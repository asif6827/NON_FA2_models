# z3_zebra_verifier_dsl_reward.py
from __future__ import annotations

import json
import math
import re

import atexit
import gc

# Workaround for occasional Z3Py shutdown/GC issues (noisy __del__ errors under sys.exit / debugger stop)
try:
    import z3.z3core as z3core  # type: ignore
    _Z3_LIB_KEEPALIVE = getattr(z3core, "_lib", None)
except Exception:
    _Z3_LIB_KEEPALIVE = None
atexit.register(lambda: gc.collect())

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Solver, Int, Distinct, And, Or, Not, Abs, sat, unsat

# ============================================================
# Utilities
# ============================================================

ORDINAL_TO_INT = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

def clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def norm_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def canon_text(s: str) -> str:
    """Aggressive canonicalization for matching values across sources."""
    s = (s or "").lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

ROLE_WORDS = {
    "lover", "lovers", "enthusiast", "enthusiasts", "drinker", "drinkers",
    "smoker", "smokers", "keeper", "keepers", "owner", "owners",
    "eater", "eaters", "fan", "fans", "person", "people",
}

def strip_role_words(s: str) -> str:
    toks = canon_text(s).split()
    while toks and toks[-1] in ROLE_WORDS:
        toks.pop()
    return " ".join(toks)

# ============================================================
# Parsing model output (<answer> JSON)
# ============================================================

def parse_answer_block(text: str) -> Dict[str, Any]:
    """
    Extract JSON from <answer>...</answer>.
    Expected top-level keys:
      - parsed_clues: list[str]
      - parsed_reasoning: list[str]
      - solution: {header: [...], rows: [[...], ...]}
    """
    m = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Missing <answer> block")
    obj = json.loads(m.group(1))
    if set(obj.keys()) != {"parsed_clues", "parsed_reasoning", "solution"}:
        raise ValueError(f"Top-level keys must be exactly parsed_clues, parsed_reasoning, solution; got {list(obj.keys())}")
    if not isinstance(obj["parsed_clues"], list) or not all(isinstance(x, str) for x in obj["parsed_clues"]):
        raise ValueError("parsed_clues must be a list[str]")
    if not isinstance(obj["parsed_reasoning"], list) or not all(isinstance(x, str) for x in obj["parsed_reasoning"]):
        raise ValueError("parsed_reasoning must be a list[str]")
    if not isinstance(obj["solution"], dict):
        raise ValueError("solution must be an object")
    return obj

# ============================================================
# Domains: prefer GT or puzzle text
# ============================================================

def domains_from_table(table: Dict[str, Any]) -> Tuple[int, Dict[str, List[str]]]:
    """
    Build domains from a ground-truth solution table.
    Returns (house_count, domains[attr]=list[str]).
    """
    header = table.get("header", [])
    rows = table.get("rows", [])
    if not header or not rows:
        return 0, {}

    # locate house column
    house_col = None
    for i, h in enumerate(header):
        if norm_key(h) == "house":
            house_col = i
            break

    n = 0
    if house_col is not None:
        for r in rows:
            if house_col < len(r):
                try:
                    n = max(n, int(str(r[house_col]).strip()))
                except Exception:
                    pass
    if n <= 0:
        n = len(rows)

    domains: Dict[str, List[str]] = {}
    for ci, h in enumerate(header):
        if norm_key(h) == "house":
            continue
        vals: List[str] = []
        for r in rows:
            if ci < len(r):
                v = str(r[ci]).strip()
                if v:
                    vals.append(v)
        # unique by canonical form
        seen = set()
        uniq = []
        for v in vals:
            cv = canon_text(v)
            if cv not in seen:
                uniq.append(v)
                seen.add(cv)
        domains[h] = uniq
    return n, domains

def domains_from_puzzle_text(puzzle_text: str) -> Tuple[int, Dict[str, List[str]]]:
    """
    Parse house_count + domains from ZebraPuzzle text.
    Looks for:
      "There are N houses..."
      "- Each person has a unique X: `a`, `b`, ..."
    """
    if not puzzle_text:
        return 0, {}

    n = 0
    m = re.search(r"There are\s+(\d+)\s+houses", puzzle_text, flags=re.IGNORECASE)
    if m:
        try:
            n = int(m.group(1))
        except Exception:
            n = 0

    domains: Dict[str, List[str]] = {}
    for line in puzzle_text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        m2 = re.search(r"^\-\s*(.+?)\s*:\s*(.+)$", line)
        if not m2:
            continue
        attr_raw = m2.group(1).strip()
        rhs = m2.group(2)
        vals = re.findall(r"`([^`]+)`", rhs)
        if vals:
            domains[attr_raw] = vals

    return n, domains

def collapse_duplicate_domains(domains: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Collapse attributes that have identical value sets (common in synthetic data)."""
    items = list(domains.items())
    used = set()
    out: Dict[str, List[str]] = {}

    def vset(vs: List[str]) -> Tuple[str, ...]:
        return tuple(sorted({canon_text(x) for x in vs}))

    for i, (ai, vi) in enumerate(items):
        if ai in used:
            continue
        used.add(ai)
        keep = ai
        si = vset(vi)
        for j in range(i + 1, len(items)):
            aj, vj = items[j]
            if aj in used:
                continue
            if vset(vj) == si:
                used.add(aj)
        out[keep] = vi
    return out

# ============================================================
# Z3 Model
# ============================================================

def build_z3_from_domains(n: int, domains: Dict[str, List[str]]) -> Tuple[Solver, Dict[str, Dict[str, Any]]]:
    """
    For each attribute attr and each value v, create Int var House(attr=v) in [1..n]
    Enforce Distinct for each attribute.
    Z[attr][canon_value] -> IntRef
    """
    S = Solver()
    Z: Dict[str, Dict[str, Any]] = {}

    for attr, values in domains.items():
        Z[attr] = {}
        vars_for_attr = []
        for v in values:
            ck = canon_text(v)
            if ck in Z[attr]:
                continue
            var = Int(f"H__{norm_key(attr)}__{ck.replace(' ', '_')}")
            Z[attr][ck] = var
            vars_for_attr.append(var)
            S.add(And(var >= 1, var <= n))
        if vars_for_attr:
            S.add(Distinct(*vars_for_attr))
    return S, Z

def build_attr_value_index(domains: Dict[str, List[str]]) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
    """
    Returns:
      attr_norm_to_attr: norm(attr) -> attr
      (attr_norm, canon(val)) -> canon(val)  (used to confirm membership)
    """
    attr_norm_to_attr: Dict[str, str] = {}
    av_index: Dict[Tuple[str, str], str] = {}
    for attr, vals in domains.items():
        attr_norm_to_attr[norm_key(attr)] = attr
        for v in vals:
            av_index[(norm_key(attr), canon_text(v))] = canon_text(v)
            # allow stripping role words for matching (optional)
            sr = strip_role_words(v)
            if sr and sr != canon_text(v):
                av_index[(norm_key(attr), sr)] = canon_text(v)
    return attr_norm_to_attr, av_index

def get_var(Z: Dict[str, Dict[str, Any]], attr: str, val: str) -> Any:
    """Fetch IntRef for (attr,val) using canonical matching."""
    a = attr
    ck = canon_text(val)
    # attr keys are original (case-sensitive); try direct then normalized match
    if a in Z and ck in Z[a]:
        return Z[a][ck]
    # fallback: try match attribute by normalized name
    a_norm = norm_key(a)
    for real_attr in Z.keys():
        if norm_key(real_attr) == a_norm and ck in Z[real_attr]:
            return Z[real_attr][ck]
    raise KeyError(f"Unknown (Attr,Val): ({attr},{val})")

# ============================================================
# DSL structures
# ============================================================

@dataclass(frozen=True)
class Term:
    attr: str
    val: str

@dataclass(frozen=True)
class ParsedClue:
    cid: int
    pred: str
    args: Tuple[Any, ...]

@dataclass(frozen=True)
class ParsedStep:
    k: int
    evidence: List[int]
    op: str  # set|not
    house: int
    attr: str
    val: str

RE_CLUE = re.compile(r"^C(?P<i>\d+)\s*=\s*(?P<pred>[a-z_]+)\((?P<args>.*)\)\.\s*$")
RE_STEP = re.compile(
    r"^S(?P<k>\d+)\s+\[(?P<ev>C\d+(?:\+C\d+)*)\]\s+(?P<op>set|not)\("
    r"(?P<h>\d+),(?P<attr>[A-Za-z][A-Za-z0-9_]*)\s*,(?P<val>[A-Za-z0-9_]+)\)\.\s*$"
)

def parse_term(s: str) -> Term:
    s = s.strip()
    if "=" not in s:
        raise ValueError(f"Bad term (missing '='): {s}")
    a, v = s.split("=", 1)
    a = a.strip()
    v = v.strip()
    if not a or not v:
        raise ValueError(f"Bad term: {s}")
    return Term(a, v)

def split_args(arg_str: str) -> List[str]:
    # predicates are simple and have no nested commas other than term pairs
    return [a.strip() for a in arg_str.split(",") if a.strip()]

def parse_parsed_clue_line(line: str) -> ParsedClue:
    m = RE_CLUE.match(line.strip())
    if not m:
        raise ValueError(f"Unparsable parsed_clues line: {line}")
    cid = int(m.group("i"))
    pred = m.group("pred")
    args_raw = split_args(m.group("args"))

    if pred in {"set", "not_set"}:
        if len(args_raw) != 3:
            raise ValueError(f"{pred} expects 3 args: {line}")
        H = int(args_raw[0])
        attr = args_raw[1]
        val = args_raw[2]
        return ParsedClue(cid=cid, pred=pred, args=(H, attr, val))

    if pred in {"immediately_left_of", "left_of", "right_of", "adjacent", "same_house"}:
        if len(args_raw) != 2:
            raise ValueError(f"{pred} expects 2 args: {line}")
        t1 = parse_term(args_raw[0])
        t2 = parse_term(args_raw[1])
        return ParsedClue(cid=cid, pred=pred, args=(t1, t2))

    if pred == "between":
        if len(args_raw) != 3:
            raise ValueError(f"between expects 3 args: {line}")
        t1 = parse_term(args_raw[0])
        t2 = parse_term(args_raw[1])
        K = int(args_raw[2])
        return ParsedClue(cid=cid, pred=pred, args=(t1, t2, K))

    raise ValueError(f"Unknown predicate '{pred}' in: {line}")

def parse_parsed_reasoning_line(line: str) -> ParsedStep:
    m = RE_STEP.match(line.strip())
    if not m:
        raise ValueError(f"Unparsable parsed_reasoning line: {line}")
    k = int(m.group("k"))
    ev = [int(x[1:]) for x in m.group("ev").split("+")]  # "C1+C3" -> [1,3]
    op = m.group("op")
    h = int(m.group("h"))
    attr = m.group("attr")
    val = m.group("val")
    return ParsedStep(k=k, evidence=ev, op=op, house=h, attr=attr, val=val)

# ============================================================
# Compile DSL to Z3 constraints
# ============================================================

def compile_term(Z: Dict[str, Dict[str, Any]], term: Term) -> Any:
    # term values in DSL are normalized with underscores; canon_text handles that.
    return get_var(Z, term.attr, term.val)

def compile_clue_constraint(Z: Dict[str, Dict[str, Any]], clue: ParsedClue) -> Any:
    p = clue.pred
    if p == "set":
        H, attr, val = clue.args
        v = get_var(Z, attr, val)
        return v == H
    if p == "not_set":
        H, attr, val = clue.args
        v = get_var(Z, attr, val)
        return v != H

    if p in {"immediately_left_of", "left_of", "right_of", "adjacent", "same_house"}:
        t1, t2 = clue.args  # type: ignore[misc]
        v1 = compile_term(Z, t1)
        v2 = compile_term(Z, t2)
        if p == "immediately_left_of":
            return v1 + 1 == v2
        if p == "left_of":
            return v1 < v2
        if p == "right_of":
            return v1 > v2
        if p == "adjacent":
            return Abs(v1 - v2) == 1
        if p == "same_house":
            return v1 == v2

    if p == "between":
        t1, t2, K = clue.args  # type: ignore[misc]
        v1 = compile_term(Z, t1)
        v2 = compile_term(Z, t2)
        # K houses strictly between => positions differ by K+1
        return Abs(v1 - v2) == (int(K) + 1)

    raise ValueError(f"Cannot compile predicate: {p}")

def compile_step_constraint(Z: Dict[str, Dict[str, Any]], step: ParsedStep) -> Any:
    v = get_var(Z, step.attr, step.val)
    if step.op == "set":
        return v == step.house
    if step.op == "not":
        return v != step.house
    raise ValueError(f"Unknown step op: {step.op}")

def negate_step_constraint(step: ParsedStep, Z: Dict[str, Dict[str, Any]]) -> Any:
    v = get_var(Z, step.attr, step.val)
    if step.op == "set":
        return v != step.house
    if step.op == "not":
        return v == step.house
    raise ValueError(f"Unknown step op: {step.op}")

# ============================================================
# Predicted solution constraints + GT accuracy
# ============================================================

def header_to_domain_attr(pred_header: List[str], domains: Dict[str, List[str]]) -> Dict[int, Optional[str]]:
    dom_keys = list(domains.keys())
    dom_norm = [(k, norm_key(k)) for k in dom_keys]
    mapping: Dict[int, Optional[str]] = {}
    for i, h in enumerate(pred_header):
        hn = norm_key(h)
        if hn == "house":
            mapping[i] = None
            continue
        best = None
        best_score = -1
        for k, kn in dom_norm:
            if hn == kn:
                best = k
                best_score = 10
                break
            if hn and kn and (hn in kn or kn in hn):
                score = min(len(hn), len(kn))
                if score > best_score:
                    best_score = score
                    best = k
        mapping[i] = best
    return mapping

def add_pred_solution_constraints(
    S: Solver,
    Z: Dict[str, Dict[str, Any]],
    domains: Dict[str, List[str]],
    pred_solution: Dict[str, Any],
) -> float:
    header = pred_solution.get("header", [])
    rows = pred_solution.get("rows", [])
    if not header or not rows:
        return 0.0

    house_col = None
    for i, h in enumerate(header):
        if norm_key(h) == "house":
            house_col = i
            break

    col_to_attr = header_to_domain_attr(header, domains)

    grounded = 0
    total_cells = 0

    for ri, row in enumerate(rows):
        if house_col is not None and house_col < len(row):
            try:
                hidx = int(str(row[house_col]).strip())
            except Exception:
                hidx = ri + 1
        else:
            hidx = ri + 1

        for ci, cell in enumerate(row):
            if ci == house_col:
                continue
            total_cells += 1
            attr = col_to_attr.get(ci)
            if not attr:
                continue
            v = str(cell).strip()
            if not v:
                continue

            ck = canon_text(v)
            var = Z.get(attr, {}).get(ck)
            if var is None:
                ck2 = strip_role_words(v)
                if ck2:
                    var = Z.get(attr, {}).get(ck2)

            if var is None:
                continue

            grounded += 1
            S.add(var == hidx)

    return float(clamp01(grounded / max(1, total_cells)))

def table_cell_accuracy(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[float, float]:
    """
    Returns (cell_acc, puzzle_acc).
    - cell_acc: fraction of GT cells (excluding House) matched by pred
    - puzzle_acc: 1.0 iff all GT cells matched and all required columns exist
    """
    gt_h = gt.get("header", [])
    gt_r = gt.get("rows", [])
    pr_h = pred.get("header", [])
    pr_r = pred.get("rows", [])
    if not gt_h or not gt_r or not pr_h or not pr_r:
        return 0.0, 0.0

    # index columns
    gt_col = {norm_key(h): i for i, h in enumerate(gt_h)}
    pr_col = {norm_key(h): i for i, h in enumerate(pr_h)}
    if "house" not in gt_col or "house" not in pr_col:
        return 0.0, 0.0

    # map rows by house
    def rows_by_house(header: List[str], rows: List[List[str]]) -> Dict[int, List[str]]:
        hc = None
        for i, h in enumerate(header):
            if norm_key(h) == "house":
                hc = i
                break
        out = {}
        for ri, row in enumerate(rows):
            if hc is not None and hc < len(row):
                try:
                    hidx = int(str(row[hc]).strip())
                except Exception:
                    hidx = ri + 1
            else:
                hidx = ri + 1
            out[hidx] = row
        return out

    gt_map = rows_by_house(gt_h, gt_r)
    pr_map = rows_by_house(pr_h, pr_r)

    total = 0
    correct = 0
    missing_cols = False
    for hk, gt_row in gt_map.items():
        pr_row = pr_map.get(hk)
        if pr_row is None:
            # all cells for this house are wrong
            for h in gt_h:
                if norm_key(h) != "house":
                    total += 1
            continue
        for h in gt_h:
            hn = norm_key(h)
            if hn == "house":
                continue
            total += 1
            gi = gt_col[hn]
            pi = pr_col.get(hn)
            if pi is None:
                missing_cols = True
                continue
            gv = canon_text(str(gt_row[gi]) if gi < len(gt_row) else "")
            pv = canon_text(str(pr_row[pi]) if pi < len(pr_row) else "")
            if gv and (gv == pv):
                correct += 1

    cell_acc = float(correct / max(1, total))
    puzzle_acc = 1.0 if (correct == total and not missing_cols) else 0.0
    return cell_acc, puzzle_acc

# ============================================================
# Information-theory-ish metrics (cheap & useful)
# ============================================================

def possible_houses_for_var(S: Solver, var: Any, n: int) -> List[int]:
    poss = []
    for h in range(1, n + 1):
        S.push()
        S.add(var == h)
        ok = (S.check() == sat)
        S.pop()
        if ok:
            poss.append(h)
    return poss

def local_info_gain_for_step(S_before: Solver, S_after: Solver, var: Any, n: int) -> float:
    """
    Local information gain: log(d_before/d_after) where d_* is number of possible houses for the step variable.
    Normalized later for reward.
    """
    db = len(possible_houses_for_var(S_before, var, n))
    da = len(possible_houses_for_var(S_after, var, n))
    if da <= 0 or db <= 0:
        return 0.0
    if da >= db:
        return 0.0
    return float(math.log(db / da))

def global_uncertainty(S: Solver, Z: Dict[str, Dict[str, Any]], n: int, max_vars: Optional[int] = None) -> float:
    """
    U = sum_v log(|possible_houses(v)|).
    Use max_vars to cap runtime if needed.
    """
    vars_list = []
    for attr in sorted(Z.keys(), key=lambda a: norm_key(a)):
        for ck in sorted(Z[attr].keys()):
            vars_list.append(Z[attr][ck])
    if max_vars is not None:
        vars_list = vars_list[:max_vars]

    U = 0.0
    for v in vars_list:
        d = len(possible_houses_for_var(S, v, n))
        if d <= 0:
            return float("inf")
        U += math.log(d)
    return U

# ============================================================
# Core evaluation (clues + steps + solution + GT)
# ============================================================

def compute_dsl_components(
    parsed_clues_lines: List[str],
    parsed_reasoning_lines: List[str],
    predicted_solution: Dict[str, Any],
    raw_clues_text: Optional[List[str]] = None,
    puzzle_text: Optional[str] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    compute_global_info: bool = True,
) -> Dict[str, Any]:
    """
    Main evaluator for your NEW prompt format.

    Returns metrics suitable for reward shaping:
      - clue_parse_rate, clue_count_match
      - clue_sat, z3_sat, assignment_cov
      - step_parse_rate, step_consistency_rate, step_entail_rate
      - avg_step_local_info_gain
      - uncertainty_reduction (global) (optional)
      - cell_acc, puzzle_acc (if GT given)
    """
    out: Dict[str, Any] = {
        "clue_parse_rate": 0.0,
        "clue_count_match": 0.0,
        "clue_grounding": 0.0,

        "assignment_cov": 0.0,
        "clue_sat": 0.0,
        "z3_sat": 0.0,
        "base_sat": 0.0,

        "step_parse_rate": 0.0,
        "step_consistency_rate": 0.0,
        "step_entail_rate": 0.0,
        "avg_step_local_info_gain": 0.0,
        "uncertainty_reduction": 0.0,

        "cell_acc": 0.0,
        "puzzle_acc": 0.0,

        "n": 0,
        "domains": {},
        "num_clue_constraints": 0,
        "num_steps": len(parsed_reasoning_lines),
        "num_steps_entailed": 0,
        "num_steps_consistent": 0,
        "num_steps_parsed": 0,
        "errors": [],
    }

    try:
        # 1) domains (prefer GT, else puzzle text, else predicted table)
        n = 0
        domains: Dict[str, List[str]] = {}
        if ground_truth:
            n, domains = domains_from_table(ground_truth)
        if (not domains) and puzzle_text:
            n2, d2 = domains_from_puzzle_text(puzzle_text)
            if d2:
                domains = d2
                if n <= 0:
                    n = n2
        if not domains:
            n, domains = domains_from_table(predicted_solution)

        domains = collapse_duplicate_domains(domains)
        if n <= 0:
            n = len(predicted_solution.get("rows", []))
        if n <= 0 or not domains:
            out["errors"].append("Missing domains/n (provide ground_truth or puzzle_text).")
            return out

        out["n"] = n
        out["domains"] = domains

        base, Z = build_z3_from_domains(n, domains)

        # 2) parse + compile clues
        parsed_clues: List[ParsedClue] = []
        parsed_ok = 0
        for line in parsed_clues_lines:
            try:
                parsed_clues.append(parse_parsed_clue_line(line))
                parsed_ok += 1
            except Exception as e:
                out["errors"].append(f"clue_parse_error: {e}")

        out["clue_parse_rate"] = float(clamp01(parsed_ok / max(1, len(parsed_clues_lines))))

        # clue count check vs raw clues if provided
        if raw_clues_text is not None:
            out["clue_count_match"] = 1.0 if (len(parsed_clues_lines) == len(raw_clues_text)) else 0.0
        else:
            out["clue_count_match"] = 1.0

        # grounding: do values mentioned in parsed clues appear in raw clue text?
        if raw_clues_text is not None and len(raw_clues_text) > 0:
            def extracted_vals_from_clue(cl: ParsedClue) -> List[str]:
                if cl.pred in {"set", "not_set"}:
                    _, _, v = cl.args
                    return [str(v)]
                if cl.pred == "between":
                    t1, t2, _ = cl.args
                    return [t1.val, t2.val]
                t1, t2 = cl.args  # type: ignore[misc]
                return [t1.val, t2.val]

            grounded = 0
            for i, cl in enumerate(parsed_clues):
                raw = raw_clues_text[i] if i < len(raw_clues_text) else ""
                raw_can = " " + canon_text(raw) + " "
                ok = True
                for v in extracted_vals_from_clue(cl):
                    vcan = canon_text(v)
                    if vcan and (f" {vcan} " not in raw_can):
                        ok = False
                        break
                grounded += 1 if ok else 0
            out["clue_grounding"] = float(grounded / max(1, len(parsed_clues)))
        else:
            out["clue_grounding"] = 0.0

        clue_constraints = []
        for cl in parsed_clues:
            try:
                clue_constraints.append(compile_clue_constraint(Z, cl))
            except Exception as e:
                out["errors"].append(f"clue_compile_error(C{cl.cid}): {e}")

        out["num_clue_constraints"] = len(clue_constraints)

        # base + clue satisfiable?
        S_base = Solver()
        S_base.add(base.assertions())
        for c in clue_constraints:
            S_base.add(c)
        out["base_sat"] = 1.0 if (S_base.check() == sat) else 0.0

        # 3) predicted assignment coverage
        S_sol = Solver()
        S_sol.add(base.assertions())
        for c in clue_constraints:
            S_sol.add(c)
        out["assignment_cov"] = add_pred_solution_constraints(S_sol, Z, domains, predicted_solution)

        # 4) clue_sat under predicted assignment: does predicted imply each clue constraint?
        if clue_constraints:
            sat_count = 0
            for c in clue_constraints:
                S_sol.push()
                S_sol.add(Not(c))
                implied = (S_sol.check() == unsat)
                S_sol.pop()
                if implied:
                    sat_count += 1
            out["clue_sat"] = float(sat_count / max(1, len(clue_constraints)))

        # 5) z3_sat: base + clues + predicted assignment SAT?
        out["z3_sat"] = 1.0 if (S_sol.check() == sat) else 0.0

        # 6) process / step verification (entailed steps)
        parsed_steps: List[ParsedStep] = []
        parsed_steps_ok = 0
        for line in parsed_reasoning_lines:
            try:
                parsed_steps.append(parse_parsed_reasoning_line(line))
                parsed_steps_ok += 1
            except Exception as e:
                out["errors"].append(f"step_parse_error: {e}")

        out["num_steps_parsed"] = parsed_steps_ok
        out["step_parse_rate"] = float(clamp01(parsed_steps_ok / max(1, len(parsed_reasoning_lines))))

        # enforce step numbering monotonicity (optional; doesn't hard-fail)
        # you can turn this into reward penalties upstream if you want.

        S_state = Solver()
        S_state.add(base.assertions())
        for c in clue_constraints:
            S_state.add(c)

        # global uncertainty before any steps (optional)
        U0 = global_uncertainty(S_state, Z, n) if compute_global_info else 0.0

        consistent = 0
        entailed = 0
        local_igs: List[float] = []

        for st in parsed_steps:
            # evidence IDs validity (optional soft check)
            if parsed_clues and any((e < 1 or e > len(parsed_clues)) for e in st.evidence):
                out["errors"].append(f"bad_evidence_in_step_S{st.k}: {st.evidence}")

            try:
                c_step = compile_step_constraint(Z, st)
                c_neg = negate_step_constraint(st, Z)
            except Exception as e:
                out["errors"].append(f"step_compile_error(S{st.k}): {e}")
                continue

            # consistency check
            S_state.push()
            S_state.add(c_step)
            is_consistent = (S_state.check() == sat)
            S_state.pop()
            if is_consistent:
                consistent += 1

            # entailment check: state entails step if NOT(step) makes UNSAT
            S_state.push()
            S_state.add(c_neg)
            is_entailed = (S_state.check() == unsat)
            S_state.pop()

            if is_entailed:
                # local info gain for the variable affected
                var = get_var(Z, st.attr, st.val)
                # create S_after = S_state + step for local IG
                S_after = Solver()
                S_after.add(S_state.assertions())
                S_after.add(c_step)
                ig = local_info_gain_for_step(S_state, S_after, var, n)
                local_igs.append(ig)

                # accept step
                S_state.add(c_step)
                entailed += 1

        out["num_steps_consistent"] = consistent
        out["num_steps_entailed"] = entailed
        out["step_consistency_rate"] = float(clamp01(consistent / max(1, len(parsed_steps))))
        out["step_entail_rate"] = float(clamp01(entailed / max(1, len(parsed_steps))))
        out["avg_step_local_info_gain"] = float(sum(local_igs) / max(1, len(local_igs))) if local_igs else 0.0

        if compute_global_info and math.isfinite(U0) and U0 > 0:
            U1 = global_uncertainty(S_state, Z, n)
            if math.isfinite(U1):
                out["uncertainty_reduction"] = float(clamp01((U0 - U1) / U0))
            else:
                out["uncertainty_reduction"] = 0.0

        # 7) GT accuracy
        if ground_truth:
            cell_acc, puzzle_acc = table_cell_accuracy(predicted_solution, ground_truth)
            out["cell_acc"] = float(cell_acc)
            out["puzzle_acc"] = float(puzzle_acc)

        return out

    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        out["errors"].append(str(e))
        return out

# ============================================================
# Curriculum reward (Logic + Cell + Puzzle + Info metrics)
# ============================================================

def curriculum_reward(
    metrics: Dict[str, Any],
    epoch: int,
    total_epochs: int,
    *,
    w_logic_max: float = 1.0,
    w_process_max: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Remedy A + C combined:

    Warmup (first 10 epochs): focus on structure + logic/process (no GT pressure).

    After warmup:
      - HARD GATE: if output is not fully parsable / not 1:1 clue count => reward = 0.0  (Remedy C)
      - If puzzle_acc == 1.0 => reward = 1.0
      - Else compute:
          cell_part = min(0.6, 0.6 * cell_acc) * (0.5 + 0.5 * z3_sat)
          z3_part   = 0.3 * z3_score
        reward = max(cell_part, z3_part)  (Remedy A)
    """
    import math

    def f(key: str, default: float = 0.0) -> float:
        try:
            return float(metrics.get(key, default))
        except Exception:
            return float(default)

    # ---- settings ----
    warmup_epochs = 10

    # keep t for logging/debugging
    t = 1.0 if total_epochs <= 1 else float(epoch) / float(max(1, total_epochs - 1))
    t = clamp01(t)

    # ---- core metrics ----
    cell_acc   = f("cell_acc")
    puzzle_acc = f("puzzle_acc")

    clue_sat = f("clue_sat")
    z3_sat   = f("z3_sat")
    base_sat = f("base_sat")

    parse_rate   = f("clue_parse_rate")
    step_parse   = f("step_parse_rate")
    clue_count   = f("clue_count_match", 1.0)   # if not provided, assume OK
    assign_cov   = f("assignment_cov")
    clue_ground  = f("clue_grounding")

    step_entail  = f("step_entail_rate")
    step_cons    = f("step_consistency_rate")

    u_red    = f("uncertainty_reduction")
    local_ig = f("avg_step_local_info_gain")
    ig_norm  = 1.0 - math.exp(-local_ig) if local_ig > 0 else 0.0

    # ---- component scores (used in warmup only) ----
    structure_score = clamp01(0.45 * parse_rate + 0.35 * assign_cov + 0.20 * clue_ground)
    logic_score     = clamp01(0.20 * base_sat + 0.45 * z3_sat + 0.35 * clue_sat)
    process_score   = clamp01(0.60 * step_entail + 0.25 * step_cons + 0.15 * u_red)
    info_score      = clamp01(0.50 * u_red + 0.50 * ig_norm)

    # =========================================================
    # Warmup: structure + logic/process
    # =========================================================
    if epoch < warmup_epochs:
        warmup_reward = (
            0.50 * (w_logic_max * logic_score) +
            0.40 * structure_score +
            0.08 * (w_process_max * process_score) +
            0.02 * info_score
        )
        warmup_cap = 0.50
        reward = clamp01(min(warmup_cap, warmup_reward))
        return reward, {
            "t": t,
            "phase": "warmup",
            "reward": reward,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "structure_score": structure_score,
            "logic_score": logic_score,
            "process_score": process_score,
            "info_score": info_score,
            "z3_sat": z3_sat,
            "clue_sat": clue_sat,
            "step_entail_rate": step_entail,
            "clue_parse_rate": parse_rate,
            "step_parse_rate": step_parse,
            "clue_count_match": clue_count,
        }

    # =========================================================
    # Post-warmup: HARD structure gate (Remedy C)
    # =========================================================
    # Require full parsability and 1:1 clue count before giving any reward.
    # (Avoids rewarding formatting in post-warmup; it becomes a prerequisite.)
    structure_ok = (
        parse_rate >= 0.999999 and
        step_parse >= 0.999999 and
        clue_count >= 0.999999
    )
    if not structure_ok:
        return 0.0, {
            "t": t,
            "phase": "invalid_structure",
            "reward": 0.0,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "clue_parse_rate": parse_rate,
            "step_parse_rate": step_parse,
            "clue_count_match": clue_count,
        }

    # =========================================================
    # Post-warmup: GT-first
    # =========================================================
    if puzzle_acc >= 1.0:
        return 1.0, {
            "t": t,
            "phase": "gt",
            "reward": 1.0,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
        }

    # ---- Z3 score (dense, for ranking/contrast) ----
    # Note: parse metrics are NOT included here because we hard-gate structure_ok above.
    z3_score = (
        0.30 * z3_sat +
        0.25 * clue_sat +
        0.20 * step_entail +
        0.15 * assign_cov +
        0.10 * u_red
    )
    z3_score = clamp01(z3_score)
    z3_part = 0.3 * z3_score  # capped at 0.3

    # ---- Cell part (your gating) ----
    cell_part = min(0.6, 0.6 * cell_acc) * (0.5 + 0.5 * z3_sat)  # <= 0.6

    # Remedy A: don't discard z3 signal when cell is small; use max
    reward = clamp01(max(cell_part, z3_part))

    return reward, {
        "t": t,
        "phase": "main",
        "reward": reward,
        "cell_acc": cell_acc,
        "puzzle_acc": puzzle_acc,
        "cell_part": cell_part,
        "z3_part": z3_part,
        "z3_score": z3_score,
        "z3_sat": z3_sat,
        "clue_sat": clue_sat,
        "step_entail_rate": step_entail,
        "assignment_cov": assign_cov,
        "uncertainty_reduction": u_red,
    }




def curriculum_reward_v2(
    metrics: Dict[str, Any],
    epoch: int,
    total_epochs: int,
    *,
    w_logic_max: float = 1.0,
    w_process_max: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """
    New curriculum (piecewise):

    Warmup (first 10 epochs): focus on structure + logic/process (no GT pressure).
    After warmup:
      1) If puzzle_acc == 1.0 => reward = 1.0
      2) Elif cell_acc > 0.0 => reward = min(0.6, 0.6 * cell_acc) * (0.5 + 0.5 * z3_sat)
      3) Else => reward based on Z3 feedback, capped at 0.3

    Returns: (reward in [0,1], breakdown dict)
    """
    # --- helpers ---
    def f(key: str, default: float = 0.0) -> float:
        try:
            return float(metrics.get(key, default))
        except Exception:
            return float(default)

    warmup_epochs = 2

    # keep t for logging/debugging
    t = 1.0 if total_epochs <= 1 else float(epoch) / float(max(1, total_epochs - 1))
    t = clamp01(t)

    # --- core metrics ---
    cell_acc = f("cell_acc")
    puzzle_acc = f("puzzle_acc")

    clue_sat = f("clue_sat")
    z3_sat = f("z3_sat")
    base_sat = f("base_sat")

    parse_rate = f("clue_parse_rate")
    assign_cov = f("assignment_cov")
    clue_ground = f("clue_grounding")

    step_entail = f("step_entail_rate")
    step_cons = f("step_consistency_rate")

    u_red = f("uncertainty_reduction")
    local_ig = f("avg_step_local_info_gain")

    # normalize local_ig to [0,1] with gentle squashing
    ig_norm = 1.0 - math.exp(-local_ig) if local_ig > 0 else 0.0

    # --- component scores (all in [0,1] ideally) ---
    structure_score = 0.45 * parse_rate + 0.35 * assign_cov + 0.20 * clue_ground
    logic_score = 0.20 * base_sat + 0.45 * z3_sat + 0.35 * clue_sat
    process_score = 0.60 * step_entail + 0.25 * step_cons + 0.15 * u_red
    info_score = 0.50 * u_red + 0.50 * ig_norm

    # clamp safety
    structure_score = clamp01(structure_score)
    logic_score = clamp01(logic_score)
    process_score = clamp01(process_score)
    info_score = clamp01(info_score)

    # =========================================================
    # Warmup: first 10 epochs => emphasize structure + logic
    # =========================================================
    if epoch < warmup_epochs:
        # scaled by w_logic_max / w_process_max if you want to tune externally
        warmup_reward = (
            0.50 * (w_logic_max * logic_score) +
            0.40 * structure_score +
            0.08 * (w_process_max * process_score) +
            0.02 * info_score
        )

        # keep warmup from becoming "good enough" (recommended)
        warmup_cap = 0.50
        reward = clamp01(min(warmup_cap, warmup_reward))

        breakdown = {
            "t": t,
            "phase": "warmup",
            "reward": reward,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "structure_score": structure_score,
            "logic_score": logic_score,
            "process_score": process_score,
            "info_score": info_score,
            "z3_sat": z3_sat,
            "clue_sat": clue_sat,
            "step_entail_rate": step_entail,
        }
        return reward, breakdown

    # =========================================================
    # Main: epoch >= 10 => GT-driven with caps
    # =========================================================
    if puzzle_acc >= 1.0:
        return 1.0, {
            "t": t,
            "phase": "gt",
            "reward": 1.0,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "structure_score": 0.0,
            "logic_score": 0.0,
            "process_score": 0.0,
            "info_score": 0.0,
            "z3_sat": 0.0,
            "clue_sat": 0.0,
            "step_entail_rate": 0.0,
        }

    if cell_acc > 0.0:
        # IMPORTANT: your requested gating
        r = min(0.8, 0.6 * cell_acc) * (0.5 + 0.5 * z3_sat)
        r = clamp01(r)
        return r, {
            "t": t,
            "phase": "cell",
            "reward": r,
            "cell_acc": cell_acc,
            "puzzle_acc": puzzle_acc,
            "structure_score": 0.0,
            "logic_score": 0.0,
            "process_score": 0.0,
            "info_score": 0.0,
            "z3_sat": z3_sat,
            "clue_sat": 0.0,
            "step_entail_rate": 0.0,
        }

    # Else: Z3 feedback only, capped at 0.3
    z3_score = (
        0.25 * z3_sat +
        0.20 * clue_sat +
        0.20 * step_entail +
        0.15 * assign_cov +
        0.10 * parse_rate +
        0.10 * u_red
    )
    z3_score = clamp01(z3_score)

    r = 0.5 * z3_score  # already capped at 0.3 because z3_score <= 1
    r = clamp01(r)

    return r, {
        "t": t,
        "phase": "z3",
        "reward": r,
        "z3_score": z3_score,
        "z3_sat": z3_sat,
        "clue_sat": clue_sat,
        "step_entail_rate": step_entail,
        "assignment_cov": assign_cov,
        "clue_parse_rate": parse_rate,
        "uncertainty_reduction": u_red,
    }



def curriculum_reward_v1(
    metrics: Dict[str, Any],
    epoch: int,
    total_epochs: int,
    *,
    w_logic_max: float = 1.0,
    w_process_max: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """
    A compact curriculum:
      - Early: emphasize cell accuracy + parsing/grounding (learn the format + mapping)
      - Mid: emphasize logic consistency (clue_sat, z3_sat) + valid entailed steps
      - Late: emphasize puzzle exactness (puzzle_acc)
      - Throughout: small info-theory bonus (uncertainty_reduction, local info gain)

    Returns: (reward in [0,1], breakdown dict)
    """
    t = 1.0 if total_epochs <= 1 else float(epoch) / float(max(1, total_epochs - 1))
    t = clamp01(t)

    cell_acc = float(metrics.get("cell_acc", 0.0))
    puzzle_acc = float(metrics.get("puzzle_acc", 0.0))
    clue_sat = float(metrics.get("clue_sat", 0.0))
    z3_sat = float(metrics.get("z3_sat", 0.0))
    parse_rate = float(metrics.get("clue_parse_rate", 0.0))
    assign_cov = float(metrics.get("assignment_cov", 0.0))
    clue_ground = float(metrics.get("clue_grounding", 0.0))

    step_entail = float(metrics.get("step_entail_rate", 0.0))
    step_cons = float(metrics.get("step_consistency_rate", 0.0))
    u_red = float(metrics.get("uncertainty_reduction", 0.0))
    local_ig = float(metrics.get("avg_step_local_info_gain", 0.0))

    # normalize local_ig to [0,1] with a gentle squashing
    ig_norm = 1.0 - math.exp(-local_ig) if local_ig > 0 else 0.0

    # weights
    w_cell = (1.0 - t) ** 1.2
    w_puzzle = t ** 1.8
    w_parse = (1.0 - t) ** 0.8
    w_logic = w_logic_max * (0.3 + 0.7 * t)
    w_process = w_process_max * (0.2 + 0.8 * t)
    w_info = 0.15  # small but persistent

    # component scores
    parse_score = 0.45 * parse_rate + 0.35 * assign_cov + 0.20 * clue_ground
    logic_score = 0.55 * z3_sat + 0.45 * clue_sat
    process_score = 0.60 * step_entail + 0.25 * step_cons + 0.15 * u_red
    info_score = 0.5 * u_red + 0.5 * ig_norm

    # weighted average (keeps reward in [0,1] if scores in [0,1])
    denom = (w_cell + w_puzzle + w_parse + w_logic + w_process + w_info)
    if denom <= 0:
        return 0.0, {}

    reward = (
        w_cell * cell_acc +
        w_puzzle * puzzle_acc +
        w_parse * parse_score +
        w_logic * logic_score +
        w_process * process_score +
        w_info * info_score
    ) / denom

    reward = float(clamp01(reward))

    breakdown = {
        "t": t,
        "cell_acc": cell_acc,
        "puzzle_acc": puzzle_acc,
        "parse_score": parse_score,
        "logic_score": logic_score,
        "process_score": process_score,
        "info_score": info_score,
        "reward": reward,
    }
    return reward, breakdown

# ============================================================
# Minimal demo usage
# ============================================================

def demo():
    # Example puzzle + GT
    ground_truth = {
        "header": ["House", "Name", "Drink"],
        "rows": [
            ["1", "Arnold", "tea"],
            ["2", "Peter", "water"],
            ["3", "Eric", "milk"]
        ]
    }

    parsed_clues = [
        "C1 = set(2,Name,Peter).",
        "C2 = immediately_left_of(Name=Arnold,Drink=water).",
        "C3 = immediately_left_of(Drink=water,Drink=milk)."
    ]

    parsed_reasoning = [
        "S1 [C1] set(2,Name,Peter).",
        "S2 [C3] not(3,Drink,water).",
        "S3 [C3] not(1,Drink,milk).",
        "S4 [C2] not(3,Name,Arnold).",
        "S5 [C2+C3] set(1,Name,Arnold)."
    ]

    predicted = ground_truth

    metrics = compute_dsl_components(
        parsed_clues_lines=parsed_clues,
        parsed_reasoning_lines=parsed_reasoning,
        predicted_solution=predicted,
        raw_clues_text=[
            "Peter is in the second house.",
            "Arnold is directly left of the one who drinks water.",
            "The water drinker is directly left of the milk drinker."
        ],
        puzzle_text=None,
        ground_truth=ground_truth,
        debug=True,
        compute_global_info=True,
    )

    r, br = curriculum_reward(metrics, epoch=3, total_epochs=10)
    print("metrics:", {k: metrics[k] for k in ["cell_acc","puzzle_acc","clue_sat","z3_sat","step_entail_rate","uncertainty_reduction","clue_parse_rate"]})
    print("reward:", r)
    print("breakdown:", br)

if __name__ == "__main__":
    demo()
