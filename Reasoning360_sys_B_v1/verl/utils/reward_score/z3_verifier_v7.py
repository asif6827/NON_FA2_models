# z3_reward_v5_fixed_puzzle_acc.py
import json
import re
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Solver, Int, Distinct, Abs, Not, sat, unsat, ArithRef


# =========================
# 0) Parsing
# =========================

def parse_answer_tag(s: str) -> Optional[str]:
    m = re.search(r"<answer>([\s\S]*?)</answer>", s, re.DOTALL)
    return m.group(1).strip() if m else None

def _repair_jsonish(chunk: str) -> str:
    c = chunk.strip()
    # key =  -> key:
    c = re.sub(r'(".*?"|\'.*?\'|[A-Za-z_][A-Za-z0-9_ ]*)\s*=\s*', r"\1: ", c)
    # '...' -> "..."
    c = re.sub(r"(?<!\\)'([^'\\]*)'", r'"\1"', c)
    # remove trailing commas before } or ]
    c = re.sub(r",\s*([}\]])", r"\1", c)
    return c

def _first_balanced_obj(text: str) -> Optional[str]:
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start:i+1]
    return None

def parse_jsonish_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()

    # strict JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    chunk = _first_balanced_obj(s) or s
    repaired = _repair_jsonish(chunk)

    # try json after repair
    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # try python literal safely
    repaired_py = (
        repaired
        .replace("null", "None")
        .replace("true", "True")
        .replace("false", "False")
    )
    try:
        obj = ast.literal_eval(repaired_py)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def parse_llm_answer(llm_output: str) -> Tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
    content = parse_answer_tag(llm_output) or llm_output
    obj = parse_jsonish_object(content)
    if not obj:
        return None, None
    reasoning = obj.get("reasoning", None)
    solution = obj.get("solution", None)
    if isinstance(reasoning, str):
        reasoning = [reasoning]
    if reasoning is not None and not isinstance(reasoning, list):
        reasoning = None
    if solution is not None and not isinstance(solution, dict):
        solution = None
    return reasoning, solution


# =========================
# 1) Normalization / structure helpers
# =========================

def _norm(x: Any) -> str:
    return str(x).strip().lower()

def structure_score(solution: Dict[str, Any]) -> float:
    """
    Checks:
      - header exists, rows exists
      - consistent row length
      - house column numeric-like
      - per-column duplicates penalty
    """
    if not isinstance(solution, dict):
        return 0.0
    header = solution.get("header")
    rows = solution.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list) or len(header) < 2 or len(rows) < 1:
        return 0.0

    n = len(rows)
    m = len(header)

    ok_rows = 0
    house_ok = 0
    for r in rows:
        if isinstance(r, list) and len(r) == m:
            ok_rows += 1
            try:
                int(r[0])
                house_ok += 1
            except Exception:
                pass

    base = (ok_rows / n) * 0.7 + (house_ok / n) * 0.3

    # duplicate penalty per attribute column (excluding house)
    dup_pen = 0.0
    for j in range(1, m):
        col = [_norm(r[j]) for r in rows if isinstance(r, list) and len(r) == m]
        if not col:
            continue
        uniq = len(set(col))
        dup_pen += (1.0 - (uniq / len(col)))

    dup_pen = dup_pen / max(1, (m - 1))
    return max(0.0, min(1.0, base * (1.0 - 0.5 * dup_pen)))

def completeness_score(solution: Dict[str, Any]) -> float:
    """
    Penalize missing/empty cells + encourage full domain coverage (n unique values per attr).
    This helps puzzle-accuracy: discourages 'degenerate' tables that Z3 can't reject.
    """
    if not isinstance(solution, dict):
        return 0.0
    header = solution.get("header")
    rows = solution.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list) or len(header) < 2 or len(rows) < 1:
        return 0.0

    n = len(rows)
    m = len(header)

    # non-empty cell ratio (excluding house)
    total = 0
    nonempty = 0
    for r in rows:
        if not isinstance(r, list) or len(r) != m:
            continue
        for j in range(1, m):
            total += 1
            if _norm(r[j]) not in ("", "none", "null", "na", "n/a"):
                nonempty += 1
    cell_fill = (nonempty / total) if total > 0 else 0.0

    # per attribute: unique count should be n
    dom = 0.0
    for j in range(1, m):
        col = [_norm(r[j]) for r in rows if isinstance(r, list) and len(r) == m]
        uniq = len(set(col))
        dom += min(1.0, uniq / max(1, n))
    dom = dom / max(1, (m - 1))

    return max(0.0, min(1.0, 0.5 * cell_fill + 0.5 * dom))


# =========================
# 2) GT scoring (cell + exact shaping)
# =========================

def normalize_grid(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict) or "header" not in data or "rows" not in data:
        return None
    try:
        header = [_norm(h) for h in data["header"]]
        ignore_cols = {"house", "position"}
        keep = [i for i, h in enumerate(header) if h not in ignore_cols] or list(range(len(header)))

        rows_norm = []
        for r in data["rows"]:
            r = [_norm(x) for x in r]
            rows_norm.append([r[i] for i in keep])

        header_kept = [header[i] for i in keep]
        return {"header": header_kept, "rows": sorted(rows_norm)}
    except Exception:
        return None

def cell_acc(pred: Dict[str, Any], gt: Dict[str, Any]) -> Tuple[float, float]:
    """
    Returns (exact_match, cell_accuracy)
    """
    np = normalize_grid(pred)
    ng = normalize_grid(gt)
    if not np or not ng:
        return 0.0, 0.0
    if np == ng:
        return 1.0, 1.0
    ph, pr = np["header"], np["rows"]
    gh, gr = ng["header"], ng["rows"]
    if ph != gh or len(pr) != len(gr):
        return 0.0, 0.0
    correct = 0
    total = 0
    for rp, rg in zip(pr, gr):
        if len(rp) != len(rg):
            return 0.0, 0.0
        total += len(rp)
        correct += sum(1 for a, b in zip(rp, rg) if a == b)
    return 0.0, (correct / total) if total > 0 else 0.0

def shaped_gt_reward(exact: float, cell: float, gamma: float = 2.0, exact_bonus: float = 0.25) -> float:
    """
    Push puzzle-accuracy:
      - if exact => 1.0
      - else => cell^gamma (steeper near 1)
      - plus small bonus only when exact
    """
    if exact >= 1.0:
        return 1.0 + exact_bonus
    return float(max(0.0, min(1.0, cell)) ** gamma)


# =========================
# 3) Z3 model + clue parsing
# =========================

def sanitize_sym(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "v"

def build_model(solution: Dict[str, Any]) -> Tuple[Solver, Dict[str, Dict[str, ArithRef]]]:
    header = solution["header"]
    rows = solution["rows"]
    n = len(rows)

    attrs = header[1:]
    values = {a: set() for a in attrs}
    for r in rows:
        for a, v in zip(attrs, r[1:]):
            values[a].add(_norm(v))

    solver = Solver()
    Z: Dict[str, Dict[str, ArithRef]] = {}
    for attr, vals in values.items():
        Z[attr] = {v: Int(f"{sanitize_sym(attr)}_{sanitize_sym(v)}") for v in vals}
        if len(Z[attr]) > 1:
            solver.add(Distinct(*Z[attr].values()))
        for vv in Z[attr].values():
            solver.add(vv >= 1, vv <= n)
    return solver, Z

def build_value_to_header(solution: Dict[str, Any]) -> Dict[str, str]:
    header = solution["header"][1:]
    rows = solution["rows"]
    v2h: Dict[str, str] = {}
    for i, h in enumerate(header):
        for r in rows:
            v2h[_norm(r[i + 1])] = h
    return v2h

def add_solution_constraints(solver: Solver, Z, solution: Dict[str, Any]) -> None:
    header = solution["header"]
    for r in solution["rows"]:
        house = int(r[0])
        for attr, val in zip(header[1:], r[1:]):
            v = _norm(val)
            if attr in Z and v in Z[attr]:
                solver.add(Z[attr][v] == house)

ALIASES = {
    "british": "brit",
    "danish": "dane",
    "first house": "house 1",
    "second house": "house 2",
    "third house": "house 3",
    "fourth house": "house 4",
    "fifth house": "house 5",
    "middle house": "house middle",
    "last house": "house last",
}

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[`.,:;]", "", s)
    for k, v in sorted(ALIASES.items(), key=lambda x: -len(x[0])):
        s = s.replace(k, v)
    s = s.replace("lives in", "in")
    s = s.replace("is in", "in")
    s = s.replace("in the", "in")

    s = s.replace("directly left of", "immediately left of")
    s = s.replace("directly right of", "immediately right of")
    s = s.replace("somewhere to the left of", "left of")
    s = s.replace("somewhere to the right of", "right of")
    s = s.replace("next to each other", "next to")

    s = s.replace("the person's child is named", "child")
    s = s.replace("person's child is named", "child")
    s = s.replace("child is named", "child")

    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@dataclass
class ValMatch:
    value: str
    start: int
    end: int

def build_match_vocabulary(v2h: Dict[str, str]) -> List[str]:
    vocab = set(v2h.keys())
    for alias, target in ALIASES.items():
        if target in v2h:
            vocab.add(alias)
    return sorted(vocab, key=len, reverse=True)

def find_values_in_text(text: str, vocab_sorted: List[str], v2h: Dict[str, str]) -> List[ValMatch]:
    matches: List[ValMatch] = []
    used = [False] * len(text)

    def can_use(a: int, b: int) -> bool:
        return not any(used[a:b])

    for v in vocab_sorted:
        for m in re.finditer(re.escape(v), text):
            a, b = m.start(), m.end()
            if can_use(a, b):
                matches.append(ValMatch(v, a, b))
                for i in range(a, b):
                    used[i] = True

    out: List[ValMatch] = []
    for m in sorted(matches, key=lambda x: x.start):
        canonical = ALIASES.get(m.value, m.value)
        if canonical in v2h:
            out.append(ValMatch(canonical, m.start, m.end))
    return out

HOUSE_RE = re.compile(r"\bhouse\s+(1|2|3|4|5|middle|last)\b")

def parse_house_index(text: str, n: int) -> Optional[int]:
    m = HOUSE_RE.search(text)
    if not m:
        return None
    token = m.group(1)
    if token == "middle":
        return (n + 1) // 2
    if token == "last":
        return n
    return int(token)

def H(val: str, Z: Dict[str, Dict[str, ArithRef]], v2h: Dict[str, str]) -> Optional[ArithRef]:
    col = v2h.get(val)
    if col is None:
        return None
    if col not in Z:
        return None
    return Z[col].get(val)

def constraints_from_sentence(sentence: str, Z, v2h: Dict[str, str], n: int, vocab: List[str]) -> List[ArithRef]:
    s = normalize_text(sentence)
    matches = find_values_in_text(s, vocab, v2h)
    out: List[ArithRef] = []

    # house-k
    hk = parse_house_index(s, n)
    if hk is not None and len(matches) >= 1:
        x = matches[0].value
        Hx = H(x, Z, v2h)
        if Hx is not None:
            return [Hx == hk]

    # next to
    if "next to" in s and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Abs(Ha - Hb) == 1]

    # immediate left/right
    if "immediately left of" in s and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha + 1 == Hb]

    if "immediately right of" in s and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha == Hb + 1]

    # somewhere left/right
    if " left of " in f" {s} " and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha < Hb]

    if " right of " in f" {s} " and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha > Hb]

    # fallback equality
    if " is " in f" {s} " and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha == Hb]

    return out


# =========================
# 4) Z3 metrics (with coverage & entailment)
# =========================

def z3_metrics(reasoning: List[str], solution: Dict[str, Any], clues: List[str]) -> Dict[str, float]:
    if not isinstance(solution, dict) or not isinstance(clues, list):
        return {"z3_sat": 0.0, "parse_cov": 0.0, "clue_sat": 0.0, "structure_score": 0.0, "completeness_score": 0.0, "reason_score": 0.0}

    base, Z = build_model(solution)
    v2h = build_value_to_header(solution)
    n = len(solution.get("rows", []))
    vocab = build_match_vocabulary(v2h)

    # parse clues -> constraints
    clue_constraints: List[ArithRef] = []
    parsed = 0
    for c in clues:
        cs = constraints_from_sentence(c, Z, v2h, n, vocab)
        if cs:
            parsed += 1
            clue_constraints.extend(cs)
    parse_cov = parsed / max(1, len(clues))

    # fixed-solution solver
    S_sol = Solver()
    S_sol.add(base.assertions())
    for c in clue_constraints:
        S_sol.add(c)
    add_solution_constraints(S_sol, Z, solution)

    z3_sat_score = 1.0 if S_sol.check() == sat else 0.0

    # entailment per parsed constraint
    if not clue_constraints:
        clue_sat = 0.0
    else:
        ok = 0
        for c in clue_constraints:
            S_sol.push()
            S_sol.add(Not(c))
            if S_sol.check() == unsat:
                ok += 1
            S_sol.pop()
        clue_sat = ok / len(clue_constraints)

    # reasoning monotonicity / contradiction vs fixed solution
    S_kb = Solver()
    S_kb.add(base.assertions())
    for c in clue_constraints:
        S_kb.add(c)

    valid = 0
    guessed = 0
    contrad = 0
    for sent in (reasoning or []):
        cs = constraints_from_sentence(sent, Z, v2h, n, vocab)
        if not cs:
            guessed += 1
            continue
        for c in cs:
            # entailed by KB?
            S_kb.push()
            S_kb.add(Not(c))
            entailed = (S_kb.check() == unsat)
            S_kb.pop()
            if entailed:
                valid += 1
                S_kb.add(c)
                continue

            # contradiction w.r.t fixed solution?
            S_sol.push()
            S_sol.add(c)
            bad = (S_sol.check() == unsat)
            S_sol.pop()
            if bad:
                contrad += 1
            else:
                guessed += 1

    total = valid + guessed + contrad
    reason_score = 0.0 if total == 0 else max(-1.0, min(1.0, (valid - guessed - 2 * contrad) / total))

    return {
        "z3_sat": float(z3_sat_score),
        "parse_cov": float(parse_cov),
        "clue_sat": float(clue_sat),
        "structure_score": float(structure_score(solution)),
        "completeness_score": float(completeness_score(solution)),
        "reason_score": float(reason_score),
    }


# =========================
# 5) Final reward (fix weights + improve puzzle-acc)
# =========================

def compute_reward(
    reasoning: List[str],
    solution: Dict[str, Any],
    clues: List[str],
    gt_solution: Optional[Dict[str, Any]] = None,
    gt_w: float = 0.7,
    z3_w: float = 0.3,
    gamma: float = 2.0,
    exact_bonus: float = 0.25,
    parse_cov_floor: float = 0.2,
) -> Dict[str, float]:
    """
    Returns dict with:
      reward_total, gt_reward, z3_reward, plus metrics.
    """

    # ---- GT component (shaped)
    exact = 0.0
    cell = 0.0
    gt_reward = 0.0
    if gt_solution is not None and isinstance(gt_solution, dict) and isinstance(solution, dict):
        exact, cell = cell_acc(solution, gt_solution)
        gt_reward = shaped_gt_reward(exact, cell, gamma=gamma, exact_bonus=exact_bonus)

    # ---- Z3 component (robust)
    m = z3_metrics(reasoning or [], solution or {}, clues or [])

    # If clue parsing is too low, don't let Z3 mislead training.
    # (This avoids “random Z3 gradients” that can reduce puzzle-acc.)
    cov_gate = 0.0 if m["parse_cov"] < parse_cov_floor else 1.0

    # Compose Z3 reward:
    # - must be satisfiable
    # - must satisfy as many parsed constraints as possible
    # - must be structurally valid + complete
    # - reasoning gives *small* bonus only (won't dominate)
    rpos = max(0.0, m["reason_score"])
    z3_reward = cov_gate * (
        m["z3_sat"]
        * (0.55 * m["clue_sat"] + 0.25 * m["structure_score"] + 0.20 * m["completeness_score"])
        * (0.85 + 0.15 * rpos)
    )

    # ---- IMPORTANT: if z3_sat=0 but parse_cov is decent, cap GT reward
    # This pushes toward globally consistent solutions (helps puzzle-acc).
    if cov_gate > 0 and m["parse_cov"] >= parse_cov_floor and m["z3_sat"] == 0.0:
        gt_reward *= 0.5  # soft penalty (not kill learning)

    # ---- combine with correct normalization
    total_w = float(gt_w + z3_w) if (gt_w + z3_w) > 0 else 1.0
    reward_total = (gt_w * gt_reward + z3_w * z3_reward) / total_w

    return {
        "reward_total": float(reward_total),
        "gt_reward": float(gt_reward),
        "z3_reward": float(z3_reward),
        "exact_match": float(exact),
        "cell_acc": float(cell),
        **{k: float(v) for k, v in m.items()},
    }


# =========================
# 6) Quick demos
# =========================

if __name__ == "__main__":
    # Demo: UNSAT case should get z3_sat=0 and GT penalty if cov_gate enabled
    clues = [
        "The person who loves yellow is the British person.",
        "The British person is in the second house.",
        "Eric is the person who loves yellow.",
    ]

    reasoning = [
        "The person who loves yellow is the British person.",
        "Eric is the person who loves yellow.",
        "So Eric is the British person.",
    ]

    pred_solution = {
        "header": ["House", "Name", "Color", "Nationality"],
        "rows": [
            ["1", "Eric", "yellow", "dane"],
            ["2", "Arnold", "red", "brit"],
        ],
    }

    gt_solution = {
        "header": ["House", "Name", "Color", "Nationality"],
        "rows": [
            ["1", "Arnold", "red", "dane"],
            ["2", "Eric", "yellow", "brit"],
        ],
    }

    out = compute_reward(reasoning, pred_solution, clues, gt_solution=gt_solution)
    print(json.dumps(out, indent=2))
