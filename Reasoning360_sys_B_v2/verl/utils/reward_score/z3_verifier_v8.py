# z3_reward_v5_no_gt.py
import json
import re
import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Solver, Int, Distinct, Abs, Not, sat, unsat, ArithRef


# =========================
# 0) Parsing (optional utility)
# =========================

def parse_answer_tag(s: str) -> Optional[str]:
    m = re.search(r"<answer>([\s\S]*?)</answer>", s, re.DOTALL)
    return m.group(1).strip() if m else None

def _repair_jsonish(chunk: str) -> str:
    c = chunk.strip()
    c = re.sub(r'(".*?"|\'.*?\'|[A-Za-z_][A-Za-z0-9_ ]*)\s*=\s*', r"\1: ", c)
    c = re.sub(r"(?<!\\)'([^'\\]*)'", r'"\1"', c)
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

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    chunk = _first_balanced_obj(s) or s
    repaired = _repair_jsonish(chunk)

    try:
        obj = json.loads(repaired)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

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

def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1.0 else x)

def structure_score(solution: Dict[str, Any]) -> float:
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
    if not isinstance(solution, dict):
        return 0.0
    header = solution.get("header")
    rows = solution.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list) or len(header) < 2 or len(rows) < 1:
        return 0.0

    n = len(rows)
    m = len(header)

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

    dom = 0.0
    for j in range(1, m):
        col = [_norm(r[j]) for r in rows if isinstance(r, list) and len(r) == m]
        uniq = len(set(col))
        dom += min(1.0, uniq / max(1, n))
    dom = dom / max(1, (m - 1))

    return max(0.0, min(1.0, 0.5 * cell_fill + 0.5 * dom))


# =========================
# 2) Z3 model + clue parsing
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

    return []


# =========================
# 3) Z3 metrics + Z3-only reward
# =========================

def z3_metrics(reasoning: List[str], solution: Dict[str, Any], clues: List[str]) -> Dict[str, float]:
    if not isinstance(solution, dict) or not isinstance(clues, list):
        return {
            "z3_sat": 0.0,
            "parse_cov": 0.0,
            "clue_sat": 0.0,
            "structure_score": 0.0,
            "completeness_score": 0.0,
            "reason_score": 0.0,
        }

    base, Z = build_model(solution)
    v2h = build_value_to_header(solution)
    n = len(solution.get("rows", []))
    vocab = build_match_vocabulary(v2h)

    clue_constraints: List[ArithRef] = []
    parsed_clues = 0
    for c in clues:
        cs = constraints_from_sentence(c, Z, v2h, n, vocab)
        if cs:
            parsed_clues += 1
            clue_constraints.extend(cs)

    parse_cov = parsed_clues / max(1, len(clues))

    S_sol = Solver()
    S_sol.add(base.assertions())
    for c in clue_constraints:
        S_sol.add(c)
    add_solution_constraints(S_sol, Z, solution)

    z3_sat_score = 1.0 if S_sol.check() == sat else 0.0

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
            S_kb.push()
            S_kb.add(Not(c))
            entailed = (S_kb.check() == unsat)
            S_kb.pop()
            if entailed:
                valid += 1
                S_kb.add(c)
                continue

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


def compute_z3_reward(
    reasoning: List[str],
    solution: Dict[str, Any],
    clues: List[str],
    parse_cov_floor: float = 0.2,
) -> Dict[str, float]:
    """
    Z3-only reward module.
    Returns: {z3_reward, ...metrics}
    """
    m = z3_metrics(reasoning or [], solution or {}, clues or [])

    # gate: if clue parsing coverage is too low, don't trust the Z3 reward
    cov_gate = 0.0 if m["parse_cov"] < parse_cov_floor else 1.0
    rpos = max(0.0, m["reason_score"])

    z3_reward = cov_gate * (
        m["z3_sat"]
        * (0.55 * m["clue_sat"] + 0.25 * m["structure_score"] + 0.20 * m["completeness_score"])
        * (0.85 + 0.15 * rpos)
    )

    m["z3_reward"] = float(_clamp01(z3_reward))
    return m


def combine_external_gt_z3(gt_score: float, z3_reward: float, gt_w: float = 0.7, z3_w: float = 0.3) -> float:
    """
    If you want a combined reward, but GT comes from another module.
    """
    total_w = float(gt_w + z3_w) if (gt_w + z3_w) > 0 else 1.0
    return float(_clamp01((gt_w * gt_score + z3_w * z3_reward) / total_w))


# =========================
# 4) Quick demo
# =========================

if __name__ == "__main__":
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
    solution = {
        "header": ["House", "Name", "Color", "Nationality"],
        "rows": [
            ["1", "Eric", "yellow", "dane"],
            ["2", "Arnold", "red", "brit"],
        ],
    }

    out = compute_z3_reward(reasoning, solution, clues, parse_cov_floor=0.1)
    print(json.dumps(out, indent=2))

    # Example combining with external GT
    external_gt_score = 0.8
    final = combine_external_gt_z3(external_gt_score, out["z3_reward"], gt_w=0.7, z3_w=0.3)
    print("combined_final_reward:", final)
