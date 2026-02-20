# zebra_reward_demo.py
# pip install z3-solver

from __future__ import annotations

import re
import json
import ast
import math
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Solver, Int, Distinct, And, Abs, Not, sat, unsat


# ============================================================
# 0) Utilities
# ============================================================

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def smoothstep(t: float) -> float:
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)

def print_exception_with_line(e: BaseException):
    print("\n[EXCEPTION]")
    traceback.print_exc()


# ============================================================
# 1) schedule(step, total_steps) for YOUR curriculum function
# ============================================================

def schedule(step: int,
             total_steps: int,
             cell_phase: float = 0.35,
             mix_phase: float = 0.25):
    """
    Returns:
      w_puz, alpha, cov_floor, parse_bonus_w

    - Early: emphasize cell accuracy
    - Mid:   smoothly transition to puzzle accuracy
    - Late:  mostly puzzle accuracy + stronger Z3 penalties
    """
    if total_steps <= 1:
        t = 1.0
    else:
        t = float(step) / float(total_steps - 1)
    t = clamp01(t)

    # how much to trust puzzle acc vs cell acc in GT anchor
    if t < cell_phase:
        w_puz = 0.05
    elif t < cell_phase + mix_phase:
        u = (t - cell_phase) / max(1e-9, mix_phase)
        w_puz = lerp(0.05, 0.95, smoothstep(u))
    else:
        w_puz = 0.95

    # penalty strength: grows over time
    alpha = lerp(1.2, 4.0, smoothstep(t))

    # minimum parse coverage to trust Z3
    cov_floor = lerp(0.10, 0.30, smoothstep(t))

    # parse bonus weight: helpful early, minimal late
    parse_bonus_w = lerp(0.25, 0.02, smoothstep(t))

    return float(w_puz), float(alpha), float(cov_floor), float(parse_bonus_w)


# ============================================================
# 2) Your curriculum reward (unchanged, uses schedule above)
# ============================================================
def curriculum_reward_cell_to_puzzle_only(
    cell_acc: float,
    puzzle_acc: float,
    epoch: int,
    total_epochs: int,
    cell_beta: float = 2.0,
    gamma_mid: float = 2.0,
) -> float:
    """
    Curriculum based ONLY on cell accuracy and puzzle accuracy.

    - Early: reward shaped cell accuracy
    - Late: reward puzzle accuracy
    - Smooth interpolation over epochs
    - Midpoint sharpening to encourage decisive solutions
    """

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    c = clamp01(cell_acc)
    p = clamp01(puzzle_acc)

    # --- Terminal shortcuts ---
    if p >= 1.0:
        return 1.0
    if p <= 0.0 and c <= 0.0:
        return 0.0

    # --- Curriculum weight (cell → puzzle) ---
    t = clamp01(epoch / max(1, total_epochs))
    w_puz = t ** 1.5  # slow early, faster late shift

    # --- Shape cell accuracy ---
    c_shaped = c ** max(1e-6, float(cell_beta))

    # --- Ground-truth anchor ---
    gt_anchor = clamp01((1.0 - w_puz) * c_shaped + w_puz * p)

    # --- Midpoint sharpening (encourage commitment) ---
    mid = 4.0 * gt_anchor * (1.0 - gt_anchor)
    sharpen = 1.0 + (mid ** max(1e-6, float(gamma_mid)))

    # --- Final reward ---
    r = clamp01(gt_anchor * sharpen)

    return float(r)

def curriculum_reward_cell_to_puzzle(
    cell_acc: float,
    puzzle_acc: float,
    z3_sat: float,
    clue_sat: float,
    parse_cov: float,
    epoch: int,
    total_epochs: int,
    cell_beta: float = 2.0,
    gamma_mid: float = 2.0,
) -> float:
    """
    - Early: GT anchor ~ shaped cell accuracy
    - Late:  GT anchor ~ puzzle accuracy
    - Z3/clue penalties ramp up over time
    - small parse bonus early
    """
    c = clamp01(cell_acc)
    p = clamp01(puzzle_acc)

    if p >= 1.0:
        return 1.0
    if p <= 0.0 and c <= 0.0:
        return 0.0

    w_puz, alpha, cov_floor, parse_bonus_w = schedule(epoch, total_epochs)

    c_shaped = c ** max(1e-6, float(cell_beta))
    gt_anchor = clamp01((1.0 - w_puz) * c_shaped + w_puz * p)

    cov = clamp01(parse_cov)
    reliable = 1.0 if cov >= cov_floor else 0.0

    z3 = clamp01(z3_sat)
    clue = clamp01(clue_sat)
    logic = (0.7 * z3 + 0.3 * clue) if reliable else 1.0

    mid = 4.0 * gt_anchor * (1.0 - gt_anchor)
    coupling = 1.0 + (mid ** max(1e-6, float(gamma_mid)))

    penalty = logic ** (alpha * coupling)

    r = gt_anchor * penalty
    r = clamp01(r + parse_bonus_w * (cov ** 1.0))
    return float(clamp01(r))


# ============================================================
# 3) Robust parsing of <answer> JSON + relaxed parsing
# ============================================================

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def _try_parse_first_obj_relaxed(text: str) -> Optional[Dict[str, Any]]:
    """
    Parses dict from text:
      1) json.loads
      2) scan for first {...} and json.loads
      3) ast.literal_eval relaxed (supports single quotes, = -> :)
    """
    if not text or not isinstance(text, str):
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    starts = [m.start() for m in re.finditer(r"\{", text)]
    for st in starts:
        for ed in range(len(text), st, -1):
            if text[ed - 1] != "}":
                continue
            chunk = text[st:ed]

            # strict json
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

            # relaxed literal
            relaxed = chunk
            relaxed = re.sub(r'("([^"]+)")\s*=\s*', r'\1: ', relaxed)
            relaxed = re.sub(r"('([^']+)')\s*=\s*", r"'\1': ", relaxed)
            relaxed = relaxed.replace("null", "None").replace("true", "True").replace("false", "False")
            try:
                obj = ast.literal_eval(relaxed)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None

def parse_answer(solution_str: str) -> Optional[Tuple[List[str], Dict[str, Any]]]:
    if not solution_str:
        return None

    m = ANSWER_RE.search(solution_str)
    content = m.group(1) if m else solution_str

    obj = _try_parse_first_obj_relaxed(content)
    if not obj or "solution" not in obj:
        return None

    reasoning = obj.get("reasoning", [])
    sol = obj.get("solution", None)

    if isinstance(reasoning, str):
        reasoning = [reasoning]
    if not isinstance(reasoning, list):
        reasoning = []
    reasoning = [str(x).strip() for x in reasoning if str(x).strip()]

    if not isinstance(sol, dict):
        return None

    return reasoning, sol


# ============================================================
# 4) Table normalization + safe "position" handling
#    Fixes: int('Arnold') etc.
# ============================================================

HOUSE_LIKE = {"house", "position", "index", "pos"}

def _norm_atom(x: Any) -> str:
    return str(x).strip().lower()

def normalize_table(sol: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(sol, dict):
        return None
    header = sol.get("header")
    rows = sol.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list) or not header:
        return None
    header_n = [str(h).strip() for h in header]
    rows_n: List[List[str]] = []
    for r in rows:
        if not isinstance(r, list):
            return None
        rows_n.append([str(x).strip() for x in r])
    return {"header": header_n, "rows": rows_n}

def infer_house_col(header: List[str], rows: List[List[str]]) -> Optional[int]:
    if not header or not rows:
        return None

    for i, h in enumerate(header[:2]):  # check first 2 cols only
        if _norm_atom(h) in HOUSE_LIKE:
            ok = 0
            for r in rows:
                if len(r) > i and re.fullmatch(r"\d+", r[i].strip()):
                    ok += 1
            if ok >= max(1, int(len(rows) * 0.6)):
                return i
    return None

def row_position(row: List[str], row_idx: int, house_col: Optional[int]) -> int:
    if house_col is not None and len(row) > house_col:
        v = row[house_col].strip()
        if re.fullmatch(r"\d+", v):
            return int(v)
    return int(row_idx + 1)

def structure_score(sol: Dict[str, Any]) -> float:
    header = sol.get("header", [])
    rows = sol.get("rows", [])
    if not header or not rows:
        return 0.0
    if len(header) < 2:
        return 0.2
    w = len(header)
    good = sum(1 for r in rows if isinstance(r, list) and len(r) == w)
    return clamp01(good / max(1, len(rows)))


# ============================================================
# 5) Build Z3 model from predicted solution table
# ============================================================

def build_value_to_header(sol: Dict[str, Any]) -> Dict[str, str]:
    header = sol["header"]
    rows = sol["rows"]
    hc = infer_house_col(header, rows)
    v2h: Dict[str, str] = {}
    for ci, col in enumerate(header):
        if hc is not None and ci == hc:
            continue
        coln = _norm_atom(col)
        for r in rows:
            if len(r) <= ci:
                continue
            v2h[_norm_atom(r[ci])] = coln
    return v2h

def build_model(sol: Dict[str, Any]) -> Tuple[Solver, Dict[str, Dict[str, Any]]]:
    header = sol["header"]
    rows = sol["rows"]
    n = len(rows)

    base = Solver()
    hc = infer_house_col(header, rows)

    Z: Dict[str, Dict[str, Any]] = {}
    for ci, col in enumerate(header):
        if hc is not None and ci == hc:
            continue
        coln = _norm_atom(col)
        Z.setdefault(coln, {})

        values = []
        for r in rows:
            if len(r) > ci:
                values.append(_norm_atom(r[ci]))

        seen = set()
        uniq = []
        for v in values:
            if v not in seen:
                seen.add(v)
                uniq.append(v)

        for v in uniq:
            Z[coln][v] = Int(f"h_{coln}_{v}".replace(" ", "_"))

        vars_ = [Z[coln][v] for v in uniq]
        if vars_:
            base.add(And(*[And(x >= 1, x <= n) for x in vars_]))
            base.add(Distinct(*vars_))

    return base, Z

def add_solution_constraints(s: Solver, Z: Dict[str, Dict[str, Any]], sol: Dict[str, Any]) -> None:
    header = sol["header"]
    rows = sol["rows"]
    hc = infer_house_col(header, rows)

    for ri, r in enumerate(rows):
        pos = row_position(r, ri, hc)
        for ci, col in enumerate(header):
            if hc is not None and ci == hc:
                continue
            coln = _norm_atom(col)
            if coln not in Z:
                continue
            if len(r) <= ci:
                continue
            v = _norm_atom(r[ci])
            if v in Z[coln]:
                s.add(Z[coln][v] == pos)


# ============================================================
# 6) Natural language -> constraints (improved clue parser)
# ============================================================

ALIASES = {
    "directly left of": "immediately left of",
    "directly right of": "immediately right of",
    "somewhere to the left of": "left of",
    "somewhere to the right of": "right of",
    "next to each other": "next to",

    "the person's child is named": "child",
    "person's child is named": "child",
    "child is named": "child",

    "is in": "in",
    "in the": "in",
}

NUM_WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
HOUSE_RE = re.compile(r"\bhouse\s+(\d+|middle|last)\b")

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[`\"“”]", "", s)
    s = re.sub(r"[.,:;!?]", "", s)
    for k, v in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(k, v)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

@dataclass
class ValMatch:
    value: str
    start: int
    end: int

def build_vocab(v2h: Dict[str, str]) -> List[str]:
    return sorted(set(v2h.keys()), key=len, reverse=True)

def find_values_in_text(text: str, vocab_sorted: List[str], v2h: Dict[str, str]) -> List[ValMatch]:
    matches: List[ValMatch] = []
    used = [False] * len(text)

    def can_use(a: int, b: int) -> bool:
        return 0 <= a < b <= len(text) and not any(used[a:b])

    for v in vocab_sorted:
        pat = re.escape(v)
        for m in re.finditer(pat, text):
            a, b = m.start(), m.end()
            if can_use(a, b):
                matches.append(ValMatch(v, a, b))
                for i in range(a, b):
                    used[i] = True

    # sorted by occurrence order
    out: List[ValMatch] = []
    for m in sorted(matches, key=lambda x: x.start):
        if m.value in v2h:
            out.append(m)
    return out

def parse_house_index(text: str, n: int) -> Optional[int]:
    m = HOUSE_RE.search(text)
    if not m:
        return None
    tok = m.group(1)
    if tok == "middle":
        return (n + 1) // 2
    if tok == "last":
        return n
    if re.fullmatch(r"\d+", tok):
        k = int(tok)
        if 1 <= k <= n:
            return k
    return None

def k_between(text: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\s+houses?\s+between\b", text)
    if m:
        return int(m.group(1))
    m2 = re.search(r"\b(one|two|three|four|five)\s+houses?\s+between\b", text)
    if m2:
        return NUM_WORD.get(m2.group(1))
    m3 = re.search(r"\b(one|two|three|four|five)\s+house\s+between\b", text)
    if m3:
        return NUM_WORD.get(m3.group(1))
    return None

def H(value: str, Z: Dict[str, Dict[str, Any]], v2h: Dict[str, str]) -> Optional[Any]:
    col = v2h.get(value)
    if col is None:
        return None
    col = col.lower()
    return Z.get(col, {}).get(value)

def constraints_from_text(text: str,
                          Z: Dict[str, Dict[str, Any]],
                          v2h: Dict[str, str],
                          n: int,
                          vocab_sorted: List[str]) -> List[Any]:
    s = normalize_text(text)
    matches = find_values_in_text(s, vocab_sorted, v2h)
    if not matches:
        return []

    # X in house k
    hk = parse_house_index(s, n)
    if hk is not None:
        x = matches[0].value
        Hx = H(x, Z, v2h)
        if Hx is not None:
            return [Hx == hk]

    # k houses between A and B
    kb = k_between(s)
    if kb is not None and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Abs(Ha - Hb) == (kb + 1)]

    # next to
    if "next to" in s and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Abs(Ha - Hb) == 1]

    # immediately left/right
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

    # left/right (non-adjacent)
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

    # Equality fallback: "A is B" -> same position
    if " is " in f" {s} " and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha == Hb]

    return []

def extract_constraints(clues: List[str],
                        Z: Dict[str, Dict[str, Any]],
                        v2h: Dict[str, str],
                        n: int) -> Tuple[List[Any], int]:
    vocab_sorted = build_vocab(v2h)
    all_cs: List[Any] = []
    parsed = 0
    for clue in (clues or []):
        cs = constraints_from_text(clue, Z, v2h, n, vocab_sorted)
        if cs:
            parsed += 1
            all_cs.extend(cs)
    return all_cs, parsed


# ============================================================
# 7) Core method: compute_z3_components(reasoning, solution, clues)
# ============================================================

def compute_z3_components(reasoning: List[str],
                          solution: Dict[str, Any],
                          clues: List[str],
                          debug: bool = False) -> Dict[str, Any]:
    """
    outputs: structure_score, parse_cov, clue_sat, z3_sat, z3_effective, reason_score
    RL-safe: never raises.
    """
    try:
        sol = normalize_table(solution)
        if not sol:
            return {"structure_score": 0.0, "parse_cov": 0.0, "clue_sat": 0.0,
                    "reason_score": 0.0, "z3_sat": 0.0, "z3_effective": 0.0,
                    "parsed_constraints": 0, "parsed_clues": 0}

        n = len(sol["rows"])
        if n <= 0:
            return {"structure_score": 0.0, "parse_cov": 0.0, "clue_sat": 0.0,
                    "reason_score": 0.0, "z3_sat": 0.0, "z3_effective": 0.0,
                    "parsed_constraints": 0, "parsed_clues": 0}

        s_score = structure_score(sol)
        base, Z = build_model(sol)
        v2h = build_value_to_header(sol)

        clue_constraints, parsed_clues = extract_constraints(clues, Z, v2h, n)
        parse_cov = parsed_clues / max(1, len(clues or []))

        # base + predicted assignment
        S_sol = Solver()
        S_sol.add(base.assertions())
        add_solution_constraints(S_sol, Z, sol)

        # clue_sat by entailment under predicted assignment
        if not clue_constraints:
            clue_sat = 0.0
        else:
            sat_count = 0
            for c in clue_constraints:
                S_sol.push()
                S_sol.add(Not(c))
                entailed = (S_sol.check() == unsat)
                S_sol.pop()
                if entailed:
                    sat_count += 1
            clue_sat = sat_count / len(clue_constraints)

        # z3_sat: full consistency
        S_all = Solver()
        S_all.add(base.assertions())
        for c in clue_constraints:
            S_all.add(c)
        add_solution_constraints(S_all, Z, sol)
        z3_sat = 1.0 if S_all.check() == sat else 0.0

        # reasoning score (lightweight)
        # KB = base + clue constraints, check entail/contradiction wrt solution
        vocab_sorted = build_vocab(v2h)
        S_kb = Solver()
        S_kb.add(base.assertions())
        for c in clue_constraints:
            S_kb.add(c)

        valid = guessed = contradicted = 0
        for sent in (reasoning or []):
            cs = constraints_from_text(sent, Z, v2h, n, vocab_sorted)
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
                contr = (S_sol.check() == unsat)
                S_sol.pop()
                if contr:
                    contradicted += 1
                else:
                    guessed += 1

        tot = valid + guessed + contradicted
        if tot == 0:
            reason_score = 0.0
        else:
            reason_score = (valid - guessed - 2 * contradicted) / tot
            reason_score = max(-1.0, min(1.0, float(reason_score)))

        # z3_effective: conservative
        # (penalize if unsat; penalize low parse coverage; require reasonable structure)
        z3_eff = clamp01(z3_sat * clue_sat * parse_cov * s_score)

        return {
            "structure_score": float(s_score),
            "parse_cov": float(parse_cov),
            "clue_sat": float(clue_sat),
            "reason_score": float(reason_score),
            "z3_sat": float(z3_sat),
            "z3_effective": float(z3_eff),
            "parsed_constraints": int(len(clue_constraints)),
            "parsed_clues": int(parsed_clues),
        }

    except Exception as e:
        if debug:
            print_exception_with_line(e)
        return {"structure_score": 0.0, "parse_cov": 0.0, "clue_sat": 0.0,
                "reason_score": 0.0, "z3_sat": 0.0, "z3_effective": 0.0,
                "parsed_constraints": 0, "parsed_clues": 0}


# ============================================================
# 8) Demo: combine Z3 metrics with your curriculum reward
# ============================================================

def demo_one(title: str,
             llm_output: str,
             clues: List[str],
             cell_acc: float,
             puzzle_acc: float,
             epoch: int,
             total_epochs: int):
    parsed = parse_answer(llm_output)
    if not parsed:
        print(f"\n=== {title} ===")
        print("FAILED to parse <answer> JSON.")
        return

    reasoning, sol = parsed
    z3_out = compute_z3_components(reasoning, sol, clues, debug=True)

    reward = curriculum_reward_cell_to_puzzle(
        cell_acc=cell_acc,
        puzzle_acc=puzzle_acc,
        z3_sat=z3_out["z3_sat"],
        clue_sat=z3_out["clue_sat"],
        parse_cov=z3_out["parse_cov"],
        epoch=epoch,
        total_epochs=total_epochs,
    )

    print(f"\n=== {title} ===")
    print("Z3:", {k: z3_out[k] for k in ["parse_cov", "clue_sat", "z3_sat", "z3_effective", "structure_score"]})
    print(f"GT: cell_acc={cell_acc:.3f}, puzzle_acc={puzzle_acc:.3f}")
    print(f"curriculum(step={epoch}/{total_epochs}) reward = {reward:.4f}")


if __name__ == "__main__":
    # -----------------------------------
    # Example A: UNSAT-ish (contradicting child/colour constraints)
    # -----------------------------------
    clues_A = [
        "The person who loves yellow is the British person.",
        "The person who has black hair is the person's child is named Fred.",
        "The person who loves yellow is the person's child is named Bella.",
        "The British person is in the second house.",
        "Eric is the person who loves yellow.",
        "The person whose birthday is in April is Eric.",
    ]

    llm_A = """
    <answer>{
      "reasoning": [
        "Clue 1 tells us that the person who loves yellow is the British person.",
        "Eric is the person who loves yellow.",
        "The British person is in the second house."
      ],
      "solution": {
        "header": ["House", "Name", "Birthday", "Color", "Nationality", "HairColor", "Child"],
        "rows": [
          ["1", "Eric", "sept", "yellow", "dane", "black", "Fred"],
          ["2", "Arnold", "april", "red", "brit", "brown", "Bella"]
        ]
      }
    }</answer>
    """.strip()

    # Example B: SAT-ish 2-house neighbor case
    clues_B = [
        "The person who loves a carnations arrangement and Eric are next to each other.",
        "The person who loves a carnations arrangement is in the first house.",
        "The person who loves eating grilled cheese is in the first house.",
    ]

    llm_B = """
    <answer>{
      "reasoning": [
        "Carnations is in house 1 by clue 2.",
        "Eric must be in house 2 because carnations and Eric are next to each other."
      ],
      "solution": {
        "header": ["House", "Name", "Lunch", "Flower"],
        "rows": [
          ["1", "Arnold", "grilled cheese", "carnations"],
          ["2", "Eric", "pizza", "daffodils"]
        ]
      }
    }</answer>
    """.strip()

    # Simulated GT scores (you said you compute these externally)
    # total_steps could be e.g. total_train_steps; use any number.
    epoch = 5
    total_epochs = 100

    demo_one(
        title="Demo A (expect strong penalty if parsing is good and UNSAT)",
        llm_output=llm_A,
        clues=clues_A,
        cell_acc=0.60,
        puzzle_acc=0.0,
        epoch=epoch,
        total_epochs=total_epochs,
    )
    print()
    print()
    epoch = 95
    demo_one(
        title="Demo B (expect decent reward)",
        llm_output=llm_B,
        clues=clues_B,
        cell_acc=0.70,
        puzzle_acc=0.0,
        epoch=epoch,
        total_epochs=total_epochs,
    )