# z3_verifier_core.py
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from z3 import sat

from z3 import Solver, Int, Distinct, Abs, Not, unsat, ArithRef


# ============================================================
# 0) Utilities
# ============================================================

def _as_str(x: Any) -> str:
    if isinstance(x, list):
        return "|".join(map(str, x))
    if isinstance(x, dict):
        return json.dumps(x, sort_keys=True)
    return str(x)

def _sanitize_sym(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "v"

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# ============================================================
# 1) Structure score + table normalization
# ============================================================

def normalize_table(solution: dict) -> dict:
    header = solution.get("header", [])
    rows = solution.get("rows", [])
    H = len(header)

    out_rows = []
    for r in rows:
        if not isinstance(r, list):
            r = [_as_str(r)]
        r = [_as_str(c).strip() for c in r]
        if len(r) > H:
            r = r[:H]
        elif len(r) < H:
            r = r + [""] * (H - len(r))
        out_rows.append(r)

    return {"header": header, "rows": out_rows}


def structure_score(solution: dict) -> float:
    try:
        header = solution.get("header", [])
        rows = solution.get("rows", [])
        if not isinstance(header, list) or not isinstance(rows, list) or len(header) < 2:
            return 0.0
        if len(rows) == 0:
            return 0.0

        H = len(header)
        bad_shape = 0
        non_scalar = 0
        for r in rows:
            if not isinstance(r, list):
                bad_shape += 1
                continue
            if len(r) != H:
                bad_shape += 1
            for c in r:
                if isinstance(c, (list, dict, tuple)):
                    non_scalar += 1

        shape_pen = bad_shape / max(1, len(rows))
        nonscalar_pen = non_scalar / max(1, len(rows) * H)

        soln = normalize_table(solution)
        rows_n = soln["rows"]

        dup = 0
        total_attr_cells = 0
        for j in range(1, H):  # exclude House column
            col = [rows_n[i][j].strip().lower() for i in range(len(rows_n))]
            total_attr_cells += len(col)
            seen = set()
            for v in col:
                if v in seen:
                    dup += 1
                else:
                    seen.add(v)

        dup_pen = dup / max(1, total_attr_cells)

        score = 1.0 - (0.45 * shape_pen + 0.35 * dup_pen + 0.20 * nonscalar_pen)
        return clamp01(score)
    except Exception:
        return 0.0


# ============================================================
# 2) Build Z3 model from predicted solution
# ============================================================

def build_model(solution: dict):
    solution = normalize_table(solution)
    header = solution["header"]
    rows = solution["rows"]
    n = len(rows)

    attrs = header[1:]
    values = {a: set() for a in attrs}
    for r in rows:
        for a, v in zip(attrs, r[1:]):
            values[a].add(v.lower())

    solver = Solver()
    Z: Dict[str, Dict[str, ArithRef]] = {}

    for attr, vals in values.items():
        Z[attr] = {v: Int(f"{attr}_{_sanitize_sym(v)}") for v in vals}
        if Z[attr]:
            solver.add(Distinct(*Z[attr].values()))
            for sym in Z[attr].values():
                solver.add(sym >= 1, sym <= n)

    return solver, Z


def build_value_to_header(solution: dict) -> Dict[str, str]:
    solution = normalize_table(solution)
    header = solution["header"][1:]
    rows = solution["rows"]

    v2h: Dict[str, str] = {}
    for i, h in enumerate(header):
        for r in rows:
            v2h[r[i + 1].lower()] = h
    return v2h


def add_solution_constraints(solver: Solver, Z, solution: dict):
    solution = normalize_table(solution)
    header = solution["header"]
    for r in solution["rows"]:
        try:
            h = int(r[0])
        except Exception:
            continue
        for attr, val in zip(header[1:], r[1:]):
            v = val.lower()
            if attr in Z and v in Z[attr]:
                solver.add(Z[attr][v] == h)


def H(token: str, Z: Dict[str, Dict[str, ArithRef]], v2h: Dict[str, str]) -> Optional[ArithRef]:
    col = v2h.get(token)
    if col is None or col not in Z or token not in Z[col]:
        return None
    return Z[col][token]


# ============================================================
# 3) Normalization + value matching (multi-word safe)
# ============================================================

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

HOUSE_RE = re.compile(r"\bhouse\s+(1|2|3|4|5|middle|last)\b")


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

    s = s.replace("birthday is in", "birthday")
    s = s.replace("birthday month is", "birthday")

    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class ValMatch:
    value: str
    start: int
    end: int


def build_vocab(v2h: Dict[str, str]) -> List[str]:
    vocab = set(v2h.keys())
    for k, v in ALIASES.items():
        if v in v2h:
            vocab.add(k)
    return sorted(vocab, key=len, reverse=True)


def find_values(text: str, vocab_sorted: List[str], v2h: Dict[str, str]) -> List[ValMatch]:
    matches: List[ValMatch] = []
    used = [False] * len(text)

    def can_use(a, b):
        return not any(used[a:b])

    for v in vocab_sorted:
        pat = re.escape(v)
        for m in re.finditer(pat, text):
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


def parse_house_index(text: str, n: int) -> Optional[int]:
    m = HOUSE_RE.search(text)
    if not m:
        return None
    t = m.group(1)
    if t == "middle":
        return (n + 1) // 2
    if t == "last":
        return n
    return int(t)


# ============================================================
# 4) Parse natural language -> constraints
# ============================================================

def constraint_from_text(text: str, Z, v2h, n: int, vocab_sorted: List[str]) -> List[ArithRef]:
    s = normalize_text(text)
    matches = find_values(s, vocab_sorted, v2h)

    tokens = [m.value for m in matches]
    seen = set()
    tokens = [t for t in tokens if not (t in seen or seen.add(t))]

    hk = parse_house_index(s, n)
    if hk is not None and len(tokens) >= 1:
        x = tokens[0]
        Hx = H(x, Z, v2h)
        if Hx is not None:
            return [Hx == hk]

    if " next to " in f" {s} " and len(tokens) >= 2:
        a, b = tokens[0], tokens[1]
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Abs(Ha - Hb) == 1]

    if " immediately left of " in f" {s} " and len(tokens) >= 2:
        a, b = tokens[0], tokens[1]
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha + 1 == Hb]

    if " immediately right of " in f" {s} " and len(tokens) >= 2:
        a, b = tokens[0], tokens[1]
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha == Hb + 1]

    if " left of " in f" {s} " and len(tokens) >= 2:
        a, b = tokens[0], tokens[1]
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha < Hb]

    if " right of " in f" {s} " and len(tokens) >= 2:
        a, b = tokens[0], tokens[1]
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha > Hb]

    # equality fallback
    if " is " in f" {s} " and len(tokens) >= 2:
        a, b = tokens[0], tokens[1]
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            return [Ha == Hb]

    return []


def extract_constraints(clues: List[str], Z, v2h, n: int, debug=False):
    vocab_sorted = build_vocab(v2h)
    all_constraints: List[ArithRef] = []
    parsed_clues = 0

    for clue in clues:
        cs = constraint_from_text(clue, Z, v2h, n, vocab_sorted)
        if cs:
            parsed_clues += 1
            all_constraints.extend(cs)

        if debug:
            s = normalize_text(clue)
            toks = [m.value for m in find_values(s, vocab_sorted, v2h)]
            print("\n[CLUE]", clue)
            print("[NORM]", s)
            print("[TOKS]", toks)
            print("[N_CONS]", len(cs))

    return all_constraints, parsed_clues


# ============================================================
# 5) New CORE method: reasoning, solution, clues -> components
# ============================================================

def compute_z3_components(reasoning: List[str], solution: dict, clues: List[str], debug=False) -> dict:
    """
    Core method requested:
      inputs: reasoning(list[str]), solution(dict table), clues(list[str])
      outputs: clue_sat, parse_cov, structure_score, reason_score, z3_effective

    RL-safe: never raises.
    """
    try:
        solution = normalize_table(solution)
        n = len(solution["rows"])
        if n <= 0:
            return {
                "structure_score": 0.0,
                "parse_cov": 0.0,
                "clue_sat": 0.0,
                "reason_score": 0.0,
                "z3_effective": 0.0,
                "parsed_constraints": 0,
                "parsed_clues": 0,
            }

        s_score = structure_score(solution)

        base, Z = build_model(solution)
        v2h = build_value_to_header(solution)
        clue_constraints, parsed_clues = extract_constraints(clues, Z, v2h, n, debug=debug)

        # parse_cov (by clue)
        parse_cov = parsed_clues / max(1, len(clues))

        # clue_sat = entailment of each parsed constraint by the predicted solution assignment
        S_sol = Solver()
        S_sol.add(base.assertions())
        add_solution_constraints(S_sol, Z, solution)

        if len(clue_constraints) == 0:
            clue_sat = 0.0
        else:
            sat_count = 0
            for c in clue_constraints:
                S_sol.push()
                S_sol.add(Not(c))
                entailed = (S_sol.check() == unsat)  # UNSAT => entailed
                S_sol.pop()
                if entailed:
                    sat_count += 1
            clue_sat = sat_count / len(clue_constraints)

        # reason_score (stepwise)
        # KB = base domain + clue constraints
        S_kb = Solver()
        S_kb.add(base.assertions())
        for c in clue_constraints:
            S_kb.add(c)

        vocab_sorted = build_vocab(v2h)

        # also need S_sol for contradiction checks
        valid = 0
        guessed = 0
        contradicted = 0

        for sent in reasoning:
            cs = constraint_from_text(sent, Z, v2h, n, vocab_sorted)
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
                    S_kb.add(c)  # monotonic acceptance
                    continue

                # contradict predicted solution assignment?
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
            r_score = 0.0
        else:
            r_score = (valid - guessed - 2 * contradicted) / tot
            r_score = max(-1.0, min(1.0, float(r_score)))

        # Effective z3 score in [0,1]
        r_pos = max(0.0, r_score)
        z3_eff = clamp01(clue_sat * parse_cov * s_score)
        z3_eff = clamp01(z3_eff * (0.75 + 0.25 * r_pos))

        return {
            "structure_score": float(s_score),
            "parse_cov": float(parse_cov),
            "clue_sat": float(clue_sat),
            "reason_score": float(r_score),
            "z3_effective": float(z3_eff),
            "parsed_constraints": int(len(clue_constraints)),
            "parsed_clues": int(parsed_clues),
        }

    except Exception:
        return {
            "structure_score": 0.0,
            "parse_cov": 0.0,
            "clue_sat": 0.0,
            "reason_score": 0.0,
            "z3_effective": 0.0,
            "parsed_constraints": 0,
            "parsed_clues": 0,
        }

def compute_z3_components_v2(reasoning: List[str], solution: dict, clues: List[str], debug=False) -> dict:
    try:
        solution = normalize_table(solution)
        n = len(solution["rows"])
        if n <= 0:
            return {"structure_score": 0.0, "parse_cov": 0.0, "clue_sat": 0.0, "reason_score": 0.0,
                    "z3_effective": 0.0, "z3_sat": 0.0, "parsed_constraints": 0, "parsed_clues": 0}

        s_score = structure_score(solution)

        base, Z = build_model(solution)
        v2h = build_value_to_header(solution)
        clue_constraints, parsed_clues = extract_constraints(clues, Z, v2h, n, debug=debug)

        parse_cov = parsed_clues / max(1, len(clues))

        # Solver with predicted assignment
        S_sol = Solver()
        S_sol.add(base.assertions())
        add_solution_constraints(S_sol, Z, solution)

        # NEW: global satisfiable check (base + clues + prediction)
        S_all = Solver()
        S_all.add(base.assertions())
        for c in clue_constraints:
            S_all.add(c)
        add_solution_constraints(S_all, Z, solution)
        z3_sat = 1.0 if S_all.check() == sat else 0.0

        # clue_sat: entailment of each parsed constraint by predicted assignment
        if len(clue_constraints) == 0:
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

        # reason_score (unchanged)
        S_kb = Solver()
        S_kb.add(base.assertions())
        for c in clue_constraints:
            S_kb.add(c)

        vocab_sorted = build_vocab(v2h)

        valid = guessed = contradicted = 0
        for sent in reasoning:
            cs = constraint_from_text(sent, Z, v2h, n, vocab_sorted)
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
            r_score = 0.0
        else:
            r_score = (valid - guessed - 2 * contradicted) / tot
            r_score = max(-1.0, min(1.0, float(r_score)))

        r_pos = max(0.0, r_score)

        # UPDATED: include z3_sat so unsat predictions get punished hard
        z3_eff = clamp01(z3_sat * clue_sat * parse_cov * s_score)
        z3_eff = clamp01(z3_eff * (0.75 + 0.25 * r_pos))

        return {
            "structure_score": float(s_score),
            "parse_cov": float(parse_cov),
            "clue_sat": float(clue_sat),
            "reason_score": float(r_score),
            "z3_effective": float(z3_eff),
            "z3_sat": float(z3_sat),
            "parsed_constraints": int(len(clue_constraints)),
            "parsed_clues": int(parsed_clues),
        }

    except Exception as e:
        print(f"Error calculating Reasoning and Clues: {e}")
        return {"structure_score": 0.0, "parse_cov": 0.0, "clue_sat": 0.0, "reason_score": 0.0,
                "z3_effective": 0.0, "z3_sat": 0.0, "parsed_constraints": 0, "parsed_clues": 0}



def combined_reward_v2(
    gt_score: float,
    z3_eff: float,
    reason_score_val: float,
    parse_cov: float = 1.0,           # add this
    acc_w: float = 0.70,
    z3_w: float = 0.30,
    gate_floor: float = 0.60,         # higher floor = safer
    cov_floor: float = 0.20,          # below this, Z3 is unreliable
    hard_fail_penalty: float = 0.50,  # applied only when Z3 is reliable and bad
) -> float:
    """
    Stable 70/30 mix:
    - Z3 influences reward strongly only when Z3 is reliable (parse_cov high).
    - Avoids crushing GT due to parsing failures.
    - Penalizes clearly inconsistent solutions (reliable unsat cases).
    """

    gt = clamp01(float(gt_score))
    z3 = clamp01(float(z3_eff))
    rpos = clamp01(max(0.0, float(reason_score_val)))
    cov = clamp01(float(parse_cov))

    # Reliability of Z3 based on parse coverage
    z3_reliable = 1.0 if cov >= cov_floor else 0.0

    # Gate signal: mainly Z3, with small reasoning bonus
    gate_signal = clamp01(0.90 * z3 + 0.10 * rpos)

    # If Z3 not reliable, don't gate GT much (set gate near 1)
    if z3_reliable < 1.0:
        gate = 1.0
    else:
        # soft gate with safer floor
        gate = gate_floor + (1.0 - gate_floor) * gate_signal

    # Base weighted mix (stable)
    total_w = acc_w + z3_w
    base = (acc_w * gt + z3_w * gate_signal) / (total_w if total_w > 0 else 1.0)

    # Apply gate only to GT part (softly)
    mixed = (acc_w * (gt * gate) + z3_w * gate_signal) / (total_w if total_w > 0 else 1.0)

    # Hard fail penalty only when Z3 is reliable AND very bad
    if z3_reliable >= 1.0 and z3 < 0.05:
        mixed *= (1.0 - hard_fail_penalty)

    return clamp01(mixed)

def combined_reward(gt_score: float, z3_eff: float, reason_score_val: float, acc_w: float = 0.70, z3_w: float = 0.30,
                    gate_floor: float = 0.15) -> float:
    """
    70/30 GT/Z3 with a soft gate (prevents GT-only hacks when logic is bad).
    """
    gt_score = float(gt_score)
    z3_eff = float(z3_eff)
    r_pos = max(0.0, float(reason_score_val))

    gate_signal = clamp01(0.85 * z3_eff + 0.15 * r_pos)
    gate = gate_floor + (1.0 - gate_floor) * gate_signal

    total = acc_w * (gt_score * gate) + z3_w * gate_signal
    return clamp01(total)

def combined_reward_simple_gt_z3(
    gt_score: float,
    z3_reward: float,
    parse_cov: float = 1.0,
    gt_w: float = 0.70,
    z3_w: float = 0.30,
    cov_floor: float = 0.20,
    z3_when_unreliable: float = 0.0,
    use_soft_gate: bool = True,
    gate_floor: float = 0.85,   # keep high so GT isn't crushed
) -> float:
    """
    Simple stable reward:
      - reward = normalized weighted mix of GT and Z3
      - if Z3 parse coverage is low, treat Z3 as unreliable (set it to z3_when_unreliable)
      - optional very soft gate on GT when Z3 is reliable and low

    Omits reasoning and structure entirely.
    """

    gt = clamp01(gt_score)
    z3 = clamp01(z3_reward)
    cov = clamp01(parse_cov)

    # Reliability of Z3 depends only on parse coverage
    reliable = (cov >= cov_floor)

    if not reliable:
        z3_eff = clamp01(z3_when_unreliable)
        gate = 1.0
    else:
        z3_eff = z3
        gate = (gate_floor + (1.0 - gate_floor) * z3_eff) if use_soft_gate else 1.0

    total_w = float(gt_w + z3_w) if (gt_w + z3_w) > 0 else 1.0
    reward = (gt_w * (gt * gate) + z3_w * z3_eff) / total_w

    return float(clamp01(reward))
def boundary_penalized_reward(
    gt_score: float,
    z3_sat: float,
    clue_sat: float = 1.0,
    parse_cov: float = 1.0,
    # Optional extra penalty channels (set to 1.0 if omitted)
    reason_score: Optional[float] = None,
    structure_score: Optional[float] = None,
    # Reliability / gating
    cov_floor: float = 0.20,
    # Strength of penalty (bigger => harsher)
    alpha: float = 2.5,
    # How much to trust each channel inside penalty
    w_z3: float = 0.55,
    w_clue: float = 0.35,
    w_parse: float = 0.10,
    # Optional "etc" weights (kept small by default)
    w_reason: float = 0.00,      # set >0 to use
    w_struct: float = 0.00,      # set >0 to use
    # Shape of GT -> penalty coupling (bigger => harsher when GT is in mid region)
    gamma: float = 2.0,
) -> float:
    """
    Guarantees:
      GT=0 -> reward=0
      GT=1 -> reward=1

    For 0<GT<1:
      reward = GT * penalty
      where penalty heavily shrinks when solver signals are bad,
      especially in the middle region.

    Notes:
      - z3_sat/clue_sat/parse_cov are expected in [0,1]
      - reason_score if provided may be [-1,1]; we only use its positive part
      - structure_score if provided in [0,1]
    """
    gt = clamp01(gt_score)

    # Hard boundary conditions
    if gt <= 0.0:
        return 0.0
    if gt >= 1.0:
        return 1.0

    z3 = clamp01(z3_sat)
    clue = clamp01(clue_sat)
    cov = clamp01(parse_cov)

    # If parsing is unreliable, do not punish based on Z3/clues strongly (avoid false penalties)
    reliable = 1.0 if cov >= cov_floor else 0.0

    # Optional channels
    if reason_score is None:
        rpos = 1.0  # neutral
    else:
        rpos = clamp01(max(0.0, float(reason_score)))  # only positive contributes

    if structure_score is None:
        st = 1.0
    else:
        st = clamp01(structure_score)

    # Normalize weights over enabled channels (so you can toggle reason/struct cleanly)
    # Always include z3/clue/parse, but scale their effect by reliability
    # Effective signals: if not reliable, use neutral values (1.0) to avoid penalizing
    z3_eff = z3 if reliable else 1.0
    clue_eff = clue if reliable else 1.0
    cov_eff = cov  # even if low, this itself is the reliability indicator

    # Weight sum (include optional channels only if their weight > 0)
    w_sum = w_z3 + w_clue + w_parse
    if w_reason > 0.0:
        w_sum += w_reason
    if w_struct > 0.0:
        w_sum += w_struct
    if w_sum <= 0:
        w_sum = 1.0

    # Combined correctness signal in [0,1]
    S = (w_z3 * z3_eff + w_clue * clue_eff + w_parse * cov_eff) / w_sum
    if w_reason > 0.0:
        S = (S * (w_sum - w_reason) + w_reason * rpos) / w_sum
    if w_struct > 0.0:
        S = (S * (w_sum - w_struct) + w_struct * st) / w_sum

    S = clamp01(S)

    # Heavy penalty: (S ** alpha) pushes down aggressively when S<1
    # Couple penalty to GT mid-region: factor peaks around 0.5, small near 0 or 1
    mid = 4.0 * gt * (1.0 - gt)     # in [0,1], maximum at gt=0.5
    coupling = 1.0 + (mid ** gamma) # in [1,2] (if gamma=2, still smooth)

    penalty = (S ** (alpha * coupling))

    reward = gt * penalty
    return float(clamp01(reward))
# ============================================================
# 6) Quick demo
# ============================================================

if __name__ == "__main__":
    clues = [
        "The person who loves yellow is the British person.",
        "The British person is in the second house.",
        "Eric is the person who loves yellow."
    ]

    reasoning= [
        "Arnold is fixed in House 1 by Clue 5.",
        "The photography enthusiast is in House 2 by Clue 6.",
        "Peter is in House 2 by Clue 3.",
        "Eric is in House 1 by Clue 4.",
        "The milk drinker must be in House 3 by Clue 1."
    ]
    solution= {
        "header": ["House", "Name", "Drink", "Hobby"],
        "rows": [
            ["1", "Arnold", "tea", "gardening"],
            ["2", "Peter", "milk", "cooking"],
            ["3", "Eric", "water", "gardening"]
        ]
    }

    '''
    comps = compute_z3_components(reasoning, solution, clues, debug=False)
    print("\ncomponents:", json.dumps(comps, indent=2))

    gt_score_example = 0
    total = combined_reward(gt_score_example, comps["z3_effective"], comps["reason_score"])
    print("\ncombined_reward:", total)

    gt_score_example = 0
    total = combined_reward_v2(gt_score_example, comps["z3_effective"], comps["reason_score"])
    print("\ncombined_reward v2:", total)
    
    
    gt_score_example = 0
    total = combined_reward_simple_gt_z3(gt_score_example, comps["z3_effective"], comps["reason_score"])
    print("\ncombined_reward v3:", total)

    gt_score_example = 1.0
    total = boundary_penalized_reward(gt_score_example, comps["z3_effective"], comps["reason_score"])
    print("\ncombined_reward v3:", total)
    '''

    gt_score_example = 1.0
    total = compute_z3_components_v2(reasoning,solution, clues)
    print("\ncombined_reward v3:", total)


