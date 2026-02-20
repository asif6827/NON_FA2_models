# z3_verifier_reasoning_solution_clues.py
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from z3 import Solver, Int, Distinct, Abs, Not, sat, unsat, ArithRef


# =========================
# 0) Helpers
# =========================

def sanitize_sym(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "v"

def _norm(x: Any) -> str:
    return str(x).strip().lower()

def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1.0 else x)


# =========================
# 1) Build Z3 model from predicted solution
# =========================

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


# =========================
# 2) Normalization + aliases
# =========================

ALIASES = {
    # nationality variants
    "british": "brit",
    "danish": "dane",
    # house words
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

    # child phrasing variants in your dataset
    s = s.replace("the person's child is named", "child")
    s = s.replace("person's child is named", "child")
    s = s.replace("child is named", "child")

    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =========================
# 3) Value matching (multi-word safe)
# =========================

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
        pattern = re.escape(v)
        for m in re.finditer(pattern, text):
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


# =========================
# 4) NL -> constraint(s)
# =========================

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
    if val not in Z[col]:
        return None
    return Z[col][val]

def _pair(matches: List[ValMatch]) -> Optional[Tuple[str, str]]:
    if len(matches) < 2:
        return None
    return matches[0].value, matches[1].value

def constraints_from_sentence(
    sentence: str,
    Z,
    v2h: Dict[str, str],
    n: int,
    vocab: List[str],
    debug: bool = False,
) -> List[ArithRef]:
    s = normalize_text(sentence)
    matches = find_values_in_text(s, vocab, v2h)

    if debug:
        print("\n[SENT]", sentence)
        print("[NORM]", s)
        print("[MATCHES]", [m.value for m in matches])

    out: List[ArithRef] = []

    # 1) house k
    house_k = parse_house_index(s, n)
    if house_k is not None and len(matches) >= 1:
        x = matches[0].value
        Hx = H(x, Z, v2h)
        if Hx is not None:
            out.append(Hx == house_k)
            return out

    # 2) next to
    if "next to" in s:
        p = _pair(matches)
        if p:
            a, b = p
            Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
            if Ha is not None and Hb is not None:
                out.append(Abs(Ha - Hb) == 1)
                return out

    # 3) immediately left/right
    if "immediately left of" in s:
        p = _pair(matches)
        if p:
            a, b = p
            Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
            if Ha is not None and Hb is not None:
                out.append(Ha + 1 == Hb)
                return out

    if "immediately right of" in s:
        p = _pair(matches)
        if p:
            a, b = p
            Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
            if Ha is not None and Hb is not None:
                out.append(Ha == Hb + 1)
                return out

    # 4) somewhere left/right
    if " left of " in f" {s} ":
        p = _pair(matches)
        if p:
            a, b = p
            Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
            if Ha is not None and Hb is not None:
                out.append(Ha < Hb)
                return out

    if " right of " in f" {s} ":
        p = _pair(matches)
        if p:
            a, b = p
            Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
            if Ha is not None and Hb is not None:
                out.append(Ha > Hb)
                return out

    # 5) equality fallback
    if " is " in f" {s} " and len(matches) >= 2:
        a, b = matches[0].value, matches[1].value
        Ha, Hb = H(a, Z, v2h), H(b, Z, v2h)
        if Ha is not None and Hb is not None:
            out.append(Ha == Hb)
            return out

    return out


def add_clue_constraints(
    solver: Solver,
    Z,
    clues: List[str],
    n: int,
    v2h: Dict[str, str],
    debug: bool = False
) -> int:
    vocab = build_match_vocabulary(v2h)
    added = 0
    for clue in clues:
        cs = constraints_from_sentence(clue, Z, v2h, n, vocab, debug=debug)
        for c in cs:
            solver.add(c)
            added += 1
    return added


# =========================
# 5) KEY ENTRYPOINT (requested):
#    takes reasoning, solution, clues
# =========================

def verify_reasoning_solution_clues_with_z3(
    reasoning: List[str],
    solution: Dict[str, Any],
    clues: List[str],
    debug: bool = False
) -> Dict[str, float]:
    """
    Returns:
      z3_sat:         1.0 if (clues + fixed solution) is satisfiable else 0.0
      parse_cov:      fraction of clues that became at least 1 constraint
      clue_sat:       fraction of parsed clue-constraints entailed by the fixed solution
      reason_score:   [-1,1] step score: valid - guessed - 2*contradicted / total
    """
    try:
        base, Z = build_model(solution)
        v2h = build_value_to_header(solution)
        n = len(solution["rows"])
        vocab = build_match_vocabulary(v2h)

        # --- build clue constraints (also keep list for clue_sat) ---
        clue_constraints: List[ArithRef] = []
        parsed_clues = 0
        for clue in clues:
            cs = constraints_from_sentence(clue, Z, v2h, n, vocab, debug=debug)
            if cs:
                parsed_clues += 1
                clue_constraints.extend(cs)

        parse_cov = parsed_clues / max(1, len(clues))

        # --- solver for fixed solution (used for SAT + entailment checks) ---
        S_sol = Solver()
        S_sol.add(base.assertions())
        for c in clue_constraints:
            S_sol.add(c)
        add_solution_constraints(S_sol, Z, solution)

        z3_sat = 1.0 if S_sol.check() == sat else 0.0

        # clue_sat: for each parsed constraint c, check if solution entails it
        # entailment test: (solution + NOT(c)) is UNSAT
        if not clue_constraints:
            clue_sat = 0.0
        else:
            sat_count = 0
            for c in clue_constraints:
                S_sol.push()
                S_sol.add(Not(c))
                if S_sol.check() == unsat:
                    sat_count += 1
                S_sol.pop()
            clue_sat = sat_count / len(clue_constraints)

        # --- reasoning score (stepwise) ---
        # KB starts with base + clue constraints (without fixing solution)
        S_kb = Solver()
        S_kb.add(base.assertions())
        for c in clue_constraints:
            S_kb.add(c)

        valid = 0
        guessed = 0
        contradicted = 0

        for sent in (reasoning or []):
            cs = constraints_from_sentence(sent, Z, v2h, n, vocab, debug=debug)
            if not cs:
                guessed += 1
                continue

            for c in cs:
                # entailed by current KB?
                S_kb.push()
                S_kb.add(Not(c))
                entailed = (S_kb.check() == unsat)
                S_kb.pop()

                if entailed:
                    valid += 1
                    S_kb.add(c)  # monotonic add
                    continue

                # contradict fixed predicted solution?
                S_sol.push()
                S_sol.add(c)
                is_contradiction = (S_sol.check() == unsat)
                S_sol.pop()

                if is_contradiction:
                    contradicted += 1
                else:
                    guessed += 1

        total = valid + guessed + contradicted
        if total == 0:
            reason_score = 0.0
        else:
            reason_score = (valid - guessed - 2 * contradicted) / total
            reason_score = max(-1.0, min(1.0, float(reason_score)))

        return {
            "z3_sat": float(z3_sat),
            "parse_cov": float(parse_cov),
            "clue_sat": float(clue_sat),
            "reason_score": float(reason_score),
        }

    except Exception:
        return {
            "z3_sat": 0.0,
            "parse_cov": 0.0,
            "clue_sat": 0.0,
            "reason_score": 0.0,
        }


# =========================
# 6) Quick demo
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
        "Therefore Eric is the British person.",
    ]

    # Intentionally WRONG solution (Eric is not brit), so z3_sat should be 0
    solution = {
        "header": ["House", "Name", "Color", "Nationality"],
        "rows": [
            ["1", "Eric", "yellow", "dane"],
            ["2", "Arnold", "red", "brit"],
        ],
    }

    out = verify_reasoning_solution_clues_with_z3(reasoning, solution, clues, debug=False)
    print("\nRESULT:", json.dumps(out, indent=2))
