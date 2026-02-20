import json
import re
import numpy as np
from z3 import *


# ============================================================
# 1. Safe Parsing (GRPO-ROBUST)
# ============================================================

def safe_parse_answer(text: str):
    try:
        text = text.strip()
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not m:
            return None
        obj = json.loads(m.group(1))
        if "reasoning" not in obj or "solution" not in obj:
            return None
        return obj["reasoning"], obj["solution"]
    except Exception:
        return None


# ============================================================
# 2. Build Z3 Model from Solution Table
# ============================================================

def build_model(solution):
    header = solution["header"]
    rows = solution["rows"]
    n = len(rows)

    attrs = header[1:]
    values = {a: set() for a in attrs}
    for r in rows:
        for a, v in zip(attrs, r[1:]):
            values[a].add(v.lower())

    solver = Solver()
    Z = {}

    for attr, vals in values.items():
        Z[attr] = {v: Int(f"{attr}_{v}") for v in vals}
        solver.add(Distinct(*Z[attr].values()))
        for v in Z[attr].values():
            solver.add(v >= 1, v <= n)

    return solver, Z


def build_value_to_header(solution):
    header = solution["header"][1:]
    rows = solution["rows"]
    v2h = {}
    for i, h in enumerate(header):
        for r in rows:
            v2h[r[i + 1].lower()] = h
    return v2h


def add_solution_constraints(solver, Z, solution):
    header = solution["header"]
    for r in solution["rows"]:
        h = int(r[0])
        for attr, val in zip(header[1:], r[1:]):
            solver.add(Z[attr][val.lower()] == h)


# ============================================================
# 3. Normalization
# ============================================================

def normalize(text: str):
    s = text.lower()
    s = re.sub(r"[.,]", "", s)
    s = re.sub(r"\b(the|a|an)\b", "", s)

    # role normalization
    s = re.sub(r"man who smokes (\w+)", r"\1", s)
    s = re.sub(r"man who keeps (\w+)", r"\1", s)
    s = re.sub(r"man who drinks (\w+)", r"\1", s)
    s = re.sub(r"(\w+) drinker", r"\1", s)

    # relations
    s = s.replace("is immediately to the right of", "right of")
    s = s.replace("is directly left of", "left of")
    s = s.replace("is immediately left of", "left of")
    s = s.replace("lives next to", "next to")
    s = s.replace("is next to", "next to")

    # positions
    s = s.replace("first house", "house 1")
    s = s.replace("middle house", "house middle")
    s = s.replace("last house", "house last")

    s = s.replace("lives in", "in")
    s = s.replace("is in", "in")

    return re.sub(r"\s+", " ", s).strip()


# ============================================================
# 4. Canonical Patterns (ORDER MATTERS)
# ============================================================

CLUE_PATTERNS = [
    ("eq_house", re.compile(r"(\w+) in house (\d+|middle|last)")),
    ("left",     re.compile(r"(\w+) left of (\w+)")),
    ("right",    re.compile(r"(\w+) right of (\w+)")),
    ("adj",      re.compile(r"(\w+) next to (\w+)")),
    ("two_away", re.compile(r"(\w+) two houses away from (\w+)")),
    ("between",  re.compile(r"(\w+) between (\w+) and (\w+)")),
    ("eq",       re.compile(r"(\w+) in (\w+)")),  # LAST
]


def extract_facts(text):
    s = normalize(text)
    for kind, pat in CLUE_PATTERNS:
        m = pat.search(s)
        if m:
            return [(kind, *m.groups())]
    return []


# ============================================================
# 5. Fact → Z3 Constraint (STRICT & SAFE)
# ============================================================

def fact_to_constraint(fact, Z, v2h, n):
    kind = fact[0]

    def H(x):
        h = v2h.get(x)
        if h is None:
            return None
        # ensure token exists in this column domain
        col = Z.get(h)
        if col is None or x not in col:
            return None
        return col[x]

    if kind == "eq":
        Ha, Hb = H(fact[1]), H(fact[2])
        if Ha is None or Hb is None:
            return None
        return Ha == Hb

    if kind == "left":
        Ha, Hb = H(fact[1]), H(fact[2])
        if Ha is None or Hb is None:
            return None
        return Ha + 1 == Hb

    if kind == "right":
        Ha, Hb = H(fact[1]), H(fact[2])
        if Ha is None or Hb is None:
            return None
        return Ha == Hb + 1

    if kind == "adj":
        Ha, Hb = H(fact[1]), H(fact[2])
        if Ha is None or Hb is None:
            return None
        return Abs(Ha - Hb) == 1

    if kind == "two_away":
        Ha, Hb = H(fact[1]), H(fact[2])
        if Ha is None or Hb is None:
            return None
        return Abs(Ha - Hb) == 2

    if kind == "between":
        Ha, Hb, Hc = H(fact[1]), H(fact[2]), H(fact[3])
        if Ha is None or Hb is None or Hc is None:
            return None
        return Or(And(Hb < Ha, Ha < Hc), And(Hc < Ha, Ha < Hb))

    if kind == "eq_house":
        Ha = H(fact[1])
        b = fact[2]
        if Ha is None:
            return None
        if b == "middle":
            return Ha == (n + 1) // 2
        if b == "last":
            return Ha == n
        return Ha == int(b)

    return None


# ============================================================
# 6. Add Clues
# ============================================================

def add_clue_constraints(solver, Z, clues, n, v2h):
    added = 0
    for c in clues:
        facts = extract_facts(c)
        if not facts:
            # uncomment for debugging:
            # print("[CLUE NOT PARSED]", c)
            continue
        for f in facts:
            z = fact_to_constraint(f, Z, v2h, n)
            if z is not None:
                solver.add(z)
                added += 1
    # uncomment for debugging:
    # print(f"[INFO] Added {added}/{len(clues)} clue constraints")
    return added


# ============================================================
# 7. Reasoning Scoring
# ============================================================

def reasoning_score(reasoning, base_solver, Z, sol_solver, n, v2h):
    valid = guessed = contradicted = 0
    solver = Solver()
    solver.add(base_solver.assertions())

    for s in reasoning:
        facts = extract_facts(s)
        if not facts:
            guessed += 1
            continue

        for f in facts:
            z = fact_to_constraint(f, Z, v2h, n)
            if z is None:
                guessed += 1
                continue

            # entailed by clues + previous accepted steps?
            solver.push()
            solver.add(Not(z))
            if solver.check() == unsat:
                valid += 1
                solver.pop()
                solver.add(z)
                continue
            solver.pop()

            # contradicts final solution?
            sol_solver.push()
            sol_solver.add(z)
            if sol_solver.check() == unsat:
                contradicted += 1
                sol_solver.pop()
                continue
            sol_solver.pop()

            guessed += 1

    tot = valid + guessed + contradicted
    return 0.0 if tot == 0 else (valid - guessed - 2 * contradicted) / tot


# ============================================================
# 8. Single-Sample Reward
# ============================================================

def zebra_reward(sample, clues):
    parsed = safe_parse_answer(sample)
    if parsed is None:
        return -1.0  # format penalty

    reasoning, solution = parsed
    n = len(solution["rows"])

    base, Z = build_model(solution)
    v2h = build_value_to_header(solution)
    add_clue_constraints(base, Z, clues, n, v2h)

    sol, _ = build_model(solution)
    add_solution_constraints(sol, Z, solution)

    sol_r = 1.0 if sol.check() == sat else 0.0
    r_r = reasoning_score(reasoning, base, Z, sol, n, v2h)

    return 0.7 * sol_r + 0.3 * max(0.0, r_r)


# ============================================================
# 9. GRPO Reward (GROUP RELATIVE)
# ============================================================

def zebra_grpo_rewards(samples, clues):
    raw = np.array([zebra_reward(s, clues) for s in samples], dtype=np.float32)

    # if all samples are format-invalid, return zeros to avoid NaNs
    if np.all(raw <= -0.99):
        return np.zeros_like(raw)

    mean = raw.mean()
    std = raw.std() + 1e-8
    return (raw - mean) / std


# ============================================================
# 10. Demo
# ============================================================

if __name__ == "__main__":
    clues = [
        "Peter is in House 2.",
        "Arnold is directly left of the water drinker.",
        "The water drinker is directly left of the milk drinker."
    ]

    samples = [
        # ✅ Perfect
        """
<answer>{
  "reasoning": [
    "Peter is in House 2.",
    "Arnold is directly left of the water drinker.",
    "The water drinker is directly left of the milk drinker."
  ],
  "solution": {
    "header": ["House", "Name", "Drink"],
    "rows": [
      ["1", "Arnold", "tea"],
      ["2", "Peter", "water"],
      ["3", "Eric", "milk"]
    ]
  }
}</answer>
""",
        # ⚠️ Correct but weak reasoning
        """
<answer>{
  "reasoning": [
    "Peter is in House 2."
  ],
  "solution": {
    "header": ["House", "Name", "Drink"],
    "rows": [
      ["1", "Arnold", "tea"],
      ["2", "Peter", "water"],
      ["3", "Eric", "milk"]
    ]
  }
}</answer>
""",
        # ❌ Wrong solution
        """
<answer>{
  "reasoning": [
    "Peter is in House 1."
  ],
  "solution": {
    "header": ["House", "Name", "Drink"],
    "rows": [
      ["1", "Peter", "water"],
      ["2", "Arnold", "tea"],
      ["3", "Eric", "milk"]
    ]
  }
}</answer>
""",
        # ❌ Invalid format
        "Peter is in House 2."
    ]

    rewards = zebra_grpo_rewards(samples, clues)
    for i, r in enumerate(rewards):
        print(f"Sample {i+1} GRPO reward: {r:.3f}")
