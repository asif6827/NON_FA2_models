"""
ZebraPuzzle Z3 Verifier
----------------------

• Verifies final solution using Z3
• Verifies natural-language reasoning step-by-step
• Penalizes guessing and contradictions
• Produces RL-ready rewards

Author: (you)
"""

import json
import re
from z3 import *


# ============================================================
# 1. Parse LLM Output
# ============================================================

def parse_answer(text: str):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        raise ValueError("Missing <answer>...</answer> block")

    obj = json.loads(m.group(1))

    if "reasoning" not in obj or "solution" not in obj:
        raise ValueError("Invalid answer format")

    return obj["reasoning"], obj["solution"]


# ============================================================
# 2. Build Generic Z3 Model from Solution Table
# ============================================================

def build_model(solution):
    header = solution["header"]
    rows = solution["rows"]

    houses = [int(r[0]) for r in rows]
    attrs = header[1:]

    values = {a: set() for a in attrs}
    for r in rows:
        for a, v in zip(attrs, r[1:]):
            values[a].add(v)

    solver = Solver()
    Z = {}

    for attr, vals in values.items():
        Z[attr] = {v: Int(f"{attr}_{v}") for v in vals}
        solver.add(Distinct(*Z[attr].values()))
        for v in Z[attr].values():
            solver.add(v >= min(houses), v <= max(houses))

    return solver, Z


# ============================================================
# 3. Inject Final Solution as Hard Constraints
# ============================================================

def add_solution_constraints(solver, Z, solution):
    header = solution["header"]
    for r in solution["rows"]:
        house = int(r[0])
        for attr, val in zip(header[1:], r[1:]):
            solver.add(Z[attr][val] == house)


# ============================================================
# 4. Extract Logical Facts from Natural Language Reasoning
# ============================================================

def extract_facts(sentence: str):
    s = sentence.lower()
    facts = []

    # X is in House Y
    m = re.search(r"(\w+) is in house (\d+)", s)
    if m:
        facts.append(("eq_house", m.group(1), int(m.group(2))))

    # X is directly left of Y
    m = re.search(r"(\w+) is directly left of (\w+)", s)
    if m:
        facts.append(("left", m.group(1), m.group(2)))

    # X is next to Y
    m = re.search(r"(\w+) is next to (\w+)", s)
    if m:
        facts.append(("adj", m.group(1), m.group(2)))

    # X drinks Y / owns Y / has Y
    m = re.search(r"(\w+) (drinks|owns|has) (\w+)", s)
    if m:
        facts.append(("eq", m.group(1), m.group(3)))

    return facts


# ============================================================
# 5. Convert Facts to Z3 Constraints
# ============================================================

def fact_to_constraint(fact, Z):
    kind = fact[0]

    if kind == "eq_house":
        ent, house = fact[1], fact[2]
        for group in Z.values():
            if ent in group:
                return group[ent] == house

    if kind == "eq":
        a, b = fact[1], fact[2]
        for group in Z.values():
            if a in group and b in group:
                return group[a] == group[b]

    if kind == "left":
        a, b = fact[1], fact[2]
        for group in Z.values():
            if a in group and b in group:
                return group[a] + 1 == group[b]

    if kind == "adj":
        a, b = fact[1], fact[2]
        for group in Z.values():
            if a in group and b in group:
                return Abs(group[a] - group[b]) == 1

    return None


# ============================================================
# 6. Step-wise Z3 Reasoning Verification
# ============================================================

def score_reasoning_steps(reasoning, solver, Z):
    valid = 0
    guessed = 0
    contradicted = 0

    for sentence in reasoning:
        facts = extract_facts(sentence)

        for f in facts:
            c = fact_to_constraint(f, Z)

            if c is None:
                guessed += 1
                continue

            # Check entailment
            solver.push()
            solver.add(Not(c))
            if solver.check() == unsat:
                valid += 1
                solver.pop()
                solver.add(c)   # monotonic reasoning
            else:
                solver.pop()
                solver.push()
                solver.add(c)
                if solver.check() == unsat:
                    contradicted += 1
                else:
                    guessed += 1
                solver.pop()

    return {
        "valid": valid,
        "guessed": guessed,
        "contradicted": contradicted
    }


# ============================================================
# 7. Solution Verification
# ============================================================

def solution_reward(solver):
    return 1.0 if solver.check() == sat else 0.0


# ============================================================
# 8. Final RL-Ready Reward Function
# ============================================================

def zebra_reward(llm_output: str):
    reasoning, solution = parse_answer(llm_output)

    solver, Z = build_model(solution)
    add_solution_constraints(solver, Z, solution)

    sol_r = solution_reward(solver)

    step_stats = score_reasoning_steps(reasoning, solver, Z)
    total_steps = sum(step_stats.values())

    if total_steps == 0:
        reasoning_r = 0.0
    else:
        reasoning_r = (
            step_stats["valid"]
            - step_stats["guessed"]
            - 2 * step_stats["contradicted"]
        ) / total_steps

    total_r = 0.7 * sol_r + 0.3 * max(0.0, reasoning_r)

    return {
        "solution_reward": sol_r,
        "reasoning_reward": reasoning_r,
        "total_reward": max(0.0, total_r),
        "step_stats": step_stats
    }


# ============================================================
# 9. Minimal Example (for testing)
# ============================================================

if __name__ == "__main__":
    example = """
    <answer>{
      "reasoning": [
        "Peter is in House 2.",
        "Arnold is directly left of the water drinker.",
        "The water drinker is in House 2."
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
    """
    print(zebra_reward(example))
