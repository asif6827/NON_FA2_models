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
    return obj["reasoning"], obj["solution"]


# ============================================================
# 2. Build Z3 Model from Solution Table
# ============================================================

def build_model(solution):
    header = solution["header"]
    rows = solution["rows"]

    houses = [int(r[0]) for r in rows]
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
            solver.add(v >= min(houses), v <= max(houses))

    return solver, Z


# ============================================================
# 3. Map value → solution header (CRITICAL FIX)
# ============================================================

def build_value_to_header(solution):
    header = solution["header"][1:]
    rows = solution["rows"]

    value_to_header = {}
    for col_idx, h in enumerate(header):
        for r in rows:
            value_to_header[r[col_idx + 1].lower()] = h
    return value_to_header


# ============================================================
# 4. Inject Final Solution Constraints
# ============================================================

def add_solution_constraints(solver, Z, solution):
    header = solution["header"]
    for r in solution["rows"]:
        house = int(r[0])
        for attr, val in zip(header[1:], r[1:]):
            solver.add(Z[attr][val.lower()] == house)


# ============================================================
# 5. Normalize Clues / Reasoning Sentences
# ============================================================

def normalize_text(text: str):
    s = text.lower()
    s = re.sub(r"[.,]", "", s)
    s = re.sub(r"\b(the|a|an)\b", "", s)

    s = re.sub(r"(\w+) drinker", r"\1", s)

    s = s.replace("is directly left of", "left of")
    s = s.replace("is immediately left of", "left of")
    s = s.replace("is immediately to the right of", "right of")
    s = s.replace("is right of", "right of")
    s = s.replace("is next to", "next to")
    s = s.replace("lives next to", "next to")

    s = s.replace("lives in", "in")
    s = s.replace("is in", "in")

    s = s.replace("first house", "house 1")
    s = s.replace("middle house", "house middle")
    s = s.replace("last house", "house last")

    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# 6. Canonical Clue / Reasoning Patterns
# ============================================================

CLUE_PATTERNS = [
    ("eq_house", re.compile(r"(\w+) in house (\d+|middle|last)")),
    ("left",     re.compile(r"(\w+) left of (\w+)")),
    ("right",    re.compile(r"(\w+) right of (\w+)")),
    ("adj",      re.compile(r"(\w+) next to (\w+)")),
    ("eq",       re.compile(r"(\w+) in (\w+)")),
]


def extract_facts(text: str):
    s = normalize_text(text)
    facts = []

    for kind, pattern in CLUE_PATTERNS:
        m = pattern.search(s)
        if not m:
            continue

        a, b = m.group(1), m.group(2)
        facts.append((kind, a, b))
        return facts

    return facts


# ============================================================
# 7. Fact → Z3 Constraint
# ============================================================

def fact_to_constraint(fact, Z, value_to_header, house_count):
    kind, a, b = fact

    ha = value_to_header.get(a)
    hb = value_to_header.get(b)

    if kind == "eq" and ha and hb:
        return Z[ha][a] == Z[hb][b]

    if kind == "left" and ha and hb:
        return Z[ha][a] + 1 == Z[hb][b]

    if kind == "right" and ha and hb:
        return Z[ha][a] == Z[hb][b] + 1

    if kind == "adj" and ha and hb:
        return Abs(Z[ha][a] - Z[hb][b]) == 1

    if kind == "eq_house" and ha:
        if b == "middle":
            return Z[ha][a] == (house_count + 1) // 2
        if b == "last":
            return Z[ha][a] == house_count
        return Z[ha][a] == int(b)

    return None


# ============================================================
# 8. Add Clues as Base Constraints
# ============================================================

def add_clue_constraints(solver, Z, clues, house_count, value_to_header):
    added = 0
    for clue in clues:
        facts = extract_facts(clue)
        if not facts:
            pass
            #print(f"[CLUE NOT PARSED] {clue}")
        for f in facts:
            c = fact_to_constraint(f, Z, value_to_header, house_count)
            if c is not None:
                solver.add(c)
                added += 1
    #print(f"[INFO] Added {added}/{len(clues)} clue constraints")
    return added


# ============================================================
# 9. Step-wise Reasoning Verification
# ============================================================

def score_reasoning_steps(reasoning, base_solver, Z, solution_solver, house_count, value_to_header):
    valid = guessed = contradicted = 0

    solver = Solver()
    solver.add(base_solver.assertions())

    for sentence in reasoning:
        facts = extract_facts(sentence)

        for f in facts:
            c = fact_to_constraint(f, Z, value_to_header, house_count)
            if c is None:
                guessed += 1
                continue

            # Entailed by clues + previous steps?
            solver.push()
            solver.add(Not(c))
            if solver.check() == unsat:
                solver.pop()
                valid += 1
                solver.add(c)
                continue
            solver.pop()

            # Contradicts final solution?
            solution_solver.push()
            solution_solver.add(c)
            if solution_solver.check() == unsat:
                solution_solver.pop()
                contradicted += 1
                continue
            solution_solver.pop()

            guessed += 1

    return dict(valid=valid, guessed=guessed, contradicted=contradicted)


# ============================================================
# 10. Final Reward Function
# ============================================================

def zebra_reward(reasoning, solution, puzzle_clues):
    
    #reasoning, solution = parse_answer(llm_output)
    house_count = len(solution["rows"])

    base_solver, Z = build_model(solution)
    value_to_header = build_value_to_header(solution)

    add_clue_constraints(base_solver, Z, puzzle_clues, house_count, value_to_header)

    solution_solver, _ = build_model(solution)
    add_solution_constraints(solution_solver, Z, solution)

    sol_reward = 1.0 if solution_solver.check() == sat else 0.0

    step_stats = score_reasoning_steps(
        reasoning,
        base_solver,
        Z,
        solution_solver,
        house_count,
        value_to_header
    )

    steps = sum(step_stats.values())
    reasoning_reward = 0.0 if steps == 0 else (
        step_stats["valid"]
        - step_stats["guessed"]
        - 2 * step_stats["contradicted"]
    ) / steps

    total_reward = 0.7 * sol_reward + 0.3 * max(0.0, reasoning_reward)

    return {
        "solution_reward": sol_reward,
        "reasoning_reward": reasoning_reward,
        "total_reward": max(0.0, total_reward),
        "step_stats": step_stats
    }


# ============================================================
# 11. Example Demonstration
# ============================================================

if __name__ == "__main__":
    clues = [
        "Peter is in House 2.",
        "Arnold is directly left of the water drinker.",
        "The water drinker is directly left of the milk drinker."
    ]

    llm_output = """
    <answer>{
      "reasoning": [
        "Peter is in House 2.",
        "Arnold is left of water.",
        "Water is left of milk."
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
    reasoning, solution = parse_answer(llm_output)
    
    print(zebra_reward(reasoning, solution, clues))
