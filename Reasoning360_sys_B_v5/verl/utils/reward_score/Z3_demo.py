# pip install z3-solver
from z3 import Int, Solver, Distinct, And, Or, sat

def solve_zebra_2houses():
    houses = [1, 2]

    # Domains
    names = ["Arnold", "Eric"]
    months = ["sept", "april"]
    mothers = ["Aniya", "Holly"]

    # Position variables: value = house number (1..2)
    pos_name = {n: Int(f"pos_name_{n}") for n in names}
    pos_month = {m: Int(f"pos_month_{m}") for m in months}
    pos_mother = {mo: Int(f"pos_mother_{mo}") for mo in mothers}

    s = Solver()

    # Each item is in some house
    for d in (pos_name, pos_month, pos_mother):
        for k, v in d.items():
            s.add(And(v >= 1, v <= 2))

    # Uniqueness within each category (AllDifferent)
    s.add(Distinct(*pos_name.values()))
    s.add(Distinct(*pos_month.values()))
    s.add(Distinct(*pos_mother.values()))

    # --- Clues ---
    # 1. The person whose mother's name is Holly is Eric.
    # Means: Eric lives in the same house as mother Holly
    s.add(pos_name["Eric"] == pos_mother["Holly"])

    # 2. Arnold is the person whose birthday is in April.
    s.add(pos_name["Arnold"] == pos_month["april"])

    # 3. Eric is not in the second house.
    s.add(pos_name["Eric"] != 2)

    # Solve
    if s.check() != sat:
        print("No solution.")
        return

    m = s.model()

    # Build per-house table
    def house_of(expr):  # model value as int
        return m[expr].as_long()

    by_house = {h: {"House": h} for h in houses}

    for n in names:
        by_house[house_of(pos_name[n])]["Name"] = n
    for mo in mothers:
        by_house[house_of(pos_mother[mo])]["Mother"] = mo
    for mon in months:
        by_house[house_of(pos_month[mon])]["Birthday"] = mon

    # Pretty print
    print("Solution:")
    for h in houses:
        row = by_house[h]
        print(f"House {h}: Name={row['Name']}, Birthday={row['Birthday']}, Mother={row['Mother']}")

    # Optional: check uniqueness
    # Block the found model (require at least one variable differs)
    block = Or(
        *[pos_name[n] != house_of(pos_name[n]) for n in names],
        *[pos_month[mon] != house_of(pos_month[mon]) for mon in months],
        *[pos_mother[mo] != house_of(pos_mother[mo]) for mo in mothers],
    )
    s2 = Solver()
    s2.add(s.assertions())
    s2.add(block)
    unique = (s2.check() != sat)
    print("Unique solution:", unique)

if __name__ == "__main__":
    solve_zebra_2houses()
