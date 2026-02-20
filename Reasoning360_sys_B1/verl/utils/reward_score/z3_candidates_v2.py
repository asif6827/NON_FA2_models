# -*- coding: utf-8 -*-
# pip install z3-solver

import itertools
import z3
from typing import List, Tuple, Dict, Set, Any

# -----------------------------
# Helpers
# -----------------------------
def canon(s: str) -> str:
    return " ".join(s.strip().lower().split())

def entailment_label(constraints: List[z3.BoolRef], atom_expr: z3.BoolRef) -> str:
    s = z3.Solver()
    s.add(constraints)
    s.add(z3.Not(atom_expr))
    return "ENTAILED" if s.check() == z3.unsat else "NOT_ENTAILED"

def is_sat(constraints: List[z3.BoolRef]) -> bool:
    s = z3.Solver()
    s.add(constraints)
    return s.check() == z3.sat

def sat_with(constraints: List[z3.BoolRef], extra: z3.BoolRef) -> bool:
    s = z3.Solver()
    s.add(constraints)
    s.add(extra)
    return s.check() == z3.sat

# -----------------------------
# Puzzle
# -----------------------------
def build_puzzle():
    n_houses = 3

    Alice, Bob, Carol = z3.Ints("Alice Bob Carol")
    Cat, Dog, Fish = z3.Ints("Cat Dog Fish")

    people = {"alice": Alice, "bob": Bob, "carol": Carol}
    pets = {"cat": Cat, "dog": Dog, "fish": Fish}
    z3_vars = {**people, **pets}

    domain = [z3.And(v >= 1, v <= n_houses) for v in z3_vars.values()]
    base = domain + [
        z3.Distinct(Alice, Bob, Carol),
        z3.Distinct(Cat, Dog, Fish),
    ]

    clues = [
        Alice == 1,
        Bob != 2,
        z3.Abs(Bob - Dog) == 1,
        Cat != 3,
    ]

    constraints = base + clues

    clue_atom_texts = {
        canon("alice == 1"),
        canon("bob != 2"),
        canon("cat != 3"),
    }

    return constraints, z3_vars, clue_atom_texts, n_houses

# -----------------------------
# 1) Base "simple atoms"
# -----------------------------
def generate_simple_atoms(z3_vars: Dict[str, Any], n_houses: int) -> List[Tuple[str, z3.BoolRef]]:
    atoms: List[Tuple[str, z3.BoolRef]] = []

    # placements / exclusions
    for name, v in z3_vars.items():
        for i in range(1, n_houses + 1):
            atoms.append((f"{name} == {i}", v == i))
            atoms.append((f"Not({name} == {i})", z3.Not(v == i)))  # explicit Not form

    # pairings across (people x pets)
    people = ["alice", "bob", "carol"]
    pets = ["cat", "dog", "fish"]
    for p in people:
        for pet in pets:
            atoms.append((f"{p} == {pet}", z3_vars[p] == z3_vars[pet]))
            atoms.append((f"Not({p} == {pet})", z3.Not(z3_vars[p] == z3_vars[pet])))

    return atoms

# -----------------------------
# 2) More involved atoms (Or / And / Implies) in a CONTROLLED way
# -----------------------------
def possible_houses(constraints: List[z3.BoolRef], v: z3.IntNumRef, n_houses: int) -> List[int]:
    """Return houses i where constraints ∧ (v==i) is SAT."""
    poss = []
    for i in range(1, n_houses + 1):
        if sat_with(constraints, v == i):
            poss.append(i)
    return poss

def generate_compound_atoms_from_domains(
    constraints: List[z3.BoolRef],
    z3_vars: Dict[str, Any],
    n_houses: int,
    max_or_size: int = 3,
) -> List[Tuple[str, z3.BoolRef]]:
    """
    Generate compound candidates derived from current domains.
    This avoids combinatorial explosion.

    Adds:
      - Or(v==i, v==j, ...) for the current possible house set (size 2..max_or_size)
      - And(Not(v==k), Not(v==m), ...) for excluded houses (only if that exclusion set is small)
      - Implies(v==i, Or(v==j, v==k)) as a "case rule" (rare; generated only when domain size is 2)
    """
    atoms: List[Tuple[str, z3.BoolRef]] = []

    for name, v in z3_vars.items():
        dom = possible_houses(constraints, v, n_houses)
        dom_set = set(dom)
        if 2 <= len(dom) <= max_or_size:
            # Or(v==d1, v==d2, ...)
            ors = [v == d for d in dom]
            text = "Or(" + ", ".join([f"{name} == {d}" for d in dom]) + ")"
            atoms.append((text, z3.Or(*ors)))

        # Small excluded-set conjunction (represents pruning)
        excluded = [i for i in range(1, n_houses + 1) if i not in dom_set]
        if 1 <= len(excluded) <= 2:  # keep it small
            ands = [z3.Not(v == e) for e in excluded]
            text = "And(" + ", ".join([f"Not({name} == {e})" for e in excluded]) + ")"
            atoms.append((text, z3.And(*ands)))

        # Optional: tiny implication rules from a 2-element domain
        # Example: if dom is {1,3}, then (v==1) -> Not(v==2) is trivial, not useful.
        # A more meaningful case-rule is: if v==d1 then Not(v==d2) (still trivial).
        # We'll show a simple form anyway, but keep it very restricted.
        if len(dom) == 2:
            d1, d2 = dom
            # Implies(v==d1, Not(v==d2)) is always true, but often entailed anyway.
            # This is mainly for experimenting with Implies in your framework.
            text = f"Implies({name} == {d1}, Not({name} == {d2}))"
            atoms.append((text, z3.Implies(v == d1, z3.Not(v == d2))))

    return atoms

# -----------------------------
# Candidate menu (ENTAILED + NOVEL) with labels
# -----------------------------
def z3_candidate_menu(
    constraints: List[z3.BoolRef],
    atoms: List[Tuple[str, z3.BoolRef]],
    used_atoms: Set[str],
    clue_atoms: Set[str],
    include_not_entailed: bool = False,
    filter_clue_restatements: bool = False,
    max_menu: int = 80,
) -> List[Dict[str, Any]]:
    if not is_sat(constraints):
        return []

    menu: List[Dict[str, Any]] = []
    for text, expr in atoms:
        c = canon(text)
        if c in used_atoms:
            continue
        if filter_clue_restatements and (c in clue_atoms):
            continue

        ent = entailment_label(constraints, expr)
        if ent == "NOT_ENTAILED" and not include_not_entailed:
            continue

        origin = "N/A"
        if ent == "ENTAILED":
            origin = "CLUE" if (c in clue_atoms) else "DERIVED"

        menu.append({"text": text, "expr": expr, "entailment": ent, "origin": origin})
        if len(menu) >= max_menu:
            break

    return menu

def print_menu(menu: List[Dict[str, Any]], title: str):
    print(f"\n=== {title} ===")
    if not menu:
        print("(empty)")
        return
    for i, m in enumerate(menu):
        print(f"{i:02d}) [{m['entailment']:^12}] [{m['origin']:^7}] {m['text']}")

# -----------------------------
# Demo
# -----------------------------
def main():
    constraints, z3_vars, clue_atoms, n_houses = build_puzzle()
    used_atoms: Set[str] = set()

    # Build atoms:
    simple_atoms = generate_simple_atoms(z3_vars, n_houses)

    # IMPORTANT: compound atoms depend on current constraints (domains).
    compound_atoms = generate_compound_atoms_from_domains(
        constraints=constraints,
        z3_vars=z3_vars,
        n_houses=n_houses,
        max_or_size=3,
    )

    all_atoms = simple_atoms + compound_atoms

    # Menu: entailed-only (recommended)
    menu = z3_candidate_menu(
        constraints=constraints,
        atoms=all_atoms,
        used_atoms=used_atoms,
        clue_atoms=clue_atoms,
        include_not_entailed=False,
        filter_clue_restatements=False,
        max_menu=80,
    )
    print_menu(menu, "Candidate Menu (ENTAILED + NOVEL), includes Or/And/Not/Implies candidates")

    # Debug: show some NOT_ENTAILED too
    menu_dbg = z3_candidate_menu(
        constraints=constraints,
        atoms=all_atoms,
        used_atoms=used_atoms,
        clue_atoms=clue_atoms,
        include_not_entailed=True,
        filter_clue_restatements=False,
        max_menu=30,
    )
    print_menu(menu_dbg, "DEBUG Menu (includes NOT_ENTAILED)")

if __name__ == "__main__":
    main()
