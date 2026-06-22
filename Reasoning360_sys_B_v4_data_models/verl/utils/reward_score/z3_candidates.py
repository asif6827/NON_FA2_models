# -*- coding: utf-8 -*-
# pip install z3-solver

import z3
from typing import List, Tuple, Dict, Set, Any

# -----------------------------
# Helpers
# -----------------------------
def canon(s: str) -> str:
    return " ".join(s.strip().lower().split())

def entailment_label(constraints: List[z3.BoolRef], atom_expr: z3.BoolRef) -> str:
    """
    Return "ENTAILED" iff constraints ⊨ atom_expr, else "NOT_ENTAILED".
    Entailment test: constraints ∧ ¬atom_expr is UNSAT.
    """
    s = z3.Solver()
    s.add(constraints)
    s.add(z3.Not(atom_expr))
    return "ENTAILED" if s.check() == z3.unsat else "NOT_ENTAILED"

def is_sat(constraints: List[z3.BoolRef]) -> bool:
    s = z3.Solver()
    s.add(constraints)
    return s.check() == z3.sat

# -----------------------------
# Toy puzzle (more involved than the earlier minimal one)
# -----------------------------
def build_puzzle():
    """
    3-house toy Zebra-style puzzle.

    People: Alice, Bob, Carol
    Pets:   Cat, Dog, Fish

    Syntactic clues:
      C1: Alice == 1
      C2: Bob != 2
      C3: Abs(Bob - Dog) == 1   (Bob next to Dog)
      C4: Cat != 3
    Plus standard all-different and domain constraints.

    This set of clues forces a unique-ish structure and yields derived facts like:
      Bob == 3, Dog == 2, Cat == 1, Fish == 3, Alice == Cat, Bob == Fish, Carol == Dog
    """
    n_houses = 3

    # Int-per-value encoding: each value's variable = its house index
    Alice, Bob, Carol = z3.Ints("Alice Bob Carol")
    Cat, Dog, Fish = z3.Ints("Cat Dog Fish")

    people = {"alice": Alice, "bob": Bob, "carol": Carol}
    pets = {"cat": Cat, "dog": Dog, "fish": Fish}
    z3_vars = {**people, **pets}

    # Domain constraints
    domain = [z3.And(v >= 1, v <= n_houses) for v in z3_vars.values()]

    # AllDifferent within each attribute group
    base = domain + [
        z3.Distinct(Alice, Bob, Carol),
        z3.Distinct(Cat, Dog, Fish),
    ]

    # --- Syntactic clues (given) ---
    clues = [
        Alice == 1,                 # C1
        Bob != 2,                   # C2
        z3.Abs(Bob - Dog) == 1,     # C3 (not an "atom" in our simple atom universe)
        Cat != 3,                   # C4
    ]

    constraints = base + clues

    # Only include clue-atoms that are in our atom universe (placements/exclusions/pairings).
    # (Abs adjacency isn't represented as a single atom in our menu)
    clue_atom_texts = {
        canon("alice == 1"),
        canon("bob != 2"),
        canon("cat != 3"),
    }

    return constraints, z3_vars, clue_atom_texts, n_houses

# -----------------------------
# Atom Universe Generator
# -----------------------------
def generate_atom_universe(z3_vars: Dict[str, Any], n_houses: int) -> List[Tuple[str, z3.BoolRef]]:
    """
    Enumerate all simple "atoms" we might ask Z3 about.

    1) Placement atoms:    v == i
    2) Exclusion atoms:    v != i
    3) Pairing atoms:      person == pet (same house)
    4) Non-pairing atoms:  person != pet

    NOTE: This universe is intentionally simple and finite. Z3 will label each
          candidate as ENTAILED or NOT_ENTAILED given the current constraints.
    """
    atoms: List[Tuple[str, z3.BoolRef]] = []

    # 1) v == i
    for name, v in z3_vars.items():
        for i in range(1, n_houses + 1):
            atoms.append((f"{name} == {i}", v == i))

    # 2) v != i
    for name, v in z3_vars.items():
        for i in range(1, n_houses + 1):
            atoms.append((f"{name} != {i}", v != i))

    # 3) Pairings / Non-pairings across groups (people x pets)
    people = ["alice", "bob", "carol"]
    pets = ["cat", "dog", "fish"]
    for p in people:
        for pet in pets:
            atoms.append((f"{p} == {pet}", z3_vars[p] == z3_vars[pet]))
            atoms.append((f"{p} != {pet}", z3_vars[p] != z3_vars[pet]))

    return atoms

# -----------------------------
# Candidate Menu (ENTAILED + NOVEL) with labels
# -----------------------------
def z3_candidate_menu(
    constraints: List[z3.BoolRef],
    atoms: List[Tuple[str, z3.BoolRef]],
    used_atoms: Set[str],
    clue_atoms: Set[str],
    include_not_entailed: bool = False,
    filter_clue_restatements: bool = False,
    max_menu: int = 60,
) -> List[Dict[str, Any]]:
    """
    Build a menu of candidates.
      - NOVEL: excludes anything in used_atoms
      - Adds entailment label: ENTAILED / NOT_ENTAILED (computed by Z3)
      - Adds origin label: CLUE / DERIVED (only meaningful for entailed ones)
      - Optionally include NOT_ENTAILED candidates (for debugging or training auxiliary losses)
      - Optionally filter clue restatements (removes atoms that match clue set)

    Returns list of dicts: {text, expr, entailment, origin}
    """
    if not is_sat(constraints):
        return []

    menu: List[Dict[str, Any]] = []

    for text, expr in atoms:
        c = canon(text)

        # NOVEL filter
        if c in used_atoms:
            continue

        # Optional: don't even show clue restatements
        if filter_clue_restatements and (c in clue_atoms):
            continue

        ent = entailment_label(constraints, expr)

        if (ent == "NOT_ENTAILED") and (not include_not_entailed):
            continue

        origin = ""
        if ent == "ENTAILED":
            origin = "CLUE" if (c in clue_atoms) else "DERIVED"
        else:
            origin = "N/A"

        menu.append({
            "text": text,
            "expr": expr,
            "entailment": ent,
            "origin": origin,
        })

        if len(menu) >= max_menu:
            break

    return menu

# -----------------------------
# Demo runner
# -----------------------------
def print_menu(menu: List[Dict[str, Any]], title: str):
    print(f"\n=== {title} ===")
    if not menu:
        print("(empty)")
        return

    for i, m in enumerate(menu):
        # Example display:
        # 03) [ENTAILED] [DERIVED] bob == 3
        print(f"{i:02d}) [{m['entailment']:^12}] [{m['origin']:^7}] {m['text']}")

def main():
    constraints, z3_vars, clue_atoms, n_houses = build_puzzle()
    atoms = generate_atom_universe(z3_vars, n_houses)

    used_atoms: Set[str] = set()

    # Step 0: show entailed-only menu (normal mode)
    menu0 = z3_candidate_menu(
        constraints=constraints,
        atoms=atoms,
        used_atoms=used_atoms,
        clue_atoms=clue_atoms,
        include_not_entailed=False,       # production: False
        filter_clue_restatements=False,   # set True if you want only DERIVED entailed facts
        max_menu=60,
    )
    print_menu(menu0, "Candidate Menu Step 0 (ENTAILED + NOVEL)")

    # Debug view: show both entailed and not-entailed
    menu0_debug = z3_candidate_menu(
        constraints=constraints,
        atoms=atoms,
        used_atoms=used_atoms,
        clue_atoms=clue_atoms,
        include_not_entailed=True,        # debugging: True
        filter_clue_restatements=False,
        max_menu=30,
    )
    print_menu(menu0_debug, "Candidate Menu Step 0 DEBUG (includes NOT_ENTAILED)")

    # Simulate "LLM chooses an index": pick the first ENTAILED+DERIVED if available; else first ENTAILED
    pick = None
    for i, m in enumerate(menu0):
        if m["entailment"] == "ENTAILED" and m["origin"] == "DERIVED":
            pick = i
            break
    if pick is None and menu0:
        pick = 0

    if pick is None:
        print("\nNo candidates to pick.")
        return

    chosen = menu0[pick]
    print(f"\n[simulate LLM] choose index {pick}: {chosen['text']}  ({chosen['entailment']}, {chosen['origin']})")

    # Mark as used (NOVEL filter) and optionally add to constraints as an "accepted reasoning step"
    used_atoms.add(canon(chosen["text"]))
    constraints = constraints + [chosen["expr"]]

    # Step 1: menu after one chosen step
    menu1 = z3_candidate_menu(
        constraints=constraints,
        atoms=atoms,
        used_atoms=used_atoms,
        clue_atoms=clue_atoms,
        include_not_entailed=False,
        filter_clue_restatements=False,
        max_menu=60,
    )
    print_menu(menu1, "Candidate Menu Step 1 (after choosing one step)")

if __name__ == "__main__":
    main()
