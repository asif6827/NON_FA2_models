# pip install z3-solver
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

from z3 import Int, Solver, Distinct, And, Abs, Or, sat, unsat


# ----------------------------
# DSL parsing
# ----------------------------

@dataclass(frozen=True)
class Atom:
    attr: str
    val: str

@dataclass(frozen=True)
class Clause:
    cid: str                 # e.g., "C1"
    pred: str                # set, not_set, same_house, left_of, right_of, adjacent, immediately_left_of, between
    args: Tuple[Any, ...]    # parsed args (ints, Atom, etc.)


_CLAUSE_RE = re.compile(r'^\s*(C\d+)\s*=\s*([a-z_]+)\((.*)\)\.\s*$')
_ATOM_RE = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([a-z0-9_]+)\s*$')


def _split_args(arg_str: str) -> List[str]:
    # Split on commas but keep nested parentheses intact (simple for our DSL)
    parts = []
    buf = []
    depth = 0
    for ch in arg_str:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _clean_token(s: str) -> str:
    s = s.strip()
    # remove wrapping backticks or quotes if present
    if (len(s) >= 2) and (
        (s[0] == s[-1] and s[0] in ("`", '"', "'"))
    ):
        s = s[1:-1].strip()
    return s

def _parse_atom(s: str) -> Atom:
    """
    Accepts:
      Mother=Holly
      Birthday Month=april
      "Birthday Month"="grilled cheese"
      `Name`=`Eric`
    """
    s = s.strip()
    if "=" not in s:
        raise ValueError(f"Invalid atom format '{s}'. Expected Attr=val.")
    attr, val = s.split("=", 1)  # split only on first '='
    attr = _clean_token(attr)
    val = _clean_token(val)

    if not attr or not val:
        raise ValueError(f"Invalid atom format '{s}'. Empty attr or val.")
    return Atom(attr=attr, val=val)

def parse_clause(line: str) -> Clause:
    """
    Parses one DSL line like:
      C1 = set(2,Name,Peter).
      C2 = immediately_left_of(Name=Arnold,Drink=water).
      C3 = between(Name=Arnold,Drink=water,1).
    """
    m = _CLAUSE_RE.match(line)
    if not m:
        raise ValueError(f"Invalid clause line: {line}")

    cid, pred, arg_str = m.group(1), m.group(2), m.group(3)
    raw_args = _split_args(arg_str)

    if pred in ("set", "not_set"):
        if len(raw_args) != 3:
            raise ValueError(f"{pred} expects 3 args: (H,Attr,Val). Got: {raw_args}")
        h = int(raw_args[0])
        attr = raw_args[1].strip()
        val = raw_args[2].strip()
        return Clause(cid=cid, pred=pred, args=(h, Atom(attr, val)))

    if pred in ("same_house", "left_of", "right_of", "adjacent", "immediately_left_of"):
        if len(raw_args) != 2:
            raise ValueError(f"{pred} expects 2 atom args. Got: {raw_args}")
        a = _parse_atom(raw_args[0])
        b = _parse_atom(raw_args[1])
        return Clause(cid=cid, pred=pred, args=(a, b))

    if pred == "between":
        if len(raw_args) != 3:
            raise ValueError(f"between expects 3 args: (AtomA,AtomB,K). Got: {raw_args}")
        a = _parse_atom(raw_args[0])
        b = _parse_atom(raw_args[1])
        k = int(raw_args[2])
        return Clause(cid=cid, pred=pred, args=(a, b, k))

    raise ValueError(f"Unknown predicate: {pred}")


# ----------------------------
# Z3 model builder
# ----------------------------

class ZebraSolver:
    """
    Model style:
      pos[attr][val] = house index (1..N)
    Ensures Distinct positions for values within each attribute.
    """

    def __init__(self, n_houses: int, attr_values: Dict[str, List[str]]):
        self.N = n_houses
        self.attr_values = attr_values
        self.s = Solver()

        # pos vars
        self.pos: Dict[str, Dict[str, Any]] = {}
        for attr, vals in self.attr_values.items():
            self.pos[attr] = {v: Int(f"pos_{attr}_{v}") for v in vals}

        # domains + uniqueness
        for attr, m in self.pos.items():
            for v, z in m.items():
                self.s.add(And(z >= 1, z <= self.N))
            self.s.add(Distinct(*m.values()))

    def _require_known(self, atom: Atom):
        if atom.attr not in self.pos:
            raise KeyError(f"Unknown attribute: {atom.attr}")
        if atom.val not in self.pos[atom.attr]:
            raise KeyError(f"Unknown value '{atom.val}' for attribute '{atom.attr}'")

    def _p(self, atom: Atom):
        self._require_known(atom)
        return self.pos[atom.attr][atom.val]

    def add_clause(self, clause: Clause):
        pred = clause.pred

        if pred == "set":
            h, atom = clause.args
            self.s.add(self._p(atom) == h)
            return

        if pred == "not_set":
            h, atom = clause.args
            self.s.add(self._p(atom) != h)
            return

        if pred == "same_house":
            a, b = clause.args
            self.s.add(self._p(a) == self._p(b))
            return

        if pred == "left_of":
            a, b = clause.args
            self.s.add(self._p(a) < self._p(b))
            return

        if pred == "right_of":
            a, b = clause.args
            self.s.add(self._p(a) > self._p(b))
            return

        if pred == "adjacent":
            a, b = clause.args
            self.s.add(Abs(self._p(a) - self._p(b)) == 1)
            return

        if pred == "immediately_left_of":
            a, b = clause.args
            self.s.add(self._p(a) + 1 == self._p(b))
            return

        if pred == "between":
            a, b, k = clause.args
            # exactly k houses strictly between => abs(posA - posB) == k + 1
            self.s.add(Abs(self._p(a) - self._p(b)) == k + 1)
            return

        raise ValueError(f"Unhandled predicate: {pred}")

    def solve(self) -> Optional[Dict[int, Dict[str, str]]]:
        if self.s.check() != sat:
            return None
        m = self.s.model()

        # invert positions to per-house assignment
        by_house: Dict[int, Dict[str, str]] = {h: {} for h in range(1, self.N + 1)}
        for attr, vals in self.attr_values.items():
            for v in vals:
                h = m[self.pos[attr][v]].as_long()
                by_house[h][attr] = v
        return by_house

    def to_table(self, by_house: Dict[int, Dict[str, str]]) -> Dict[str, Any]:
        header = ["House"] + list(self.attr_values.keys())
        rows = []
        for h in range(1, self.N + 1):
            row = [str(h)]
            for attr in self.attr_values.keys():
                row.append(by_house[h][attr])
            rows.append(row)
        return {"header": header, "rows": rows}

    # Optional: entailment check for a reasoning step
    def entails_set(self, h: int, atom: Atom) -> bool:
        """
        Returns True if constraints logically imply pos(atom)==h,
        i.e., adding pos(atom)!=h makes the system UNSAT.
        """
        self._require_known(atom)
        s2 = Solver()
        s2.add(self.s.assertions())
        s2.add(self._p(atom) != h)
        return s2.check() == unsat

    def entails_not(self, h: int, atom: Atom) -> bool:
        """
        Returns True if constraints imply pos(atom)!=h,
        i.e., adding pos(atom)==h makes the system UNSAT.
        """
        self._require_known(atom)
        s2 = Solver()
        s2.add(self.s.assertions())
        s2.add(self._p(atom) == h)
        return s2.check() == unsat


# ----------------------------
# End-to-end: from your LLM JSON to Z3 solve
# ----------------------------

def solve_from_llm_output(llm_json: Dict[str, Any], n_houses: int, attr_values: Dict[str, List[str]]) -> Dict[str, Any]:
    parsed_clues: List[str] = llm_json["parsed_clues"]
    clauses = [parse_clause(line) for line in parsed_clues]

    zs = ZebraSolver(n_houses=n_houses, attr_values=attr_values)
    for c in clauses:
        zs.add_clause(c)

    sol = zs.solve()
    if sol is None:
        return {"sat": False, "solution": None}

    return {"sat": True, "solution": zs.to_table(sol)}


# ----------------------------
# Example: your 2-house puzzle
# ----------------------------

if __name__ == "__main__":
    # Attributes derived from the puzzle text (you can auto-extract later)
    Puzzle = """There are 2 houses, numbered 1 to 2 from left to right, as seen from across the street. Each house is occupied by a different person.
              Each house has a unique attribute for each of the following characteristics:\n 
              - Each person has a unique name: Arnold, Eric 
              - Each person has a unique birthday month: sept, april 
              - The mothers' names in different houses are unique: Aniya, Holly
              
              ## Clues:
              1. The person whose mother's name is Holly is Eric.
              2. Arnold is the person whose birthday is in April.
              3. Eric is not in the second house."""

    attr_values = {
        "Name": ["Arnold", "Eric"],
        "Birthday_Month": ["sept", "april"],
        "Mother": ["Aniya", "Holly"],
    }
    N = 2

    # This is what your LLM should output in parsed_clues for the given puzzle:
    llm_json = {
        "parsed_clues": [
            "C1 = same_house(Mother=Holly,Name=Eric).",
            "C2 = same_house(Name=Arnold,Birthday_Month=april).",
            "C3 = not_set(2,Name,Eric).",
        ],
        "parsed_reasoning": [],
        "solution": {"header": [], "rows": []},
    }

    out = solve_from_llm_output(llm_json, n_houses=N, attr_values=attr_values)
    print(json.dumps(out, indent=2))
