# z3_reward_package_minimal.py
# pip install z3-solver

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Int, Solver, Distinct, And, Abs, Or, sat


# =============================================================================
# Parsing: parsed_clues DSL
# =============================================================================

@dataclass(frozen=True)
class Atom:
    attr: str
    val: str

@dataclass(frozen=True)
class Clause:
    cid: str
    pred: str
    args: Tuple[Any, ...]


_CLAUSE_RE = re.compile(r"^\s*(C\d+)\s*=\s*([a-z_]+)\((.*)\)\.\s*$")


def _split_args(arg_str: str) -> List[str]:
    parts, buf, depth = [], [], 0
    for ch in arg_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _clean_token(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and (s[0] == s[-1] and s[0] in ("`", '"', "'")):
        s = s[1:-1].strip()
    return s


def parse_atom(s: str) -> Atom:
    s = s.strip()
    if "=" not in s:
        raise ValueError(f"Invalid atom format '{s}'. Expected Attr=val.")
    attr, val = s.split("=", 1)
    attr = _clean_token(attr)
    val = _clean_token(val)
    if not attr or not val:
        raise ValueError(f"Invalid atom format '{s}'. Empty attr/val.")
    return Atom(attr=attr, val=val)


def parse_clause(line: str) -> Clause:
    m = _CLAUSE_RE.match(line)
    if not m:
        raise ValueError(f"Invalid clause line: {line}")

    cid, pred, arg_str = m.group(1), m.group(2), m.group(3)
    raw_args = _split_args(arg_str)

    if pred in ("set", "not_set"):
        if len(raw_args) != 3:
            raise ValueError(f"{pred} expects 3 args (H,Attr,Val). Got: {raw_args}")
        h = int(_clean_token(raw_args[0]))
        attr = _clean_token(raw_args[1])
        val = _clean_token(raw_args[2])
        return Clause(cid=cid, pred=pred, args=(h, Atom(attr, val)))

    if pred in ("same_house", "left_of", "right_of", "adjacent", "immediately_left_of"):
        if len(raw_args) != 2:
            raise ValueError(f"{pred} expects 2 atom args. Got: {raw_args}")
        a = parse_atom(raw_args[0])
        b = parse_atom(raw_args[1])
        return Clause(cid=cid, pred=pred, args=(a, b))

    if pred == "between":
        if len(raw_args) != 3:
            raise ValueError(f"between expects 3 args (AtomA,AtomB,K). Got: {raw_args}")
        a = parse_atom(raw_args[0])
        b = parse_atom(raw_args[1])
        k = int(_clean_token(raw_args[2]))
        return Clause(cid=cid, pred=pred, args=(a, b, k))

    raise ValueError(f"Unknown predicate: {pred}")


# =============================================================================
# Z3 model
# =============================================================================

class ZebraSolver:
    """
    pos[attr][val] = house index (1..N). Distinct per attribute.
    """

    def __init__(self, n_houses: int, attribute_values: Dict[str, List[str]]):
        self.N = n_houses
        self.attribute_values = attribute_values
        self.s = Solver()

        self.pos: Dict[str, Dict[str, Any]] = {}
        for attr, vals in self.attribute_values.items():
            self.pos[attr] = {v: Int(f"pos_{attr}_{v}") for v in vals}

        for _, m in self.pos.items():
            for _, z in m.items():
                self.s.add(And(z >= 1, z <= self.N))
            self.s.add(Distinct(*m.values()))

    def _require_known(self, atom: Atom):
        if atom.attr not in self.pos:
            raise KeyError(f"Unknown attribute: {atom.attr}")
        if atom.val not in self.pos[atom.attr]:
            raise KeyError(f"Unknown value '{atom.val}' for attribute '{atom.attr}'")

    def p(self, atom: Atom):
        self._require_known(atom)
        return self.pos[atom.attr][atom.val]

    def add_clause(self, clause: Clause):
        pred = clause.pred

        if pred == "set":
            h, atom = clause.args
            self.s.add(self.p(atom) == h); return

        if pred == "not_set":
            h, atom = clause.args
            self.s.add(self.p(atom) != h); return

        if pred == "same_house":
            a, b = clause.args
            self.s.add(self.p(a) == self.p(b)); return

        if pred == "left_of":
            a, b = clause.args
            self.s.add(self.p(a) < self.p(b)); return

        if pred == "right_of":
            a, b = clause.args
            self.s.add(self.p(a) > self.p(b)); return

        if pred == "adjacent":
            a, b = clause.args
            self.s.add(Abs(self.p(a) - self.p(b)) == 1); return

        if pred == "immediately_left_of":
            a, b = clause.args
            self.s.add(self.p(a) + 1 == self.p(b)); return

        if pred == "between":
            a, b, k = clause.args
            self.s.add(Abs(self.p(a) - self.p(b)) == k + 1); return

        raise ValueError(f"Unhandled predicate: {pred}")

    def is_sat(self) -> bool:
        return self.s.check() == sat

    def solve_table(self) -> Optional[Dict[str, Any]]:
        if self.s.check() != sat:
            return None
        m = self.s.model()

        by_house: Dict[int, Dict[str, str]] = {h: {} for h in range(1, self.N + 1)}
        for attr, vals in self.attribute_values.items():
            for v in vals:
                h = m[self.pos[attr][v]].as_long()
                by_house[h][attr] = v

        header = ["House"] + list(self.attribute_values.keys())
        rows = []
        for h in range(1, self.N + 1):
            rows.append([str(h)] + [by_house[h][attr] for attr in self.attribute_values.keys()])
        return {"header": header, "rows": rows}

    def is_unique(self) -> bool:
        """Checks if there is exactly one model (by blocking the current model)."""
        if self.s.check() != sat:
            return False
        m = self.s.model()

        diffs = []
        for _, mvars in self.pos.items():
            for _, z in mvars.items():
                diffs.append(z != m[z].as_long())

        s2 = Solver()
        s2.add(self.s.assertions())
        s2.add(Or(*diffs))  # enforce at least one var differs
        return s2.check() != sat  # if no second model, unique


# =============================================================================
# Helpers
# =============================================================================

def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        x = float(x)
    except Exception:
        return None
    return max(0.0, min(1.0, x))

def _validate_domain(n_houses: int, attribute_values: Dict[str, List[str]]) -> Tuple[bool, str]:
    if not isinstance(n_houses, int) or n_houses <= 0:
        return False, "n_houses must be positive int"
    if not isinstance(attribute_values, dict) or not attribute_values:
        return False, "attribute_values must be non-empty dict"
    for attr, vals in attribute_values.items():
        if not isinstance(attr, str) or not attr.strip():
            return False, "attribute name must be non-empty str"
        if not isinstance(vals, list) or len(vals) != n_houses:
            return False, f"attribute '{attr}' must have exactly N={n_houses} values"
        if len(set(vals)) != len(vals):
            return False, f"attribute '{attr}' has duplicate values"
    return True, "ok"

def _tables_equal(a: Any, b: Any) -> bool:
    return isinstance(a, dict) and isinstance(b, dict) and a.get("header") == b.get("header") and a.get("rows") == b.get("rows")

def _cell_accuracy(solution: Any, z3_table: Any) -> float:
    """Fraction of non-house cells that match between solution and z3_table."""
    try:
        if not isinstance(solution, dict) or not isinstance(z3_table, dict):
            return 0.0
        if solution.get("header") != z3_table.get("header"):
            return 0.0
        rows_a = solution.get("rows", [])
        rows_b = z3_table.get("rows", [])
        if len(rows_a) != len(rows_b) or len(rows_a) == 0:
            return 0.0
        total = 0
        match = 0
        for ra, rb in zip(rows_a, rows_b):
            if not (isinstance(ra, list) and isinstance(rb, list)) or len(ra) != len(rb):
                return 0.0
            for j in range(1, len(ra)):  # skip House column
                total += 1
                if ra[j] == rb[j]:
                    match += 1
        return match / total if total > 0 else 0.0
    except Exception:
        return 0.0


# =============================================================================
# Minimal Z3-based reward: your formula + "honesty gate"
# =============================================================================

def compute_z3_reward(
    parsed_clues: List[str],
    parsed_reasoning: List[str],  # unused in this minimal fix, kept for interface stability
    solution: Dict[str, Any],
    attribute_values: Dict[str, List[str]],
    n_houses: int,
    puzzle_accuracy: Optional[float] = None,  # (0..1)
    cell_accuracy: Optional[float] = None,    # (0..1)
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Minimal fix to improve Puzzle-Accuracy:
      reward = 1.0 if puzzle_accuracy == 1 else min(0.95, 0.2 + 0.75 * cell_acc)

    + Z3 honesty gate:
      If constraints are UNSAT or the provided solution doesn't match Z3's implied solution,
      cap the base reward (and make UNSAT slightly negative).
    """
    cap_when_inconsistent = 0.25
    unsat_negative = -0.1
    if weights:
        cap_when_inconsistent = float(weights.get("cap_when_inconsistent", cap_when_inconsistent))
        unsat_negative = float(weights.get("unsat_negative", unsat_negative))

    metrics: Dict[str, Any] = {
        "domain_ok": 0,
        "domain_msg": "",
        "parse_cov": 0.0,
        "clues_parsed_ok": 0,
        "clues_parse_error": None,
        "sat": 0,
        "unique": 0,
        "solution_match": 0,
        "cell_acc_z3": 0.0,
        "puzzle_accuracy_used": 0.0,
        "cell_accuracy_used": 0.0,
        "reward_reason": "",
        "reward": 0.0,
    }

    # Domain validation
    ok, msg = _validate_domain(n_houses, attribute_values)
    metrics["domain_ok"] = int(ok)
    metrics["domain_msg"] = msg
    if not ok:
        metrics["reward"] = -1.0
        metrics["reward_reason"] = "invalid_domain"
        return -1.0, metrics

    # Parse clues
    if not isinstance(parsed_clues, list) or len(parsed_clues) == 0:
        metrics["reward"] = -1.0
        metrics["reward_reason"] = "no_or_bad_clues"
        return -1.0, metrics

    clauses: List[Clause] = []
    parsed_ok = 0
    first_err = None
    for line in parsed_clues:
        try:
            clauses.append(parse_clause(line))
            parsed_ok += 1
        except Exception as e:
            if first_err is None:
                first_err = str(e)

    metrics["clues_parsed_ok"] = parsed_ok
    metrics["clues_parse_error"] = first_err
    metrics["parse_cov"] = parsed_ok / max(1, len(parsed_clues))

    if parsed_ok == 0:
        metrics["reward"] = -1.0
        metrics["reward_reason"] = "all_clues_unparseable"
        return -1.0, metrics

    # Build Z3
    try:
        zs = ZebraSolver(n_houses=n_houses, attribute_values=attribute_values)
        for c in clauses:
            zs.add_clause(c)
    except Exception as e:
        metrics["reward"] = -1.0
        metrics["reward_reason"] = "z3_build_failed"
        metrics["domain_msg"] = f"z3_build_failed: {e}"
        return -1.0, metrics

    # SAT + Z3-derived solution
    sat_ok = zs.is_sat()
    metrics["sat"] = int(sat_ok)

    z3_table = zs.solve_table() if sat_ok else None
    solution_match = int(z3_table is not None and _tables_equal(solution, z3_table))
    metrics["solution_match"] = solution_match
    metrics["unique"] = int(zs.is_unique() if sat_ok else False)

    cell_acc_z3 = _cell_accuracy(solution, z3_table) if z3_table is not None else 0.0
    metrics["cell_acc_z3"] = float(cell_acc_z3)

    # Accuracy inputs (use provided when present; for cell acc use max with z3-derived)
    pa_in = _clip01(puzzle_accuracy)
    ca_in = _clip01(cell_accuracy)

    pa_used = pa_in if pa_in is not None else float(solution_match)
    ca_used = max(ca_in if ca_in is not None else 0.0, float(cell_acc_z3))

    metrics["puzzle_accuracy_used"] = float(pa_used)
    metrics["cell_accuracy_used"] = float(ca_used)

    # Your original reward shape
    if pa_used >= 1.0:
        reward = 1.0
        # tiny penalty if "correct" but constraints don't support it
        if (not sat_ok) or (solution_match == 0):
            reward = max(0.8, reward - 0.2)
            metrics["reward_reason"] = "puzzle_correct_but_inconsistent"
        else:
            metrics["reward_reason"] = "puzzle_correct"
        metrics["reward"] = float(reward)
        return float(reward), metrics

    base = min(0.95, 0.2 + 0.75 * ca_used)

    # Minimal Z3 honesty gate (key improvement)
    if (not sat_ok) or (solution_match == 0):
        reward = min(base, cap_when_inconsistent)
        if not sat_ok:
            reward = min(reward, unsat_negative)  # ensure <0 for UNSAT
            metrics["reward_reason"] = "unsat_capped_negative"
        else:
            metrics["reward_reason"] = "solution_mismatch_capped"
    else:
        reward = base
        metrics["reward_reason"] = "base_ok"

    metrics["reward"] = float(reward)
    return float(reward), metrics


# =============================================================================
# Working examples
# =============================================================================

def ex_good_correct() -> Dict[str, Any]:
    return {
        "n_houses": 3,
        "attribute_values": {
            "Name": ["Arnold", "Eric", "Peter"],
            "Drink": ["milk", "water", "tea"],
            "Hobby": ["photography", "cooking", "gardening"],
        },
        "parsed_clues": [
            "C1 = set(2,Name,Peter).",
            "C2 = immediately_left_of(Name=Arnold,Drink=water).",
            "C3 = immediately_left_of(Drink=water,Drink=milk).",
        ],
        "parsed_reasoning": [],
        "solution": {
            "header": ["House", "Name", "Drink", "Hobby"],
            "rows": [
                ["1", "Arnold", "water", "gardening"],
                ["2", "Peter", "milk", "cooking"],
                ["3", "Eric", "tea", "photography"],
            ],
        },
        "puzzle_accuracy": 1.0,
        "cell_accuracy": 1.0,
    }

def ex_wrong_solution_high_cell_but_mismatch() -> Dict[str, Any]:
    # Puzzle wrong but cell_accuracy provided high -> Z3 mismatch cap should prevent high reward
    ex = ex_good_correct()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.9  # tries to game your dense reward
    ex["solution"] = {
        "header": ["House", "Name", "Drink", "Hobby"],
        "rows": [
            ["1", "Arnold", "milk", "gardening"],   # wrong
            ["2", "Peter", "water", "cooking"],     # wrong
            ["3", "Eric", "tea", "photography"],    # maybe right
        ],
    }
    return ex

def ex_unsat_constraints() -> Dict[str, Any]:
    # Contradiction: Peter is forced in house 2 and also not allowed in house 2
    ex = ex_good_correct()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    ex["parsed_clues"] = ex["parsed_clues"] + [
        "C4 = not_set(2,Name,Peter).",
    ]
    return ex


if __name__ == "__main__":
    examples = [
        ("GOOD / correct", ex_good_correct()),
        ("WRONG solution but high cell_acc (should be capped)", ex_wrong_solution_high_cell_but_mismatch()),
        ("UNSAT constraints (should be negative)", ex_unsat_constraints()),
    ]

    for title, d in examples:
        r, m = compute_z3_reward(
            parsed_clues=d["parsed_clues"],
            parsed_reasoning=d["parsed_reasoning"],
            solution=d["solution"],
            attribute_values=d["attribute_values"],
            n_houses=d["n_houses"],
            puzzle_accuracy=d.get("puzzle_accuracy"),
            cell_accuracy=d.get("cell_accuracy"),
            weights={"cap_when_inconsistent": 0.25, "unsat_negative": -0.1},
        )
        print("\n" + "=" * 90)
        print(title)
        print("reward:", r)
        print("metrics:", {k: m.get(k) for k in [
            "sat", "unique", "parse_cov", "solution_match",
            "puzzle_accuracy_used", "cell_accuracy_used",
            "cell_acc_z3", "reward_reason"
        ]})
