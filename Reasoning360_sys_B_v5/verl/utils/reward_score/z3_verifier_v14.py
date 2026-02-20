# z3_reward_module_v2.py
# pip install z3-solver

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Int, Solver, Distinct, And, Abs, Or, sat, unsat


# =============================================================================
# DSL: parsed_clues parsing
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
        raise ValueError(f"Invalid atom format '{s}'. Empty attr or val.")
    return Atom(attr=attr, val=val)


def parse_clause(line: str) -> Clause:
    m = _CLAUSE_RE.match(line)
    if not m:
        raise ValueError(f"Invalid clause line: {line}")

    cid, pred, arg_str = m.group(1), m.group(2), m.group(3)
    raw_args = _split_args(arg_str)

    if pred in ("set", "not_set"):
        if len(raw_args) != 3:
            raise ValueError(f"{pred} expects 3 args: (H,Attr,Val). Got: {raw_args}")
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
            raise ValueError(f"between expects 3 args: (AtomA,AtomB,K). Got: {raw_args}")
        a = parse_atom(raw_args[0])
        b = parse_atom(raw_args[1])
        k = int(_clean_token(raw_args[2]))
        return Clause(cid=cid, pred=pred, args=(a, b, k))

    raise ValueError(f"Unknown predicate: {pred}")


# =============================================================================
# DSL: parsed_reasoning parsing
# =============================================================================

_REASON_RE = re.compile(
    r"^\s*S(\d+)\s*\[([^\]]+)\]\s*(set|not)\(\s*([0-9]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)\.\s*$"
)

@dataclass(frozen=True)
class ReasonStep:
    k: int
    evidence: str
    op: str   # "set" or "not"
    h: int
    atom: Atom

def parse_reason_step(line: str) -> ReasonStep:
    m = _REASON_RE.match(line)
    if not m:
        raise ValueError(f"Invalid reasoning line: {line}")
    k = int(m.group(1))
    evidence = _clean_token(m.group(2))
    op = m.group(3)
    h = int(m.group(4))
    attr = _clean_token(m.group(5))
    val = _clean_token(m.group(6))
    return ReasonStep(k=k, evidence=evidence, op=op, h=h, atom=Atom(attr, val))


# =============================================================================
# Z3 model
# =============================================================================

class ZebraSolver:
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
        if self.s.check() != sat:
            return False
        m = self.s.model()
        diffs = []
        for _, mvars in self.pos.items():
            for _, z in mvars.items():
                diffs.append(z != m[z].as_long())
        s2 = Solver()
        s2.add(self.s.assertions())
        s2.add(Or(*diffs))
        return s2.check() != sat

    def entails_set(self, h: int, atom: Atom) -> bool:
        self._require_known(atom)
        s2 = Solver()
        s2.add(self.s.assertions())
        s2.add(self.p(atom) != h)
        return s2.check() == unsat

    def entails_not(self, h: int, atom: Atom) -> bool:
        self._require_known(atom)
        s2 = Solver()
        s2.add(self.s.assertions())
        s2.add(self.p(atom) == h)
        return s2.check() == unsat

    def add_reason_step_as_constraint(self, step: ReasonStep):
        if step.op == "set":
            self.s.add(self.p(step.atom) == step.h)
        else:
            self.s.add(self.p(step.atom) != step.h)


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
    try:
        if not isinstance(solution, dict) or not isinstance(z3_table, dict):
            return 0.0
        if solution.get("header") != z3_table.get("header"):
            return 0.0
        rows_a = solution.get("rows", [])
        rows_b = z3_table.get("rows", [])
        if len(rows_a) != len(rows_b) or len(rows_a) == 0:
            return 0.0
        total, match = 0, 0
        for ra, rb in zip(rows_a, rows_b):
            if not (isinstance(ra, list) and isinstance(rb, list)) or len(ra) != len(rb):
                return 0.0
            for j in range(1, len(ra)):  # skip House
                total += 1
                if ra[j] == rb[j]:
                    match += 1
        return match / total if total > 0 else 0.0
    except Exception:
        return 0.0


# =============================================================================
# Reward (now accepts puzzle_accuracy + cell_accuracy inputs)
# =============================================================================

def compute_z3_reward(
    parsed_clues: List[str],
    parsed_reasoning: List[str],
    solution: Dict[str, Any],
    attribute_values: Dict[str, List[str]],
    n_houses: int,
    puzzle_accuracy: Optional[float] = None,  # NEW (0..1)
    cell_accuracy: Optional[float] = None,    # NEW (0..1)
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns (reward, metrics).

    - puzzle_accuracy is the primary correctness signal (drives reward).
    - Z3 still judges SAT/UNSAT, uniqueness, reasoning entailment, and
      (optionally) solution consistency with constraints via solution_match.
    """

    # weights used when puzzle is correct (or near correct)
    w = {
        "puzzle_acc": 1.2,
        "unique": 0.2,
        "parse_cov": 0.25,
        "reason_entail": 0.25,
        "inconsistency_pen": 0.2,  # penalty when puzzle_acc=1 but constraints don't support solution
        "parse_miss_pen": 0.2,
    }
    if weights:
        w.update(weights)

    # stable metrics defaults
    metrics: Dict[str, Any] = {
        "domain_ok": 0,
        "domain_msg": "",
        "num_clues": 0,
        "clues_parsed_ok": 0,
        "clues_parse_error": None,
        "parse_cov": 0.0,
        "sat": 0,
        "unique": 0,
        "z3_solution": None,
        "solution_match": 0,
        "reason_entail_rate": 0.0,
        "reason_steps_total": 0,
        "reason_steps_entailed": 0,
        "reason_steps_bad_parse_or_eval": 0,
        "reason_first_error": None,
        "puzzle_accuracy_in": puzzle_accuracy,
        "cell_accuracy_in": cell_accuracy,
        "puzzle_accuracy_used": 0.0,
        "cell_accuracy_used": 0.0,
        "reward": 0.0,
        "reward_reason": "",
    }

    # 1) Domain validation
    ok, msg = _validate_domain(n_houses, attribute_values)
    metrics["domain_ok"] = int(ok)
    metrics["domain_msg"] = msg
    if not ok:
        metrics["reward"] = -1.0
        metrics["reward_reason"] = "invalid_domain"
        return -1.0, metrics

    # 2) Parse clues
    if not isinstance(parsed_clues, list) or len(parsed_clues) == 0:
        metrics["reward"] = -1.0
        metrics["reward_reason"] = "no_or_bad_clues"
        return -1.0, metrics

    metrics["num_clues"] = len(parsed_clues)

    clauses: List[Clause] = []
    parsed_ok = 0
    first_parse_error = None
    for line in parsed_clues:
        try:
            clauses.append(parse_clause(line))
            parsed_ok += 1
        except Exception as e:
            if first_parse_error is None:
                first_parse_error = str(e)

    parse_cov = parsed_ok / max(1, len(parsed_clues))
    metrics["clues_parsed_ok"] = parsed_ok
    metrics["clues_parse_error"] = first_parse_error
    metrics["parse_cov"] = parse_cov

    if parsed_ok == 0:
        metrics["reward"] = -1.0
        metrics["reward_reason"] = "all_clues_unparseable"
        return -1.0, metrics

    # 3) Z3 build + SAT gate
    try:
        zs = ZebraSolver(n_houses=n_houses, attribute_values=attribute_values)
        for c in clauses:
            zs.add_clause(c)
    except Exception as e:
        metrics["reward"] = -1.0
        metrics["reward_reason"] = "z3_build_failed"
        metrics["domain_msg"] = f"z3_build_failed: {e}"
        return -1.0, metrics

    sat_ok = zs.is_sat()
    metrics["sat"] = int(sat_ok)
    if not sat_ok:
        metrics["reward"] = -0.6
        metrics["reward_reason"] = "unsat_constraints"
        return -0.6, metrics

    # 4) Z3 solve + uniqueness
    z3_table = zs.solve_table()
    metrics["z3_solution"] = z3_table
    unique = zs.is_unique()
    metrics["unique"] = int(unique)

    # 5) Consistency check: does LLM solution match the Z3-implied solution?
    sol_match = int(z3_table is not None and _tables_equal(solution, z3_table))
    metrics["solution_match"] = sol_match

    # computed cell-accuracy vs z3 (useful even if you supply external cell_accuracy)
    computed_cell_acc = _cell_accuracy(solution, z3_table) if z3_table is not None else 0.0

    # 6) Reasoning entailment rate
    entail_ok = 0
    entail_total = 0
    bad_reason_steps = 0
    first_reason_error = None

    zs_reason = ZebraSolver(n_houses=n_houses, attribute_values=attribute_values)
    for c in clauses:
        zs_reason.add_clause(c)

    if isinstance(parsed_reasoning, list) and len(parsed_reasoning) > 0:
        steps: List[ReasonStep] = []
        for line in parsed_reasoning:
            try:
                steps.append(parse_reason_step(line))
            except Exception as e:
                bad_reason_steps += 1
                if first_reason_error is None:
                    first_reason_error = str(e)

        steps.sort(key=lambda s: s.k)

        for st in steps:
            entail_total += 1
            try:
                ok_ent = zs_reason.entails_set(st.h, st.atom) if st.op == "set" else zs_reason.entails_not(st.h, st.atom)
                if ok_ent:
                    entail_ok += 1
                    zs_reason.add_reason_step_as_constraint(st)
            except Exception as e:
                bad_reason_steps += 1
                if first_reason_error is None:
                    first_reason_error = str(e)

    reason_entail_rate = (entail_ok / entail_total) if entail_total > 0 else 0.0
    metrics["reason_entail_rate"] = reason_entail_rate
    metrics["reason_steps_total"] = entail_total
    metrics["reason_steps_entailed"] = entail_ok
    metrics["reason_steps_bad_parse_or_eval"] = bad_reason_steps
    metrics["reason_first_error"] = first_reason_error

    # 7) Choose which puzzle/cell accuracy to use
    pa_in = _clip01(puzzle_accuracy)
    ca_in = _clip01(cell_accuracy)

    # If not provided, fall back to Z3-based versions:
    pa_used = pa_in if pa_in is not None else float(sol_match)  # 0/1 if derived from exact match
    ca_used = ca_in if ca_in is not None else float(computed_cell_acc)

    metrics["puzzle_accuracy_used"] = pa_used
    metrics["cell_accuracy_used"] = ca_used

    # 8) Reward shaping
    # Main objective: maximize puzzle accuracy
    if pa_used < 1.0:
        # HARD CAP when puzzle is wrong: cannot get high reward.
        shaped = (
            0.10 * parse_cov
            + 0.10 * reason_entail_rate
            + 0.20 * ca_used
            - 0.10 * (1.0 - parse_cov)
        )
        reward = max(-1.0, min(0.2, float(shaped) - 0.05))
        metrics["reward"] = reward
        metrics["reward_reason"] = "puzzle_incorrect_capped"
        return reward, metrics

    # If puzzle is correct, give high reward, but keep constraints honest
    inconsistency_pen = w["inconsistency_pen"] * (1 - sol_match)  # penalize if constraints don't support the solution
    reward = (
        w["puzzle_acc"] * pa_used
        + w["unique"] * int(unique)
        + w["parse_cov"] * parse_cov
        + w["reason_entail"] * reason_entail_rate
        - w["parse_miss_pen"] * (1.0 - parse_cov)
        - inconsistency_pen
    )
    reward = max(-1.0, min(1.0, float(reward)))
    metrics["reward"] = reward
    metrics["reward_reason"] = "ok"
    return reward, metrics


# =============================================================================
# Examples (quick testing)
# =============================================================================

def _base_domain():
    return {
        "Name": ["Arnold", "Eric"],
        "Birthday Month": ["sept", "april"],
        "Mother": ["Aniya", "Holly"],
    }

def _example_good_with_external_acc() -> Dict[str, Any]:
    # Correct answer, provide puzzle_accuracy/cell_accuracy explicitly
    return {
        "n_houses": 2,
        "attribute_values": _base_domain(),
        "parsed_clues": [
            "C1 = same_house(Mother=Holly,Name=Eric).",
            "C2 = same_house(Name=Arnold,Birthday Month=april).",
            "C3 = not_set(2,Name,Eric).",
        ],
        "parsed_reasoning": [
            "S1 [C3] set(1,Name,Eric).",
            "S2 [C1+C3] set(1,Mother,Holly).",
            "S3 [C3] set(2,Name,Arnold).",
            "S4 [C2] set(2,Birthday Month,april).",
        ],
        "solution": {
            "header": ["House", "Name", "Birthday Month", "Mother"],
            "rows": [
                ["1", "Eric", "sept", "Holly"],
                ["2", "Arnold", "april", "Aniya"],
            ],
        },
        "puzzle_accuracy": 1.0,
        "cell_accuracy": 1.0,
    }

def _example_wrong_solution_with_external_acc() -> Dict[str, Any]:
    # Wrong final table, explicitly pass puzzle_accuracy=0
    ex = _example_good_with_external_acc()
    ex["solution"] = {
        "header": ["House", "Name", "Birthday Month", "Mother"],
        "rows": [
            ["1", "Arnold", "sept", "Holly"],
            ["2", "Eric", "april", "Aniya"],
        ],
    }
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    return ex

def _example_correct_table_but_bad_constraints() -> Dict[str, Any]:
    # Table is correct (puzzle_accuracy=1), but constraints are under/incorrect so solution_match=0
    ex = _example_good_with_external_acc()
    ex["parsed_clues"] = [
        "C1 = not_set(2,Name,Eric).",  # too weak to force full solution
    ]
    ex["puzzle_accuracy"] = 1.0
    ex["cell_accuracy"] = 1.0
    return ex
def _example_pa0_ca0_unsat() -> Dict[str, Any]:
    # Z3 UNSAT -> reward = -0.6 (negative) regardless of supplied accuracies
    ex = _example_good_with_external_acc()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    ex["parsed_clues"] = ex["parsed_clues"] + [
        "C4 = not_set(1,Name,Eric).",  # together with not_set(2,Name,Eric) makes Eric impossible
    ]
    return ex

def _example_pa0_ca0_domain_invalid_values_count() -> Dict[str, Any]:
    # Domain invalid: attribute list not length N -> reward = -1.0
    ex = _example_good_with_external_acc()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    ex["attribute_values"] = {
        "Name": ["Arnold", "Eric"],
        "Birthday Month": ["sept"],          # ❌ wrong length (should be 2)
        "Mother": ["Aniya", "Holly"],
    }
    return ex

def _example_pa0_ca0_domain_duplicates() -> Dict[str, Any]:
    # Domain invalid: duplicates in an attribute -> reward = -1.0
    ex = _example_good_with_external_acc()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    ex["attribute_values"] = {
        "Name": ["Eric", "Eric"],            # ❌ duplicates
        "Birthday Month": ["sept", "april"],
        "Mother": ["Aniya", "Holly"],
    }
    return ex

def _example_pa0_ca0_all_clues_unparseable() -> Dict[str, Any]:
    # None of the clues parse -> reward = -1.0 (negative)
    ex = _example_good_with_external_acc()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    ex["parsed_clues"] = [
        "C1 = samehouse(Mother=Holly,Name=Eric).",   # ❌ wrong predicate name
        "C2 = foo(bar).",                            # ❌ unknown predicate / bad format
        "C3 = notset(2,Name,Eric).",                 # ❌ wrong predicate name
    ]
    return ex

def _example_pa0_ca0_z3_build_failed_unknown_attr() -> Dict[str, Any]:
    # Clue parses, but refers to unknown attribute -> Z3 build fails -> reward = -1.0
    ex = _example_good_with_external_acc()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    ex["parsed_clues"] = [
        "C1 = same_house(UnknownAttr=Holly,Name=Eric).",  # ❌ UnknownAttr not in attribute_values
    ]
    return ex

def _example_pa0_ca0_z3_build_failed_unknown_value() -> Dict[str, Any]:
    # Clue parses, but value not in domain -> Z3 build fails -> reward = -1.0
    ex = _example_good_with_external_acc()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    ex["parsed_clues"] = [
        "C1 = same_house(Mother=Sarah,Name=Eric).",        # ❌ Sarah not in Mother domain
    ]
    return ex

def _example_pa0_ca0_sat_but_capped_negative() -> Dict[str, Any]:
    # SAT, but puzzle_acc=0 forces capped shaping.
    # With parse_cov=0.0 (all clues unparseable) we'd return -1.0 already.
    # So we create a SAT case with parse_cov=1 but no reasoning and bad accuracies.
    # In our cap formula: shaped = 0.10*1 + 0 + 0 + 0 - 0 - 0.05 = 0.05 -> not negative.
    # To make it negative while still SAT, we need parse_cov low enough but >0.
    # Example: 1 parsable, 9 unparseable -> parse_cov=0.1 => shaped = 0.01 - 0.09 - 0.05 = -0.13
    ex = _example_good_with_external_acc()
    ex["puzzle_accuracy"] = 0.0
    ex["cell_accuracy"] = 0.0
    ex["parsed_reasoning"] = []
    ex["parsed_clues"] = [
        "C1 = not_set(2,Name,Eric).",  # parsable
        # 9 unparseable junk lines:
        "C2 = BAD(bad).",
        "C3 = BAD(bad).",
        "C4 = BAD(bad).",
        "C5 = BAD(bad).",
        "C6 = BAD(bad).",
        "C7 = BAD(bad).",
        "C8 = BAD(bad).",
        "C9 = BAD(bad).",
        "C10 = BAD(bad).",
    ]
    return ex

if __name__ == "__main__":
    tests = [
        ("PA0 CA0 UNSAT (expect reward -0.6)", _example_pa0_ca0_unsat()),
        ("PA0 CA0 invalid domain length (expect -1.0)", _example_pa0_ca0_domain_invalid_values_count()),
        ("PA0 CA0 invalid domain duplicates (expect -1.0)", _example_pa0_ca0_domain_duplicates()),
        ("PA0 CA0 all clues unparseable (expect -1.0)", _example_pa0_ca0_all_clues_unparseable()),
        ("PA0 CA0 unknown attribute in clue (expect -1.0)", _example_pa0_ca0_z3_build_failed_unknown_attr()),
        ("PA0 CA0 unknown value in clue (expect -1.0)", _example_pa0_ca0_z3_build_failed_unknown_value()),
        ("PA0 CA0 SAT but parse_cov low => negative capped", _example_pa0_ca0_sat_but_capped_negative()),
    ]

    for title, data in tests:
        r, m = compute_z3_reward(
            parsed_clues=data["parsed_clues"],
            parsed_reasoning=data["parsed_reasoning"],
            solution=data["solution"],
            attribute_values=data["attribute_values"],
            n_houses=data["n_houses"],
            puzzle_accuracy=data.get("puzzle_accuracy"),
            cell_accuracy=data.get("cell_accuracy"),
        )

        print("\n" + "=" * 100)
        print(title)
        print("reward:", r)
        keys = [
            "sat", "unique", "parse_cov", "solution_match",
            "puzzle_accuracy_used", "cell_accuracy_used",
            "reason_entail_rate", "reward_reason"
        ]
        print("metrics:", {k: m.get(k) for k in keys})
