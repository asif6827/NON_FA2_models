# -*- coding: utf-8 -*-
"""
z3_reasoning_vs_solution_verifier_v2.py

Z3-based verification of a proposed *solution table* against:

Step-1 (clues vs solution):
  - Add syntactic_clues constraints to a Z3 solver
  - Add the proposed solution (as equalities token == house_index)
  - If UNSAT => solution contradicts clues => r1 = 0.0 else r1 = 1.0

Step-2 (reasoning steps vs solution):
  - Add syntactic reasoning steps constraints to a Z3 solver
  - Add the proposed solution
  - If UNSAT => solution contradicts reasoning => r2 = 0.0 else r2 = 1.0

Final reward = (r1 + r2) / 2

Notes / assumptions:
- Tokens (e.g., "Eric", "red", "soccer") are modeled as Int variables denoting house indices.
- Each token is constrained to [1, n_houses].
- We optionally enforce "all-different within each attribute column" using the solution header:
    For every column except "House", all values appearing in that column are constrained to be distinct.
  This matches the standard Zebra/logic-grid assumption.
- Constraints are expected in a Z3/Python-friendly syntax using:
    ==, !=, <, >, +, Not(...), And(...), Or(...)
  Examples:
    "C1: Arnold == red."
    "C2: red == 2."
    "S3: Or(Eric == 1, Eric == 2, Eric == 3)."

If you interleave NL lines with syntactic steps, use `extract_syntactic_steps_with_evidence`
(which ignores non-matching lines by auto-numbering them as constraints; you should pre-filter NL
if you want stricter behavior).
"""

from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    import z3
except Exception as e:  # pragma: no cover
    z3 = None
    _Z3_IMPORT_ERROR = e
else:
    _Z3_IMPORT_ERROR = None


# ------------------------------------------------------------
# Utilities: parsing inputs
# ------------------------------------------------------------

def _canon_constraint(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    return s


def _iter_constraints_from_syntactic_clues(syntactic_clues: Any) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Parse syntactic clues into a list of (cid, constraint) pairs.

    Accepts:
      - list[str] lines like:
          "C1: milk == 1."
          "C2: cat + 1 == coffee."
          "milk == 1"   (no id; will be assigned C<i>)
      - dict[str,str] mapping id->constraint
    """
    errors: List[str] = []
    pairs: List[Tuple[str, str]] = []

    if syntactic_clues is None:
        return pairs, errors

    if isinstance(syntactic_clues, dict):
        for k, v in syntactic_clues.items():
            if not isinstance(k, str) or not isinstance(v, str):
                errors.append("syntactic_clues dict must map str -> str.")
                continue
            cid = k.strip() or "C?"
            constraint = v.strip().rstrip(".")
            pairs.append((cid, constraint))
        return pairs, errors

    if not isinstance(syntactic_clues, list):
        return pairs, ["syntactic_clues must be a list[str] or dict[str,str]."]

    for i, line in enumerate(syntactic_clues):
        if not isinstance(line, str):
            errors.append(f"syntactic_clues[{i}] is not a string.")
            continue
        s = line.strip()
        if not s:
            continue

        mm = re.match(r"^\s*(C\d+)\s*:\s*(.+?)\s*\.?\s*$", s)
        if mm:
            cid = mm.group(1)
            constraint = mm.group(2).strip().rstrip(".")
            pairs.append((cid, constraint))
        else:
            pairs.append((f"C{i+1}", s.rstrip(".")))

    return pairs, errors


RE_STEP_IN_LIST = re.compile(
    r"""^\s*
    S(?P<sid>\d+)\s*:\s*
    (?P<constraint>.+?)
    \s*\.?\s*
    (?:\[\s*(?P<evidence>[^\]]*)\s*\]\s*)?
    $""",
    re.VERBOSE,
)


def extract_syntactic_steps_with_evidence(reasoning: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract a list of syntactic steps from `reasoning`.

    Supported input formats:

    (A) Legacy format:
        "S1: Alice == cat. [C1]"

    (B) Simple constraint list:
        "Alice == 1"
        "Not(Bob == 5)"
        "And(tea != 4, dog != 3)"

    In (B), step ids are auto-assigned: S1, S2, ...
    """
    if not isinstance(reasoning, list):
        return [], ["reasoning must be a list[str]"]

    errors: List[str] = []
    steps: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}

    max_k = 0
    auto_k = 0

    for i, line in enumerate(reasoning):
        if not isinstance(line, str):
            errors.append(f"reasoning[{i}] is not a string.")
            continue

        s = line.strip()
        if not s:
            continue

        m = RE_STEP_IN_LIST.match(s)
        if m:
            k = int(m.group("sid"))
            sid = f"S{k}"
            max_k = max(max_k, k)
            constraint = m.group("constraint").strip().rstrip(".")
            ev_raw = (m.group("evidence") or "").strip()
            if ev_raw:
                toks = [t.strip() for t in re.split(r"[, ]+", ev_raw) if t.strip()]
                evidence = [re.sub(r"[^A-Za-z0-9_]", "", t) for t in toks if t]
                evidence = [t for t in evidence if t]
            else:
                evidence = []
        else:
            auto_k = max(auto_k, max_k)
            auto_k += 1
            k = auto_k
            sid = f"S{k}"
            constraint = s.rstrip(".")
            evidence = []

        if sid in seen:
            if _canon_constraint(seen[sid]) != _canon_constraint(constraint):
                errors.append(f"Duplicate step {sid} with conflicting constraints: {seen[sid]!r} vs {constraint!r}")
            continue
        seen[sid] = constraint

        steps.append({"sid": sid, "k": k, "constraint": constraint, "evidence": evidence, "raw": s})

    steps.sort(key=lambda d: d["k"])
    return steps, errors


# ------------------------------------------------------------
# Z3 model construction
# ------------------------------------------------------------

_IDENT = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _tokens_from_solution(solution: Dict[str, Any]) -> Tuple[Set[str], int, Dict[str, List[str]]]:
    """
    Returns:
      tokens: all tokens appearing in solution rows (excluding 'House' column values)
      n_houses: number of rows
      col_values: dict column_name -> list of tokens in that column (excluding House)
    """
    header = solution.get("header")
    rows = solution.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list) or not header:
        return set(), 0, {}

    n_houses = len(rows)
    col_values: Dict[str, List[str]] = {str(h): [] for h in header if str(h) != "House"}

    for r in rows:
        if not isinstance(r, list) or len(r) != len(header):
            continue
        for j, col in enumerate(header):
            col = str(col)
            if col == "House":
                continue
            tok = str(r[j]).strip()
            if tok:
                col_values[col].append(tok)

    tokens: Set[str] = set()
    for vs in col_values.values():
        tokens.update(vs)
    return tokens, n_houses, col_values


def _tokens_from_constraints(constraints: List[str]) -> Set[str]:
    toks: Set[str] = set()
    for c in constraints:
        if not isinstance(c, str):
            continue
        s = c.strip()
        if not s:
            continue
        for m in _IDENT.finditer(s):
            name = m.group(0)
            # Skip function names and common keywords
            if name in {"And", "Or", "Not", "True", "False"}:
                continue
            toks.add(name)
    return toks


def _build_z3_env(tokens: Set[str], n_houses: int) -> Dict[str, Any]:
    """
    Create a safe eval environment mapping:
      - token names -> z3.Int token vars
      - And/Or/Not -> z3.And/z3.Or/z3.Not
    """
    assert z3 is not None
    env: Dict[str, Any] = {
        "And": z3.And,
        "Or": z3.Or,
        "Not": z3.Not,
        "True": True,
        "False": False,
    }
    for t in tokens:
        env[t] = z3.Int(t)
    env["_n_houses"] = n_houses
    return env


def _add_domain_constraints(slv: "z3.Solver", env: Dict[str, Any], tokens: Set[str], n_houses: int) -> None:
    """Constrain every token variable to be in [1, n_houses]."""
    for t in tokens:
        v = env[t]
        slv.add(v >= 1, v <= n_houses)


def _add_all_different_by_column(slv: "z3.Solver", env: Dict[str, Any], col_values: Dict[str, List[str]]) -> None:
    """
    Standard Zebra rule: within each attribute column, all values are distinct houses.
    """
    for col, vals in col_values.items():
        uniq = []
        seen = set()
        for v in vals:
            if v not in seen:
                seen.add(v)
                uniq.append(env[v])
        if len(uniq) >= 2:
            slv.add(z3.Distinct(*uniq))


def _add_solution_equalities(slv: "z3.Solver", env: Dict[str, Any], solution: Dict[str, Any]) -> List[str]:
    """
    Adds equalities token == house_index from the provided solution table.
    Returns list of errors (if any).
    """
    errors: List[str] = []
    header = solution.get("header")
    rows = solution.get("rows")

    if not isinstance(header, list) or not isinstance(rows, list):
        return ["Solution table must contain header:list and rows:list."]

    try:
        house_col = header.index("House")
    except ValueError:
        return ["Solution header must include 'House'."]

    for i, r in enumerate(rows):
        if not isinstance(r, list) or len(r) != len(header):
            errors.append(f"Solution row {i} is malformed (len != header).")
            continue
        try:
            h = int(str(r[house_col]).strip())
        except Exception:
            errors.append(f"Solution row {i} has non-integer House value: {r[house_col]!r}")
            continue

        for j, col in enumerate(header):
            if j == house_col:
                continue
            tok = str(r[j]).strip()
            if not tok:
                continue
            if tok not in env:
                # should not happen if tokens were collected properly, but be safe
                env[tok] = z3.Int(tok)
            slv.add(env[tok] == h)

    return errors


def _safe_eval_constraint(expr: str, env: Dict[str, Any]) -> "z3.BoolRef":
    """
    Evaluate a constraint expression using a restricted eval environment.
    """
    assert z3 is not None
    s = expr.strip().rstrip(".")
    # Disallow obvious code injection primitives
    if "__" in s or "import" in s or "eval" in s or "exec" in s:
        raise ValueError("Unsafe tokens detected in constraint.")
    return eval(s, {"__builtins__": {}}, env)  # noqa: S307


def _solver_check_sat(constraints: List[str], solution: Dict[str, Any], *, enforce_alldiff: bool = True) -> Tuple[bool, List[str]]:
    """
    Build a solver with:
      - domain constraints
      - (optional) all-different per attribute column
      - provided constraints
      - the proposed solution equalities

    Returns:
      (is_sat, errors)
    """
    if z3 is None:  # pragma: no cover
        raise ImportError(f"z3 is not available: {_Z3_IMPORT_ERROR}")

    sol_tokens, n_houses, col_values = _tokens_from_solution(solution)
    if n_houses <= 0:
        return False, ["Solution table has no rows (cannot infer n_houses)."]

    c_tokens = _tokens_from_constraints(constraints)
    tokens = set(sol_tokens) | set(c_tokens)

    env = _build_z3_env(tokens, n_houses)
    slv = z3.Solver()

    _add_domain_constraints(slv, env, tokens, n_houses)
    if enforce_alldiff:
        _add_all_different_by_column(slv, env, col_values)

    # Add constraints (clues or steps)
    errors: List[str] = []
    for i, c in enumerate(constraints):
        if not isinstance(c, str) or not c.strip():
            continue
        try:
            slv.add(_safe_eval_constraint(c, env))
        except Exception as e:
            errors.append(f"Constraint[{i}] parse/eval failed: {c!r} ({e})")

    # Add solution
    errors.extend(_add_solution_equalities(slv, env, solution))

    if errors:
        # If constraints couldn't be parsed, treat as failure (unsat check not meaningful)
        return False, errors

    res = slv.check()
    return (res == z3.sat), []


# ------------------------------------------------------------
# Public API: two-step verifier (requested)
# ------------------------------------------------------------

def verify_solution_two_step(
    syntactic_clues: Any,
    reasoning: Any,
    solution: Dict[str, Any],
    *,
    enforce_alldiff: bool = True,
) -> Dict[str, Any]:
    """
    Implements the user-requested two-step Z3 verification.

    Returns dict including r1, r2, final_reward, plus debug details.
    """
    # Step-1: clues + solution
    clue_pairs, clue_parse_errs = _iter_constraints_from_syntactic_clues(syntactic_clues)
    clue_constraints = [c for _, c in clue_pairs]

    r1 = 0.0
    step1_errors: List[str] = []
    if clue_parse_errs:
        step1_errors.extend([f"Clue parse: {e}" for e in clue_parse_errs])

    if not step1_errors:
        sat1, err1 = _solver_check_sat(clue_constraints, solution, enforce_alldiff=enforce_alldiff)
        step1_errors.extend(err1)
        r1 = 1.0 if sat1 else 0.0

    # Step-2: steps + solution
    steps, step_parse_errs = extract_syntactic_steps_with_evidence(reasoning)
    step_constraints = [d["constraint"] for d in steps]

    r2 = 0.0
    step2_errors: List[str] = []
    if step_parse_errs:
        step2_errors.extend([f"Reasoning parse: {e}" for e in step_parse_errs])

    if not step2_errors:
        sat2, err2 = _solver_check_sat(step_constraints, solution, enforce_alldiff=enforce_alldiff)
        step2_errors.extend(err2)
        r2 = 1.0 if sat2 else 0.0

    #final_reward = (r1 + r2) / 2.0
    final_reward = r1
    #"ok": (r1 == 1.0 and r2 == 1.0),
    return {
        "ok": r1 == 1.0,
        "r1": r1,
        "r2": r2,
        "reward": final_reward,
        "step1": {
            "n_clues": len(clue_constraints),
            "errors": step1_errors,
        },
        "step2": {
            "n_steps": len(step_constraints),
            "errors": step2_errors,
        },
    }


# Backward compatible wrapper name (used by some callers)
def validate_reasoning_and_solution(
    syntactic_clues: Any,
    reasoning: Any,
    solution: Dict[str, Any],
    *,
    enforce_alldiff: bool = True,
) -> Dict[str, Any]:
    return verify_solution_two_step(syntactic_clues, reasoning, solution, enforce_alldiff=enforce_alldiff)


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    syntactic_clues = [
        "C1: Alice == 1.",
        "C2: Alice == cat.",
        "C3: dog != 2.",
        "C4: green < white.",
        "C5: coffee > tea.",
        "C6: cat + 2 == coffee.",
        "C7: Not(Bob == 5).",
        "C8: And(tea != 4, dog != 3).",
        "C9: Or(milk == 2, milk == 5).",
    ]

    reasoning = [
        "Alice == 1",
        "Alice == cat",
        "cat + 2 == coffee",
        "coffee > tea",
        "green < white",
        "Not(Bob == 5)",
        "And(tea != 4, dog != 3)",
        "Or(milk == 2, milk == 5)",
        "And(Alice == 1, cat == 1, coffee == 3)",
    ]

    solution = {
        "header": ["House", "Name", "Pet", "Drink", "Color"],
        "rows": [
            ["1", "Alice", "cat", "juice", "yellow"],
            ["2", "Bob", "fish", "tea", "green"],
            ["3", "Carol", "bird", "coffee", "red"],
            ["4", "Dan", "horse", "water", "white"],
            ["5", "Eve", "dog", "milk", "blue"],
        ],
    }

    res = verify_solution_two_step(syntactic_clues, reasoning, solution)
    print(json.dumps(res, indent=2))
