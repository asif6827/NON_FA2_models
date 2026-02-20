
# -*- coding: utf-8 -*-
"""
z3_reasoning_validator_v13_verifier_GT_clues_v4.py

Validates a proposed solution table against:
  (A) syntactic_clues (C1:, C2:, ...)
  (B) syntactic reasoning steps provided as a list of strings:
        ["S1: ...", "S2: ...", ...]
      (If you interleave NL + syntactic, you should pre-filter NL lines
       before calling this module.)

This module intentionally keeps the validation "lightweight":
- It does NOT require Z3, and it does NOT attempt full DSL parsing.
- It checks a useful subset of constraint forms using the final solution table:
    token == k
    token != k
    token < token / token > token
    A + d == B
    A == B (same house)
    Not(<subexpr>)
    And(arg1, arg2, ...)
    Or(arg1, arg2, ...)

If a constraint form is unsupported, it is reported as an error (configurable).
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _canon_constraint(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    return s

def _extract_solution_positions(solution: Dict[str, Any]) -> Dict[str, int]:
    """
    Build token -> house_index map from the solution table.
    Expects:
      solution["header"] includes "House"
      solution["rows"] rows aligned with header
    """
    header = solution.get("header")
    rows = solution.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list):
        return {}

    try:
        house_col = header.index("House")
    except ValueError:
        return {}

    pos: Dict[str, int] = {}
    for r in rows:
        if not isinstance(r, list) or len(r) != len(header):
            continue
        try:
            h = int(str(r[house_col]).strip())
        except Exception:
            continue
        for j, val in enumerate(r):
            if j == house_col:
                continue
            tok = str(val).strip()
            if tok:
                pos[tok] = h
    return pos

# ------------------------------------------------------------
# Parse syntactic clues
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Parse syntactic steps from interleaved reasoning list
# ------------------------------------------------------------

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

    Expected input form (strict by default):
      reasoning = [
        "S1: Alice == cat. [C1]",
        "S2: cat + 1 == coffee. [C2]",
        "S3: And(cat == 1, coffee == 2). [S2]"
      ]

    Returns:
      (steps, errors)
      where steps is a list of dicts:
        {
          "sid": "S1",
          "k": 1,
          "constraint": "Alice == cat",
          "evidence": ["C1"],
          "raw": "S1: Alice == cat. [C1]"
        }
    """
    if not isinstance(reasoning, list):
        return [], ["reasoning must be a list[str]"]

    errors: List[str] = []
    steps: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}

    for i, line in enumerate(reasoning):
        if not isinstance(line, str):
            errors.append(f"reasoning[{i}] is not a string.")
            continue

        s = line.strip()
        m = RE_STEP_IN_LIST.match(s)
        if not m:
            errors.append(f"Unparseable reasoning step at reasoning[{i}]: {s!r}")
            continue

        k = int(m.group("sid"))
        sid = f"S{k}"
        constraint = m.group("constraint").strip().rstrip(".")
        # Evidence is OPTIONAL in v4 input format.
        # If present, we parse it; if absent, we simply proceed.
        ev_raw = (m.group("evidence") or "").strip()
        if ev_raw:
            toks = [t.strip() for t in re.split(r"[, ]+", ev_raw) if t.strip()]
            evidence = [re.sub(r"[^A-Za-z0-9_]", "", t) for t in toks if t]
            evidence = [t for t in evidence if t]
        else:
            evidence = []

        # Detect conflicting duplicates
        if sid in seen:
            if _canon_constraint(seen[sid]) != _canon_constraint(constraint):
                errors.append(
                    f"Duplicate step {sid} with conflicting constraints: {seen[sid]!r} vs {constraint!r}"
                )
            # still record the first one only
            continue
        seen[sid] = constraint

        steps.append(
            {
                "sid": sid,
                "k": k,
                "constraint": constraint,
                "evidence": evidence,
                "raw": s,
            }
        )

    # Sort by numeric id for stable downstream handling
    steps.sort(key=lambda d: d["k"])
    return steps, errors


def extract_syntactic_steps_from_reasoning_list(reasoning: Any) -> Tuple[Dict[str, str], List[str]]:
    """Backward-compatible wrapper: returns {sid: constraint}, errors."""
    steps, errors = extract_syntactic_steps_with_evidence(reasoning)
    return {d["sid"]: d["constraint"] for d in steps}, errors
# ------------------------------------------------------------
# Constraint evaluation against a solution table
# ------------------------------------------------------------

def _split_top_level_args(s: str) -> List[str]:
    """
    Split a comma-separated argument list at top level, respecting nested parentheses.
    Example: "A==1,Not(B==2),Or(C==1,D==2)" -> ["A==1", "Not(B==2)", "Or(C==1,D==2)"]
    """
    args: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                args.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        args.append(tail)
    return args

def _eval_atomic(c: str, sol_pos: Dict[str, int]) -> Tuple[Optional[bool], Optional[str]]:
    """
    Evaluate an atomic constraint against solution mapping.
    Returns (value, error_reason). value=None indicates "unsupported".
    """
    def _get(tok: str) -> Optional[int]:
        return sol_pos.get(tok)

    # token == k
    mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)==(\d+)", c)
    if mm:
        tok, k = mm.group(1), int(mm.group(2))
        ht = _get(tok)
        if ht is None:
            return False, f"token `{tok}` not found in solution"
        return (ht == k), None

    # token != k
    mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)!=(\d+)", c)
    if mm:
        tok, k = mm.group(1), int(mm.group(2))
        ht = _get(tok)
        if ht is None:
            return False, f"token `{tok}` not found in solution"
        return (ht != k), None

    # A + d == B
    mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\+(\d+)==([A-Za-z_][A-Za-z0-9_]*)", c)
    if mm:
        a, d, b = mm.group(1), int(mm.group(2)), mm.group(3)
        ha, hb = _get(a), _get(b)
        if ha is None or hb is None:
            return False, f"token(s) not found in solution: `{a}`, `{b}`"
        return (ha + d == hb), None

    # A < B or A > B
    mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(<|>)([A-Za-z_][A-Za-z0-9_]*)", c)
    if mm:
        a, op, b = mm.group(1), mm.group(2), mm.group(3)
        ha, hb = _get(a), _get(b)
        if ha is None or hb is None:
            return False, f"token(s) not found in solution: `{a}`, `{b}`"
        return ((ha < hb) if op == "<" else (ha > hb)), None

    # A == B (same house), excluding token==k
    mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)==([A-Za-z_][A-Za-z0-9_]*)", c)
    if mm and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*==\d+", c):
        a, b = mm.group(1), mm.group(2)
        ha, hb = _get(a), _get(b)
        if ha is None or hb is None:
            return False, f"token(s) not found in solution: `{a}`, `{b}`"
        return (ha == hb), None

    return None, None

def eval_constraint_against_solution(constraint: str, sol_pos: Dict[str, int]) -> Tuple[Optional[bool], Optional[str]]:
    """
    Evaluate supported constraint forms against sol_pos.
    Returns:
      (True/False/None, reason)
      - value None means "unsupported"
    """
    c = _canon_constraint(constraint)

    # Not(...)
    if c.startswith("Not(") and c.endswith(")"):
        inner = c[4:-1].strip()
        v, reason = eval_constraint_against_solution(inner, sol_pos)
        if v is None:
            return None, None
        return (not v), reason

    # And(...)
    if c.startswith("And(") and c.endswith(")"):
        inner = c[4:-1]
        args = _split_top_level_args(inner)
        if not args:
            return None, None
        for a in args:
            v, reason = eval_constraint_against_solution(a, sol_pos)
            if v is None:
                return None, None
            if v is False:
                return False, reason
        return True, None

    # Or(...)
    if c.startswith("Or(") and c.endswith(")"):
        inner = c[3:-1]
        args = _split_top_level_args(inner)
        if not args:
            return None, None
        any_true = False
        last_reason = None
        for a in args:
            v, reason = eval_constraint_against_solution(a, sol_pos)
            if v is None:
                return None, None
            if v is True:
                any_true = True
                break
            last_reason = reason
        return (True if any_true else False), last_reason

    # Atomic
    return _eval_atomic(c, sol_pos)

# ------------------------------------------------------------
# Public validators
# ------------------------------------------------------------

def validate_constraints_against_solution(
    pairs: List[Tuple[str, str]],
    solution: Dict[str, Any],
    *,
    label: str,
    fail_on_unsupported: bool = True,
) -> Dict[str, Any]:
    """
    Validate a list of (id, constraint) pairs against a solution.
    """
    errors: List[str] = []
    checked: List[Dict[str, Any]] = []

    sol_pos = _extract_solution_positions(solution)
    if not sol_pos:
        errors.append("Solution table could not be parsed (missing header/rows/House).")

    for cid, constraint in pairs:
        checked.append({"id": cid, "constraint": constraint})
        if not sol_pos:
            continue

        v, reason = eval_constraint_against_solution(constraint, sol_pos)
        if v is None:
            if fail_on_unsupported:
                errors.append(f"{label} {cid}: unsupported constraint form: `{constraint}`")
            continue
        if v is False:
            if reason:
                errors.append(f"{label} {cid}: solution contradicts `{constraint}` ({reason}).")
            else:
                errors.append(f"{label} {cid}: solution contradicts `{constraint}`.")
    return {"ok": len(errors) == 0, "errors": errors, "checked": checked}

def validate_solution_against_reasoning_steps_and_syntactic_clues(
    syntactic_clues: Any,
    reasoning: Any,
    solution: Dict[str, Any],
    *,
    fail_on_unsupported: bool = True,
) -> Dict[str, Any]:
    """
    Validate solution against:
      - syntactic_clues (C<i>: ...)
      - syntactic reasoning steps (S<i>: ...)

    v4 input format:
      syntactic_clues = ["C1: ...", ...]
      reasoning = ["S1: ...", "S2: ...", ...]   # evidence brackets optional and ignored
    """
    # clues
    clue_pairs, clue_errs = _iter_constraints_from_syntactic_clues(syntactic_clues)
    clues_result = validate_constraints_against_solution(
        clue_pairs,
        solution,
        label="Clue",
        fail_on_unsupported=fail_on_unsupported,
    )
    if clue_errs:
        clues_result["errors"] = [f"Clue parse: {e}" for e in clue_errs] + clues_result["errors"]
        clues_result["ok"] = len(clues_result["errors"]) == 0


    # steps
    steps, step_parse_errs = extract_syntactic_steps_with_evidence(reasoning)

    step_pairs = [(d["sid"], d["constraint"]) for d in steps]
    steps_result = validate_constraints_against_solution(
        step_pairs,
        solution,
        label="Step",
        fail_on_unsupported=fail_on_unsupported,
    )

    # Attach parse errors
    if step_parse_errs:
        steps_result["errors"] = [f"Reasoning parse: {e}" for e in step_parse_errs] + steps_result["errors"]
    steps_result["ok"] = len(steps_result["errors"]) == 0

    ok = bool(clues_result.get("ok")) and bool(steps_result.get("ok"))
    return {
        "ok": ok,
        "clues_validation": clues_result,
        "steps_validation": steps_result,
        "n_steps_parsed": len(step_pairs),
    }


def validate_reasoning_and_solution(
    syntactic_clues: Any,
    reasoning: Any,
    solution: Dict[str, Any],
    *,
    fail_on_unsupported: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper matching your requested signature."""
    return validate_solution_against_reasoning_steps_and_syntactic_clues(
        syntactic_clues,
        reasoning,
        solution,
        fail_on_unsupported=fail_on_unsupported,
    )

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    # Example matching the required input form
    syntactic_clues = [
        "C1: Alice == cat.",
        "C2: cat + 1 == coffee."
    ]

    reasoning = [
        "S1: Alice == cat.",
        "S2: cat + 1 == coffee.",
        "S3: And(cat == 1, coffee == 2)."
    ]

    solution = {
        "header": ["House", "Name", "Pet", "Drink"],
        "rows": [
            ["1", "Alice", "cat", "tea"],
            ["2", "Bob", "dog", "coffee"]
        ]
    }

    res = validate_reasoning_and_solution(syntactic_clues, reasoning, solution)
    print("ok:", res["ok"])
    print("\n== clues_validation ==")
    for e in res["clues_validation"]["errors"]:
        print("-", e)
    print("\n== steps_validation ==")
    for e in res["steps_validation"]["errors"]:
        print("-", e)
