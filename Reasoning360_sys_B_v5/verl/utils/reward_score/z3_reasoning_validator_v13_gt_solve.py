#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V13.1_gt_solve.py

Z3 solver/validator for "syntactic_clues" (Z3-like textual constraints) and
interleaved "reasoning" where ONLY syntactic S<k>: lines are validated.

Key outputs (requested):
- base_sat_full: SAT after adding all successfully-parsed clues (non-conflict-tolerant).
- base_sat_raw:  SAT after adding clues under conflict-tolerant mode (skip conflicts).
- base_sat_full_GT: True iff full SAT and Z3 solution matches ground-truth.
- base_sat_raw_GT:  True iff raw SAT and Z3 solution matches ground-truth (conflict-tolerant).
- reward: if base_sat_full_GT then 1.0*pass_rate_strict_gt_validated else 0.2*pass_rate_strict_gt_validated

Also retains v12-style metrics listed by the user.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import And, Bool, Distinct, Int, Not, Or, Solver, sat, unsat  # type: ignore
from z3 import And as Z3And, Or as Z3Or, Not as Z3Not, Implies as Z3Implies, Xor as Z3Xor, If as Z3If


# ----------------------------- Normalization -----------------------------
_SLINE_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"""^[\s"'`“”‘’()\[\]{}<>]+|[\s"'`“”‘’()\[\]{}<>]+$""")
_TRAIL_PUNCT_RE = re.compile(r"""[.,;:!?]+$""")
_SEP_RE = re.compile(r"""[\s\-\/]+""")
_MULTI_US_RE = re.compile(r"_+")
_CANON_MONTHS = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
}

_MONTH_ALIASES = {
    # Jan
    "jan": "january", "jan.": "january",
    # Feb
    "feb": "february", "feb.": "february",
    # Mar
    "mar": "march", "mar.": "march",
    # Apr
    "apr": "april", "apr.": "april",
    # May
    "may": "may",
    # Jun
    "jun": "june", "jun.": "june",
    # Jul
    "jul": "july", "jul.": "july",
    # Aug
    "aug": "august", "aug.": "august",
    # Sep
    "sep": "september", "sep.": "september",
    "sept": "september", "sept.": "september",
    # Oct
    "oct": "october", "oct.": "october",
    # Nov
    "nov": "november", "nov.": "november",
    # Dec
    "dec": "december", "dec.": "december",
}

# =================================================================================
#           !.....COMPUTE DISTINCT STEPS....!


# ------------------- Step line parsing -------------------

# Accept: "S1: ..." or "S1 ...", keep only syntactic lines
_SLINE_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)

def _extract_syntactic_expr(line: str) -> Optional[Tuple[int, str]]:
    """
    Returns (k, expr) for lines like:
      S1: Arnold == cat. [C5]
    Ignores NL lines without S-prefix.
    """
    m = _SLINE_RE.match((line or "").strip())
    if not m:
        return None

    k = int(m.group(1))
    tail = m.group(2).strip()

    # remove evidence bracket if present
    if "[" in tail:
        tail = tail.split("[", 1)[0].strip()

    # strip trailing period
    if tail.endswith("."):
        tail = tail[:-1].strip()

    return (k, tail) if tail else None

from typing import Set
from z3 import Z3_OP_UNINTERPRETED

def _vars_in(phi) -> Set[str]:
    """Return variable names (tokens) referenced in phi."""
    names: Set[str] = set()

    def walk(e):
        if e.num_args() == 0:
            # leaf node: variable or numeral
            if e.decl().kind() == Z3_OP_UNINTERPRETED:
                names.add(e.decl().name())
            return
        for c in e.children():
            walk(c)

    walk(phi)
    return names
# ------------------- Relaxed base model (IMPORTANT) -------------------

def build_relaxed_base_solver_v13_style(
    n_houses: int,
    attribute_values: Dict[str, List[str]],
    *,
    allow_unassigned: bool = True,
    timeout_s: float = 2.0,
) -> Tuple[Solver, Dict[str, Any], Dict[str, List[Any]]]:
    """
    Same *token->Int(token)* mapping as v13, but with relaxed assumptions:

      - Domain: 0..N (0=unassigned) if allow_unassigned else 1..N
      - Uniqueness: conditional distinctness when allow_unassigned:
            (xi==0) OR (xj==0) OR (xi != xj)

    Returns:
      base_solver, var_map, attr_vars
    """
    N = int(n_houses)
    var_map: Dict[str, Any] = {}
    attr_vars: Dict[str, List[Any]] = {}

    for attr, values in (attribute_values or {}).items():
        vars_: List[Any] = []
        for v in values:
            tok = _norm_token(str(v))
            if tok in var_map:
                raise ValueError(f"Duplicate token across attributes after normalization: {tok!r}")
            iv = Int(tok)          # IMPORTANT: v13 uses Int(tok)
            var_map[tok] = iv
            vars_.append(iv)
        attr_vars[str(attr)] = vars_

    s = Solver()
    s.set("timeout", int(float(timeout_s) * 1000))

    for vars_ in attr_vars.values():
        # domain
        for x in vars_:
            if allow_unassigned:
                s.add(And(x >= 0, x <= N))
            else:
                s.add(And(x >= 1, x <= N))

        # uniqueness
        if len(vars_) > 1:
            if allow_unassigned:
                for i in range(len(vars_)):
                    for j in range(i + 1, len(vars_)):
                        xi, xj = vars_[i], vars_[j]
                        s.add(Or(xi == 0, xj == 0, xi != xj))
            else:
                # strict uniqueness
                # (same as Distinct but written pairwise to avoid extra import)
                for i in range(len(vars_)):
                    for j in range(i + 1, len(vars_)):
                        s.add(vars_[i] != vars_[j])

    return s, var_map, attr_vars


# ------------------- DSL step -> Z3 -------------------

_DSL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)\s*$")

def _split_top_level_args(s: str) -> List[str]:
    args, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        args.append("".join(cur).strip())
    return args

def _parse_attrval(s: str) -> Tuple[str, str]:
    # "Attr=Val"
    if "=" not in s:
        raise ValueError(f"Expected Attr=Val, got {s!r}")
    a, v = s.split("=", 1)
    return a.strip(), v.strip()

def _dsl_to_z3(expr: str, var_map: Dict[str, Any]):
    """
    Supports your predicate DSL forms with value-only token mapping (v13 style).

    Note: We ignore Attr names in set/not_set because the variable is keyed by VALUE token.
    """
    m = _DSL_RE.match(expr.strip().rstrip("."))
    if not m:
        return None  # not DSL

    op = m.group(1).strip()
    args = _split_top_level_args(m.group(2))

    op_norm = op.lower()

    def V(val: str):
        tok = _norm_token(val)
        if tok not in var_map:
            raise KeyError(f"Unknown value token: {tok!r}")
        return var_map[tok]

    if op_norm == "set":
        # set(H,Attr,Val)
        if len(args) != 3:
            raise ValueError(f"set expects 3 args, got {args!r}")
        h = int(args[0])
        val = args[2]
        return V(val) == h

    if op_norm == "not_set":
        if len(args) != 3:
            raise ValueError(f"not_set expects 3 args, got {args!r}")
        h = int(args[0])
        val = args[2]
        return V(val) != h

    if op_norm in ("same_house", "not_same_house", "left_of", "right_of",
                   "immediately_left_of", "immediately_right_of", "adjacent"):
        if len(args) < 2:
            raise ValueError(f"{op} expects >=2 args, got {args!r}")

        _, va = _parse_attrval(args[0])
        _, vb = _parse_attrval(args[1])

        A = V(va)
        B = V(vb)

        if op_norm == "same_house":
            return A == B
        if op_norm == "not_same_house":
            return A != B
        if op_norm == "left_of":
            return A < B
        if op_norm == "right_of":
            return A > B
        if op_norm == "immediately_left_of":
            return A + 1 == B
        if op_norm == "immediately_right_of":
            return B + 1 == A
        if op_norm == "adjacent":
            return Abs(A - B) == 1

    if op_norm == "between":
        # between(A=va,B=vb,K) where K is #houses strictly between
        if len(args) != 3:
            raise ValueError(f"between expects 3 args, got {args!r}")
        _, va = _parse_attrval(args[0])
        _, vb = _parse_attrval(args[1])
        k = int(args[2])
        return Abs(V(va) - V(vb)) == (k + 1)

    # Unknown predicate => treat as non-DSL (let caller handle)
    return None


# ------------------- Entailment check -------------------

def _status_under_gamma(gamma: Solver, phi) -> str:
    """
    Returns:
      ENTAILED if Gamma ⊨ phi
      CONTRADICTION if Gamma ⊨ ¬phi
      NOT_ENTAILED otherwise
    """
    s1 = Solver()
    s1.append(gamma.assertions())
    s1.add(Not(phi))
    if s1.check() == unsat:
        return "ENTAILED"

    s2 = Solver()
    s2.append(gamma.assertions())
    s2.add(phi)
    if s2.check() == unsat:
        return "CONTRADICTION"

    return "NOT_ENTAILED"


# ------------------- Public API -------------------

def count_distinct_reasoning_steps_v13_relaxed(
    reasoning_lines,
    *,
    n_houses,
    attribute_values,
    timeout_s=2.0,
    allow_unassigned=False,
    # NEW:
    syntactic_clues=None,
    distinct_from_syntactic_clues: bool = False,
):
    """
    Distinct = (a) consistent with already-kept steps under `gamma`, AND
               (b) either NOT_ENTAILED, OR (ENTAILED but introduces new tokens),
               (c) optionally: NOT equivalent to any syntactic clue (if enabled).

    If distinct_from_syntactic_clues=True, you must pass syntactic_clues (list[str]).
    """
    base_solver, var_map, _ = build_relaxed_base_solver_v13_style(
        n_houses=int(n_houses),
        attribute_values=attribute_values,
        allow_unassigned=allow_unassigned,
        timeout_s=timeout_s,
    )

    gamma = Solver()
    gamma.set("timeout", int(float(timeout_s) * 1000))
    gamma.add(base_solver.assertions())

    # ---------------- NEW: precompute "clue constraints" for restatement filtering ----------------
    clue_sexprs = set()
    clue_parse_errors = []
    if distinct_from_syntactic_clues:
        for raw in syntactic_clues or []:
            try:
                _, cexpr = _extract_cid_and_expr(raw)
                cphi = _dsl_to_z3(cexpr, var_map)
                if cphi is None:
                    cphi = _parse_constraint(cexpr, var_map)
                clue_sexprs.add(cphi.sexpr())
            except Exception as e:
                clue_parse_errors.append(f"{raw} -> {type(e).__name__}: {e}")

    from typing import Set
    seen_tokens: Set[str] = set()
    seen_step_sexprs: Set[str] = set()  # helps de-dupe identical steps early

    n_steps_total = 0
    n_steps_parsed_ok = 0
    n_steps_distinct = 0

    kept_steps = []
    kept_reasoning_steps = []
    skipped_steps = []

    for line in reasoning_lines or []:
        parsed = _extract_syntactic_expr(line)
        if parsed is None:
            continue

        n_steps_total += 1
        k, expr = parsed

        try:
            phi = _dsl_to_z3(expr, var_map)
            if phi is None:
                phi = _parse_constraint(expr, var_map)
            n_steps_parsed_ok += 1
        except Exception as e:
            skipped_steps.append({
                "k": k, "raw": line, "expr": expr,
                "status": "PARSE_ERROR",
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        phi_sexpr = phi.sexpr()

        # NEW: skip if it restates a clue (exact structural match)
        if distinct_from_syntactic_clues and (phi_sexpr in clue_sexprs):
            skipped_steps.append({
                "k": k, "raw": line, "expr": expr,
                "status": "RESTATE_CLUE",
            })
            continue

        # NEW: skip exact duplicate step expression (cheap early de-dupe)
        if phi_sexpr in seen_step_sexprs:
            skipped_steps.append({
                "k": k, "raw": line, "expr": expr,
                "status": "DUPLICATE_STEP",
            })
            continue

        status = _status_under_gamma(gamma, phi)

        # "consistent with previous kept steps": contradiction => always skip
        if status == "CONTRADICTION":
            skipped_steps.append({"k": k, "raw": line, "expr": expr, "status": status})
            continue

        tokset = _vars_in(phi)
        introduces_new_tokens = len(tokset - seen_tokens) > 0

        # keep rule (your original intent), plus we only add if consistent (handled above)
        if status == "NOT_ENTAILED" or (status == "ENTAILED" and introduces_new_tokens):
            gamma.add(phi)
            seen_tokens |= tokset
            seen_step_sexprs.add(phi_sexpr)

            n_steps_distinct += 1
            kept_steps.append({"k": k, "raw": line, "expr": expr, "status": status})
            kept_reasoning_steps.append(line)
        else:
            skipped_steps.append({"k": k, "raw": line, "expr": expr, "status": status})

    out = {
        "n_steps_total": n_steps_total,
        "n_steps_parsed_ok": n_steps_parsed_ok,
        "n_steps_distinct": n_steps_distinct,
        "kept_steps": kept_steps,
        "kept_reasoning_steps": kept_reasoning_steps,
        "skipped_steps": skipped_steps,
        "base_mode": "relaxed_unassigned_0" if allow_unassigned else "strict_1_to_N",
    }

    # Optional debugging signal if clue parsing failed
    if distinct_from_syntactic_clues:
        out["clue_parse_errors_for_distinct_filter"] = clue_parse_errors

    return out


def count_distinct_reasoning_steps_v13_relaxed_v_old(
    reasoning_lines,
    *,
    n_houses,
    attribute_values,
    timeout_s=2.0,
    allow_unassigned=False,
):
    base_solver, var_map, _ = build_relaxed_base_solver_v13_style(
        n_houses=int(n_houses),
        attribute_values=attribute_values,
        allow_unassigned=allow_unassigned,
        timeout_s=timeout_s,
    )

    gamma = Solver()
    gamma.set("timeout", int(float(timeout_s) * 1000))
    gamma.add(base_solver.assertions())

    seen_tokens: Set[str] = set()

    n_steps_total = 0
    n_steps_parsed_ok = 0
    n_steps_distinct = 0

    kept_steps = []
    kept_reasoning_steps = []
    skipped_steps = []

    for line in reasoning_lines or []:
        parsed = _extract_syntactic_expr(line)
        if parsed is None:
            continue

        n_steps_total += 1
        k, expr = parsed

        try:
            phi = _dsl_to_z3(expr, var_map)
            if phi is None:
                phi = _parse_constraint(expr, var_map)
            n_steps_parsed_ok += 1
        except Exception as e:
            skipped_steps.append({
                "k": k, "raw": line, "expr": expr,
                "status": "PARSE_ERROR",
                "error": f"{type(e).__name__}: {e}",
            })
            continue

        status = _status_under_gamma(gamma, phi)

        # --- new: trace-novelty rule ---
        tokset = _vars_in(phi)
        introduces_new_tokens = len(tokset - seen_tokens) > 0

        if status == "NOT_ENTAILED" or (status == "ENTAILED" and introduces_new_tokens):
            # keep it as a "distinct trace step"
            #if status =="NOT_ENTAILED":
            gamma.add(phi)  # safe even if entailed
            seen_tokens |= tokset
            n_steps_distinct += 1
            kept_steps.append({"k": k, "raw": line, "expr": expr, "status": status})
            kept_reasoning_steps.append(line)
        else:
            skipped_steps.append({"k": k, "raw": line, "expr": expr, "status": status})

    return {
        "n_steps_total": n_steps_total,
        "n_steps_parsed_ok": n_steps_parsed_ok,
        "n_steps_distinct": n_steps_distinct,
        "kept_steps": kept_steps,
        "kept_reasoning_steps": kept_reasoning_steps,
        "skipped_steps": skipped_steps,
        "base_mode": "relaxed_unassigned_0" if allow_unassigned else "strict_1_to_N",
    }




# =================================================================================



def _normalize_month_token(value: str) -> str:
    """
    If `value` looks like a month name/abbreviation, normalize to Title Case month.
    Examples:
      "sept"  -> "September"
      "april" -> "April"
      "Sep."  -> "September"
    Otherwise returns original value unchanged.
    """
    if not isinstance(value, str):
        return value

    original = value
    s = value.strip()
    if not s:
        return original

    # strip trailing punctuation like "sept," or "sep."
    s_clean = re.sub(r"[,\.\!;\:\)\]]+$", "", s).strip()
    s_lower = s_clean.lower()

    # full month name
    if s_lower in _CANON_MONTHS:
        return _CANON_MONTHS[s_lower]

    # alias/abbreviation
    if s_lower in _MONTH_ALIASES:
        return _CANON_MONTHS[_MONTH_ALIASES[s_lower]]

    return original

def normalize_months_in_rows(z3_solution: dict) -> dict:
    """
    Takes only one input (z3_solution) and scans ALL cells in ALL columns.
    Any cell that matches a month name/abbreviation is normalized.
    """
    rows = z3_solution.get("rows", [])

    for row in rows:
        if not isinstance(row, list):
            continue
        for i, cell in enumerate(row):
            row[i] = _normalize_month_token(cell)

    return z3_solution

def _norm_token(s: str) -> str:
    """
    Normalize a single token/value:
    - strip surrounding quotes/brackets
    - strip trailing punctuation (.,;:!?)
    - collapse internal whitespace/hyphen/slash into underscores
    - collapse multiple underscores
    - strip leading/trailing underscores
    - lowercase (case-insensitive matching)
    """
    s = s.strip()
    s = s.rstrip(".")
    s = _PUNCT_RE.sub("", s)
    s = _TRAIL_PUNCT_RE.sub("", s)
    s = _SEP_RE.sub("_", s.strip())
    s = _MULTI_US_RE.sub("_", s)
    s = s.strip("_")
    s = s.lower()
    return s

def _norm_cell(s: str) -> str:
    """Normalize a solution/GT cell (same as token normalization)."""
    return _norm_token(s)


# ----------------------------- Parsing constraints -----------------------------

_CID_RE = re.compile(r"^\s*(C\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)

def _extract_cid_and_expr(line: str) -> Tuple[str, str]:
    raw = line.strip()
    raw = raw.strip()
    if raw.endswith("."):
        raw = raw[:-1]
    m = _CID_RE.match(raw)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return "C?", raw


_FUNC_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*(.*)\s*\)\s*$")

def _split_top_level_args(s: str) -> List[str]:
    args = []
    depth = 0
    cur = []
    for ch in s:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(cur).strip()
            if part:
                args.append(part)
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


from z3 import And as Z3And, Or as Z3Or, Not as Z3Not, Implies as Z3Implies  # add to imports if needed

_FUNC_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$")

def _parse_bool_expr(expr: str, var_map: Dict[str, Any]):
    e = expr.strip()

    # unwrap redundant parentheses
    while e.startswith("(") and e.endswith(")"):
        inner = e[1:-1].strip()
        if inner.count("(") == inner.count(")"):
            e = inner
        else:
            break

    m = _FUNC_RE.match(e)
    if m:
        fn = m.group(1).lower()
        args = _split_top_level_args(m.group(2))

        parsed_args = [_parse_bool_expr(a, var_map) for a in args]

        if fn == "and":
            return Z3And(*parsed_args)
        if fn == "or":
            return Z3Or(*parsed_args)
        if fn == "not":
            if len(parsed_args) != 1:
                raise ValueError("Not() expects exactly one argument")
            return Z3Not(parsed_args[0])
        if fn == "implies":
            if len(parsed_args) != 2:
                raise ValueError("Implies() expects exactly two arguments")
            return Z3Implies(parsed_args[0], parsed_args[1])
        if fn == "xor":
            if len(parsed_args) != 2:
                raise ValueError("Xor() expects exactly two arguments")
            return Z3Xor(parsed_args[0], parsed_args[1])
        if fn == "if":
            if len(parsed_args) != 3:
                raise ValueError("If() expects exactly three arguments")
            return Z3If(parsed_args[0], parsed_args[1], parsed_args[2])

        raise ValueError(f"Unsupported boolean operator: {fn}")

    # leaf → atomic constraint
    return _parse_atomic_constraint(e, var_map)


def _parse_atomic_constraint(expr: str, var_map: Dict[str, Any]) -> Any:
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1]

    # A + k == B   (k can be any integer, e.g., 2, -1)
    m = re.match(r"^(.+?)\s*\+\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        left_raw, k_raw, right_raw = m.group(1), m.group(2), m.group(3)
        Ltok = _norm_token(left_raw)
        Rtok = _norm_token(right_raw)
        if Ltok not in var_map:
            raise KeyError(f"Unknown left token: {Ltok!r}")
        if Rtok not in var_map:
            raise KeyError(f"Unknown right token: {Rtok!r}")
        L = var_map[Ltok]
        R = var_map[Rtok]
        return L + int(k_raw) == R

    for op in ("==", "!=", "<", ">"):
        parts = e.split(op)
        if len(parts) == 2:
            left_raw, right_raw = parts[0].strip(), parts[1].strip()
            Ltok = _norm_token(left_raw)
            if Ltok not in var_map:
                raise KeyError(f"Unknown left token: {Ltok!r}")
            L = var_map[Ltok]

            if right_raw.isdigit():
                R = int(right_raw)
            else:
                Rtok = _norm_token(right_raw)
                if Rtok not in var_map:
                    raise KeyError(f"Unknown right token: {Rtok!r}")
                R = var_map[Rtok]

            if op == "==": return L == R
            if op == "!=": return L != R
            if op == "<":  return L < R
            if op == ">":  return L > R

    raise ValueError(f"Unrecognized constraint syntax: {expr!r}")


def _parse_constraint(expr: str, var_map: Dict[str, Any]) -> Any:
    return _parse_bool_expr(expr, var_map)
# ----------------------------- Z3 Model building -----------------------------

@dataclass
class BuildResult:
    solver: Solver
    var_map: Dict[str, Any]
    attr_vars: Dict[str, List[Any]]
    clue_parse_errors: List[str]
    clue_skipped_oov: List[str]
    clue_skipped_under: List[str]
    clue_skipped_conflict: List[str]
    added_clues: List[str]  # raw lines that were added
    clue_id_to_text: Dict[str, str]  # cid -> raw line
    trackers: Dict[str, str]  # tracker_name -> raw line


def _build_base_solver(n: int, attribute_values: Dict[str, List[str]], timeout_s: float) -> Tuple[Solver, Dict[str, Any], Dict[str, List[Any]]]:
    var_map: Dict[str, Any] = {}
    attr_vars: Dict[str, List[Any]] = {}

    for attr, values in attribute_values.items():
        vars_: List[Any] = []
        for v in values:
            tok = _norm_token(v)
            if tok in var_map:
                raise ValueError(f"Duplicate token across attributes after normalization: {tok!r}")
            iv = Int(tok)
            var_map[tok] = iv
            vars_.append(iv)
        attr_vars[attr] = vars_

    s = Solver()
    s.set("timeout", int(timeout_s * 1000))
    for vars_ in attr_vars.values():
        s.add(Distinct(vars_))
        for v in vars_:
            s.add(And(v >= 1, v <= n))
    return s, var_map, attr_vars


def _add_clues(
    *,
    base_solver: Solver,
    n: int,
    var_map: Dict[str, Any],
    syntactic_clues: List[str],
    timeout_s: float,
    conflict_tolerant: bool,
) -> BuildResult:
    solver = Solver()
    solver.set("timeout", int(timeout_s * 1000))
    solver.add(base_solver.assertions())

    clue_parse_errors: List[str] = []
    clue_skipped_oov: List[str] = []
    clue_skipped_under: List[str] = []
    clue_skipped_conflict: List[str] = []
    added_clues: List[str] = []
    clue_id_to_text: Dict[str, str] = {}
    trackers: Dict[str, str] = {}

    for raw_line in syntactic_clues or []:
        cid, expr = _extract_cid_and_expr(raw_line)
        clue_id_to_text[cid] = raw_line

        # Underconstrained: must contain an operator we support
        if not any(op in expr for op in ("==", "!=", "<", ">", "+")):
            clue_skipped_under.append(raw_line)
            continue

        try:
            z3c = _parse_constraint(expr, var_map)
        except KeyError as e:
            clue_skipped_oov.append(f"{raw_line} -> {str(e)}")
            continue
        except Exception as e:
            clue_parse_errors.append(f"{raw_line} -> {type(e).__name__}: {e}")
            continue

        if conflict_tolerant:
            tmp = Solver()
            tmp.set("timeout", int(timeout_s * 1000))
            tmp.add(solver.assertions())
            tmp.add(z3c)
            if tmp.check() != sat:
                clue_skipped_conflict.append(raw_line)
                continue

        # Add as HARD constraint + also track for UNSAT cores
        solver.add(z3c)
        tr = Bool(f"clue_{cid}")
        solver.assert_and_track(z3c, tr)
        trackers[str(tr)] = raw_line
        added_clues.append(raw_line)

    return BuildResult(
        solver=solver,
        var_map=var_map,
        attr_vars={},  # filled by caller if needed
        clue_parse_errors=clue_parse_errors,
        clue_skipped_oov=clue_skipped_oov,
        clue_skipped_under=clue_skipped_under,
        clue_skipped_conflict=clue_skipped_conflict,
        added_clues=added_clues,
        clue_id_to_text=clue_id_to_text,
        trackers=trackers,
    )


def _model_to_solution_table(model, n: int, attribute_values: Dict[str, List[str]], var_map: Dict[str, Any]) -> Dict[str, Any]:
    token_to_attr: Dict[str, str] = {}
    for attr, vals in attribute_values.items():
        for v in vals:
            token_to_attr[_norm_token(v)] = attr

    house_to_attr_val: Dict[int, Dict[str, str]] = {h: {} for h in range(1, n + 1)}
    for tok, zv in var_map.items():
        h = int(model.eval(zv, model_completion=True).as_long())
        attr = token_to_attr.get(tok)
        if attr is not None:
            house_to_attr_val[h][attr] = tok

    header = ["House"] + list(attribute_values.keys())
    rows: List[List[str]] = []
    for h in range(1, n + 1):
        row = [str(h)]
        for attr in attribute_values.keys():
            row.append(house_to_attr_val[h].get(attr, ""))
        rows.append(row)
    return {"header": header, "rows": rows}



def normalize_header(data_sample):
    """
    Replaces sport-related header names with 'Sport' in:
      data_sample["ground_truth"]["header"]
    """
    header = data_sample.get("header", [])
    sports_aliases = {"FavoriteSports", "Sports", "FavoriteSport"}

    data_sample["header"] = ["Sport" if h in sports_aliases else h for h in header]
    return data_sample

def validate_solution_against_ground_truth(
    z3_solution: Optional[Dict[str, Any]],
    ground_truth: Optional[Dict[str, Any]],
) -> Tuple[bool, Dict[str, Any]]:
    if not z3_solution or not ground_truth:
        return False, {"error": "Missing z3_solution or ground_truth"}

    z_header = z3_solution.get("header", [])
    g_header = ground_truth.get("header", [])
    header_match = [ _norm_cell(x) for x in z_header ] == [ _norm_cell(x) for x in g_header ]

    z_rows = z3_solution.get("rows", [])
    g_rows = ground_truth.get("rows", [])

    mismatches: List[Dict[str, Any]] = []
    # Build z3 lookup
    z_by_house: Dict[int, List[str]] = {}
    for r in z_rows:
        if not r:
            continue
        z_by_house[int(r[0])] = r

    for gr in g_rows:
        if not gr:
            continue
        h = int(gr[0])
        zr = z_by_house.get(h)
        if zr is None:
            mismatches.append({"house": h, "col": "House", "gt": gr[0], "z3": None})
            continue
        # compare cell-by-cell using normalized comparison
        for j in range(1, min(len(gr), len(zr))):
            col = g_header[j] if j < len(g_header) else f"col_{j}"
            gt_raw = gr[j]
            z_raw = zr[j]
            gt_norm = _norm_cell(gt_raw)
            z_norm = _norm_cell(z_raw)
            if gt_norm != z_norm:
                mismatches.append({
                    "house": h,
                    "col": col,
                    "gt": gt_raw,
                    "z3": z_raw,
                    "gt_norm": gt_norm,
                    "z3_norm": z_norm,
                })

    ok = header_match and len(mismatches) == 0
    return ok, {"header_match": header_match, "cell_mismatches": mismatches, "ok": ok}


# ----------------------------- Reasoning validation -----------------------------

_STEP_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.*?)\s*(\[(.*?)\])?\s*$", re.IGNORECASE)

def _parse_step_line(line: str) -> Tuple[bool, Optional[str], Optional[str], List[str], Optional[str]]:
    """
    Returns (parsed_ok, step_id, expr, evidence_list, parse_error)
    Only parses syntactic lines that start with 'S'.
    """
    m = _STEP_RE.match(line.strip())
    if not m:
        return False, None, None, [], "Not a syntactic step line"
    sid = f"S{m.group(1)}"
    expr = (m.group(2) or "").strip()
    if expr.endswith("."):
        expr = expr[:-1].strip()
    evidence_raw = (m.group(4) or "").strip()
    ev: List[str] = []
    if evidence_raw:
        ev = [x.strip() for x in evidence_raw.split("+") if x.strip()]
        ev = [e.upper() for e in ev]
    return True, sid, expr, ev, None


def _entailment_status(solver: Solver, constraint) -> str:
    """
    Returns one of: ENTAILED, NOT_ENTAILED, CONTRADICTION
    Γ ⊨ φ  iff Γ ∧ ¬φ is UNSAT  -> ENTAILED
    Γ ⊨ ¬φ iff Γ ∧ φ is UNSAT   -> CONTRADICTION
    else NOT_ENTAILED
    """
    t1 = Solver()
    t1.add(solver.assertions())
    t1.add(Not(constraint))
    if t1.check() == unsat:
        return "ENTAILED"

    t2 = Solver()
    t2.add(solver.assertions())
    t2.add(constraint)
    if t2.check() == unsat:
        return "CONTRADICTION"

    return "NOT_ENTAILED"


def validate_reasoning_steps_syntactic_only(
    *,
    reasoning: Optional[List[str]],
    base_solver_with_clues: Solver,
    var_map: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    """
    Validates ONLY syntactic S<k>: lines. Natural language lines are ignored.

    strict: entailment against base_solver_with_clues
    chain: entailment against (base_solver_with_clues + previously accepted step constraints, if consistent)
    """
    steps_dicts: List[Dict[str, Any]] = []
    n_steps_total = 0
    n_steps_parsed_ok = 0
    n_steps_entailed_strict = 0
    n_steps_entailed_chain = 0
    n_steps_contradiction_chain = 0

    chain_solver = Solver()
    chain_solver.set("timeout", int(timeout_s * 1000))
    chain_solver.add(base_solver_with_clues.assertions())

    for line in reasoning or []:
        _S_STEP_RE = re.compile(r"^\s*s(\d+)\s*:\s*", re.IGNORECASE)

        if not _S_STEP_RE.match(line.strip().lower()):
            continue  # ignore natural language
        n_steps_total += 1

        parsed_ok, sid, expr, ev, perr = _parse_step_line(line)
        step_info: Dict[str, Any] = {
            "index": n_steps_total,
            "raw": line,
            "parsed_ok": False,
            "parse_error": perr,
            "strict_status": "PARSE_ERROR",
            "chain_status": "PARSE_ERROR",
            "strict_valid": None,
            "chain_valid": None,
            "evidence": ev,
            "expr": expr if expr is not None else None,
        }
        if not parsed_ok or expr is None:
            steps_dicts.append(step_info)
            continue

        # parse expr -> z3 constraint
        try:
            z3c = _parse_constraint(expr, var_map)
        except Exception as e:
            step_info["parse_error"] = f"{type(e).__name__}: {e}"
            steps_dicts.append(step_info)
            continue

        step_info["parsed_ok"] = True
        step_info["parse_error"] = None
        n_steps_parsed_ok += 1

        strict_status = _entailment_status(base_solver_with_clues, z3c)
        chain_status = _entailment_status(chain_solver, z3c)

        step_info["strict_status"] = strict_status
        step_info["chain_status"] = chain_status
        step_info["strict_valid"] = (strict_status == "ENTAILED")
        step_info["chain_valid"] = (chain_status == "ENTAILED")

        if strict_status == "ENTAILED":
            n_steps_entailed_strict += 1
        if chain_status == "ENTAILED":
            n_steps_entailed_chain += 1
        if chain_status == "CONTRADICTION":
            n_steps_contradiction_chain += 1

        # Update chain solver: add step if it is parsed and not contradictory (treat as "assumption" in chain)
        if chain_status != "CONTRADICTION":
            chain_solver.add(z3c)

        steps_dicts.append(step_info)

    return {
        "steps": steps_dicts,
        "n_steps_total": n_steps_total,
        "n_steps_parsed_ok": n_steps_parsed_ok,
        "n_steps_entailed_strict": n_steps_entailed_strict,
        "n_steps_entailed_chain": n_steps_entailed_chain,
        "n_steps_contradiction_chain": n_steps_contradiction_chain,
    }


# ----------------------------- Top-level solve/validate -----------------------------

def solve_and_validate_payload(payload: Dict[str, Any], *, timeout_s: float = 2.0, conflict_tolerant_clues: bool = False) -> Dict[str, Any]:
    """
    Computes both FULL and RAW builds:
      FULL: add all parseable clues (no conflict-tolerant skipping unless parse error / oov / underconstrained).
      RAW:  add clues with conflict-tolerant skipping (always enabled for RAW, independent of passed flag).
    Reasoning is validated against FULL if FULL is SAT; else against RAW if RAW is SAT.
    """
    report: Dict[str, Any] = {}

    n = int(payload["n_houses"])
    attribute_values = payload["attribute_values"]
    syntactic_clues = payload.get("syntactic_clues") or []
    reasoning = payload.get("reasoning") or []
    ground_truth = payload.get("ground_truth")
    ground_truth = normalize_header(ground_truth)

    base_solver, var_map, attr_vars = _build_base_solver(n, attribute_values, timeout_s)
    # RAW base_sat_raw uses conflict-tolerant clues always (per user requirement)

    #report["n_steps_total_unfiltered"] = len(reasoning)
    try:

        #out_distinct_steps = count_distinct_reasoning_steps_v13_relaxed_v_old(reasoning, n_houses=n, attribute_values=attribute_values, timeout_s=2.0,)
        out_distinct_steps = count_distinct_reasoning_steps_v13_relaxed(
            reasoning,
            n_houses=n,
            attribute_values=attribute_values,
            timeout_s=2.0,
            allow_unassigned=False,  # or False, your choice
            syntactic_clues=syntactic_clues,
            distinct_from_syntactic_clues=False,
        )

        reasoning = out_distinct_steps["kept_reasoning_steps"]

        report["log_compute_distinct_steps"] = out_distinct_steps

        #print("selected steps = ", reasoning)
    except Exception as dis_Error:
        print("Error in computing distinct reasoning steps = {}".format(dis_Error))


    raw_build = _add_clues(
        base_solver=base_solver,
        n=n,
        var_map=var_map,
        syntactic_clues=syntactic_clues,
        timeout_s=timeout_s,
        conflict_tolerant=True,
    )
    raw_build.attr_vars = attr_vars

    # FULL build uses non-conflict-tolerant (unless user wants; keep parameter conflict_tolerant_clues for full as requested)
    full_build = _add_clues(
        base_solver=base_solver,
        n=n,
        var_map=var_map,
        syntactic_clues=syntactic_clues,
        timeout_s=timeout_s,
        conflict_tolerant=bool(conflict_tolerant_clues),
    )
    full_build.attr_vars = attr_vars

    # base SAT flags
    base_sat_raw = (raw_build.solver.check() == sat)
    base_sat_full = (full_build.solver.check() == sat)
    report["base_sat_raw"] = base_sat_raw
    report["base_sat_full"] = base_sat_full

    # Unsat debug + cores (for full only, since raw is conflict-tolerant)
    base_unsat_debug: Dict[str, Any] = {}
    if not base_sat_full:
        try:
            core = full_build.solver.unsat_core()
            core_ids = []
            core_text = []
            for c in core:
                key = str(c)
                cid = key.replace("clue_", "")
                core_ids.append(cid)
                core_text.append(full_build.trackers.get(key, ""))
            base_unsat_debug["full_unsat_core_ids"] = core_ids
            base_unsat_debug["full_unsat_core_text"] = core_text
        except Exception:
            pass
    report["base_unsat_debug"] = base_unsat_debug

    # Metrics from v12 re clues (use FULL's lists; RAW is available but not requested separately)
    report["clue_parse_errors"] = full_build.clue_parse_errors
    report["clue_skipped_oov"] = full_build.clue_skipped_oov
    report["clue_skipped_underconstrained"] = full_build.clue_skipped_under
    report["clue_skipped_conflict"] = full_build.clue_skipped_conflict

    # Solve and compare against GT for FULL and RAW
    z3_solution_full = None
    gt_valid_full = False
    gt_details_full: Dict[str, Any] = {}
    if base_sat_full:
        m = full_build.solver.model()
        z3_solution_full = _model_to_solution_table(m, n, attribute_values, var_map)
        z3_solution_full = normalize_header(z3_solution_full)
        gt_valid_full, gt_details_full = validate_solution_against_ground_truth(z3_solution_full, ground_truth)

    z3_solution_raw = None
    gt_valid_raw = False
    gt_details_raw: Dict[str, Any] = {}
    if base_sat_raw:
        m = raw_build.solver.model()
        z3_solution_raw = _model_to_solution_table(m, n, attribute_values, var_map)
        z3_solution_raw = normalize_header(z3_solution_raw)
        gt_valid_raw, gt_details_raw = validate_solution_against_ground_truth(z3_solution_raw, ground_truth)

    report["z3_solution"] = z3_solution_full if z3_solution_full is not None else z3_solution_raw
    # v12: gt_solution_valid/gt_solution_details refer to chosen solution (prefer full)
    report["gt_solution_valid"] = bool(gt_valid_full if base_sat_full else gt_valid_raw)
    report["gt_solution_details"] = gt_details_full if base_sat_full else gt_details_raw

    report["base_sat_full_GT"] = bool(base_sat_full and gt_valid_full)
    report["base_sat_raw_GT"] = bool(base_sat_raw and gt_valid_raw)

    # Reasoning validation: ONLY syntactic lines. Use FULL solver if SAT else RAW if SAT.
    reasoning_solver = None
    if base_sat_full:
        reasoning_solver = full_build.solver
    elif base_sat_raw:
        reasoning_solver = raw_build.solver

    steps_dicts: List[Dict[str, Any]] = []
    n_steps_total = 0
    n_steps_parsed_ok = 0
    n_steps_entailed_strict = 0
    n_steps_entailed_chain = 0
    n_steps_contradiction_chain = 0

    if reasoning_solver is not None:
        rv = validate_reasoning_steps_syntactic_only(
            reasoning=reasoning,
            base_solver_with_clues=reasoning_solver,
            var_map=var_map,
            timeout_s=timeout_s,
        )
        steps_dicts = rv["steps"]
        n_steps_total = rv["n_steps_total"]
        n_steps_parsed_ok = rv["n_steps_parsed_ok"]
        n_steps_entailed_strict = rv["n_steps_entailed_strict"]
        n_steps_entailed_chain = rv["n_steps_entailed_chain"]
        n_steps_contradiction_chain = rv["n_steps_contradiction_chain"]

    # v12-style step metrics
    report["steps"] = steps_dicts
    report["n_steps_total"] = n_steps_total
    report["n_steps_parsed_ok"] = n_steps_parsed_ok
    report["n_steps_entailed_strict"] = n_steps_entailed_strict
    report["n_steps_entailed_chain"] = n_steps_entailed_chain
    report["n_steps_contradiction_chain"] = n_steps_contradiction_chain

    # gt_valid/gt_factor naming from v12
    gt_valid = report["gt_solution_valid"]
    _gt_factor = 1.0 if gt_valid else 0.0
    report["gt_valid"] = gt_valid
    report["gt_factor"] = _gt_factor

    # Pass rates validated by GT
    pass_rate_strict = (n_steps_entailed_strict / n_steps_total) if n_steps_total > 0 else 0.0
    pass_rate_chain = (n_steps_entailed_chain / n_steps_total) if n_steps_total > 0 else 0.0
    report["pass_rate_strict_gt_validated"] = pass_rate_strict * _gt_factor
    report["pass_rate_chain_gt_validated"] = pass_rate_chain * _gt_factor

    # n_steps_entailed_strict already present; include as requested explicitly (v12 name)
    report["n_steps_entailed_strict"] = n_steps_entailed_strict
    report["n_steps_contradiction_chain"] = n_steps_contradiction_chain

    # Failure reason (lightweight)
    if report["base_sat_full_GT"]:
        failure_reason = "NONE"
    elif not base_sat_full:
        failure_reason = "CLUES_UNSAT"
    elif base_sat_full and not gt_valid_full:
        failure_reason = "GT_MISMATCH"
    else:
        failure_reason = "UNKNOWN"
    report["failure_reason"] = failure_reason

    return report


# ----------------------------- Self-tests -----------------------------

def run_self_tests() -> int:
    # Minimal sanity tests; return nonzero on failure

    payload = {
        "n_houses": 6,
        "attribute_values": {
            "Name": ["Arnold", "Peter", "Bob", "Eric", "Carol", "Alice"],
            "Animal": ["horse", "rabbit", "fish", "cat", "bird", "dog"],
            "Occupation": ["engineer", "nurse", "lawyer", "teacher", "artist", "doctor"],
            "Sport": ["basketball", "volleyball", "soccer", "tennis", "baseball", "swimming"],
            "Height": ["average", "tall", "short", "very_short", "very_tall", "super_tall"]
        },
        "syntactic_clues": [
            "C1: engineer == dog.",
            "C2: average < short.",
            "C3: average + 1 == rabbit.",
            "C4: tall < very_short.",
            "C5: Arnold == cat.",
            "C6: horse == teacher.",
            "C7: Carol == soccer.",
            "C8: tall == volleyball.",
            "C9: lawyer == 5.",
            "C10: tennis == teacher.",
            "C11: average == swimming.",
            "C12: baseball + 1 == engineer.",
            "C13: Peter == nurse.",
            "C14: Bob > artist.",
            "C15: teacher + 1 == soccer.",
            "C16: rabbit == Alice.",
            "C17: fish == Carol.",
            "C18: baseball == 1.",
            "C19: cat > very_short.",
            "C20: super_tall == 5."
        ],
        "reasoning": [
            "We start from the strongest absolute placement: the baseball fan is fixed in the first house.",
            "S1: baseball == 1. [C18]",

            "The baseball fan is immediately left of the engineer, so the engineer must be in house 2.",
            "S2: engineer == 2. [C12+S1]",

            "The engineer owns the dog, placing the dog in house 2.",
            "S3: dog == 2. [C1+S2]",

            "Another absolute clue places the lawyer in house 5.",
            "S4: lawyer == 5. [C9]",

            "The super tall person is also fixed in house 5.",
            "S5: super_tall == 5. [C20]",

            "From the teacher–soccer adjacency, the teacher must be immediately left of the soccer fan and therefore cannot be in house 6.",
            "S6: Or(teacher == 1, teacher == 2, teacher == 3, teacher == 4, teacher == 5). [C15]",

            "Because the lawyer is already in house 5 and occupations are unique, the teacher cannot be in house 5.",
            "S7: teacher != 5. [S4+Distinct(Occupation)]",

            "Because the engineer is already in house 2, the teacher cannot also be in house 2.",
            "S8: teacher != 2. [S2+Distinct(Occupation)]",

            "Combining these eliminations, the teacher must be in house 1, 3, or 4.",
            "S9: Or(teacher == 1, teacher == 3, teacher == 4). [S6+S7+S8]",

            "Using adjacency, we can express the possible consequences of each remaining teacher placement.",
            "S10: Implies(teacher == 1, soccer == 2). [C15]",
            "S11: Implies(teacher == 3, soccer == 4). [C15]",
            "S12: Implies(teacher == 4, soccer == 5). [C15]",

            "Carol is the soccer fan and also the fish owner, so fish and soccer must be in the same house.",
            "S13: Carol == soccer. [C7]",
            "S14: fish == Carol. [C17]",
            "S15: fish == soccer. [S13+S14]",

            "House 2 already contains the dog, so fish cannot be in house 2. This eliminates the case teacher == 1.",
            "S16: soccer != 2. [S3+S15+Distinct(Animal)]",
            "S17: teacher != 1. [S10+S16]",

            "Thus the teacher must be in either house 3 or house 4.",
            "S18: Or(teacher == 3, teacher == 4). [S9+S17]",

            "We now perform an explicit case split on the remaining possibilities.",
            "S19: Or(teacher == 3, teacher == 4). [CaseSplit]",

            "Case A: assume the teacher is in house 3. Then soccer is in house 4, and Carol and fish are also in house 4.",
            "S20: Implies(teacher == 3, soccer == 4). [C15]",
            "S21: Implies(teacher == 3, Carol == 4). [S13+S20]",
            "S22: Implies(teacher == 3, fish == 4). [S15+S20]",

            "Under this assumption, horse and tennis are also forced into house 3, which leads to a contradiction once the rabbit–average chain is propagated. Therefore this case is invalid.",
            "S23: teacher != 3. [CaseA contradiction]",

            "Only Case B remains: the teacher must be in house 4, forcing soccer into house 5.",
            "S24: teacher == 4. [S18+S23]",
            "S25: soccer == 5. [C15+S24]",

            "Carol is therefore in house 5, and the fish is also in house 5.",
            "S26: Carol == 5. [C7+S25]",
            "S27: fish == 5. [C17+S26]",

            "Since the teacher is in house 4, the horse and tennis must also be in house 4.",
            "S28: horse == 4. [C6+S24]",
            "S29: tennis == 4. [C10+S24]",

            "The rabbit is Alice and must be immediately right of the average-height person. Given existing animal placements, the rabbit can only be in house 3.",
            "S30: rabbit == 3. [C3+Distinct(Animal)]",
            "S31: Alice == 3. [C16+S30]",

            "Thus the average height and swimming are in house 2.",
            "S32: average == 2. [C3+S30]",
            "S33: swimming == 2. [C11+S32]",

            "Tall corresponds to volleyball and must be left of very short. The only valid placement is tall and volleyball in house 3.",
            "S34: volleyball == 3. [C8+Distinct(Sport)]",
            "S35: tall == 3. [C8+S34]",

            "The remaining sports force basketball into house 6.",
            "S36: basketball == 6. [Distinct(Sport)]",

            "Peter is the nurse, and since house 2 is engineer and house 4 is teacher, Peter must be in house 1.",
            "S37: Peter == 1. [C13+Distinct(Occupation)]",

            "Arnold is the cat lover, and the cat must be to the right of the very short person. This forces Arnold into house 6.",
            "S38: Arnold == 6. [C5+C19]",

            "The remaining name assignments give Eric in house 2 and Bob in house 4.",
            "S39: Eric == 2. [Distinct(Name)]",
            "S40: Bob == 4. [Distinct(Name)]",

            "The remaining animal bird must be in house 1.",
            "S41: bird == 1. [Distinct(Animal)]",

            "Completing heights using ordering constraints gives very short in house 4, short in house 6, and very tall in house 1.",
            "S42: very_short == 4. [C4+S35]",
            "S43: short == 6. [C2+S32]",
            "S44: very_tall == 1. [Distinct(Height)]",

            "All attributes are now consistently assigned, yielding the final solution."
        ],
        "ground_truth": {
            "header": ["House", "Name", "Animal", "Occupation", "Sport", "Height"],
            "rows": [
                ["1", "Peter", "bird", "nurse", "baseball", "very_tall"],
                ["2", "Eric", "dog", "engineer", "swimming", "average"],
                ["3", "Alice", "rabbit", "artist", "volleyball", "tall"],
                ["4", "Bob", "horse", "teacher", "tennis", "very_short"],
                ["5", "Carol", "fish", "lawyer", "soccer", "super_tall"],
                ["6", "Arnold", "cat", "doctor", "basketball", "short"]
            ]
        }
    }
    r1 = solve_and_validate_payload(payload, timeout_s=2.0)

    print(json.dumps(r1, indent=2))
    print(5*"\n")



    return 0


def main() -> None:
    run_self_tests()
    #payload = _load_json(args.input_json)

    #print(json.dumps(report, ensure_ascii=False, indent=2))

    '''
    attribute_values = {
        "Name": ["Eric", "Arnold"],
        "Car": ["ford_f150", "tesla_model_3"],
    }
    reasoning = [
        "S1: Eric > tesla_model_3. [C1]",
        "S2: ford_f150  + 5 == Eric. [C2]",
        "S3: tesla_model_3 == 2. [C3]",
    ]

    out = count_distinct_reasoning_steps_v13_relaxed(
        reasoning,
        n_houses=2,
        attribute_values=attribute_values,
        timeout_s=2.0,
    )
    
    import json
    print(json.dumps(out, indent=2))
    '''

if __name__ == "__main__":
    main()