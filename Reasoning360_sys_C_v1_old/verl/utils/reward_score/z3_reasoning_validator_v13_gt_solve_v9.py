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

import z3  # type: ignore
import argparse
import json
import sys
import os
import re
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import And, Bool, Distinct, Int, Not, Or, Solver, sat, unsat  # type: ignore
from z3 import And as Z3And, Or as Z3Or, Not as Z3Not, Implies as Z3Implies, Xor as Z3Xor, If as Z3If


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)


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

    Special-case for implications (A => B):
      We treat an implication as "entailed" only if it is BOTH:
        1) Gamma ∧ A ∧ ¬B is UNSAT  (no counterexample where A holds but B fails)
        2) Gamma ∧ A ∧ B is SAT     (A is feasible and B can hold under A; avoids vacuous truth)
    """
    # --- Special handling for Implies(A, B) ---
    try:
        if hasattr(phi, "decl") and phi.decl().name() == "=>" and phi.num_args() == 2:
            A = phi.arg(0)
            B = phi.arg(1)

            # --- Non-vacuous CONTRADICTION check for implication ---
            s_full = Solver()
            s_full.append(gamma.assertions())
            s_full.add(phi)
            if s_full.check() == unsat:
                # For a genuine contradiction, A must be forced (no model with ¬A)
                s_notA = Solver()
                s_notA.append(gamma.assertions())
                s_notA.add(Not(A))
                if s_notA.check() != unsat:
                    return "NOT_ENTAILED"

                # And there must be a concrete witness of violation under A: A ∧ ¬B is SAT
                s_witness_bad = Solver()
                s_witness_bad.append(gamma.assertions())
                s_witness_bad.add(A)
                s_witness_bad.add(Not(B))
                if s_witness_bad.check() != sat:
                    return "NOT_ENTAILED"

                return "CONTRADICTION"

            # --- Non-vacuous ENTAILED check for implication ---
            s_counter = Solver()
            s_counter.append(gamma.assertions())
            s_counter.add(A)
            s_counter.add(Not(B))
            if s_counter.check() != unsat:
                # Counterexample exists => not entailed
                return "NOT_ENTAILED"

            s_witness = Solver()
            s_witness.append(gamma.assertions())
            s_witness.add(A)
            s_witness.add(B)
            if s_witness.check() != sat:
                # No witness where A and B can both hold => treat as not-entailed (vacuous / degenerate)
                return "NOT_ENTAILED"

            return "ENTAILED"
            return "ENTAILED"
    except Exception:
        # Fall back to generic entailment check below
        pass

    # --- Generic entailment check ---
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


# ---------------- helper: semantic clue-equivalence via XOR (S <=> C) ----------------
def _equiv_to_any_clue_via_xor(phi) -> bool:
    """
    Returns True if phi is logically equivalent to any clue cphi under *base axioms only*.
    Equivalence test: UNSAT( base ∧ Xor(phi, cphi) ).
    """
    if not distinct_from_syntactic_clues_semantic_xor or not clue_phis:
        return False

    for cphi in clue_phis:
        tt = Solver()
        tt.set("timeout", int(float(timeout_s) * 1000))
        tt.add(base_assertions)
        tt.add(Z3Xor(phi, cphi))
        if tt.check() == unsat:
            return True
    return False

# ---------------- helper: tautology under base axioms only ----------------
def _is_tautology_under_base(phi) -> bool:
    """
    Returns True if phi is entailed by *base axioms only* (domain/distinctness),
    i.e., UNSAT(base ∧ Not(phi)). Such steps add no information and should be omitted.
    Example: Or(x==1, x==2) in a 2-house domain.
    """
    if not omit_tautologies:
        return False
    tt = Solver()
    tt.set("timeout", int(float(timeout_s) * 1000))
    tt.add(base_assertions)
    tt.add(Z3Not(phi))
    return tt.check() == unsat

# ---------------- helpers for "token novelty" ----------------

    import re
    TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    def _tokens_in_constraint(expr: str):
        # best-effort tokens (filters out keywords/operators and pure numbers)
        toks = TOKEN_RE.findall(expr)
        bad = {"And", "Or", "Not", "Implies", "Abs", "Distinct", "sat", "unsat"}
        out = []
        for t in toks:
            if t in bad:
                continue
            if t.isdigit():
                continue
            out.append(t)
        return set(out)

    seen_tokens = set()
    seen_step_sexprs = set()

    n_steps_total = 0
    n_steps_parsed_ok = 0
    n_steps_distinct = 0
    n_steps_entailed = 0

    kept_steps = []
    kept_reasoning_steps = []
    skipped_steps = []

    for line in (reasoning_lines or []):
        parsed = _extract_syntactic_expr(line)  # expects "S<i>: ...."
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

        # 1) exact duplicate step
        if phi_sexpr in seen_step_sexprs:
            skipped_steps.append({
                "k": k, "raw": line, "expr": expr,
                "status": "DUPLICATE_STEP",
                "error": "exact duplicate sexpr",
            })
            continue

        # 2) optionally forbid restating clues
        if distinct_from_syntactic_clues:
            # Fast check: exact normalized match
            if phi_sexpr in clue_sexprs:
                skipped_steps.append({
                    "k": k, "raw": line, "expr": expr,
                    "status": "RESTATES_CLUE",
                    "error": "exact match to a clue constraint (sexpr)",
                })
                continue

            # Stronger check (optional): semantic equivalence via UNSAT(Xor)
            if _equiv_to_any_clue_via_xor(phi):
                skipped_steps.append({
                    "k": k, "raw": line, "expr": expr,
                    "status": "RESTATES_CLUE",
                    "error": "semantically equivalent to a clue constraint (UNSAT(Xor))",
                })
                continue

        # 2b) omit tautologies under base axioms (adds no information)
        if _is_tautology_under_base(phi):
            skipped_steps.append({
                "k": k, "raw": line, "expr": expr,
                "status": "TAUTOLOGY",
                "error": "entailed by base axioms only (UNSAT(base ∧ Not(phi)))",
            })
            continue

        status = _status_under_gamma(chain_gamma, phi)  # ENTAILED / CONTRADICTION / NOT_ENTAILED

        # CONTRADICTION steps are always skipped and MUST NOT become precursors.
        if status == "CONTRADICTION":
            skipped_steps.append({
                "k": k, "raw": line, "expr": expr,
                "status": "CONTRADICTION",
                "error": "ChainGamma entails Not(phi) (non-vacuous implication handling applies if phi is Implies)",
            })
            continue

        # If entailed_only, reject non-entailed steps (and do NOT add them as precursors).
        if entailed_only and status != "ENTAILED":
            skipped_steps.append({
                "k": k, "raw": line, "expr": expr,
                "status": "NOT_ENTAILED",
                "error": "not entailed by (base + clues + prior kept steps)",
            })
            continue

        # Optional novelty heuristic (prevents “entailed spam”)
        if require_token_novelty:
            toks = _tokens_in_constraint(expr)
            new_toks = toks - seen_tokens
            if len(new_toks) == 0:
                skipped_steps.append({
                    "k": k, "raw": line, "expr": expr,
                    "status": "NON_NOVEL",
                    "error": "no new tokens vs already-kept steps",
                })
                continue
            seen_tokens |= toks

        # Keep it
        seen_step_sexprs.add(phi_sexpr)
        if status == "ENTAILED":
            n_steps_entailed += 1
        n_steps_distinct += 1

        kept_steps.append({
            "k": k,
            "raw": line,
            "expr": expr,
            "status": status,
        })
        kept_reasoning_steps.append(line)

        # Update ChainGamma: add this kept step as a precursor for later steps.
        chain_gamma.add(phi)

    return {
        "n_steps_total": n_steps_total,
        "n_steps_parsed_ok": n_steps_parsed_ok,
        "n_steps_distinct": n_steps_distinct,
        "n_steps_entailed": n_steps_entailed,
        "clue_constraints_added": clue_constraints_added,
        "clue_parse_errors": clue_parse_errors,
        "kept_steps": kept_steps,
        "kept_reasoning_steps": kept_reasoning_steps,
        "skipped_steps": skipped_steps,
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



def _parse_term_literal(raw: str, var_map: Dict[str, Any], *, side: str) -> Any:
    """Parse either an integer literal (e.g., 2, -1) or a known symbol in var_map.

    This is intentionally permissive so we can accept expressions like:
      - Arnold == 2
      - 2 == Arnold
      - Arnold < 3
      - 3 < Arnold
    """
    r = raw.strip()
    if re.fullmatch(r"-?\d+", r):
        return int(r)
    tok = _norm_token(r)
    if tok not in var_map:
        raise KeyError(f"Unknown {side} token: {tok!r}")
    return var_map[tok]


def _parse_atomic_constraint(expr: str, var_map: Dict[str, Any]) -> Any:
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1]

    # A + k == B   (k can be any integer, e.g., 2, -1)
    m = re.match(r"^(.+?)\s*\+\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        left_raw, k_raw, right_raw = m.group(1), m.group(2), m.group(3)
        L = _parse_term_literal(left_raw, var_map, side="left")
        R = _parse_term_literal(right_raw, var_map, side="right")
        return L + int(k_raw) == R

    for op in ("==", "!=", "<", ">"):
        parts = e.split(op)
        if len(parts) == 2:
            left_raw, right_raw = parts[0].strip(), parts[1].strip()
            L = _parse_term_literal(left_raw, var_map, side="left")

            R = _parse_term_literal(right_raw, var_map, side="right")

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

    Generic rule:
      Γ ⊨ φ  iff Γ ∧ ¬φ is UNSAT  -> ENTAILED
      Γ ⊨ ¬φ iff Γ ∧ φ is UNSAT   -> CONTRADICTION
      else NOT_ENTAILED

    Special-case for implications (A => B):
      We treat an implication as "entailed" only if it is BOTH:
        1) Γ ∧ A ∧ ¬B is UNSAT  (no counterexample where A holds but B fails)
        2) Γ ∧ A ∧ B is SAT     (A is feasible and B can hold under A; avoids vacuous truth)
    """
    # --- Special handling for Implies(A, B) ---
    try:
        if hasattr(constraint, "decl") and constraint.decl().name() == "=>" and constraint.num_args() == 2:
            A = constraint.arg(0)
            B = constraint.arg(1)

            # --- Non-vacuous CONTRADICTION check for implication ---
            t_full = Solver()
            t_full.add(solver.assertions())
            t_full.add(constraint)
            if t_full.check() == unsat:
                # For a genuine contradiction, A must be forced (no model with ¬A)
                t_notA = Solver()
                t_notA.add(solver.assertions())
                t_notA.add(Not(A))
                if t_notA.check() != unsat:
                    return "NOT_ENTAILED"

                # And there must be a concrete witness of violation under A: A ∧ ¬B is SAT
                t_witness_bad = Solver()
                t_witness_bad.add(solver.assertions())
                t_witness_bad.add(A)
                t_witness_bad.add(Not(B))
                if t_witness_bad.check() != sat:
                    return "NOT_ENTAILED"

                return "CONTRADICTION"

            # --- Non-vacuous ENTAILED check for implication ---
            t_counter = Solver()
            t_counter.add(solver.assertions())
            t_counter.add(A)
            t_counter.add(Not(B))
            if t_counter.check() != unsat:
                return "NOT_ENTAILED"

            t_witness = Solver()
            t_witness.add(solver.assertions())
            t_witness.add(A)
            t_witness.add(B)
            if t_witness.check() != sat:
                return "NOT_ENTAILED"

            return "ENTAILED"
            return "ENTAILED"
    except Exception:
        # Fall back to generic checks below
        pass

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


    # Check Validity of the base solver
    try:
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
        base_sat_full = (full_build.solver.check() == sat)
        #print("base sat full = {}".format(base_sat_full))
        # Solve and compare against GT for FULL and RAW
        z3_solution_full = None
        gt_valid_full = False
        gt_details_full: Dict[str, Any] = {}
        if base_sat_full:
            m = full_build.solver.model()
            z3_solution_full = _model_to_solution_table(m, n, attribute_values, var_map)
            z3_solution_full = normalize_header(z3_solution_full)
            gt_valid_full, gt_details_full = validate_solution_against_ground_truth(z3_solution_full, ground_truth)


        report["z3_solution"] = z3_solution_full if z3_solution_full is not None else {}
        report["gt_solution_details"] = gt_details_full if base_sat_full else []
        report["base_sat_full_GT"] = bool(base_sat_full and gt_valid_full)
        report["parse_status"] = "SAT_CHECK_SUCCESS"

    except Exception as e:
        logger.error(f"Error in validity of the base solver: {e}")
        report["z3_solution"] = {}
        report["gt_solution_details"] = {}
        report["base_sat_full_GT"] = 0.0
        report["parse_status"] = "SAT_CHECK_FAIL"

    if report["base_sat_full_GT"]:
        try:
            out_distinct_steps = count_distinct_reasoning_steps_v13_relaxed(
                reasoning_lines=reasoning,
                n_houses=n,
                attribute_values=attribute_values,
                syntactic_clues=syntactic_clues,
                distinct_from_syntactic_clues_semantic_xor=True,  # enable semantic (UNSAT(Xor)) clue equivalence too
                entailed_only=True,
                require_token_novelty=False,
                omit_tautologies=True,)

            report["n_steps_total"] =  out_distinct_steps["n_steps_total"]
            report["n_steps_parsed_ok"] = out_distinct_steps["n_steps_parsed_ok"]
            report["n_steps_valid"] = out_distinct_steps["n_steps_valid"]
            report["n_steps_novel_inc_clues"] = out_distinct_steps["n_steps_novel_inc_clues"]
            report["list_all_steps"] = out_distinct_steps["list_all_steps"]
            report["list_steps_non_valid"] = out_distinct_steps["list_steps_non_valid"]
            report["list_novel_steps_inc_clues"] = out_distinct_steps["list_novel_steps_inc_clues"]
            report["n_non_valid_contradiction"] = out_distinct_steps["n_non_valid_contradiction"]
            report["parse_status"] = "NOVELTY_CHECK_SUCCESS"

            #print("selected steps = ", reasoning)
        except Exception as dis_Error:
            logger.error("Error in computing distinct reasoning steps = {}".format(dis_Error))

            #report["base_sat_full_GT"] = 0.0
            report["n_steps_total"] =  0
            report["n_steps_parsed_ok"] = 0
            report["n_steps_valid"] = 0
            report["n_steps_novel_inc_clues"] = 0
            report["list_steps_non_valid"] = []
            report["list_novel_steps_inc_clues"] = []
            report["n_non_valid_contradiction"] = 0
            report["parse_status"] = "NOVELTY_CHECK_FAIL"

    return report


# ----------------------------- Self-tests -----------------------------


# ============================================================
# PATCH v13_8: Fix count_distinct_reasoning_steps_v13_relaxed
# - Always distinct from syntactic_clues (sexpr + optional semantic XOR under base axioms)
# - Do NOT use CONTRADICTION steps as precursors (not added to chain_gamma)
# - Omit tautologies under base axioms: UNSAT(base ∧ Not(phi))
# - Ensure function returns a dict (was previously truncated)
# ============================================================

_SID_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)

def _extract_sid_and_expr(line: str) -> Tuple[int, str]:
    raw = line.strip()
    if raw.endswith("."):
        raw = raw[:-1]
    m = _SID_RE.match(raw)
    if not m:
        return -1, raw
    return int(m.group(1)), m.group(2).strip()

def _sanitize_var_name(tok: str) -> str:
    t = tok.strip().lower()
    t = re.sub(r"[^a-z0-9_]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "v"
    return t

def _build_var_map_from_attribute_values(attribute_values: Dict[str, List[str]]) -> Dict[str, Any]:
    var_map: Dict[str, Any] = {}
    used_names: Set[str] = set()
    for _, vals in attribute_values.items():
        for v in vals:
            base = _sanitize_var_name(v)
            name = base
            k = 2
            while name in used_names:
                name = f"{base}_{k}"
                k += 1
            used_names.add(name)
            z = Int(name)
            # map a few common normalizations to the same z3 var
            keys = set()
            keys.add(v)
            keys.add(v.strip())
            keys.add(v.strip().lower())
            keys.add(_sanitize_var_name(v))
            for key in keys:
                var_map[key] = z
    return var_map

def _build_base_axioms(n_houses: int, attribute_values: Dict[str, List[str]], var_map: Dict[str, Any]) -> List[BoolRef]:
    axioms: List[BoolRef] = []
    # ranges
    for key, z in var_map.items():
        # var_map contains repeated references; only constrain unique z3 vars
        pass
    uniq_vars = list({id(v): v for v in var_map.values()}.values())
    for v in uniq_vars:
        axioms.append(And(v >= 1, v <= n_houses))

    # Distinct per attribute
    for _, vals in attribute_values.items():
        vs = []
        for tok in vals:
            if tok in var_map:
                vs.append(var_map[tok])
            elif tok.strip().lower() in var_map:
                vs.append(var_map[tok.strip().lower()])
        if len(vs) >= 2:
            axioms.append(Distinct(*vs))
    return axioms

def _is_tautology_under_base(base_axioms: List[BoolRef], phi: BoolRef) -> bool:
    s = Solver()
    s.add(base_axioms)
    s.add(Not(phi))
    return s.check() == unsat

def _is_semantic_equiv_to_any_clue_xor(base_axioms: List[BoolRef], phi: BoolRef, clue_phis: List[BoolRef]) -> bool:
    # phi <=> c  iff  UNSAT(base ∧ Xor(phi,c))
    for cphi in clue_phis:
        s = Solver()
        s.add(base_axioms)
        s.add(Z3Xor(phi, cphi))
        if s.check() == unsat:
            return True
    return False

# ============================================================
# Helper: entailment check that avoids "unsat premises => everything entailed"
# ============================================================
def _solver_is_sat(assertions, timeout_ms: int | None = None) -> bool:
    s = Solver()
    if timeout_ms is not None:
        s.set("timeout", int(timeout_ms))
    s.add(assertions)
    return s.check() == sat


def _status_under_gamma_strong(solver: Solver, phi: BoolRef, *, timeout_ms: int | None = None) -> str:
    """Return ENTAILED / NOT_ENTAILED / CONTRADICTION / PREMISES_UNSAT / UNKNOWN.

    Strong entailment variant:
      - If premises are UNSAT, return PREMISES_UNSAT (avoid explosion).
      - For ENTAILED on non-implication constraints, require both:
          (1) Γ ∧ ¬phi is UNSAT
          (2) Γ ∧ phi is SAT
      - For implication constraints Implies(A,B), delegate to _status_under_gamma (which
        already uses the non-vacuous rule A∧¬B UNSAT AND A∧B SAT), but still guards
        against UNSAT premises.
    """
    base_assertions = list(solver.assertions())
    if not _solver_is_sat(base_assertions, timeout_ms):
        return "PREMISES_UNSAT"

    # implication special-case
    try:
        if is_app_of(phi, Z3Implies):
            return _status_under_gamma(solver, phi)
    except Exception:
        pass

    # CONTRADICTION: Γ ∧ phi UNSAT
    s1 = Solver()
    if timeout_ms is not None:
        s1.set("timeout", int(timeout_ms))
    s1.add(base_assertions)
    s1.add(phi)
    r1 = s1.check()
    if r1 == unsat:
        return "CONTRADICTION"
    if r1 == z3.unknown:
        return "UNKNOWN"

    # ENTAILED: Γ ∧ ¬phi UNSAT, with witness SAT already ensured above
    s2 = Solver()
    if timeout_ms is not None:
        s2.set("timeout", int(timeout_ms))
    s2.add(base_assertions)
    s2.add(Z3Not(phi))
    r2 = s2.check()
    if r2 == unsat:
        return "ENTAILED"
    if r2 == z3.unknown:
        return "UNKNOWN"
    return "NOT_ENTAILED"


def _is_tautology_under_base_v2(base_axioms, phi: BoolRef, *, timeout_ms: int | None = None) -> bool:
    """Tautology under base axioms Γ0 (no clues, no steps): UNSAT(Γ0 ∧ ¬phi)."""
    s = Solver()
    if timeout_ms is not None:
        s.set("timeout", int(timeout_ms))
    s.add(base_axioms)
    s.add(Z3Not(phi))
    return s.check() == unsat




def count_distinct_reasoning_steps_v13_relaxed(
    reasoning_lines,
    *,
    n_houses,
    attribute_values,
    timeout_s=2.0,
    allow_unassigned=False,
    syntactic_clues=None,
    distinct_from_syntactic_clues: bool = False,  # ignored; always enforced
    distinct_from_syntactic_clues_semantic_xor: bool = False,
    entailed_only: bool = True,
    require_token_novelty: bool = False,
    omit_tautologies: bool = True,
):
    """Analyze reasoning steps and return validity/novelty breakdown.

    Implemented definitions:

    Validity (implied by clues only):
      Γ_clues = base_axioms + syntactic_clues
      A step is VALID iff _status_under_gamma_strong(Γ_clues, step) == ENTAILED.

    Novelty (w.r.t previous steps, excluding contradictions, NO clues):
      Γ_steps = base_axioms + previous steps (excluding any step marked CONTRADICTION w.r.t Γ_steps)
      A step is NOVEL iff _status_under_gamma_strong(Γ_steps, step) != ENTAILED.

    Returned lists:
      - novel_steps_inc_clues: steps that are both VALID (entailed by clues) and NOVEL w.r.t Γ_steps
      - novel_steps_exc_clues: steps that are NOVEL w.r.t Γ_steps (validity w.r.t clues not required)

      - skipped_steps_exc_clues: all steps not included in novel_steps_exc_clues, with reasons
      - skipped_steps_inc_clues: all steps not included in novel_steps_inc_clues, with reasons

    Also returns clue_parse_errors for debugging.
    """
    syntactic_clues = syntactic_clues or []
    reasoning_lines = reasoning_lines or []

    var_map = _build_var_map_from_attribute_values(attribute_values)
    base_axioms = _build_base_axioms(n_houses, attribute_values, var_map)

    timeout_ms = int(timeout_s * 1000)

    # Base solver (Γ0)
    base_solver = Solver()
    base_solver.set("timeout", timeout_ms)
    base_solver.add(base_axioms)

    # Γ_clues = base + clues
    gamma_clues = Solver()
    gamma_clues.set("timeout", timeout_ms)
    gamma_clues.append(base_solver.assertions())

    # Γ_steps = base + previous non-contradictory steps (NO clues)
    steps_gamma = Solver()
    steps_gamma.set("timeout", timeout_ms)
    steps_gamma.append(base_solver.assertions())

    # Parse clues for Γ_clues and clue distinctness
    clue_sexprs: Set[str] = set()
    clue_phis: List[BoolRef] = []
    clue_parse_errors: List[str] = []

    for raw in syntactic_clues:
        try:
            _, cexpr = _extract_cid_and_expr(raw)
            cphi = _dsl_to_z3(cexpr, var_map)
            if cphi is None:
                cphi = _parse_constraint(cexpr, var_map)
            gamma_clues.add(cphi)
            clue_phis.append(cphi)
            clue_sexprs.add(cphi.sexpr())
        except Exception as e:
            clue_parse_errors.append(f"{raw} -> {type(e).__name__}: {e}")

    # Tracking
    seen_step_sexprs: Set[str] = set()
    seen_tokens: Set[str] = set()

    valid_steps: List[dict] = []
    novel_steps_inc_clues: List[dict] = []
    novel_steps_exc_clues: List[dict] = []
    skipped_steps_inc_clues: List[dict] = []
    skipped_steps_exc_clues: List[dict] = []

    n_total = len(reasoning_lines)
    n_parsed_ok = 0
    all_step_exprs: List[str] = []
    step_parse_errors: List[dict] = []
    non_valid_steps: List[dict] = []

    def _extract_tokens(s: str) -> Set[str]:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", s))

    for raw in reasoning_lines:
        k, expr = _extract_sid_and_expr(raw)
        all_step_exprs.append(expr)

        # --- Parse step ---
        try:
            phi = _dsl_to_z3(expr, var_map)
            if phi is None:
                phi = _parse_constraint(expr, var_map)
            if phi is None:
                raise ValueError("Failed to parse step to z3 constraint.")
        except Exception as e:
            entry = {"k": k, "raw": raw, "expr": expr, "status": "PARSE_ERROR", "error": f"{type(e).__name__}: {e}"}
            step_parse_errors.append(entry)
            # PARSE_ERROR is non-valid by definition
            non_valid_steps.append({
                "k": k,
                "raw": raw,
                "expr": expr,
                "validity_status": "PARSE_ERROR",
                "reason": entry["error"],
            })
            skipped_steps_inc_clues.append(entry)
            skipped_steps_exc_clues.append(entry)
            continue

        n_parsed_ok += 1
        phi_sexpr = phi.sexpr()

        # --- Duplicate step ---
        if phi_sexpr in seen_step_sexprs:
            # If a step duplicates a previous expression, we still want correct labeling.
            # In particular: if the duplicated step is a CONTRADICTION w.r.t. previously accepted steps,
            # it should be reported as CONTRADICTION (not merely DUPLICATE_STEP).
            try:
                # TAUTOLOGY is always valid (but not novel).
                if _is_tautology_under_base_v2(base_axioms, phi, timeout_ms=timeout_ms):
                    valid_steps.append({
                        "k": k, "raw": raw, "expr": expr,
                        "validity_status": "TAUTOLOGY",
                        "sexpr": phi_sexpr,
                        "reason": "Tautology under base axioms (Base ∧ ¬phi is UNSAT).",
                    })
                    entry = {"k": k, "raw": raw, "expr": expr, "status": "TAUTOLOGY", "error": None}
                    skipped_steps_inc_clues.append(entry)
                    skipped_steps_exc_clues.append(entry)
                    continue

                # Check contradiction against previously accepted steps first.
                step_status = _status_under_gamma_strong(steps_gamma, phi, timeout_ms=timeout_ms)
                if step_status == "CONTRADICTION":
                    non_valid_steps.append({
                        "k": k, "raw": raw, "expr": expr,
                        "validity_status": "CONTRADICTION",
                        "sexpr": phi_sexpr,
                        "reason": "Contradicts previously accepted steps (Γ_steps ∧ phi is UNSAT).",
                    })
                    entry = {"k": k, "raw": raw, "expr": expr, "status": "CONTRADICTION", "error": None}
                    skipped_steps_inc_clues.append(entry)
                    skipped_steps_exc_clues.append(entry)
                    continue

            except Exception as e:
                non_valid_steps.append({
                    "k": k, "raw": raw, "expr": expr,
                    "validity_status": "VALIDITY_CHECK_ERROR",
                    "sexpr": phi_sexpr,
                    "reason": f"{type(e).__name__}: {e}",
                })
                entry = {"k": k, "raw": raw, "expr": expr, "status": "DUPLICATE_STEP", "error": None}
                skipped_steps_inc_clues.append(entry)
                skipped_steps_exc_clues.append(entry)
                continue

            # Not a contradiction under previously accepted steps => treat as a duplicate.
            # (Still account for validity stats w.r.t. clues.)
            try:
                validity_status = _status_under_gamma_strong(gamma_clues, phi, timeout_ms=timeout_ms)
                if validity_status == "ENTAILED":
                    valid_steps.append({"k": k, "raw": raw, "expr": expr, "validity_status": validity_status, "sexpr": phi_sexpr})
                else:
                    if validity_status == "CONTRADICTION":
                        reason = "Contradicts clues (Γ_clues ∧ phi is UNSAT)."
                    elif validity_status == "NOT_ENTAILED":
                        reason = "Not implied by clues (Γ_clues ∧ ¬phi is SAT; counterexample exists)."
                    elif validity_status == "PREMISES_UNSAT":
                        reason = "Clue premises are UNSAT; validity is undefined (avoid explosion)."
                    elif validity_status == "UNKNOWN":
                        reason = "Z3 returned unknown under clue premises."
                    else:
                        reason = f"Validity status: {validity_status}"
                    non_valid_steps.append({
                        "k": k, "raw": raw, "expr": expr,
                        "validity_status": validity_status,
                        "sexpr": phi_sexpr,
                        "reason": reason,
                    })
            except Exception as e:
                non_valid_steps.append({
                    "k": k, "raw": raw, "expr": expr,
                    "validity_status": "VALIDITY_CHECK_ERROR",
                    "sexpr": phi_sexpr,
                    "reason": f"{type(e).__name__}: {e}",
                })

            entry = {"k": k, "raw": raw, "expr": expr, "status": "DUPLICATE_STEP", "error": None}
            skipped_steps_inc_clues.append(entry)
            skipped_steps_exc_clues.append(entry)
            continue

        # --- Distinct from clues (always) ---
        if phi_sexpr in clue_sexprs:
            # Clue restatements should still be accounted for in validity stats.
            try:
                if _is_tautology_under_base_v2(base_axioms, phi, timeout_ms=timeout_ms):
                    valid_steps.append({
                        "k": k, "raw": raw, "expr": expr,
                        "validity_status": "TAUTOLOGY",
                        "sexpr": phi_sexpr,
                        "reason": "Tautology under base axioms (Base ∧ ¬phi is UNSAT).",
                    })
                else:
                    validity_status = _status_under_gamma_strong(gamma_clues, phi, timeout_ms=timeout_ms)
                    if validity_status == "ENTAILED":
                        valid_steps.append({"k": k, "raw": raw, "expr": expr, "validity_status": validity_status, "sexpr": phi_sexpr})
                    else:
                        if validity_status == "CONTRADICTION":
                            reason = "Contradicts clues (Γ_clues ∧ phi is UNSAT)."
                        elif validity_status == "NOT_ENTAILED":
                            reason = "Not implied by clues (Γ_clues ∧ ¬phi is SAT; counterexample exists)."
                        elif validity_status == "PREMISES_UNSAT":
                            reason = "Clue premises are UNSAT; validity is undefined (avoid explosion)."
                        elif validity_status == "UNKNOWN":
                            reason = "Z3 returned unknown under clue premises."
                        else:
                            reason = f"Validity status: {validity_status}"
                        non_valid_steps.append({"k": k, "raw": raw, "expr": expr, "validity_status": validity_status, "sexpr": phi_sexpr, "reason": reason})
            except Exception as e:
                non_valid_steps.append({
                    "k": k, "raw": raw, "expr": expr,
                    "validity_status": "VALIDITY_CHECK_ERROR",
                    "reason": f"{type(e).__name__}: {e}",
                })

            entry = {"k": k, "raw": raw, "expr": expr, "status": "RESTATES_CLUE", "error": None}
            skipped_steps_inc_clues.append(entry)
            skipped_steps_exc_clues.append(entry)
            seen_step_sexprs.add(phi_sexpr)
            continue

        if distinct_from_syntactic_clues_semantic_xor:
            try:
                if _is_semantic_equiv_to_any_clue_xor(base_axioms, phi, clue_phis):
                    # Semantic clue restatements should still be accounted for in validity stats.
                    try:
                        if _is_tautology_under_base_v2(base_axioms, phi, timeout_ms=timeout_ms):
                            valid_steps.append({
                                "k": k, "raw": raw, "expr": expr,
                                "validity_status": "TAUTOLOGY",
                                "sexpr": phi_sexpr,
                                "reason": "Tautology under base axioms (Base ∧ ¬phi is UNSAT).",
                            })
                        else:
                            validity_status = _status_under_gamma_strong(gamma_clues, phi, timeout_ms=timeout_ms)
                            if validity_status == "ENTAILED":
                                valid_steps.append({"k": k, "raw": raw, "expr": expr, "validity_status": validity_status, "sexpr": phi_sexpr})
                            else:
                                if validity_status == "CONTRADICTION":
                                    reason = "Contradicts clues (Γ_clues ∧ phi is UNSAT)."
                                elif validity_status == "NOT_ENTAILED":
                                    reason = "Not implied by clues (Γ_clues ∧ ¬phi is SAT; counterexample exists)."
                                elif validity_status == "PREMISES_UNSAT":
                                    reason = "Clue premises are UNSAT; validity is undefined (avoid explosion)."
                                elif validity_status == "UNKNOWN":
                                    reason = "Z3 returned unknown under clue premises."
                                else:
                                    reason = f"Validity status: {validity_status}"
                                non_valid_steps.append({"k": k, "raw": raw, "expr": expr, "validity_status": validity_status, "sexpr": phi_sexpr, "reason": reason})
                    except Exception as e:
                        non_valid_steps.append({
                            "k": k, "raw": raw, "expr": expr,
                            "validity_status": "VALIDITY_CHECK_ERROR",
                            "reason": f"{type(e).__name__}: {e}",
                        })

                    entry = {"k": k, "raw": raw, "expr": expr, "status": "RESTATES_CLUE_SEMANTIC", "error": None}
                    skipped_steps_inc_clues.append(entry)
                    skipped_steps_exc_clues.append(entry)
                    seen_step_sexprs.add(phi_sexpr)
                    continue
            except Exception as e:
                entry = {"k": k, "raw": raw, "expr": expr, "status": "CLUE_EQUIV_CHECK_ERROR", "error": f"{type(e).__name__}: {e}"}
                skipped_steps_inc_clues.append(entry)
                skipped_steps_exc_clues.append(entry)
                seen_step_sexprs.add(phi_sexpr)
                continue

        # --- Token novelty (optional) ---
        if require_token_novelty:
            toks = _extract_tokens(expr)
            if toks.issubset(seen_tokens):
                entry = {"k": k, "raw": raw, "expr": expr, "status": "TOKEN_NOT_NOVEL", "error": None}
                skipped_steps_inc_clues.append(entry)
                skipped_steps_exc_clues.append(entry)
                seen_step_sexprs.add(phi_sexpr)
                continue

        # --- Tautology under base (optional) ---
        if omit_tautologies:
            try:
                if _is_tautology_under_base_v2(base_axioms, phi, timeout_ms=timeout_ms):
                    entry = {"k": k, "raw": raw, "expr": expr, "status": "TAUTOLOGY", "error": None}
                    # TAUTOLOGY should be treated as valid (per caller requirement)
                    valid_steps.append({
                        "k": k,
                        "raw": raw,
                        "expr": expr,
                        "validity_status": "TAUTOLOGY",
                        "sexpr": phi_sexpr,
                        "reason": "Tautology under base axioms (Base ∧ ¬phi is UNSAT).",
                    })
                    skipped_steps_inc_clues.append(entry)
                    skipped_steps_exc_clues.append(entry)
                    seen_step_sexprs.add(phi_sexpr)
                    continue
            except Exception as e:
                entry = {"k": k, "raw": raw, "expr": expr, "status": "TAUTOLOGY_CHECK_ERROR", "error": f"{type(e).__name__}: {e}"}
                skipped_steps_inc_clues.append(entry)
                skipped_steps_exc_clues.append(entry)
                seen_step_sexprs.add(phi_sexpr)
                continue

        # --- Validity: implied by clues (strong) ---
        validity_status = _status_under_gamma_strong(gamma_clues, phi, timeout_ms=timeout_ms)
        is_valid = (validity_status == "ENTAILED")
        if is_valid:
            valid_steps.append({"k": k, "raw": raw, "expr": expr, "validity_status": validity_status, "sexpr": phi_sexpr})
        else:
            # Record all non-valid steps with explicit reason
            if validity_status == "CONTRADICTION":
                reason = "Contradicts clues (Γ_clues ∧ phi is UNSAT)."
            elif validity_status == "NOT_ENTAILED":
                reason = "Not implied by clues (Γ_clues ∧ ¬phi is SAT; counterexample exists)."
            elif validity_status == "PREMISES_UNSAT":
                reason = "Clue premises are UNSAT; validity is undefined (avoid explosion)."
            elif validity_status == "UNKNOWN":
                reason = "Z3 returned unknown under clue premises."
            else:
                reason = f"Validity status: {validity_status}"
            non_valid_steps.append({"k": k, "raw": raw, "expr": expr, "validity_status": validity_status, "sexpr": phi_sexpr, "reason": reason})

        # --- Novelty: not implied by previous steps (NO clues), excluding contradictions ---
        steps_status = _status_under_gamma_strong(steps_gamma, phi, timeout_ms=timeout_ms)

        if steps_status == "CONTRADICTION":
            entry = {"k": k, "raw": raw, "expr": expr, "status": "CONTRADICTION", "error": None}
            skipped_steps_exc_clues.append(entry)
            skipped_steps_inc_clues.append(entry)
            seen_step_sexprs.add(phi_sexpr)
            if require_token_novelty:
                seen_tokens |= _extract_tokens(expr)
            continue

        is_novel_wrt_prev = (steps_status != "ENTAILED")

        # Novelty selection (requested semantics):
        #   - inc_clues: VALID (entailed by clues) + NOVEL (not implied by prev non-contradictory steps, NO clues).
        #   - exc_clues: NOVEL only (validity w.r.t. clues NOT required).
        inc_included = bool(is_valid and is_novel_wrt_prev)
        exc_included = bool(is_novel_wrt_prev)

        if inc_included or exc_included:
            entry = {
                "k": k, "raw": raw, "expr": expr,
                "validity_status": validity_status,
                "steps_status": steps_status,
                "sexpr": phi_sexpr,
            }
            if inc_included:
                novel_steps_inc_clues.append(entry)
            else:
                # Not included in inc_clues because it is not valid w.r.t clues.
                skipped_steps_inc_clues.append({
                    "k": k, "raw": raw, "expr": expr,
                    "status": "NOT_IMPLIED_BY_CLUES",
                    "error": None,
                })

            if exc_included:
                novel_steps_exc_clues.append(entry)
        else:
            # Not novel w.r.t previous steps => excluded from both lists.
            entry_exc = {"k": k, "raw": raw, "expr": expr, "status": "IMPLIED_BY_PREV_STEPS", "error": None}
            skipped_steps_exc_clues.append(entry_exc)

            status_inc = "IMPLIED_BY_PREV_STEPS" if is_valid else "NOT_IMPLIED_BY_CLUES"
            entry_inc = {"k": k, "raw": raw, "expr": expr, "status": status_inc, "error": None}
            skipped_steps_inc_clues.append(entry_inc)

        # Add to Γ_steps only if not contradiction w.r.t prev steps
        steps_gamma.add(phi)

        # Update seen
        seen_step_sexprs.add(phi_sexpr)
        if require_token_novelty:
            seen_tokens |= _extract_tokens(expr)

    # If entailed_only is requested, we treat "invalid under clues" as skipped for inc_clues,
    # but still allow exc_clues novelty analysis.

        # Build output lists in the schema requested by the caller.
    list_all_steps = all_step_exprs
    list_steps_valid = [x["expr"] for x in valid_steps]
    list_steps_non_valid = non_valid_steps

    list_novel_steps_inc_clues = [x["expr"] for x in novel_steps_inc_clues]
    list_novel_steps_exc_clues = [x["expr"] for x in novel_steps_exc_clues]

    list_skipped_steps_inc_clues = skipped_steps_inc_clues
    list_skipped_steps_exc_clues = skipped_steps_exc_clues

    list_clue_parse_errors = clue_parse_errors
    list_step_parse_errors = step_parse_errors

    return {
        "n_steps_total": n_total,
        "n_steps_parsed_ok": n_parsed_ok,
        "n_steps_valid": len(list_steps_valid),
        "n_steps_novel_inc_clues": len(list_novel_steps_inc_clues),
        "n_steps_novel_exc_clues": len(list_novel_steps_exc_clues),
        "n_non_valid_contradiction": len([x for x in list_steps_non_valid if x['validity_status'] == 'CONTRADICTION']),
        #"n_novel_inc_clues_contradiction": len([x for x in list_novel_steps_inc_clues if x['steps_status'] == 'CONTRADICTION']),

        "list_all_steps": list_all_steps,
        "list_steps_valid": list_steps_valid,
        "list_steps_non_valid": list_steps_non_valid,
        "list_novel_steps_inc_clues": list_novel_steps_inc_clues,
        "list_novel_steps_exc_clues": list_novel_steps_exc_clues,
        "list_skipped_steps_inc_clues": list_skipped_steps_inc_clues,
        "list_skipped_steps_exc_clues": list_skipped_steps_exc_clues,
        "list_clue_parse_errors": list_clue_parse_errors,
        "list_step_parse_errors": list_step_parse_errors,
    }

if __name__ == "__main__":
    # ============================================================
    # Quick sanity tests for:
    # - clue restatement filtering
    # - contradiction-as-precursor exclusion
    # - base-axiom tautology omission (UNSAT(base ∧ Not(phi)))
    # ============================================================

    # -----------------------
    # Test A: 3-house mini puzzle (existing style)
    # -----------------------
    '''
    n_houses = 2
    attribute_values = {
        "Name": ["Eric", "Arnold"],
        "Hobby": ["photography", "gardening"],
        "PhoneModel": ["iphone 13", "samsung galaxy s21"],
        "Children": ["Bella", "Fred"],
        "Animal": ["cat", "horse"],
        "Vacation": ["beach", "mountain"],
    }

    syntactic_clues = [
        "C1: cat == Fred.",
        "C2: horse == gardening.",
        "C3: horse == photography.",
        "C4: horse + 1 == beach.",
        "C5: horse + 1 == iphone 13."
    ]

    reasoning_lines = [
      "C1: cat == Fred.",
      "C2: horse == gardening.",
      "C3: horse == photography.",
      "C4: horse + 1 == beach.",
      "C5: horse + 1 == iphone 13.",
      "S1: cat == Fred.",
      "S2: horse == gardening.",
      "S3: horse == photography.",
      "S4: horse + 1 == beach.",
      "S5: horse + 1 == iphone 13.",
      "C1: cat == Fred.",
      "C2: horse == gardening.",
      "C3: horse == photography.",
      "C4: horse + 1 == beach.",
      "C5: horse + 1 == iphone 13.",
      "S1: cat == Fred.",
      "S2: horse == gardening.",
      "S3: horse == photography.",
      "S4: horse + 1 == beach.",
      "S5: horse + 1 == iphone 13.",
      "C1: cat == Fred.",
      "C2: horse == gardening.",
      "C3: horse == photography.",
      "C4: horse + 1 == beach.",
      "C5: horse + 1 == iphone 13.",
      "S1: cat == Fred.",
      "S2: horse == gardening.",
      "S3: horse == photography.",
      "S4: horse + 1 == beach.",
      "S5: horse + 1 == iphone 13."
    ]
    outA = count_distinct_reasoning_steps_v13_relaxed(
        reasoning_lines=reasoning_lines,
        n_houses=n_houses,
        attribute_values=attribute_values,
        syntactic_clues=syntactic_clues,
        distinct_from_syntactic_clues_semantic_xor=True,  # enable semantic (UNSAT(Xor)) clue equivalence too
        entailed_only=True,
        require_token_novelty=False,
        omit_tautologies=True,
    )

    print(json.dumps(outA, indent=2))




    '''
    
    
    n_houses = 3
    attribute_values = {
        "Name": ["Peter", "Eric", "Arnold"],
        "Color": ["red", "white", "yellow"],
        "Children": ["Fred", "Meredith", "Bella"],
    }

    syntactic_clues = [
        "C1: Arnold == red.",
        "C2: red == 2.",
        "C3: Bella == 1.",
        "C4: Fred < Eric.",
        "C5: white == Meredith.",
    ]

    reasoning_lines = [
        "S1: red == Arnold.",              # ENTAILED (from C1 + C2)
        "S2: 2 == Arnold.",  # ENTAILED (from C1 + C2)
        "I am NL step",
        "S3: Not(Eric == 1).",           # ENTAILED (from C4 + 3-house domain)
        "S4: Or(Eric ==1, Eric == 2, Eric == 3).", # ENTAILED (from S2 + domain)
        "S5: Not(Eric == 2).",           # ENTAILED (since Arnold==2 and names are Distinct)
        "S6: Eric == 1.",                # ENTAILED (from S3 + S4)
        "S7: Eric == 3.",  # ENTAILED (from S3 + S4)
        "S8: Eric == 1.",  # ENTAILED (from S3 + S4)
        # ---- negative / filtering tests ----
        "S9: Arnold == 2.",              # DUPLICATE_STEP
        "S10: red == 2.",                 # RESTATES_CLUE (exact match to C2)
        "S11: Or(eric ==1, Eric ==, Eric == 3).",   # PARSE_ERROR
    ]

    outA = count_distinct_reasoning_steps_v13_relaxed(
        reasoning_lines=reasoning_lines,
        n_houses=n_houses,
        attribute_values=attribute_values,
        syntactic_clues=syntactic_clues,
        distinct_from_syntactic_clues_semantic_xor=True,  # enable semantic (UNSAT(Xor)) clue equivalence too
        entailed_only=True,
        require_token_novelty=False,
        omit_tautologies=True,
    )

    print(json.dumps(outA, indent=2))

    #print("\n=== Test A (3-house) ===")
    #print("list_novel_steps_inc_clues:", outA["list_novel_steps_inc_clues"])
    #print("skipped_inc_clues:", [(x["k"], x["status"], x.get("expr"), x.get("error")) for x in outA["list_skipped_steps_inc_clues"]])

    # -----------------------
    # Test B: 2-house tautology stress-test (GENERALIZED)
    # Demonstrates base-axiom tautologies with:
    #  - Or(x==1, x==2) is ALWAYS TRUE under domain 1..2
    #  - Or(P, Not(P)) is ALWAYS TRUE (boolean tautology)
    #  - Implies(A,B) can be tautology if A is impossible under base
    # -----------------------
    n_houses_B = 2
    attribute_values_B = {
        "Name": ["Arnold", "Bob"],
        "Height": ["short", "tall"],
    }
    syntactic_clues_B = []  # IMPORTANT: "empty model" here means NO clues/steps, only base axioms (ranges + Distinct)

    reasoning_lines_B = [
        "S1: Or(Arnold == 1, Arnold == 2).",                 # TAUTOLOGY under base (domain is exactly {1,2})
        "S2: Or(Bob == 1, Not(Bob == 1)).",                  # TAUTOLOGY (P ∨ ¬P)
        "S3: Implies(Arnold == 3, Bob == 1).",               # TAUTOLOGY under base (Arnold==3 is impossible under 2-house domain)
        "S4: Arnold == 1.",                                  # NOT_ENTAILED under base-only (should be rejected if entailed_only=True)
        "S5: Not(Arnold == 1).",                             # NOT_ENTAILED under base-only (also rejected)
    ]

    outB = count_distinct_reasoning_steps_v13_relaxed(
        reasoning_lines=reasoning_lines_B,
        n_houses=n_houses_B,
        attribute_values=attribute_values_B,
        syntactic_clues=syntactic_clues_B,
        distinct_from_syntactic_clues_semantic_xor=True,
        entailed_only=True,
        require_token_novelty=False,
        omit_tautologies=True,
    )

    #print("\n=== Test B (2-house tautologies under base axioms) ===")
    #print("list_novel_steps_inc_clues:", outB["list_novel_steps_inc_clues"])
    #print("skipped_inc_clues:", [(x["k"], x["status"], x.get("expr"), x.get("error")) for x in outB["list_skipped_steps_inc_clues"]])


    n_houses_C = 3
    attribute_values_C = {
        "Name": ["Alice", "Bob", "Carol"],
    }

    syntactic_clues_C = [
        "C1: Alice != 1.",
    ]

    reasoning_lines_C = [
        "S1: Alice != 1.",   # restates clue (valid but not novel)
        "S2: Alice == 2.",   # GUESS: novel wrt steps, but NOT entailed by clues
    ]

    outC = count_distinct_reasoning_steps_v13_relaxed(
        reasoning_lines=reasoning_lines_C,
        n_houses=n_houses_C,
        attribute_values=attribute_values_C,
        syntactic_clues=syntactic_clues_C,
        distinct_from_syntactic_clues_semantic_xor=True,
        entailed_only=True,
        require_token_novelty=False,
        omit_tautologies=True,
    )

    #print(json.dumps(outC, indent=2))
