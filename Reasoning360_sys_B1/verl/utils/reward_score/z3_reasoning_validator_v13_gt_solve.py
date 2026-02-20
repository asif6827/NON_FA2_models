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


# ----------------------------- Normalization -----------------------------

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

def _parse_constraint(expr: str, var_map: Dict[str, Any]) -> Any:
    """
    Parse constraint forms (tokens are normalized individually):
      A == B
      A != B
      A < B
      A > B
      A + 1 == B
      A == 5
    Returns a Z3 BoolRef.
    """
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1]
    # A + 1 == B
    m = re.match(r"^(.+?)\s*\+\s*1\s*==\s*(.+?)$", e)
    if m:
        left_raw, right_raw = m.group(1), m.group(2)
        L = var_map[_norm_token(left_raw)]
        R = var_map[_norm_token(right_raw)]
        return L + 1 == R

    # binary ops
    for op in ("==", "!=", "<", ">"):
        # split only once
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

            if op == "==":
                return L == R
            if op == "!=":
                return L != R
            if op == "<":
                return L < R
            if op == ">":
                return L > R

    raise ValueError(f"Unrecognized constraint syntax: {expr!r}")


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
    tc1 = {
        "n_houses": 2,
        "attribute_values": {"Name": ["Arnold", "Eric"], "Pet": ["cAt", "dog"]},
        "syntactic_clues": ["C1: Arnold == cat.", "C2: Eric == dog."],
        "reasoning": ["Arnold is cat.", "S1: Arnold == cat. [C1]", "Eric is dog.", "S2: Eric == dog. [C2]"],
        "ground_truth": {"header": ["House", "Name", "Pet"], "rows": [["1", "Arnold", "cat"], ["2", "Eric", "dog"]]},
    }

    payload = {
        "n_houses": 2,
        "attribute_values": {
            "Name": [
                "Arnold",
                "Eric"
            ],
            "BookGenre": [
                "science fiction",
                "mystery"
            ],
            "Vacation": [
                "mountain",
                "beach"
            ],
            "Smoothie": [
                "desert",
                "cherry"
            ],
            "Children": [
                "Fred",
                "Bella"
            ],
            "Sport": [
                "basketball",
                "soccer"
            ]
        },
        "syntactic_clues": [
            "C1: Bella == Eric.",
            "C2: soccer == Fred.",
            "C3: cherry == mystery.",
            "C4: beach == mystery.",
            "C5: mountain == 2.",
            "C6: Eric > Arnold."
        ],
        "reasoning": [
            "S1: Bella == Eric. [C1]",
            "S2: soccer == Fred. [C2]",
            "S3: cherry == mystery. [C3]",
            "S4: beach == mystery. [C4]",
            "S5: mountain == 2. [C5]",
            "S6: Eric > Arnold. [C6]"
        ],
        "ground_truth": {
            "header": [
                "House",
                "Name",
                "BookGenre",
                "Vacation",
                "Smoothie",
                "Children",
                "FavoriteSport"
            ],
            "rows": [
                [
                    "1",
                    "Arnold",
                    "mystery",
                    "beach",
                    "cherry",
                    "Fred",
                    "soccer"
                ],
                [
                    "2",
                    "Eric",
                    "science fiction",
                    "mountain",
                    "desert",
                    "Bella",
                    "basketball"
                ]
            ]
        },
        "puzzle_text": [
            "There are 2 houses, numbered 1 to 2 from left to right, as seen from across the street. Each house is occupied by a different person. Each house has a unique attribute for each of the following characteristics:",
            " - Each person has a unique name: `Arnold`, `Eric`",
            " - People have unique favorite book genres: `science fiction`, `mystery`",
            " - Each person prefers a unique type of vacation: `mountain`, `beach`",
            " - Everyone has a favorite smoothie: `desert`, `cherry`",
            " - Each mother is accompanied by their child: `Fred`, `Bella`",
            " - People have unique favorite sports: `basketball`, `soccer`",
            "",
            "## Clues:",
            "1. The person's child is named Bella is Eric.",
            "2. The person who loves soccer is the person's child is named Fred.",
            "3. The person who likes Cherry smoothies is the person who loves mystery books.",
            "4. The person who loves beach vacations is the person who loves mystery books.",
            "5. The person who enjoys mountain retreats is in the second house.",
            "6. Eric is somewhere to the right of Arnold."
        ]
    }
    r1 = solve_and_validate_payload(payload, timeout_s=2.0)

    print(json.dumps(r1, indent=2))
    print(5*"\n")



    return 0


def main() -> None:
    run_self_tests()

    #payload = _load_json(args.input_json)

    #print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()