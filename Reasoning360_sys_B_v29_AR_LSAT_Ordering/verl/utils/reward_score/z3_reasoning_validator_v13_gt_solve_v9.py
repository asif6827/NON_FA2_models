#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AR-LSAT ordering Z3 validator.

Expected payload:
{
  "problem_type": "ordering",
  "world_model": {"entities": [...], "domains": {"positions": [...]}, ...},
  "rules": [...],
  "facts": [...],
  "question_semantics": {"question_type": "could_be_true", ...},
  "options": {"A": "...", ...},
  "reasoning": [...],
  "solution": {"selected_option": "A"},
  "ground_truth": "A" or {"answer": "A"}
}
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Set

import z3  # type: ignore
from z3 import And, Distinct, Int, Not, Or, Solver, Xor, Implies, sat, unsat  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_FUNC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")


def normalize_header(data_sample: Any) -> Any:
    # kept for backward import compatibility
    return data_sample


def normalize_months_in_rows(z3_solution: dict) -> dict:
    # kept for backward import compatibility
    return z3_solution


def _norm_token(x: Any) -> str:
    s = str(x).strip().strip("`'\"“”‘’")
    s = re.sub(r"[.,;:!?]+$", "", s)
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _split_top_level_args(s: str) -> List[str]:
    args, buf, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
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


def _strip_step_expr(line: str) -> Optional[Tuple[int, str]]:
    m = _STEP_RE.match((line or "").strip())
    if not m:
        return None
    expr = m.group(2).strip()
    if expr.endswith("."):
        expr = expr[:-1].strip()
    return int(m.group(1)), expr


def _infer_n_positions(world_model: Dict[str, Any]) -> int:
    domains = world_model.get("domains", {}) if isinstance(world_model, dict) else {}
    positions = domains.get("positions") or domains.get("position") or []
    vals = []
    for p in positions:
        try:
            vals.append(int(str(p)))
        except Exception:
            pass
    if vals:
        return max(vals)
    entities = world_model.get("entities", []) if isinstance(world_model, dict) else []
    return max(1, len(entities))


def _build_base(world_model: Dict[str, Any], timeout_s: float) -> Tuple[Solver, Dict[str, Any], int]:
    entities = world_model.get("entities", []) if isinstance(world_model, dict) else []
    n = _infer_n_positions(world_model)
    var_map: Dict[str, Any] = {}
    for ent in entities:
        tok = _norm_token(ent)
        if tok:
            var_map[tok] = Int(tok)
            # preserve raw and case variants too
            var_map[str(ent)] = var_map[tok]
            var_map[str(ent).lower()] = var_map[tok]

    s = Solver()
    s.set("timeout", int(float(timeout_s) * 1000))
    uniq_vars = list({id(v): v for v in var_map.values()}.values())
    for v in uniq_vars:
        s.add(And(v >= 1, v <= n))
    return s, var_map, n


def _term(raw: str, var_map: Dict[str, Any]) -> Any:
    r = raw.strip()
    if re.fullmatch(r"-?\d+", r):
        return int(r)
    key = _norm_token(r)
    if key in var_map:
        return var_map[key]
    if r in var_map:
        return var_map[r]
    if r.lower() in var_map:
        return var_map[r.lower()]
    raise KeyError(f"Unknown token: {r!r}")


def _parse_expr(expr: str, var_map: Dict[str, Any], options: Optional[Dict[str, str]] = None) -> Any:
    e = str(expr).strip()
    if e.endswith("."):
        e = e[:-1].strip()

    m_status = re.fullmatch(r"(Sat|Unsat)\(Option_([A-Z])\)", e, flags=re.IGNORECASE)
    if m_status:
        # Reasoning status steps are handled outside; parsing as a Bool marker is enough if needed.
        return z3.Bool(f"{m_status.group(1).lower()}_option_{m_status.group(2).upper()}")

    m = _FUNC_RE.match(e)
    if m:
        fn = m.group(1).lower()
        args = _split_top_level_args(m.group(2))
        if fn == "and":
            return And(*[_parse_expr(a, var_map, options) for a in args])
        if fn == "or":
            return Or(*[_parse_expr(a, var_map, options) for a in args])
        if fn == "not":
            if len(args) != 1:
                raise ValueError("Not expects one argument")
            return Not(_parse_expr(args[0], var_map, options))
        if fn == "implies":
            if len(args) != 2:
                raise ValueError("Implies expects two arguments")
            return Implies(_parse_expr(args[0], var_map, options), _parse_expr(args[1], var_map, options))
        if fn == "xor":
            if len(args) != 2:
                raise ValueError("Xor expects two arguments")
            return Xor(_parse_expr(args[0], var_map, options), _parse_expr(args[1], var_map, options))
        if fn == "distinct":
            return Distinct(*[_term(a, var_map) for a in args])
        raise ValueError(f"Unsupported function: {fn}")

    m = re.match(r"^(.+?)\s*([+-])\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        left, sign, k, right = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        return (_term(left, var_map) + k == _term(right, var_map)) if sign == "+" else (_term(left, var_map) - k == _term(right, var_map))

    for op in ("<=", ">=", "==", "!=", "<", ">"):
        parts = e.split(op)
        if len(parts) == 2:
            L, R = _term(parts[0], var_map), _term(parts[1], var_map)
            if op == "==": return L == R
            if op == "!=": return L != R
            if op == "<": return L < R
            if op == ">": return L > R
            if op == "<=": return L <= R
            if op == ">=": return L >= R

    raise ValueError(f"Unrecognized expression: {expr!r}")


def _solver_with(assertions: List[Any], timeout_s: float) -> Solver:
    s = Solver()
    s.set("timeout", int(float(timeout_s) * 1000))
    s.add(assertions)
    return s


def _status_under_gamma(gamma_assertions: List[Any], phi: Any, timeout_s: float) -> str:
    base = _solver_with(gamma_assertions, timeout_s)
    if base.check() != sat:
        return "PREMISES_UNSAT"

    s_contra = _solver_with(gamma_assertions, timeout_s)
    s_contra.add(phi)
    r1 = s_contra.check()
    if r1 == unsat:
        return "CONTRADICTION"
    if r1 == z3.unknown:
        return "UNKNOWN"

    s_ent = _solver_with(gamma_assertions, timeout_s)
    s_ent.add(Not(phi))
    r2 = s_ent.check()
    if r2 == unsat:
        return "ENTAILED"
    if r2 == z3.unknown:
        return "UNKNOWN"
    return "NOT_ENTAILED"


def _is_option_sat(base_assertions: List[Any], opt_phi: Any, timeout_s: float) -> bool:
    s = _solver_with(base_assertions, timeout_s)
    s.add(opt_phi)
    return s.check() == sat


def _eval_option(question_type: str, base_assertions: List[Any], opt_phi: Any, timeout_s: float) -> bool:
    qt = (question_type or "").lower()
    if qt in {"could_be_true", "acceptability", "partial_acceptability", "other"}:
        return _is_option_sat(base_assertions, opt_phi, timeout_s)
    if qt in {"cannot_be_true", "must_be_false"}:
        return not _is_option_sat(base_assertions, opt_phi, timeout_s)
    if qt in {"must_be_true", "must_follow"}:
        s = _solver_with(base_assertions, timeout_s)
        s.add(Not(opt_phi))
        return s.check() == unsat
    if qt == "could_be_false":
        s = _solver_with(base_assertions, timeout_s)
        s.add(Not(opt_phi))
        return s.check() == sat
    return _is_option_sat(base_assertions, opt_phi, timeout_s)


def _selected_from_ground_truth(gt: Any) -> Optional[str]:
    if isinstance(gt, str):
        return gt.strip().upper()
    if isinstance(gt, dict):
        for k in ("answer", "selected_option", "ground_truth_option"):
            if gt.get(k):
                return str(gt[k]).strip().upper()
    return None


def solve_and_validate_payload(payload: Dict[str, Any], *, timeout_s: float = 2.0, conflict_tolerant_clues: bool = False) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "parse_status": "INIT",
        "base_sat_full_GT": 0.0,
        "n_steps_total": 0,
        "n_steps_parsed_ok": 0,
        "n_steps_valid": 0,
        "n_steps_novel_inc_clues": 0,
        "n_non_valid_contradiction": 0,
        "list_steps_non_valid": [],
        "list_novel_steps_inc_clues": [],
    }

    try:
        if payload.get("problem_type") != "ordering":
            report["parse_status"] = "UNSUPPORTED_PROBLEM_TYPE"
            return report

        world_model = payload.get("world_model") or {}
        rules = payload.get("rules") or []
        facts = payload.get("facts") or []
        reasoning = payload.get("reasoning") or []
        options = payload.get("options") or {}
        solution = payload.get("solution") or {}
        selected = str(solution.get("selected_option", "")).strip().upper()
        gt_selected = _selected_from_ground_truth(payload.get("ground_truth"))
        qsem = payload.get("question_semantics") or {}
        question_type = str(qsem.get("question_type") or payload.get("question_type") or "other")

        base_solver, var_map, n_positions = _build_base(world_model, timeout_s)

        parsed_rules: List[Any] = []
        parse_errors: List[str] = []
        for raw in list(rules) + list(facts):
            try:
                parsed_rules.append(_parse_expr(raw, var_map, options))
            except Exception as e:
                parse_errors.append(f"{raw} -> {type(e).__name__}: {e}")

        base_assertions = list(base_solver.assertions()) + parsed_rules
        base_sat = (_solver_with(base_assertions, timeout_s).check() == sat)

        option_results: Dict[str, Any] = {}
        parsed_options: Dict[str, Any] = {}
        for label, raw in options.items():
            lab = str(label).strip().upper()
            try:
                phi = _parse_expr(raw, var_map, options)
                parsed_options[lab] = phi
                option_results[lab] = _eval_option(question_type, base_assertions, phi, timeout_s)
            except Exception as e:
                option_results[lab] = {"parse_error": f"{type(e).__name__}: {e}"}

        selected_option_valid_by_semantics = bool(option_results.get(selected) is True)
        selected_option_matches_gt = bool(gt_selected and selected == gt_selected)

        # Validate syntactic reasoning lines only. Ignore Sat/Unsat status steps for entailment counts,
        # but count them as parsed if they match actual option feasibility.
        valid_steps: List[dict] = []
        non_valid_steps: List[dict] = []
        novel_steps: List[dict] = []
        seen_sexprs: Set[str] = set()
        step_assertions: List[Any] = []
        n_total = 0
        n_parsed = 0
        n_contra = 0

        for line in reasoning:
            parsed_line = _strip_step_expr(line)
            if not parsed_line:
                continue
            n_total += 1
            k, expr = parsed_line

            m_status = re.fullmatch(r"(Sat|Unsat)\(Option_([A-Z])\)", expr, flags=re.IGNORECASE)
            if m_status:
                n_parsed += 1
                label = m_status.group(2).upper()
                expected = bool(option_results.get(label) is True)
                observed = m_status.group(1).lower() == "sat"
                ok = expected == observed
                entry = {"k": k, "raw": line, "expr": expr, "validity_status": "ENTAILED" if ok else "CONTRADICTION"}
                if ok:
                    valid_steps.append(entry)
                    # option status steps are not counted novel process deductions
                else:
                    n_contra += 1
                    non_valid_steps.append(entry)
                continue

            try:
                phi = _parse_expr(expr, var_map, options)
                n_parsed += 1
            except Exception as e:
                non_valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": "PARSE_ERROR", "reason": f"{type(e).__name__}: {e}"})
                continue

            status = _status_under_gamma(base_assertions, phi, timeout_s)
            sexpr = phi.sexpr()
            if status == "ENTAILED":
                valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": status})
                if sexpr not in seen_sexprs:
                    novel_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": status})
            elif status == "CONTRADICTION":
                n_contra += 1
                non_valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": status})
            else:
                non_valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": status})
            seen_sexprs.add(sexpr)
            if status != "CONTRADICTION":
                step_assertions.append(phi)

        report.update({
            "parse_status": "AR_LSAT_ORDERING_SUCCESS",
            "base_sat": bool(base_sat),
            "base_parse_errors": parse_errors,
            "option_results": option_results,
            "selected_option_valid_by_semantics": selected_option_valid_by_semantics,
            "selected_option_matches_gt": selected_option_matches_gt,
            "base_sat_full_GT": bool(base_sat and selected_option_valid_by_semantics and selected_option_matches_gt),
            "n_positions": n_positions,
            "n_steps_total": n_total,
            "n_steps_parsed_ok": n_parsed,
            "n_steps_valid": len(valid_steps),
            "n_steps_novel_inc_clues": len(novel_steps),
            "n_non_valid_contradiction": n_contra,
            "list_steps_non_valid": non_valid_steps,
            "list_novel_steps_inc_clues": [x["expr"] for x in novel_steps],
        })
        return report

    except Exception as e:
        report["parse_status"] = "Z3_EXCEPTION"
        report["error"] = f"{type(e).__name__}: {e}"
        logger.exception("AR-LSAT ordering validator crashed")
        return report


if __name__ == "__main__":
    payload = {
        "problem_type": "ordering",
        "world_model": {"entities": ["A", "B", "C", "D"], "domains": {"positions": ["1", "2", "3", "4"]}},
        "rules": ["Distinct(A, B, C, D)", "A < B", "C == A + 1", "D != 1"],
        "facts": ["B == 4"],
        "question_semantics": {"question_type": "could_be_true"},
        "options": {"A": "A == 2", "B": "C == 4"},
        "reasoning": ["B is fixed fourth.", "S1: B == 4.", "Option A is possible.", "S2: Sat(Option_A)."],
        "solution": {"selected_option": "A"},
        "ground_truth": "A",
    }
    print(json.dumps(solve_and_validate_payload(payload), indent=2))
