#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR-LSAT Z3 validator for ordering and assignment payloads.

This is a drop-in replacement for z3_reasoning_validator_v13_gt_solve_v9.py.
It keeps the ordering validator behavior and adds support for assignment-style
payloads such as:

  Assign(A, P1)
  Not(Assign(A, P1))
  Assign(B, P1) == Assign(C, P1)
  Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))
  Sat(Option_A)
  Unsat(Not(Option_A))
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import z3
from z3 import (
    And,
    AtLeast,
    AtMost,
    Bool,
    BoolVal,
    Distinct,
    Implies,
    Int,
    Not,
    Or,
    PbEq,
    Solver,
    Xor,
    sat,
    unsat,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_FUNC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)


def normalize_header(data_sample):
    return data_sample


def normalize_months_in_rows(z3_solution: dict) -> dict:
    return z3_solution


def _norm_token(x: Any) -> str:
    s = str(x).strip().strip("`'\"“”‘’")
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _split_top_level_args(s: str) -> List[str]:
    args: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses")
            buf.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                args.append(part)
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        raise ValueError("Unbalanced parentheses")
    tail = "".join(buf).strip()
    if tail:
        args.append(tail)
    return args


def _split_top_level_binary(expr: str, op: str) -> Optional[Tuple[str, str]]:
    depth = 0
    i = 0
    while i <= len(expr) - len(op):
        ch = expr[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses")
            i += 1
            continue
        if depth == 0 and expr.startswith(op, i):
            left = expr[:i].strip()
            right = expr[i + len(op):].strip()
            if left and right:
                return left, right
        i += 1
    return None


def _selected_from_ground_truth(gt: Any) -> Optional[str]:
    if isinstance(gt, str):
        return gt.strip().upper()
    if isinstance(gt, dict):
        for k in ("answer", "selected_option", "ground_truth_option"):
            if gt.get(k):
                return str(gt[k]).strip().upper()
    return None


def _selected_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    sol = payload.get("solution") or {}
    if isinstance(sol, dict) and sol.get("selected_option"):
        return str(sol["selected_option"]).strip().upper()
    return None


# -----------------------------------------------------------------------------
# Shared solver helpers
# -----------------------------------------------------------------------------

def _solver_check(assertions: List[Any], timeout_s: float):
    s = Solver()
    s.set("timeout", int(timeout_s * 1000))
    s.add(assertions)
    return s.check()


def _is_sat(assertions: List[Any], timeout_s: float) -> bool:
    return _solver_check(assertions, timeout_s) == sat


def _is_unsat(assertions: List[Any], timeout_s: float) -> bool:
    return _solver_check(assertions, timeout_s) == unsat


def _status_under_gamma(gamma_assertions: List[Any], phi: Any, timeout_s: float) -> str:
    if not _is_sat(gamma_assertions, timeout_s):
        return "PREMISES_UNSAT"
    if _is_unsat(gamma_assertions + [phi], timeout_s):
        return "CONTRADICTION"
    if _is_unsat(gamma_assertions + [Not(phi)], timeout_s):
        return "ENTAILED"
    return "NOT_ENTAILED"


def _is_tautology(base_assertions: List[Any], phi: Any, timeout_s: float) -> bool:
    return _is_sat(base_assertions, timeout_s) and _is_unsat(base_assertions + [Not(phi)], timeout_s)


def _equiv_to_any_rule(base_assertions: List[Any], phi: Any, rule_phis: List[Any], timeout_s: float) -> bool:
    for r in rule_phis:
        if _is_unsat(base_assertions + [Xor(phi, r)], timeout_s):
            return True
    return False


def _evaluate_option(question_type: str, option_phi: Any, base_plus_rules_facts: List[Any], timeout_s: float) -> bool:
    qt = (question_type or "").strip().lower()
    if qt in {"could_be_true", "acceptability", "partial_acceptability", "valid_complete_assignment"}:
        return _is_sat(base_plus_rules_facts + [option_phi], timeout_s)
    if qt in {"must_be_true", "must_follow"}:
        return _is_unsat(base_plus_rules_facts + [Not(option_phi)], timeout_s)
    if qt in {"cannot_be_true", "must_be_false"}:
        return _is_unsat(base_plus_rules_facts + [option_phi], timeout_s)
    if qt == "could_be_false":
        return _is_sat(base_plus_rules_facts + [Not(option_phi)], timeout_s)
    return _is_sat(base_plus_rules_facts + [option_phi], timeout_s)


def _extract_step_expr(line: str) -> Optional[Tuple[int, str]]:
    m = _STEP_RE.match((line or "").strip())
    if not m:
        return None
    k = int(m.group(1))
    expr = m.group(2).strip()
    if "[" in expr:
        expr = expr.split("[", 1)[0].strip()
    if expr.endswith("."):
        expr = expr[:-1].strip()
    return (k, expr) if expr else None


def _option_status_expr(expr: str) -> Optional[Tuple[str, str]]:
    """Return (mode, option_label).

    Supported modes:
      sat_option, unsat_option, sat_not_option, unsat_not_option
    """
    m = re.fullmatch(r"\s*(Sat|Unsat)\(Option_([A-Z])\)\s*", expr, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}_option", m.group(2).upper()

    m = re.fullmatch(r"\s*(Sat|Unsat)\(\s*Not\(\s*Option_([A-Z])\s*\)\s*\)\s*", expr, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1).lower()}_not_option", m.group(2).upper()

    return None


def _check_option_status(mode: str, opt_phi: Any, gamma_valid: List[Any], timeout_s: float) -> bool:
    if mode == "sat_option":
        return _is_sat(gamma_valid + [opt_phi], timeout_s)
    if mode == "unsat_option":
        return _is_unsat(gamma_valid + [opt_phi], timeout_s)
    if mode == "sat_not_option":
        return _is_sat(gamma_valid + [Not(opt_phi)], timeout_s)
    if mode == "unsat_not_option":
        return _is_unsat(gamma_valid + [Not(opt_phi)], timeout_s)
    return False


def _parse_constraints(lines: List[str], parser_fn: Callable[[str], Any]) -> Tuple[List[Any], List[Dict[str, str]]]:
    phis: List[Any] = []
    errors: List[Dict[str, str]] = []
    for raw in lines or []:
        try:
            phis.append(parser_fn(str(raw)))
        except Exception as e:
            errors.append({"raw": str(raw), "error": f"{type(e).__name__}: {e}"})
    return phis, errors


def _parse_options(options: Dict[str, str], parser_fn: Callable[[str], Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    out: Dict[str, Any] = {}
    errs: List[Dict[str, str]] = []
    for label, expr in (options or {}).items():
        lab = str(label).strip().upper()
        try:
            out[lab] = parser_fn(str(expr))
        except Exception as e:
            errs.append({"label": lab, "raw": str(expr), "error": f"{type(e).__name__}: {e}"})
    return out, errs


def _validate_reasoning_steps(
    reasoning: List[str],
    *,
    parser_fn: Callable[[str], Any],
    base_assertions: List[Any],
    rule_fact_phis: List[Any],
    rule_phis: List[Any],
    option_phis: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    gamma_valid = base_assertions + rule_fact_phis
    gamma_steps = list(base_assertions)
    seen = set()

    n_total = 0
    n_parsed = 0
    valid_steps: List[Dict[str, Any]] = []
    novel_steps: List[Dict[str, Any]] = []
    non_valid: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []

    for line in reasoning or []:
        parsed = _extract_step_expr(line)
        if parsed is None:
            continue
        n_total += 1
        k, expr = parsed

        status_expr = _option_status_expr(expr)
        if status_expr:
            n_parsed += 1
            mode, label = status_expr
            opt_phi = option_phis.get(label)
            if opt_phi is None:
                non_valid.append({"k": k, "raw": line, "expr": expr, "validity_status": "UNKNOWN_OPTION"})
                continue
            option_valid = _check_option_status(mode, opt_phi, gamma_valid, timeout_s)
            if option_valid:
                valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": "OPTION_STATUS_VALID"})
            else:
                non_valid.append({"k": k, "raw": line, "expr": expr, "validity_status": "OPTION_STATUS_INVALID"})
            continue

        try:
            phi = parser_fn(expr)
            n_parsed += 1
        except Exception as e:
            entry = {"k": k, "raw": line, "expr": expr, "status": "PARSE_ERROR", "error": f"{type(e).__name__}: {e}"}
            parse_errors.append(entry)
            non_valid.append({"k": k, "raw": line, "expr": expr, "validity_status": "PARSE_ERROR", "reason": entry["error"]})
            continue

        sexpr = phi.sexpr()
        if sexpr in seen:
            continue
        seen.add(sexpr)

        if _is_tautology(base_assertions, phi, timeout_s):
            valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": "TAUTOLOGY"})
            continue
        if _equiv_to_any_rule(base_assertions, phi, rule_phis, timeout_s):
            valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": "RESTATES_RULE"})
            continue

        validity = _status_under_gamma(gamma_valid, phi, timeout_s)
        if validity == "ENTAILED":
            valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": validity})
        else:
            non_valid.append({"k": k, "raw": line, "expr": expr, "validity_status": validity})

        step_status = _status_under_gamma(gamma_steps, phi, timeout_s)
        if step_status == "CONTRADICTION":
            continue
        is_novel = step_status != "ENTAILED"
        if validity == "ENTAILED" and is_novel:
            novel_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": validity, "steps_status": step_status})
        gamma_steps.append(phi)

    consistency_score = (len(valid_steps) / max(n_total, 1)) if n_total else 0.0

    return {
        "n_steps_total": n_total,
        "n_steps_parsed_ok": n_parsed,
        "n_steps_valid": len(valid_steps),
        "n_steps_novel_inc_clues": len(novel_steps),
        "n_non_valid_contradiction": len([x for x in non_valid if x.get("validity_status") == "CONTRADICTION"]),
        "consistency_score": float(consistency_score),
        "list_steps_valid": [x.get("expr") for x in valid_steps],
        "list_steps_non_valid": non_valid,
        "list_novel_steps_inc_clues": [x.get("expr") for x in novel_steps],
        "list_step_parse_errors": parse_errors,
    }


# -----------------------------------------------------------------------------
# Ordering support, preserved from your original validator
# -----------------------------------------------------------------------------

def _extract_entities_positions(world_model: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    entities = [_norm_token(e) for e in (world_model.get("entities") or [])]
    domains = world_model.get("domains") or {}
    raw_positions = domains.get("positions") or domains.get("position") or []
    positions: List[int] = []
    for p in raw_positions:
        try:
            positions.append(int(str(p).strip()))
        except Exception:
            pass
    if not positions and entities:
        positions = list(range(1, len(entities) + 1))
    return entities, sorted(set(positions))


def _make_ordering_base(world_model: Dict[str, Any], timeout_s: float) -> Tuple[Solver, Dict[str, Any], List[Any], int, int]:
    entities, positions = _extract_entities_positions(world_model)
    if not entities:
        raise ValueError("world_model.entities is empty")
    if not positions:
        raise ValueError("world_model.domains.positions is empty")

    var_map: Dict[str, Any] = {}
    for e in entities:
        if e in var_map:
            raise ValueError(f"Duplicate entity after normalization: {e}")
        var_map[e] = Int(e)

    lo, hi = min(positions), max(positions)
    base_assertions: List[Any] = [And(v >= lo, v <= hi) for v in var_map.values()]
    if len(var_map) > 1:
        base_assertions.append(Distinct(*list(var_map.values())))

    s = Solver()
    s.set("timeout", int(timeout_s * 1000))
    s.add(base_assertions)
    return s, var_map, base_assertions, len(positions), len(entities)


def _ordering_term(raw: str, var_map: Dict[str, Any]):
    r = raw.strip()
    if re.fullmatch(r"-?\d+", r):
        return int(r)
    tok = _norm_token(r)
    if tok not in var_map:
        raise KeyError(f"Unknown token: {tok!r}")
    return var_map[tok]


def _parse_ordering_atomic(expr: str, var_map: Dict[str, Any]):
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1].strip()

    m = re.match(r"^(.+?)\s*\+\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        return _ordering_term(m.group(1), var_map) + int(m.group(2)) == _ordering_term(m.group(3), var_map)

    m = re.match(r"^(.+?)\s*\-\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        return _ordering_term(m.group(1), var_map) - int(m.group(2)) == _ordering_term(m.group(3), var_map)

    for op in ("<=", ">=", "==", "!=", "<", ">"):
        split = _split_top_level_binary(e, op)
        if split:
            left, right = split
            L, R = _ordering_term(left, var_map), _ordering_term(right, var_map)
            if op == "==":
                return L == R
            if op == "!=":
                return L != R
            if op == "<":
                return L < R
            if op == ">":
                return L > R
            if op == "<=":
                return L <= R
            if op == ">=":
                return L >= R
    raise ValueError(f"Unrecognized atomic expression: {expr!r}")


def _parse_ordering_expr(expr: str, var_map: Dict[str, Any]):
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1].strip()

    m = _FUNC_RE.match(e)
    if m:
        fn = m.group(1).lower()
        args = _split_top_level_args(m.group(2))
        if fn == "distinct":
            return Distinct(*[_ordering_term(a, var_map) for a in args])
        parsed = [_parse_ordering_expr(a, var_map) for a in args]
        if fn == "and":
            return And(*parsed)
        if fn == "or":
            return Or(*parsed)
        if fn == "not":
            if len(parsed) != 1:
                raise ValueError("Not expects one argument")
            return Not(parsed[0])
        if fn == "implies":
            if len(parsed) != 2:
                raise ValueError("Implies expects two arguments")
            return Implies(parsed[0], parsed[1])
        if fn == "xor":
            if len(parsed) != 2:
                raise ValueError("Xor expects two arguments")
            return Xor(parsed[0], parsed[1])
        raise ValueError(f"Unsupported function: {fn}")
    return _parse_ordering_atomic(e, var_map)


def _solve_ordering_payload(payload: Dict[str, Any], *, timeout_s: float) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "base_sat_full_GT": False,
        "parse_status": "INIT",
        "n_steps_total": 0,
        "n_steps_parsed_ok": 0,
        "n_steps_valid": 0,
        "n_steps_novel_inc_clues": 0,
        "n_non_valid_contradiction": 0,
        "consistency_score": 0.0,
    }
    try:
        _, var_map, base_assertions, n_positions, n_entities = _make_ordering_base(payload.get("world_model") or {}, timeout_s)
        parser_fn = lambda s: _parse_ordering_expr(s, var_map)
        rule_phis, rule_errors = _parse_constraints(payload.get("rules") or [], parser_fn)
        fact_phis, fact_errors = _parse_constraints(payload.get("facts") or [], parser_fn)
        option_phis, option_errors_list = _parse_options(payload.get("options") or {}, parser_fn)

        rule_fact_phis = rule_phis + fact_phis
        base_plus_rules_facts = base_assertions + rule_fact_phis
        base_sat = _is_sat(base_plus_rules_facts, timeout_s)

        selected = _selected_from_payload(payload)
        gt = _selected_from_ground_truth(payload.get("ground_truth"))
        question_type = ((payload.get("question_semantics") or {}).get("question_type") or payload.get("question_type") or "could_be_true")
        selected_phi = option_phis.get(selected or "")
        solver_selected_ok = bool(selected_phi is not None and base_sat and _evaluate_option(question_type, selected_phi, base_plus_rules_facts, timeout_s))
        gt_match = bool(selected and gt and selected == gt)

        report.update({
            "base_sat_full_GT": bool(base_sat and solver_selected_ok and gt_match),
            "base_sat": bool(base_sat),
            "solver_selected_ok": bool(solver_selected_ok),
            "selected_option": selected,
            "ground_truth_option": gt,
            "question_type": question_type,
            "rule_parse_errors": rule_errors,
            "fact_parse_errors": fact_errors,
            "option_parse_errors": option_errors_list,
            "n_positions": n_positions,
            "n_entities": n_entities,
        })

        step_report = _validate_reasoning_steps(
            payload.get("reasoning") or [],
            parser_fn=parser_fn,
            base_assertions=base_assertions,
            rule_fact_phis=rule_fact_phis,
            rule_phis=rule_phis,
            option_phis=option_phis,
            timeout_s=timeout_s,
        )
        report.update(step_report)
        report["parse_status"] = "AR_LSAT_ORDERING_SUCCESS"
        return report
    except Exception as e:
        report["parse_status"] = "Z3_EXCEPTION"
        report["error"] = f"{type(e).__name__}: {e}"
        return report


# -----------------------------------------------------------------------------
# Assignment support
# -----------------------------------------------------------------------------

def _extract_entities_values(world_model: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    entities = [_norm_token(e) for e in (world_model.get("entities") or [])]
    domains = world_model.get("domains") or {}

    raw_values = []
    for key in ("values", "assignments", "projects", "rooms", "days", "colors", "groups", "slots", "tasks", "courses", "teams"):
        if isinstance(domains.get(key), list) and domains.get(key):
            raw_values = domains.get(key)
            break

    if not raw_values and isinstance(domains, dict):
        # Fallback: use the first non-empty list that is not simply the entities list.
        ent_set = set(entities)
        for v in domains.values():
            if isinstance(v, list) and v:
                candidate = [_norm_token(x) for x in v]
                if set(candidate) != ent_set:
                    raw_values = v
                    break

    values = [_norm_token(v) for v in raw_values]
    return entities, values


def _assignment_value_exactly_one_required(world_model: Dict[str, Any]) -> bool:
    assumptions = world_model.get("structural_assumptions") or []
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    text = " ".join(str(x).lower() for x in assumptions)
    return bool(
        "one-to-one" in text
        or "bijective" in text
        or re.search(r"(each|every)\s+(value|project|room|day|slot|task|course|team|group)\b.*\bexactly\s+one\b", text)
    )


def _make_assignment_base(world_model: Dict[str, Any], timeout_s: float) -> Tuple[Solver, Dict[Tuple[str, str], Any], List[Any], int, int]:
    entities, values = _extract_entities_values(world_model)
    if not entities:
        raise ValueError("world_model.entities is empty")
    if not values:
        raise ValueError("world_model.domains.values/assignments/projects/etc. is empty")

    assign_vars: Dict[Tuple[str, str], Any] = {}
    for e in entities:
        for v in values:
            assign_vars[(e, v)] = Bool(f"Assign__{e}__{v}")

    base_assertions: List[Any] = []

    # Default LSAT assignment assumption: every entity receives exactly one value.
    for e in entities:
        base_assertions.append(PbEq([(assign_vars[(e, v)], 1) for v in values], 1))

    # Only add reverse uniqueness when explicitly requested; many LSAT grouping/assignment
    # tasks allow multiple entities to share the same value.
    if _assignment_value_exactly_one_required(world_model):
        for v in values:
            base_assertions.append(PbEq([(assign_vars[(e, v)], 1) for e in entities], 1))

    s = Solver()
    s.set("timeout", int(timeout_s * 1000))
    s.add(base_assertions)
    return s, assign_vars, base_assertions, len(values), len(entities)


def _assignment_var(entity_raw: str, value_raw: str, assign_vars: Dict[Tuple[str, str], Any]):
    e = _norm_token(entity_raw)
    v = _norm_token(value_raw)
    key = (e, v)
    if key not in assign_vars:
        known_entities = sorted({x for x, _ in assign_vars.keys()})
        known_values = sorted({y for _, y in assign_vars.keys()})
        raise KeyError(f"Unknown Assign({e}, {v}). Known entities={known_entities}; known values={known_values}")
    return assign_vars[key]


def _parse_assignment_expr(expr: str, assign_vars: Dict[Tuple[str, str], Any]):
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1].strip()

    if e.lower() == "true":
        return BoolVal(True)
    if e.lower() == "false":
        return BoolVal(False)

    # Top-level Boolean equality/inequality, e.g. Assign(B,P1) == Assign(C,P1).
    for op in ("==", "!="):
        split = _split_top_level_binary(e, op)
        if split:
            left, right = split
            L = _parse_assignment_expr(left, assign_vars)
            R = _parse_assignment_expr(right, assign_vars)
            return (L == R) if op == "==" else (L != R)

    m = _FUNC_RE.match(e)
    if not m:
        raise ValueError(f"Unrecognized assignment expression: {expr!r}")

    fn = m.group(1).lower()
    args = _split_top_level_args(m.group(2))

    if fn == "assign":
        if len(args) != 2:
            raise ValueError("Assign expects exactly two arguments: Assign(entity, value)")
        return _assignment_var(args[0], args[1], assign_vars)

    if fn == "not":
        if len(args) != 1:
            raise ValueError("Not expects one argument")
        return Not(_parse_assignment_expr(args[0], assign_vars))

    if fn == "and":
        return And(*[_parse_assignment_expr(a, assign_vars) for a in args])
    if fn == "or":
        return Or(*[_parse_assignment_expr(a, assign_vars) for a in args])
    if fn == "implies":
        if len(args) != 2:
            raise ValueError("Implies expects two arguments")
        return Implies(_parse_assignment_expr(args[0], assign_vars), _parse_assignment_expr(args[1], assign_vars))
    if fn == "xor":
        if len(args) != 2:
            raise ValueError("Xor expects two arguments")
        return Xor(_parse_assignment_expr(args[0], assign_vars), _parse_assignment_expr(args[1], assign_vars))

    if fn in {"exactly", "atleast", "atmost"}:
        if len(args) < 2:
            raise ValueError(f"{fn} expects a count plus one or more Boolean arguments")
        try:
            k = int(str(args[0]).strip())
        except Exception:
            raise ValueError(f"First argument to {fn} must be an integer")
        parsed = [_parse_assignment_expr(a, assign_vars) for a in args[1:]]
        if fn == "exactly":
            return PbEq([(p, 1) for p in parsed], k)
        if fn == "atleast":
            return AtLeast(*parsed, k)
        if fn == "atmost":
            return AtMost(*parsed, k)

    raise ValueError(f"Unsupported assignment function: {fn}")


def _solve_assignment_payload(payload: Dict[str, Any], *, timeout_s: float) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "base_sat_full_GT": False,
        "parse_status": "INIT",
        "n_steps_total": 0,
        "n_steps_parsed_ok": 0,
        "n_steps_valid": 0,
        "n_steps_novel_inc_clues": 0,
        "n_non_valid_contradiction": 0,
        "consistency_score": 0.0,
    }
    try:
        _, assign_vars, base_assertions, n_values, n_entities = _make_assignment_base(payload.get("world_model") or {}, timeout_s)
        parser_fn = lambda s: _parse_assignment_expr(s, assign_vars)
        rule_phis, rule_errors = _parse_constraints(payload.get("rules") or [], parser_fn)
        fact_phis, fact_errors = _parse_constraints(payload.get("facts") or [], parser_fn)
        option_phis, option_errors_list = _parse_options(payload.get("options") or {}, parser_fn)

        rule_fact_phis = rule_phis + fact_phis
        base_plus_rules_facts = base_assertions + rule_fact_phis
        base_sat = _is_sat(base_plus_rules_facts, timeout_s)

        selected = _selected_from_payload(payload)
        gt = _selected_from_ground_truth(payload.get("ground_truth"))
        question_type = ((payload.get("question_semantics") or {}).get("question_type") or payload.get("question_type") or "could_be_true")
        selected_phi = option_phis.get(selected or "")
        solver_selected_ok = bool(selected_phi is not None and base_sat and _evaluate_option(question_type, selected_phi, base_plus_rules_facts, timeout_s))
        gt_match = bool(selected and gt and selected == gt)

        report.update({
            "base_sat_full_GT": bool(base_sat and solver_selected_ok and gt_match),
            "base_sat": bool(base_sat),
            "solver_selected_ok": bool(solver_selected_ok),
            "selected_option": selected,
            "ground_truth_option": gt,
            "question_type": question_type,
            "rule_parse_errors": rule_errors,
            "fact_parse_errors": fact_errors,
            "option_parse_errors": option_errors_list,
            "n_values": n_values,
            "n_entities": n_entities,
        })

        step_report = _validate_reasoning_steps(
            payload.get("reasoning") or [],
            parser_fn=parser_fn,
            base_assertions=base_assertions,
            rule_fact_phis=rule_fact_phis,
            rule_phis=rule_phis,
            option_phis=option_phis,
            timeout_s=timeout_s,
        )
        report.update(step_report)
        report["parse_status"] = "AR_LSAT_ASSIGNMENT_SUCCESS"
        return report
    except Exception as e:
        report["parse_status"] = "Z3_EXCEPTION"
        report["error"] = f"{type(e).__name__}: {e}"
        return report


# -----------------------------------------------------------------------------
# Public entry point used by reward file
# -----------------------------------------------------------------------------

def solve_and_validate_payload(payload: Dict[str, Any], *, timeout_s: float = 2.0, conflict_tolerant_clues: bool = False) -> Dict[str, Any]:
    problem_type = str(payload.get("problem_type") or "").strip().lower()
    if problem_type == "ordering":
        return _solve_ordering_payload(payload, timeout_s=timeout_s)
    if problem_type == "assignment":
        return _solve_assignment_payload(payload, timeout_s=timeout_s)

    return {
        "base_sat_full_GT": False,
        "parse_status": "Z3_EXCEPTION",
        "error": f"Unsupported problem_type={payload.get('problem_type')!r}. Expected 'ordering' or 'assignment'.",
        "n_steps_total": 0,
        "n_steps_parsed_ok": 0,
        "n_steps_valid": 0,
        "n_steps_novel_inc_clues": 0,
        "n_non_valid_contradiction": 0,
        "consistency_score": 0.0,
    }


if __name__ == "__main__":
    sample = {
        "problem_type": "assignment",
        "world_model": {
            "entities": ["A", "B", "C"],
            "domains": {"values": ["P1", "P2", "P3"]},
            "structural_assumptions": ["each entity is assigned exactly one value"],
        },
        "rules": [
            "Not(Assign(A, P1))",
            "Assign(B, P1) == Assign(C, P1)",
            "Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))",
        ],
        "facts": [],
        "question_semantics": {"question_type": "could_be_true"},
        "options": {"A": "Assign(A, P2)", "B": "Assign(A, P1)", "C": "Assign(B, P2)"},
        "reasoning": [
            "A is not assigned to P1 by the first rule.",
            "S1: Not(Assign(A, P1)).",
            "Exactly one employee is assigned to P2.",
            "S2: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2)).",
            "Option A can be extended to a full valid assignment.",
            "S3: Sat(Option_A).",
        ],
        "solution": {"selected_option": "A"},
        "ground_truth": "A",
    }
    print(json.dumps(solve_and_validate_payload(sample), indent=2, ensure_ascii=False, default=str))
