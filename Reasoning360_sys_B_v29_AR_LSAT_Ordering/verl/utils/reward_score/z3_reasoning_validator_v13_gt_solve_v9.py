#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR-LSAT ORDERING Z3 validator.

Fixes included:
- supports ordering only: problem_type == "ordering"
- supports Sat(Option_A), Unsat(Option_A), Sat(Not(Option_A)), Unsat(Not(Option_A))
- supports counting operators AtLeast/AtMost/Exactly for Boolean expressions
- reports diagnostic parse counts so low BASE_sat_full_GT can be debugged
"""
from __future__ import annotations

import re
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import z3
    from z3 import And, Or, Not, Implies, Xor, Distinct, Int, Solver, AtLeast, AtMost, sat, unsat
except Exception:  # pragma: no cover
    z3 = None

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
    args, buf, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1; buf.append(ch)
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


def _selected_from_ground_truth(gt: Any) -> Optional[str]:
    if isinstance(gt, str):
        return gt.strip().upper()
    if isinstance(gt, dict):
        for k in ("answer", "selected_option", "ground_truth_option"):
            if gt.get(k) is not None:
                return str(gt[k]).strip().upper()
    return None


def _selected_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    sol = payload.get("solution") or {}
    if isinstance(sol, dict) and sol.get("selected_option") is not None:
        return str(sol["selected_option"]).strip().upper()
    return None


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


def _make_base(world_model: Dict[str, Any], timeout_s: float):
    if z3 is None:
        raise RuntimeError("z3-solver is not installed")
    entities, positions = _extract_entities_positions(world_model)
    if not entities:
        raise ValueError("world_model.entities is empty")
    if not positions:
        raise ValueError("world_model.domains.positions is empty")
    var_map: Dict[str, Any] = {e: Int(e) for e in entities}
    if len(var_map) != len(entities):
        raise ValueError("Duplicate entity after normalization")
    lo, hi = min(positions), max(positions)
    base_assertions: List[Any] = [And(v >= lo, v <= hi) for v in var_map.values()]
    if len(var_map) > 1:
        base_assertions.append(Distinct(*list(var_map.values())))
    return var_map, base_assertions, len(positions), len(entities)


def _term(raw: str, var_map: Dict[str, Any]):
    r = raw.strip()
    if re.fullmatch(r"-?\d+", r):
        return int(r)
    tok = _norm_token(r)
    if tok not in var_map:
        raise KeyError(f"Unknown token: {tok!r}")
    return var_map[tok]


def _parse_atomic(expr: str, var_map: Dict[str, Any]):
    e = str(expr).strip().rstrip(".")
    # A + d == B / A - d == B
    m = re.match(r"^(.+?)\s*\+\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        return _term(m.group(1), var_map) + int(m.group(2)) == _term(m.group(3), var_map)
    m = re.match(r"^(.+?)\s*\-\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        return _term(m.group(1), var_map) - int(m.group(2)) == _term(m.group(3), var_map)
    for op in ("<=", ">=", "==", "!=", "<", ">"):
        if op in e:
            left, right = e.split(op, 1)
            L, R = _term(left, var_map), _term(right, var_map)
            return {"==": L == R, "!=": L != R, "<": L < R, ">": L > R, "<=": L <= R, ">=": L >= R}[op]
    raise ValueError(f"Unrecognized atomic expression: {expr!r}")


def _parse_expr(expr: str, var_map: Dict[str, Any]):
    e = str(expr).strip().rstrip(".")
    m = _FUNC_RE.match(e)
    if m:
        fn = m.group(1).lower()
        args = _split_top_level_args(m.group(2))
        if fn == "distinct":
            return Distinct(*[_term(a, var_map) for a in args])
        if fn in {"atleast", "atmost", "exactly"}:
            if len(args) < 2:
                raise ValueError(f"{fn} expects k and Boolean expressions")
            k = int(str(args[0]).strip())
            parsed = [_parse_expr(a, var_map) for a in args[1:]]
            if fn == "atleast":
                return AtLeast(*parsed, k)
            if fn == "atmost":
                return AtMost(*parsed, k)
            return And(AtLeast(*parsed, k), AtMost(*parsed, k))
        parsed = [_parse_expr(a, var_map) for a in args]
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
    return _parse_atomic(e, var_map)


def _parse_constraints(lines: List[str], var_map: Dict[str, Any]):
    phis, errors = [], []
    for raw in lines or []:
        try:
            phis.append(_parse_expr(str(raw), var_map))
        except Exception as e:
            errors.append({"raw": str(raw), "error": f"{type(e).__name__}: {e}"})
    return phis, errors


def _parse_options(options: Dict[str, str], var_map: Dict[str, Any]):
    out, errs = {}, []
    for label, expr in (options or {}).items():
        lab = str(label).strip().upper().rstrip(".")
        try:
            out[lab] = _parse_expr(str(expr), var_map)
        except Exception as e:
            errs.append({"label": lab, "raw": str(expr), "error": f"{type(e).__name__}: {e}"})
    return out, errs


def _solver_check(assertions: List[Any], timeout_s: float):
    s = Solver(); s.set("timeout", int(timeout_s * 1000)); s.add(assertions); return s.check()


def _is_sat(assertions: List[Any], timeout_s: float) -> bool:
    return _solver_check(assertions, timeout_s) == sat


def _is_unsat(assertions: List[Any], timeout_s: float) -> bool:
    return _solver_check(assertions, timeout_s) == unsat


def _evaluate_option(question_type: str, option_phi: Any, gamma: List[Any], timeout_s: float) -> bool:
    qt = (question_type or "").strip().lower()
    if qt in {"could_be_true", "acceptability", "partial_acceptability", "valid_complete_assignment"}:
        return _is_sat(gamma + [option_phi], timeout_s)
    if qt in {"must_be_true", "must_follow"}:
        return _is_unsat(gamma + [Not(option_phi)], timeout_s)
    if qt in {"cannot_be_true", "must_be_false"}:
        return _is_unsat(gamma + [option_phi], timeout_s)
    if qt == "could_be_false":
        return _is_sat(gamma + [Not(option_phi)], timeout_s)
    return _is_sat(gamma + [option_phi], timeout_s)


def _extract_step_expr(line: str) -> Optional[Tuple[int, str]]:
    m = _STEP_RE.match((line or "").strip())
    if not m:
        return None
    expr = m.group(2).strip()
    if "[" in expr:
        expr = expr.split("[", 1)[0].strip()
    expr = expr.rstrip(".").strip()
    return (int(m.group(1)), expr) if expr else None


def _option_status_expr(expr: str) -> Optional[Tuple[str, bool, str]]:
    e = (expr or "").strip().rstrip(".")
    m = re.fullmatch(r"\s*(Sat|Unsat)\(\s*(Not\()?\s*Option_([A-Z])\s*\)?\s*\)\s*", e, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower(), bool(m.group(2)), m.group(3).upper()


def _option_status_is_true(status: str, is_negated: bool, option_phi: Any, gamma: List[Any], timeout_s: float) -> bool:
    phi = Not(option_phi) if is_negated else option_phi
    if status == "sat":
        return _is_sat(gamma + [phi], timeout_s)
    if status == "unsat":
        return _is_unsat(gamma + [phi], timeout_s)
    return False


def _supports_selected_solution(question_type: str, status: str, is_negated: bool, label: str, selected: Optional[str]) -> bool:
    if not selected or label != selected:
        return False
    qt = (question_type or "").strip().lower()
    if qt in {"could_be_true", "acceptability", "partial_acceptability", "valid_complete_assignment"}:
        return status == "sat" and not is_negated
    if qt in {"must_be_true", "must_follow"}:
        return status == "unsat" and is_negated
    if qt in {"cannot_be_true", "must_be_false"}:
        return status == "unsat" and not is_negated
    if qt == "could_be_false":
        return status == "sat" and is_negated
    return status == "sat" and not is_negated


def _status_under_gamma(gamma: List[Any], phi: Any, timeout_s: float) -> str:
    if not _is_sat(gamma, timeout_s):
        return "PREMISES_UNSAT"
    if _is_unsat(gamma + [phi], timeout_s):
        return "CONTRADICTION"
    if _is_unsat(gamma + [Not(phi)], timeout_s):
        return "ENTAILED"
    return "NOT_ENTAILED"


def _validate_reasoning_steps(reasoning: List[str], *, var_map: Dict[str, Any], base_assertions: List[Any], rule_fact_phis: List[Any], rule_phis: List[Any], option_phis: Dict[str, Any], question_type: str, selected_option: Optional[str], timeout_s: float) -> Dict[str, Any]:
    gamma_valid = base_assertions + rule_fact_phis
    gamma_steps = list(base_assertions)
    n_total = n_parsed = 0
    valid_steps, novel_steps, non_valid, parse_errors, solution_support_steps = [], [], [], [], []
    seen = set()
    for line in reasoning or []:
        parsed = _extract_step_expr(line)
        if parsed is None:
            continue
        n_total += 1
        k, expr = parsed
        status_expr = _option_status_expr(expr)
        if status_expr:
            n_parsed += 1
            status, is_negated, label = status_expr
            opt_phi = option_phis.get(label)
            if opt_phi is None:
                non_valid.append({"k": k, "raw": line, "expr": expr, "validity_status": "UNKNOWN_OPTION"})
                continue
            option_valid = _option_status_is_true(status, is_negated, opt_phi, gamma_valid, timeout_s)
            supports = bool(option_valid and _supports_selected_solution(question_type, status, is_negated, label, selected_option))
            entry = {"k": k, "raw": line, "expr": expr, "validity_status": "OPTION_STATUS_VALID" if option_valid else "OPTION_STATUS_INVALID", "supports_selected_solution": supports}
            (valid_steps if option_valid else non_valid).append(entry)
            if supports:
                solution_support_steps.append(entry)
            continue
        try:
            phi = _parse_expr(expr, var_map)
            n_parsed += 1
        except Exception as e:
            parse_errors.append({"k": k, "raw": line, "expr": expr, "error": f"{type(e).__name__}: {e}"})
            non_valid.append({"k": k, "raw": line, "expr": expr, "validity_status": "PARSE_ERROR"})
            continue
        sx = phi.sexpr()
        if sx in seen:
            continue
        seen.add(sx)
        validity = _status_under_gamma(gamma_valid, phi, timeout_s)
        if validity == "ENTAILED":
            valid_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": validity})
        else:
            non_valid.append({"k": k, "raw": line, "expr": expr, "validity_status": validity})
        step_status = _status_under_gamma(gamma_steps, phi, timeout_s)
        if step_status != "CONTRADICTION":
            if validity == "ENTAILED" and step_status != "ENTAILED":
                novel_steps.append({"k": k, "raw": line, "expr": expr, "validity_status": validity, "steps_status": step_status})
            gamma_steps.append(phi)
    return {
        "n_steps_total": n_total,
        "n_steps_parsed_ok": n_parsed,
        "n_steps_valid": len(valid_steps),
        "n_steps_novel_inc_clues": len(novel_steps),
        "n_non_valid_contradiction": len([x for x in non_valid if x.get("validity_status") == "CONTRADICTION"]),
        "list_steps_valid": [x.get("expr") for x in valid_steps],
        "list_steps_non_valid": non_valid,
        "list_novel_steps_inc_clues": [x.get("expr") for x in novel_steps],
        "list_step_parse_errors": parse_errors,
        "consistency_score": 1.0 if solution_support_steps else 0.0,
        "solution_support_steps": solution_support_steps,
    }


def solve_and_validate_payload(payload: Dict[str, Any], *, timeout_s: float = 2.0, conflict_tolerant_clues: bool = False) -> Dict[str, Any]:
    report: Dict[str, Any] = {"base_sat_full_GT": False, "parse_status": "INIT", "n_steps_total": 0, "n_steps_parsed_ok": 0, "n_steps_valid": 0, "n_steps_novel_inc_clues": 0, "n_non_valid_contradiction": 0, "consistency_score": 0.0, "solution_support_steps": []}
    try:
        if payload.get("problem_type") != "ordering":
            raise ValueError("This validator only supports problem_type='ordering'.")
        var_map, base_assertions, n_positions, n_entities = _make_base(payload.get("world_model") or {}, timeout_s)
        rule_phis, rule_errors = _parse_constraints(payload.get("rules") or [], var_map)
        fact_phis, fact_errors = _parse_constraints(payload.get("facts") or [], var_map)
        option_phis, option_errors = _parse_options(payload.get("options") or {}, var_map)
        rule_fact_phis = rule_phis + fact_phis
        gamma = base_assertions + rule_fact_phis
        base_sat = _is_sat(gamma, timeout_s)
        selected = _selected_from_payload(payload)
        gt = _selected_from_ground_truth(payload.get("ground_truth"))
        question_type = ((payload.get("question_semantics") or {}).get("question_type") or payload.get("question_type") or "could_be_true")
        selected_phi = option_phis.get(selected or "")
        solver_selected_ok = bool(selected_phi is not None and base_sat and _evaluate_option(question_type, selected_phi, gamma, timeout_s))
        gt_match = bool(selected and gt and selected == gt)
        report.update({
            "base_sat_full_GT": bool(base_sat and solver_selected_ok and gt_match),
            "base_sat": bool(base_sat),
            "solver_selected_ok": bool(solver_selected_ok),
            "gt_match": bool(gt_match),
            "selected_option": selected,
            "ground_truth_option": gt,
            "question_type": question_type,
            "rule_parse_errors": rule_errors,
            "fact_parse_errors": fact_errors,
            "option_parse_errors": option_errors,
            "n_rule_parse_errors": len(rule_errors),
            "n_fact_parse_errors": len(fact_errors),
            "n_option_parse_errors": len(option_errors),
            "selected_option_parse_ok": bool(selected_phi is not None),
            "n_positions": n_positions,
            "n_entities": n_entities,
        })
        step_report = _validate_reasoning_steps(payload.get("reasoning") or [], var_map=var_map, base_assertions=base_assertions, rule_fact_phis=rule_fact_phis, rule_phis=rule_phis, option_phis=option_phis, question_type=question_type, selected_option=selected, timeout_s=timeout_s)
        report.update(step_report)
        report["parse_status"] = "AR_LSAT_ORDERING_SUCCESS"
        return report
    except Exception as e:
        report["parse_status"] = "Z3_EXCEPTION"
        report["error"] = f"{type(e).__name__}: {e}"
        return report
