#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AR-LSAT ORDERING validator.

This replaces the ZebraPuzzle-specific payload format with the AR-LSAT format:
{
  "problem_type": "ordering",
  "world_model": {"entities": [...], "domains": {"positions": [...]}, ...},
  "rules": [...],
  "facts": [...],
  "question_semantics": {"question_type": "..."},
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
from typing import Any, Dict, List, Optional, Tuple

import z3
from z3 import (
    And as Z3And,
    BoolRef,
    Distinct,
    Implies as Z3Implies,
    Int,
    Not as Z3Not,
    Or as Z3Or,
    Solver,
    Xor as Z3Xor,
    sat,
    unsat,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_FUNC_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.DOTALL)
_TOKEN_RE = re.compile(r"^[A-Za-z_]\w*$")


def normalize_header(data_sample):
    """Compatibility stub retained for imports from older reward code."""
    if not isinstance(data_sample, dict):
        return data_sample
    header = data_sample.get("header", [])
    sports_aliases = {"FavoriteSports", "Sports", "FavoriteSport"}
    data_sample["header"] = ["Sport" if h in sports_aliases else h for h in header]
    return data_sample


def normalize_months_in_rows(z3_solution: dict) -> dict:
    """Compatibility stub retained for imports from older reward code."""
    return z3_solution


def _norm_token(x: Any) -> str:
    s = str(x).strip().strip("`'\"“”‘’")
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _split_top_level_args(s: str) -> List[str]:
    args: List[str] = []
    depth = 0
    buf: List[str] = []
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


def _extract_entities_and_positions(world_model: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    entities = [_norm_token(e) for e in world_model.get("entities", [])]
    domains = world_model.get("domains", {}) or {}
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


def _make_ordering_base(world_model: Dict[str, Any], timeout_s: float) -> Tuple[Solver, Dict[str, Any], List[BoolRef]]:
    entities, positions = _extract_entities_and_positions(world_model)
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
    base_assertions: List[BoolRef] = []
    for z in var_map.values():
        base_assertions.append(Z3And(z >= lo, z <= hi))
    if len(var_map) > 1:
        base_assertions.append(Distinct(*list(var_map.values())))

    s = Solver()
    s.set("timeout", int(float(timeout_s) * 1000))
    s.add(base_assertions)
    return s, var_map, base_assertions


def _term(raw: str, var_map: Dict[str, Any]):
    r = raw.strip()
    if re.fullmatch(r"-?\d+", r):
        return int(r)
    tok = _norm_token(r)
    if tok not in var_map:
        raise KeyError(f"Unknown token: {tok!r}")
    return var_map[tok]


def _parse_atomic(expr: str, var_map: Dict[str, Any]):
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1].strip()

    # A + k == B
    m = re.match(r"^(.+?)\s*\+\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        return _term(m.group(1), var_map) + int(m.group(2)) == _term(m.group(3), var_map)

    # A - k == B
    m = re.match(r"^(.+?)\s*\-\s*(-?\d+)\s*==\s*(.+?)$", e)
    if m:
        return _term(m.group(1), var_map) - int(m.group(2)) == _term(m.group(3), var_map)

    for op in ("<=", ">=", "==", "!=", "<", ">"):
        # split once at top level; atoms do not contain nested ops except handled above
        if op in e:
            left, right = e.split(op, 1)
            L, R = _term(left, var_map), _term(right, var_map)
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

    raise ValueError(f"Unrecognized atomic ordering expression: {expr!r}")


def _parse_expr(expr: str, var_map: Dict[str, Any]):
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1].strip()

    # unwrap harmless outer parentheses
    while e.startswith("(") and e.endswith(")"):
        inner = e[1:-1].strip()
        try:
            _split_top_level_args(inner)
            e = inner
        except Exception:
            break

    m = _FUNC_RE.match(e)
    if m:
        fn = m.group(1).strip().lower()
        args = _split_top_level_args(m.group(2))

        if fn == "distinct":
            if len(args) < 2:
                raise ValueError("Distinct expects at least 2 args")
            return Distinct(*[_term(a, var_map) for a in args])

        parsed = [_parse_expr(a, var_map) for a in args]
        if fn == "and":
            return Z3And(*parsed)
        if fn == "or":
            return Z3Or(*parsed)
        if fn == "not":
            if len(parsed) != 1:
                raise ValueError("Not expects exactly 1 arg")
            return Z3Not(parsed[0])
        if fn == "implies":
            if len(parsed) != 2:
                raise ValueError("Implies expects exactly 2 args")
            return Z3Implies(parsed[0], parsed[1])
        if fn == "xor":
            if len(parsed) != 2:
                raise ValueError("Xor expects exactly 2 args")
            return Z3Xor(parsed[0], parsed[1])

        raise ValueError(f"Unsupported function: {fn}")

    return _parse_atomic(e, var_map)


def _solver_result(assertions: List[Any], timeout_s: float):
    s = Solver()
    s.set("timeout", int(float(timeout_s) * 1000))
    s.add(assertions)
    return s.check()


def _is_sat(assertions: List[Any], timeout_s: float) -> bool:
    return _solver_result(assertions, timeout_s) == sat


def _is_unsat(assertions: List[Any], timeout_s: float) -> bool:
    return _solver_result(assertions, timeout_s) == unsat


def _option_sat(base_solver: Solver, option_phi, timeout_s: float) -> bool:
    return _is_sat(list(base_solver.assertions()) + [option_phi], timeout_s)


def _entailment_status(assertions: List[Any], phi, timeout_s: float) -> str:
    # Avoid explosion from inconsistent premises.
    if not _is_sat(assertions, timeout_s):
        return "PREMISES_UNSAT"
    if _is_unsat(assertions + [phi], timeout_s):
        return "CONTRADICTION"
    if _is_unsat(assertions + [Z3Not(phi)], timeout_s):
        return "ENTAILED"
    return "NOT_ENTAILED"


def _equiv_under_base(base_assertions: List[Any], phi, psi, timeout_s: float) -> bool:
    return _is_unsat(base_assertions + [Z3Xor(phi, psi)], timeout_s)


def _is_tautology(base_assertions: List[Any], phi, timeout_s: float) -> bool:
    return _is_unsat(base_assertions + [Z3Not(phi)], timeout_s)


def _extract_gt_option(x: Any) -> Optional[str]:
    if isinstance(x, dict):
        x = x.get("answer") or x.get("selected_option")
    if x is None:
        return None
    s = str(x).strip().upper()
    return s if re.fullmatch(r"[A-Z]", s) else None


def _question_semantics_ok(question_type: str, base_solver: Solver, option_phi, timeout_s: float) -> bool:
    qt = (question_type or "").strip().lower()
    if qt in {"could_be_true", "acceptability", "partial_acceptability", "other"}:
        return _option_sat(base_solver, option_phi, timeout_s)
    if qt in {"cannot_be_true", "must_be_false"}:
        return not _option_sat(base_solver, option_phi, timeout_s)
    if qt in {"must_be_true", "must_follow"}:
        return _is_unsat(list(base_solver.assertions()) + [Z3Not(option_phi)], timeout_s)
    if qt == "could_be_false":
        return _is_sat(list(base_solver.assertions()) + [Z3Not(option_phi)], timeout_s)
    return _option_sat(base_solver, option_phi, timeout_s)


def _validate_reasoning_ordering(
    *,
    reasoning: List[str],
    base_solver: Solver,
    base_only_solver: Solver,
    var_map: Dict[str, Any],
    rule_fact_phis: List[Any],
    options_phis: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    n_total = 0
    n_parsed_ok = 0
    n_valid = 0
    n_novel = 0
    n_contra = 0
    list_steps_valid: List[str] = []
    list_steps_non_valid: List[Dict[str, Any]] = []
    list_novel_steps_inc_clues: List[str] = []
    list_step_parse_errors: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    seen_sexprs = set()
    steps_solver = Solver()
    steps_solver.set("timeout", int(float(timeout_s) * 1000))
    steps_solver.add(base_only_solver.assertions())

    for raw in reasoning or []:
        parsed = _extract_step_expr(raw)
        if parsed is None:
            continue
        n_total += 1
        k, expr = parsed

        m_opt = re.match(r"^\s*(Sat|Unsat)\s*\(\s*Option_([A-Z])\s*\)\s*$", expr, re.IGNORECASE)
        if m_opt:
            n_parsed_ok += 1
            op = m_opt.group(1).lower()
            label = m_opt.group(2).upper()
            option_phi = options_phis.get(label)
            if option_phi is None:
                item = {"k": k, "raw": raw, "expr": expr, "validity_status": "OPTION_NOT_FOUND"}
                list_steps_non_valid.append(item)
                skipped.append(item)
                continue
            is_sat_now = _option_sat(base_solver, option_phi, timeout_s)
            ok = (op == "sat" and is_sat_now) or (op == "unsat" and not is_sat_now)
            if ok:
                n_valid += 1
                n_novel += 1
                list_steps_valid.append(expr)
                list_novel_steps_inc_clues.append(expr)
            else:
                item = {"k": k, "raw": raw, "expr": expr, "validity_status": "BAD_OPTION_STATUS", "solver_sat": is_sat_now}
                list_steps_non_valid.append(item)
                skipped.append(item)
            continue

        try:
            phi = _parse_expr(expr, var_map)
            n_parsed_ok += 1
        except Exception as e:
            item = {"k": k, "raw": raw, "expr": expr, "status": "PARSE_ERROR", "error": f"{type(e).__name__}: {e}"}
            list_step_parse_errors.append(item)
            list_steps_non_valid.append({**item, "validity_status": "PARSE_ERROR"})
            skipped.append(item)
            continue

        sexpr = phi.sexpr()
        if sexpr in seen_sexprs:
            skipped.append({"k": k, "raw": raw, "expr": expr, "status": "DUPLICATE_STEP"})
            continue

        base_assertions = list(base_only_solver.assertions())
        full_assertions = list(base_solver.assertions())

        try:
            if any(_equiv_under_base(base_assertions, phi, c, timeout_s) for c in rule_fact_phis):
                status = _entailment_status(full_assertions, phi, timeout_s)
                if status == "ENTAILED":
                    n_valid += 1
                    list_steps_valid.append(expr)
                skipped.append({"k": k, "raw": raw, "expr": expr, "status": "RESTATES_RULE_OR_FACT"})
                seen_sexprs.add(sexpr)
                continue
        except Exception:
            pass

        try:
            if _is_tautology(base_assertions, phi, timeout_s):
                skipped.append({"k": k, "raw": raw, "expr": expr, "status": "TAUTOLOGY"})
                seen_sexprs.add(sexpr)
                continue
        except Exception:
            pass

        validity_status = _entailment_status(full_assertions, phi, timeout_s)
        if validity_status == "ENTAILED":
            n_valid += 1
            list_steps_valid.append(expr)
        else:
            if validity_status == "CONTRADICTION":
                n_contra += 1
            list_steps_non_valid.append({"k": k, "raw": raw, "expr": expr, "validity_status": validity_status})

        steps_status = _entailment_status(list(steps_solver.assertions()), phi, timeout_s)
        if validity_status == "ENTAILED" and steps_status != "ENTAILED" and steps_status != "CONTRADICTION":
            n_novel += 1
            list_novel_steps_inc_clues.append(expr)

        if steps_status != "CONTRADICTION":
            steps_solver.add(phi)
        seen_sexprs.add(sexpr)

    return {
        "n_steps_total": n_total,
        "n_steps_parsed_ok": n_parsed_ok,
        "n_steps_valid": n_valid,
        "n_steps_novel_inc_clues": n_novel,
        "n_non_valid_contradiction": n_contra,
        "list_steps_valid": list_steps_valid,
        "list_steps_non_valid": list_steps_non_valid,
        "list_novel_steps_inc_clues": list_novel_steps_inc_clues,
        "list_step_parse_errors": list_step_parse_errors,
        "list_skipped_steps_inc_clues": skipped,
    }


def solve_and_validate_payload(payload: Dict[str, Any], *, timeout_s: float = 2.0, conflict_tolerant_clues: bool = False) -> Dict[str, Any]:
    """Validate an AR-LSAT ordering payload and return reward-friendly metrics."""
    report: Dict[str, Any] = {
        "parse_status": "INIT",
        "base_sat_full_GT": 0.0,
        "option_correct": 0.0,
        "base_rules_facts_sat": 0.0,
        "selected_option_semantics_ok": 0.0,
        "selected_option": None,
        "ground_truth_option": None,
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
        question_type = ((payload.get("question_semantics") or {}).get("question_type") or "").strip()

        selected = _extract_gt_option(solution)
        gt = _extract_gt_option(payload.get("ground_truth"))
        report["selected_option"] = selected
        report["ground_truth_option"] = gt
        report["option_correct"] = 1.0 if selected and gt and selected == gt else 0.0

        base_only_solver, var_map, _ = _make_ordering_base(world_model, timeout_s)
        full_solver = Solver()
        full_solver.set("timeout", int(float(timeout_s) * 1000))
        full_solver.add(base_only_solver.assertions())

        rule_fact_phis: List[Any] = []
        parse_errors: List[Dict[str, Any]] = []
        for expr in list(rules) + list(facts):
            try:
                phi = _parse_expr(str(expr), var_map)
                full_solver.add(phi)
                rule_fact_phis.append(phi)
            except Exception as e:
                parse_errors.append({"expr": expr, "error": f"{type(e).__name__}: {e}"})
        report["rule_fact_parse_errors"] = parse_errors

        base_sat = full_solver.check() == sat
        report["base_rules_facts_sat"] = 1.0 if base_sat else 0.0
        if not base_sat:
            report["parse_status"] = "BASE_UNSAT"
            return report

        options_phis: Dict[str, Any] = {}
        option_parse_errors: List[Dict[str, Any]] = []
        for label, expr in options.items():
            try:
                options_phis[str(label).strip().upper()] = _parse_expr(str(expr), var_map)
            except Exception as e:
                option_parse_errors.append({"option": label, "expr": expr, "error": f"{type(e).__name__}: {e}"})
        report["option_parse_errors"] = option_parse_errors

        selected_phi = options_phis.get(selected) if selected else None
        if selected_phi is not None:
            sem_ok = _question_semantics_ok(question_type, full_solver, selected_phi, timeout_s)
            report["selected_option_semantics_ok"] = 1.0 if sem_ok else 0.0
        else:
            sem_ok = False

        reason_out = _validate_reasoning_ordering(
            reasoning=reasoning,
            base_solver=full_solver,
            base_only_solver=base_only_solver,
            var_map=var_map,
            rule_fact_phis=rule_fact_phis,
            options_phis=options_phis,
            timeout_s=timeout_s,
        )
        report.update(reason_out)

        report["base_sat_full_GT"] = 1.0 if (report["option_correct"] == 1.0 and sem_ok) else 0.0
        report["parse_status"] = "AR_LSAT_ORDERING_SUCCESS"
        return report

    except Exception as e:
        report["parse_status"] = "AR_LSAT_ORDERING_FAIL"
        report["error"] = f"{type(e).__name__}: {e}"
        return report


if __name__ == "__main__":
    payload = {
        "problem_type": "ordering",
        "world_model": {"entities": ["A", "B", "C", "D"], "domains": {"positions": ["1", "2", "3", "4"]}},
        "rules": ["Distinct(A, B, C, D)", "A < B", "C == A + 1", "D != 1"],
        "facts": ["B == 4"],
        "question_semantics": {"question_type": "could_be_true"},
        "options": {"A": "A == 2", "B": "C == 4"},
        "reasoning": ["B is fourth.", "S1: B == 4.", "C follows A immediately.", "S2: C == A + 1."],
        "solution": {"selected_option": "A"},
        "ground_truth": {"answer": "A"},
    }
    print(json.dumps(solve_and_validate_payload(payload), indent=2))
