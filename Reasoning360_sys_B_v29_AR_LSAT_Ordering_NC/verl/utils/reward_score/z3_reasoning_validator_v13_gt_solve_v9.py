#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR-LSAT ORDERING Z3 validator for process-reward computation.

Core semantics
--------------
* One integer position variable is created per entity.
* Every entity must occupy one of the explicitly declared positions.
* Entity positions are pairwise distinct unless world_model allows ties.
* Every answer option is evaluated independently under the question semantics.
* The solver answer is the unique option satisfying those semantics.
* BASE_sat_full_GT does not depend on the LLM-selected option. It means:
    - the formalization is complete,
    - the base theory is satisfiable,
    - exactly one option is solver-correct, and
    - that solver-derived option matches the official ground truth.
* Reasoning validity is checked against base constraints + rules + facts.
* Novelty is checked against previous non-contradictory steps without rules/facts.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import z3
    from z3 import (
        Abs,
        And,
        AtLeast,
        AtMost,
        BoolVal,
        Distinct,
        Implies,
        Int,
        Not,
        Or,
        Solver,
        Xor,
        sat,
        unsat,
    )
except Exception:  # pragma: no cover
    z3 = None

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_FUNC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_OPTION_STATUS_RE = re.compile(
    r"^\s*(Sat|Unsat)\(\s*(Not\()?\s*(?:Option[_\-:\s]*)?([A-E])\s*\)?\s*\)\s*$",
    re.IGNORECASE,
)


def normalize_header(data_sample):
    return data_sample


def normalize_months_in_rows(z3_solution: dict) -> dict:
    return z3_solution


def _norm_token(x: Any) -> str:
    text = str(x).strip().strip("`'\"“”‘’")
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise ValueError(f"Invalid empty token after normalization: {x!r}")
    return text


def _normalize_option_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, int) and 0 <= x <= 4:
        return chr(ord("A") + x)
    text = str(x).strip().upper()
    if not text:
        return None
    compact = text.replace("-", "_").replace(" ", "_").replace(":", "_")
    match = re.fullmatch(r"(?:SELECTED_)?(?:OPTION|ANSWER|CHOICE)?_?([A-E])", compact)
    if match:
        return match.group(1)
    match = re.search(r"(?:OPTION|ANSWER|CHOICE|SELECTED_OPTION)\s*[:_\-\s]*([A-E])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-E])\b\s*[\.)\]]?\s*$", text)
    if match:
        return match.group(1)
    letters = re.findall(r"\b([A-E])\b", text)
    return letters[-1] if letters else None


def _selected_from_ground_truth(gt: Any) -> Optional[str]:
    if isinstance(gt, (str, int)):
        return _normalize_option_label(gt)
    if isinstance(gt, dict):
        for key in ("answer", "selected_option", "ground_truth_option", "label"):
            if gt.get(key) is not None:
                return _selected_from_ground_truth(gt[key])
    return None


def _selected_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    solution = payload.get("solution") or {}
    if isinstance(solution, dict):
        return _normalize_option_label(solution.get("selected_option"))
    return None


def _split_top_level_args(text: str) -> List[str]:
    args: List[str] = []
    buffer: List[str] = []
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
            buffer.append(char)
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses")
            buffer.append(char)
        elif char == "," and depth == 0:
            part = "".join(buffer).strip()
            if part:
                args.append(part)
            buffer = []
        else:
            buffer.append(char)
    if depth != 0:
        raise ValueError("Unbalanced parentheses")
    tail = "".join(buffer).strip()
    if tail:
        args.append(tail)
    return args


def _find_top_level_operator(expr: str, operators: Sequence[str]) -> Optional[Tuple[str, str, str]]:
    depth = 0
    index = 0
    while index < len(expr):
        char = expr[index]
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses")
            index += 1
            continue
        if depth == 0:
            for operator in operators:
                if expr.startswith(operator, index):
                    left = expr[:index].strip()
                    right = expr[index + len(operator):].strip()
                    if left and right:
                        return left, operator, right
        index += 1
    if depth != 0:
        raise ValueError("Unbalanced parentheses")
    return None


def _unwrap_parentheses(expr: str) -> str:
    text = expr.strip()
    changed = True
    while changed and text.startswith("(") and text.endswith(")"):
        changed = False
        depth = 0
        balanced_outer = True
        for idx, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and idx != len(text) - 1:
                    balanced_outer = False
                    break
        if balanced_outer and depth == 0:
            text = text[1:-1].strip()
            changed = True
    return text


def _extract_entities_positions(world_model: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    entities = [_norm_token(entity) for entity in (world_model.get("entities") or [])]
    if len(set(entities)) != len(entities):
        raise ValueError("Duplicate entities after normalization")

    domains = world_model.get("domains") or {}
    raw_positions = domains.get("positions") or domains.get("position") or []
    positions: List[int] = []
    for raw in raw_positions:
        try:
            positions.append(int(str(raw).strip()))
        except Exception as exc:
            raise ValueError(f"Position must be an integer: {raw!r}") from exc

    if not positions and entities:
        positions = list(range(1, len(entities) + 1))

    positions = sorted(set(positions))
    return entities, positions


def _ties_allowed(world_model: Dict[str, Any]) -> bool:
    explicit = world_model.get("allow_ties")
    if explicit is not None:
        return bool(explicit)
    assumptions = world_model.get("structural_assumptions") or []
    if isinstance(assumptions, str):
        assumptions = [assumptions]
    text = " ".join(str(item).lower() for item in assumptions)
    return "ties allowed" in text or "same position allowed" in text


def _make_base(world_model: Dict[str, Any], timeout_s: float):
    if z3 is None:
        raise RuntimeError("z3-solver is not installed")

    entities, positions = _extract_entities_positions(world_model)
    if not entities:
        raise ValueError("world_model.entities is empty")
    if not positions:
        raise ValueError("world_model.domains.positions is empty")
    if not _ties_allowed(world_model) and len(entities) > len(positions):
        raise ValueError("More entities than available positions while ties are disallowed")

    var_map: Dict[str, Any] = {entity: Int(entity) for entity in entities}
    base_assertions: List[Any] = []

    # Use membership in the explicit position set rather than min/max bounds.
    # This correctly handles non-contiguous domains such as [1, 2, 4, 5].
    for variable in var_map.values():
        base_assertions.append(Or(*[variable == position for position in positions]))

    if not _ties_allowed(world_model) and len(var_map) > 1:
        base_assertions.append(Distinct(*var_map.values()))

    return var_map, positions, base_assertions, len(positions), len(entities)


def _entity(raw: str, var_map: Dict[str, Any]):
    token = _norm_token(raw)
    if token not in var_map:
        raise KeyError(f"Unknown ordering entity: {token!r}; known={sorted(var_map)}")
    return var_map[token]


def _parse_int_term(expr: str, var_map: Dict[str, Any], positions: Sequence[int]):
    text = _unwrap_parentheses(str(expr).strip())

    if re.fullmatch(r"-?\d+", text):
        return int(text)

    # Position(A) is a readable alias for the entity's integer variable.
    match = _FUNC_RE.match(text)
    if match and match.group(1).lower() in {"pos", "position"}:
        args = _split_top_level_args(match.group(2))
        if len(args) != 1:
            raise ValueError("Position/Pos expects exactly one entity")
        return _entity(args[0], var_map)

    # Top-level arithmetic. These operators are intentionally limited.
    split = _find_top_level_operator(text, ("+", "-"))
    if split is not None:
        left, operator, right = split
        left_term = _parse_int_term(left, var_map, positions)
        right_term = _parse_int_term(right, var_map, positions)
        return left_term + right_term if operator == "+" else left_term - right_term

    return _entity(text, var_map)


def _parse_expr(expr: str, var_map: Dict[str, Any], positions: Sequence[int]):
    text = _unwrap_parentheses(str(expr).strip().rstrip("."))
    if text.lower() == "true":
        return BoolVal(True)
    if text.lower() == "false":
        return BoolVal(False)

    # Comparisons are parsed before generic function calls so that expressions
    # such as Position(A) == Position(B) + 1 are handled correctly.
    comparison = _find_top_level_operator(text, ("<=", ">=", "==", "!=", "<", ">"))
    if comparison is not None:
        left, operator, right = comparison
        left_term = _parse_int_term(left, var_map, positions)
        right_term = _parse_int_term(right, var_map, positions)
        return {
            "==": left_term == right_term,
            "!=": left_term != right_term,
            "<": left_term < right_term,
            ">": left_term > right_term,
            "<=": left_term <= right_term,
            ">=": left_term >= right_term,
        }[operator]

    match = _FUNC_RE.match(text)
    if not match:
        raise ValueError(f"Unrecognized ordering expression: {expr!r}")

    function = match.group(1).lower()
    args = _split_top_level_args(match.group(2))

    if function == "distinct":
        if len(args) < 2:
            raise ValueError("Distinct expects at least two entities")
        return Distinct(*[_entity(arg, var_map) for arg in args])

    if function in {"before", "earlierthan", "precedes"}:
        if len(args) != 2:
            raise ValueError("Before expects exactly two entities")
        return _entity(args[0], var_map) < _entity(args[1], var_map)

    if function in {"after", "laterthan", "follows"}:
        if len(args) != 2:
            raise ValueError("After expects exactly two entities")
        return _entity(args[0], var_map) > _entity(args[1], var_map)

    if function in {"immediatelybefore", "directlybefore"}:
        if len(args) != 2:
            raise ValueError("ImmediatelyBefore expects exactly two entities")
        return _entity(args[0], var_map) + 1 == _entity(args[1], var_map)

    if function in {"immediatelyafter", "directlyafter"}:
        if len(args) != 2:
            raise ValueError("ImmediatelyAfter expects exactly two entities")
        return _entity(args[0], var_map) == _entity(args[1], var_map) + 1

    if function in {"adjacent", "nextto"}:
        if len(args) != 2:
            raise ValueError("Adjacent expects exactly two entities")
        return Abs(_entity(args[0], var_map) - _entity(args[1], var_map)) == 1

    if function == "between":
        if len(args) != 3:
            raise ValueError("Between(A, B, C) expects exactly three entities")
        # B is between A and C.
        a, b, c = (_entity(arg, var_map) for arg in args)
        return Or(And(a < b, b < c), And(c < b, b < a))

    if function in {"distance", "apartby"}:
        if len(args) != 3:
            raise ValueError("Distance expects entity, entity, integer distance")
        try:
            distance = int(str(args[2]).strip())
        except Exception as exc:
            raise ValueError("Distance's third argument must be an integer") from exc
        if distance < 0:
            raise ValueError("Distance must be non-negative")
        return Abs(_entity(args[0], var_map) - _entity(args[1], var_map)) == distance

    if function == "first":
        if len(args) != 1:
            raise ValueError("First expects exactly one entity")
        return _entity(args[0], var_map) == min(positions)

    if function == "last":
        if len(args) != 1:
            raise ValueError("Last expects exactly one entity")
        return _entity(args[0], var_map) == max(positions)

    if function in {"atposition", "inposition"}:
        if len(args) != 2:
            raise ValueError("AtPosition expects entity and integer position")
        try:
            position = int(str(args[1]).strip())
        except Exception as exc:
            raise ValueError("AtPosition's second argument must be an integer") from exc
        return _entity(args[0], var_map) == position

    if function in {"and", "or", "not", "implies", "xor"}:
        parsed = [_parse_expr(arg, var_map, positions) for arg in args]
        if function == "and":
            if len(parsed) < 2:
                raise ValueError("And expects at least two arguments")
            return And(*parsed)
        if function == "or":
            if len(parsed) < 2:
                raise ValueError("Or expects at least two arguments")
            return Or(*parsed)
        if function == "not":
            if len(parsed) != 1:
                raise ValueError("Not expects exactly one argument")
            return Not(parsed[0])
        if function == "implies":
            if len(parsed) != 2:
                raise ValueError("Implies expects exactly two arguments")
            return Implies(parsed[0], parsed[1])
        if len(parsed) != 2:
            raise ValueError("Xor expects exactly two arguments")
        return Xor(parsed[0], parsed[1])

    if function in {"atleast", "atmost", "exactly"}:
        if len(args) < 2:
            raise ValueError(f"{function} expects k and Boolean expressions")
        try:
            count = int(str(args[0]).strip())
        except Exception as exc:
            raise ValueError(f"{function}'s first argument must be an integer") from exc
        parsed = [_parse_expr(arg, var_map, positions) for arg in args[1:]]
        if function == "atleast":
            return AtLeast(*parsed, count)
        if function == "atmost":
            return AtMost(*parsed, count)
        return And(AtLeast(*parsed, count), AtMost(*parsed, count))

    raise ValueError(f"Unsupported ordering function: {function}")


def _parse_constraints(
    lines: Iterable[str], var_map: Dict[str, Any], positions: Sequence[int]
) -> Tuple[List[Any], List[Dict[str, str]]]:
    formulas: List[Any] = []
    errors: List[Dict[str, str]] = []
    for raw in lines or []:
        try:
            formulas.append(_parse_expr(str(raw), var_map, positions))
        except Exception as exc:
            errors.append({"raw": str(raw), "error": f"{type(exc).__name__}: {exc}"})
    return formulas, errors


def _parse_options(
    options: Dict[str, str], var_map: Dict[str, Any], positions: Sequence[int]
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    formulas: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []
    for raw_label, expression in (options or {}).items():
        label = _normalize_option_label(raw_label)
        if label is None:
            errors.append({"label": str(raw_label), "raw": str(expression), "error": "Invalid option label"})
            continue
        if label in formulas:
            errors.append({"label": label, "raw": str(expression), "error": "Duplicate normalized option label"})
            continue
        try:
            formulas[label] = _parse_expr(str(expression), var_map, positions)
        except Exception as exc:
            errors.append({"label": label, "raw": str(expression), "error": f"{type(exc).__name__}: {exc}"})
    return formulas, errors


def _solver_check(assertions: Sequence[Any], timeout_s: float):
    solver = Solver()
    solver.set("timeout", int(float(timeout_s) * 1000))
    solver.add(list(assertions))
    return solver.check()


def _is_sat(assertions: Sequence[Any], timeout_s: float) -> bool:
    return _solver_check(assertions, timeout_s) == sat


def _is_unsat(assertions: Sequence[Any], timeout_s: float) -> bool:
    return _solver_check(assertions, timeout_s) == unsat


def _status_under_gamma(gamma: Sequence[Any], phi: Any, timeout_s: float) -> str:
    base_status = _solver_check(gamma, timeout_s)
    if base_status == unsat:
        return "PREMISES_UNSAT"
    if z3 is not None and base_status == z3.unknown:
        return "UNKNOWN"
    positive = _solver_check([*gamma, phi], timeout_s)
    if positive == unsat:
        return "CONTRADICTION"
    if z3 is not None and positive == z3.unknown:
        return "UNKNOWN"
    negative = _solver_check([*gamma, Not(phi)], timeout_s)
    if negative == unsat:
        return "ENTAILED"
    if z3 is not None and negative == z3.unknown:
        return "UNKNOWN"
    return "NOT_ENTAILED"


def _is_tautology(base_assertions: Sequence[Any], phi: Any, timeout_s: float) -> bool:
    return _is_sat(base_assertions, timeout_s) and _is_unsat([*base_assertions, Not(phi)], timeout_s)


def _equiv_to_any_rule(base_assertions: Sequence[Any], phi: Any, rules: Sequence[Any], timeout_s: float) -> bool:
    for rule in rules:
        if _is_unsat([*base_assertions, Xor(phi, rule)], timeout_s):
            return True
    return False


def _evaluate_option(question_type: str, option_phi: Any, gamma: Sequence[Any], timeout_s: float) -> bool:
    question = str(question_type or "").strip().lower()
    if question in {"could_be_true", "acceptability", "partial_acceptability", "valid_complete_assignment"}:
        return _is_sat([*gamma, option_phi], timeout_s)
    if question in {"must_be_true", "must_follow"}:
        return _is_unsat([*gamma, Not(option_phi)], timeout_s)
    if question in {"cannot_be_true", "must_be_false"}:
        return _is_unsat([*gamma, option_phi], timeout_s)
    if question == "could_be_false":
        return _is_sat([*gamma, Not(option_phi)], timeout_s)
    raise ValueError(f"Unsupported or missing question_type: {question_type!r}")


def _solve_options(
    question_type: str,
    option_phis: Dict[str, Any],
    gamma: Sequence[Any],
    timeout_s: float,
) -> Tuple[Dict[str, bool], Dict[str, str]]:
    evaluations: Dict[str, bool] = {}
    errors: Dict[str, str] = {}
    for label, formula in sorted(option_phis.items()):
        try:
            evaluations[label] = bool(_evaluate_option(question_type, formula, gamma, timeout_s))
        except Exception as exc:
            evaluations[label] = False
            errors[label] = f"{type(exc).__name__}: {exc}"
    return evaluations, errors


def _extract_step_expr(line: str) -> Optional[Tuple[int, str]]:
    match = _STEP_RE.match(str(line or "").strip())
    if not match:
        return None
    expression = match.group(2).strip()
    if "[" in expression:
        expression = expression.split("[", 1)[0].strip()
    expression = expression.rstrip(".").strip()
    return (int(match.group(1)), expression) if expression else None


def _option_status_expr(expr: str) -> Optional[Tuple[str, bool, str]]:
    match = _OPTION_STATUS_RE.fullmatch(str(expr or "").strip().rstrip("."))
    if not match:
        return None
    return match.group(1).lower(), bool(match.group(2)), match.group(3).upper()


def _option_status_is_true(
    status: str,
    is_negated: bool,
    option_phi: Any,
    gamma: Sequence[Any],
    timeout_s: float,
) -> bool:
    formula = Not(option_phi) if is_negated else option_phi
    if status == "sat":
        return _is_sat([*gamma, formula], timeout_s)
    if status == "unsat":
        return _is_unsat([*gamma, formula], timeout_s)
    return False


def _supports_solver_answer(
    question_type: str,
    status: str,
    is_negated: bool,
    label: str,
    solver_answer: Optional[str],
) -> bool:
    if not solver_answer or label != solver_answer:
        return False
    question = str(question_type or "").strip().lower()
    if question in {"could_be_true", "acceptability", "partial_acceptability", "valid_complete_assignment"}:
        return status == "sat" and not is_negated
    if question in {"must_be_true", "must_follow"}:
        return status == "unsat" and is_negated
    if question in {"cannot_be_true", "must_be_false"}:
        return status == "unsat" and not is_negated
    if question == "could_be_false":
        return status == "sat" and is_negated
    return False


def _validate_reasoning_steps(
    reasoning: Sequence[str],
    *,
    var_map: Dict[str, Any],
    positions: Sequence[int],
    base_assertions: Sequence[Any],
    rule_fact_phis: Sequence[Any],
    rule_phis: Sequence[Any],
    option_phis: Dict[str, Any],
    question_type: str,
    solver_answer: Optional[str],
    timeout_s: float,
) -> Dict[str, Any]:
    gamma_valid = [*base_assertions, *rule_fact_phis]
    gamma_steps: List[Any] = list(base_assertions)  # no rules/facts for novelty
    seen: set[str] = set()

    n_total = 0
    n_parsed = 0
    valid_steps: List[Dict[str, Any]] = []
    novel_steps: List[Dict[str, Any]] = []
    non_valid: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []
    support_steps: List[Dict[str, Any]] = []
    skipped_steps: List[Dict[str, Any]] = []

    for raw_line in reasoning or []:
        parsed = _extract_step_expr(str(raw_line))
        if parsed is None:
            continue
        n_total += 1
        step_number, expression = parsed

        option_status = _option_status_expr(expression)
        if option_status is not None:
            n_parsed += 1
            status, is_negated, label = option_status
            option_phi = option_phis.get(label)
            if option_phi is None:
                non_valid.append({
                    "k": step_number,
                    "raw": raw_line,
                    "expr": expression,
                    "validity_status": "UNKNOWN_OPTION",
                })
                continue
            option_valid = _option_status_is_true(status, is_negated, option_phi, gamma_valid, timeout_s)
            supports = bool(
                option_valid
                and _supports_solver_answer(
                    question_type, status, is_negated, label, solver_answer
                )
            )
            entry = {
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "option_label": label,
                "status_operator": status,
                "is_negated": is_negated,
                "supports_solver_answer": supports,
                "validity_status": "OPTION_STATUS_VALID" if option_valid else "OPTION_STATUS_INVALID",
            }
            if option_valid:
                valid_steps.append(entry)
                if supports:
                    support_steps.append(entry)
            else:
                non_valid.append(entry)
            continue

        try:
            phi = _parse_expr(expression, var_map, positions)
            n_parsed += 1
        except Exception as exc:
            error = {
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "status": "PARSE_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            parse_errors.append(error)
            non_valid.append({
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "validity_status": "PARSE_ERROR",
                "reason": error["error"],
            })
            continue

        sexpr = phi.sexpr()
        if sexpr in seen:
            skipped_steps.append({
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "status": "DUPLICATE_STEP",
            })
            continue
        seen.add(sexpr)

        validity = _status_under_gamma(gamma_valid, phi, timeout_s)
        is_valid = validity == "ENTAILED"
        if is_valid:
            valid_steps.append({
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "validity_status": validity,
            })
        else:
            non_valid.append({
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "validity_status": validity,
            })

        # Tautologies and direct/semantic rule restatements are valid but not novel.
        if _is_tautology(base_assertions, phi, timeout_s):
            skipped_steps.append({
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "status": "TAUTOLOGY",
            })
            continue
        if _equiv_to_any_rule(base_assertions, phi, rule_phis, timeout_s):
            skipped_steps.append({
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "status": "RESTATES_RULE",
            })
            continue

        step_status = _status_under_gamma(gamma_steps, phi, timeout_s)
        if step_status == "CONTRADICTION":
            # Contradictory steps never become premises for later novelty checks.
            skipped_steps.append({
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "status": "CONTRADICTION_WITH_PREVIOUS_STEPS",
            })
            continue

        if is_valid and step_status != "ENTAILED":
            novel_steps.append({
                "k": step_number,
                "raw": raw_line,
                "expr": expression,
                "validity_status": validity,
                "steps_status": step_status,
            })

        # Previous non-contradictory steps are included, even if not clue-valid,
        # matching the stated novelty definition.
        gamma_steps.append(phi)

    return {
        "n_steps_total": n_total,
        "n_steps_parsed_ok": n_parsed,
        "n_steps_valid": len(valid_steps),
        "n_steps_novel_inc_clues": len(novel_steps),
        "n_non_valid_contradiction": sum(
            1 for item in non_valid if item.get("validity_status") == "CONTRADICTION"
        ),
        "list_steps_valid": [item.get("expr") for item in valid_steps],
        "list_steps_non_valid": non_valid,
        "list_novel_steps_inc_clues": [item.get("expr") for item in novel_steps],
        "list_step_parse_errors": parse_errors,
        "list_skipped_steps": skipped_steps,
        "consistency_score": 1.0 if support_steps else 0.0,
        "solution_support_steps": support_steps,
    }


def solve_and_validate_payload(
    payload: Dict[str, Any],
    *,
    timeout_s: float = 2.0,
    conflict_tolerant_clues: bool = False,
) -> Dict[str, Any]:
    del conflict_tolerant_clues  # retained for call-site compatibility

    report: Dict[str, Any] = {
        "base_sat_full_GT": False,
        "parse_status": "INIT",
        "n_steps_total": 0,
        "n_steps_parsed_ok": 0,
        "n_steps_valid": 0,
        "n_steps_novel_inc_clues": 0,
        "n_non_valid_contradiction": 0,
        "consistency_score": 0.0,
        "solution_support_steps": [],
    }

    try:
        if str(payload.get("problem_type") or "").strip().lower() != "ordering":
            raise ValueError("This validator only supports problem_type='ordering'.")

        var_map, positions, base_assertions, n_positions, n_entities = _make_base(
            payload.get("world_model") or {}, timeout_s
        )

        rule_phis, rule_errors = _parse_constraints(
            payload.get("rules") or [], var_map, positions
        )
        fact_phis, fact_errors = _parse_constraints(
            payload.get("facts") or [], var_map, positions
        )
        option_phis, option_errors = _parse_options(
            payload.get("options") or {}, var_map, positions
        )

        rule_fact_phis = [*rule_phis, *fact_phis]
        gamma = [*base_assertions, *rule_fact_phis]
        base_status = _solver_check(gamma, timeout_s)
        base_sat = base_status == sat

        selected = _selected_from_payload(payload)
        ground_truth = _selected_from_ground_truth(payload.get("ground_truth"))
        question_type = (
            (payload.get("question_semantics") or {}).get("question_type")
            or payload.get("question_type")
        )
        if not question_type:
            raise ValueError("Missing question_type; option semantics cannot be determined")

        option_evaluations, option_evaluation_errors = _solve_options(
            question_type, option_phis, gamma, timeout_s
        ) if base_sat else ({label: False for label in option_phis}, {})

        solver_correct_options = [
            label for label, is_correct in sorted(option_evaluations.items()) if is_correct
        ]
        solver_has_unique_answer = len(solver_correct_options) == 1
        solver_answer = solver_correct_options[0] if solver_has_unique_answer else None

        formalization_complete = bool(
            len(rule_errors) == 0
            and len(fact_errors) == 0
            and len(option_errors) == 0
            and len(option_evaluation_errors) == 0
            and bool(option_phis)
        )
        solver_matches_gt = bool(
            solver_answer is not None
            and ground_truth is not None
            and solver_answer == ground_truth
        )
        model_matches_solver = bool(
            selected is not None
            and solver_answer is not None
            and selected == solver_answer
        )
        model_matches_gt = bool(
            selected is not None
            and ground_truth is not None
            and selected == ground_truth
        )

        report.update({
            "base_sat_full_GT": bool(
                base_sat
                and formalization_complete
                and solver_has_unique_answer
                and solver_matches_gt
            ),
            "base_sat": bool(base_sat),
            "base_solver_status": str(base_status),
            "formalization_complete": formalization_complete,
            "solver_answer": solver_answer,
            "solver_correct_options": solver_correct_options,
            "solver_has_unique_answer": solver_has_unique_answer,
            "solver_matches_gt": solver_matches_gt,
            "model_matches_solver": model_matches_solver,
            "model_matches_gt": model_matches_gt,
            "answer_correct": bool(model_matches_solver and model_matches_gt),
            "option_evaluations": option_evaluations,
            "option_evaluation_errors": option_evaluation_errors,
            "selected_option": selected,
            "ground_truth_option": ground_truth,
            "question_type": question_type,
            "rule_parse_errors": rule_errors,
            "fact_parse_errors": fact_errors,
            "option_parse_errors": option_errors,
            "n_rule_parse_errors": len(rule_errors),
            "n_fact_parse_errors": len(fact_errors),
            "n_option_parse_errors": len(option_errors),
            "selected_option_parse_ok": bool(selected in option_phis if selected else False),
            "n_positions": n_positions,
            "n_entities": n_entities,
        })

        report.update(
            _validate_reasoning_steps(
                payload.get("reasoning") or [],
                var_map=var_map,
                positions=positions,
                base_assertions=base_assertions,
                rule_fact_phis=rule_fact_phis,
                rule_phis=rule_phis,
                option_phis=option_phis,
                question_type=question_type,
                solver_answer=solver_answer,
                timeout_s=timeout_s,
            )
        )

        report["parse_status"] = "AR_LSAT_ORDERING_SUCCESS"
        return report

    except Exception as exc:
        logger.exception("AR-LSAT ordering validation failed")
        report["parse_status"] = "Z3_EXCEPTION"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report


if __name__ == "__main__":
    # This puzzle has the unique ordering A=1, C=2, D=3, B=4.
    sample = {
        "problem_type": "ordering",
        "world_model": {
            "entities": ["A", "B", "C", "D"],
            "domains": {"positions": [1, 2, 3, 4]},
            "structural_assumptions": ["each entity occupies exactly one distinct position"],
        },
        "rules": [
            "Before(A, B)",
            "ImmediatelyBefore(A, C)",
            "Not(AtPosition(D, 1))",
        ],
        "facts": ["AtPosition(B, 4)"],
        "question_semantics": {"question_type": "could_be_true"},
        "options": {
            "A": "AtPosition(A, 1)",
            "B": "AtPosition(C, 4)",
            "C": "AtPosition(D, 2)",
        },
        "reasoning": [
            "B is fixed in fourth position by the question condition.",
            "S1: AtPosition(B, 4).",
            "A must immediately precede C.",
            "S2: ImmediatelyBefore(A, C).",
            "The only complete ordering places A first.",
            "S3: AtPosition(A, 1).",
            "Option A is satisfiable under the complete ordering theory.",
            "S4: Sat(Option_A).",
        ],
        "solution": {"selected_option": "A"},
        "ground_truth": "A",
    }
    print(json.dumps(solve_and_validate_payload(sample), indent=2, default=str))
