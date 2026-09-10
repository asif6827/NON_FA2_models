from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

try:
    from verl.utils.reward_score import (
        z3_reasoning_validator_v13_gt_solve_v10_nlspa as _z3v
    )
except Exception:
    import z3_reasoning_validator_v13_gt_solve_v10_nlspa as _z3v


MISSING_PA_DEFAULTS = {
    "pa_present": 0.0,
    "PA_n_total": 0,
    "PA_n_evaluated": 0,

    "PA_n_resolved_cells": 0,
    "PA_n_supported_cells": 0,
    "PA_n_unsupported_cells": 0,

    "PA_n_monotonicity_compared_cells": 0,
    "PA_n_monotonicity_preserved_cells": 0,

    "pa_prefix_support_score": 0.0,
    "pa_monotonicity_score": 0.0,
    "pa_monotonicity_applicable": 0.0,
    "pa_supported_progress_score": 0.0,

    "pa_reward": 0.0,
    "reward_status": "missing_or_failed",

    "list_pa_unsupported_cells": [],
    "list_pa_monotonicity_violations": [],
    "list_valid_s_added_to_prefix": [],
    "list_valid_s_prefix_parse_errors": [],
    "list_clue_parse_errors": [],
    "list_pa_errors": [],
    "pa_details": [],
}


_S_KEY_RE = re.compile(r"^S(\d+)$")
_PA_KEY_RE = re.compile(r"^PA(\d+)$")


def _canon_expr(expr: Any) -> str:
    s = str(expr).strip()
    if s.endswith("."):
        s = s[:-1].strip()
    return re.sub(r"\s+", "", s)


def _expr_to_z3(expr: str, var_map: Dict[str, Any]):
    raw = str(expr).strip().rstrip(".").strip()
    phi = _z3v._dsl_to_z3(raw, var_map)
    if phi is None:
        phi = _z3v._parse_constraint(raw, var_map)
    if phi is None:
        raise ValueError(f"Could not parse constraint: {expr!r}")
    return phi


def _clone_solver(solver: z3.Solver, timeout_s: float) -> z3.Solver:
    out = z3.Solver()
    out.set("timeout", int(timeout_s * 1000))
    out.add(solver.assertions())
    return out


def _entailment_status(solver: z3.Solver, phi, timeout_s: float) -> str:
    premise_status = solver.check()
    if premise_status == z3.unsat:
        return "PREMISES_UNSAT"
    if premise_status == z3.unknown:
        return "UNKNOWN"

    s_neg = _clone_solver(solver, timeout_s)
    s_neg.add(z3.Not(phi))
    neg_status = s_neg.check()
    if neg_status == z3.unsat:
        return "ENTAILED"

    s_pos = _clone_solver(solver, timeout_s)
    s_pos.add(phi)
    pos_status = s_pos.check()
    if pos_status == z3.unsat:
        return "CONTRADICTION"

    if neg_status == z3.unknown or pos_status == z3.unknown:
        return "UNKNOWN"
    return "NOT_ENTAILED"


def _extract_s_steps(reasoning: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    steps: List[Tuple[int, str, str]] = []
    for key, value in reasoning.items():
        m = _S_KEY_RE.fullmatch(str(key))
        if not m or not isinstance(value, str):
            continue
        steps.append((int(m.group(1)), str(key), value))
    return steps


def _valid_s_indices_from_z3(reasoning: Dict[str, Any], z3_out: Dict[str, Any]) -> Set[int]:
    s_steps = _extract_s_steps(reasoning)
    valid_exprs = z3_out.get("list_steps_valid")
    n_valid_expected = int(z3_out.get("n_steps_valid", 0) or 0)

    if isinstance(valid_exprs, list):
        counts = Counter(_canon_expr(x) for x in valid_exprs)
        valid_indices: Set[int] = set()
        for idx, _, expr in s_steps:
            c = _canon_expr(expr)
            if counts[c] > 0:
                valid_indices.add(idx)
                counts[c] -= 1
        if len(valid_indices) != n_valid_expected:
            raise ValueError(
                "Could not map all z3_out valid S steps back to reasoning keys: "
                f"expected={n_valid_expected}, mapped={len(valid_indices)}"
            )
        return valid_indices

    non_valid = z3_out.get("list_steps_non_valid")
    if isinstance(non_valid, list):
        non_valid_ids: Set[int] = set()
        for item in non_valid:
            if not isinstance(item, dict):
                continue
            try:
                k = int(item.get("k"))
            except Exception:
                continue
            if k > 0:
                non_valid_ids.add(k)
        all_ids = {idx for idx, _, _ in s_steps}
        candidate_valid = all_ids - non_valid_ids
        if len(candidate_valid) == n_valid_expected:
            return candidate_valid

    if n_valid_expected == 0:
        return set()

    raise ValueError(
        "z3_out does not contain enough per-step information to identify "
        "which S_i steps are valid."
    )


def _build_base_clue_solver(*, n_houses: int, attribute_values: Dict[str, List[str]], syntactic_clues: List[str], timeout_s: float) -> Tuple[z3.Solver, Dict[str, Any], List[str]]:
    var_map = _z3v._build_var_map_from_attribute_values(attribute_values)
    base_axioms = _z3v._build_base_axioms(n_houses, attribute_values, var_map)

    solver = z3.Solver()
    solver.set("timeout", int(timeout_s * 1000))
    solver.add(base_axioms)

    clue_errors: List[str] = []
    for raw_clue in syntactic_clues or []:
        try:
            _, expr = _z3v._extract_cid_and_expr(raw_clue)
            phi = _expr_to_z3(expr, var_map)
            solver.add(phi)
        except Exception as exc:
            clue_errors.append(f"{raw_clue} -> {type(exc).__name__}: {exc}")
    return solver, var_map, clue_errors


def _resolve_pa_cells(pa_key: str, pa: Dict[str, Any], *, n_houses: int, attribute_values: Dict[str, List[str]]) -> Tuple[Dict[Tuple[int, str], str], List[str]]:
    errors: List[str] = []
    resolved: Dict[Tuple[int, str], str] = {}

    if not isinstance(pa, dict):
        return {}, [f"{pa_key} is not a dictionary."]

    header = pa.get("header")
    rows = pa.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list):
        return {}, [f"{pa_key} must contain list-valued header and rows."]

    if len(rows) != n_houses:
        errors.append(f"{pa_key} has {len(rows)} rows; expected {n_houses}.")
    if not header or header[0] != "House":
        errors.append(f"{pa_key} header must start with 'House'.")
    for attr in header[1:]:
        if attr not in attribute_values:
            errors.append(f"{pa_key} contains unknown attribute column {attr!r}.")

    for row_i, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            errors.append(f"{pa_key} row {row_i} is not a list.")
            continue
        if len(row) != len(header):
            errors.append(f"{pa_key} row {row_i} has {len(row)} cells; expected {len(header)}.")
            continue
        try:
            house = int(row[0])
        except Exception:
            errors.append(f"{pa_key} row {row_i} has invalid House value {row[0]!r}.")
            continue
        if house != row_i:
            errors.append(f"{pa_key} row {row_i} has House={house}; expected {row_i}.")

        for col_idx, attr in enumerate(header[1:], start=1):
            if attr not in attribute_values:
                continue
            value = row[col_idx]
            if value == "?":
                continue
            if value not in attribute_values[attr]:
                errors.append(
                    f"{pa_key} house {house}, {attr}: value {value!r} is outside attribute_values."
                )
                continue
            resolved[(house, attr)] = value

    return resolved, errors


def reward_PA(payload: Dict[str, Any], timeout_s: float = 5.0) -> Dict[str, Any]:
    out = copy.deepcopy(MISSING_PA_DEFAULTS)

    if not isinstance(payload, dict):
        out["reward_status"] = "invalid_payload"
        return out

    try:
        n_houses = int(payload.get("n_houses"))
    except Exception:
        out["reward_status"] = "invalid_n_houses"
        return out
    if n_houses <= 0:
        out["reward_status"] = "invalid_n_houses"
        return out

    attribute_values = payload.get("attribute_values") or {}
    syntactic_clues = payload.get("syntactic_clues") or []
    reasoning = payload.get("reasoning") or {}
    z3_out = payload.get("z3_out") or {}

    if not isinstance(attribute_values, dict) or not attribute_values:
        out["reward_status"] = "invalid_attribute_values"
        return out
    if not isinstance(reasoning, dict):
        out["reward_status"] = "reasoning_not_dict"
        return out

    pa_keys = [key for key in reasoning.keys() if _PA_KEY_RE.fullmatch(str(key))]
    out["PA_n_total"] = len(pa_keys)
    out["pa_present"] = 1.0 if pa_keys else 0.0
    if not pa_keys:
        out["reward_status"] = "no_pa"
        return out

    try:
        valid_s_indices = _valid_s_indices_from_z3(reasoning, z3_out)
    except Exception as exc:
        out["reward_status"] = "missing_or_unmappable_s_validation"
        out["list_pa_errors"].append(f"{type(exc).__name__}: {exc}")
        return out

    try:
        prefix_solver, var_map, clue_errors = _build_base_clue_solver(
            n_houses=n_houses,
            attribute_values=attribute_values,
            syntactic_clues=syntactic_clues,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        out["reward_status"] = "base_solver_build_error"
        out["list_pa_errors"].append(f"{type(exc).__name__}: {exc}")
        return out

    out["list_clue_parse_errors"] = clue_errors
    if clue_errors:
        out["reward_status"] = "clue_parse_error"
        return out

    base_status = prefix_solver.check()
    if base_status == z3.unsat:
        out["reward_status"] = "base_clues_unsat"
        return out
    if base_status == z3.unknown:
        out["reward_status"] = "base_clues_unknown"
        return out

    total_puzzle_cells = n_houses * len(attribute_values)
    pa_support_scores: List[float] = []
    total_resolved = total_supported = total_unsupported = 0
    monotonic_compared = monotonic_preserved = 0
    committed_cells: Dict[Tuple[int, str], str] = {}
    last_pa_supported_count = 0
    last_pa_key: Optional[str] = None
    last_s_seen = 0

    for key, value in reasoning.items():
        key_str = str(key)

        sm = _S_KEY_RE.fullmatch(key_str)
        if sm:
            s_idx = int(sm.group(1))
            last_s_seen = s_idx
            if s_idx not in valid_s_indices:
                continue
            if not isinstance(value, str):
                out["list_valid_s_prefix_parse_errors"].append({
                    "S": key_str, "error": "S value is not a string."
                })
                continue
            try:
                phi = _expr_to_z3(value, var_map)
                tmp = _clone_solver(prefix_solver, timeout_s)
                tmp.add(phi)
                status = tmp.check()
                if status != z3.sat:
                    out["list_valid_s_prefix_parse_errors"].append({
                        "S": key_str,
                        "expr": value,
                        "error": (
                            "z3_out marked this S step valid, but adding it "
                            f"to the PA prefix produced {status}."
                        ),
                    })
                    continue
                prefix_solver.add(phi)
                out["list_valid_s_added_to_prefix"].append(key_str)
            except Exception as exc:
                out["list_valid_s_prefix_parse_errors"].append({
                    "S": key_str,
                    "expr": value,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            continue

        pm = _PA_KEY_RE.fullmatch(key_str)
        if not pm:
            continue

        resolved_cells, pa_errors = _resolve_pa_cells(
            key_str,
            value,
            n_houses=n_houses,
            attribute_values=attribute_values,
        )
        if pa_errors:
            out["list_pa_errors"].extend(pa_errors)
            out["pa_details"].append({
                "pa": key_str,
                "after_s": last_s_seen,
                "evaluated": False,
                "errors": pa_errors,
            })
            continue

        prefix_status = prefix_solver.check()
        if prefix_status != z3.sat:
            out["pa_details"].append({
                "pa": key_str,
                "after_s": last_s_seen,
                "evaluated": False,
                "errors": [f"Prefix solver status={prefix_status}"],
            })
            out["list_pa_errors"].append(f"{key_str}: prefix solver status={prefix_status}")
            continue

        supported_count = unsupported_count = 0
        cell_details: List[Dict[str, Any]] = []

        for (house, attr), cell_value in resolved_cells.items():
            try:
                zvar = var_map[cell_value]
                phi_cell = (zvar == house)
                status = _entailment_status(prefix_solver, phi_cell, timeout_s)
            except Exception as exc:
                status = "CHECK_ERROR"
                out["list_pa_unsupported_cells"].append({
                    "pa": key_str,
                    "after_s": last_s_seen,
                    "house": house,
                    "attribute": attr,
                    "value": cell_value,
                    "status": status,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                unsupported_count += 1
                cell_details.append({
                    "house": house, "attribute": attr,
                    "value": cell_value, "status": status,
                })
                continue

            if status == "ENTAILED":
                supported_count += 1
            else:
                unsupported_count += 1
                out["list_pa_unsupported_cells"].append({
                    "pa": key_str,
                    "after_s": last_s_seen,
                    "house": house,
                    "attribute": attr,
                    "value": cell_value,
                    "status": status,
                })

            cell_details.append({
                "house": house,
                "attribute": attr,
                "value": cell_value,
                "status": status,
            })

        resolved_count = len(resolved_cells)
        support_score = supported_count / resolved_count if resolved_count > 0 else 0.0
        pa_support_scores.append(support_score)
        total_resolved += resolved_count
        total_supported += supported_count
        total_unsupported += unsupported_count

        for cell_key, old_value in committed_cells.items():
            monotonic_compared += 1
            new_value = resolved_cells.get(cell_key, "?")
            if new_value == old_value:
                monotonic_preserved += 1
            else:
                house, attr = cell_key
                out["list_pa_monotonicity_violations"].append({
                    "pa": key_str,
                    "house": house,
                    "attribute": attr,
                    "previous_value": old_value,
                    "current_value": new_value,
                })

        for cell_key, cell_value in resolved_cells.items():
            if cell_key not in committed_cells:
                committed_cells[cell_key] = cell_value

        out["PA_n_evaluated"] += 1
        last_pa_supported_count = supported_count
        last_pa_key = key_str
        out["pa_details"].append({
            "pa": key_str,
            "after_s": last_s_seen,
            "evaluated": True,
            "resolved_cells": resolved_count,
            "supported_cells": supported_count,
            "unsupported_cells": unsupported_count,
            "prefix_support_score": support_score,
            "cell_details": cell_details,
        })

    out["PA_n_resolved_cells"] = total_resolved
    out["PA_n_supported_cells"] = total_supported
    out["PA_n_unsupported_cells"] = total_unsupported
    out["PA_n_monotonicity_compared_cells"] = monotonic_compared
    out["PA_n_monotonicity_preserved_cells"] = monotonic_preserved

    if out["PA_n_evaluated"] == 0:
        out["reward_status"] = "no_evaluable_pa"
        return out

    pa_prefix_support_score = sum(pa_support_scores) / len(pa_support_scores) if pa_support_scores else 0.0

    if monotonic_compared > 0:
        pa_monotonicity_score = monotonic_preserved / monotonic_compared
        monotonicity_applicable = 1.0
    else:
        pa_monotonicity_score = 0.0
        monotonicity_applicable = 0.0

    pa_supported_progress_score = (
        last_pa_supported_count / total_puzzle_cells
        if total_puzzle_cells > 0 else 0.0
    )

    out["pa_prefix_support_score"] = float(pa_prefix_support_score)
    out["pa_monotonicity_score"] = float(pa_monotonicity_score)
    out["pa_monotonicity_applicable"] = float(monotonicity_applicable)
    out["pa_supported_progress_score"] = float(pa_supported_progress_score)

    if monotonicity_applicable:
        pa_reward = (
            0.60 * pa_prefix_support_score
            + 0.20 * pa_monotonicity_score
            + 0.20 * pa_supported_progress_score
        )
    else:
        pa_reward = (
            0.75 * pa_prefix_support_score
            + 0.25 * pa_supported_progress_score
        )

    out["pa_reward"] = float(max(0.0, min(1.0, pa_reward)))
    out["reward_status"] = (
        "success"
        if not out["list_pa_errors"]
        and not out["list_valid_s_prefix_parse_errors"]
        else "partial_success"
    )
    if last_pa_key is not None:
        out["last_pa_evaluated"] = last_pa_key
    return out
