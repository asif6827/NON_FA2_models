#!/usr/bin/env python3
"""
Analyze two ZebraLogic GPT logs (Original prompt vs NSS prompt) beyond accuracy.

The script performs four analyses:

1) Wrong-cell severity by ZebraLogic difficulty.
2) First erroneous S-step (GT-inconsistent) + deduction type.
3) Four-way reasoning-vs-answer taxonomy:
       R✓ A✓, R✗ A✗, R✓ A✗, R✗ A✓
4) Paired error taxonomy for NSS fixes vs NSS regressions, including
   PA failures for the NSS trace.

Important interpretation
------------------------
"Reasoning correct" here means:
    every extracted S_i is parseable/evaluable and TRUE under the final
    ground-truth assignment.

This is a GT-consistency analysis. It does NOT prove that an S_i is entailed
by the clues or by the preceding reasoning prefix. For formal entailment,
reuse the project's Z3 validator.

Default inputs:
    ./Input_Logs/gpt51_outputs_test_700_temp_0.jsonl
    ./Input_Logs/gpt51_outputs_test_700_mlxl_nss_temp_0.jsonl

Example:
    python Analyze_prompt_error_patterns.py

or:
    python Analyze_prompt_error_patterns.py \
        --original ./Input_Logs/gpt51_outputs_test_700_temp_0.jsonl \
        --nss ./Input_Logs/gpt51_outputs_test_700_mlxl_nss_temp_0.jsonl \
        --output-dir ./Outputs/Error_Analysis
"""

import argparse
import ast
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# ZebraLogic difficulty definitions supplied by the user / paper
# ============================================================================

DIFFICULTY_SIZES = {
    "Small": {
        "2x2", "2x3", "2x4", "2x5", "2x6", "3x2", "3x3", "4x2"
    },
    "Medium": {
        "3x4", "3x5", "3x6", "4x3", "4x4", "5x2", "6x2"
    },
    "Large": {
        "4x5", "5x3", "4x6", "5x4", "6x3"
    },
    "XL": {
        "5x5", "6x4", "5x6", "6x5", "6x6"
    },
}

DIFFICULTY_ORDER = ["Small", "Medium", "Large", "XL", "Unknown"]


def normalize_size(size: Any) -> str:
    """Normalize '6*2', '6x2', '6×2' -> '6x2'."""
    s = str(size or "").strip().lower().replace("×", "x").replace("*", "x")
    s = re.sub(r"\s+", "", s)
    m = re.fullmatch(r"(\d+)x(\d+)", s)
    return f"{int(m.group(1))}x{int(m.group(2))}" if m else s


def difficulty_of(size: Any) -> str:
    s = normalize_size(size)
    for difficulty, sizes in DIFFICULTY_SIZES.items():
        if s in sizes:
            return difficulty
    return "Unknown"


# ============================================================================
# Generic parsing / normalization
# ============================================================================


def canonicalize(value: Any) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[\s\-_]+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def parse_outer_record(line: str) -> Dict[str, Any]:
    """Parse one JSONL record, retaining compatibility with old wrapped rows."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as original_error:
        stripped = line.strip()
        if stripped.startswith('{"prompt": {'):
            try:
                record = json.loads(stripped + "}")
            except json.JSONDecodeError:
                raise original_error
        else:
            raise original_error

    if isinstance(record, dict) and isinstance(record.get("prompt"), dict):
        record = record["prompt"]
    if not isinstance(record, dict):
        raise ValueError("JSONL row is not a JSON object")
    return record


def read_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(parse_outer_record(line))
            except Exception as e:
                raise ValueError(f"Failed parsing {path} line {line_no}: {e}") from e
    return records


def _balanced_json_value(text: str, marker: str) -> Optional[Any]:
    """
    Extract a JSON object/list immediately after marker, e.g. '"reasoning"'.
    Useful when the complete <answer> payload is malformed but the target value
    is still valid JSON.
    """
    if not isinstance(text, str):
        return None
    marker_pos = text.find(marker)
    if marker_pos < 0:
        return None
    colon = text.find(":", marker_pos + len(marker))
    if colon < 0:
        return None

    obj_pos = text.find("{", colon)
    arr_pos = text.find("[", colon)
    starts = [p for p in (obj_pos, arr_pos) if p >= 0]
    if not starts:
        return None
    start = min(starts)
    opening = text[start]
    closing = "}" if opening == "{" else "]"

    stack = []
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            top = stack[-1]
            if (top == "{" and ch != "}") or (top == "[" and ch != "]"):
                return None
            stack.pop()
            if not stack:
                chunk = text[start:i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return None
    return None


def extract_payload_components(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recover full answer payload if possible; otherwise recover reasoning and
    solution independently.
    """
    text = record.get("llm_output", "")
    payload = None
    full_payload_ok = False

    if isinstance(text, str):
        match = re.search(
            r"<answer>\s*(\{.*\})\s*</answer>",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            try:
                candidate = json.loads(match.group(1))
                if isinstance(candidate, dict):
                    payload = candidate
                    full_payload_ok = True
            except json.JSONDecodeError:
                pass

    if payload is not None:
        reasoning = payload.get("reasoning")
        solution = payload.get("solution")
        method = "full_answer_json"
    else:
        reasoning = _balanced_json_value(text, '"reasoning"')
        solution = _balanced_json_value(text, '"solution"')
        method = "fallback_components"

    return {
        "payload": payload,
        "reasoning": reasoning,
        "solution": solution if isinstance(solution, dict) else None,
        "full_payload_ok": full_payload_ok,
        "extraction_method": method,
    }


# ============================================================================
# Strict table evaluation
# ============================================================================


def normalize_table_strict(table: Any) -> Optional[Tuple[Dict[int, Dict[str, str]], List[str]]]:
    """
    Normalize table by House. Reject malformed rows, duplicate houses, duplicate
    canonicalized headers, or missing House column.
    """
    if not isinstance(table, dict):
        return None
    header = table.get("header")
    rows = table.get("rows")
    if not isinstance(header, list) or not header or not isinstance(rows, list):
        return None

    headers = [canonicalize(h) for h in header]
    if len(headers) != len(set(headers)):
        return None
    if "house" not in headers:
        return None
    house_idx = headers.index("house")
    attrs = [h for i, h in enumerate(headers) if i != house_idx]

    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != len(header):
            return None
        try:
            house = int(str(row[house_idx]).strip())
        except (TypeError, ValueError):
            return None
        if house in out:
            return None

        values = {}
        for i, h in enumerate(headers):
            if i == house_idx:
                continue
            values[h] = canonicalize(row[i])
        out[house] = values

    return (out, attrs) if out else None


def compute_case_accuracy(gt: Any, prediction: Any) -> Dict[str, Any]:
    gt_norm = normalize_table_strict(gt)
    if gt_norm is None:
        raise ValueError("Invalid ground-truth table")
    gt_by_house, gt_cols = gt_norm
    total_cells = sum(len(v) for v in gt_by_house.values())

    pred_norm = normalize_table_strict(prediction) if prediction is not None else None
    if pred_norm is None:
        return {
            "puzzle_accuracy": 0.0,
            "cell_accuracy": 0.0,
            "correct_cells": 0,
            "wrong_cells": total_cells,
            "total_cells": total_cells,
            "structure_valid": False,
        }

    pred_by_house, pred_cols = pred_norm
    correct = 0
    for house, gt_values in gt_by_house.items():
        pred_values = pred_by_house.get(house, {})
        for col, gt_value in gt_values.items():
            if pred_values.get(col) == gt_value:
                correct += 1

    exact = (
        correct == total_cells
        and set(pred_by_house) == set(gt_by_house)
        and set(pred_cols) == set(gt_cols)
    )
    return {
        "puzzle_accuracy": 1.0 if exact else 0.0,
        "cell_accuracy": correct / total_cells if total_cells else 0.0,
        "correct_cells": correct,
        "wrong_cells": total_cells - correct,
        "total_cells": total_cells,
        "structure_valid": True,
    }


def severity_bucket(metrics: Dict[str, Any]) -> str:
    if metrics["puzzle_accuracy"] == 1.0:
        return "correct"
    if not metrics["structure_valid"]:
        return "structural/unparseable"
    wrong = int(metrics["wrong_cells"])
    if wrong <= 1:
        return "1 cell"
    if wrong <= 3:
        return "2-3 cells"
    if wrong <= 6:
        return "4-6 cells"
    return "7+ cells"


# ============================================================================
# GT environment and S-step evaluator
# ============================================================================


def gt_entity_to_house(gt: Dict[str, Any]) -> Tuple[Dict[str, int], set]:
    norm = normalize_table_strict(gt)
    if norm is None:
        raise ValueError("Invalid GT")
    by_house, _ = norm
    env: Dict[str, int] = {}
    ambiguous = set()
    for house, values in by_house.items():
        for value in values.values():
            key = canonicalize(value)
            if not key:
                continue
            if key in env and env[key] != house:
                ambiguous.add(key)
            else:
                env[key] = house
    for key in ambiguous:
        env.pop(key, None)
    return env, ambiguous


class StepEvalError(Exception):
    pass


def eval_s_ast(node: ast.AST, env: Dict[str, int]) -> Any:
    if isinstance(node, ast.Expression):
        return eval_s_ast(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise StepEvalError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        key = canonicalize(node.id)
        if key not in env:
            raise StepEvalError(f"Unknown entity token: {node.id}")
        return env[key]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_s_ast(node.operand, env)
    if isinstance(node, ast.BinOp):
        left = eval_s_ast(node.left, env)
        right = eval_s_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        raise StepEvalError(f"Unsupported arithmetic op: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = eval_s_ast(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = eval_s_ast(comparator, env)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            else:
                raise StepEvalError(f"Unsupported comparison: {type(op).__name__}")
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id.lower()
        if name == "and":
            return all(bool(eval_s_ast(arg, env)) for arg in node.args)
        if name == "or":
            return any(bool(eval_s_ast(arg, env)) for arg in node.args)
        if name == "not":
            if len(node.args) != 1:
                raise StepEvalError("Not() requires one argument")
            return not bool(eval_s_ast(node.args[0], env))
        raise StepEvalError(f"Unsupported function: {node.func.id}")
    raise StepEvalError(f"Unsupported AST node: {type(node).__name__}")


def evaluate_s_expression(expr: str, env: Dict[str, int]) -> Tuple[Optional[bool], Optional[str]]:
    clean = str(expr).strip()
    if clean.endswith("."):
        clean = clean[:-1].strip()
    try:
        tree = ast.parse(clean, mode="eval")
        value = eval_s_ast(tree, env)
        if not isinstance(value, bool):
            raise StepEvalError("Expression did not evaluate to Boolean")
        return value, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def extract_s_steps(reasoning: Any) -> List[Dict[str, Any]]:
    steps = []
    if isinstance(reasoning, list):
        for order, item in enumerate(reasoning):
            if not isinstance(item, str):
                continue
            m = re.match(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", item, flags=re.I | re.S)
            if m:
                steps.append({
                    "key": f"S{int(m.group(1))}",
                    "k": int(m.group(1)),
                    "expr": m.group(2).strip(),
                    "order": order,
                })
    elif isinstance(reasoning, dict):
        for order, (key, value) in enumerate(reasoning.items()):
            m = re.fullmatch(r"S(\d+)", str(key), flags=re.I)
            if m and isinstance(value, str):
                steps.append({
                    "key": f"S{int(m.group(1))}",
                    "k": int(m.group(1)),
                    "expr": value.strip(),
                    "order": order,
                })
    return steps


def _contains_arithmetic(node: ast.AST) -> bool:
    return any(isinstance(n, ast.BinOp) for n in ast.walk(node))


def classify_deduction_type(expr: str) -> str:
    clean = str(expr).strip().rstrip(".").strip()
    try:
        tree = ast.parse(clean, mode="eval")
        root = tree.body
    except Exception:
        return "unparseable"

    if isinstance(root, ast.Call) and isinstance(root.func, ast.Name):
        name = root.func.id.lower()
        if name == "or":
            return "compound_disjunction"
        if name == "and":
            return "compound_conjunction"
        if name == "not":
            return "negative_elimination"
        return "unsupported_call"

    if isinstance(root, ast.Compare):
        if _contains_arithmetic(root):
            return "relative_position/distance"
        if any(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in root.ops):
            return "ordering"
        if any(isinstance(op, ast.NotEq) for op in root.ops):
            return "negative_elimination"
        if len(root.ops) == 1 and isinstance(root.ops[0], ast.Eq):
            left, right = root.left, root.comparators[0]
            left_num = isinstance(left, ast.Constant) and isinstance(left.value, (int, float))
            right_num = isinstance(right, ast.Constant) and isinstance(right.value, (int, float))
            if left_num or right_num:
                return "direct_house_assignment"
            return "entity_equality/co_location"
    return "other"


def analyze_reasoning(reasoning: Any, gt: Dict[str, Any]) -> Dict[str, Any]:
    env, ambiguous = gt_entity_to_house(gt)
    steps = extract_s_steps(reasoning)
    false_steps = []
    unknown_steps = []
    step_results = []

    for position, step in enumerate(steps, 1):
        value, error = evaluate_s_expression(step["expr"], env)
        item = dict(step)
        item.update({
            "position": position,
            "eval": value,
            "eval_error": error,
            "deduction_type": classify_deduction_type(step["expr"]),
        })
        step_results.append(item)
        if value is False:
            false_steps.append(item)
        elif value is None:
            unknown_steps.append(item)

    if not steps:
        status = "no_s_steps"
        reasoning_ok = False
    elif false_steps:
        status = "has_gt_inconsistent_s"
        reasoning_ok = False
    elif unknown_steps:
        status = "unverifiable_s_present"
        reasoning_ok = False
    else:
        status = "all_s_gt_consistent"
        reasoning_ok = True

    first_false = false_steps[0] if false_steps else None
    first_unknown = unknown_steps[0] if unknown_steps else None

    return {
        "reasoning_ok": reasoning_ok,
        "reasoning_status": status,
        "n_s_steps": len(steps),
        "n_false_s_steps": len(false_steps),
        "n_unverifiable_s_steps": len(unknown_steps),
        "first_false_s": first_false,
        "first_unverifiable_s": first_unknown,
        "step_results": step_results,
        "ambiguous_gt_tokens": sorted(ambiguous),
    }


# ============================================================================
# PA analysis (NSS)
# ============================================================================


def extract_pa_steps(reasoning: Any) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(reasoning, dict):
        return out
    for order, (key, value) in enumerate(reasoning.items()):
        m = re.fullmatch(r"PA(\d+)", str(key), flags=re.I)
        if m:
            out.append({
                "key": f"PA{int(m.group(1))}",
                "k": int(m.group(1)),
                "order": order,
                "pa": value,
            })
    return out


def pa_cell_map(pa: Any) -> Tuple[Optional[Dict[Tuple[int, str], str]], Optional[str]]:
    if not isinstance(pa, dict):
        return None, "PA is not a dict"
    header = pa.get("header")
    rows = pa.get("rows")
    if not isinstance(header, list) or not header or not isinstance(rows, list):
        return None, "Missing/invalid PA header or rows"

    headers = [canonicalize(h) for h in header]
    if len(headers) != len(set(headers)):
        return None, "Duplicate PA headers"
    if "house" not in headers:
        return None, "PA has no House column"
    hidx = headers.index("house")

    cells: Dict[Tuple[int, str], str] = {}
    seen_houses = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != len(header):
            return None, "PA row length mismatch"
        try:
            house = int(str(row[hidx]).strip())
        except Exception:
            return None, "Invalid PA house value"
        if house in seen_houses:
            return None, "Duplicate PA house"
        seen_houses.add(house)
        for i, col in enumerate(headers):
            if i == hidx:
                continue
            raw = str(row[i]).strip()
            if raw == "?":
                cells[(house, col)] = "?"
            else:
                cells[(house, col)] = canonicalize(raw)
    return cells, None


def analyze_pa(reasoning: Any, gt: Dict[str, Any]) -> Dict[str, Any]:
    pa_steps = extract_pa_steps(reasoning)
    gt_norm = normalize_table_strict(gt)
    if gt_norm is None:
        raise ValueError("Invalid GT for PA analysis")
    gt_by_house, gt_cols = gt_norm

    gt_cells = {
        (house, col): value
        for house, vals in gt_by_house.items()
        for col, value in vals.items()
    }

    previous_resolved: Dict[Tuple[int, str], str] = {}
    total_wrong = 0
    total_resolved = 0
    structure_errors = []
    monotonicity_violations = []
    pa_details = []
    first_gt_error = None

    for pa_step in pa_steps:
        cells, structure_error = pa_cell_map(pa_step["pa"])
        detail = {
            "key": pa_step["key"],
            "order": pa_step["order"],
            "structure_error": structure_error,
            "resolved_cells": 0,
            "wrong_gt_cells": [],
            "monotonicity_violations": [],
        }

        if structure_error is not None:
            structure_errors.append({"key": pa_step["key"], "error": structure_error})
            pa_details.append(detail)
            continue

        assert cells is not None
        current_resolved = {k: v for k, v in cells.items() if v != "?"}
        detail["resolved_cells"] = len(current_resolved)
        total_resolved += len(current_resolved)

        for cell_key, value in current_resolved.items():
            gt_value = gt_cells.get(cell_key)
            if gt_value is None or value != gt_value:
                violation = {
                    "house": cell_key[0],
                    "attribute": cell_key[1],
                    "value": value,
                    "gt": gt_value,
                }
                detail["wrong_gt_cells"].append(violation)
                total_wrong += 1
                if first_gt_error is None:
                    first_gt_error = {
                        "pa_key": pa_step["key"],
                        "order": pa_step["order"],
                        **violation,
                    }

        # Monotonicity: a previously resolved cell should remain resolved and equal.
        for cell_key, old_value in previous_resolved.items():
            new_value = cells.get(cell_key)
            if new_value is None:
                continue
            if new_value == "?":
                vtype = "forgetting"
            elif new_value != old_value:
                vtype = "revision"
            else:
                continue
            violation = {
                "from_previous": old_value,
                "to_current": new_value,
                "house": cell_key[0],
                "attribute": cell_key[1],
                "type": vtype,
                "pa_key": pa_step["key"],
            }
            monotonicity_violations.append(violation)
            detail["monotonicity_violations"].append(violation)

        # Carry all prior commitments forward conceptually; overwrite only with
        # a new resolved value so a '?' does not erase the remembered commitment.
        previous_resolved.update(current_resolved)
        pa_details.append(detail)

    if not pa_steps:
        failure_mode = "no_pa"
    elif structure_errors:
        failure_mode = "pa_structure_error"
    elif total_wrong and monotonicity_violations:
        failure_mode = "pa_gt_error_and_monotonicity"
    elif total_wrong:
        failure_mode = "pa_gt_error"
    elif monotonicity_violations:
        failure_mode = "pa_monotonicity_only"
    else:
        failure_mode = "pa_clean_vs_gt"

    return {
        "n_pa": len(pa_steps),
        "n_pa_resolved_cells": total_resolved,
        "n_pa_wrong_gt_cells": total_wrong,
        "n_pa_structure_errors": len(structure_errors),
        "n_pa_monotonicity_violations": len(monotonicity_violations),
        "first_pa_gt_error": first_gt_error,
        "pa_failure_mode": failure_mode,
        "structure_errors": structure_errors,
        "monotonicity_violations": monotonicity_violations,
        "pa_details": pa_details,
    }


# ============================================================================
# Case analysis and paired taxonomy
# ============================================================================


def analyze_case(record: Dict[str, Any], system_name: str) -> Dict[str, Any]:
    components = extract_payload_components(record)
    gt = record.get("ground_truth", {})
    metrics = compute_case_accuracy(gt, components["solution"])
    reasoning = analyze_reasoning(components["reasoning"], gt)
    pa = analyze_pa(components["reasoning"], gt) if system_name.lower() == "nss" else None

    answer_ok = metrics["puzzle_accuracy"] == 1.0
    r_ok = reasoning["reasoning_ok"]
    if r_ok and answer_ok:
        four_way = "R✓ A✓"
    elif (not r_ok) and (not answer_ok):
        four_way = "R✗ A✗"
    elif r_ok and (not answer_ok):
        four_way = "R✓ A✗"
    else:
        four_way = "R✗ A✓"

    if answer_ok:
        error_mode = "correct"
    elif not metrics["structure_valid"]:
        error_mode = "final_solution_structure_or_parse_failure"
    elif reasoning["reasoning_status"] == "has_gt_inconsistent_s":
        error_mode = "reasoning_GT_error"
    elif reasoning["reasoning_status"] == "all_s_gt_consistent":
        error_mode = "reasoning_to_answer_drift"
    elif reasoning["reasoning_status"] == "unverifiable_s_present":
        error_mode = "reasoning_unverifiable"
    else:
        error_mode = "no_s_reasoning"

    return {
        "system": system_name,
        "id": str(record.get("id", "UNKNOWN")),
        "index": record.get("index"),
        "size": normalize_size(record.get("size")),
        "difficulty": difficulty_of(record.get("size")),
        "ground_truth": gt,
        "llm_output": record.get("llm_output", ""),
        "solution": components["solution"],
        "full_payload_ok": components["full_payload_ok"],
        "extraction_method": components["extraction_method"],
        "answer": metrics,
        "severity": severity_bucket(metrics),
        "reasoning": reasoning,
        "pa": pa,
        "four_way": four_way,
        "error_mode": error_mode,
    }


def failure_origin_nss(case: Dict[str, Any]) -> str:
    reasoning = case["reasoning"]
    pa = case.get("pa") or {}
    first_s = reasoning.get("first_false_s")
    first_pa = pa.get("first_pa_gt_error")

    if first_s and first_pa:
        if first_pa["order"] < first_s["order"]:
            return "PA_GT_error_before_first_false_S"
        if first_s["order"] < first_pa["order"]:
            return "false_S_before_PA_GT_error"
        return "S_and_PA_error_same_order"
    if first_pa:
        return "PA_GT_error_without_false_S"
    if first_s:
        return "false_S_without_PA_GT_error"
    if pa.get("n_pa_monotonicity_violations", 0) > 0:
        return "PA_monotonicity_without_GT_cell_error"
    return "no_detected_S_or_PA_GT_error"


# ============================================================================
# Summaries
# ============================================================================


def mean_safe(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median_safe(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def difficulty_groups(cases: List[Dict[str, Any]]):
    groups = defaultdict(list)
    for case in cases:
        groups[case["difficulty"]].append(case)
    return groups


def summarize_wrong_cell_severity(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {}
    groups = difficulty_groups(cases)
    for difficulty in DIFFICULTY_ORDER:
        subset = groups.get(difficulty, [])
        if not subset:
            continue
        wrong = [c for c in subset if c["answer"]["puzzle_accuracy"] == 0.0]
        wrong_cells = [float(c["answer"]["wrong_cells"]) for c in wrong]
        wrong_rates = [
            c["answer"]["wrong_cells"] / c["answer"]["total_cells"]
            if c["answer"]["total_cells"] else 0.0
            for c in wrong
        ]
        buckets = Counter(c["severity"] for c in wrong)
        output[difficulty] = {
            "n": len(subset),
            "wrong_puzzles": len(wrong),
            "mean_wrong_cells": mean_safe(wrong_cells),
            "median_wrong_cells": median_safe(wrong_cells),
            "mean_wrong_cell_rate": mean_safe(wrong_rates),
            "mean_cell_accuracy_on_wrong": mean_safe([c["answer"]["cell_accuracy"] for c in wrong]),
            "severity_buckets": dict(buckets),
        }
    return output


def summarize_first_s_errors(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {}
    groups = difficulty_groups(cases)
    for difficulty in DIFFICULTY_ORDER:
        subset = groups.get(difficulty, [])
        if not subset:
            continue
        first_false = [c["reasoning"]["first_false_s"] for c in subset if c["reasoning"]["first_false_s"]]
        positions = [x["position"] for x in first_false]
        normalized_positions = []
        types = Counter()
        for c in subset:
            x = c["reasoning"]["first_false_s"]
            if x:
                n = max(c["reasoning"]["n_s_steps"], 1)
                normalized_positions.append(x["position"] / n)
                types[x["deduction_type"]] += 1
        output[difficulty] = {
            "n": len(subset),
            "cases_with_false_s": len(first_false),
            "cases_all_s_gt_consistent": sum(c["reasoning"]["reasoning_status"] == "all_s_gt_consistent" for c in subset),
            "cases_unverifiable_s": sum(c["reasoning"]["reasoning_status"] == "unverifiable_s_present" for c in subset),
            "cases_no_s": sum(c["reasoning"]["reasoning_status"] == "no_s_steps" for c in subset),
            "mean_first_false_position": mean_safe(positions),
            "mean_first_false_normalized_position": mean_safe(normalized_positions),
            "first_error_deduction_types": dict(types),
        }
    return output


def summarize_four_way(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    output = {}
    groups = difficulty_groups(cases)
    for difficulty in DIFFICULTY_ORDER:
        subset = groups.get(difficulty, [])
        if not subset:
            continue
        counts = Counter(c["four_way"] for c in subset)
        output[difficulty] = {
            "n": len(subset),
            "R✓ A✓": counts["R✓ A✓"],
            "R✗ A✗": counts["R✗ A✗"],
            "R✓ A✗": counts["R✓ A✗"],
            "R✗ A✓": counts["R✗ A✓"],
            "unverifiable_reasoning": sum(c["reasoning"]["reasoning_status"] == "unverifiable_s_present" for c in subset),
            "no_s_reasoning": sum(c["reasoning"]["reasoning_status"] == "no_s_steps" for c in subset),
        }
    return output


def summarize_paired(original_cases: List[Dict[str, Any]], nss_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_orig = {c["id"]: c for c in original_cases}
    by_nss = {c["id"]: c for c in nss_cases}
    if set(by_orig) != set(by_nss):
        missing_o = sorted(set(by_nss) - set(by_orig))[:10]
        missing_n = sorted(set(by_orig) - set(by_nss))[:10]
        raise ValueError(f"ID sets differ. Missing Original={missing_o}; Missing NSS={missing_n}")

    paired = []
    for cid in [c["id"] for c in original_cases]:
        o, n = by_orig[cid], by_nss[cid]
        if o["size"] != n["size"]:
            raise ValueError(f"Size mismatch for {cid}: {o['size']} vs {n['size']}")
        if o["ground_truth"] != n["ground_truth"]:
            raise ValueError(f"Ground truth mismatch for {cid}")

        oc = o["answer"]["puzzle_accuracy"] == 1.0
        nc = n["answer"]["puzzle_accuracy"] == 1.0
        if oc and nc:
            outcome = "both_correct"
        elif (not oc) and (not nc):
            outcome = "both_wrong"
        elif (not oc) and nc:
            outcome = "nss_fix"
        else:
            outcome = "nss_regression"
        paired.append({"id": cid, "difficulty": o["difficulty"], "size": o["size"], "outcome": outcome, "original": o, "nss": n})

    fixes = [p for p in paired if p["outcome"] == "nss_fix"]
    regressions = [p for p in paired if p["outcome"] == "nss_regression"]

    def paired_bucket_stats(rows):
        d = defaultdict(Counter)
        for p in rows:
            d[p["difficulty"]][p["outcome"]] += 1
        return {k: dict(v) for k, v in d.items()}

    fix_loser_error_modes = Counter(p["original"]["error_mode"] for p in fixes)
    fix_first_s_types = Counter(
        p["original"]["reasoning"]["first_false_s"]["deduction_type"]
        for p in fixes
        if p["original"]["reasoning"]["first_false_s"]
    )

    regression_loser_error_modes = Counter(p["nss"]["error_mode"] for p in regressions)
    regression_first_s_types = Counter(
        p["nss"]["reasoning"]["first_false_s"]["deduction_type"]
        for p in regressions
        if p["nss"]["reasoning"]["first_false_s"]
    )
    regression_pa_modes = Counter((p["nss"].get("pa") or {}).get("pa_failure_mode", "no_pa_analysis") for p in regressions)
    regression_origins = Counter(failure_origin_nss(p["nss"]) for p in regressions)

    # Paired outcome by difficulty
    diff_outcomes = {}
    for difficulty in DIFFICULTY_ORDER:
        rows = [p for p in paired if p["difficulty"] == difficulty]
        if not rows:
            continue
        c = Counter(p["outcome"] for p in rows)
        diff_outcomes[difficulty] = {
            "n": len(rows),
            "both_correct": c["both_correct"],
            "both_wrong": c["both_wrong"],
            "nss_fixes": c["nss_fix"],
            "nss_regressions": c["nss_regression"],
            "net_nss": c["nss_fix"] - c["nss_regression"],
        }

    return {
        "paired": paired,
        "fixes": fixes,
        "regressions": regressions,
        "overall_outcomes": dict(Counter(p["outcome"] for p in paired)),
        "difficulty_outcomes": diff_outcomes,
        "nss_fix_original_error_modes": dict(fix_loser_error_modes),
        "nss_fix_original_first_s_error_types": dict(fix_first_s_types),
        "nss_regression_nss_error_modes": dict(regression_loser_error_modes),
        "nss_regression_nss_first_s_error_types": dict(regression_first_s_types),
        "nss_regression_pa_failure_modes": dict(regression_pa_modes),
        "nss_regression_failure_origins": dict(regression_origins),
    }


# ============================================================================
# Report writers
# ============================================================================


def write_summary(path: Path, original_cases, nss_cases, paired_summary):
    sev_o = summarize_wrong_cell_severity(original_cases)
    sev_n = summarize_wrong_cell_severity(nss_cases)
    s_o = summarize_first_s_errors(original_cases)
    s_n = summarize_first_s_errors(nss_cases)
    fw_o = summarize_four_way(original_cases)
    fw_n = summarize_four_way(nss_cases)

    with path.open("w", encoding="utf-8") as f:
        def p(x=""):
            f.write(str(x) + "\n")
            print(x)

        p("=" * 120)
        p("GPT PROMPT ERROR ANALYSIS — ORIGINAL vs NSS")
        p("=" * 120)
        p()
        p("Definition note: R✓ means all extracted S_i steps were parseable/evaluable and TRUE under the GT assignment.")
        p("This is GT-consistency, not formal clue-prefix entailment.")
        p()

        p("1. WRONG-CELL SEVERITY BY DIFFICULTY")
        p("-" * 120)
        header = f"{'Difficulty':<10} {'System':<9} {'N':>5} {'Wrong':>6} {'MeanWrong':>10} {'Median':>8} {'WrongRate':>10} {'CAccWrong':>10}"
        p(header)
        p("-" * len(header))
        for d in DIFFICULTY_ORDER:
            for system, data in (("Original", sev_o), ("NSS", sev_n)):
                if d not in data:
                    continue
                x = data[d]
                p(f"{d:<10} {system:<9} {x['n']:>5} {x['wrong_puzzles']:>6} {x['mean_wrong_cells']:>10.2f} {x['median_wrong_cells']:>8.2f} {x['mean_wrong_cell_rate']:>10.3f} {x['mean_cell_accuracy_on_wrong']:>10.3f}")
                p(f"{'':<10} {'':<9} severity={x['severity_buckets']}")
        p()

        p("2. FIRST GT-INCONSISTENT S-STEP + DEDUCTION TYPE")
        p("-" * 120)
        header = f"{'Difficulty':<10} {'System':<9} {'N':>5} {'FalseS':>7} {'AllSOK':>7} {'Unverif':>8} {'NoS':>5} {'MeanPos':>8} {'NormPos':>8}"
        p(header)
        p("-" * len(header))
        for d in DIFFICULTY_ORDER:
            for system, data in (("Original", s_o), ("NSS", s_n)):
                if d not in data:
                    continue
                x = data[d]
                p(f"{d:<10} {system:<9} {x['n']:>5} {x['cases_with_false_s']:>7} {x['cases_all_s_gt_consistent']:>7} {x['cases_unverifiable_s']:>8} {x['cases_no_s']:>5} {x['mean_first_false_position']:>8.2f} {x['mean_first_false_normalized_position']:>8.3f}")
                p(f"{'':<10} {'':<9} first-error types={x['first_error_deduction_types']}")
        p()

        p("3. FOUR-WAY REASONING-vs-ANSWER TAXONOMY")
        p("-" * 120)
        header = f"{'Difficulty':<10} {'System':<9} {'N':>5} {'R✓A✓':>7} {'R✗A✗':>7} {'R✓A✗':>7} {'R✗A✓':>7} {'Unverif':>8} {'NoS':>5}"
        p(header)
        p("-" * len(header))
        for d in DIFFICULTY_ORDER:
            for system, data in (("Original", fw_o), ("NSS", fw_n)):
                if d not in data:
                    continue
                x = data[d]
                p(f"{d:<10} {system:<9} {x['n']:>5} {x['R✓ A✓']:>7} {x['R✗ A✗']:>7} {x['R✓ A✗']:>7} {x['R✗ A✓']:>7} {x['unverifiable_reasoning']:>8} {x['no_s_reasoning']:>5}")
        p()

        p("4. PAIRED ERROR TAXONOMY — NSS FIXES vs NSS REGRESSIONS")
        p("-" * 120)
        o = paired_summary["overall_outcomes"]
        p(f"Both correct:       {o.get('both_correct', 0)}")
        p(f"Both wrong:         {o.get('both_wrong', 0)}")
        p(f"NSS fixes:          {o.get('nss_fix', 0)}  (Original wrong, NSS correct)")
        p(f"NSS regressions:    {o.get('nss_regression', 0)}  (Original correct, NSS wrong)")
        p(f"Net NSS:            {o.get('nss_fix', 0) - o.get('nss_regression', 0):+d}")
        p()
        p("Paired outcomes by difficulty:")
        for d, x in paired_summary["difficulty_outcomes"].items():
            p(f"  {d:<8} N={x['n']:>3}  both_correct={x['both_correct']:>3}  both_wrong={x['both_wrong']:>3}  NSS_fixes={x['nss_fixes']:>3}  NSS_regressions={x['nss_regressions']:>3}  net_NSS={x['net_nss']:+d}")
        p()

        p("When NSS FIXES an Original error — what failed in Original?")
        p(f"  Original error modes: {paired_summary['nss_fix_original_error_modes']}")
        p(f"  Original first-S error types: {paired_summary['nss_fix_original_first_s_error_types']}")
        p()

        p("When NSS REGRESSES — what failed in NSS?")
        p(f"  NSS error modes: {paired_summary['nss_regression_nss_error_modes']}")
        p(f"  NSS first-S error types: {paired_summary['nss_regression_nss_first_s_error_types']}")
        p(f"  NSS PA failure modes: {paired_summary['nss_regression_pa_failure_modes']}")
        p(f"  NSS detected failure origin: {paired_summary['nss_regression_failure_origins']}")
        p()

        p("Interpretation of key error modes:")
        p("  reasoning_GT_error        = at least one S_i is false under GT.")
        p("  reasoning_to_answer_drift = every S_i is GT-consistent, but final grid is wrong.")
        p("  reasoning_unverifiable    = no false S_i found, but at least one S_i could not be evaluated.")
        p("  PA_GT_error               = a resolved PA cell disagrees with GT.")
        p("  PA_monotonicity           = a later PA forgets or changes an earlier resolved cell.")


def write_case_jsonl(path: Path, original_cases, nss_cases):
    with path.open("w", encoding="utf-8") as f:
        for o, n in zip(original_cases, nss_cases):
            compact = {
                "id": o["id"],
                "size": o["size"],
                "difficulty": o["difficulty"],
                "original": {
                    "answer": o["answer"],
                    "severity": o["severity"],
                    "four_way": o["four_way"],
                    "error_mode": o["error_mode"],
                    "reasoning_status": o["reasoning"]["reasoning_status"],
                    "first_false_s": o["reasoning"]["first_false_s"],
                    "n_unverifiable_s": o["reasoning"]["n_unverifiable_s_steps"],
                },
                "nss": {
                    "answer": n["answer"],
                    "severity": n["severity"],
                    "four_way": n["four_way"],
                    "error_mode": n["error_mode"],
                    "reasoning_status": n["reasoning"]["reasoning_status"],
                    "first_false_s": n["reasoning"]["first_false_s"],
                    "n_unverifiable_s": n["reasoning"]["n_unverifiable_s_steps"],
                    "pa": n["pa"],
                },
            }
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")


def write_discordant_cases(path: Path, rows: List[Dict[str, Any]], title: str):
    with path.open("w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * 120 + "\n\n")
        for i, p in enumerate(rows, 1):
            o, n = p["original"], p["nss"]
            f.write(f"CASE {i} | ID={p['id']} | size={p['size']} | difficulty={p['difficulty']}\n")
            f.write("-" * 120 + "\n")
            f.write(f"Outcome: {p['outcome']}\n")
            f.write(f"Original: error_mode={o['error_mode']}, severity={o['severity']}, four_way={o['four_way']}\n")
            f.write(f"Original first false S: {json.dumps(o['reasoning']['first_false_s'], ensure_ascii=False)}\n")
            f.write(f"NSS: error_mode={n['error_mode']}, severity={n['severity']}, four_way={n['four_way']}\n")
            f.write(f"NSS first false S: {json.dumps(n['reasoning']['first_false_s'], ensure_ascii=False)}\n")
            f.write(f"NSS PA mode: {(n.get('pa') or {}).get('pa_failure_mode')}\n")
            f.write(f"NSS failure origin: {failure_origin_nss(n)}\n")
            f.write(f"NSS first PA GT error: {json.dumps((n.get('pa') or {}).get('first_pa_gt_error'), ensure_ascii=False)}\n")
            f.write(f"NSS PA monotonicity violations: {json.dumps((n.get('pa') or {}).get('monotonicity_violations', []), ensure_ascii=False)}\n")
            f.write("\nGROUND TRUTH:\n")
            f.write(json.dumps(o["ground_truth"], indent=2, ensure_ascii=False) + "\n")
            f.write("\nORIGINAL SOLUTION:\n")
            f.write(json.dumps(o["solution"], indent=2, ensure_ascii=False) + "\n")
            f.write("\nNSS SOLUTION:\n")
            f.write(json.dumps(n["solution"], indent=2, ensure_ascii=False) + "\n")
            f.write("\nORIGINAL LLM OUTPUT:\n")
            f.write(str(o["llm_output"]) + "\n")
            f.write("\nNSS LLM OUTPUT:\n")
            f.write(str(n["llm_output"]) + "\n")
            f.write("\n" + "=" * 120 + "\n\n\n")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze Original vs NSS ZebraLogic error patterns")
    parser.add_argument(
        "--original",
        default="./Input_Logs/gpt51_outputs_test_700_temp_0.jsonl",
        help="Original-prompt JSONL log",
    )
    parser.add_argument(
        "--nss",
        default="./Input_Logs/gpt51_outputs_test_700_mlxl_nss_temp_0.jsonl",
        help="NSS-prompt JSONL log",
    )
    parser.add_argument(
        "--output-dir",
        default="./Outputs/Error_Analysis",
        help="Directory for analysis outputs",
    )
    args = parser.parse_args()

    original_path = Path(args.original)
    nss_path = Path(args.nss)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_records = read_records(original_path)
    nss_records = read_records(nss_path)

    # Analyze by ID rather than assuming line-order equivalence.
    original_cases = [analyze_case(r, "Original") for r in original_records]
    nss_cases_unsorted = [analyze_case(r, "NSS") for r in nss_records]
    nss_map = {c["id"]: c for c in nss_cases_unsorted}

    if len({c["id"] for c in original_cases}) != len(original_cases):
        raise ValueError("Duplicate IDs in Original log")
    if len(nss_map) != len(nss_cases_unsorted):
        raise ValueError("Duplicate IDs in NSS log")

    missing = [c["id"] for c in original_cases if c["id"] not in nss_map]
    extra = [cid for cid in nss_map if cid not in {c["id"] for c in original_cases}]
    if missing or extra:
        raise ValueError(f"Logs have different ID sets. Missing NSS={missing[:10]}, extra NSS={extra[:10]}")

    nss_cases = [nss_map[c["id"]] for c in original_cases]
    paired = summarize_paired(original_cases, nss_cases)

    summary_path = out_dir / "prompt_error_analysis_summary.txt"
    case_jsonl_path = out_dir / "case_level_error_analysis.jsonl"
    fixes_path = out_dir / "nss_fixes_detailed.txt"
    regressions_path = out_dir / "nss_regressions_detailed.txt"

    write_summary(summary_path, original_cases, nss_cases, paired)
    write_case_jsonl(case_jsonl_path, original_cases, nss_cases)
    write_discordant_cases(
        fixes_path,
        paired["fixes"],
        "NSS FIXES — Original wrong, NSS correct",
    )
    write_discordant_cases(
        regressions_path,
        paired["regressions"],
        "NSS REGRESSIONS — Original correct, NSS wrong",
    )

    print("\nSaved:")
    for p in (summary_path, case_jsonl_path, fixes_path, regressions_path):
        print("  ", p.resolve())


if __name__ == "__main__":
    main()
