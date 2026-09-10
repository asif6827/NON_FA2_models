#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paired error-difference analysis for two ZebraLogic prompt logs.

This script compares two aligned JSONL logs:
  - Original prompt
  - NSS / PA prompt

It focuses on cases where the final puzzle outcome differs:

  NSS FIX:
      Original final answer is wrong
      NSS final answer is correct

  NSS REGRESSION:
      Original final answer is correct
      NSS final answer is wrong

For those paired cases, it analyzes:

1. First genuine S-step error
   - A genuine S-step error is an S_i that is FALSE under the ground-truth
     entity-to-house assignment.
   - The script reports the first GT-wrong step, its normalized position,
     and its deduction type.

2. Error timing difference
   - Which system makes the first genuine S-step error earlier?
   - delta_first_error_position = NSS normalized first-error position
                                  - Original normalized first-error position
   - Positive means NSS stays GT-consistent longer.
   - If one trace has no GT-wrong S-step, it is treated separately.

3. Error cascade / recovery
   After the first GT-wrong S-step, the script measures:
   - number of later GT-wrong S-steps
   - number of later GT-consistent S-steps
   - whether any recovery occurs
   - whether the final S-step is GT-consistent
   - final wrong-cell count

4. Reasoning-to-answer drift
   - Final answer is wrong BUT no GT-wrong S-step was detected.
   - This does NOT prove the reasoning is formally valid; it means the
     displayed S-trace never directly contradicts the GT.

5. NSS PA analysis
   For NSS traces containing PA_i checkpoints:
   - first PA containing a GT-wrong resolved cell
   - number of wrong resolved PA cells
   - PA monotonicity violations:
       * forgetting: known cell -> "?"
       * revision:   known cell -> different known value
   - ordering between first S error and first PA error:
       S_BEFORE_PA
       PA_BEFORE_S
       SAME_REGION
       S_ERROR_ONLY
       PA_ERROR_ONLY
       NO_S_OR_PA_ERROR

6. Deduction-type taxonomy for the first GT-wrong S-step
   - direct_house_assignment
   - negative_elimination
   - entity_equality_or_colocation
   - ordering
   - relative_position_or_distance
   - compound_disjunction
   - compound_conjunction
   - negation
   - other

7. Reasoning-length comparison
   - number of S-steps in Original vs NSS
   - whether NSS fixes/regressions correlate with longer traces

Outputs:
  ./Outputs/Paired_Error_Analysis/
      paired_error_analysis_summary.txt
      paired_case_analysis.jsonl
      nss_fixes_detailed.txt
      nss_regressions_detailed.txt

      NSS_FIXES/
          size_summary.txt
          <size>/
              <case>.txt

      NSS_REGRESSES/
          size_summary.txt
          <size>/
              <case>.txt

Requirements:
  Python 3.9+
  No Z3 dependency is required for this script.

Typical usage:

  python Analyze_paired_error_differences.py

or:

  python Analyze_paired_error_differences.py \
      --original ./Input_Logs/gpt51_outputs_test_700_temp_0.jsonl \
      --nss ./Input_Logs/gpt51_outputs_test_700_mlxl_nss_temp_0.jsonl \
      --output-dir ./Outputs/Paired_Error_Analysis

The script automatically uses:
    ./Input_Logs/pid_to_puzzle_dic.json

You can override it with:
    --source-dataset <other .json/.jsonl/.parquet file>
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =============================================================================
# ZebraLogic difficulty buckets
# =============================================================================

DIFFICULTY_BY_SIZE = {
    # Small
    "2x2": "Small", "2x3": "Small", "2x4": "Small", "2x5": "Small",
    "2x6": "Small", "3x2": "Small", "3x3": "Small", "4x2": "Small",

    # Medium
    "3x4": "Medium", "3x5": "Medium", "3x6": "Medium", "4x3": "Medium",
    "4x4": "Medium", "5x2": "Medium", "6x2": "Medium",

    # Large
    "4x5": "Large", "5x3": "Large", "4x6": "Large", "5x4": "Large",
    "6x3": "Large",

    # XL
    "5x5": "XL", "6x4": "XL", "5x6": "XL", "6x5": "XL", "6x6": "XL",
}

DIFFICULTY_ORDER = ["Small", "Medium", "Large", "XL", "Overall"]


def normalize_size(size: Any) -> str:
    s = str(size).strip().lower().replace("*", "x").replace("×", "x")
    return re.sub(r"\s+", "", s)


# =============================================================================
# Generic JSON / payload extraction
# =============================================================================

def parse_outer_record(line: str) -> Dict[str, Any]:
    """
    Parse one outer JSONL line.

    A small legacy repair is retained for an occasional record wrapped as:
        {"prompt": {...}
    with one final brace missing.
    """
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
        raise ValueError("Outer JSONL record is not a dictionary.")

    return record


def extract_answer_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(
        r"<answer\b[^>]*>(.*?)</answer\s*>",
        text,
        flags=re.I | re.S,
    )
    return m.group(1).strip() if m else text.strip()


def balanced_json_value(text: str, marker: str) -> Any:
    """
    Recover the JSON object/list immediately following `marker`.
    """
    if not isinstance(text, str):
        return None

    pos = text.find(marker)
    if pos < 0:
        return None

    colon = text.find(":", pos + len(marker))
    if colon < 0:
        return None

    i = colon + 1
    while i < len(text) and text[i].isspace():
        i += 1

    if i >= len(text) or text[i] not in "[{":
        return None

    opener = text[i]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_string = False
    escaped = False

    for j in range(i, len(text)):
        ch = text[j]

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
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    return None

    return None


def extract_payload_components(record: Dict[str, Any]) -> Dict[str, Any]:
    text = record.get("llm_output", "")
    answer_text = extract_answer_text(text)

    try:
        payload = json.loads(answer_text)
        if isinstance(payload, dict):
            return {
                "payload": payload,
                "reasoning": payload.get("reasoning"),
                "solution": payload.get("solution"),
                "n_houses": payload.get("n_houses"),
                "attribute_values": payload.get("attribute_values"),
                "syntactic_clues": payload.get("syntactic_clues"),
                "full_payload_ok": True,
                "extraction_method": "full_answer_json",
            }
    except Exception:
        pass

    # Best-effort component recovery if the complete answer JSON is malformed.
    return {
        "payload": None,
        "reasoning": balanced_json_value(answer_text, '"reasoning"'),
        "solution": balanced_json_value(answer_text, '"solution"'),
        "attribute_values": balanced_json_value(answer_text, '"attribute_values"'),
        "syntactic_clues": balanced_json_value(answer_text, '"syntactic_clues"'),
        "n_houses": None,
        "full_payload_ok": False,
        "extraction_method": "fallback_components",
    }


# =============================================================================
# Canonicalization / strict table accuracy
# =============================================================================

def canonicalize(value: Any) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[\s\-_]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def normalize_table_strict(
    table: Any,
) -> Optional[Tuple[Dict[int, Dict[str, str]], List[str]]]:
    if not isinstance(table, dict):
        return None

    header = table.get("header")
    rows = table.get("rows")

    if not isinstance(header, list) or not header or not isinstance(rows, list):
        return None

    headers = [canonicalize(h) for h in header]

    if len(headers) != len(set(headers)):
        return None

    try:
        house_idx = headers.index("house")
    except ValueError:
        return None

    attrs = [h for i, h in enumerate(headers) if i != house_idx]

    out: Dict[int, Dict[str, str]] = {}

    for row in rows:
        if not isinstance(row, list) or len(row) != len(headers):
            return None

        try:
            house = int(row[house_idx])
        except Exception:
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


def compute_case_accuracy(gt: Any, pred: Any) -> Dict[str, Any]:
    gt_norm = normalize_table_strict(gt)

    if gt_norm is None:
        raise ValueError("Invalid ground truth table.")

    gt_by_house, gt_cols = gt_norm
    total = sum(len(v) for v in gt_by_house.values())

    pred_norm = normalize_table_strict(pred)

    if pred_norm is None:
        return {
            "puzzle_accuracy": 0.0,
            "cell_accuracy": 0.0,
            "correct_cells": 0,
            "wrong_cells": total,
            "total_cells": total,
            "structural_failure": True,
        }

    pred_by_house, pred_cols = pred_norm
    correct = 0

    for h, gt_vals in gt_by_house.items():
        pv = pred_by_house.get(h, {})
        for col, val in gt_vals.items():
            if pv.get(col) == val:
                correct += 1

    exact = (
        correct == total
        and set(pred_by_house) == set(gt_by_house)
        and set(pred_cols) == set(gt_cols)
    )

    return {
        "puzzle_accuracy": 1.0 if exact else 0.0,
        "cell_accuracy": correct / total if total else 0.0,
        "correct_cells": correct,
        "wrong_cells": total - correct,
        "total_cells": total,
        "structural_failure": False,
    }


# =============================================================================
# Ground-truth entity environment
# =============================================================================

def gt_entity_to_house(gt: Dict[str, Any]) -> Tuple[Dict[str, int], set]:
    norm = normalize_table_strict(gt)

    if norm is None:
        raise ValueError("Invalid GT table.")

    by_house, _ = norm
    env: Dict[str, int] = {}
    ambiguous = set()

    for house, vals in by_house.items():
        for value in vals.values():
            key = canonicalize(value)
            if key in env and env[key] != house:
                ambiguous.add(key)
            else:
                env[key] = house

    for key in ambiguous:
        env.pop(key, None)

    return env, ambiguous


# =============================================================================
# S-step extraction
# =============================================================================

def extract_s_steps(reasoning: Any) -> List[Dict[str, Any]]:
    """
    Return ordered S-step records:
        {
            "k": int,
            "key": "S3",
            "expr": "...",
            "nl": "...",
            "order": raw reasoning order
        }

    Supports:
      NSS dictionary format:
        {"NL1": "...", "S1": "...", "PA1": {...}, ...}

      Original list format:
        ["NL text", "S1: ...", "NL text", "S2: ...", ...]
    """
    steps: List[Dict[str, Any]] = []

    if isinstance(reasoning, dict):
        for order, (key, value) in enumerate(reasoning.items()):
            m = re.fullmatch(r"S(\d+)", str(key), flags=re.I)
            if not m or not isinstance(value, str):
                continue

            k = int(m.group(1))
            nl = reasoning.get(f"NL{k}", "")
            if not isinstance(nl, str):
                nl = ""

            steps.append({
                "k": k,
                "key": f"S{k}",
                "expr": value.strip(),
                "nl": nl.strip(),
                "order": order,
            })

        steps.sort(key=lambda x: x["order"])
        return steps

    if isinstance(reasoning, list):
        pending_nl: List[str] = []

        for order, item in enumerate(reasoning):
            if not isinstance(item, str):
                continue

            m = re.match(
                r"^\s*S(\d+)\s*:\s*(.+?)\s*$",
                item,
                flags=re.I | re.S,
            )

            if m:
                k = int(m.group(1))
                steps.append({
                    "k": k,
                    "key": f"S{k}",
                    "expr": m.group(2).strip(),
                    "nl": " ".join(pending_nl).strip(),
                    "order": order,
                })
                pending_nl = []
            else:
                clean = re.sub(
                    r"^\s*NL\d+\s*:\s*",
                    "",
                    item,
                    flags=re.I,
                )
                pending_nl.append(clean.strip())

        return steps

    return steps


# =============================================================================
# Evaluate S_i against GT
# =============================================================================

class GTEvalError(Exception):
    pass


def eval_ast_gt(node: ast.AST, env: Dict[str, int]) -> Any:
    if isinstance(node, ast.Expression):
        return eval_ast_gt(node.body, env)

    if isinstance(node, ast.Name):
        key = canonicalize(node.id)
        if key not in env:
            raise GTEvalError(f"Unknown entity token: {node.id}")
        return env[key]

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, bool)):
            return node.value
        raise GTEvalError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_ast_gt(node.operand, env)

    if isinstance(node, ast.BinOp):
        l = eval_ast_gt(node.left, env)
        r = eval_ast_gt(node.right, env)

        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r

        raise GTEvalError(
            f"Unsupported arithmetic operator: {type(node.op).__name__}"
        )

    if isinstance(node, ast.Compare):
        left = eval_ast_gt(node.left, env)

        for op, comp in zip(node.ops, node.comparators):
            right = eval_ast_gt(comp, env)

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
                raise GTEvalError(
                    f"Unsupported comparison: {type(op).__name__}"
                )

            if not ok:
                return False

            left = right

        return True

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id.lower()

        if fn == "and":
            return all(bool(eval_ast_gt(a, env)) for a in node.args)

        if fn == "or":
            return any(bool(eval_ast_gt(a, env)) for a in node.args)

        if fn == "not":
            if len(node.args) != 1:
                raise GTEvalError("Not requires exactly one argument.")
            return not bool(eval_ast_gt(node.args[0], env))

        raise GTEvalError(f"Unsupported function: {node.func.id}")

    raise GTEvalError(f"Unsupported AST node: {type(node).__name__}")


def eval_expr_gt(
    expr: Any,
    env: Dict[str, int],
) -> Tuple[Optional[bool], Optional[str]]:
    clean = str(expr).strip().rstrip(".").strip()

    try:
        tree = ast.parse(clean, mode="eval")
        value = eval_ast_gt(tree, env)

        if not isinstance(value, bool):
            raise GTEvalError("Expression did not evaluate to Boolean.")

        return value, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# =============================================================================
# Deduction type
# =============================================================================

def deduction_type(expr: Any) -> str:
    clean = str(expr).strip().rstrip(".").strip()

    try:
        tree = ast.parse(clean, mode="eval")
        node = tree.body
    except Exception:
        return "unparseable"

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id.lower()

        if fn == "or":
            return "compound_disjunction"
        if fn == "and":
            return "compound_conjunction"
        if fn == "not":
            return "negation"

        return "other_function"

    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = node.ops[0]
        left = node.left
        right = node.comparators[0]

        # Entity == integer
        if isinstance(op, ast.Eq):
            if (
                isinstance(left, ast.Name)
                and isinstance(right, ast.Constant)
                and isinstance(right.value, int)
            ) or (
                isinstance(right, ast.Name)
                and isinstance(left, ast.Constant)
                and isinstance(left.value, int)
            ):
                return "direct_house_assignment"

            if isinstance(left, ast.Name) and isinstance(right, ast.Name):
                return "entity_equality_or_colocation"

            if isinstance(left, ast.BinOp) or isinstance(right, ast.BinOp):
                return "relative_position_or_distance"

            return "equality_other"

        if isinstance(op, ast.NotEq):
            return "negative_elimination"

        if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            if isinstance(left, ast.BinOp) or isinstance(right, ast.BinOp):
                return "relative_position_or_distance"
            return "ordering"

    return "other"


# =============================================================================
# S-trace diagnostics
# =============================================================================

def analyze_s_trace(
    reasoning: Any,
    gt_env: Dict[str, int],
) -> Dict[str, Any]:
    steps = extract_s_steps(reasoning)

    details: List[Dict[str, Any]] = []

    for pos, step in enumerate(steps, 1):
        gt_value, err = eval_expr_gt(step["expr"], gt_env)

        details.append({
            **step,
            "position": pos,
            "normalized_position": (
                pos / len(steps) if steps else None
            ),
            "gt_consistent": gt_value,
            "gt_eval_error": err,
            "deduction_type": deduction_type(step["expr"]),
        })

    false_steps = [
        d for d in details
        if d["gt_consistent"] is False
    ]

    first_false = false_steps[0] if false_steps else None

    if first_false is None:
        later_false = 0
        later_true = 0
        later_unverifiable = 0
        recovery_any = False
        last_verifiable_consistent = None

    else:
        after = [
            d for d in details
            if d["position"] > first_false["position"]
        ]

        later_false = sum(
            d["gt_consistent"] is False
            for d in after
        )

        later_true = sum(
            d["gt_consistent"] is True
            for d in after
        )

        later_unverifiable = sum(
            d["gt_consistent"] is None
            for d in after
        )

        recovery_any = later_true > 0

        verifiable_after_or_at = [
            d for d in details
            if d["position"] >= first_false["position"]
            and d["gt_consistent"] is not None
        ]

        last_verifiable_consistent = (
            verifiable_after_or_at[-1]["gt_consistent"]
            if verifiable_after_or_at
            else None
        )

    return {
        "n_s_steps": len(steps),
        "n_gt_correct_s": sum(
            d["gt_consistent"] is True
            for d in details
        ),
        "n_gt_wrong_s": len(false_steps),
        "n_unverifiable_s": sum(
            d["gt_consistent"] is None
            for d in details
        ),

        "first_false_s": first_false,
        "first_false_position": (
            first_false["position"]
            if first_false
            else None
        ),
        "first_false_normalized_position": (
            first_false["normalized_position"]
            if first_false
            else None
        ),
        "first_false_type": (
            first_false["deduction_type"]
            if first_false
            else None
        ),

        "n_later_gt_wrong_s": int(later_false),
        "n_later_gt_correct_s": int(later_true),
        "n_later_unverifiable_s": int(later_unverifiable),

        "recovery_any": bool(recovery_any),
        "last_verifiable_consistent_after_error": (
            last_verifiable_consistent
        ),

        "step_details": details,
    }


# =============================================================================
# PA extraction / diagnostics
# =============================================================================

def extract_pa_steps(reasoning: Any) -> List[Dict[str, Any]]:
    """
    NSS PA_i checkpoints are expected inside the reasoning dictionary.
    """
    if not isinstance(reasoning, dict):
        return []

    out = []

    for order, (key, value) in enumerate(reasoning.items()):
        m = re.fullmatch(r"PA(\d+)", str(key), flags=re.I)

        if not m or not isinstance(value, dict):
            continue

        out.append({
            "k": int(m.group(1)),
            "key": f"PA{int(m.group(1))}",
            "pa": value,
            "order": order,
        })

    out.sort(key=lambda x: x["order"])
    return out


def pa_grid_by_house(
    pa: Any,
) -> Optional[Tuple[Dict[int, Dict[str, str]], List[str]]]:
    """
    Parse PA allowing "?" unresolved cells.
    """
    if not isinstance(pa, dict):
        return None

    header = pa.get("header")
    rows = pa.get("rows")

    if not isinstance(header, list) or not header or not isinstance(rows, list):
        return None

    headers = [canonicalize(h) for h in header]

    if len(headers) != len(set(headers)):
        return None

    try:
        house_idx = headers.index("house")
    except ValueError:
        return None

    attrs = [h for i, h in enumerate(headers) if i != house_idx]
    out: Dict[int, Dict[str, str]] = {}

    for row in rows:
        if not isinstance(row, list) or len(row) != len(headers):
            return None

        try:
            house = int(row[house_idx])
        except Exception:
            return None

        if house in out:
            return None

        vals = {}

        for i, h in enumerate(headers):
            if i == house_idx:
                continue

            raw = str(row[i]).strip()

            if raw == "?":
                vals[h] = "?"
            else:
                vals[h] = canonicalize(raw)

        out[house] = vals

    return (out, attrs) if out else None


def analyze_pa_trace(
    reasoning: Any,
    gt: Dict[str, Any],
) -> Dict[str, Any]:
    pas = extract_pa_steps(reasoning)
    gt_norm = normalize_table_strict(gt)

    if gt_norm is None:
        raise ValueError("Invalid GT while evaluating PA.")

    gt_by_house, gt_attrs = gt_norm

    pa_details: List[Dict[str, Any]] = []
    monotonic_violations: List[Dict[str, Any]] = []

    previous_grid = None
    first_error = None

    total_resolved = 0
    total_wrong = 0

    for idx, item in enumerate(pas):
        parsed = pa_grid_by_house(item["pa"])

        detail = {
            "k": item["k"],
            "key": item["key"],
            "order": item["order"],
            "parse_ok": parsed is not None,
            "resolved_cells": 0,
            "wrong_cells": 0,
            "wrong_cell_details": [],
            "monotonicity_violations": [],
        }

        if parsed is None:
            pa_details.append(detail)
            continue

        grid, attrs = parsed

        for house, vals in grid.items():
            gt_vals = gt_by_house.get(house, {})

            for attr, value in vals.items():
                if value == "?":
                    continue

                detail["resolved_cells"] += 1
                total_resolved += 1

                gt_value = gt_vals.get(attr)

                if gt_value != value:
                    detail["wrong_cells"] += 1
                    total_wrong += 1

                    cell_err = {
                        "house": house,
                        "attribute": attr,
                        "pa_value": value,
                        "gt_value": gt_value,
                    }

                    detail["wrong_cell_details"].append(cell_err)

                    if first_error is None:
                        first_error = {
                            "pa_key": item["key"],
                            "pa_k": item["k"],
                            "order": item["order"],
                            "house": house,
                            "attribute": attr,
                            "pa_value": value,
                            "gt_value": gt_value,
                        }

        # Monotonicity relative to previous parsed PA
        if previous_grid is not None:
            prev_grid, prev_attrs = previous_grid

            common_houses = set(prev_grid) & set(grid)
            common_attrs = set(prev_attrs) & set(attrs)

            for house in sorted(common_houses):
                for attr in sorted(common_attrs):
                    prev_value = prev_grid[house].get(attr, "?")
                    curr_value = grid[house].get(attr, "?")

                    if prev_value == "?":
                        continue

                    if curr_value == "?":
                        violation = {
                            "type": "forgetting",
                            "from_pa": pas[idx - 1]["key"],
                            "to_pa": item["key"],
                            "house": house,
                            "attribute": attr,
                            "previous": prev_value,
                            "current": curr_value,
                        }
                        detail["monotonicity_violations"].append(violation)
                        monotonic_violations.append(violation)

                    elif curr_value != prev_value:
                        violation = {
                            "type": "revision",
                            "from_pa": pas[idx - 1]["key"],
                            "to_pa": item["key"],
                            "house": house,
                            "attribute": attr,
                            "previous": prev_value,
                            "current": curr_value,
                        }
                        detail["monotonicity_violations"].append(violation)
                        monotonic_violations.append(violation)

        previous_grid = parsed
        pa_details.append(detail)

    return {
        "n_pa": len(pas),
        "n_pa_resolved_cells": total_resolved,
        "n_pa_wrong_cells": total_wrong,
        "first_pa_error": first_error,
        "n_monotonicity_violations": len(monotonic_violations),
        "n_forgetting": sum(
            x["type"] == "forgetting"
            for x in monotonic_violations
        ),
        "n_revision": sum(
            x["type"] == "revision"
            for x in monotonic_violations
        ),
        "monotonicity_violations": monotonic_violations,
        "pa_details": pa_details,
    }


# =============================================================================
# S-error vs PA-error ordering
# =============================================================================

def classify_s_pa_order(
    s_trace: Dict[str, Any],
    pa_trace: Dict[str, Any],
) -> str:
    first_s = s_trace.get("first_false_s")
    first_pa = pa_trace.get("first_pa_error")

    if first_s is None and first_pa is None:
        return "NO_S_OR_PA_ERROR"

    if first_s is not None and first_pa is None:
        return "S_ERROR_ONLY"

    if first_s is None and first_pa is not None:
        return "PA_ERROR_ONLY"

    s_order = first_s.get("order")
    pa_order = first_pa.get("order")

    if s_order is None or pa_order is None:
        return "ORDER_UNKNOWN"

    if s_order < pa_order:
        return "S_BEFORE_PA"

    if pa_order < s_order:
        return "PA_BEFORE_S"

    return "SAME_REGION"


# =============================================================================
# Per-system / paired-case analysis
# =============================================================================

def analyze_system_case(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    gt = record.get("ground_truth", {})
    comp = extract_payload_components(record)
    solution = comp.get("solution")
    reasoning = comp.get("reasoning")

    accuracy = compute_case_accuracy(gt, solution)
    env, ambiguous_gt = gt_entity_to_house(gt)

    s_trace = analyze_s_trace(
        reasoning,
        env,
    )

    pa_trace = analyze_pa_trace(
        reasoning,
        gt,
    )

    reasoning_to_answer_drift = (
        accuracy["puzzle_accuracy"] == 0.0
        and s_trace["n_gt_wrong_s"] == 0
    )

    return {
        "id": str(record.get("id", "UNKNOWN")),
        "size": normalize_size(record.get("size", "")),
        "difficulty": DIFFICULTY_BY_SIZE.get(
            normalize_size(record.get("size", "")),
            "Unknown",
        ),
        "full_payload_ok": comp.get("full_payload_ok"),
        "extraction_method": comp.get("extraction_method"),
        "accuracy": accuracy,
        "s_trace": s_trace,
        "pa_trace": pa_trace,
        "reasoning_to_answer_drift": reasoning_to_answer_drift,
        "ambiguous_gt_tokens": sorted(ambiguous_gt),
    }


def normalized_first_error_comparison(
    original: Dict[str, Any],
    nss: Dict[str, Any],
) -> Dict[str, Any]:
    o = original["s_trace"]["first_false_normalized_position"]
    n = nss["s_trace"]["first_false_normalized_position"]

    if o is None and n is None:
        return {
            "category": "NO_GT_WRONG_S_IN_EITHER",
            "delta_nss_minus_original": None,
        }

    if o is not None and n is None:
        return {
            "category": "ORIGINAL_ERROR_ONLY",
            "delta_nss_minus_original": None,
        }

    if o is None and n is not None:
        return {
            "category": "NSS_ERROR_ONLY",
            "delta_nss_minus_original": None,
        }

    delta = float(n) - float(o)

    eps = 1e-12

    if delta > eps:
        category = "NSS_ERROR_LATER"
    elif delta < -eps:
        category = "NSS_ERROR_EARLIER"
    else:
        category = "SAME_NORMALIZED_POSITION"

    return {
        "category": category,
        "delta_nss_minus_original": delta,
    }


def paired_outcome(
    original: Dict[str, Any],
    nss: Dict[str, Any],
) -> str:
    o = original["accuracy"]["puzzle_accuracy"] == 1.0
    n = nss["accuracy"]["puzzle_accuracy"] == 1.0

    if o and n:
        return "BOTH_CORRECT"

    if (not o) and (not n):
        return "BOTH_WRONG"

    if (not o) and n:
        return "NSS_FIX"

    return "NSS_REGRESSION"


def analyze_pair(
    original_record: Dict[str, Any],
    nss_record: Dict[str, Any],
) -> Dict[str, Any]:
    original = analyze_system_case(original_record)
    nss = analyze_system_case(nss_record)

    if original["id"] != nss["id"]:
        raise ValueError(
            f"ID mismatch: {original['id']} != {nss['id']}"
        )

    if original["size"] != nss["size"]:
        raise ValueError(
            f"Size mismatch for {original['id']}: "
            f"{original['size']} != {nss['size']}"
        )

    outcome = paired_outcome(original, nss)

    return {
        "id": original["id"],
        "size": original["size"],
        "difficulty": original["difficulty"],
        "outcome": outcome,

        "first_error_comparison": normalized_first_error_comparison(
            original,
            nss,
        ),

        "nss_s_pa_order": classify_s_pa_order(
            nss["s_trace"],
            nss["pa_trace"],
        ),

        "original": original,
        "nss": nss,
    }


# =============================================================================
# Log loading / alignment
# =============================================================================

def read_records(path: Path) -> List[Dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(parse_outer_record(line))
            except Exception as exc:
                raise ValueError(
                    f"Could not parse {path}:{line_no}: {exc}"
                ) from exc

    return records


def align_by_id(
    original_records: Sequence[Dict[str, Any]],
    nss_records: Sequence[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    omap = {
        str(r.get("id")): r
        for r in original_records
    }

    nmap = {
        str(r.get("id")): r
        for r in nss_records
    }

    if len(omap) != len(original_records):
        raise ValueError("Duplicate IDs in Original log.")

    if len(nmap) != len(nss_records):
        raise ValueError("Duplicate IDs in NSS log.")

    if set(omap) != set(nmap):
        only_o = sorted(set(omap) - set(nmap))[:20]
        only_n = sorted(set(nmap) - set(omap))[:20]

        raise ValueError(
            "ID sets differ.\n"
            f"Only Original: {only_o}\n"
            f"Only NSS: {only_n}"
        )

    return [
        (r, nmap[str(r.get("id"))])
        for r in original_records
    ]


# =============================================================================
# Aggregation helpers
# =============================================================================

def subset(
    pairs: Sequence[Dict[str, Any]],
    *,
    outcome: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows = list(pairs)

    if outcome is not None:
        rows = [
            r for r in rows
            if r["outcome"] == outcome
        ]

    if difficulty is not None:
        rows = [
            r for r in rows
            if r["difficulty"] == difficulty
        ]

    return rows


def safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [
        float(v)
        for v in values
        if v is not None
        and isinstance(v, (int, float))
        and math.isfinite(float(v))
    ]

    if not vals:
        return None

    return sum(vals) / len(vals)


def fmt_num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def summarize_subset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    first_cmp = Counter(
        r["first_error_comparison"]["category"]
        for r in rows
    )

    original_first_types = Counter(
        r["original"]["s_trace"]["first_false_type"]
        for r in rows
        if r["original"]["s_trace"]["first_false_type"] is not None
    )

    nss_first_types = Counter(
        r["nss"]["s_trace"]["first_false_type"]
        for r in rows
        if r["nss"]["s_trace"]["first_false_type"] is not None
    )

    s_pa_order = Counter(
        r["nss_s_pa_order"]
        for r in rows
    )

    return {
        "n": len(rows),

        "first_error_comparison": first_cmp,

        "mean_delta_first_error_position": safe_mean([
            r["first_error_comparison"][
                "delta_nss_minus_original"
            ]
            for r in rows
        ]),

        "original_first_error_types": original_first_types,
        "nss_first_error_types": nss_first_types,

        "original_reasoning_to_answer_drift": sum(
            r["original"]["reasoning_to_answer_drift"]
            for r in rows
        ),

        "nss_reasoning_to_answer_drift": sum(
            r["nss"]["reasoning_to_answer_drift"]
            for r in rows
        ),

        "original_recovery_any": sum(
            r["original"]["s_trace"]["recovery_any"]
            for r in rows
        ),

        "nss_recovery_any": sum(
            r["nss"]["s_trace"]["recovery_any"]
            for r in rows
        ),

        "mean_original_gt_wrong_s": safe_mean([
            r["original"]["s_trace"]["n_gt_wrong_s"]
            for r in rows
        ]),

        "mean_nss_gt_wrong_s": safe_mean([
            r["nss"]["s_trace"]["n_gt_wrong_s"]
            for r in rows
        ]),

        "mean_original_later_wrong_s_after_first": safe_mean([
            r["original"]["s_trace"]["n_later_gt_wrong_s"]
            for r in rows
            if r["original"]["s_trace"]["first_false_s"] is not None
        ]),

        "mean_nss_later_wrong_s_after_first": safe_mean([
            r["nss"]["s_trace"]["n_later_gt_wrong_s"]
            for r in rows
            if r["nss"]["s_trace"]["first_false_s"] is not None
        ]),

        "mean_original_wrong_final_cells": safe_mean([
            r["original"]["accuracy"]["wrong_cells"]
            for r in rows
        ]),

        "mean_nss_wrong_final_cells": safe_mean([
            r["nss"]["accuracy"]["wrong_cells"]
            for r in rows
        ]),

        "mean_original_s_steps": safe_mean([
            r["original"]["s_trace"]["n_s_steps"]
            for r in rows
        ]),

        "mean_nss_s_steps": safe_mean([
            r["nss"]["s_trace"]["n_s_steps"]
            for r in rows
        ]),

        "mean_nss_pa_count": safe_mean([
            r["nss"]["pa_trace"]["n_pa"]
            for r in rows
        ]),

        "nss_pa_error_cases": sum(
            r["nss"]["pa_trace"]["first_pa_error"] is not None
            for r in rows
        ),

        "nss_pa_monotonicity_violation_cases": sum(
            r["nss"]["pa_trace"]["n_monotonicity_violations"] > 0
            for r in rows
        ),

        "nss_s_pa_order": s_pa_order,
    }



def extract_original_puzzle_text(record: Dict[str, Any]) -> str:
    """
    Best-effort extraction of the original natural-language puzzle text
    from a log record.

    Checks direct puzzle fields first, then nested metadata, then
    prompt/messages structures.
    """
    if not isinstance(record, dict):
        return "[Puzzle text unavailable: record is not a dictionary]"

    direct_keys = (
        "puzzle_text",
        "puzzle",
        "question",
        "problem",
        "input",
    )

    for key in direct_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for parent_key in ("extra_info", "metadata", "data", "example"):
        parent = record.get(parent_key)
        if not isinstance(parent, dict):
            continue
        for key in direct_keys:
            value = parent.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    def extract_from_messages(messages: Any) -> Optional[str]:
        if not isinstance(messages, list):
            return None
        user_contents: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip().lower()
            content = msg.get("content")
            if role == "user" and isinstance(content, str) and content.strip():
                user_contents.append(content.strip())
        return user_contents[-1] if user_contents else None

    candidate = extract_from_messages(record.get("messages"))
    if candidate:
        return candidate

    prompt = record.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    candidate = extract_from_messages(prompt)
    if candidate:
        return candidate

    if isinstance(prompt, dict):
        candidate = extract_from_messages(prompt.get("messages"))
        if candidate:
            return candidate
        for key in direct_keys:
            value = prompt.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raw_prompt = record.get("raw_prompt")
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        return raw_prompt.strip()

    candidate = extract_from_messages(raw_prompt)
    if candidate:
        return candidate

    return "[Puzzle text unavailable in this log record]"




# =============================================================================
# Optional source-dataset lookup for original natural-language puzzle text
# =============================================================================

def load_source_puzzle_map(
    source_path: Optional[Path],
    *,
    id_field: str = "id",
    puzzle_field: str = "puzzle",
) -> Dict[str, str]:
    """
    Load original puzzle text keyed by puzzle ID from an optional source dataset.

    Supported file types:
      - .jsonl
      - .json
      - .parquet

    The ZebraLogic source dataset used in this project commonly contains:
        id, size, puzzle, solution, ...

    If source_path is None, returns an empty dictionary.
    """
    if source_path is None:
        return {}

    source_path = Path(source_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source puzzle dataset does not exist: {source_path}"
        )

    rows: List[Dict[str, Any]] = []
    suffix = source_path.suffix.lower()

    if suffix == ".jsonl":
        with source_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise ValueError(
                        f"Could not parse source JSONL "
                        f"{source_path}:{line_no}: {exc}"
                    ) from exc

                if isinstance(row, dict):
                    rows.append(row)

    elif suffix == ".json":
        with source_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            # Standard list-of-records JSON:
            # [
            #   {"id": "...", "puzzle": "..."},
            #   ...
            # ]
            rows = [
                x for x in obj
                if isinstance(x, dict)
            ]

        elif isinstance(obj, dict):
            if isinstance(obj.get("data"), list):
                # Wrapped list-of-records JSON:
                # {"data": [{"id": "...", "puzzle": "..."}, ...]}
                rows = [
                    x for x in obj["data"]
                    if isinstance(x, dict)
                ]

            else:
                # IMPORTANT: support the exact structure used by
                # ./Input_Logs/pid_to_puzzle_dic.json:
                #
                # {
                #   "lgp-test-2x6-15": "There are 2 houses ...",
                #   "lgp-test-2x6-15_sol": {...},
                #   ...
                # }
                #
                # The puzzle text is stored DIRECTLY as a string value.
                # Solution entries end in "_sol" and are dictionaries.
                for key, value in obj.items():

                    # Direct pid -> puzzle text mapping.
                    if isinstance(value, str) and value.strip():
                        rows.append({
                            id_field: str(key).strip(),
                            puzzle_field: value.strip(),
                        })
                        continue

                    # Also retain compatibility with ID -> record mappings.
                    if isinstance(value, dict):
                        row = dict(value)
                        row.setdefault(id_field, key)
                        rows.append(row)

        else:
            raise ValueError(
                f"Unsupported JSON structure in {source_path}"
            )

    elif suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Reading a Parquet source dataset requires pandas "
                "and a parquet engine such as pyarrow."
            ) from exc

        df = pd.read_parquet(source_path)

        if id_field not in df.columns:
            raise ValueError(
                f"Parquet source is missing ID field {id_field!r}. "
                f"Columns={list(df.columns)}"
            )

        if puzzle_field not in df.columns:
            raise ValueError(
                f"Parquet source is missing puzzle field {puzzle_field!r}. "
                f"Columns={list(df.columns)}"
            )

        for _, row in df.iterrows():
            rows.append({
                id_field: row[id_field],
                puzzle_field: row[puzzle_field],
            })

    else:
        raise ValueError(
            "Unsupported source dataset type. "
            "Use .jsonl, .json, or .parquet."
        )

    puzzle_map: Dict[str, str] = {}

    for row in rows:
        pid = row.get(id_field)
        puzzle = row.get(puzzle_field)

        if pid is None:
            continue

        if isinstance(puzzle, str) and puzzle.strip():
            puzzle_map[str(pid)] = puzzle.strip()
            continue

        # Fallback for datasets that use puzzle_text/question/etc.
        fallback = extract_original_puzzle_text(row)

        if (
            isinstance(fallback, str)
            and fallback.strip()
            and not fallback.startswith("[Puzzle text unavailable")
        ):
            puzzle_map[str(pid)] = fallback.strip()

    return puzzle_map


def get_puzzle_text_for_case(
    pid: str,
    original_raw: Dict[str, Any],
    nss_raw: Dict[str, Any],
    source_puzzle_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, str]:
    """
    Return:
        (puzzle_text, source_label)

    Priority:
      1. external source dataset lookup by ID
      2. Original output-log record
      3. NSS output-log record
      4. clear unavailable message
    """
    source_puzzle_map = source_puzzle_map or {}

    if pid in source_puzzle_map:
        return source_puzzle_map[pid], "SOURCE_DATASET"

    for label, record in (
        ("ORIGINAL_LOG", original_raw),
        ("NSS_LOG", nss_raw),
    ):
        candidate = extract_original_puzzle_text(record)

        if (
            isinstance(candidate, str)
            and candidate.strip()
            and not candidate.startswith("[Puzzle text unavailable")
        ):
            return candidate.strip(), label

    return (
        "[Original natural-language puzzle text is not stored in either "
        "generation log. Supply --source-dataset to recover it by puzzle ID.]",
        "UNAVAILABLE",
    )



# =============================================================================
# Reporting
# =============================================================================

def print_counter(
    p,
    title: str,
    counter: Counter,
    *,
    indent: str = "    ",
) -> None:
    p(title)

    if not counter:
        p(indent + "(none)")
        return

    for key, value in counter.most_common():
        p(f"{indent}{str(key):<36} {value:>6}")


def write_summary(
    path: Path,
    pairs: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as out:

        def p(text: str = ""):
            print(text)
            out.write(str(text) + "\n")

        outcomes = Counter(
            r["outcome"]
            for r in pairs
        )

        p("=" * 130)
        p("PAIRED ERROR-DIFFERENCE ANALYSIS")
        p("=" * 130)
        p()
        p("Definitions:")
        p("  NSS_FIX        = Original final puzzle wrong, NSS final puzzle correct.")
        p("  NSS_REGRESSION = Original final puzzle correct, NSS final puzzle wrong.")
        p("  GT-wrong S_i   = S_i evaluates FALSE under the complete ground-truth assignment.")
        p("  Recovery       = after the first GT-wrong S_i, at least one later S-step is GT-consistent.")
        p("  Reasoning-to-answer drift = final answer wrong but no GT-wrong S_i is observed.")
        p("                           This does NOT prove formal validity; it means no displayed S_i contradicts GT.")
        p()

        p("=" * 130)
        p("A. OVERALL PAIRED OUTCOMES")
        p("=" * 130)
        p(f"Total aligned puzzles : {len(pairs)}")
        p(f"Both correct          : {outcomes['BOTH_CORRECT']}")
        p(f"Both wrong            : {outcomes['BOTH_WRONG']}")
        p(f"NSS fixes             : {outcomes['NSS_FIX']}")
        p(f"NSS regressions       : {outcomes['NSS_REGRESSION']}")
        p(f"Net NSS wins          : {outcomes['NSS_FIX'] - outcomes['NSS_REGRESSION']}")
        p()

        # Difficulty-level paired outcomes
        p("Paired outcomes by difficulty:")
        p(
            f"{'Difficulty':<12}"
            f"{'N':>6}"
            f"{'Both✓':>9}"
            f"{'Both✗':>9}"
            f"{'NSS Fix':>10}"
            f"{'NSS Reg':>10}"
            f"{'Net NSS':>10}"
        )
        p("-" * 66)

        for difficulty in ["Small", "Medium", "Large", "XL"]:
            rows = subset(
                pairs,
                difficulty=difficulty,
            )
            c = Counter(r["outcome"] for r in rows)

            p(
                f"{difficulty:<12}"
                f"{len(rows):>6}"
                f"{c['BOTH_CORRECT']:>9}"
                f"{c['BOTH_WRONG']:>9}"
                f"{c['NSS_FIX']:>10}"
                f"{c['NSS_REGRESSION']:>10}"
                f"{c['NSS_FIX'] - c['NSS_REGRESSION']:>10}"
            )

        p()

        # Main paired error groups
        for outcome, title in (
            (
                "NSS_FIX",
                "B. NSS FIXES: ORIGINAL WRONG -> NSS CORRECT",
            ),
            (
                "NSS_REGRESSION",
                "C. NSS REGRESSIONS: ORIGINAL CORRECT -> NSS WRONG",
            ),
        ):
            p("=" * 130)
            p(title)
            p("=" * 130)

            for difficulty_label in [
                "Overall",
                "Small",
                "Medium",
                "Large",
                "XL",
            ]:
                difficulty = (
                    None
                    if difficulty_label == "Overall"
                    else difficulty_label
                )

                rows = subset(
                    pairs,
                    outcome=outcome,
                    difficulty=difficulty,
                )

                s = summarize_subset(rows)

                p()
                p(f"[{difficulty_label}]  N = {s['n']}")
                p("-" * 90)

                if s["n"] == 0:
                    continue

                print_counter(
                    p,
                    "First GT-wrong S timing comparison:",
                    s["first_error_comparison"],
                )

                p(
                    "Mean normalized first-error delta "
                    "(NSS - Original, paired cases where both have an error): "
                    f"{fmt_num(s['mean_delta_first_error_position'], 4)}"
                )

                p()
                print_counter(
                    p,
                    "Original first GT-wrong deduction type:",
                    s["original_first_error_types"],
                )

                p()
                print_counter(
                    p,
                    "NSS first GT-wrong deduction type:",
                    s["nss_first_error_types"],
                )

                p()
                p("Reasoning / cascade metrics:")
                p(
                    f"    Original traces with recovery after first error : "
                    f"{s['original_recovery_any']}"
                )
                p(
                    f"    NSS traces with recovery after first error      : "
                    f"{s['nss_recovery_any']}"
                )
                p(
                    f"    Original reasoning->answer drift cases          : "
                    f"{s['original_reasoning_to_answer_drift']}"
                )
                p(
                    f"    NSS reasoning->answer drift cases               : "
                    f"{s['nss_reasoning_to_answer_drift']}"
                )
                p(
                    f"    Mean Original GT-wrong S count                  : "
                    f"{fmt_num(s['mean_original_gt_wrong_s'])}"
                )
                p(
                    f"    Mean NSS GT-wrong S count                       : "
                    f"{fmt_num(s['mean_nss_gt_wrong_s'])}"
                )
                p(
                    f"    Mean later wrong S after first error, Original  : "
                    f"{fmt_num(s['mean_original_later_wrong_s_after_first'])}"
                )
                p(
                    f"    Mean later wrong S after first error, NSS       : "
                    f"{fmt_num(s['mean_nss_later_wrong_s_after_first'])}"
                )
                p(
                    f"    Mean Original wrong final cells                 : "
                    f"{fmt_num(s['mean_original_wrong_final_cells'])}"
                )
                p(
                    f"    Mean NSS wrong final cells                      : "
                    f"{fmt_num(s['mean_nss_wrong_final_cells'])}"
                )

                p()
                p("Reasoning length:")
                p(
                    f"    Mean Original S steps : "
                    f"{fmt_num(s['mean_original_s_steps'])}"
                )
                p(
                    f"    Mean NSS S steps      : "
                    f"{fmt_num(s['mean_nss_s_steps'])}"
                )
                p(
                    f"    Mean NSS PA count     : "
                    f"{fmt_num(s['mean_nss_pa_count'])}"
                )

                p()
                p("NSS PA diagnostics:")
                p(
                    f"    Cases with >=1 GT-wrong PA cell          : "
                    f"{s['nss_pa_error_cases']}"
                )
                p(
                    f"    Cases with PA monotonicity violation      : "
                    f"{s['nss_pa_monotonicity_violation_cases']}"
                )

                print_counter(
                    p,
                    "    First S-error vs first PA-error ordering:",
                    s["nss_s_pa_order"],
                    indent="        ",
                )

            p()

        p("=" * 130)
        p("D. INTERPRETATION GUIDE")
        p("=" * 130)
        p()
        p("Evidence that NSS helps the reasoning trajectory would look like:")
        p("  - In NSS_FIX cases, Original commonly has the first GT-wrong S while NSS has no GT-wrong S.")
        p("  - When both systems err, NSS first error occurs later (positive normalized delta).")
        p("  - NSS has smaller downstream error cascades and/or more recovery.")
        p("  - NSS fixes show clean PA trajectories.")
        p()
        p("Evidence that PA itself may introduce regressions would look like:")
        p("  - In NSS_REGRESSION cases, PA_BEFORE_S is common.")
        p("  - A GT-wrong PA cell appears before the first GT-wrong NSS S-step.")
        p("  - PA monotonicity violations are concentrated in regressions.")
        p()
        p("Evidence that PA mostly records pre-existing S errors would look like:")
        p("  - In NSS_REGRESSION cases, S_BEFORE_PA dominates.")
        p("  - The first GT-wrong S occurs before the first GT-wrong PA cell.")
        p()
        p("Reasoning-to-answer drift is separately informative:")
        p("  - Original wrong + no GT-wrong S, while NSS correct, can indicate that NSS/PA improved state-to-answer transfer.")
        p("  - NSS wrong + no GT-wrong S, while Original correct, can indicate NSS introduced final-answer/state-construction problems.")
        p()


def write_case_jsonl(
    path: Path,
    pairs: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def format_step(step: Optional[Dict[str, Any]]) -> str:
    if not step:
        return "None"

    return (
        f"{step.get('key')} "
        f"(pos={step.get('position')}, "
        f"norm={fmt_num(step.get('normalized_position'), 4)}, "
        f"type={step.get('deduction_type')}): "
        f"{step.get('expr')}"
    )


def write_detailed_cases(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    original_raw_by_id: Dict[str, Dict[str, Any]],
    nss_raw_by_id: Dict[str, Dict[str, Any]],
    source_puzzle_map: Optional[Dict[str, str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, 1):
            pid = row["id"]

            out.write("=" * 140 + "\n")
            out.write(
                f"CASE {idx} | ID={pid} | "
                f"SIZE={row['size']} | "
                f"DIFFICULTY={row['difficulty']} | "
                f"OUTCOME={row['outcome']}\n"
            )
            out.write("=" * 140 + "\n\n")

            o = row["original"]
            n = row["nss"]

            out.write("PAIR SUMMARY\n")
            out.write("-" * 80 + "\n")
            out.write(
                f"First-error comparison: "
                f"{row['first_error_comparison']}\n"
            )
            out.write(
                f"NSS first S/PA error ordering: "
                f"{row['nss_s_pa_order']}\n"
            )
            out.write("\n")

            puzzle_text, puzzle_source = get_puzzle_text_for_case(
                pid,
                original_raw_by_id[pid],
                nss_raw_by_id[pid],
                source_puzzle_map,
            )

            out.write("ORIGINAL PUZZLE TEXT\n")
            out.write("-" * 80 + "\n")
            out.write(f"[source={puzzle_source}]\n")
            out.write(puzzle_text + "\n\n")

            out.write("ORIGINAL\n")
            out.write("-" * 80 + "\n")
            out.write(
                f"Puzzle accuracy: {o['accuracy']['puzzle_accuracy']} | "
                f"Cell accuracy: {o['accuracy']['cell_accuracy']:.4f} | "
                f"Wrong final cells: {o['accuracy']['wrong_cells']}\n"
            )
            out.write(
                f"S steps={o['s_trace']['n_s_steps']} | "
                f"GT-wrong S={o['s_trace']['n_gt_wrong_s']} | "
                f"Recovery={o['s_trace']['recovery_any']} | "
                f"Reasoning->answer drift={o['reasoning_to_answer_drift']}\n"
            )
            out.write(
                "First GT-wrong S: "
                + format_step(o["s_trace"]["first_false_s"])
                + "\n\n"
            )

            out.write("NSS\n")
            out.write("-" * 80 + "\n")
            out.write(
                f"Puzzle accuracy: {n['accuracy']['puzzle_accuracy']} | "
                f"Cell accuracy: {n['accuracy']['cell_accuracy']:.4f} | "
                f"Wrong final cells: {n['accuracy']['wrong_cells']}\n"
            )
            out.write(
                f"S steps={n['s_trace']['n_s_steps']} | "
                f"GT-wrong S={n['s_trace']['n_gt_wrong_s']} | "
                f"Recovery={n['s_trace']['recovery_any']} | "
                f"Reasoning->answer drift={n['reasoning_to_answer_drift']}\n"
            )
            out.write(
                "First GT-wrong S: "
                + format_step(n["s_trace"]["first_false_s"])
                + "\n"
            )
            out.write(
                f"PA count={n['pa_trace']['n_pa']} | "
                f"GT-wrong PA cells={n['pa_trace']['n_pa_wrong_cells']} | "
                f"Monotonicity violations={n['pa_trace']['n_monotonicity_violations']}\n"
            )
            out.write(
                f"First PA error: "
                f"{n['pa_trace']['first_pa_error']}\n"
            )

            if n["pa_trace"]["monotonicity_violations"]:
                out.write("PA monotonicity violations:\n")
                for v in n["pa_trace"]["monotonicity_violations"]:
                    out.write(f"  {v}\n")

            out.write("\nGROUND TRUTH\n")
            out.write("-" * 80 + "\n")
            out.write(
                json.dumps(
                    original_raw_by_id[pid].get("ground_truth", {}),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n\n"
            )

            out.write("ORIGINAL FULL LLM OUTPUT\n")
            out.write("-" * 80 + "\n")
            out.write(
                str(
                    original_raw_by_id[pid].get(
                        "llm_output",
                        "",
                    )
                )
                + "\n\n"
            )

            out.write("NSS FULL LLM OUTPUT\n")
            out.write("-" * 80 + "\n")
            out.write(
                str(
                    nss_raw_by_id[pid].get(
                        "llm_output",
                        "",
                    )
                )
                + "\n\n\n"
            )


# =============================================================================
# Human-readable paired trace export by outcome and exact puzzle size
# =============================================================================


def safe_filename_component(value: Any) -> str:
    """Convert an arbitrary puzzle ID / label into a filesystem-safe component."""
    s = str(value).strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s or "UNKNOWN"


def format_solution_table(table: Any) -> str:
    """Return a readable plain-text representation of a Zebra table."""
    if not isinstance(table, dict):
        return json.dumps(table, indent=2, ensure_ascii=False)

    header = table.get("header")
    rows = table.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list):
        return json.dumps(table, indent=2, ensure_ascii=False)

    string_rows = [[str(x) for x in header]]
    for row in rows:
        if isinstance(row, list):
            string_rows.append([str(x) for x in row])

    if not string_rows:
        return json.dumps(table, indent=2, ensure_ascii=False)

    n_cols = max(len(r) for r in string_rows)
    widths = [0] * n_cols
    for row in string_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    lines = []
    for ridx, row in enumerate(string_rows):
        padded = [
            (row[i] if i < len(row) else "").ljust(widths[i])
            for i in range(n_cols)
        ]
        lines.append(" | ".join(padded))
        if ridx == 0:
            lines.append("-+-".join("-" * w for w in widths))

    return "\n".join(lines)


def format_reasoning_trace_readable(
    reasoning: Any,
    *,
    gt_env: Dict[str, int],
) -> str:
    """Render Original/NSS reasoning with GT annotations on every S_i."""
    lines: List[str] = []

    def s_annotation(expr: Any) -> str:
        gt_value, gt_error = eval_expr_gt(expr, gt_env)
        if gt_value is True:
            status = "GT-CORRECT"
        elif gt_value is False:
            status = "GT-WRONG"
        else:
            status = "UNVERIFIABLE"

        dtype = deduction_type(expr)
        error_text = f" | {gt_error}" if gt_error else ""
        return f"[{status} | type={dtype}{error_text}]"

    if isinstance(reasoning, dict):
        for key, value in reasoning.items():
            skey = str(key)
            if re.fullmatch(r"NL\d+", skey, flags=re.I):
                lines.append(f"{skey}: {value}")
            elif re.fullmatch(r"S\d+", skey, flags=re.I):
                lines.append(f"{skey}: {value}  {s_annotation(value)}")
            elif re.fullmatch(r"PA\d+", skey, flags=re.I):
                lines.append("")
                lines.append(f"{skey}:")
                lines.append(format_solution_table(value))
                lines.append("")
            else:
                lines.append(f"{skey}: {value}")
        return "\n".join(lines)

    if isinstance(reasoning, list):
        for item in reasoning:
            if not isinstance(item, str):
                lines.append(str(item))
                continue

            m = re.match(
                r"^\s*S(\d+)\s*:\s*(.+?)\s*$",
                item,
                flags=re.I | re.S,
            )
            if m:
                key = f"S{int(m.group(1))}"
                expr = m.group(2).strip()
                lines.append(f"{key}: {expr}  {s_annotation(expr)}")
            else:
                lines.append(item)
        return "\n".join(lines)

    return json.dumps(reasoning, indent=2, ensure_ascii=False)


def extract_display_components(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract components used for readable trace output."""
    comp = extract_payload_components(record)
    return {
        "reasoning": comp.get("reasoning"),
        "solution": comp.get("solution"),
        "syntactic_clues": comp.get("syntactic_clues"),
        "attribute_values": comp.get("attribute_values"),
        "n_houses": comp.get("n_houses"),
        "full_payload_ok": comp.get("full_payload_ok"),
        "extraction_method": comp.get("extraction_method"),
    }


def write_one_readable_paired_trace(
    path: Path,
    row: Dict[str, Any],
    original_raw: Dict[str, Any],
    nss_raw: Dict[str, Any],
    source_puzzle_map: Optional[Dict[str, str]] = None,
) -> None:
    """Write one paired Original-vs-NSS case as a readable TXT trace."""
    path.parent.mkdir(parents=True, exist_ok=True)

    pid = row["id"]
    gt = original_raw.get("ground_truth", {})
    try:
        gt_env, _ = gt_entity_to_house(gt)
    except Exception:
        gt_env = {}

    original_comp = extract_display_components(original_raw)
    nss_comp = extract_display_components(nss_raw)
    o = row["original"]
    n = row["nss"]

    with path.open("w", encoding="utf-8") as out:
        def p(text: str = ""):
            out.write(str(text) + "\n")

        p("=" * 140)
        p(
            f"PAIRED TRACE | ID={pid} | SIZE={row['size']} | "
            f"DIFFICULTY={row['difficulty']} | OUTCOME={row['outcome']}"
        )
        p("=" * 140)
        p()

        puzzle_text, puzzle_source = get_puzzle_text_for_case(
            pid,
            original_raw,
            nss_raw,
            source_puzzle_map,
        )

        p("ORIGINAL PUZZLE TEXT")
        p("-" * 100)
        p(f"[source={puzzle_source}]")
        p(puzzle_text)
        p()

        p("PAIR-LEVEL DIAGNOSTICS")
        p("-" * 100)
        p(f"First-error comparison : {row['first_error_comparison']}")
        p(f"NSS S-vs-PA ordering   : {row['nss_s_pa_order']}")
        p()

        p("FINAL OUTCOME")
        p("-" * 100)
        p(
            f"Original: PAcc={o['accuracy']['puzzle_accuracy']:.1f}, "
            f"CAcc={o['accuracy']['cell_accuracy']:.4f}, "
            f"wrong_cells={o['accuracy']['wrong_cells']}"
        )
        p(
            f"NSS     : PAcc={n['accuracy']['puzzle_accuracy']:.1f}, "
            f"CAcc={n['accuracy']['cell_accuracy']:.4f}, "
            f"wrong_cells={n['accuracy']['wrong_cells']}"
        )
        p()

        p("GROUND TRUTH")
        p("-" * 100)
        p(format_solution_table(gt))
        p()

        for system_name, comp in (("ORIGINAL", original_comp), ("NSS", nss_comp)):
            p(f"SYNTACTIC CLUES — {system_name}")
            p("-" * 100)
            clues = comp.get("syntactic_clues")
            if isinstance(clues, list):
                for clue in clues:
                    p(str(clue))
            else:
                p(json.dumps(clues, indent=2, ensure_ascii=False))
            p()

        p("=" * 140)
        p("ORIGINAL REASONING TRACE")
        p("=" * 140)
        p(
            f"S-steps={o['s_trace']['n_s_steps']} | "
            f"GT-wrong S={o['s_trace']['n_gt_wrong_s']} | "
            f"GT-correct S={o['s_trace']['n_gt_correct_s']} | "
            f"unverifiable S={o['s_trace']['n_unverifiable_s']}"
        )
        p("First GT-wrong S: " + format_step(o["s_trace"]["first_false_s"]))
        p(
            f"Recovery after first error: {o['s_trace']['recovery_any']} | "
            f"reasoning->answer drift: {o['reasoning_to_answer_drift']}"
        )
        p()
        p(format_reasoning_trace_readable(original_comp.get("reasoning"), gt_env=gt_env))
        p()

        p("ORIGINAL FINAL SOLUTION")
        p("-" * 100)
        p(format_solution_table(original_comp.get("solution")))
        p()

        p("=" * 140)
        p("NSS REASONING TRACE")
        p("=" * 140)
        p(
            f"S-steps={n['s_trace']['n_s_steps']} | "
            f"GT-wrong S={n['s_trace']['n_gt_wrong_s']} | "
            f"GT-correct S={n['s_trace']['n_gt_correct_s']} | "
            f"unverifiable S={n['s_trace']['n_unverifiable_s']}"
        )
        p("First GT-wrong S: " + format_step(n["s_trace"]["first_false_s"]))
        p(
            f"Recovery after first error: {n['s_trace']['recovery_any']} | "
            f"reasoning->answer drift: {n['reasoning_to_answer_drift']}"
        )
        p(
            f"PA count={n['pa_trace']['n_pa']} | "
            f"GT-wrong PA cells={n['pa_trace']['n_pa_wrong_cells']} | "
            f"PA monotonicity violations={n['pa_trace']['n_monotonicity_violations']}"
        )
        p(f"First PA error: {n['pa_trace']['first_pa_error']}")
        p()
        p(format_reasoning_trace_readable(nss_comp.get("reasoning"), gt_env=gt_env))
        p()

        p("NSS FINAL SOLUTION")
        p("-" * 100)
        p(format_solution_table(nss_comp.get("solution")))
        p()

        if n["pa_trace"]["monotonicity_violations"]:
            p("NSS PA MONOTONICITY VIOLATIONS")
            p("-" * 100)
            for violation in n["pa_trace"]["monotonicity_violations"]:
                p(json.dumps(violation, ensure_ascii=False))
            p()


def write_outcome_size_summary(
    root_dir: Path,
    rows: Sequence[Dict[str, Any]],
    outcome_label: str,
) -> None:
    """Write counts for one paired outcome, grouped by difficulty and exact size."""
    root_dir.mkdir(parents=True, exist_ok=True)

    by_size = Counter(row["size"] for row in rows)
    by_difficulty = Counter(row["difficulty"] for row in rows)

    with (root_dir / "size_summary.txt").open("w", encoding="utf-8") as out:
        out.write("=" * 100 + "\n")
        out.write(f"{outcome_label} — CASE COUNTS BY PUZZLE SIZE\n")
        out.write("=" * 100 + "\n\n")

        out.write("BY DIFFICULTY\n")
        out.write("-" * 60 + "\n")
        for difficulty in ["Small", "Medium", "Large", "XL", "Unknown"]:
            if by_difficulty[difficulty]:
                out.write(f"{difficulty:<12} {by_difficulty[difficulty]:>6}\n")

        out.write("\nBY EXACT SIZE\n")
        out.write("-" * 60 + "\n")
        out.write(f"{'Size':<12}{'Difficulty':<14}{'Cases':>8}\n")
        out.write("-" * 34 + "\n")

        def size_sort_key(size: str):
            m = re.fullmatch(r"(\d+)x(\d+)", size)
            return (int(m.group(1)), int(m.group(2))) if m else (999, 999)

        for size in sorted(by_size, key=size_sort_key):
            difficulty = DIFFICULTY_BY_SIZE.get(size, "Unknown")
            out.write(f"{size:<12}{difficulty:<14}{by_size[size]:>8}\n")


def write_readable_traces_by_outcome_and_size(
    output_dir: Path,
    paired_rows: Sequence[Dict[str, Any]],
    original_raw_by_id: Dict[str, Dict[str, Any]],
    nss_raw_by_id: Dict[str, Dict[str, Any]],
    source_puzzle_map: Optional[Dict[str, str]] = None,
) -> Dict[str, int]:
    """
    Create:

      output_dir/
        NSS_FIXES/
          size_summary.txt
          2x2/
            001_<id>.txt
            ...
          5x4/
            ...

        NSS_REGRESSES/
          size_summary.txt
          ...

    Every case file contains GT, clues, annotated Original trace,
    annotated NSS trace (including PAs), final solutions, and paired diagnostics.
    """
    configs = [
        ("NSS_FIX", "NSS_FIXES"),
        ("NSS_REGRESSION", "NSS_REGRESSES"),
    ]

    written_counts: Dict[str, int] = {}

    for outcome_code, dirname in configs:
        rows = [row for row in paired_rows if row["outcome"] == outcome_code]
        root = output_dir / dirname
        root.mkdir(parents=True, exist_ok=True)

        write_outcome_size_summary(root, rows, dirname)

        counters: Counter = Counter()
        for row in rows:
            size = row["size"] or "UNKNOWN"
            size_dir = root / safe_filename_component(size)
            size_dir.mkdir(parents=True, exist_ok=True)

            counters[size] += 1
            pid = row["id"]
            filename = f"{counters[size]:03d}_{safe_filename_component(pid)}.txt"

            write_one_readable_paired_trace(
                size_dir / filename,
                row,
                original_raw_by_id[pid],
                nss_raw_by_id[pid],
                source_puzzle_map,
            )

        written_counts[dirname] = len(rows)

    return written_counts


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original",
        default="./Input_Logs/gpt51_outputs_test_700_temp_0.jsonl",
        help="Original-prompt JSONL log.",
    )

    parser.add_argument(
        "--nss",
        default="./Input_Logs/gpt51_outputs_test_700_mlxl_nss_temp_0.jsonl",
        help="NSS / PA-prompt JSONL log.",
    )

    parser.add_argument(
        "--output-dir",
        default="./Outputs/Paired_Error_Analysis",
        help="Output folder.",
    )

    parser.add_argument(
        "--source-dataset",
        default="./Input_Logs/pid_to_puzzle_dic.json",
        help=(
            "Original ZebraLogic puzzle lookup file. Default: "
            "./Input_Logs/pid_to_puzzle_dic.json. The expected format is "
            "a JSON mapping from puzzle ID to natural-language puzzle text; "
            "separate <pid>_sol entries are safely ignored for puzzle-text lookup."
        ),
    )

    parser.add_argument(
        "--source-id-field",
        default="id",
        help="ID field/column in --source-dataset.",
    )

    parser.add_argument(
        "--source-puzzle-field",
        default="puzzle",
        help="Natural-language puzzle field/column in --source-dataset.",
    )

    args = parser.parse_args()

    original_path = Path(args.original)
    nss_path = Path(args.nss)
    out_dir = Path(args.output_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_path = (
        Path(args.source_dataset)
        if args.source_dataset
        else None
    )

    if source_path is not None and source_path.exists():
        source_puzzle_map = load_source_puzzle_map(
            source_path,
            id_field=args.source_id_field,
            puzzle_field=args.source_puzzle_field,
        )

        print(
            f"Loaded original puzzle text for "
            f"{len(source_puzzle_map)} IDs from "
            f"{source_path}"
        )

    else:
        source_puzzle_map = {}

        if source_path is not None:
            print(
                f"WARNING: source puzzle file not found: {source_path}"
            )
        else:
            print(
                "No source puzzle file supplied."
            )

    original_records = read_records(
        original_path
    )

    nss_records = read_records(
        nss_path
    )

    aligned = align_by_id(
        original_records,
        nss_records,
    )

    # --------------------------------------------------------------
    # Puzzle-text ID alignment diagnostics
    # --------------------------------------------------------------
    log_ids = {
        str(r.get("id", "")).strip()
        for r in original_records
    }

    source_ids = set(source_puzzle_map.keys())

    matched_puzzle_ids = log_ids & source_ids
    missing_puzzle_ids = log_ids - source_ids

    print("\nPUZZLE-TEXT ID ALIGNMENT")
    print("=" * 80)
    print(f"Log puzzle IDs          : {len(log_ids)}")
    print(f"Puzzle-text lookup IDs  : {len(source_ids)}")
    print(f"Matched puzzle IDs      : {len(matched_puzzle_ids)}")
    print(f"Missing puzzle IDs      : {len(missing_puzzle_ids)}")

    if missing_puzzle_ids:
        print("\nFirst 20 missing puzzle IDs:")
        for missing_pid in sorted(missing_puzzle_ids)[:20]:
            print("  ", repr(missing_pid))

    # Concrete diagnostic for the example discussed during development.
    example_pid = "lgp-test-2x6-15"
    if example_pid in log_ids:
        print(
            f"\nDiagnostic {example_pid}: "
            f"{'FOUND' if example_pid in source_puzzle_map else 'NOT FOUND'} "
            f"in puzzle-text lookup."
        )

    original_raw_by_id = {
        str(r.get("id")): r
        for r in original_records
    }

    nss_raw_by_id = {
        str(r.get("id")): r
        for r in nss_records
    }

    paired: List[Dict[str, Any]] = []

    total = len(aligned)

    for i, (orec, nrec) in enumerate(
        aligned,
        1,
    ):
        if (
            i == 1
            or i % 50 == 0
            or i == total
        ):
            print(
                f"Analyzing paired case "
                f"{i}/{total} ..."
            )

        row = analyze_pair(
            orec,
            nrec,
        )

        paired.append(row)

    fixes = [
        r for r in paired
        if r["outcome"] == "NSS_FIX"
    ]

    regressions = [
        r for r in paired
        if r["outcome"] == "NSS_REGRESSION"
    ]

    summary_path = (
        out_dir
        / "paired_error_analysis_summary.txt"
    )

    case_jsonl_path = (
        out_dir
        / "paired_case_analysis.jsonl"
    )

    fixes_path = (
        out_dir
        / "nss_fixes_detailed.txt"
    )

    regressions_path = (
        out_dir
        / "nss_regressions_detailed.txt"
    )

    write_summary(
        summary_path,
        paired,
    )

    write_case_jsonl(
        case_jsonl_path,
        paired,
    )

    write_detailed_cases(
        fixes_path,
        fixes,
        original_raw_by_id,
        nss_raw_by_id,
        source_puzzle_map,
    )

    write_detailed_cases(
        regressions_path,
        regressions,
        original_raw_by_id,
        nss_raw_by_id,
        source_puzzle_map,
    )

    readable_counts = write_readable_traces_by_outcome_and_size(
        out_dir,
        paired,
        original_raw_by_id,
        nss_raw_by_id,
        source_puzzle_map,
    )

    print("\nReadable paired traces written by outcome and exact puzzle size:")
    print(f"  NSS_FIXES     : {readable_counts.get('NSS_FIXES', 0)} cases")
    print(f"  NSS_REGRESSES : {readable_counts.get('NSS_REGRESSES', 0)} cases")

    print("\nOutputs written to:")

    for p in (
        summary_path,
        case_jsonl_path,
        fixes_path,
        regressions_path,
        out_dir / "NSS_FIXES",
        out_dir / "NSS_REGRESSES",
    ):
        print("  ", p.resolve())


if __name__ == "__main__":
    main()
