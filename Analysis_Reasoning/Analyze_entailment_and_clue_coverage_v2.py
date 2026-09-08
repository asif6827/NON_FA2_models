#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two ZebraLogic prompt logs using GLOBAL puzzle entailment plus
explicit-trace diagnostics.

PRIMARY METHODOLOGY
===================

Let:
  S_i      = the i-th symbolic reasoning step
  G        = the complete ground-truth assignment
  B        = ZebraLogic base constraints (domain + uniqueness)
  C_all    = all parseable syntactic clue constraints in the puzzle
  P        = B + C_all, i.e. the complete formal puzzle theory

Primary S_i classification:

  1) ENTAILED
       G |= S_i  AND  P |= S_i

       The step is correct under GT and is logically implied by the complete
       puzzle constraints. This is counted as GOOD reasoning even if the model
       does not explicitly name the clue(s) that justify it.

  2) GT_CONSISTENT_NOT_ENTAILED
       G |= S_i  AND  P !|= S_i

       The step agrees with GT but is not implied by the complete formal puzzle
       theory. This is NOT called "premature" merely because the model failed to
       cite a clue. It is better interpreted as an unsupported/lucky/hidden-
       assumption step, subject to the caveat that clue parsing or non-uniqueness
       may affect the result.

  3) GT_INCONSISTENT
       G !|= S_i

       The step is false under the ground-truth solution and is treated as a
       genuine reasoning error.

  4) UNVERIFIABLE
       The step cannot be reliably classified because of parsing, token,
       constraint, or solver failure.

SECONDARY TRACE-SUPPORT DIAGNOSTIC
==================================

The model sees all clues from the start, so explicit clue citation must NOT be
used to decide whether a correct step is logically entailed.

However, for interpretability we additionally compute whether the displayed
reasoning trace explicitly supports S_i.

Let:
  C_ref<=i  = clue IDs explicitly referenced in NL up to S_i
  S^E_<i    = earlier S_j steps classified ENTAILED
  T_i       = B + C_ref<=i + S^E_<i

Then, for an ENTAILED S_i:
  TRACE_SUPPORTED
      if T_i |= S_i

  GLOBAL_ENTAILED_NOT_TRACE_SUPPORTED
      if P |= S_i but T_i !|= S_i

The second case is NOT treated as a reasoning error. It means the step is a
valid consequence of the puzzle, but the explicit NL/S trace does not expose
enough of the supporting chain.

CLUE COVERAGE
=============

For each puzzle:
    unique clue IDs explicitly referenced in NL reasoning
    -----------------------------------------------------
              unique clue IDs in syntactic_clues

Clue Coverage is a descriptive "explicit-use" metric only. Low clue coverage
does NOT imply poor reasoning, because the model may validly infer a fact without
explicitly naming every supporting clue.

The script compares Original vs NSS overall and by ZebraLogic difficulty:
Small, Medium, Large, XL.

Requirements:
    pip install z3-solver

Typical usage:
    python Analyze_entailment_and_clue_coverage_v2.py

or:
    python Analyze_entailment_and_clue_coverage_v2.py \
      --original ./Input_Logs/gpt51_outputs_test_700_temp_0.jsonl \
      --nss ./Input_Logs/gpt51_outputs_test_700_mlxl_nss_temp_0.jsonl \
      --output-dir ./Outputs/Entailment_Analysis_v2
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import z3
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This analysis requires z3-solver. Install it in the environment with "
        "`pip install z3-solver` (or run in the same environment as your Z3 "
        "reward validator)."
    ) from exc


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
    s = re.sub(r"\s+", "", s)
    return s


# =============================================================================
# Generic JSON / answer extraction
# =============================================================================


def parse_outer_record(line: str) -> Dict[str, Any]:
    return json.loads(line)


def _extract_answer_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    m = re.search(r"<answer\b[^>]*>(.*?)</answer\s*>", text, flags=re.I | re.S)
    return m.group(1).strip() if m else text


def _balanced_json_value(text: str, marker: str) -> Any:
    """Recover a JSON object/list immediately following a JSON key marker."""
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
    answer_text = _extract_answer_text(text)
    payload = None
    full_payload_ok = False

    try:
        candidate = json.loads(answer_text)
        if isinstance(candidate, dict):
            payload = candidate
            full_payload_ok = True
    except Exception:
        payload = None

    if payload is not None:
        return {
            "payload": payload,
            "n_houses": payload.get("n_houses"),
            "attribute_values": payload.get("attribute_values"),
            "syntactic_clues": payload.get("syntactic_clues"),
            "reasoning": payload.get("reasoning"),
            "solution": payload.get("solution"),
            "full_payload_ok": True,
            "extraction_method": "full_answer_json",
        }

    # Best-effort component recovery for malformed full JSON.
    n_houses = None
    m = re.search(r'"n_houses"\s*:\s*(\d+)', answer_text)
    if m:
        n_houses = int(m.group(1))

    return {
        "payload": None,
        "n_houses": n_houses,
        "attribute_values": _balanced_json_value(answer_text, '"attribute_values"'),
        "syntactic_clues": _balanced_json_value(answer_text, '"syntactic_clues"'),
        "reasoning": _balanced_json_value(answer_text, '"reasoning"'),
        "solution": _balanced_json_value(answer_text, '"solution"'),
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


def normalize_table_strict(table: Any) -> Optional[Tuple[Dict[int, Dict[str, str]], List[str]]]:
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
        values: Dict[str, str] = {}
        for i, h in enumerate(headers):
            if i == house_idx:
                continue
            values[h] = canonicalize(row[i])
        out[house] = values
    return (out, attrs) if out else None


def compute_case_accuracy(gt: Any, pred: Any) -> Dict[str, Any]:
    gt_norm = normalize_table_strict(gt)
    if gt_norm is None:
        raise ValueError("Invalid ground truth table")
    gt_by_house, gt_cols = gt_norm
    total = sum(len(v) for v in gt_by_house.values())

    pred_norm = normalize_table_strict(pred)
    if pred_norm is None:
        return {"puzzle_accuracy": 0.0, "cell_accuracy": 0.0, "correct_cells": 0, "total_cells": total}

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
        "total_cells": total,
    }


# =============================================================================
# GT environment
# =============================================================================


def gt_entity_to_house(gt: Dict[str, Any]) -> Tuple[Dict[str, int], Set[str]]:
    norm = normalize_table_strict(gt)
    if norm is None:
        raise ValueError("Invalid GT")
    by_house, _ = norm
    env: Dict[str, int] = {}
    ambiguous: Set[str] = set()
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
# Reasoning extraction and clue references
# =============================================================================


def extract_s_nl_pairs(reasoning: Any) -> List[Dict[str, Any]]:
    """Return ordered {k, key, nl, expr, order} records for list/dict formats."""
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
            steps.append({"k": k, "key": f"S{k}", "nl": nl, "expr": value.strip(), "order": order})
        return steps

    if isinstance(reasoning, list):
        pending_nl: List[str] = []
        for order, item in enumerate(reasoning):
            if not isinstance(item, str):
                continue
            m = re.match(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", item, flags=re.I | re.S)
            if m:
                k = int(m.group(1))
                nl = " ".join(pending_nl).strip()
                pending_nl = []
                steps.append({"k": k, "key": f"S{k}", "nl": nl, "expr": m.group(2).strip(), "order": order})
            else:
                clean = re.sub(r"^\s*NL\d+\s*:\s*", "", item, flags=re.I)
                pending_nl.append(clean.strip())
        return steps

    return steps


def extract_clue_refs(text: Any, valid_ids: Optional[Set[int]] = None) -> Set[int]:
    """Extract references such as 'Clue 3', 'Clues 1 and 2', 'C4'.

    The regex intentionally consumes only the numeric list immediately after
    the word clue(s), so house numbers later in the sentence are not mistaken
    for clue IDs.
    """
    s = str(text or "")
    ids: Set[int] = set()

    for x in re.findall(r"\bC\s*(\d+)\b", s, flags=re.I):
        ids.add(int(x))

    clue_list_re = re.compile(
        r"\bclues?\s+(\d+(?:(?:\s*,\s*\d+)|(?:\s*,?\s*(?:and|&)\s*\d+))*)",
        flags=re.I,
    )
    for m in clue_list_re.finditer(s):
        for x in re.findall(r"\d+", m.group(1)):
            ids.add(int(x))

    if valid_ids is not None:
        ids &= valid_ids
    return ids


# =============================================================================
# Z3 variable model + AST parser
# =============================================================================


def _aliases(value: Any) -> Set[str]:
    raw = str(value).strip()
    can = canonicalize(raw)
    return {raw, raw.lower(), can, can.lower()}


def build_z3_model(
    n_houses: int,
    attribute_values: Dict[str, List[Any]],
) -> Tuple[Dict[str, Any], Set[str], List[Any]]:
    """Build the Zebra variable model.

    If the same textual value appears in multiple attributes (e.g. "red" in
    Color and HairColor), that alias is intrinsically ambiguous in the emitted
    DSL. We keep the case analyzable, but any clue/S-step that uses that
    ambiguous token becomes UNVERIFIABLE instead of crashing the whole case.
    """
    alias_candidates: Dict[str, List[Any]] = defaultdict(list)
    attr_vars: Dict[str, List[Any]] = {}
    all_vars: List[Any] = []

    for ai, (attr, values) in enumerate(attribute_values.items()):
        if not isinstance(values, list):
            raise ValueError(f"attribute_values[{attr!r}] is not a list")
        vars_for_attr = []
        for vi, value in enumerate(values):
            zv = z3.Int(f"v_{ai}_{vi}")
            vars_for_attr.append(zv)
            all_vars.append(zv)
            for a in _aliases(value):
                if all(not z3.eq(existing, zv) for existing in alias_candidates[a]):
                    alias_candidates[a].append(zv)
        attr_vars[str(attr)] = vars_for_attr

    alias_map: Dict[str, Any] = {}
    ambiguous_aliases: Set[str] = set()
    for alias, vars_ in alias_candidates.items():
        if len(vars_) == 1:
            alias_map[alias] = vars_[0]
        else:
            ambiguous_aliases.add(alias)

    axioms: List[Any] = []
    for v in all_vars:
        axioms.append(z3.And(v >= 1, v <= int(n_houses)))
    for vals in attr_vars.values():
        if len(vals) >= 2:
            axioms.append(z3.Distinct(*vals))
    return alias_map, ambiguous_aliases, axioms


def lookup_z3_name(name: str, alias_map: Dict[str, Any], ambiguous_aliases: Optional[Set[str]] = None):
    candidates = [name, name.lower(), canonicalize(name), canonicalize(name).lower()]
    if ambiguous_aliases:
        for c in candidates:
            if c in ambiguous_aliases:
                raise KeyError(f"Ambiguous entity token across attributes: {name!r}")
    for c in candidates:
        if c in alias_map:
            return alias_map[c]
    raise KeyError(f"Unknown entity token: {name!r}")


def ast_to_z3(node: ast.AST, alias_map: Dict[str, Any]):
    if isinstance(node, ast.Expression):
        return ast_to_z3(node.body, alias_map)
    if isinstance(node, ast.Name):
        return lookup_z3_name(node.id, alias_map)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return z3.BoolVal(node.value)
        if isinstance(node.value, int):
            return z3.IntVal(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -ast_to_z3(node.operand, alias_map)
    if isinstance(node, ast.BinOp):
        l = ast_to_z3(node.left, alias_map)
        r = ast_to_z3(node.right, alias_map)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        raise ValueError(f"Unsupported arithmetic op: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = ast_to_z3(node.left, alias_map)
        clauses = []
        for op, comp in zip(node.ops, node.comparators):
            right = ast_to_z3(comp, alias_map)
            if isinstance(op, ast.Eq):
                clauses.append(left == right)
            elif isinstance(op, ast.NotEq):
                clauses.append(left != right)
            elif isinstance(op, ast.Lt):
                clauses.append(left < right)
            elif isinstance(op, ast.LtE):
                clauses.append(left <= right)
            elif isinstance(op, ast.Gt):
                clauses.append(left > right)
            elif isinstance(op, ast.GtE):
                clauses.append(left >= right)
            else:
                raise ValueError(f"Unsupported comparison: {type(op).__name__}")
            left = right
        return clauses[0] if len(clauses) == 1 else z3.And(*clauses)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id.lower()
        args = [ast_to_z3(a, alias_map) for a in node.args]
        if fn == "and":
            if len(args) < 2:
                raise ValueError("And requires >=2 args")
            return z3.And(*args)
        if fn == "or":
            if len(args) < 2:
                raise ValueError("Or requires >=2 args")
            return z3.Or(*args)
        if fn == "not":
            if len(args) != 1:
                raise ValueError("Not requires 1 arg")
            return z3.Not(args[0])
        raise ValueError(f"Unsupported function: {node.func.id}")
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def parse_z3_expr(expr: Any, alias_map: Dict[str, Any]):
    clean = str(expr).strip().rstrip(".").strip()
    tree = ast.parse(clean, mode="eval")
    zexpr = ast_to_z3(tree, alias_map)
    if not z3.is_bool(zexpr):
        raise ValueError("Expression does not produce a Boolean constraint")
    return zexpr


# =============================================================================
# GT evaluator for S_i
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
        raise GTEvalError(f"Unsupported arithmetic op: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = eval_ast_gt(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = eval_ast_gt(comp, env)
            if isinstance(op, ast.Eq): ok = left == right
            elif isinstance(op, ast.NotEq): ok = left != right
            elif isinstance(op, ast.Lt): ok = left < right
            elif isinstance(op, ast.LtE): ok = left <= right
            elif isinstance(op, ast.Gt): ok = left > right
            elif isinstance(op, ast.GtE): ok = left >= right
            else: raise GTEvalError(f"Unsupported comparison: {type(op).__name__}")
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
                raise GTEvalError("Not requires one arg")
            return not bool(eval_ast_gt(node.args[0], env))
        raise GTEvalError(f"Unsupported function: {node.func.id}")
    raise GTEvalError(f"Unsupported AST node: {type(node).__name__}")


def eval_expr_gt(expr: Any, env: Dict[str, int]) -> Tuple[Optional[bool], Optional[str]]:
    clean = str(expr).strip().rstrip(".").strip()
    try:
        value = eval_ast_gt(ast.parse(clean, mode="eval"), env)
        if not isinstance(value, bool):
            raise GTEvalError("Expression did not evaluate to bool")
        return value, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# =============================================================================
# Syntactic clue parsing
# =============================================================================


def parse_clue_groups(
    syntactic_clues: Any,
    alias_map: Dict[str, Any],
) -> Tuple[Dict[int, List[Any]], Dict[int, List[str]], List[str]]:
    groups: Dict[int, List[Any]] = defaultdict(list)
    errors: Dict[int, List[str]] = defaultdict(list)
    raw_unidentified: List[str] = []

    if not isinstance(syntactic_clues, list):
        return {}, {}, ["syntactic_clues is not a list"]

    for raw in syntactic_clues:
        text = str(raw).strip()
        m = re.match(r"^\s*C(\d+)\s*:\s*(.+?)\s*$", text, flags=re.I | re.S)
        if not m:
            raw_unidentified.append(text)
            continue
        cid = int(m.group(1))
        expr = m.group(2).strip().rstrip(".").strip()
        try:
            groups[cid].append(parse_z3_expr(expr, alias_map))
        except Exception as exc:
            errors[cid].append(f"{expr} :: {type(exc).__name__}: {exc}")

    return dict(groups), dict(errors), raw_unidentified


# =============================================================================
# Solver helpers
# =============================================================================


def solver_status(constraints: Sequence[Any], timeout_ms: int) -> Any:
    s = z3.Solver()
    s.set("timeout", int(timeout_ms))
    s.add(*constraints)
    return s.check()


def is_entailed(constraints: Sequence[Any], proposition: Any, timeout_ms: int) -> Tuple[Optional[bool], str]:
    """
    Return (entailed?, status).
    entailed=None means prefix/proof check was unknown or prefix inconsistent.
    """
    s = z3.Solver()
    s.set("timeout", int(timeout_ms))
    s.add(*constraints)
    base_status = s.check()
    if base_status == z3.unknown:
        return None, "prefix_unknown"
    if base_status == z3.unsat:
        return None, "prefix_unsat"

    s.push()
    s.add(z3.Not(proposition))
    proof_status = s.check()
    s.pop()
    if proof_status == z3.unsat:
        return True, "entailed"
    if proof_status == z3.sat:
        return False, "not_entailed"
    return None, "proof_unknown"


# =============================================================================
# Per-case reasoning analysis
# =============================================================================


def analyze_case_reasoning(
    record: Dict[str, Any],
    *,
    timeout_ms: int,
) -> Dict[str, Any]:
    """
    Analyze every S_i using the complete puzzle theory as the PRIMARY
    entailment test.

    Primary theory:
        P = B + all parseable syntactic clues

    Secondary explicit-trace theory:
        T_i = B + clues explicitly referenced up to i
                + earlier S_j classified ENTAILED

    Important:
      - Explicit clue citation is NOT required for ENTAILED.
      - TRACE support is reported separately and is not a correctness label.
    """
    comp = extract_payload_components(record)
    gt = record.get("ground_truth", {})
    pred = comp.get("solution")
    accuracy = compute_case_accuracy(gt, pred)

    size = normalize_size(record.get("size", ""))
    difficulty = DIFFICULTY_BY_SIZE.get(size, "Unknown")

    n_houses = comp.get("n_houses")
    attrs = comp.get("attribute_values")
    clues = comp.get("syntactic_clues")
    reasoning = comp.get("reasoning")

    result: Dict[str, Any] = {
        "id": str(record.get("id", "UNKNOWN")),
        "size": size,
        "difficulty": difficulty,
        "puzzle_accuracy": accuracy["puzzle_accuracy"],
        "cell_accuracy": accuracy["cell_accuracy"],
        "full_payload_ok": bool(comp.get("full_payload_ok")),
        "extraction_method": comp.get("extraction_method"),
        "analysis_ok": False,
        "analysis_error": None,

        "n_s_steps": 0,
        "n_entailed": 0,
        "n_gt_consistent_not_entailed": 0,
        "n_gt_inconsistent": 0,
        "n_unverifiable": 0,

        # Secondary trace-support diagnostics, only for globally ENTAILED steps.
        "n_trace_supported": 0,
        "n_global_entailed_not_trace_supported": 0,
        "n_trace_support_unverifiable": 0,

        # Explicit clue-reference diagnostics.
        "n_clues_total": 0,
        "n_clues_covered": 0,
        "clue_coverage": 0.0,
        "n_steps_with_explicit_clue_ref": 0,

        "step_details": [],
        "clue_parse_errors": {},
        "unidentified_clues": [],
        "unknown_clue_refs": [],
    }

    try:
        if not isinstance(n_houses, int) or n_houses <= 0:
            raise ValueError("missing/invalid n_houses")
        if not isinstance(attrs, dict) or not attrs:
            raise ValueError("missing/invalid attribute_values")

        alias_map, ambiguous_aliases, base_axioms = build_z3_model(n_houses, attrs)
        clue_groups, clue_errors, unidentified = parse_clue_groups(clues, alias_map)

        clue_ids: Set[int] = set(clue_groups) | set(clue_errors)
        if not clue_ids:
            raise ValueError("no identifiable syntactic clue IDs")

        env, ambiguous_gt = gt_entity_to_house(gt)
        pairs = extract_s_nl_pairs(reasoning)

        # All parseable formal clue constraints. These define the PRIMARY theory.
        all_parsed_clue_constraints = [
            c
            for cid in sorted(clue_groups)
            for c in clue_groups[cid]
        ]

        # If any clue failed to parse, a negative global-entailment result is
        # incomplete. Positive entailment remains sound because a subset already
        # suffices to prove the proposition.
        any_clue_parse_error = bool(clue_errors)

        global_theory: List[Any] = list(base_axioms) + list(all_parsed_clue_constraints)

        active_clues: Set[int] = set()
        covered_clues: Set[int] = set()
        unknown_refs: Set[int] = set()

        # Earlier globally ENTAILED S_j are safe to use in the secondary trace
        # support state.
        accepted_entailed_s: List[Any] = []

        step_details: List[Dict[str, Any]] = []
        counts = defaultdict(int)

        for pos, step in enumerate(pairs, 1):
            refs_raw = extract_clue_refs(step.get("nl", ""), valid_ids=None)
            refs_known = refs_raw & clue_ids
            refs_unknown = refs_raw - clue_ids

            if refs_known:
                counts["steps_with_ref"] += 1

            active_clues |= refs_known
            covered_clues |= refs_known
            unknown_refs |= refs_unknown

            gt_value, gt_error = eval_expr_gt(step["expr"], env)

            detail = dict(step)
            detail.update({
                "position": pos,
                "clue_refs_current": sorted(refs_known),
                "clue_refs_active": sorted(active_clues),
                "unknown_clue_refs_current": sorted(refs_unknown),

                "gt_consistent": gt_value,
                "gt_eval_error": gt_error,

                "classification": None,

                # PRIMARY global entailment
                "global_entails": None,
                "global_entailment_status": None,

                # SECONDARY explicit-trace support
                "trace_supported": None,
                "trace_support_status": None,
            })

            # ---------------------------------------------------------
            # 1) GT consistency
            # ---------------------------------------------------------
            if gt_value is False:
                detail["classification"] = "GT_INCONSISTENT"
                counts["gt_inconsistent"] += 1
                step_details.append(detail)
                continue

            if gt_value is None:
                detail["classification"] = "UNVERIFIABLE"
                counts["unverifiable"] += 1
                step_details.append(detail)
                continue

            # ---------------------------------------------------------
            # 2) Parse S_i into Z3
            # ---------------------------------------------------------
            try:
                zstep = parse_z3_expr(step["expr"], alias_map)
            except Exception as exc:
                detail["classification"] = "UNVERIFIABLE"
                detail["global_entailment_status"] = (
                    f"S_parse_error: {type(exc).__name__}: {exc}"
                )
                counts["unverifiable"] += 1
                step_details.append(detail)
                continue

            # ---------------------------------------------------------
            # 3) PRIMARY classification: full puzzle entailment
            # ---------------------------------------------------------
            global_ent, global_status = is_entailed(
                global_theory,
                zstep,
                timeout_ms,
            )
            detail["global_entails"] = global_ent
            detail["global_entailment_status"] = global_status

            if global_ent is True:
                detail["classification"] = "ENTAILED"
                counts["entailed"] += 1

            elif global_ent is False:
                # If one or more clues failed to parse, the formal global theory
                # is incomplete. A negative result is therefore not reliable.
                if any_clue_parse_error:
                    detail["classification"] = "UNVERIFIABLE"
                    detail["global_entailment_status"] = (
                        "not_entailed_with_incomplete_clue_parse"
                    )
                    counts["unverifiable"] += 1
                    step_details.append(detail)
                    continue

                detail["classification"] = "GT_CONSISTENT_NOT_ENTAILED"
                counts["gt_consistent_not_entailed"] += 1

            else:
                detail["classification"] = "UNVERIFIABLE"
                counts["unverifiable"] += 1
                step_details.append(detail)
                continue

            # ---------------------------------------------------------
            # 4) SECONDARY trace-support diagnostic
            #
            # This does NOT change the primary classification.
            # ---------------------------------------------------------
            if detail["classification"] == "ENTAILED":
                trace_constraints: List[Any] = list(base_axioms)

                active_clue_parse_errors = {
                    cid: clue_errors[cid]
                    for cid in active_clues
                    if cid in clue_errors
                }

                for cid in sorted(active_clues):
                    trace_constraints.extend(clue_groups.get(cid, []))

                trace_constraints.extend(accepted_entailed_s)

                trace_ent, trace_status = is_entailed(
                    trace_constraints,
                    zstep,
                    timeout_ms,
                )
                detail["trace_supported"] = trace_ent
                detail["trace_support_status"] = trace_status

                if trace_ent is True:
                    counts["trace_supported"] += 1

                elif trace_ent is False:
                    if active_clue_parse_errors:
                        # Missing referenced clue constraints make a negative
                        # trace-support result incomplete.
                        detail["trace_supported"] = None
                        detail["trace_support_status"] = (
                            "trace_not_entailed_with_active_clue_parse_error"
                        )
                        detail["active_clue_parse_errors"] = active_clue_parse_errors
                        counts["trace_support_unverifiable"] += 1
                    else:
                        counts["global_entailed_not_trace_supported"] += 1

                else:
                    counts["trace_support_unverifiable"] += 1

                # Since the step is a valid consequence of the complete puzzle,
                # it is safe to make it available to later trace-support checks.
                accepted_entailed_s.append(zstep)

            step_details.append(detail)

        result.update({
            "analysis_ok": True,

            "n_s_steps": len(pairs),
            "n_entailed": counts["entailed"],
            "n_gt_consistent_not_entailed": counts["gt_consistent_not_entailed"],
            "n_gt_inconsistent": counts["gt_inconsistent"],
            "n_unverifiable": counts["unverifiable"],

            "n_trace_supported": counts["trace_supported"],
            "n_global_entailed_not_trace_supported":
                counts["global_entailed_not_trace_supported"],
            "n_trace_support_unverifiable":
                counts["trace_support_unverifiable"],

            "n_clues_total": len(clue_ids),
            "n_clues_covered": len(covered_clues),
            "clue_coverage":
                len(covered_clues) / len(clue_ids) if clue_ids else 0.0,
            "n_steps_with_explicit_clue_ref": counts["steps_with_ref"],

            "step_details": step_details,
            "clue_parse_errors": clue_errors,
            "unidentified_clues": unidentified,
            "unknown_clue_refs": sorted(unknown_refs),
            "ambiguous_gt_tokens": sorted(ambiguous_gt),
            "ambiguous_dsl_aliases": sorted(ambiguous_aliases),
            "any_clue_parse_error": any_clue_parse_error,
        })

    except Exception as exc:
        result["analysis_error"] = f"{type(exc).__name__}: {exc}"

    return result


# =============================================================================
# Log I/O and aggregation
# =============================================================================


def read_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = parse_outer_record(line)
            except Exception as exc:
                raise ValueError(
                    f"Failed JSONL parse at {path}:{line_no}: {exc}"
                ) from exc
            records.append(rec)
    return records


def align_by_id(
    a: Sequence[Dict[str, Any]],
    b: Sequence[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    ma = {str(r.get("id")): r for r in a}
    mb = {str(r.get("id")): r for r in b}

    if len(ma) != len(a) or len(mb) != len(b):
        raise ValueError("Duplicate IDs found in one of the logs")

    if set(ma) != set(mb):
        only_a = sorted(set(ma) - set(mb))[:10]
        only_b = sorted(set(mb) - set(ma))[:10]
        raise ValueError(
            f"ID sets differ. only original={only_a}, only NSS={only_b}"
        )

    # Preserve original-file order.
    return [(r, mb[str(r.get("id"))]) for r in a]


def aggregate(
    cases: Sequence[Dict[str, Any]],
    difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    rows = [
        c
        for c in cases
        if difficulty is None or c.get("difficulty") == difficulty
    ]
    ok = [c for c in rows if c.get("analysis_ok")]

    sums = defaultdict(float)
    clue_cov_sum = 0.0
    clue_num = 0
    clue_den = 0

    for c in ok:
        for k in (
            "n_s_steps",
            "n_entailed",
            "n_gt_consistent_not_entailed",
            "n_gt_inconsistent",
            "n_unverifiable",
            "n_trace_supported",
            "n_global_entailed_not_trace_supported",
            "n_trace_support_unverifiable",
        ):
            sums[k] += float(c.get(k, 0))

        clue_cov_sum += float(c.get("clue_coverage", 0.0))
        clue_num += int(c.get("n_clues_covered", 0))
        clue_den += int(c.get("n_clues_total", 0))

    # Primary verifiable labels.
    verifiable = (
        sums["n_entailed"]
        + sums["n_gt_consistent_not_entailed"]
        + sums["n_gt_inconsistent"]
    )

    # Secondary trace-support denominator = globally entailed steps for which
    # trace support was either proven or disproven.
    trace_verifiable = (
        sums["n_trace_supported"]
        + sums["n_global_entailed_not_trace_supported"]
    )

    n_rows = len(rows)

    pacc = (
        sum(float(c.get("puzzle_accuracy", 0.0)) for c in rows) / n_rows
        if n_rows
        else 0.0
    )
    cacc = (
        sum(float(c.get("cell_accuracy", 0.0)) for c in rows) / n_rows
        if n_rows
        else 0.0
    )

    return {
        "n_cases": n_rows,
        "n_analysis_ok": len(ok),

        "puzzle_accuracy": pacc,
        "macro_cell_accuracy": cacc,

        "n_s_steps": int(sums["n_s_steps"]),
        "n_verifiable_steps": int(verifiable),

        "n_entailed": int(sums["n_entailed"]),
        "n_gt_consistent_not_entailed":
            int(sums["n_gt_consistent_not_entailed"]),
        "n_gt_inconsistent": int(sums["n_gt_inconsistent"]),
        "n_unverifiable": int(sums["n_unverifiable"]),

        "entailed_rate":
            sums["n_entailed"] / verifiable if verifiable else 0.0,

        "gt_consistent_not_entailed_rate":
            sums["n_gt_consistent_not_entailed"] / verifiable
            if verifiable
            else 0.0,

        "gt_inconsistent_rate":
            sums["n_gt_inconsistent"] / verifiable if verifiable else 0.0,

        "unverifiable_rate_all_steps":
            sums["n_unverifiable"] / sums["n_s_steps"]
            if sums["n_s_steps"]
            else 0.0,

        # Trace-support diagnostics
        "n_trace_supported": int(sums["n_trace_supported"]),
        "n_global_entailed_not_trace_supported":
            int(sums["n_global_entailed_not_trace_supported"]),
        "n_trace_support_unverifiable":
            int(sums["n_trace_support_unverifiable"]),
        "trace_support_rate":
            sums["n_trace_supported"] / trace_verifiable
            if trace_verifiable
            else 0.0,

        # Explicit clue coverage
        "macro_clue_coverage":
            clue_cov_sum / len(ok) if ok else 0.0,
        "micro_clue_coverage":
            clue_num / clue_den if clue_den else 0.0,
        "clues_covered": clue_num,
        "clues_total": clue_den,
    }


def winner_higher(a: float, b: float, eps: float = 1e-12) -> str:
    if abs(a - b) <= eps:
        return "Tie"
    return "NSS" if b > a else "Original"


def winner_lower(a: float, b: float, eps: float = 1e-12) -> str:
    if abs(a - b) <= eps:
        return "Tie"
    return "NSS" if b < a else "Original"


def reasoning_quality_winner(
    orig: Dict[str, Any],
    nss: Dict[str, Any],
) -> Tuple[str, Dict[str, str]]:
    """
    Primary reasoning-quality comparison.

    Higher ENTAILED rate is better.
    Lower GT_INCONSISTENT rate is better.
    Lower GT_CONSISTENT_NOT_ENTAILED rate is better, but this third criterion
    should be interpreted cautiously because it can reflect non-uniqueness or
    incomplete formalization rather than a definite reasoning error.
    """
    wins = {
        "entailed_rate":
            winner_higher(orig["entailed_rate"], nss["entailed_rate"]),

        "gt_consistent_not_entailed_rate":
            winner_lower(
                orig["gt_consistent_not_entailed_rate"],
                nss["gt_consistent_not_entailed_rate"],
            ),

        "gt_inconsistent_rate":
            winner_lower(
                orig["gt_inconsistent_rate"],
                nss["gt_inconsistent_rate"],
            ),
    }

    ow = sum(v == "Original" for v in wins.values())
    nw = sum(v == "NSS" for v in wins.values())

    if ow == nw:
        return "Mixed/Tie", wins

    return ("NSS" if nw > ow else "Original"), wins


def build_verdict(
    orig: Dict[str, Any],
    nss: Dict[str, Any],
    label: str,
) -> List[str]:
    task = winner_higher(
        orig["puzzle_accuracy"],
        nss["puzzle_accuracy"],
    )

    rq, rq_parts = reasoning_quality_winner(orig, nss)

    coverage = winner_higher(
        orig["macro_clue_coverage"],
        nss["macro_clue_coverage"],
    )

    trace_support = winner_higher(
        orig["trace_support_rate"],
        nss["trace_support_rate"],
    )

    lines = [
        (
            f"{label}: task-performance winner = {task}; "
            f"primary reasoning-quality winner = {rq}."
        ),
        (
            "  Primary reasoning-quality subcriteria: "
            f"entailed-rate={rq_parts['entailed_rate']}, "
            "GT-consistent-not-entailed-rate="
            f"{rq_parts['gt_consistent_not_entailed_rate']}, "
            f"GT-inconsistent-rate={rq_parts['gt_inconsistent_rate']}."
        ),
        (
            f"  Secondary diagnostics: explicit-clue-coverage winner = "
            f"{coverage}; explicit-trace-support winner = {trace_support}."
        ),
    ]

    if task == rq and task in {"Original", "NSS"}:
        lines.append(
            f"  Overall evidence favors {task}: it wins both exact puzzle "
            "performance and the majority of the primary reasoning-quality "
            "diagnostics."
        )

    elif (
        task in {"Original", "NSS"}
        and rq in {"Original", "NSS"}
        and task != rq
    ):
        lines.append(
            f"  Evidence is mixed: {task} solves more puzzles, while {rq} "
            "has the stronger primary reasoning-quality profile."
        )

    else:
        lines.append(
            "  No single prompt dominates both task performance and primary "
            "reasoning quality."
        )

    lines.append(
        "  Do NOT treat lower explicit clue coverage or lower explicit trace "
        "support as an automatic reasoning failure: all puzzle clues are "
        "available to the model from the start."
    )

    return lines


# =============================================================================
# Formal definitions + reporting
# =============================================================================


FORMAL_DEFINITIONS = r"""
================================================================================
FORMAL DEFINITIONS USED IN THIS ANALYSIS
================================================================================

Let:

  S_i      = the i-th symbolic reasoning step

  G        = the complete ground-truth assignment

  B        = ZebraLogic base constraints
             (domain constraints + within-attribute uniqueness constraints)

  C_all    = all parseable syntactic clue constraints for the puzzle

  P        = the complete formal puzzle theory:

                 P = B + C_all


PRIMARY CLASSIFICATION OF S_i
-----------------------------

1. ENTAILED

   S_i is ENTAILED if it is true under the ground truth and is logically
   implied by the complete puzzle theory.

       G |= S_i
       and
       P |= S_i

   Z3 operational test:

       P AND NOT(S_i)

   must be UNSAT.

   Interpretation:
       S_i is a correct logical consequence of the puzzle.

   IMPORTANT:
       The model does NOT need to explicitly cite the supporting clue in NL.
       Since the model receives all puzzle clues from the start, a logically
       valid deduction is counted as good even if the trace does not spell out
       every clue used to derive it.


2. GT_CONSISTENT_NOT_ENTAILED

   S_i is GT-consistent but NOT entailed if:

       G |= S_i
       and
       P !|= S_i

   Operationally:

       P AND NOT(S_i)

   is SAT.

   Interpretation:
       The statement agrees with the provided GT assignment, but the complete
       formal puzzle constraints do not force it.

   This is NOT automatically called "premature" merely because the model did
   not cite a clue.

   Possible explanations include:
       - an unsupported/lucky conclusion,
       - a hidden assumption,
       - a non-unique puzzle,
       - or incomplete formalization of the puzzle constraints.

   In a uniquely solvable, fully parsed ZebraLogic puzzle, this category should
   normally be rare.


3. GT_INCONSISTENT

   S_i is GT_INCONSISTENT if it is false under the complete ground truth:

       G !|= S_i

   Interpretation:
       S_i is genuinely inconsistent with the correct puzzle solution and is
       treated as a reasoning error.


4. UNVERIFIABLE

   S_i is UNVERIFIABLE when its GT truth or logical entailment cannot be
   reliably determined.

   Typical causes:
       - malformed symbolic syntax,
       - unknown / ambiguous entity token,
       - unsupported logical construction,
       - clue parsing failure that makes a negative proof incomplete,
       - Z3 timeout / UNKNOWN,
       - constraint-construction failure.

   UNVERIFIABLE is reported separately and is NOT automatically counted as
   GT_INCONSISTENT.


SECONDARY EXPLICIT-TRACE SUPPORT
--------------------------------

The primary ENTAILED label uses ALL puzzle clues because all clues are visible
to the model from the beginning.

For interpretability only, we also ask whether the model's DISPLAYED reasoning
trace explicitly supports an ENTAILED S_i.

Let:

  C_ref<=i = clue IDs explicitly referenced in NL up to and including S_i

  S^E_<i   = earlier S_j, j < i, classified ENTAILED

  T_i      = explicit trace-support state:

                 T_i = B + C_ref<=i + S^E_<i


For a globally ENTAILED S_i:

  TRACE_SUPPORTED
      if T_i |= S_i

  GLOBAL_ENTAILED_NOT_TRACE_SUPPORTED
      if P |= S_i but T_i !|= S_i


IMPORTANT:

  GLOBAL_ENTAILED_NOT_TRACE_SUPPORTED is NOT considered a reasoning error.

  It means the symbolic step is a valid consequence of the complete puzzle,
  but the displayed NL/S trace does not explicitly expose enough of the
  supporting chain.


CLUE COVERAGE
-------------

Let:

  C_total = number of unique syntactic clue IDs in the puzzle

  C_used  = number of unique clue IDs explicitly referenced by NL reasoning

Then:

                 Clue Coverage = C_used / C_total


Clue Coverage measures EXPLICIT clue citation/use in the displayed trace.

It is a descriptive breadth metric only.

Lower Clue Coverage does NOT imply worse logical reasoning, because a model may
derive a valid ENTAILED fact without explicitly naming every supporting clue.

================================================================================
"""


def print_formal_definitions() -> None:
    print(FORMAL_DEFINITIONS)


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def write_summary(
    path: Path,
    original_cases: Sequence[Dict[str, Any]],
    nss_cases: Sequence[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as out:

        def p(s: str = ""):
            print(s)
            out.write(s + "\n")

        p(FORMAL_DEFINITIONS.rstrip())
        p()

        p("=" * 170)
        p("ZEBRALOGIC GLOBAL-ENTAILMENT + EXPLICIT-TRACE + CLUE-COVERAGE ANALYSIS")
        p("=" * 170)
        p()

        p(
            "Primary labels use the COMPLETE puzzle theory B + all parseable "
            "syntactic clues."
        )
        p(
            "Explicit clue references are used only for the secondary trace-"
            "support and clue-coverage diagnostics."
        )
        p()

        header = (
            f"{'Difficulty':<10} {'System':<10} {'N':>5} "
            f"{'PAcc':>9} {'CAcc':>9} "
            f"{'S':>7} {'Entailed':>10} {'GT-ok/NotEnt':>13} "
            f"{'GT-wrong':>10} {'Unverif':>9} "
            f"{'Ent%':>8} {'NotEnt%':>9} {'GTerr%':>8} "
            f"{'TraceSup%':>10} {'ClueCov':>9}"
        )

        p(header)
        p("-" * len(header))

        agg_by_label: Dict[
            str,
            Tuple[Dict[str, Any], Dict[str, Any]]
        ] = {}

        for label in DIFFICULTY_ORDER:
            diff = None if label == "Overall" else label

            oa = aggregate(original_cases, diff)
            na = aggregate(nss_cases, diff)

            agg_by_label[label] = (oa, na)

            for sys_name, a in (("Original", oa), ("NSS", na)):
                p(
                    f"{label:<10} "
                    f"{sys_name:<10} "
                    f"{a['n_cases']:>5} "
                    f"{fmt_pct(a['puzzle_accuracy']):>9} "
                    f"{fmt_pct(a['macro_cell_accuracy']):>9} "
                    f"{a['n_s_steps']:>7} "
                    f"{a['n_entailed']:>10} "
                    f"{a['n_gt_consistent_not_entailed']:>13} "
                    f"{a['n_gt_inconsistent']:>10} "
                    f"{a['n_unverifiable']:>9} "
                    f"{fmt_pct(a['entailed_rate']):>8} "
                    f"{fmt_pct(a['gt_consistent_not_entailed_rate']):>9} "
                    f"{fmt_pct(a['gt_inconsistent_rate']):>8} "
                    f"{fmt_pct(a['trace_support_rate']):>10} "
                    f"{fmt_pct(a['macro_clue_coverage']):>9}"
                )

            p()

        p()
        p("=" * 170)
        p("SECONDARY EXPLICIT-TRACE SUPPORT")
        p("=" * 170)
        p(
            "These numbers apply only to S_i already classified ENTAILED by "
            "the complete puzzle theory."
        )
        p(
            "GLOBAL_ONLY means the deduction is logically valid, but the "
            "displayed trace does not explicitly expose enough support."
        )
        p()

        trace_header = (
            f"{'Difficulty':<10} {'System':<10} "
            f"{'Trace-supported':>17} {'Global-only':>14} "
            f"{'Trace-unverif':>15} {'TraceSup%':>11}"
        )
        p(trace_header)
        p("-" * len(trace_header))

        for label in DIFFICULTY_ORDER:
            oa, na = agg_by_label[label]
            for sys_name, a in (("Original", oa), ("NSS", na)):
                p(
                    f"{label:<10} {sys_name:<10} "
                    f"{a['n_trace_supported']:>17} "
                    f"{a['n_global_entailed_not_trace_supported']:>14} "
                    f"{a['n_trace_support_unverifiable']:>15} "
                    f"{fmt_pct(a['trace_support_rate']):>11}"
                )
            p()

        p()
        p("=" * 170)
        p("PROMPT COMPARISON / JUSTIFICATION")
        p("=" * 170)
        p(
            "Primary reasoning quality uses: higher ENTAILED rate, lower "
            "GT_CONSISTENT_NOT_ENTAILED rate, and lower GT_INCONSISTENT rate."
        )
        p(
            "Trace support and Clue Coverage are descriptive diagnostics only "
            "and are NOT used as primary correctness penalties."
        )
        p()

        for label in DIFFICULTY_ORDER:
            oa, na = agg_by_label[label]

            for line in build_verdict(oa, na, label):
                p(line)

            p(
                f"  PAcc: Original={fmt_pct(oa['puzzle_accuracy'])}, "
                f"NSS={fmt_pct(na['puzzle_accuracy'])}; "
                f"Entailed: {fmt_pct(oa['entailed_rate'])} vs "
                f"{fmt_pct(na['entailed_rate'])}; "
                f"GT-ok/NotEnt: "
                f"{fmt_pct(oa['gt_consistent_not_entailed_rate'])} vs "
                f"{fmt_pct(na['gt_consistent_not_entailed_rate'])}; "
                f"GT-error: {fmt_pct(oa['gt_inconsistent_rate'])} vs "
                f"{fmt_pct(na['gt_inconsistent_rate'])}; "
                f"TraceSup: {fmt_pct(oa['trace_support_rate'])} vs "
                f"{fmt_pct(na['trace_support_rate'])}; "
                f"ClueCov: {fmt_pct(oa['macro_clue_coverage'])} vs "
                f"{fmt_pct(na['macro_clue_coverage'])}."
            )
            p()

        p()
        p("=" * 170)
        p("ANALYSIS COMPLETENESS")
        p("=" * 170)

        for label in DIFFICULTY_ORDER:
            oa, na = agg_by_label[label]

            p(
                f"{label:<10} "
                f"Original analyzed {oa['n_analysis_ok']}/{oa['n_cases']} "
                f"cases; NSS analyzed {na['n_analysis_ok']}/{na['n_cases']} "
                f"cases. Unverifiable S rate: "
                f"Original={fmt_pct(oa['unverifiable_rate_all_steps'])}, "
                f"NSS={fmt_pct(na['unverifiable_rate_all_steps'])}."
            )


def write_case_jsonl(
    path: Path,
    pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for o, n in pairs:
            rec = {
                "id": o["id"],
                "size": o["size"],
                "difficulty": o["difficulty"],
                "original": o,
                "nss": n,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_step_details(
    path: Path,
    cases: Sequence[Dict[str, Any]],
    system_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as out:
        out.write(FORMAL_DEFINITIONS)
        out.write("\n\n")

        for c in cases:
            out.write("=" * 130 + "\n")
            out.write(
                f"ID: {c['id']}\n"
                f"SIZE: {c['size']}  "
                f"DIFFICULTY: {c['difficulty']}  "
                f"SYSTEM: {system_name}\n"
            )
            out.write(
                f"Puzzle Accuracy: {c['puzzle_accuracy']}  "
                f"Clue Coverage: {c['clue_coverage']:.4f}\n"
            )

            if not c.get("analysis_ok"):
                out.write(
                    f"ANALYSIS ERROR: {c.get('analysis_error')}\n\n"
                )
                continue

            for s in c.get("step_details", []):
                out.write(
                    f"{s['key']} | {s['classification']} | "
                    f"refs_now={s['clue_refs_current']} | "
                    f"active_refs={s['clue_refs_active']}\n"
                )

                out.write(f"  NL: {s.get('nl', '')}\n")
                out.write(f"  S : {s.get('expr', '')}\n")

                out.write(
                    f"  GT-consistent: {s.get('gt_consistent')} | "
                    f"GLOBAL entails: {s.get('global_entails')} "
                    f"({s.get('global_entailment_status')})\n"
                )

                if s.get("classification") == "ENTAILED":
                    out.write(
                        f"  Explicit trace support: "
                        f"{s.get('trace_supported')} "
                        f"({s.get('trace_support_status')})\n"
                    )

            out.write("\n\n")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
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
        default="./Outputs/Entailment_Analysis_v2",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=2000,
        help="Z3 timeout per SAT/entailment query",
    )
    args = parser.parse_args()

    # Print the exact formal definitions used by the analysis.
    print_formal_definitions()

    original_path = Path(args.original)
    nss_path = Path(args.nss)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_records = read_records(original_path)
    nss_records = read_records(nss_path)
    aligned = align_by_id(original_records, nss_records)

    original_cases: List[Dict[str, Any]] = []
    nss_cases: List[Dict[str, Any]] = []
    paired_cases: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    total = len(aligned)
    for i, (oraw, nraw) in enumerate(aligned, 1):
        if i == 1 or i % 50 == 0 or i == total:
            print(f"Analyzing {i}/{total} ...")
        oa = analyze_case_reasoning(oraw, timeout_ms=args.timeout_ms)
        na = analyze_case_reasoning(nraw, timeout_ms=args.timeout_ms)

        if oa["size"] != na["size"]:
            raise ValueError(f"Size mismatch for {oa['id']}: {oa['size']} vs {na['size']}")
        if oa["difficulty"] != na["difficulty"]:
            raise ValueError(f"Difficulty mismatch for {oa['id']}")

        original_cases.append(oa)
        nss_cases.append(na)
        paired_cases.append((oa, na))

    summary_path = out_dir / "entailment_clue_coverage_summary.txt"
    write_summary(summary_path, original_cases, nss_cases)
    write_case_jsonl(out_dir / "case_level_entailment_analysis.jsonl", paired_cases)
    write_step_details(out_dir / "original_step_entailment_details.txt", original_cases, "Original")
    write_step_details(out_dir / "nss_step_entailment_details.txt", nss_cases, "NSS")

    print("\nOutputs written to:")
    for p in (
        summary_path,
        out_dir / "case_level_entailment_analysis.jsonl",
        out_dir / "original_step_entailment_details.txt",
        out_dir / "nss_step_entailment_details.txt",
    ):
        print("  ", p.resolve())


if __name__ == "__main__":
    main()
