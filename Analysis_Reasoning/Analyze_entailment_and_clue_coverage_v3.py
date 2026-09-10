#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two ZebraLogic prompt logs using STRONG IMPLICATION for step-level
reasoning analysis.

PRIMARY METHODOLOGY
===================

Strong implication follows the requested definition:

    A =>_Phi B

iff BOTH conditions hold:

    SAT( Phi U {A, B} )
    UNSAT( Phi U {A, NOT B} )

The first SAT condition prevents vacuous implication from an inconsistent
precursor/background theory.

For each puzzle/system:

1. FILTER THE CASE USING THE SYNTACTIC CLUES

   Build the Zebra background theory Phi from the base constraints:
       - every attribute value is assigned to one house
       - house indices are in 1..N
       - values within each attribute are pairwise distinct

   Parse ALL syntactic clues C.

   Keep the case for reasoning analysis only if the Z3 clue theory determines
   exactly the supplied GT assignment:

       SAT( Phi U C U GT )
       UNSAT( Phi U C U {NOT GT} )

   Thus the formal clue solution is the GT solution; cases with clue parse
   failures, inconsistency, solver UNKNOWN, or an alternative solution are
   filtered out.

2. CHECK S_i BY STRONG IMPLICATION

   Let E_<i be the set of earlier S_j (j < i) that were themselves accepted as
   strongly implied.

   Define the precursor for S_i as:

       A_i = C AND E_<i

   and the background theory as:

       Phi = Zebra base constraints.

   For S_1:

       A_1 = C

   For later steps:

       A_i = C AND S_1^E AND ... AND S_{i-1}^E

   Then S_i is ENTAILED iff:

       A_i =>_Phi S_i

   i.e. iff:

       SAT( Phi U {A_i, S_i} )
       UNSAT( Phi U {A_i, NOT S_i} )

   Only earlier steps that passed this strong-implication test are admitted into
   E_<i. GT-wrong, unsupported, or unverifiable steps never contaminate the
   accepted reasoning prefix.

3. STEP LABELS

   ENTAILED
       S_i is GT-consistent and A_i =>_Phi S_i.

   GT_CONSISTENT_NOT_STRONGLY_IMPLIED
       S_i is true under GT, but A_i =>_Phi S_i does not hold.
       This is NOT called "premature". It is simply a GT-correct step that the
       formal precursor does not strongly imply.

   GT_INCONSISTENT
       S_i is false under GT.

   UNVERIFIABLE
       Parsing/token/solver/filter-related uncertainty prevents a reliable
       classification.

4. CLUE COVERAGE

   Clue Coverage is kept as a separate descriptive metric:

       unique clue IDs explicitly referenced in NL reasoning
       -----------------------------------------------------
                   unique syntactic clue IDs

   Clue coverage does NOT affect entailment, because all syntactic clues are in
   A_i from the first S-step onward.

IMPORTANT COMPARISON RULE
=========================

Reasoning metrics are reported:
  (a) per system on that system's filter-passing cases, and
  (b) on the COMMON FILTER-PASS subset where both Original and NSS clue theories
      solve to GT. The common subset is the preferred apples-to-apples prompt
      comparison because it avoids comparing different filtered puzzle sets.

Requirements:
    pip install z3-solver

Typical usage:
    python Analyze_strong_implication_and_clue_coverage_v3.py

or:
    python Analyze_strong_implication_and_clue_coverage_v3.py \\
      --original ./Input_Logs/gpt51_outputs_test_700_temp_0.jsonl \\
      --nss ./Input_Logs/gpt51_outputs_test_700_mlxl_nss_temp_0.jsonl \\
      --output-dir ./Outputs/Strong_Implication_Analysis
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
# Strong implication + GT-clue filter
# =============================================================================


def solver_status(constraints: Sequence[Any], timeout_ms: int) -> Any:
    s = z3.Solver()
    s.set("timeout", int(timeout_ms))
    s.add(*constraints)
    return s.check()


def _status_name(status: Any) -> str:
    if status == z3.sat:
        return "sat"
    if status == z3.unsat:
        return "unsat"
    if status == z3.unknown:
        return "unknown"
    return str(status)


def strong_implication(
    phi_constraints: Sequence[Any],
    precursor_constraints: Sequence[Any],
    conclusion: Any,
    timeout_ms: int,
) -> Tuple[Optional[bool], Dict[str, str]]:
    """
    Check A =>_Phi B using the strong implication definition.

    Here:
        Phi = conjunction(phi_constraints)
        A   = conjunction(precursor_constraints)
        B   = conclusion

    Strong implication holds iff:
        Phi U {A, B}      is SAT
        Phi U {A, NOT B}  is UNSAT

    Returns:
        (True, details)   if strong implication holds
        (False, details)  if both checks are decisive and implication fails
        (None, details)   if either check is UNKNOWN
    """
    s_pos = z3.Solver()
    s_pos.set("timeout", int(timeout_ms))
    s_pos.add(*phi_constraints)
    s_pos.add(*precursor_constraints)
    s_pos.add(conclusion)
    pos = s_pos.check()

    s_neg = z3.Solver()
    s_neg.set("timeout", int(timeout_ms))
    s_neg.add(*phi_constraints)
    s_neg.add(*precursor_constraints)
    s_neg.add(z3.Not(conclusion))
    neg = s_neg.check()

    details = {
        "phi_A_B": _status_name(pos),
        "phi_A_notB": _status_name(neg),
    }

    if pos == z3.unknown or neg == z3.unknown:
        return None, details

    if pos == z3.sat and neg == z3.unsat:
        return True, details

    return False, details


def build_gt_constraints(
    gt: Dict[str, Any],
    attribute_values: Dict[str, List[Any]],
    alias_map: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, int]]:
    """Build a complete Z3 conjunction corresponding to the supplied GT grid."""
    env, ambiguous_gt = gt_entity_to_house(gt)
    if ambiguous_gt:
        raise ValueError(
            "GT contains ambiguous entity values across houses: "
            + ", ".join(sorted(ambiguous_gt))
        )

    constraints: List[Any] = []
    expected = 0
    gt_positions: Dict[str, int] = {}

    for _attr, values in attribute_values.items():
        if not isinstance(values, list):
            raise ValueError("attribute_values contains a non-list domain")
        for value in values:
            expected += 1
            key = canonicalize(value)
            if key not in env:
                raise ValueError(f"Entity {value!r} from attribute_values is missing in GT")
            zv = lookup_z3_name(str(value), alias_map)
            house = int(env[key])
            constraints.append(zv == house)
            gt_positions[key] = house

    if len(constraints) != expected:
        raise ValueError("Could not build a complete GT assignment")

    return constraints, gt_positions


def check_clue_theory_equals_gt(
    *,
    base_axioms: Sequence[Any],
    clue_constraints: Sequence[Any],
    gt_constraints: Sequence[Any],
    clue_errors: Dict[int, List[str]],
    unidentified_clues: Sequence[str],
    timeout_ms: int,
) -> Dict[str, Any]:
    """
    Filter required by the analysis.

    PASS iff the complete parsed clue theory has exactly the supplied GT
    assignment over the Zebra variables:

        SAT(Phi + C + GT)
        UNSAT(Phi + C + NOT(GT))

    This is stronger and safer than comparing one arbitrary Z3 model to GT.
    """
    out = {
        "pass": False,
        "reason": None,
        "clue_theory_status": None,
        "clue_plus_gt_status": None,
        "alternative_to_gt_status": None,
    }

    if clue_errors:
        out["reason"] = "CLUE_PARSE_ERROR"
        return out
    if unidentified_clues:
        out["reason"] = "UNIDENTIFIED_CLUE"
        return out
    if not clue_constraints:
        out["reason"] = "NO_PARSED_CLUES"
        return out
    if not gt_constraints:
        out["reason"] = "NO_GT_CONSTRAINTS"
        return out

    theory = list(base_axioms) + list(clue_constraints)
    st = solver_status(theory, timeout_ms)
    out["clue_theory_status"] = _status_name(st)
    if st == z3.unknown:
        out["reason"] = "CLUE_THEORY_UNKNOWN"
        return out
    if st == z3.unsat:
        out["reason"] = "CLUE_THEORY_UNSAT"
        return out

    with_gt = solver_status(theory + list(gt_constraints), timeout_ms)
    out["clue_plus_gt_status"] = _status_name(with_gt)
    if with_gt == z3.unknown:
        out["reason"] = "GT_MATCH_UNKNOWN"
        return out
    if with_gt == z3.unsat:
        out["reason"] = "GT_NOT_SOLUTION_OF_CLUES"
        return out

    gt_formula = z3.And(*gt_constraints) if len(gt_constraints) > 1 else gt_constraints[0]
    alt = solver_status(theory + [z3.Not(gt_formula)], timeout_ms)
    out["alternative_to_gt_status"] = _status_name(alt)

    if alt == z3.unknown:
        out["reason"] = "GT_UNIQUENESS_UNKNOWN"
        return out
    if alt == z3.sat:
        out["reason"] = "ALTERNATIVE_SOLUTION_EXISTS"
        return out

    out["pass"] = True
    out["reason"] = "PASS_Z3_SOLUTION_EQUALS_GT"
    return out


# =============================================================================
# Per-case reasoning analysis
# =============================================================================


def analyze_case_reasoning(
    record: Dict[str, Any],
    *,
    timeout_ms: int,
) -> Dict[str, Any]:
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

        # Mandatory clue/GT filter.
        "filter_pass": False,
        "filter_reason": None,
        "filter_details": {},

        # S-step metrics, populated only on filter-pass cases.
        "n_s_steps": 0,
        "n_entailed": 0,
        "n_gt_consistent_not_strongly_implied": 0,
        "n_gt_inconsistent": 0,
        "n_unverifiable": 0,

        # Explicit clue-reference coverage; independent of entailment.
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

        all_clue_constraints = [
            c for cid in sorted(clue_groups) for c in clue_groups[cid]
        ]

        gt_constraints, _gt_positions = build_gt_constraints(gt, attrs, alias_map)

        filt = check_clue_theory_equals_gt(
            base_axioms=base_axioms,
            clue_constraints=all_clue_constraints,
            gt_constraints=gt_constraints,
            clue_errors=clue_errors,
            unidentified_clues=unidentified,
            timeout_ms=timeout_ms,
        )

        result["filter_pass"] = bool(filt["pass"])
        result["filter_reason"] = filt["reason"]
        result["filter_details"] = filt
        result["clue_parse_errors"] = clue_errors
        result["unidentified_clues"] = unidentified
        result["ambiguous_dsl_aliases"] = sorted(ambiguous_aliases)
        result["n_clues_total"] = len(clue_ids)

        pairs = extract_s_nl_pairs(reasoning)

        # Clue coverage is descriptive and can be computed even if filter fails.
        covered_clues: Set[int] = set()
        unknown_refs: Set[int] = set()
        steps_with_ref = 0
        for step in pairs:
            refs_raw = extract_clue_refs(step.get("nl", ""), valid_ids=None)
            refs_known = refs_raw & clue_ids
            refs_unknown = refs_raw - clue_ids
            if refs_known:
                steps_with_ref += 1
            covered_clues |= refs_known
            unknown_refs |= refs_unknown

        result["n_clues_covered"] = len(covered_clues)
        result["clue_coverage"] = (
            len(covered_clues) / len(clue_ids) if clue_ids else 0.0
        )
        result["n_steps_with_explicit_clue_ref"] = steps_with_ref
        result["unknown_clue_refs"] = sorted(unknown_refs)

        # The requested filter is a hard gate for S-step implication analysis.
        if not result["filter_pass"]:
            result["analysis_ok"] = True
            result["n_s_steps"] = len(pairs)
            return result

        env, ambiguous_gt = gt_entity_to_house(gt)
        result["ambiguous_gt_tokens"] = sorted(ambiguous_gt)

        accepted_implied_s: List[Any] = []
        accepted_implied_keys: List[str] = []
        step_details: List[Dict[str, Any]] = []
        counts = defaultdict(int)

        for pos, step in enumerate(pairs, 1):
            gt_value, gt_error = eval_expr_gt(step["expr"], env)

            detail = dict(step)
            detail.update({
                "position": pos,
                "gt_consistent": gt_value,
                "gt_eval_error": gt_error,
                "classification": None,
                "precursor": {
                    "all_syntactic_clues": True,
                    "previous_strongly_implied_steps": list(accepted_implied_keys),
                },
                "strong_implication": None,
                "strong_implication_checks": None,
            })

            # GT false = genuine reasoning error. It is never admitted to prefix.
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

            try:
                zstep = parse_z3_expr(step["expr"], alias_map)
            except Exception as exc:
                detail["classification"] = "UNVERIFIABLE"
                detail["strong_implication_checks"] = {
                    "parse_error": f"{type(exc).__name__}: {exc}"
                }
                counts["unverifiable"] += 1
                step_details.append(detail)
                continue

            # Strong-implication notation:
            #   Phi = base_axioms
            #   A_i = all clues AND all earlier strongly implied S_j
            #   B_i = current S_i
            precursor_constraints = list(all_clue_constraints) + list(accepted_implied_s)

            implied, checks = strong_implication(
                phi_constraints=base_axioms,
                precursor_constraints=precursor_constraints,
                conclusion=zstep,
                timeout_ms=timeout_ms,
            )

            detail["strong_implication"] = implied
            detail["strong_implication_checks"] = checks

            if implied is True:
                detail["classification"] = "ENTAILED"
                counts["entailed"] += 1
                accepted_implied_s.append(zstep)
                accepted_implied_keys.append(step["key"])

            elif implied is False:
                detail["classification"] = "GT_CONSISTENT_NOT_STRONGLY_IMPLIED"
                counts["gt_consistent_not_strongly_implied"] += 1

            else:
                detail["classification"] = "UNVERIFIABLE"
                counts["unverifiable"] += 1

            step_details.append(detail)

        result.update({
            "analysis_ok": True,
            "n_s_steps": len(pairs),
            "n_entailed": counts["entailed"],
            "n_gt_consistent_not_strongly_implied":
                counts["gt_consistent_not_strongly_implied"],
            "n_gt_inconsistent": counts["gt_inconsistent"],
            "n_unverifiable": counts["unverifiable"],
            "step_details": step_details,
        })

    except Exception as exc:
        result["analysis_error"] = f"{type(exc).__name__}: {exc}"

    return result


# =============================================================================
# Log I/O, aggregation, and paired comparison
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

    return [(r, mb[str(r.get("id"))]) for r in a]


def aggregate(
    cases: Sequence[Dict[str, Any]],
    difficulty: Optional[str] = None,
    *,
    require_filter_pass: bool = False,
) -> Dict[str, Any]:
    rows = [
        c for c in cases
        if (difficulty is None or c.get("difficulty") == difficulty)
    ]

    filtered = [c for c in rows if c.get("filter_pass")]
    reason_rows = filtered if require_filter_pass else [c for c in rows if c.get("filter_pass")]

    sums = defaultdict(float)
    clue_cov_sum = 0.0
    clue_num = 0
    clue_den = 0

    for c in reason_rows:
        for k in (
            "n_s_steps",
            "n_entailed",
            "n_gt_consistent_not_strongly_implied",
            "n_gt_inconsistent",
            "n_unverifiable",
        ):
            sums[k] += float(c.get(k, 0))
        clue_cov_sum += float(c.get("clue_coverage", 0.0))
        clue_num += int(c.get("n_clues_covered", 0))
        clue_den += int(c.get("n_clues_total", 0))

    verifiable = (
        sums["n_entailed"]
        + sums["n_gt_consistent_not_strongly_implied"]
        + sums["n_gt_inconsistent"]
    )

    n_rows = len(rows)
    pacc_all = (
        sum(float(c.get("puzzle_accuracy", 0.0)) for c in rows) / n_rows
        if n_rows else 0.0
    )
    cacc_all = (
        sum(float(c.get("cell_accuracy", 0.0)) for c in rows) / n_rows
        if n_rows else 0.0
    )

    n_fr = len(reason_rows)
    pacc_filtered = (
        sum(float(c.get("puzzle_accuracy", 0.0)) for c in reason_rows) / n_fr
        if n_fr else 0.0
    )

    return {
        "n_cases": n_rows,
        "n_filter_pass": len(filtered),
        "filter_pass_rate": len(filtered) / n_rows if n_rows else 0.0,
        "puzzle_accuracy_all": pacc_all,
        "macro_cell_accuracy_all": cacc_all,
        "puzzle_accuracy_filtered": pacc_filtered,

        "n_s_steps": int(sums["n_s_steps"]),
        "n_verifiable_steps": int(verifiable),
        "n_entailed": int(sums["n_entailed"]),
        "n_gt_consistent_not_strongly_implied":
            int(sums["n_gt_consistent_not_strongly_implied"]),
        "n_gt_inconsistent": int(sums["n_gt_inconsistent"]),
        "n_unverifiable": int(sums["n_unverifiable"]),

        "entailed_rate": sums["n_entailed"] / verifiable if verifiable else 0.0,
        "gt_consistent_not_strongly_implied_rate":
            sums["n_gt_consistent_not_strongly_implied"] / verifiable
            if verifiable else 0.0,
        "gt_inconsistent_rate":
            sums["n_gt_inconsistent"] / verifiable if verifiable else 0.0,
        "unverifiable_rate_all_steps":
            sums["n_unverifiable"] / sums["n_s_steps"] if sums["n_s_steps"] else 0.0,

        "macro_clue_coverage": clue_cov_sum / n_fr if n_fr else 0.0,
        "micro_clue_coverage": clue_num / clue_den if clue_den else 0.0,
    }


def aggregate_common_pairs(
    paired_cases: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    difficulty: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
    selected = [
        (o, n) for o, n in paired_cases
        if (difficulty is None or o.get("difficulty") == difficulty)
        and o.get("filter_pass") and n.get("filter_pass")
    ]
    oc = [o for o, _ in selected]
    nc = [n for _, n in selected]
    return aggregate(oc, None), aggregate(nc, None), len(selected)


def filter_reason_counts(
    cases: Sequence[Dict[str, Any]],
    difficulty: Optional[str] = None,
) -> Dict[str, int]:
    out = defaultdict(int)
    for c in cases:
        if difficulty is not None and c.get("difficulty") != difficulty:
            continue
        out[str(c.get("filter_reason"))] += 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def winner_higher(a: float, b: float, eps: float = 1e-12) -> str:
    if abs(a - b) <= eps:
        return "Tie"
    return "NSS" if b > a else "Original"


def winner_lower(a: float, b: float, eps: float = 1e-12) -> str:
    if abs(a - b) <= eps:
        return "Tie"
    return "NSS" if b < a else "Original"


FORMAL_DEFINITIONS = r"""
================================================================================
FORMAL DEFINITIONS USED BY THIS SCRIPT
================================================================================

Let:

  Phi      = ZebraLogic background/base theory (domain + uniqueness constraints)

  C        = conjunction of ALL parsed syntactic clues

  S_i      = the i-th symbolic reasoning step

  E_<i     = earlier S_j, j < i, that were themselves accepted as ENTAILED
             by the strong-implication test

  A_i      = precursor formula for S_i:

                 A_i = C AND E_<i

             Hence A_1 = C.


1. STRONG IMPLICATION
---------------------

Let Phi be a background theory and A and B be formulas.
We say that A strongly implies B under Phi, written

                 A =>_Phi B

iff BOTH conditions hold:

  (i)   Phi U {A, B}      is SATISFIABLE, and
  (ii)  Phi U {A, NOT B}  is UNSATISFIABLE.

The SAT condition (i) excludes vacuous implication caused by an inconsistent
Phi U {A}.

For each reasoning step S_i, this script checks:

                 A_i =>_Phi S_i

Operationally with Z3:

  Check 1:  SAT( Phi + A_i + S_i )
  Check 2:  UNSAT( Phi + A_i + NOT(S_i) )

Only if BOTH checks hold is S_i classified ENTAILED.


2. CLUE / GT FILTER
-------------------

Before any S-step analysis, the syntactic clue theory must determine the
provided GT assignment exactly.

Let GT be the conjunction of all entity-to-house assignments in the ground
truth. The case passes the filter only if:

  SAT( Phi + C + GT )
  UNSAT( Phi + C + NOT(GT) )

Thus, among analyzed cases, the Z3 solution induced by the syntactic clues is
exactly GT. Cases with clue parse failure, inconsistent clues, solver UNKNOWN,
or alternative solutions are excluded from S-step implication analysis.


3. ENTAILED
-----------

S_i is ENTAILED iff:

  (a) S_i is true under GT, and
  (b) A_i =>_Phi S_i.

Only ENTAILED earlier S_j are added to E_<i for later steps.


4. GT_CONSISTENT_NOT_STRONGLY_IMPLIED
-------------------------------------

S_i is GT-consistent but not strongly implied iff:

  (a) S_i is true under GT, but
  (b) A_i =>_Phi S_i does NOT hold.

This category is NOT called premature. A GT-correct deduction is treated as
good when it is strongly implied by the clue/prefix state. If it is not
strongly implied, the script reports that fact without assuming that the model
was necessarily "guessing".


5. GT_INCONSISTENT
------------------

S_i is GT_INCONSISTENT iff it is false under the supplied GT assignment.
It is treated as a genuine reasoning error and is never added to E_<i.


6. UNVERIFIABLE
---------------

S_i is UNVERIFIABLE when parsing, token resolution, or a Z3 UNKNOWN result
prevents a reliable implication classification. It is reported separately and
is not automatically counted as a reasoning error.


7. CLUE COVERAGE
----------------

Clue Coverage is descriptive only:

     number of unique clue IDs explicitly referenced in NL reasoning
     ----------------------------------------------------------------
                 number of unique syntactic clue IDs

All syntactic clues C are used by the strong-implication check regardless of
whether the model explicitly cites them in NL. Therefore Clue Coverage does NOT
affect ENTAILED / GT-wrong classification.

================================================================================
"""


def print_formal_definitions() -> None:
    print(FORMAL_DEFINITIONS)


def write_summary(
    path: Path,
    original_cases: Sequence[Dict[str, Any]],
    nss_cases: Sequence[Dict[str, Any]],
    paired_cases: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as out:
        def p(s: str = ""):
            print(s)
            out.write(s + "\n")

        p(FORMAL_DEFINITIONS.rstrip())
        p()
        p("=" * 160)
        p("A. CLUE-THEORY FILTER: DOES Z3(SYNTACTIC CLUES) DETERMINE GT?")
        p("=" * 160)
        p(
            f"{'Difficulty':<10} {'System':<10} {'N':>6} {'FilterPass':>11} "
            f"{'PassRate':>10} {'PAcc(all)':>11} {'CAcc(all)':>11}"
        )
        p("-" * 76)

        agg_all = {}
        for label in DIFFICULTY_ORDER:
            diff = None if label == "Overall" else label
            oa = aggregate(original_cases, diff)
            na = aggregate(nss_cases, diff)
            agg_all[label] = (oa, na)
            for name, a in (("Original", oa), ("NSS", na)):
                p(
                    f"{label:<10} {name:<10} {a['n_cases']:>6} "
                    f"{a['n_filter_pass']:>11} {fmt_pct(a['filter_pass_rate']):>10} "
                    f"{fmt_pct(a['puzzle_accuracy_all']):>11} "
                    f"{fmt_pct(a['macro_cell_accuracy_all']):>11}"
                )
            p()

        p()
        p("Filter-failure reasons (Overall):")
        p(f"  Original: {filter_reason_counts(original_cases)}")
        p(f"  NSS     : {filter_reason_counts(nss_cases)}")

        p()
        p("=" * 160)
        p("B. STRONG-IMPLICATION REASONING METRICS ON EACH SYSTEM'S FILTER-PASS CASES")
        p("=" * 160)
        header = (
            f"{'Difficulty':<10} {'System':<10} {'PassN':>6} {'S':>7} "
            f"{'Entailed':>10} {'GT-ok/NotSI':>12} {'GT-wrong':>10} {'Unverif':>9} "
            f"{'Ent%':>8} {'NotSI%':>8} {'GTerr%':>8} {'ClueCov':>9}"
        )
        p(header)
        p("-" * len(header))
        for label in DIFFICULTY_ORDER:
            diff = None if label == "Overall" else label
            oa = aggregate(original_cases, diff, require_filter_pass=True)
            na = aggregate(nss_cases, diff, require_filter_pass=True)
            for name, a in (("Original", oa), ("NSS", na)):
                p(
                    f"{label:<10} {name:<10} {a['n_filter_pass']:>6} "
                    f"{a['n_s_steps']:>7} {a['n_entailed']:>10} "
                    f"{a['n_gt_consistent_not_strongly_implied']:>12} "
                    f"{a['n_gt_inconsistent']:>10} {a['n_unverifiable']:>9} "
                    f"{fmt_pct(a['entailed_rate']):>8} "
                    f"{fmt_pct(a['gt_consistent_not_strongly_implied_rate']):>8} "
                    f"{fmt_pct(a['gt_inconsistent_rate']):>8} "
                    f"{fmt_pct(a['macro_clue_coverage']):>9}"
                )
            p()

        p()
        p("=" * 160)
        p("C. APPLES-TO-APPLES: COMMON FILTER-PASS CASES")
        p("=" * 160)
        p("Only puzzles for which BOTH prompt outputs have Z3(clues) == GT are compared below.")
        p()
        header2 = (
            f"{'Difficulty':<10} {'System':<10} {'CommonN':>8} {'PAcc':>9} "
            f"{'S':>7} {'Ent%':>9} {'NotSI%':>9} {'GTerr%':>9} "
            f"{'Unverif%':>10} {'ClueCov':>9}"
        )
        p(header2)
        p("-" * len(header2))

        common_aggs = {}
        for label in DIFFICULTY_ORDER:
            diff = None if label == "Overall" else label
            oa, na, ncommon = aggregate_common_pairs(paired_cases, diff)
            common_aggs[label] = (oa, na, ncommon)
            for name, a in (("Original", oa), ("NSS", na)):
                p(
                    f"{label:<10} {name:<10} {ncommon:>8} "
                    f"{fmt_pct(a['puzzle_accuracy_all']):>9} "
                    f"{a['n_s_steps']:>7} {fmt_pct(a['entailed_rate']):>9} "
                    f"{fmt_pct(a['gt_consistent_not_strongly_implied_rate']):>9} "
                    f"{fmt_pct(a['gt_inconsistent_rate']):>9} "
                    f"{fmt_pct(a['unverifiable_rate_all_steps']):>10} "
                    f"{fmt_pct(a['macro_clue_coverage']):>9}"
                )
            p()

        p()
        p("=" * 160)
        p("D. PROMPT COMPARISON / JUSTIFICATION")
        p("=" * 160)
        p("Interpretation hierarchy:")
        p("  1) Higher clue-filter pass rate = better formalization of the puzzle clues into a GT-equivalent Z3 theory.")
        p("  2) On COMMON filter-pass cases: higher ENTAILED rate and lower GT-wrong rate indicate stronger symbolic reasoning.")
        p("  3) Lower UNVERIFIABLE rate indicates better parser/DSL robustness.")
        p("  4) Puzzle Accuracy remains the task-level outcome; Clue Coverage is descriptive only.")
        p()

        for label in DIFFICULTY_ORDER:
            oa_all, na_all = agg_all[label]
            oa, na, ncommon = common_aggs[label]

            task_winner = winner_higher(
                oa_all["puzzle_accuracy_all"], na_all["puzzle_accuracy_all"]
            )
            filter_winner = winner_higher(
                oa_all["filter_pass_rate"], na_all["filter_pass_rate"]
            )
            ent_winner = winner_higher(oa["entailed_rate"], na["entailed_rate"])
            gterr_winner = winner_lower(
                oa["gt_inconsistent_rate"], na["gt_inconsistent_rate"]
            )
            unver_winner = winner_lower(
                oa["unverifiable_rate_all_steps"], na["unverifiable_rate_all_steps"]
            )

            p(
                f"{label}: task={task_winner}; clue-filter={filter_winner}; "
                f"commonN={ncommon}; entailed={ent_winner}; "
                f"GT-error={gterr_winner}; unverifiable={unver_winner}."
            )
            p(
                f"  PAcc(all): Original={fmt_pct(oa_all['puzzle_accuracy_all'])}, "
                f"NSS={fmt_pct(na_all['puzzle_accuracy_all'])}; "
                f"FilterPass: {fmt_pct(oa_all['filter_pass_rate'])} vs "
                f"{fmt_pct(na_all['filter_pass_rate'])}."
            )
            if ncommon:
                p(
                    f"  Common-pass reasoning: Entailed={fmt_pct(oa['entailed_rate'])} vs "
                    f"{fmt_pct(na['entailed_rate'])}; "
                    f"GT-ok/NotSI={fmt_pct(oa['gt_consistent_not_strongly_implied_rate'])} vs "
                    f"{fmt_pct(na['gt_consistent_not_strongly_implied_rate'])}; "
                    f"GT-error={fmt_pct(oa['gt_inconsistent_rate'])} vs "
                    f"{fmt_pct(na['gt_inconsistent_rate'])}; "
                    f"Unverif={fmt_pct(oa['unverifiable_rate_all_steps'])} vs "
                    f"{fmt_pct(na['unverifiable_rate_all_steps'])}."
                )
            p()

        p("NOTE: Because filter-pass cases are constrained so clues uniquely determine GT, a GT-consistent formula over the modeled Zebra variables should normally be strongly implied. Therefore GT-ok/NotSI should be near zero; non-zero values are useful diagnostics for parser/modeling mismatches rather than a 'premature deduction' penalty.")


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
                f"ID: {c['id']}\nSIZE: {c['size']}  DIFFICULTY: {c['difficulty']}  SYSTEM: {system_name}\n"
            )
            out.write(
                f"Puzzle Accuracy: {c['puzzle_accuracy']}  "
                f"Filter pass: {c['filter_pass']} ({c['filter_reason']})  "
                f"Clue Coverage: {c['clue_coverage']:.4f}\n"
            )
            if c.get("analysis_error"):
                out.write(f"ANALYSIS ERROR: {c.get('analysis_error')}\n\n")
                continue
            if not c.get("filter_pass"):
                out.write(f"FILTER DETAILS: {json.dumps(c.get('filter_details', {}), ensure_ascii=False)}\n\n")
                continue
            for s in c.get("step_details", []):
                out.write(
                    f"{s['key']} | {s['classification']} | "
                    f"precursor_previous_entailed={s['precursor']['previous_strongly_implied_steps']}\n"
                )
                out.write(f"  NL: {s.get('nl', '')}\n")
                out.write(f"  S : {s.get('expr', '')}\n")
                out.write(
                    f"  GT-consistent: {s.get('gt_consistent')} | "
                    f"strong implication: {s.get('strong_implication')} | "
                    f"checks={s.get('strong_implication_checks')}\n"
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
        default="./Outputs/Strong_Implication_Analysis",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=2000,
        help="Z3 timeout per SAT/strong-implication query",
    )
    args = parser.parse_args()

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
            raise ValueError(
                f"Size mismatch for {oa['id']}: {oa['size']} vs {na['size']}"
            )
        if oa["difficulty"] != na["difficulty"]:
            raise ValueError(f"Difficulty mismatch for {oa['id']}")

        original_cases.append(oa)
        nss_cases.append(na)
        paired_cases.append((oa, na))

    summary_path = out_dir / "strong_implication_clue_coverage_summary.txt"
    write_summary(summary_path, original_cases, nss_cases, paired_cases)
    write_case_jsonl(
        out_dir / "case_level_strong_implication_analysis.jsonl",
        paired_cases,
    )
    write_step_details(
        out_dir / "original_step_strong_implication_details.txt",
        original_cases,
        "Original",
    )
    write_step_details(
        out_dir / "nss_step_strong_implication_details.txt",
        nss_cases,
        "NSS",
    )

    print("\nOutputs written to:")
    for p in (
        summary_path,
        out_dir / "case_level_strong_implication_analysis.jsonl",
        out_dir / "original_step_strong_implication_details.txt",
        out_dir / "nss_step_strong_implication_details.txt",
    ):
        print("  ", p.resolve())


if __name__ == "__main__":
    main()
