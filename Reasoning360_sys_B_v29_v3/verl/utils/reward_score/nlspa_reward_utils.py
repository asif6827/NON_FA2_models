# -*- coding: utf-8 -*-
"""
nlspa_reward_utils.py

Utilities for Zebra reasoning in the new schema:

"reasoning": {
    "NL1": "...",
    "S1": "Arnold == 2.",
    "PA1": {"header": [...], "rows": [...]},
    "NL2": "...",
    "S2": "...",
    ...
}

Design goals
------------
1. Validate NL_i / S_i / PA_i emitted-key structure.
2. Convert S_i entries to the legacy "S1: ..." list expected by the
   existing Z3 reasoning validator.
3. Validate and score partial-answer grids (PA_i).
4. Compute S-step PRM ratios using the number of emitted S steps rather
   than house x attribute grid size.

No Z3 dependency is required in this module.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_NL_KEY_RE = re.compile(r"^NL(\d+)$")
_S_KEY_RE = re.compile(r"^S(\d+)$")
_PA_KEY_RE = re.compile(r"^PA(\d+)$")


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def _canon_token(x: Any) -> str:
    """
    Canonical form used only for grid comparison.

    Makes:
        "high school", "high_school", "High-School"
    comparable without changing the model payload itself.
    """
    s = str(x).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _normalize_grid(grid: Any) -> Optional[Dict[str, List[List[str]]]]:
    if not isinstance(grid, dict):
        return None
    header = grid.get("header")
    rows = grid.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list):
        return None

    out_rows: List[List[str]] = []
    for row in rows:
        if not isinstance(row, list):
            return None
        out_rows.append([str(v) for v in row])

    return {
        "header": [str(h) for h in header],
        "rows": out_rows,
    }


def extract_s_steps(reasoning: Any) -> List[Dict[str, Any]]:
    """Return S entries in emitted order."""
    if not isinstance(reasoning, dict):
        return []

    steps: List[Dict[str, Any]] = []
    for key, value in reasoning.items():
        m = _S_KEY_RE.fullmatch(str(key))
        if not m:
            continue
        steps.append({
            "sid": str(key),
            "k": int(m.group(1)),
            "expr": value,
        })
    return steps


def reasoning_to_legacy_s_lines(reasoning: Any) -> List[str]:
    """
    Adapter for the existing Z3 validator.

    New:
        "S3": "Eric == 3."

    Legacy:
        "S3: Eric == 3."
    """
    lines: List[str] = []
    for step in extract_s_steps(reasoning):
        expr = step["expr"]
        if not isinstance(expr, str):
            continue
        expr = expr.strip()
        if not expr:
            continue
        if not expr.endswith("."):
            expr += "."
        lines.append(f'{step["sid"]}: {expr}')
    return lines


def extract_pa_entries(reasoning: Any) -> List[Dict[str, Any]]:
    """Return PA entries and the latest preceding S index."""
    if not isinstance(reasoning, dict):
        return []

    out: List[Dict[str, Any]] = []
    last_s = 0
    for key, value in reasoning.items():
        sm = _S_KEY_RE.fullmatch(str(key))
        if sm:
            last_s = int(sm.group(1))
            continue

        pm = _PA_KEY_RE.fullmatch(str(key))
        if pm:
            out.append({
                "pid": str(key),
                "k": int(pm.group(1)),
                "after_s": last_s,
                "grid": value,
            })
    return out


def _strip_constraint_prefix(expr: str) -> str:
    s = str(expr).strip().rstrip(".")
    s = re.sub(r"^\s*[CS]\d+\s*:\s*", "", s, flags=re.IGNORECASE)
    return s.strip()


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
                return []
            buf.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if not part:
                return []
            args.append(part)
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        return []
    tail = "".join(buf).strip()
    if tail:
        args.append(tail)
    return args


def _extract_positive_equalities(expr: str) -> List[Tuple[str, str]]:
    """
    Extract only explicitly positive equality atoms from:
        A == 2
        A == B
        And(A == 2, B == C)

    Or/Not/!=/</> are intentionally NOT converted into PA support facts.
    """
    s = _strip_constraint_prefix(expr)

    m = re.fullmatch(r"And\((.*)\)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        out: List[Tuple[str, str]] = []
        for arg in _split_top_level_args(m.group(1)):
            out.extend(_extract_positive_equalities(arg))
        return out

    # Do not infer support from disjunctions or negations.
    if re.match(r"^(Or|Not)\s*\(", s, flags=re.IGNORECASE):
        return []

    m = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*|\d+)\s*==\s*"
        r"([A-Za-z_][A-Za-z0-9_]*|\d+)",
        s,
    )
    if not m:
        return []
    return [(m.group(1), m.group(2))]


class _EqState:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.house: Dict[str, int] = {}

    def _add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        self._add(x)
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        ha = self.house.get(ra)
        hb = self.house.get(rb)
        self.parent[rb] = ra
        if ha is None and hb is not None:
            self.house[ra] = hb
        elif ha is not None:
            self.house[ra] = ha
        self.house.pop(rb, None)

    def fix(self, token: str, h: int) -> None:
        r = self.find(token)
        existing = self.house.get(r)
        # Conflicts are left unresolved for support purposes rather than
        # silently overriding an earlier explicit state.
        if existing is None:
            self.house[r] = int(h)
        elif existing != int(h):
            self.house[r] = -1

    def supported_house(self, token: str) -> Optional[int]:
        if token not in self.parent:
            return None
        h = self.house.get(self.find(token))
        if h is None or h < 1:
            return None
        return int(h)


def _build_explicit_equality_state(
    syntactic_clues: List[str],
    s_lines: List[str],
) -> _EqState:
    state = _EqState()
    pairs: List[Tuple[str, str]] = []
    for expr in list(syntactic_clues or []) + list(s_lines or []):
        pairs.extend(_extract_positive_equalities(expr))

    # Union token-token equalities first.
    for a, b in pairs:
        if not a.isdigit() and not b.isdigit():
            state.union(a, b)

    # Then apply token-house equalities.
    for a, b in pairs:
        if a.isdigit() and not b.isdigit():
            state.fix(b, int(a))
        elif b.isdigit() and not a.isdigit():
            state.fix(a, int(b))

    return state


def pa_prefix_support(
    pa_grid: Any,
    *,
    syntactic_clues: List[str],
    s_lines_prefix: List[str],
) -> Dict[str, Any]:
    """
    Score whether resolved PA cells are explicitly supported at the checkpoint.

    Support closure includes only positive equality information from clues and
    preceding S steps. This avoids giving credit for filling the entire unique
    final solution early merely because a full Z3 solve could derive it.
    """
    norm = _normalize_grid(pa_grid)
    if norm is None:
        return {
            "resolved_cells": 0,
            "supported_cells": 0,
            "unsupported_cells": 0,
            "support_ratio": 0.0,
        }

    state = _build_explicit_equality_state(
        syntactic_clues=syntactic_clues,
        s_lines=s_lines_prefix,
    )

    resolved = supported = 0
    unsupported: List[Dict[str, Any]] = []

    header = norm["header"]
    for i, row in enumerate(norm["rows"]):
        if len(row) != len(header):
            continue
        try:
            h = int(str(row[0]).strip())
        except Exception:
            continue

        for j in range(1, len(header)):
            token = str(row[j]).strip()
            if token == "?":
                continue
            resolved += 1
            sh = state.supported_house(token)
            if sh == h:
                supported += 1
            else:
                unsupported.append({
                    "house": h,
                    "column": header[j],
                    "value": token,
                    "supported_house": sh,
                })

    return {
        "resolved_cells": resolved,
        "supported_cells": supported,
        "unsupported_cells": len(unsupported),
        "support_ratio": (supported / resolved) if resolved else 1.0,
        "unsupported": unsupported,
    }


def validate_partial_grid(
    grid: Any,
    *,
    n_houses: int,
    expected_header: List[str],
    attribute_values: Dict[str, List[Any]],
) -> Dict[str, Any]:
    """
    Structural PA validation.

    Unknown cells may be exactly "?".
    House cells may not be unknown.
    """
    errors: List[str] = []
    norm = _normalize_grid(grid)

    if norm is None:
        return {
            "ok": False,
            "errors": ["PA must be an object with header:list and rows:list."],
            "resolved_cells": 0,
            "unknown_cells": 0,
            "total_attribute_cells": max(int(n_houses), 0) * max(len(expected_header) - 1, 0),
        }

    header = norm["header"]
    rows = norm["rows"]

    if header != [str(h) for h in expected_header]:
        errors.append("PA header must exactly match solution_header.")

    if len(rows) != int(n_houses):
        errors.append(
            f"PA must contain exactly {n_houses} rows; found {len(rows)}."
        )

    header_len = len(expected_header)
    expected_domains = {
        str(col): {_canon_token(v) for v in values}
        for col, values in (attribute_values or {}).items()
    }

    resolved = 0
    unknown = 0

    # Uniqueness per attribute column.
    seen_by_col: Dict[int, set] = {
        j: set() for j in range(1, header_len)
    }

    for i, row in enumerate(rows):
        if len(row) != header_len:
            errors.append(
                f"PA row {i + 1} has {len(row)} cells; expected {header_len}."
            )
            continue

        expected_house = str(i + 1)
        if str(row[0]).strip() != expected_house:
            errors.append(
                f"PA row {i + 1} House must be {expected_house!r}; "
                f"found {row[0]!r}."
            )

        for j in range(1, header_len):
            value = str(row[j]).strip()
            col = str(expected_header[j])

            if value == "?":
                unknown += 1
                continue

            if value in {"", "None", "null"}:
                errors.append(
                    f"PA row {i + 1}, column {col}: unresolved cells must be '?'."
                )
                continue

            resolved += 1
            canon = _canon_token(value)
            domain = expected_domains.get(col)

            if domain is None:
                errors.append(
                    f"PA column {col!r} is not present in attribute_values."
                )
            elif canon not in domain:
                errors.append(
                    f"PA value {value!r} is outside the domain for {col!r}."
                )

            if canon in seen_by_col[j]:
                errors.append(
                    f"PA value {value!r} appears more than once in column {col!r}."
                )
            seen_by_col[j].add(canon)

    return {
        "ok": not errors,
        "errors": errors,
        "resolved_cells": int(resolved),
        "unknown_cells": int(unknown),
        "total_attribute_cells": int(max(n_houses, 0) * max(header_len - 1, 0)),
    }


def validate_reasoning_schema(
    reasoning: Any,
    *,
    n_houses: int,
    expected_header: List[str],
    attribute_values: Dict[str, List[Any]],
    require_terminal_period: bool = True,
) -> Dict[str, Any]:
    """
    Validate emitted key order:

        NL1, S1, [PA1], NL2, S2, [PA2], ...

    A PA must immediately follow a completed S step.
    """
    errors: List[Dict[str, Any]] = []

    if not isinstance(reasoning, dict) or not reasoning:
        return {
            "ok": False,
            "errors": [{"code": "REASONING_NOT_OBJECT",
                        "message": "reasoning must be a non-empty JSON object."}],
            "n_nl": 0,
            "n_s": 0,
            "n_pa": 0,
            "s_lines": [],
            "pa_entries": [],
        }

    keys = list(reasoning.keys())
    expected_pair = 1
    expected_pa = 1
    expecting = "NL"
    previous_kind: Optional[str] = None

    n_nl = n_s = n_pa = 0

    for pos, key in enumerate(keys):
        key = str(key)
        value = reasoning[key]

        nlm = _NL_KEY_RE.fullmatch(key)
        sm = _S_KEY_RE.fullmatch(key)
        pam = _PA_KEY_RE.fullmatch(key)

        if nlm:
            n_nl += 1
            k = int(nlm.group(1))
            if expecting != "NL":
                errors.append({
                    "position": pos,
                    "key": key,
                    "code": "NL_WITHOUT_PREVIOUS_S",
                    "message": f"{key} appears before S{expected_pair - 1} is completed.",
                })
            if k != expected_pair:
                errors.append({
                    "position": pos,
                    "key": key,
                    "code": "NL_NON_CONSECUTIVE",
                    "message": f"Expected NL{expected_pair}, found {key}.",
                })
            if not isinstance(value, str) or not value.strip():
                errors.append({
                    "position": pos, "key": key,
                    "code": "NL_INVALID_VALUE",
                    "message": "NL value must be a non-empty string.",
                })
            elif require_terminal_period and not value.strip().endswith("."):
                errors.append({
                    "position": pos, "key": key,
                    "code": "NL_MISSING_PERIOD",
                    "message": "NL value must end with a period.",
                })
            expecting = "S"
            previous_kind = "NL"
            continue

        if sm:
            n_s += 1
            k = int(sm.group(1))
            if expecting != "S":
                errors.append({
                    "position": pos, "key": key,
                    "code": "S_WITHOUT_NL",
                    "message": f"{key} must immediately follow NL{expected_pair}.",
                })
            if k != expected_pair:
                errors.append({
                    "position": pos, "key": key,
                    "code": "S_NON_CONSECUTIVE",
                    "message": f"Expected S{expected_pair}, found {key}.",
                })
            if not isinstance(value, str) or not value.strip():
                errors.append({
                    "position": pos, "key": key,
                    "code": "S_INVALID_VALUE",
                    "message": "S value must be a non-empty string.",
                })
            else:
                text = value.strip()
                if text.lower().startswith(key.lower() + ":"):
                    errors.append({
                        "position": pos, "key": key,
                        "code": "S_REDUNDANT_PREFIX",
                        "message": f"{key} value must not repeat the '{key}:' prefix.",
                    })
                if require_terminal_period and not text.endswith("."):
                    errors.append({
                        "position": pos, "key": key,
                        "code": "S_MISSING_PERIOD",
                        "message": "S value must end with a period.",
                    })
            expected_pair += 1
            expecting = "NL"
            previous_kind = "S"
            continue

        if pam:
            n_pa += 1
            k = int(pam.group(1))
            if previous_kind != "S" or expecting != "NL":
                errors.append({
                    "position": pos, "key": key,
                    "code": "PA_BAD_POSITION",
                    "message": f"{key} may appear only immediately after a completed NL/S pair.",
                })
            if k != expected_pa:
                errors.append({
                    "position": pos, "key": key,
                    "code": "PA_NON_CONSECUTIVE",
                    "message": f"Expected PA{expected_pa}, found {key}.",
                })

            grid_result = validate_partial_grid(
                value,
                n_houses=n_houses,
                expected_header=expected_header,
                attribute_values=attribute_values,
            )
            for err in grid_result["errors"]:
                errors.append({
                    "position": pos, "key": key,
                    "code": "PA_GRID_INVALID",
                    "message": err,
                })

            expected_pa += 1
            previous_kind = "PA"
            continue

        errors.append({
            "position": pos,
            "key": key,
            "code": "UNKNOWN_REASONING_KEY",
            "message": "Allowed keys are only NLi, Si, and PAi.",
        })
        previous_kind = "UNKNOWN"

    if expecting == "S":
        errors.append({
            "position": len(keys),
            "key": None,
            "code": "UNPAIRED_FINAL_NL",
            "message": f"NL{expected_pair} has no matching S{expected_pair}.",
        })

    return {
        "ok": not errors,
        "errors": errors,
        "n_entries": len(keys),
        "n_nl": n_nl,
        "n_s": n_s,
        "n_pa": n_pa,
        "n_pairs": min(n_nl, n_s),
        "s_lines": reasoning_to_legacy_s_lines(reasoning),
        "pa_entries": extract_pa_entries(reasoning),
    }


def _ground_truth_lookup(
    ground_truth: Dict[str, Any],
) -> Tuple[Optional[List[str]], Dict[Tuple[int, int], str]]:
    grid = _normalize_grid(ground_truth)
    if grid is None:
        return None, {}

    header = grid["header"]
    lookup: Dict[Tuple[int, int], str] = {}
    for i, row in enumerate(grid["rows"]):
        if len(row) != len(header):
            continue
        for j in range(1, len(header)):
            lookup[(i, j)] = _canon_token(row[j])
    return header, lookup


def score_partial_answers(
    reasoning: Any,
    *,
    n_houses: int,
    expected_header: List[str],
    attribute_values: Dict[str, List[Any]],
    ground_truth: Dict[str, Any],
    syntactic_clues: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    PA PRM metrics.

    Correctness is evaluated only over RESOLVED PA cells.
    Unknown '?' cells are neutral.

    Metrics:
      - pa_structure_score
      - pa_cell_precision
      - pa_final_coverage
      - pa_effective_progress = correct resolved cells / total grid cells
      - pa_monotonicity_score
      - pa_transition_progress_score
      - pa_prm_score

    If no PA is emitted, pa_present=0 and pa_prm_score=0.
    The caller can gate the PA contribution with pa_present so optional
    PAs are neutral rather than penalized.
    """
    entries = extract_pa_entries(reasoning)
    total_cells = max(int(n_houses), 0) * max(len(expected_header) - 1, 0)

    if not entries:
        return {
            "pa_present": 0.0,
            "PA_n_total": 0.0,
            "PA_n_structurally_valid": 0.0,
            "PA_resolved_cells": 0.0,
            "PA_correct_resolved_cells": 0.0,
            "PA_incorrect_resolved_cells": 0.0,
            "pa_structure_score": 0.0,
            "pa_cell_precision": 0.0,
            "pa_final_coverage": 0.0,
            "pa_effective_progress": 0.0,
            "pa_monotonicity_score": 0.0,
            "pa_transition_progress_score": 0.0,
            "pa_prefix_support_score": 0.0,
            "PA_supported_resolved_cells": 0.0,
            "PA_unsupported_resolved_cells": 0.0,
            "pa_prm_score": 0.0,
            "pa_details": [],
        }

    gt_header, gt_lookup = _ground_truth_lookup(ground_truth)
    header_ok = gt_header is not None and [
        str(x) for x in expected_header
    ] == gt_header

    structure_valid = 0
    resolved_total = 0
    correct_total = 0
    incorrect_total = 0
    supported_total = 0
    unsupported_total = 0
    details: List[Dict[str, Any]] = []

    all_s_lines = reasoning_to_legacy_s_lines(reasoning)

    previous_grid: Optional[Dict[str, Any]] = None
    monotonic_transitions = 0
    transition_count = 0
    progress_transitions = 0

    last_coverage = 0.0
    last_effective = 0.0

    for index, entry in enumerate(entries):
        grid = entry["grid"]
        structure = validate_partial_grid(
            grid,
            n_houses=n_houses,
            expected_header=expected_header,
            attribute_values=attribute_values,
        )
        if structure["ok"]:
            structure_valid += 1

        norm = _normalize_grid(grid)
        resolved = correct = incorrect = 0

        if norm is not None and header_ok:
            for i, row in enumerate(norm["rows"]):
                if len(row) != len(expected_header):
                    continue
                for j in range(1, len(expected_header)):
                    value = str(row[j]).strip()
                    if value == "?":
                        continue
                    resolved += 1
                    if _canon_token(value) == gt_lookup.get((i, j)):
                        correct += 1
                    else:
                        incorrect += 1

        resolved_total += resolved
        correct_total += correct
        incorrect_total += incorrect

        coverage = (resolved / total_cells) if total_cells else 0.0
        precision = (correct / resolved) if resolved else 1.0
        effective = (correct / total_cells) if total_cells else 0.0

        monotonic_ok = True
        progress_ok = True

        if previous_grid is not None and norm is not None:
            transition_count += 1
            prev = _normalize_grid(previous_grid)
            if prev is None:
                monotonic_ok = False
                progress_ok = False
            else:
                newly_resolved = 0
                for i in range(min(len(prev["rows"]), len(norm["rows"]))):
                    prow = prev["rows"][i]
                    crow = norm["rows"][i]
                    for j in range(1, min(len(prow), len(crow))):
                        pv = str(prow[j]).strip()
                        cv = str(crow[j]).strip()
                        if pv != "?" and cv != pv:
                            monotonic_ok = False
                        if pv == "?" and cv != "?":
                            newly_resolved += 1
                progress_ok = newly_resolved > 0

            if monotonic_ok:
                monotonic_transitions += 1
            if progress_ok:
                progress_transitions += 1

        support = pa_prefix_support(
            grid,
            syntactic_clues=list(syntactic_clues or []),
            s_lines_prefix=all_s_lines[: int(entry["after_s"])],
        )
        supported_total += int(support["supported_cells"])
        unsupported_total += int(support["unsupported_cells"])

        details.append({
            "pid": entry["pid"],
            "after_s": entry["after_s"],
            "structure_ok": bool(structure["ok"]),
            "resolved_cells": resolved,
            "correct_resolved_cells": correct,
            "incorrect_resolved_cells": incorrect,
            "prefix_supported_cells": int(support["supported_cells"]),
            "prefix_unsupported_cells": int(support["unsupported_cells"]),
            "prefix_support_ratio": _clamp01(support["support_ratio"]),
            "coverage": _clamp01(coverage),
            "precision": _clamp01(precision),
            "effective_progress": _clamp01(effective),
            "monotonic_from_previous": bool(monotonic_ok),
            "adds_new_information": bool(progress_ok),
            "errors": structure["errors"],
            "unsupported": support.get("unsupported", []),
        })

        previous_grid = grid
        last_coverage = coverage
        last_effective = effective

    n_pa = len(entries)
    structure_score = structure_valid / n_pa
    cell_precision = (
        correct_total / resolved_total if resolved_total else 1.0
    )
    monotonicity_score = (
        1.0 if transition_count == 0
        else monotonic_transitions / transition_count
    )
    transition_progress_score = (
        1.0 if transition_count == 0
        else progress_transitions / transition_count
    )
    prefix_support_score = (
        supported_total / resolved_total if resolved_total else 1.0
    )

    # Correctness and checkpoint support dominate. Coverage/progress has
    # deliberately low weight so the model is not rewarded for aggressively
    # filling future cells before they are established.
    pa_prm = (
        0.15 * structure_score
        + 0.30 * cell_precision
        + 0.25 * prefix_support_score
        + 0.15 * monotonicity_score
        + 0.10 * transition_progress_score
        + 0.05 * _clamp01(last_effective)
    )

    return {
        "pa_present": 1.0,
        "PA_n_total": float(n_pa),
        "PA_n_structurally_valid": float(structure_valid),
        "PA_resolved_cells": float(resolved_total),
        "PA_correct_resolved_cells": float(correct_total),
        "PA_incorrect_resolved_cells": float(incorrect_total),
        "pa_structure_score": _clamp01(structure_score),
        "pa_cell_precision": _clamp01(cell_precision),
        "pa_final_coverage": _clamp01(last_coverage),
        "pa_effective_progress": _clamp01(last_effective),
        "pa_monotonicity_score": _clamp01(monotonicity_score),
        "pa_transition_progress_score": _clamp01(transition_progress_score),
        "pa_prefix_support_score": _clamp01(prefix_support_score),
        "PA_supported_resolved_cells": float(supported_total),
        "PA_unsupported_resolved_cells": float(unsupported_total),
        "pa_prm_score": _clamp01(pa_prm),
        "pa_details": details,
    }


def compute_s_prm_metrics(
    z3_out: Dict[str, Any],
    *,
    n_s_steps: int,
    reasoning_solution_consistency: float = 0.0,
) -> Dict[str, float]:
    """
    S-step PRM normalized by ACTUAL emitted S-step count.

    This replaces grid-size normalization for process metrics.
    """
    n_s = max(int(n_s_steps), 0)
    if n_s == 0:
        return {
            "s_parse_ratio": 0.0,
            "s_validity_ratio": 0.0,
            "s_novelty_ratio": 0.0,
            "s_contradiction_ratio": 0.0,
            "s_prm_score": 0.0,
        }

    parsed = float(z3_out.get("n_steps_parsed_ok", 0) or 0)
    valid = float(z3_out.get("n_steps_valid", 0) or 0)
    novel = float(z3_out.get("n_steps_novel_inc_clues", 0) or 0)
    contra = float(z3_out.get("n_non_valid_contradiction", 0) or 0)

    parse_ratio = _clamp01(parsed / n_s)
    validity_ratio = _clamp01(valid / n_s)
    novelty_ratio = _clamp01(novel / n_s)
    contradiction_ratio = _clamp01(contra / n_s)
    sol_consistency = _clamp01(reasoning_solution_consistency)

    score = (
        0.10 * parse_ratio
        + 0.35 * validity_ratio
        + 0.30 * novelty_ratio
        + 0.25 * sol_consistency
        - 0.30 * contradiction_ratio
    )

    return {
        "s_parse_ratio": parse_ratio,
        "s_validity_ratio": validity_ratio,
        "s_novelty_ratio": novelty_ratio,
        "s_contradiction_ratio": contradiction_ratio,
        "s_prm_score": _clamp01(score),
    }


def combine_process_prm(
    *,
    s_prm_score: float,
    pa_prm_score: float,
    pa_present: float,
) -> float:
    """
    Conservative combined PRM.

    PA is optional. When absent, S-PRM carries the whole process signal.
    When present, weight S/PA as 70/30.
    """
    s = _clamp01(s_prm_score)
    if float(pa_present) <= 0.0:
        return s
    return _clamp01(
        0.70 * s + 0.30 * _clamp01(pa_prm_score)
    )
