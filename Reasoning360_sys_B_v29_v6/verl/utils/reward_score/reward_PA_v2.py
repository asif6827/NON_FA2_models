from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import z3


# ============================================================
# Defaults
# ============================================================

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

    # Debugging
    "list_pa_unsupported_cells": [],
    "list_pa_monotonicity_violations": [],
    "list_valid_s_added_to_prefix": [],
    "list_valid_s_prefix_parse_errors": [],
    "list_clue_parse_errors": [],
    "list_pa_errors": [],
    "pa_details": [],
}


# ============================================================
# Key patterns
# ============================================================

_S_KEY_RE = re.compile(r"^S(\d+)$", re.IGNORECASE)
_PA_KEY_RE = re.compile(r"^PA(\d+)$", re.IGNORECASE)
_CID_RE = re.compile(r"^\s*(C\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FUNC_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$")


# ============================================================
# Self-contained Zebra/Z3 helpers
#
# These replace the previous dependency on:
#   z3_reasoning_validator_v13_gt_solve_v10_nlspa as _z3v
# ============================================================


def _sanitize_var_name(token: str) -> str:
    """Create a safe Z3 variable name from an entity token."""
    t = str(token).strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^A-Za-z0-9_]", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "v"
    if t[0].isdigit():
        t = f"v_{t}"
    return t


def _normalized_value_token(value: Any) -> str:
    """Normalization used for aliases in syntactic clue / S-step lookup."""
    return re.sub(r"\s+", "_", str(value).strip())


def _token_aliases(value: Any) -> Set[str]:
    """
    Return common aliases for one attribute value.

    attribute_values itself remains unchanged. These aliases only help the
    solver recognize normalized reasoning tokens such as "grilled_cheese"
    when attribute_values contains "grilled cheese".
    """
    raw = str(value)
    stripped = raw.strip()
    underscored = _normalized_value_token(stripped)
    sanitized = _sanitize_var_name(stripped)

    return {
        raw,
        stripped,
        stripped.lower(),
        underscored,
        underscored.lower(),
        sanitized,
    }


def _build_var_map_from_attribute_values(
    attribute_values: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Build a value-token -> Z3 Int map.

    Every entity token denotes its unknown house position.
    Multiple textual aliases for the same attribute value point to the same
    Z3 variable.
    """
    var_map: Dict[str, Any] = {}
    canonical_owner: Dict[str, Tuple[str, str]] = {}

    for attr, values in attribute_values.items():
        if not isinstance(values, list):
            raise ValueError(f"attribute_values[{attr!r}] must be a list.")

        for value in values:
            canonical = _sanitize_var_name(value)

            if canonical in canonical_owner:
                prev_attr, prev_value = canonical_owner[canonical]
                if prev_attr != attr or str(prev_value) != str(value):
                    raise ValueError(
                        "Duplicate entity token after normalization: "
                        f"{value!r} ({attr}) conflicts with "
                        f"{prev_value!r} ({prev_attr})."
                    )
            else:
                canonical_owner[canonical] = (attr, str(value))

            zvar = z3.Int(canonical)
            for alias in _token_aliases(value):
                if alias in var_map and not z3.eq(var_map[alias], zvar):
                    raise ValueError(
                        f"Alias collision for entity token {alias!r}."
                    )
                var_map[alias] = zvar

    return var_map


def _lookup_var(token: str, var_map: Dict[str, Any]):
    """Resolve an emitted entity token to its Z3 variable."""
    candidates = [
        str(token),
        str(token).strip(),
        str(token).strip().lower(),
        _normalized_value_token(token),
        _normalized_value_token(token).lower(),
        _sanitize_var_name(token),
    ]

    for key in candidates:
        if key in var_map:
            return var_map[key]

    raise KeyError(f"Unknown entity token: {token!r}")


def _build_base_axioms(
    n_houses: int,
    attribute_values: Dict[str, List[str]],
    var_map: Dict[str, Any],
) -> List[Any]:
    """
    Zebra base constraints:
      1) every entity occupies one house in 1..N;
      2) values of each attribute occupy distinct houses.
    """
    axioms: List[Any] = []

    unique_vars: Dict[str, Any] = {}
    for zvar in var_map.values():
        unique_vars[zvar.decl().name()] = zvar

    for zvar in unique_vars.values():
        axioms.append(z3.And(zvar >= 1, zvar <= n_houses))

    for attr, values in attribute_values.items():
        vars_for_attr = [_lookup_var(str(v), var_map) for v in values]
        if len(vars_for_attr) >= 2:
            axioms.append(z3.Distinct(*vars_for_attr))

    return axioms


def _extract_cid_and_expr(line: str) -> Tuple[str, str]:
    raw = str(line).strip()
    if raw.endswith("."):
        raw = raw[:-1].strip()

    m = _CID_RE.match(raw)
    if m:
        return m.group(1).upper(), m.group(2).strip()

    return "C?", raw


def _split_top_level_args(text: str) -> List[str]:
    """Split comma-separated function arguments while respecting nesting."""
    args: List[str] = []
    buf: List[str] = []
    depth = 0

    for ch in text:
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


def _parse_operand(token: str, var_map: Dict[str, Any]):
    token = token.strip()
    if re.fullmatch(r"\d+", token):
        return z3.IntVal(int(token))
    if not _IDENT_RE.fullmatch(token):
        raise ValueError(f"Invalid operand: {token!r}")
    return _lookup_var(token, var_map)


def _parse_atomic(expr: str, var_map: Dict[str, Any]):
    """
    Parse supported atomic syntax:

      A == B
      A != B
      A < B
      A > B
      A == H
      A != H
      A < H
      A > H
      A + d == B

    H is an integer house index. The surrounding base axioms constrain all
    entity variables to valid house positions.
    """
    e = expr.strip()

    # Directed distance: A + d == B
    m = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*([1-9]\d*)\s*==\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)",
        e,
    )
    if m:
        left = _lookup_var(m.group(1), var_map)
        distance = int(m.group(2))
        right = _lookup_var(m.group(3), var_map)
        return left + distance == right

    # General binary relation.
    m = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<|>)\s*"
        r"([A-Za-z_][A-Za-z0-9_]*|\d+)",
        e,
    )
    if not m:
        raise ValueError(f"Unrecognized atomic constraint syntax: {expr!r}")

    left = _lookup_var(m.group(1), var_map)
    op = m.group(2)
    right = _parse_operand(m.group(3), var_map)

    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == ">":
        return left > right

    raise ValueError(f"Unsupported operator: {op!r}")


def _parse_constraint(expr: str, var_map: Dict[str, Any]):
    """
    Parse the current NL/S/PA prompt's solver language:

      atomic: ==, !=, <, >, +d ==
      boolean: Not(...), And(...), Or(...)
    """
    e = str(expr).strip().rstrip(".").strip()
    if not e:
        raise ValueError("Empty constraint expression.")

    fm = _FUNC_RE.fullmatch(e)
    if fm:
        func = fm.group(1).lower()
        inner = fm.group(2).strip()
        args = _split_top_level_args(inner)

        if func == "not":
            if len(args) != 1:
                raise ValueError("Not(...) requires exactly one argument.")
            return z3.Not(_parse_constraint(args[0], var_map))

        if func == "and":
            if len(args) < 2:
                raise ValueError("And(...) requires at least two arguments.")
            return z3.And(*[_parse_constraint(a, var_map) for a in args])

        if func == "or":
            if len(args) < 2:
                raise ValueError("Or(...) requires at least two arguments.")
            return z3.Or(*[_parse_constraint(a, var_map) for a in args])

        raise ValueError(f"Unsupported Boolean operator: {fm.group(1)!r}")

    return _parse_atomic(e, var_map)


def _dsl_to_z3(expr: str, var_map: Dict[str, Any]):
    """
    Compatibility helper retained locally.

    The updated prompt uses textual Z3-like constraints rather than the old
    predicate DSL, so this simply delegates to _parse_constraint().
    """
    return _parse_constraint(expr, var_map)


# ============================================================
# Generic PA reward helpers
# ============================================================


def _canon_expr(expr: Any) -> str:
    s = str(expr).strip()
    if s.endswith("."):
        s = s[:-1].strip()
    return re.sub(r"\s+", "", s)


def _expr_to_z3(expr: str, var_map: Dict[str, Any]):
    raw = str(expr).strip().rstrip(".").strip()
    phi = _dsl_to_z3(raw, var_map)
    if phi is None:
        phi = _parse_constraint(raw, var_map)
    if phi is None:
        raise ValueError(f"Could not parse constraint: {expr!r}")
    return phi


def _clone_solver(solver: z3.Solver, timeout_s: float) -> z3.Solver:
    out = z3.Solver()
    out.set("timeout", int(timeout_s * 1000))
    out.add(solver.assertions())
    return out


def _entailment_status(solver: z3.Solver, phi, timeout_s: float) -> str:
    """
    Return one of:
      ENTAILED, CONTRADICTION, NOT_ENTAILED, PREMISES_UNSAT, UNKNOWN

    The explicit premise-SAT check prevents logical explosion from making
    every PA cell appear entailed when the prefix itself is inconsistent.
    """
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


def _valid_s_indices_from_z3(
    reasoning: Dict[str, Any],
    z3_out: Dict[str, Any],
) -> Set[int]:
    """
    Identify which S_i steps were marked valid by the existing S validator.

    Preferred source:
        z3_out["list_steps_valid"]  -> list of valid S expressions

    Fallback:
        all S ids minus ids in z3_out["list_steps_non_valid"]
    """
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


def _build_base_clue_solver(
    *,
    n_houses: int,
    attribute_values: Dict[str, List[str]],
    syntactic_clues: List[str],
    timeout_s: float,
) -> Tuple[z3.Solver, Dict[str, Any], List[str]]:
    """Build BASE domain/uniqueness constraints + all parseable clues."""
    var_map = _build_var_map_from_attribute_values(attribute_values)
    base_axioms = _build_base_axioms(n_houses, attribute_values, var_map)

    solver = z3.Solver()
    solver.set("timeout", int(timeout_s * 1000))
    solver.add(base_axioms)

    clue_errors: List[str] = []

    for raw_clue in syntactic_clues or []:
        try:
            _, expr = _extract_cid_and_expr(raw_clue)
            phi = _expr_to_z3(expr, var_map)
            solver.add(phi)
        except Exception as exc:
            clue_errors.append(
                f"{raw_clue} -> {type(exc).__name__}: {exc}"
            )

    return solver, var_map, clue_errors


def _resolve_pa_cells(
    pa_key: str,
    pa: Dict[str, Any],
    *,
    n_houses: int,
    attribute_values: Dict[str, List[str]],
) -> Tuple[Dict[Tuple[int, str], str], List[str]]:
    """
    Convert a PA grid into:
        (house, attribute) -> resolved value

    Unknown '?' cells are omitted.
    """
    errors: List[str] = []
    resolved: Dict[Tuple[int, str], str] = {}

    if not isinstance(pa, dict):
        return {}, [f"{pa_key} is not a dictionary."]

    header = pa.get("header")
    rows = pa.get("rows")

    if not isinstance(header, list) or not isinstance(rows, list):
        return {}, [f"{pa_key} must contain list-valued header and rows."]

    if len(rows) != n_houses:
        errors.append(
            f"{pa_key} has {len(rows)} rows; expected {n_houses}."
        )

    if not header or header[0] != "House":
        errors.append(f"{pa_key} header must start with 'House'.")

    for attr in header[1:]:
        if attr not in attribute_values:
            errors.append(
                f"{pa_key} contains unknown attribute column {attr!r}."
            )

    for row_i, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            errors.append(f"{pa_key} row {row_i} is not a list.")
            continue

        if len(row) != len(header):
            errors.append(
                f"{pa_key} row {row_i} has {len(row)} cells; "
                f"expected {len(header)}."
            )
            continue

        try:
            house = int(row[0])
        except Exception:
            errors.append(
                f"{pa_key} row {row_i} has invalid House value {row[0]!r}."
            )
            continue

        if house != row_i:
            errors.append(
                f"{pa_key} row {row_i} has House={house}; expected {row_i}."
            )

        for col_idx, attr in enumerate(header[1:], start=1):
            if attr not in attribute_values:
                continue

            value = row[col_idx]
            if value == "?":
                continue

            # Accept exact or normalized form of a provided domain value.
            allowed = attribute_values[attr]
            allowed_alias_to_raw: Dict[str, str] = {}
            for av in allowed:
                for alias in _token_aliases(av):
                    allowed_alias_to_raw[alias] = str(av)

            value_str = str(value)
            candidate_aliases = _token_aliases(value_str)
            matching_raw = None
            for alias in candidate_aliases:
                if alias in allowed_alias_to_raw:
                    matching_raw = allowed_alias_to_raw[alias]
                    break

            if matching_raw is None:
                errors.append(
                    f"{pa_key} house {house}, {attr}: value {value!r} "
                    "is outside attribute_values."
                )
                continue

            # Store the emitted value. _lookup_var() can resolve its aliases.
            resolved[(house, attr)] = value_str

    return resolved, errors


# ============================================================
# Main PA reward
# ============================================================


def reward_PA(payload: Dict[str, Any], timeout_s: float = 5.0) -> Dict[str, Any]:
    """
    Evaluate structured PA checkpoints using three signals:

      1) Prefix support
         A resolved PA cell must be entailed by:
             BASE + syntactic_clues + VALID emitted S-prefix

      2) PA monotonicity
         Once a cell is resolved in an earlier PA, later PAs must preserve it.

      3) Supported progress
         Fraction of all non-House puzzle cells that are supported in the
         final evaluated PA.

    Validity of S_i is taken from payload['z3_out']; only valid S_i steps are
    added to the evolving PA-prefix solver.

    Reward:
      if monotonicity is applicable:
          0.60 * support + 0.20 * monotonicity + 0.20 * progress

      otherwise (e.g. only one PA):
          0.75 * support + 0.25 * progress
    """
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

    pa_keys = [
        key for key in reasoning.keys()
        if _PA_KEY_RE.fullmatch(str(key))
    ]

    out["PA_n_total"] = len(pa_keys)
    out["pa_present"] = 1.0 if pa_keys else 0.0

    if not pa_keys:
        out["reward_status"] = "no_pa"
        return out

    # ------------------------------------------------------------
    # Which S_i are valid?
    # ------------------------------------------------------------
    try:
        valid_s_indices = _valid_s_indices_from_z3(reasoning, z3_out)
    except Exception as exc:
        out["reward_status"] = "missing_or_unmappable_s_validation"
        out["list_pa_errors"].append(
            f"{type(exc).__name__}: {exc}"
        )
        return out

    # ------------------------------------------------------------
    # Build BASE + syntactic clues.
    # ------------------------------------------------------------
    try:
        prefix_solver, var_map, clue_errors = _build_base_clue_solver(
            n_houses=n_houses,
            attribute_values=attribute_values,
            syntactic_clues=syntactic_clues,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        out["reward_status"] = "base_solver_build_error"
        out["list_pa_errors"].append(
            f"{type(exc).__name__}: {exc}"
        )
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
    total_resolved = 0
    total_supported = 0
    total_unsupported = 0

    monotonic_compared = 0
    monotonic_preserved = 0

    # Stores all previously committed non-'?' PA cells.
    committed_cells: Dict[Tuple[int, str], str] = {}

    last_pa_supported_count = 0
    last_pa_key: Optional[str] = None
    last_s_seen = 0

    # ------------------------------------------------------------
    # Walk the emitted reasoning trajectory IN INSERTION ORDER.
    # ------------------------------------------------------------
    for key, value in reasoning.items():
        key_str = str(key)

        # --------------------------------------------------------
        # S_i: add only if z3_out says it is valid.
        # --------------------------------------------------------
        sm = _S_KEY_RE.fullmatch(key_str)
        if sm:
            s_idx = int(sm.group(1))
            last_s_seen = s_idx

            if s_idx not in valid_s_indices:
                continue

            if not isinstance(value, str):
                out["list_valid_s_prefix_parse_errors"].append({
                    "S": key_str,
                    "error": "S value is not a string.",
                })
                continue

            try:
                phi = _expr_to_z3(value, var_map)

                # A supposedly valid S_i must keep the prefix SAT.
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

        # --------------------------------------------------------
        # PA_i checkpoint.
        # --------------------------------------------------------
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
            out["list_pa_errors"].append(
                f"{key_str}: prefix solver status={prefix_status}"
            )
            continue

        # --------------------------------------------------------
        # 1) Prefix support.
        # --------------------------------------------------------
        supported_count = 0
        unsupported_count = 0
        cell_details: List[Dict[str, Any]] = []

        for (house, attr), cell_value in resolved_cells.items():
            try:
                zvar = _lookup_var(cell_value, var_map)
                phi_cell = (zvar == house)
                status = _entailment_status(
                    prefix_solver,
                    phi_cell,
                    timeout_s,
                )
            except Exception as exc:
                status = "CHECK_ERROR"
                unsupported_count += 1

                out["list_pa_unsupported_cells"].append({
                    "pa": key_str,
                    "after_s": last_s_seen,
                    "house": house,
                    "attribute": attr,
                    "value": cell_value,
                    "status": status,
                    "error": f"{type(exc).__name__}: {exc}",
                })

                cell_details.append({
                    "house": house,
                    "attribute": attr,
                    "value": cell_value,
                    "status": status,
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
        support_score = (
            supported_count / resolved_count
            if resolved_count > 0
            else 0.0
        )

        pa_support_scores.append(support_score)
        total_resolved += resolved_count
        total_supported += supported_count
        total_unsupported += unsupported_count

        # --------------------------------------------------------
        # 2) Monotonicity.
        #
        # Only previously resolved cells are compared. A later PA must
        # preserve every earlier commitment.
        # --------------------------------------------------------
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

        # Add newly committed cells. Existing commitments are deliberately
        # not overwritten, so future comparisons remain anchored to the
        # earliest explicit commitment.
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

    # ------------------------------------------------------------
    # Aggregate metrics.
    # ------------------------------------------------------------
    out["PA_n_resolved_cells"] = total_resolved
    out["PA_n_supported_cells"] = total_supported
    out["PA_n_unsupported_cells"] = total_unsupported

    out["PA_n_monotonicity_compared_cells"] = monotonic_compared
    out["PA_n_monotonicity_preserved_cells"] = monotonic_preserved

    if out["PA_n_evaluated"] == 0:
        out["reward_status"] = "no_evaluable_pa"
        return out

    pa_prefix_support_score = (
        sum(pa_support_scores) / len(pa_support_scores)
        if pa_support_scores
        else 0.0
    )

    if monotonic_compared > 0:
        pa_monotonicity_score = (
            monotonic_preserved / monotonic_compared
        )
        monotonicity_applicable = 1.0
    else:
        pa_monotonicity_score = 0.0
        monotonicity_applicable = 0.0

    # Progress is measured only on the final emitted/evaluated PA.
    pa_supported_progress_score = (
        last_pa_supported_count / total_puzzle_cells
        if total_puzzle_cells > 0
        else 0.0
    )

    out["pa_prefix_support_score"] = float(pa_prefix_support_score)
    out["pa_monotonicity_score"] = float(pa_monotonicity_score)
    out["pa_monotonicity_applicable"] = float(monotonicity_applicable)
    out["pa_supported_progress_score"] = float(
        pa_supported_progress_score
    )

    # ------------------------------------------------------------
    # PA reward formulation.
    # ------------------------------------------------------------
    if monotonicity_applicable:
        pa_reward = (
            0.60 * pa_prefix_support_score
            + 0.20 * pa_monotonicity_score
            + 0.20 * pa_supported_progress_score
        )
    else:
        # Do not grant a free monotonicity reward when no PA->PA comparison
        # exists. Renormalize the two applicable components instead.
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


# ============================================================
# Demonstrations
# ============================================================


def _demo_base_payload() -> Dict[str, Any]:
    """
    Small deliberately underconstrained example.

    This makes it possible to demonstrate that a PA cell can be unsupported
    even if it happens to agree with some eventual solution.
    """
    return {
        "n_houses": 3,
        "attribute_values": {
            "Name": ["Peter", "Eric", "Arnold"],
            "Color": ["red", "white", "yellow"],
            "Children": ["Fred", "Meredith", "Bella"],
        },
        "syntactic_clues": [
            "C1: Arnold == red.",
            "C2: red == 2.",
            "C3: Bella == 1.",
            "C4: Fred < Eric.",
            "C5: white == Meredith.",
        ],
        "ground_truth": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "Peter", "yellow", "Bella"],
                ["2", "Arnold", "red", "Fred"],
                ["3", "Eric", "white", "Meredith"],
            ],
        },
    }


def _print_demo(title: str, result: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    keys = [
        "reward_status",
        "pa_present",
        "PA_n_total",
        "PA_n_evaluated",
        "PA_n_resolved_cells",
        "PA_n_supported_cells",
        "PA_n_unsupported_cells",
        "pa_prefix_support_score",
        "pa_monotonicity_score",
        "pa_supported_progress_score",
        "pa_reward",
    ]

    for key in keys:
        print(f"{key:36s}: {result.get(key)}")

    if result.get("list_valid_s_added_to_prefix"):
        print("valid S added to prefix:", result["list_valid_s_added_to_prefix"])

    if result.get("list_pa_unsupported_cells"):
        print("unsupported PA cells:")
        for x in result["list_pa_unsupported_cells"]:
            print("  ", x)

    if result.get("list_pa_monotonicity_violations"):
        print("monotonicity violations:")
        for x in result["list_pa_monotonicity_violations"]:
            print("  ", x)

    if result.get("list_pa_errors"):
        print("PA errors:")
        for x in result["list_pa_errors"]:
            print("  ", x)


if __name__ == "__main__":
    base = _demo_base_payload()

    # --------------------------------------------------------
    # Demo 1: clean two-PA trajectory.
    # --------------------------------------------------------
    p1 = copy.deepcopy(base)
    p1["reasoning"] = {
        "NL1": "Clues 1 and 2 place Arnold in house 2.",
        "S1": "Arnold == 2.",
        "NL2": "Fred is left of Eric, so Eric cannot be in house 1.",
        "S2": "Eric != 1.",
        "NL3": "Arnold occupies house 2, so Eric must occupy house 3.",
        "S3": "Eric == 3.",
        "PA1": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "?", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "Eric", "?", "?"],
            ],
        },
        "NL4": "The remaining person Peter must occupy house 1.",
        "S4": "Peter == 1.",
        "PA2": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "Peter", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "Eric", "?", "?"],
            ],
        },
    }
    p1["z3_out"] = {
        "n_steps_valid": 4,
        "list_steps_valid": [
            "Arnold == 2",
            "Eric != 1",
            "Eric == 3",
            "Peter == 1",
        ],
        "list_steps_non_valid": [],
    }
    _print_demo("DEMO 1 - Clean two-PA trajectory", reward_PA(p1))

    # --------------------------------------------------------
    # Demo 2: unsupported PA assignment.
    # S1 is valid, but PA1 prematurely claims Eric == 3.
    # --------------------------------------------------------
    p2 = copy.deepcopy(base)
    p2["reasoning"] = {
        "NL1": "Clues 1 and 2 place Arnold in house 2.",
        "S1": "Arnold == 2.",
        "PA1": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "?", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "Eric", "?", "?"],
            ],
        },
    }
    p2["z3_out"] = {
        "n_steps_valid": 1,
        "list_steps_valid": ["Arnold == 2"],
        "list_steps_non_valid": [],
    }
    _print_demo("DEMO 2 - Unsupported PA cell", reward_PA(p2))

    # --------------------------------------------------------
    # Demo 3: monotonicity violation.
    # PA1 commits Arnold=2, PA2 reverts it to '?'.
    # --------------------------------------------------------
    p3 = copy.deepcopy(base)
    p3["reasoning"] = {
        "NL1": "Clues 1 and 2 place Arnold in house 2.",
        "S1": "Arnold == 2.",
        "PA1": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "?", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "?", "?", "?"],
            ],
        },
        "NL2": "Fred is left of Eric, so Eric cannot occupy house 1.",
        "S2": "Eric != 1.",
        "PA2": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "?", "?", "Bella"],
                ["2", "?", "red", "?"],
                ["3", "?", "?", "?"],
            ],
        },
    }
    p3["z3_out"] = {
        "n_steps_valid": 2,
        "list_steps_valid": ["Arnold == 2", "Eric != 1"],
        "list_steps_non_valid": [],
    }
    _print_demo("DEMO 3 - PA monotonicity violation", reward_PA(p3))

    # --------------------------------------------------------
    # Demo 4: invalid S is NOT added to PA prefix.
    # --------------------------------------------------------
    p4 = copy.deepcopy(base)
    p4["reasoning"] = {
        "NL1": "Clues 1 and 2 place Arnold in house 2.",
        "S1": "Arnold == 2.",
        "NL2": "Peter is guessed to be in house 1.",
        "S2": "Peter == 1.",
        "PA1": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "Peter", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "?", "?", "?"],
            ],
        },
    }
    p4["z3_out"] = {
        "n_steps_valid": 1,
        "list_steps_valid": ["Arnold == 2"],
        "list_steps_non_valid": [
            {
                "k": 2,
                "expr": "Peter == 1",
                "validity_status": "NOT_ENTAILED",
            }
        ],
    }
    _print_demo("DEMO 4 - Invalid S excluded from prefix", reward_PA(p4))

    # --------------------------------------------------------
    # Demo 5: no PA.
    # --------------------------------------------------------
    p5 = copy.deepcopy(base)
    p5["reasoning"] = {
        "NL1": "Clues 1 and 2 place Arnold in house 2.",
        "S1": "Arnold == 2.",
    }
    p5["z3_out"] = {
        "n_steps_valid": 1,
        "list_steps_valid": ["Arnold == 2"],
        "list_steps_non_valid": [],
    }
    _print_demo("DEMO 5 - No PA emitted", reward_PA(p5))
