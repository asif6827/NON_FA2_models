from __future__ import annotations

import ast
import copy
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import z3


# =============================================================================
# Defaults
# =============================================================================

MISSING_PA_DEFAULTS = {
    "pa_present": 0.0,
    "PA_n_total": 0,
    "PA_n_evaluated": 0,
    "PA_n_resolved_cells": 0,

    # Requested PA reward components
    "pa_base_check": 0.0,
    "pa_accuracy": 0.0,
    "pa_consistency": 0.0,
    "pa_coverage": 0.0,
    "pa_gate_pass": 0.0,
    "pa_reward": 0.0,

    # Accuracy diagnostics
    "PA_n_gt_correct_cells": 0,
    "PA_n_gt_wrong_cells": 0,

    # Consistency diagnostics
    "PA_n_consistency_checked_cells": 0,
    "PA_n_consistent_cells": 0,
    "PA_n_inconsistent_cells": 0,

    # Coverage diagnostics
    "PA_n_newly_filled_cells": 0,
    "PA_total_solution_cells": 0,

    # Debugging
    "list_pa_gt_wrong_cells": [],
    "list_pa_inconsistent_cells": [],
    "list_pa_coverage_transitions": [],
    "list_s_prefix_errors": [],
    "list_pa_errors": [],
    "pa_details": [],

    "reward_status": "missing_or_failed",
}


# =============================================================================
# Key patterns
# =============================================================================

_S_KEY_RE = re.compile(r"^S(\d+)$", re.IGNORECASE)
_PA_KEY_RE = re.compile(r"^PA(\d+)$", re.IGNORECASE)


# =============================================================================
# Entity / Z3 helpers
# =============================================================================

def _sanitize_token(value: Any) -> str:
    """Canonical token for Z3 variable names and PA/GT comparison."""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    if not s:
        s = "v"
    if s[0].isdigit():
        s = f"v_{s}"

    return s


def _header_key(value: Any) -> str:
    """Canonical header key insensitive to spaces/underscores/punctuation."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _token_aliases(value: Any) -> Set[str]:
    raw = str(value)
    stripped = raw.strip()
    underscored = re.sub(r"\s+", "_", stripped)

    return {
        raw,
        stripped,
        stripped.lower(),
        underscored,
        underscored.lower(),
        _sanitize_token(stripped),
    }


def _build_var_map(
    attribute_values: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Map every domain value to one Z3 Int variable representing its house.
    """
    var_map: Dict[str, Any] = {}
    canonical_owner: Dict[str, Tuple[str, str]] = {}

    for attr, values in attribute_values.items():
        if not isinstance(values, list):
            raise ValueError(
                f"attribute_values[{attr!r}] must be a list."
            )

        for value in values:
            canonical = _sanitize_token(value)

            if canonical in canonical_owner:
                prev_attr, prev_value = canonical_owner[canonical]
                if prev_attr != str(attr) or prev_value != str(value):
                    raise ValueError(
                        "Duplicate entity token after normalization: "
                        f"{value!r} ({attr}) conflicts with "
                        f"{prev_value!r} ({prev_attr})."
                    )
            else:
                canonical_owner[canonical] = (str(attr), str(value))

            zvar = z3.Int(canonical)

            for alias in _token_aliases(value):
                if alias in var_map and not z3.eq(var_map[alias], zvar):
                    raise ValueError(
                        f"Alias collision for entity token {alias!r}."
                    )
                var_map[alias] = zvar

    return var_map


def _lookup_var(
    token: Any,
    var_map: Dict[str, Any],
):
    candidates = [
        str(token),
        str(token).strip(),
        str(token).strip().lower(),
        re.sub(r"\s+", "_", str(token).strip()),
        re.sub(r"\s+", "_", str(token).strip()).lower(),
        _sanitize_token(token),
    ]

    for candidate in candidates:
        if candidate in var_map:
            return var_map[candidate]

    raise KeyError(f"Unknown entity token: {token!r}")


def _build_base_axioms(
    n_houses: int,
    attribute_values: Dict[str, List[str]],
    var_map: Dict[str, Any],
) -> List[Any]:
    """
    Structural Zebra constraints only:

      1. every entity occupies one house in 1..N;
      2. values within each attribute occupy distinct houses.

    The complete syntactic-clue set is intentionally NOT inserted into the
    PA-consistency solver.
    """
    axioms: List[Any] = []

    unique_vars: Dict[str, Any] = {}
    for zvar in var_map.values():
        unique_vars[zvar.decl().name()] = zvar

    for zvar in unique_vars.values():
        axioms.append(
            z3.And(
                zvar >= 1,
                zvar <= n_houses,
            )
        )

    for _, values in attribute_values.items():
        vars_for_attr = [
            _lookup_var(value, var_map)
            for value in values
        ]
        if len(vars_for_attr) >= 2:
            axioms.append(z3.Distinct(*vars_for_attr))

    return axioms


def _new_base_solver(
    n_houses: int,
    attribute_values: Dict[str, List[str]],
    timeout_s: float,
) -> Tuple[z3.Solver, Dict[str, Any]]:
    var_map = _build_var_map(attribute_values)

    solver = z3.Solver()
    solver.set(
        "timeout",
        int(max(float(timeout_s), 0.001) * 1000),
    )
    solver.add(
        _build_base_axioms(
            n_houses,
            attribute_values,
            var_map,
        )
    )

    return solver, var_map


def _clone_solver(
    solver: z3.Solver,
    timeout_s: float,
) -> z3.Solver:
    out = z3.Solver()
    out.set(
        "timeout",
        int(max(float(timeout_s), 0.001) * 1000),
    )
    out.add(solver.assertions())
    return out


# =============================================================================
# Safe parser for symbolic S_i expressions
# =============================================================================

def _ast_to_z3(
    node: ast.AST,
    var_map: Dict[str, Any],
):
    """
    Supported examples:

        Arnold == 2
        Eric != 1
        Fred < Eric
        Bella + 1 == Eric
        Bella == Eric + 1
        Bella == Eric - 1

        Not(Eric == 1)
        And(Arnold == 1, Eric == 2)
        Or(Bella + 1 == Eric, Eric + 1 == Bella)
    """

    if isinstance(node, ast.Expression):
        return _ast_to_z3(node.body, var_map)

    if isinstance(node, ast.Name):
        return _lookup_var(node.id, var_map)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return z3.BoolVal(node.value)
        if isinstance(node.value, int):
            return z3.IntVal(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_ast_to_z3(node.operand, var_map)
        raise ValueError(
            f"Unsupported unary operator: {type(node.op).__name__}"
        )

    if isinstance(node, ast.BinOp):
        left = _ast_to_z3(node.left, var_map)
        right = _ast_to_z3(node.right, var_map)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right

        raise ValueError(
            f"Unsupported arithmetic operator: {type(node.op).__name__}"
        )

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Chained comparisons are not supported.")

        left = _ast_to_z3(node.left, var_map)
        right = _ast_to_z3(node.comparators[0], var_map)
        op = node.ops[0]

        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right

        raise ValueError(
            f"Unsupported comparison operator: {type(op).__name__}"
        )

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = node.func.id.lower()
        args = [_ast_to_z3(arg, var_map) for arg in node.args]

        if func == "not":
            if len(args) != 1:
                raise ValueError("Not(...) requires exactly one argument.")
            return z3.Not(args[0])

        if func == "and":
            if len(args) < 2:
                raise ValueError("And(...) requires at least two arguments.")
            return z3.And(*args)

        if func == "or":
            if len(args) < 2:
                raise ValueError("Or(...) requires at least two arguments.")
            return z3.Or(*args)

        raise ValueError(
            f"Unsupported Boolean operator: {node.func.id!r}"
        )

    raise ValueError(
        f"Unsupported symbolic syntax: {type(node).__name__}"
    )


def _expr_to_z3(
    expr: Any,
    var_map: Dict[str, Any],
):
    raw = str(expr).strip().rstrip(".").strip()

    if not raw:
        raise ValueError("Empty symbolic reasoning step.")

    try:
        tree = ast.parse(raw, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Invalid symbolic syntax: {raw!r}"
        ) from exc

    phi = _ast_to_z3(tree, var_map)

    if not z3.is_bool(phi):
        raise ValueError(
            f"S-step is not Boolean: {raw!r}"
        )

    return phi


# =============================================================================
# PA / GT parsing
# =============================================================================

def _domain_value_lookup(
    value: Any,
    allowed_values: List[str],
) -> Optional[str]:
    """
    Match an emitted PA value to its canonical raw domain value.
    """
    aliases_to_raw: Dict[str, str] = {}

    for allowed in allowed_values:
        for alias in _token_aliases(allowed):
            aliases_to_raw[str(alias).lower()] = str(allowed)

    for alias in _token_aliases(value):
        key = str(alias).lower()
        if key in aliases_to_raw:
            return aliases_to_raw[key]

    return None


def _resolve_pa_cells(
    pa_key: str,
    pa: Any,
    *,
    n_houses: int,
    attribute_values: Dict[str, List[str]],
) -> Tuple[Dict[Tuple[int, str], str], List[str]]:
    """
    Convert PA into:
        (house, attribute) -> canonical domain value

    '?' cells are omitted.
    """
    errors: List[str] = []
    resolved: Dict[Tuple[int, str], str] = {}

    if not isinstance(pa, dict):
        return {}, [f"{pa_key} is not a dictionary."]

    header = pa.get("header")
    rows = pa.get("rows")

    if not isinstance(header, list) or not isinstance(rows, list):
        return {}, [
            f"{pa_key} must contain list-valued 'header' and 'rows'."
        ]

    if len(rows) != n_houses:
        errors.append(
            f"{pa_key} has {len(rows)} rows; expected {n_houses}."
        )

    if not header or _header_key(header[0]) != "house":
        errors.append(
            f"{pa_key} header must start with 'House'."
        )
        return {}, errors

    attr_by_key = {
        _header_key(attr): str(attr)
        for attr in attribute_values.keys()
    }

    pa_columns: List[Tuple[int, str]] = []

    for col_idx, header_attr in enumerate(header[1:], start=1):
        attr = attr_by_key.get(_header_key(header_attr))

        if attr is None:
            errors.append(
                f"{pa_key} contains unknown attribute column "
                f"{header_attr!r}."
            )
            continue

        pa_columns.append((col_idx, attr))

    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            errors.append(
                f"{pa_key} row {row_idx} is not a list."
            )
            continue

        if len(row) != len(header):
            errors.append(
                f"{pa_key} row {row_idx} has {len(row)} cells; "
                f"expected {len(header)}."
            )
            continue

        try:
            house = int(row[0])
        except Exception:
            errors.append(
                f"{pa_key} row {row_idx} has invalid House={row[0]!r}."
            )
            continue

        if house != row_idx:
            errors.append(
                f"{pa_key} row {row_idx} has House={house}; "
                f"expected {row_idx}."
            )

        if house < 1 or house > n_houses:
            errors.append(
                f"{pa_key} row {row_idx} has out-of-range House={house}."
            )
            continue

        for col_idx, attr in pa_columns:
            value = row[col_idx]

            if str(value).strip() == "?":
                continue

            canonical_value = _domain_value_lookup(
                value,
                attribute_values[attr],
            )

            if canonical_value is None:
                errors.append(
                    f"{pa_key} house {house}, {attr}: value {value!r} "
                    "is outside attribute_values."
                )
                continue

            resolved[(house, attr)] = canonical_value

    return resolved, errors


def _build_gt_cell_map(
    ground_truth: Any,
    *,
    n_houses: int,
    attribute_values: Dict[str, List[str]],
) -> Tuple[Dict[Tuple[int, str], str], List[str]]:
    """
    Convert GT into:
        (house, attribute) -> normalized GT value
    """
    errors: List[str] = []
    gt_cells: Dict[Tuple[int, str], str] = {}

    if not isinstance(ground_truth, dict):
        return {}, ["ground_truth is not a dictionary."]

    header = ground_truth.get("header")
    rows = ground_truth.get("rows")

    if not isinstance(header, list) or not isinstance(rows, list):
        return {}, [
            "ground_truth must contain list-valued 'header' and 'rows'."
        ]

    if not header or _header_key(header[0]) != "house":
        return {}, [
            "ground_truth header must start with 'House'."
        ]

    attr_by_key = {
        _header_key(attr): str(attr)
        for attr in attribute_values.keys()
    }

    gt_columns: List[Tuple[int, str]] = []

    for col_idx, header_attr in enumerate(header[1:], start=1):
        attr = attr_by_key.get(_header_key(header_attr))
        if attr is not None:
            gt_columns.append((col_idx, attr))

    represented = {attr for _, attr in gt_columns}
    missing = [
        str(attr)
        for attr in attribute_values.keys()
        if attr not in represented
    ]

    if missing:
        errors.append(
            f"ground_truth is missing expected attribute columns: {missing}"
        )

    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            errors.append(
                f"ground_truth row {row_idx} is not a list."
            )
            continue

        if len(row) != len(header):
            errors.append(
                f"ground_truth row {row_idx} has {len(row)} cells; "
                f"expected {len(header)}."
            )
            continue

        try:
            house = int(row[0])
        except Exception:
            errors.append(
                f"ground_truth row {row_idx} has invalid House={row[0]!r}."
            )
            continue

        if house < 1 or house > n_houses:
            errors.append(
                f"ground_truth row {row_idx} has out-of-range House={house}."
            )
            continue

        for col_idx, attr in gt_columns:
            gt_cells[(house, attr)] = _sanitize_token(row[col_idx])

    expected = n_houses * len(attribute_values)

    if len(gt_cells) != expected:
        errors.append(
            f"ground_truth yielded {len(gt_cells)} comparable cells; "
            f"expected {expected}."
        )

    return gt_cells, errors


# =============================================================================
# PA consistency
# =============================================================================

def _cell_consistency_status(
    prefix_solver: z3.Solver,
    phi_cell,
    timeout_s: float,
) -> str:
    """
    For a PA cell c and prior S-prefix A:

        c is CONSISTENT iff

            SAT(A ∧ c)
            and
            UNSAT(A ∧ ¬c)

    This is the non-vacuous implication test used by the reward.
    """
    premise_status = prefix_solver.check()

    if premise_status == z3.unsat:
        return "PREFIX_UNSAT"

    if premise_status == z3.unknown:
        return "UNKNOWN"

    positive = _clone_solver(prefix_solver, timeout_s)
    positive.add(phi_cell)
    positive_status = positive.check()

    negative = _clone_solver(prefix_solver, timeout_s)
    negative.add(z3.Not(phi_cell))
    negative_status = negative.check()

    if (
        positive_status == z3.sat
        and negative_status == z3.unsat
    ):
        return "CONSISTENT"

    if positive_status == z3.unsat:
        return "CONTRADICTION"

    if (
        positive_status == z3.unknown
        or negative_status == z3.unknown
    ):
        return "UNKNOWN"

    return "NOT_IMPLIED"


# =============================================================================
# Main PA reward
# =============================================================================

def reward_PA(
    payload: Dict[str, Any],
    timeout_s: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute PA reward using exactly four quantities.

    A. BASE_CHECK [binary]
       The last PA must contain at least one resolved non-'?' entry.

    B. ACCURACY [binary]
       All resolved cells across all PAs must match GT.

    C. CONSISTENCY [binary]
       For PA_i, construct:
           Zebra structural BASE + every S_j appearing before PA_i.

       For every resolved PA_i cell c, require:
           SAT(prefix_i AND c)
           UNSAT(prefix_i AND NOT(c))

       All resolved PA cells must pass.

    D. COVERAGE [continuous]
       For each PA_i -> PA_{i+1}, count positions changing:
           ? -> resolved

       Each position earns credit at most once.

       coverage =
           unique newly-filled cells after PA_1
           ------------------------------------
           n_houses * number_of_attributes

    FINAL:
        gate = BASE_CHECK * ACCURACY * CONSISTENCY

        pa_reward = coverage if gate == 1 else 0.0

    Notes:
      - syntactic_clues are not added directly to the PA consistency solver;
      - z3_out is not used by this PA reward;
      - there is no monotonicity reward and no backup/alternative reward.
    """

    out = copy.deepcopy(MISSING_PA_DEFAULTS)

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------

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
    reasoning = payload.get("reasoning") or {}
    ground_truth = payload.get("ground_truth") or {}

    if not isinstance(attribute_values, dict) or not attribute_values:
        out["reward_status"] = "invalid_attribute_values"
        return out

    if not isinstance(reasoning, dict):
        out["reward_status"] = "reasoning_not_dict"
        return out

    if not isinstance(ground_truth, dict) or not ground_truth:
        out["reward_status"] = "invalid_ground_truth"
        return out

    pa_keys = [
        str(key)
        for key in reasoning.keys()
        if _PA_KEY_RE.fullmatch(str(key))
    ]

    out["PA_n_total"] = len(pa_keys)
    out["pa_present"] = 1.0 if pa_keys else 0.0

    if not pa_keys:
        out["reward_status"] = "no_pa"
        return out

    total_solution_cells = n_houses * len(attribute_values)
    out["PA_total_solution_cells"] = int(total_solution_cells)

    # -------------------------------------------------------------------------
    # GT map for PA accuracy
    # -------------------------------------------------------------------------

    gt_cells, gt_errors = _build_gt_cell_map(
        ground_truth,
        n_houses=n_houses,
        attribute_values=attribute_values,
    )

    if gt_errors:
        out["list_pa_errors"].extend(gt_errors)
        out["reward_status"] = "ground_truth_structure_error"
        return out

    # -------------------------------------------------------------------------
    # Base-only Z3 solver for PA consistency
    # -------------------------------------------------------------------------

    try:
        prefix_solver, var_map = _new_base_solver(
            n_houses,
            attribute_values,
            timeout_s,
        )
    except Exception as exc:
        out["list_pa_errors"].append(
            f"{type(exc).__name__}: {exc}"
        )
        out["reward_status"] = "base_solver_build_error"
        return out

    base_status = prefix_solver.check()

    if base_status == z3.unsat:
        out["reward_status"] = "base_unsat"
        return out

    if base_status == z3.unknown:
        out["reward_status"] = "base_unknown"
        return out

    # -------------------------------------------------------------------------
    # Walk the emitted reasoning trajectory in insertion order
    # -------------------------------------------------------------------------

    parsed_pa_sequence: List[
        Tuple[str, Dict[Tuple[int, str], str]]
    ] = []

    accuracy_ok = True
    consistency_ok = True
    prefix_healthy = True

    total_resolved = 0
    total_gt_correct = 0
    total_gt_wrong = 0

    total_consistency_checked = 0
    total_consistent = 0
    total_inconsistent = 0

    last_s_seen = 0

    for key, value in reasoning.items():
        key_str = str(key)

        # ---------------------------------------------------------------------
        # S_j: add every preceding symbolic step to the current PA prefix
        # ---------------------------------------------------------------------

        sm = _S_KEY_RE.fullmatch(key_str)

        if sm:
            last_s_seen = int(sm.group(1))

            if not isinstance(value, str):
                prefix_healthy = False
                consistency_ok = False

                out["list_s_prefix_errors"].append({
                    "S": key_str,
                    "error": "S value is not a string.",
                })
                continue

            try:
                phi_s = _expr_to_z3(value, var_map)

                tmp = _clone_solver(prefix_solver, timeout_s)
                tmp.add(phi_s)
                status = tmp.check()

                # A contradictory/unknown prefix cannot be used to prove PA
                # cells. Fail the consistency gate safely.
                if status != z3.sat:
                    prefix_healthy = False
                    consistency_ok = False

                    out["list_s_prefix_errors"].append({
                        "S": key_str,
                        "expr": value,
                        "error": (
                            "Adding this S-step to the reasoning prefix "
                            f"produced {status}."
                        ),
                    })
                    continue

                prefix_solver.add(phi_s)

            except Exception as exc:
                prefix_healthy = False
                consistency_ok = False

                out["list_s_prefix_errors"].append({
                    "S": key_str,
                    "expr": value,
                    "error": f"{type(exc).__name__}: {exc}",
                })

            continue

        # ---------------------------------------------------------------------
        # PA_i checkpoint
        # ---------------------------------------------------------------------

        if not _PA_KEY_RE.fullmatch(key_str):
            continue

        resolved_cells, pa_errors = _resolve_pa_cells(
            key_str,
            value,
            n_houses=n_houses,
            attribute_values=attribute_values,
        )

        if pa_errors:
            accuracy_ok = False
            consistency_ok = False

            out["list_pa_errors"].extend(pa_errors)

            out["pa_details"].append({
                "pa": key_str,
                "after_s": last_s_seen,
                "evaluated": False,
                "errors": pa_errors,
            })

            # Preserve the actual PA position in the chain.
            parsed_pa_sequence.append((key_str, {}))
            continue

        out["PA_n_evaluated"] += 1
        parsed_pa_sequence.append((key_str, resolved_cells))

        pa_gt_correct = 0
        pa_gt_wrong = 0
        pa_consistent = 0
        pa_inconsistent = 0

        cell_details: List[Dict[str, Any]] = []

        # ---------------------------------------------------------------------
        # Accuracy + consistency for every resolved cell
        # ---------------------------------------------------------------------

        for (house, attr), cell_value in resolved_cells.items():
            total_resolved += 1

            # ACCURACY ---------------------------------------------------------

            pa_value_norm = _sanitize_token(cell_value)
            gt_value_norm = gt_cells.get((house, attr))

            gt_match = (
                gt_value_norm is not None
                and pa_value_norm == gt_value_norm
            )

            if gt_match:
                total_gt_correct += 1
                pa_gt_correct += 1
            else:
                total_gt_wrong += 1
                pa_gt_wrong += 1
                accuracy_ok = False

                out["list_pa_gt_wrong_cells"].append({
                    "pa": key_str,
                    "after_s": last_s_seen,
                    "house": house,
                    "attribute": attr,
                    "value": cell_value,
                    "gt_value": gt_value_norm,
                })

            # CONSISTENCY ------------------------------------------------------

            total_consistency_checked += 1

            try:
                zvar = _lookup_var(cell_value, var_map)
                phi_cell = (zvar == house)

                if not prefix_healthy:
                    consistency_status = "PREFIX_INVALID"
                else:
                    consistency_status = _cell_consistency_status(
                        prefix_solver,
                        phi_cell,
                        timeout_s,
                    )

            except Exception as exc:
                consistency_status = "CHECK_ERROR"

                out["list_pa_inconsistent_cells"].append({
                    "pa": key_str,
                    "after_s": last_s_seen,
                    "house": house,
                    "attribute": attr,
                    "value": cell_value,
                    "status": consistency_status,
                    "error": f"{type(exc).__name__}: {exc}",
                })

                total_inconsistent += 1
                pa_inconsistent += 1
                consistency_ok = False

                cell_details.append({
                    "house": house,
                    "attribute": attr,
                    "value": cell_value,
                    "gt_match": gt_match,
                    "consistency_status": consistency_status,
                })
                continue

            if consistency_status == "CONSISTENT":
                total_consistent += 1
                pa_consistent += 1
            else:
                total_inconsistent += 1
                pa_inconsistent += 1
                consistency_ok = False

                out["list_pa_inconsistent_cells"].append({
                    "pa": key_str,
                    "after_s": last_s_seen,
                    "house": house,
                    "attribute": attr,
                    "value": cell_value,
                    "status": consistency_status,
                })

            cell_details.append({
                "house": house,
                "attribute": attr,
                "value": cell_value,
                "gt_match": gt_match,
                "consistency_status": consistency_status,
            })

        out["pa_details"].append({
            "pa": key_str,
            "after_s": last_s_seen,
            "evaluated": True,
            "resolved_cells": len(resolved_cells),
            "gt_correct_cells": pa_gt_correct,
            "gt_wrong_cells": pa_gt_wrong,
            "consistent_cells": pa_consistent,
            "inconsistent_cells": pa_inconsistent,
            "cell_details": cell_details,
        })

    # =========================================================================
    # A. BASE_CHECK
    # =========================================================================

    final_pa_resolved_count = (
        len(parsed_pa_sequence[-1][1])
        if parsed_pa_sequence
        else 0
    )

    pa_base_check = (
        1.0
        if final_pa_resolved_count > 0
        else 0.0
    )

    # =========================================================================
    # B. ACCURACY
    # =========================================================================

    pa_accuracy = (1.0 if accuracy_ok else 0.0)

    # =========================================================================
    # C. CONSISTENCY
    # =========================================================================

    pa_consistency = (1.0 if consistency_ok else 0.0)

    # =========================================================================
    # D. COVERAGE
    # =========================================================================

    unique_newly_filled: Set[Tuple[int, str]] = set()

    for idx in range(len(parsed_pa_sequence) - 1):
        pa_i_key, pa_i_cells = parsed_pa_sequence[idx]
        pa_next_key, pa_next_cells = parsed_pa_sequence[idx + 1]

        previous_positions = set(pa_i_cells.keys())
        next_positions = set(pa_next_cells.keys())

        newly_filled = next_positions - previous_positions

        # A position can receive reward only once over the entire chain.
        newly_rewarded = newly_filled - unique_newly_filled
        unique_newly_filled.update(newly_rewarded)

        out["list_pa_coverage_transitions"].append({
            "from_pa": pa_i_key,
            "to_pa": pa_next_key,
            "previous_resolved_cells": len(previous_positions),
            "next_resolved_cells": len(next_positions),
            "newly_filled_cells": len(newly_filled),
            "newly_rewarded_unique_cells": len(newly_rewarded),
            "newly_rewarded_positions": [
                {
                    "house": house,
                    "attribute": attr,
                }
                for house, attr in sorted(newly_rewarded)
            ],
        })

    n_newly_filled = len(unique_newly_filled)

    pa_coverage = (
        n_newly_filled / total_solution_cells
        if total_solution_cells > 0
        else 0.0
    )

    pa_coverage = float(
        max(0.0, min(1.0, pa_coverage))
    )

    # =========================================================================
    # Combined gate and final reward
    # =========================================================================

    pa_gate_pass = (
        1.0
        if (
            pa_base_check == 1.0
            and pa_accuracy == 1.0
            and pa_consistency == 1.0
        )
        else 0.0
    )

    pa_reward = (
        pa_coverage
        if pa_gate_pass == 1.0
        else 0.0
    )

    # =========================================================================
    # Output
    # =========================================================================

    out["pa_base_check"] = float(pa_base_check)
    out["pa_accuracy"] = float(pa_accuracy)
    out["pa_consistency"] = float(pa_consistency)
    out["pa_coverage"] = float(pa_coverage)
    out["pa_gate_pass"] = float(pa_gate_pass)
    out["pa_reward"] = float(pa_reward)

    out["PA_n_resolved_cells"] = int(total_resolved)

    out["PA_n_gt_correct_cells"] = int(total_gt_correct)
    out["PA_n_gt_wrong_cells"] = int(total_gt_wrong)

    out["PA_n_consistency_checked_cells"] = int(
        total_consistency_checked
    )
    out["PA_n_consistent_cells"] = int(total_consistent)
    out["PA_n_inconsistent_cells"] = int(total_inconsistent)

    out["PA_n_newly_filled_cells"] = int(n_newly_filled)

    if out["PA_n_evaluated"] == 0:
        out["reward_status"] = "no_evaluable_pa"

    elif pa_gate_pass == 1.0:
        out["reward_status"] = "success"

    else:
        failed_gates: List[str] = []

        if pa_base_check == 0.0:
            failed_gates.append("BASE_CHECK")

        if pa_accuracy == 0.0:
            failed_gates.append("ACCURACY")

        if pa_consistency == 0.0:
            failed_gates.append("CONSISTENCY")

        out["reward_status"] = (
            "gate_failed:" + ",".join(failed_gates)
        )

    return out


# =============================================================================
# Quick tests
# =============================================================================

def _print_test_result(
    title: str,
    result: Dict[str, Any],
) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    for key in [
        "reward_status",
        "pa_present",
        "PA_n_total",
        "PA_n_evaluated",
        "pa_base_check",
        "pa_accuracy",
        "pa_consistency",
        "pa_coverage",
        "pa_gate_pass",
        "pa_reward",
        "PA_n_resolved_cells",
        "PA_n_gt_correct_cells",
        "PA_n_gt_wrong_cells",
        "PA_n_consistency_checked_cells",
        "PA_n_consistent_cells",
        "PA_n_inconsistent_cells",
        "PA_n_newly_filled_cells",
        "PA_total_solution_cells",
    ]:
        print(f"{key:36s}: {result.get(key)}")

    if result.get("list_pa_gt_wrong_cells"):
        print("\nGT-wrong PA cells:")
        for item in result["list_pa_gt_wrong_cells"]:
            print("  ", item)

    if result.get("list_pa_inconsistent_cells"):
        print("\nInconsistent / unsupported PA cells:")
        for item in result["list_pa_inconsistent_cells"]:
            print("  ", item)

    if result.get("list_pa_coverage_transitions"):
        print("\nCoverage transitions:")
        for item in result["list_pa_coverage_transitions"]:
            print("  ", item)

    if result.get("list_s_prefix_errors"):
        print("\nS-prefix errors:")
        for item in result["list_s_prefix_errors"]:
            print("  ", item)


def _base_test_payload() -> Dict[str, Any]:
    """
    Two houses x two attributes => 4 non-House solution cells.

    GT:
        House 1 = Arnold, red
        House 2 = Eric, blue
    """
    return {
        "n_houses": 2,
        "attribute_values": {
            "Name": ["Arnold", "Eric"],
            "Color": ["red", "blue"],
        },
        "syntactic_clues": [
            "C1: Arnold == 1.",
            "C2: red == 1.",
        ],
        "ground_truth": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "Eric", "blue"],
            ],
        },
        # z3_out may still be supplied by the unchanged caller, but reward_PA()
        # intentionally does not use it.
        "z3_out": {},
    }


if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # TEST 1: clean trajectory
    #
    # PA1 resolves Arnold/red in House 1.
    # PA2 fills Eric/blue in House 2.
    #
    # Newly filled after PA1 = 2 cells
    # Total solution cells      = 4
    # Expected coverage/reward  = 2/4 = 0.5
    # -------------------------------------------------------------------------

    p1 = _base_test_payload()

    p1["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "NL2": "Red is in house 1.",
        "S2": "red == 1.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "?", "?"],
            ],
        },

        "NL3": "Eric must occupy the remaining house.",
        "S3": "Eric == 2.",

        "NL4": "Blue must occupy the remaining house.",
        "S4": "blue == 2.",

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "Eric", "blue"],
            ],
        },
    }

    r1 = reward_PA(p1)
    _print_test_result(
        "TEST 1 - Clean trajectory: expected reward = 0.5",
        r1,
    )

    assert r1["pa_base_check"] == 1.0
    assert r1["pa_accuracy"] == 1.0
    assert r1["pa_consistency"] == 1.0
    assert abs(r1["pa_coverage"] - 0.5) < 1e-9
    assert abs(r1["pa_reward"] - 0.5) < 1e-9

    # -------------------------------------------------------------------------
    # TEST 2: GT-correct but NOT supported by prior S-prefix
    #
    # red=1 is correct in GT, but before PA1 the S-prefix only says Arnold=1.
    # Therefore Accuracy passes, Consistency fails, reward = 0.
    # -------------------------------------------------------------------------

    p2 = _base_test_payload()

    p2["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "?", "?"],
            ],
        },

        "NL2": "Eric is in house 2.",
        "S2": "Eric == 2.",

        "NL3": "Blue is in house 2.",
        "S3": "blue == 2.",

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "Eric", "blue"],
            ],
        },
    }

    r2 = reward_PA(p2)
    _print_test_result(
        "TEST 2 - Accurate but unsupported PA cell: expected reward = 0",
        r2,
    )

    assert r2["pa_base_check"] == 1.0
    assert r2["pa_accuracy"] == 1.0
    assert r2["pa_consistency"] == 0.0
    assert r2["pa_reward"] == 0.0

    # -------------------------------------------------------------------------
    # TEST 3: PA contains GT-wrong cells
    #
    # PA2 swaps the names. Accuracy must fail, therefore reward = 0.
    # -------------------------------------------------------------------------

    p3 = _base_test_payload()

    p3["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "NL2": "Red is in house 1.",
        "S2": "red == 1.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "?", "?"],
            ],
        },

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Eric", "red"],
                ["2", "Arnold", "?"],
            ],
        },
    }

    r3 = reward_PA(p3)
    _print_test_result(
        "TEST 3 - GT-wrong PA cells: expected reward = 0",
        r3,
    )

    assert r3["pa_base_check"] == 1.0
    assert r3["pa_accuracy"] == 0.0
    assert r3["pa_reward"] == 0.0

    # -------------------------------------------------------------------------
    # TEST 4: final PA is completely unknown
    #
    # BASE_CHECK must fail even if an earlier PA contained valid assignments.
    # -------------------------------------------------------------------------

    p4 = _base_test_payload()

    p4["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "?"],
                ["2", "?", "?"],
            ],
        },

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "?", "?"],
                ["2", "?", "?"],
            ],
        },
    }

    r4 = reward_PA(p4)
    _print_test_result(
        "TEST 4 - Last PA all '?': expected BASE_CHECK=0 and reward=0",
        r4,
    )

    assert r4["pa_base_check"] == 0.0
    assert r4["pa_reward"] == 0.0

    # -------------------------------------------------------------------------
    # TEST 5: Coverage = 0.00
    #
    # PA2 adds nothing beyond PA1.
    #
    # Total solution cells = 4
    # Newly filled cells    = 0
    #
    # Expected:
    #   BASE_CHECK  = 1
    #   ACCURACY    = 1
    #   CONSISTENCY = 1
    #   COVERAGE    = 0.00
    #   REWARD      = 0.00
    # -------------------------------------------------------------------------

    p5 = _base_test_payload()

    p5["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "?"],
                ["2", "?", "?"],
            ],
        },

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "?"],
                ["2", "?", "?"],
            ],
        },
    }

    r5 = reward_PA(p5)

    _print_test_result(
        "TEST 5 - No incremental progress: expected coverage = 0.00",
        r5,
    )

    assert r5["pa_base_check"] == 1.0
    assert r5["pa_accuracy"] == 1.0
    assert r5["pa_consistency"] == 1.0
    assert abs(r5["pa_coverage"] - 0.00) < 1e-9
    assert abs(r5["pa_reward"] - 0.00) < 1e-9


    # -------------------------------------------------------------------------
    # TEST 6: Coverage = 0.25
    #
    # PA1 already contains 3/4 cells.
    # PA2 adds exactly one new cell.
    #
    # Newly filled cells = 1
    # Total cells        = 4
    #
    # Coverage = 1/4 = 0.25
    # -------------------------------------------------------------------------

    p6 = _base_test_payload()

    p6["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "NL2": "Red is in house 1.",
        "S2": "red == 1.",

        "NL3": "Eric is therefore in house 2.",
        "S3": "Eric == 2.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "Eric", "?"],
            ],
        },

        "NL4": "Blue is in the remaining house.",
        "S4": "blue == 2.",

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "Eric", "blue"],
            ],
        },
    }

    r6 = reward_PA(p6)

    _print_test_result(
        "TEST 6 - One newly filled cell: expected coverage = 0.25",
        r6,
    )

    assert r6["pa_base_check"] == 1.0
    assert r6["pa_accuracy"] == 1.0
    assert r6["pa_consistency"] == 1.0
    assert r6["PA_n_newly_filled_cells"] == 1
    assert abs(r6["pa_coverage"] - 0.25) < 1e-9
    assert abs(r6["pa_reward"] - 0.25) < 1e-9


    # -------------------------------------------------------------------------
    # TEST 7: Coverage = 0.50
    #
    # PA1 contains 2/4 cells.
    # PA2 adds the remaining 2 cells.
    #
    # Coverage = 2/4 = 0.50
    # -------------------------------------------------------------------------

    p7 = _base_test_payload()

    p7["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "NL2": "Red is in house 1.",
        "S2": "red == 1.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "?", "?"],
            ],
        },

        "NL3": "Eric occupies the remaining house.",
        "S3": "Eric == 2.",

        "NL4": "Blue occupies the remaining house.",
        "S4": "blue == 2.",

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "Eric", "blue"],
            ],
        },
    }

    r7 = reward_PA(p7)

    _print_test_result(
        "TEST 7 - Two newly filled cells: expected coverage = 0.50",
        r7,
    )

    assert r7["pa_base_check"] == 1.0
    assert r7["pa_accuracy"] == 1.0
    assert r7["pa_consistency"] == 1.0
    assert r7["PA_n_newly_filled_cells"] == 2
    assert abs(r7["pa_coverage"] - 0.50) < 1e-9
    assert abs(r7["pa_reward"] - 0.50) < 1e-9


    # -------------------------------------------------------------------------
    # TEST 8: Coverage = 0.75
    #
    # PA1 contains only Arnold=1.
    # PA2 adds red=1.
    # PA3 adds Eric=2 and blue=2.
    #
    # Total new cells = 3
    # Coverage        = 3/4 = 0.75
    #
    # This also checks accumulation across MULTIPLE PA transitions.
    # -------------------------------------------------------------------------

    p8 = _base_test_payload()

    p8["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "?"],
                ["2", "?", "?"],
            ],
        },

        "NL2": "Red is in house 1.",
        "S2": "red == 1.",

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "?", "?"],
            ],
        },

        "NL3": "Eric occupies house 2.",
        "S3": "Eric == 2.",

        "NL4": "Blue occupies house 2.",
        "S4": "blue == 2.",

        "PA3": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "Eric", "blue"],
            ],
        },
    }

    r8 = reward_PA(p8)

    _print_test_result(
        "TEST 8 - Multi-PA accumulation: expected coverage = 0.75",
        r8,
    )

    assert r8["pa_base_check"] == 1.0
    assert r8["pa_accuracy"] == 1.0
    assert r8["pa_consistency"] == 1.0
    assert r8["PA_n_newly_filled_cells"] == 3
    assert abs(r8["pa_coverage"] - 0.75) < 1e-9
    assert abs(r8["pa_reward"] - 0.75) < 1e-9


    # -------------------------------------------------------------------------
    # TEST 9: Coverage = 1.00
    #
    # PA1 is completely empty.
    # All four cells are established by S-steps before PA2.
    # PA2 then fills all four cells.
    #
    # Newly filled = 4
    # Coverage     = 4/4 = 1.00
    #
    # NOTE:
    # BASE_CHECK only applies to the LAST PA, so an empty PA1 is allowed.
    # -------------------------------------------------------------------------

    p9 = _base_test_payload()

    p9["reasoning"] = {
        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "?", "?"],
                ["2", "?", "?"],
            ],
        },

        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "NL2": "Eric is in house 2.",
        "S2": "Eric == 2.",

        "NL3": "Red is in house 1.",
        "S3": "red == 1.",

        "NL4": "Blue is in house 2.",
        "S4": "blue == 2.",

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "Eric", "blue"],
            ],
        },
    }

    r9 = reward_PA(p9)

    _print_test_result(
        "TEST 9 - Full incremental coverage: expected coverage = 1.00",
        r9,
    )

    assert r9["pa_base_check"] == 1.0
    assert r9["pa_accuracy"] == 1.0
    assert r9["pa_consistency"] == 1.0
    assert r9["PA_n_newly_filled_cells"] == 4
    assert abs(r9["pa_coverage"] - 1.00) < 1e-9
    assert abs(r9["pa_reward"] - 1.00) < 1e-9


    # -------------------------------------------------------------------------
    # TEST 10: Forget + refill should NOT increase coverage twice.
    #
    # PA1:
    #   Arnold known
    #
    # PA2:
    #   adds red              -> +1 coverage cell
    #
    # PA3:
    #   red becomes '?'
    #
    # PA4:
    #   red is filled again
    #
    # red must NOT receive coverage for a second time.
    #
    # Expected:
    #   unique newly-filled cells = 1
    #   coverage = 1/4 = 0.25
    # -------------------------------------------------------------------------

    p10 = _base_test_payload()

    p10["reasoning"] = {
        "NL1": "Arnold is in house 1.",
        "S1": "Arnold == 1.",

        "PA1": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "?"],
                ["2", "?", "?"],
            ],
        },

        "NL2": "Red is in house 1.",
        "S2": "red == 1.",

        "PA2": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "?", "?"],
            ],
        },

        "PA3": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "?"],
                ["2", "?", "?"],
            ],
        },

        "PA4": {
            "header": ["House", "Name", "Color"],
            "rows": [
                ["1", "Arnold", "red"],
                ["2", "?", "?"],
            ],
        },
    }

    r10 = reward_PA(p10)

    _print_test_result(
        "TEST 10 - Refill does not double-count: expected coverage = 0.25",
        r10,
    )

    assert r10["pa_base_check"] == 1.0
    assert r10["pa_accuracy"] == 1.0
    assert r10["pa_consistency"] == 1.0
    assert r10["PA_n_newly_filled_cells"] == 1
    assert abs(r10["pa_coverage"] - 0.25) < 1e-9
    assert abs(r10["pa_reward"] - 0.25) < 1e-9


    print("\nAll extended PA reward tests passed.")

    print("\nAll PA reward tests passed.")
