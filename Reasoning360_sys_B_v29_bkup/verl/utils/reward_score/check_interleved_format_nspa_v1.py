import copy
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------------------------------------------------------------
# Reasoning-key patterns for the new NL_i / S_i / PA_i object format.
# -----------------------------------------------------------------------------
_NL_KEY_RE = re.compile(r"^NL(\d+)$")
_S_KEY_RE = re.compile(r"^S(\d+)$")
_PA_KEY_RE = re.compile(r"^PA(\d+)$")

# Detect the old value style, which is no longer allowed under an S_i key:
#   "S1": "S1: Arnold == 2."   <- invalid
_OLD_S_VALUE_PREFIX_RE = re.compile(r"^\s*S\d+\s*:")

# Conservative normalized identifier: Arnold, red, pall_mall, house3, etc.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

FormatCheckResult = Tuple[float, Dict[str, Any]]


def _success(*, n_nl: int, n_s: int, n_pa: int) -> FormatCheckResult:
    """Return the requested successful format reward."""
    return (
        1.0,
        {
            "success": "All NL/S/PA format checks passed.",
            "n_nl": n_nl,
            "n_s": n_s,
            "n_pa": n_pa,
        },
    )


def _failure(reason: str) -> FormatCheckResult:
    """Return the requested failed format reward with a diagnostic reason."""
    return 0.0, {"Failure_reason": reason}


# -----------------------------------------------------------------------------
# S-expression grammar helpers
# -----------------------------------------------------------------------------

def _split_top_level_args(s: str) -> List[str]:
    """Split a comma-separated argument list while respecting nested parentheses."""
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
    elif s.strip():
        # Non-empty input ending in a top-level comma is malformed.
        return []

    return args


def _is_ident(x: str) -> bool:
    return bool(_IDENT_RE.fullmatch(x.strip()))


def _is_house_index(x: str, n_houses: int) -> bool:
    if not x.isdigit():
        return False
    h = int(x)
    return 1 <= h <= n_houses


def _parse_atomic(
    expr: str,
    n_houses: int,
    identifiers: Optional[List[str]] = None,
) -> bool:
    """
    Parse one allowed atomic expression.

    Allowed forms (whitespace flexible):
        A == B
        A != B
        A < B
        A > B
        A + d == B       (d is a positive integer)
        A == H           (H in 1..n_houses)
        A != H           (H in 1..n_houses)

    Notes:
        - For < and >, RHS must be an identifier, not a number.
        - For + d ==, both sides must be identifiers.
        - identifiers, when provided, is populated with all entity tokens.
    """
    e = expr.strip()

    # Directed distance: A + d == B
    m = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*([1-9]\d*)\s*==\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)",
        e,
    )
    if m:
        left, d, right = m.group(1), m.group(2), m.group(3)
        if not (_is_ident(left) and int(d) > 0 and _is_ident(right)):
            return False
        if identifiers is not None:
            identifiers.extend([left, right])
        return True

    # Binary operator: A (op) RHS
    m = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<|>)\s*"
        r"([A-Za-z_][A-Za-z0-9_]*|\d+)",
        e,
    )
    if not m:
        return False

    left, op, rhs = m.group(1), m.group(2), m.group(3)
    if not _is_ident(left):
        return False

    if identifiers is not None:
        identifiers.append(left)

    if op in ("<", ">"):
        # Ordering is between two entity-position variables only.
        if not _is_ident(rhs):
            return False
        if identifiers is not None:
            identifiers.append(rhs)
        return True

    if op in ("==", "!="):
        if rhs.isdigit():
            # New prompt allows both A == H and A != H.
            return _is_house_index(rhs, n_houses)

        if not _is_ident(rhs):
            return False
        if identifiers is not None:
            identifiers.append(rhs)
        return True

    return False


def _parse_expr(
    expr: str,
    n_houses: int,
    identifiers: Optional[List[str]] = None,
) -> bool:
    """
    Parse the restricted solver-checkable expression grammar.

    Boolean forms:
        Not(e)
        And(e1, e2, ..., en)   with n >= 2
        Or(e1, e2, ..., en)    with n >= 2

    Nested Boolean expressions are allowed.
    """
    e = expr.strip()
    if not e:
        return False

    # Not(...)
    if e.startswith("Not(") and e.endswith(")"):
        inner = e[len("Not(") : -1].strip()
        return bool(inner) and _parse_expr(inner, n_houses, identifiers)

    # And(...)
    if e.startswith("And(") and e.endswith(")"):
        inner = e[len("And(") : -1].strip()
        args = _split_top_level_args(inner)
        return len(args) >= 2 and all(
            _parse_expr(arg, n_houses, identifiers) for arg in args
        )

    # Or(...)
    if e.startswith("Or(") and e.endswith(")"):
        inner = e[len("Or(") : -1].strip()
        args = _split_top_level_args(inner)
        return len(args) >= 2 and all(
            _parse_expr(arg, n_houses, identifiers) for arg in args
        )

    return _parse_atomic(e, n_houses, identifiers)


# -----------------------------------------------------------------------------
# Domain / NL / S validation helpers
# -----------------------------------------------------------------------------

def _validate_checker_inputs(
    *,
    n_houses: int,
    expected_header: List[str],
    attribute_values: Dict[str, List[str]],
) -> Optional[str]:
    """Validate the metadata needed by the format checker itself."""
    if not isinstance(n_houses, int) or n_houses <= 0:
        return "n_houses must be a positive integer."

    if not isinstance(expected_header, list) or not expected_header:
        return "expected_header must be a non-empty list."

    if not all(isinstance(x, str) and x for x in expected_header):
        return "Every expected_header entry must be a non-empty string."

    if expected_header.count("House") != 1:
        return "expected_header must contain exactly one 'House' column."

    if not isinstance(attribute_values, dict) or not attribute_values:
        return "attribute_values must be a non-empty dict."

    for attr in expected_header:
        if attr == "House":
            continue
        if attr not in attribute_values:
            return f"Header attribute '{attr}' is missing from attribute_values."
        values = attribute_values[attr]
        if not isinstance(values, list) or not values:
            return f"attribute_values['{attr}'] must be a non-empty list."
        if not all(isinstance(v, str) and v for v in values):
            return f"All values in attribute_values['{attr}'] must be non-empty strings."

    return None


def _allowed_entity_tokens(attribute_values: Dict[str, List[str]]) -> Set[str]:
    """Flatten all puzzle-domain entity values into one exact-match token set."""
    return {
        value
        for values in attribute_values.values()
        if isinstance(values, list)
        for value in values
        if isinstance(value, str)
    }


def _validate_nl_value(key: str, value: Any) -> Optional[str]:
    """Check the structural NL_i requirements that can be reliably format-checked."""
    if not isinstance(value, str) or not value.strip():
        return f"{key} must contain a non-empty natural-language string."

    text = value.strip()
    if "\n" in text or "\r" in text:
        return f"{key} must be a single-line natural-language sentence."

    if not text.endswith("."):
        return f"{key} must end with a period."

    return None


def _validate_s_value(
    key: str,
    value: Any,
    *,
    n_houses: int,
    allowed_tokens: Set[str],
) -> Optional[str]:
    """Validate S_i punctuation, restricted grammar, and entity-token domains."""
    if not isinstance(value, str) or not value.strip():
        return f"{key} must contain a non-empty syntactic constraint string."

    text = value.strip()
    if "\n" in text or "\r" in text:
        return f"{key} must contain exactly one single-line syntactic constraint."

    if _OLD_S_VALUE_PREFIX_RE.match(text):
        return (
            f"{key} value must contain only the constraint; do not repeat an S-step "
            f"prefix inside the value."
        )

    if not text.endswith("."):
        return f"{key} must end with a period."

    expr = text[:-1].strip()
    if not expr:
        return f"{key} has an empty syntactic expression before the final period."

    identifiers: List[str] = []
    if not _parse_expr(expr, n_houses, identifiers):
        return f"{key} uses invalid or unsupported S grammar: '{expr}'."

    for token in identifiers:
        if token not in allowed_tokens:
            return (
                f"{key} contains entity token '{token}' that is not present in "
                "attribute_values."
            )

    return None


# -----------------------------------------------------------------------------
# PA validation helpers
# -----------------------------------------------------------------------------

def _validate_pa(
    key: str,
    pa: Any,
    *,
    n_houses: int,
    expected_header: List[str],
    attribute_values: Dict[str, List[str]],
) -> Optional[str]:
    """Validate PA grid structure, cell domains, and within-PA uniqueness."""
    if not isinstance(pa, dict):
        return f"{key} must be a JSON object/dict, not a string or other value."

    if set(pa.keys()) != {"header", "rows"} or len(pa) != 2:
        return f"{key} must contain exactly two keys: 'header' and 'rows'."

    header = pa.get("header")
    rows = pa.get("rows")

    if header != expected_header:
        return f"{key}.header must be exactly identical to expected_header."

    if not isinstance(rows, list):
        return f"{key}.rows must be a list."

    if len(rows) != n_houses:
        return f"{key}.rows must contain exactly {n_houses} rows; found {len(rows)}."

    n_cols = len(expected_header)
    house_col = expected_header.index("House")

    # Validate row shapes, House cells, and cell domains.
    for row_idx, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            return f"{key}.rows[{row_idx - 1}] must be a list."

        if len(row) != n_cols:
            return (
                f"{key}.rows[{row_idx - 1}] must contain exactly {n_cols} cells; "
                f"found {len(row)}."
            )

        if not all(isinstance(cell, str) for cell in row):
            return f"{key}.rows[{row_idx - 1}] must contain strings only."

        expected_house = str(row_idx)
        if row[house_col] != expected_house:
            return (
                f"{key} House cell for row {row_idx} must be '{expected_house}', "
                f"found '{row[house_col]}'."
            )

        for col_idx, attr in enumerate(expected_header):
            if attr == "House":
                continue

            cell = row[col_idx]
            if cell == "?":
                continue

            if cell not in attribute_values[attr]:
                return (
                    f"{key} has invalid value '{cell}' at house {row_idx}, column "
                    f"'{attr}'. Expected '?' or a value from attribute_values['{attr}']."
                )

    # Every resolved non-'?' value must be unique within each attribute column.
    for col_idx, attr in enumerate(expected_header):
        if attr == "House":
            continue

        seen: Set[str] = set()
        for row_idx, row in enumerate(rows, start=1):
            cell = row[col_idx]
            if cell == "?":
                continue
            if cell in seen:
                return (
                    f"{key} column '{attr}' contains duplicate resolved value "
                    f"'{cell}' (duplicate encountered at house {row_idx})."
                )
            seen.add(cell)

    return None


def _validate_pa_monotonicity(
    previous_key: str,
    previous_pa: Dict[str, Any],
    current_key: str,
    current_pa: Dict[str, Any],
    *,
    expected_header: List[str],
) -> Optional[str]:
    """Require every later PA to preserve all previously resolved cells."""
    prev_rows = previous_pa["rows"]
    curr_rows = current_pa["rows"]

    for row_idx, (prev_row, curr_row) in enumerate(
        zip(prev_rows, curr_rows), start=1
    ):
        for col_idx, attr in enumerate(expected_header):
            if attr == "House":
                continue

            old = prev_row[col_idx]
            new = curr_row[col_idx]

            if old == "?":
                # '?' may stay '?' or become resolved.
                continue

            if new != old:
                if new == "?":
                    return (
                        f"{current_key} is non-monotonic relative to {previous_key}: "
                        f"house {row_idx}, column '{attr}' reverted from resolved "
                        f"value '{old}' to '?'."
                    )
                return (
                    f"{current_key} is non-monotonic relative to {previous_key}: "
                    f"house {row_idx}, column '{attr}' changed from '{old}' to '{new}'."
                )

    return None


# -----------------------------------------------------------------------------
# Main NL/S/PA format checker
# -----------------------------------------------------------------------------

def check_interleaved_reasoning(
    reasoning: Any,
    *,
    n_houses: int,
    expected_header: List[str],
    attribute_values: Dict[str, List[str]],
) -> FormatCheckResult:
    """
    Validate the updated NL_i / S_i / PA_i interleaved reasoning FORMAT.

    This checker intentionally handles format/state constraints only:
        1. reasoning is a non-empty dict (insertion order is the trajectory)
        2. only NL<i>, S<i>, PA<i> keys are allowed
        3. strict NL_i -> S_i pairing
        4. consecutive NL/S numbering starting from 1
        5. PA placement only immediately after a completed NL/S pair
        6. consecutive PA numbering starting from 1
        7. NL format: non-empty, single-line, final period
        8. S grammar: restricted solver-checkable grammar, final period
        9. S entity tokens must occur in attribute_values
       10. PA grid structure: exact header, n_houses rows, row sizes, House cells
       11. PA cell domains: '?' or value from the corresponding attribute domain
       12. PA uniqueness for every resolved attribute value
       13. PA monotonicity across successive checkpoints

    It does NOT check:
        - whether an S_i deduction is logically entailed
        - whether a PA_i resolved cell is supported by the reasoning prefix
        - final-solution correctness

    Returns:
        Success:
            (1.0, {"success": "...", "n_nl": ..., "n_s": ..., "n_pa": ...})
        Failure:
            (0.0, {"Failure_reason": "..."})
    """
    input_error = _validate_checker_inputs(
        n_houses=n_houses,
        expected_header=expected_header,
        attribute_values=attribute_values,
    )
    if input_error:
        return _failure(input_error)

    # 1) reasoning is dict
    if not isinstance(reasoning, dict):
        return _failure("reasoning must be a JSON object/dict, not a list or string.")
    if not reasoning:
        return _failure("reasoning must be a non-empty dict.")

    items = list(reasoning.items())  # Python preserves emitted/insertion order.
    keys = [k for k, _ in items]

    # 2) allowed keys only
    for key in keys:
        if not isinstance(key, str) or not (
            _NL_KEY_RE.fullmatch(key)
            or _S_KEY_RE.fullmatch(key)
            or _PA_KEY_RE.fullmatch(key)
        ):
            return _failure(
                f"Disallowed reasoning key '{key}'. Allowed keys are NL<i>, S<i>, PA<i>."
            )

    allowed_tokens = _allowed_entity_tokens(attribute_values)

    expected_step = 1
    expected_pa = 1
    expecting_s = False
    active_nl_key: Optional[str] = None
    previous_key: Optional[str] = None

    n_nl = 0
    n_s = 0
    n_pa = 0

    previous_pa_key: Optional[str] = None
    previous_pa: Optional[Dict[str, Any]] = None

    for key, value in items:
        nl_match = _NL_KEY_RE.fullmatch(key)
        s_match = _S_KEY_RE.fullmatch(key)
        pa_match = _PA_KEY_RE.fullmatch(key)

        # ------------------------------------------------------------------
        # NL_i
        # ------------------------------------------------------------------
        if nl_match:
            idx = int(nl_match.group(1))

            if expecting_s:
                return _failure(
                    f"Expected S{expected_step} immediately after {active_nl_key}, "
                    f"but found '{key}'."
                )

            if idx != expected_step:
                return _failure(
                    f"NL numbering error: expected NL{expected_step}, found {key}."
                )

            nl_error = _validate_nl_value(key, value)
            if nl_error:
                return _failure(nl_error)

            n_nl += 1
            expecting_s = True
            active_nl_key = key
            previous_key = key
            continue

        # ------------------------------------------------------------------
        # S_i
        # ------------------------------------------------------------------
        if s_match:
            idx = int(s_match.group(1))

            if not expecting_s:
                return _failure(
                    f"{key} must be immediately preceded by its matching NL{idx}."
                )

            if idx != expected_step:
                return _failure(
                    f"NL/S index mismatch: expected S{expected_step} after "
                    f"{active_nl_key}, found {key}."
                )

            s_error = _validate_s_value(
                key,
                value,
                n_houses=n_houses,
                allowed_tokens=allowed_tokens,
            )
            if s_error:
                return _failure(s_error)

            n_s += 1
            expecting_s = False
            active_nl_key = None
            expected_step += 1
            previous_key = key
            continue

        # ------------------------------------------------------------------
        # PA_i
        # ------------------------------------------------------------------
        assert pa_match is not None
        idx = int(pa_match.group(1))

        # A PA cannot interrupt NL_i -> S_i.
        if expecting_s:
            return _failure(
                f"{key} cannot appear between {active_nl_key} and its matching "
                f"S{expected_step}."
            )

        # A PA must be immediately after a completed S_i, not at the beginning,
        # not after NL_i, and not consecutively after another PA.
        if previous_key is None or _S_KEY_RE.fullmatch(previous_key) is None:
            return _failure(
                f"{key} must appear immediately after a completed NL_i/S_i pair."
            )

        if idx != expected_pa:
            return _failure(
                f"PA numbering error: expected PA{expected_pa}, found {key}."
            )

        pa_error = _validate_pa(
            key,
            value,
            n_houses=n_houses,
            expected_header=expected_header,
            attribute_values=attribute_values,
        )
        if pa_error:
            return _failure(pa_error)

        # 13) monotonicity across successive valid PAs
        if previous_pa is not None and previous_pa_key is not None:
            monotonicity_error = _validate_pa_monotonicity(
                previous_pa_key,
                previous_pa,
                key,
                value,
                expected_header=expected_header,
            )
            if monotonicity_error:
                return _failure(monotonicity_error)

        previous_pa = value
        previous_pa_key = key
        expected_pa += 1
        n_pa += 1
        previous_key = key

    # No dangling final NL_i is permitted in the new prompt.
    if expecting_s:
        return _failure(
            f"Reasoning ends with {active_nl_key}; matching S{expected_step} is missing."
        )

    if n_nl == 0 or n_s == 0:
        return _failure("reasoning must contain at least one complete NL1/S1 pair.")

    if n_nl != n_s:
        # Defensive check; normally caught earlier by the state machine.
        return _failure(
            f"NL/S count mismatch: found {n_nl} NL steps and {n_s} S steps."
        )

    return _success(n_nl=n_nl, n_s=n_s, n_pa=n_pa)


# -----------------------------------------------------------------------------
# Regression-style examples
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    N_HOUSES = 3
    HEADER = ["House", "Name", "Color", "Children"]
    ATTRIBUTE_VALUES = {
        "Name": ["Peter", "Eric", "Arnold"],
        "Color": ["red", "white", "yellow"],
        "Children": ["Fred", "Meredith", "Bella"],
    }

    # 1) VALID: full NL/S/PA trajectory with two monotonic PAs.
    valid_full = {
        "NL1": "Clues place Arnold in house 2.",
        "S1": "Arnold == 2.",
        "NL2": "Eric cannot occupy house 1.",
        "S2": "Eric != 1.",
        "NL3": "The remaining placement forces Eric into house 3.",
        "S3": "Eric == 3.",
        "PA1": {
            "header": HEADER,
            "rows": [
                ["1", "?", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "Eric", "?", "?"],
            ],
        },
        "NL4": "Peter must therefore occupy house 1.",
        "S4": "Peter == 1.",
        "NL5": "Fred must be in house 1 or house 2.",
        "S5": "Or(Fred == 1, Fred == 2).",
        "NL6": "Child uniqueness then fixes Fred in house 2.",
        "S6": "Fred == 2.",
        "PA2": {
            "header": HEADER,
            "rows": [
                ["1", "Peter", "?", "Bella"],
                ["2", "Arnold", "red", "Fred"],
                ["3", "Eric", "?", "?"],
            ],
        },
    }

    # 2) VALID: PA is optional.
    valid_without_pa = {
        "NL1": "Arnold occupies house 2.",
        "S1": "Arnold == 2.",
        "NL2": "Eric is not in house 1.",
        "S2": "Not(Eric == 1).",
    }

    # 3) INVALID: reasoning contains a disallowed key.
    invalid_key = {
        "NL1": "Arnold occupies house 2.",
        "S1": "Arnold == 2.",
        "STATE1": {},
    }

    # 4) INVALID: NL/S pairing/index mismatch.
    invalid_pairing = {
        "NL1": "Arnold occupies house 2.",
        "S2": "Arnold == 2.",
    }

    # 5) INVALID: PA inserted between NL1 and S1.
    invalid_pa_placement = {
        "NL1": "Arnold occupies house 2.",
        "PA1": {
            "header": HEADER,
            "rows": [
                ["1", "?", "?", "?"],
                ["2", "Arnold", "?", "?"],
                ["3", "?", "?", "?"],
            ],
        },
        "S1": "Arnold == 2.",
    }

    # 6) INVALID: NL value does not end with a period.
    invalid_nl_format = {
        "NL1": "Arnold occupies house 2",
        "S1": "Arnold == 2.",
    }

    # 7) INVALID: unsupported S grammar.
    invalid_s_grammar = {
        "NL1": "A hypothetical implication is attempted.",
        "S1": "Implies(Eric == 1, Eric == 2).",
    }

    # 8) INVALID: syntactically valid S expression uses an unknown entity token.
    invalid_s_domain = {
        "NL1": "An unknown person is placed in house 2.",
        "S1": "John == 2.",
    }

    # 9) INVALID: duplicate resolved value in a PA attribute column.
    invalid_pa_uniqueness = {
        "NL1": "Arnold occupies house 2.",
        "S1": "Arnold == 2.",
        "PA1": {
            "header": HEADER,
            "rows": [
                ["1", "Arnold", "?", "?"],
                ["2", "Arnold", "?", "?"],
                ["3", "?", "?", "?"],
            ],
        },
    }

    # 10) INVALID: PA2 reverts a previously resolved cell to '?'.
    invalid_pa_monotonicity = copy.deepcopy(valid_full)
    invalid_pa_monotonicity["PA2"]["rows"][1][1] = "?"  # Arnold -> ?

    tests = [
        ("1_valid_full", valid_full, 1.0),
        ("2_valid_without_pa", valid_without_pa, 1.0),
        ("3_invalid_key", invalid_key, 0.0),
        ("4_invalid_pairing", invalid_pairing, 0.0),
        ("5_invalid_pa_placement", invalid_pa_placement, 0.0),
        ("6_invalid_nl_format", invalid_nl_format, 0.0),
        ("7_invalid_s_grammar", invalid_s_grammar, 0.0),
        ("8_invalid_s_domain", invalid_s_domain, 0.0),
        ("9_invalid_pa_uniqueness", invalid_pa_uniqueness, 0.0),
        ("10_invalid_pa_monotonicity", invalid_pa_monotonicity, 0.0),
    ]

    passed = 0
    for name, reasoning, expected_reward in tests:
        reward, info = check_interleaved_reasoning(
            reasoning,
            n_houses=N_HOUSES,
            expected_header=HEADER,
            attribute_values=ATTRIBUTE_VALUES,
        )
        test_ok = reward == expected_reward
        passed += int(test_ok)
        print(
            f"{name:30s} -> reward={reward:.1f} "
            f"expected={expected_reward:.1f} "
            f"{'PASS' if test_ok else 'FAIL'}"
        )
        print(f"    {info}")

    print("\n" + "=" * 80)
    print(f"Passed {passed}/{len(tests)} NL/S/PA format-check tests.")
