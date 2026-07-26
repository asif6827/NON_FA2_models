import re
from typing import Any, Dict, List, Optional, Tuple, Union

_STEP_RE = re.compile(r"^\s*S(\d+)\s*:\s+(.+?)\s*$", re.IGNORECASE | re.DOTALL)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INT_RE = re.compile(r"^-?\d+$")
_FUNC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_OPTION_RE = re.compile(r"^Option[_\-\s:]*([A-E])$", re.IGNORECASE)


def _split_top_level_args(text: str) -> Optional[List[str]]:
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
                return None
            buffer.append(char)
        elif char == "," and depth == 0:
            part = "".join(buffer).strip()
            if not part:
                return None
            args.append(part)
            buffer = []
        else:
            buffer.append(char)

    if depth != 0:
        return None

    tail = "".join(buffer).strip()
    if tail:
        args.append(tail)
    elif text.strip():
        return None

    return args


def _split_top_level_binary(text: str, operators: Tuple[str, ...]) -> Optional[Tuple[str, str, str]]:
    depth = 0
    index = 0
    ordered_ops = sorted(operators, key=len, reverse=True)

    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                return None
            index += 1
            continue

        if depth == 0:
            for operator in ordered_ops:
                if text.startswith(operator, index):
                    left = text[:index].strip()
                    right = text[index + len(operator):].strip()
                    if left and right:
                        return left, operator, right
        index += 1

    return None


def _is_identifier(value: str) -> bool:
    return bool(_IDENT_RE.fullmatch(value.strip()))


def _is_position_index(value: str, n_houses: int) -> bool:
    if not _INT_RE.fullmatch(value.strip()):
        return False
    number = int(value)
    return n_houses <= 0 or 1 <= number <= n_houses


def _parse_position_term(term: str, n_houses: int) -> bool:
    """Validate an integer-valued ordering term."""
    value = term.strip()

    if _INT_RE.fullmatch(value):
        return _is_position_index(value, n_houses)

    if _is_identifier(value):
        return True

    match = _FUNC_RE.fullmatch(value)
    if match and match.group(1).lower() == "position":
        args = _split_top_level_args(match.group(2))
        return args is not None and len(args) == 1 and _is_identifier(args[0])

    arithmetic = _split_top_level_binary(value, ("+", "-"))
    if arithmetic is not None:
        left, operator, right = arithmetic
        del operator
        # Offset arithmetic must combine one position term and one integer.
        return (
            _parse_position_term(left, n_houses)
            and bool(_INT_RE.fullmatch(right.strip()))
        )

    return False


def _parse_option_reference(value: str) -> bool:
    return bool(_OPTION_RE.fullmatch(value.strip()))


def _parse_expr(expr: str, n_houses: int = 0) -> bool:
    value = str(expr).strip().rstrip(".").strip()
    if not value:
        return False

    comparison = _split_top_level_binary(value, ("==", "!=", "<=", ">=", "<", ">"))
    if comparison is not None:
        left, _, right = comparison
        return _parse_position_term(left, n_houses) and _parse_position_term(right, n_houses)

    match = _FUNC_RE.fullmatch(value)
    if not match:
        return False

    function = match.group(1).lower()
    args = _split_top_level_args(match.group(2))
    if args is None:
        return False

    if function in {"and", "or"}:
        return len(args) >= 2 and all(_parse_expr(arg, n_houses) for arg in args)

    if function == "not":
        return len(args) == 1 and _parse_expr(args[0], n_houses)

    if function in {"implies", "xor"}:
        return len(args) == 2 and all(_parse_expr(arg, n_houses) for arg in args)

    if function == "distinct":
        return len(args) >= 2 and all(
            _parse_position_term(arg, n_houses) and not _INT_RE.fullmatch(arg.strip())
            for arg in args
        )

    if function in {"atleast", "atmost", "exactly"}:
        if len(args) < 2 or not re.fullmatch(r"\d+", args[0].strip()):
            return False
        count = int(args[0])
        formulas = args[1:]
        if count > len(formulas):
            return False
        return all(_parse_expr(arg, n_houses) for arg in formulas)

    if function in {"sat", "unsat"}:
        if len(args) != 1:
            return False
        argument = args[0].strip()
        if _parse_option_reference(argument):
            return True
        nested = _FUNC_RE.fullmatch(argument)
        return bool(
            nested
            and nested.group(1).lower() == "not"
            and (nested_args := _split_top_level_args(nested.group(2))) is not None
            and len(nested_args) == 1
            and _parse_option_reference(nested_args[0])
        )

    # Ordering predicates.
    if function in {"before", "after", "immediatelybefore", "immediatelyafter", "adjacent"}:
        return len(args) == 2 and all(_is_identifier(arg) for arg in args)

    if function == "between":
        return len(args) == 3 and all(_is_identifier(arg) for arg in args)

    if function == "distance":
        return (
            len(args) == 3
            and _is_identifier(args[0])
            and _is_identifier(args[1])
            and bool(re.fullmatch(r"\d+", args[2].strip()))
            and int(args[2]) >= 0
        )

    if function in {"first", "last"}:
        return len(args) == 1 and _is_identifier(args[0])

    if function == "atposition":
        return (
            len(args) == 2
            and _is_identifier(args[0])
            and _is_position_index(args[1], n_houses)
        )

    return False


def check_interleaved_reasoning_detailed(
    reasoning: Union[List[str], None],
    *,
    n_houses: int = 0,
    require_terminal_period: bool = True,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []

    if not isinstance(n_houses, int) or n_houses < 0:
        errors.append({
            "index": None,
            "code": "INVALID_DOMAIN_SIZE",
            "message": "n_houses must be a non-negative integer.",
        })

    if not isinstance(reasoning, list):
        errors.append({
            "index": None,
            "code": "NOT_A_LIST",
            "message": "reasoning must be a list of strings.",
        })
        return {"ok": False, "errors": errors, "n_entries": 0, "n_pairs": 0}

    if not reasoning:
        errors.append({
            "index": None,
            "code": "EMPTY_REASONING",
            "message": "reasoning must contain at least one NL/formal pair.",
        })
        return {"ok": False, "errors": errors, "n_entries": 0, "n_pairs": 0}

    if len(reasoning) % 2 != 0:
        errors.append({
            "index": len(reasoning) - 1,
            "code": "UNPAIRED_ENTRY",
            "message": "reasoning must contain complete natural-language/formal pairs; its length must be even.",
        })

    expected_step = 1
    for index, item in enumerate(reasoning):
        if not isinstance(item, str) or not item.strip():
            errors.append({
                "index": index,
                "code": "EMPTY_OR_NON_STRING",
                "message": "every reasoning entry must be a non-empty string.",
            })
            continue

        text = item.strip()
        step_match = _STEP_RE.fullmatch(text)

        if require_terminal_period and not text.endswith("."):
            errors.append({
                "index": index,
                "code": "MISSING_TERMINAL_PERIOD",
                "message": "every reasoning entry must end with a period.",
            })

        if index % 2 == 0:
            if step_match:
                errors.append({
                    "index": index,
                    "code": "FORMAL_IN_NL_SLOT",
                    "message": "even-indexed entries must be natural-language explanations, not S-prefixed formal steps.",
                })
            continue

        if not step_match:
            errors.append({
                "index": index,
                "code": "MISSING_STEP_PREFIX",
                "message": f"formal entry must begin with S{expected_step}:.",
            })
            continue

        step_number = int(step_match.group(1))
        expression = step_match.group(2).strip().rstrip(".").strip()

        if step_number != expected_step:
            errors.append({
                "index": index,
                "code": "NON_CONSECUTIVE_STEP",
                "message": f"expected S{expected_step}, but found S{step_number}.",
            })

        if not _parse_expr(expression, n_houses):
            errors.append({
                "index": index,
                "code": "INVALID_FORMAL_EXPRESSION",
                "message": f"unsupported or malformed Ordering expression: {expression!r}.",
            })

        expected_step += 1

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_entries": len(reasoning),
        "n_pairs": len(reasoning) // 2,
    }


def check_interleaved_reasoning(
    reasoning: Union[List[str], None],
    *,
    n_houses: int = 0,
) -> bool:
    """Backward-compatible Boolean interface used by the reward scorer."""
    return bool(check_interleaved_reasoning_detailed(
        reasoning,
        n_houses=n_houses,
        require_terminal_period=True,
    )["ok"])


def _run_demo() -> None:
    demos = {
        "valid_predicates": [
            "B is fixed in fourth position.",
            "S1: AtPosition(B, 4).",
            "A immediately precedes C.",
            "S2: ImmediatelyBefore(A, C).",
            "A is before B and C is adjacent to D.",
            "S3: And(Before(A, B), Adjacent(C, D)).",
            "Option A is satisfiable.",
            "S4: Sat(Option_A).",
        ],
        "valid_arithmetic": [
            "C occurs immediately after A.",
            "S1: Position(C) == Position(A) + 1.",
            "A and B are two positions apart.",
            "S2: Distance(A, B, 2).",
            "B lies between A and C.",
            "S3: Between(A, B, C).",
        ],
        "invalid_starts_formal": [
            "S1: AtPosition(B, 4).",
            "B is fixed in fourth position.",
        ],
        "invalid_unpaired": [
            "B is fixed in fourth position.",
            "S1: AtPosition(B, 4).",
            "This explanation has no corresponding formal step.",
        ],
        "invalid_step_number": [
            "B is fixed in fourth position.",
            "S2: AtPosition(B, 4).",
        ],
        "invalid_position": [
            "B is outside the declared position domain.",
            "S1: AtPosition(B, 5).",
        ],
        "invalid_natural_language_formula": [
            "A precedes B.",
            "S1: A is before B.",
        ],
    }

    for name, reasoning in demos.items():
        result = check_interleaved_reasoning_detailed(reasoning, n_houses=4)
        codes = [error["code"] for error in result["errors"]]
        print(f"{name:32s} ok={result['ok']!s:5s} errors={codes}")


if __name__ == "__main__":
    _run_demo()
