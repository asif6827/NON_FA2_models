# -*- coding: utf-8 -*-
"""Strict interleaved-format checker for AR-LSAT Grouping outputs.

Expected structure:
    natural-language explanation
    S1: formal expression.
    natural-language explanation
    S2: formal expression.
    ...

The public ``check_interleaved_reasoning`` function remains backward compatible
and returns only a Boolean. Use ``check_interleaved_reasoning_detailed`` for
error diagnostics.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

_STEP_RE = re.compile(r"^\s*S\s*(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_ASSIGN_RE = re.compile(
    rf"^Assign\(\s*{_IDENT}\s*,\s*{_IDENT}\s*\)$",
    re.IGNORECASE,
)
_GROUP_REL_RE = re.compile(
    rf"^(SameGroup|DifferentGroup|Together|Apart)\(\s*{_IDENT}\s*,\s*{_IDENT}\s*\)$",
    re.IGNORECASE,
)
_OPTION_LABEL_RE = re.compile(r"^Option[_\-\s:]*([A-E])$", re.IGNORECASE)


def _error(index: Optional[int], code: str, message: str, raw: Any = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"index": index, "code": code, "message": message}
    if raw is not None:
        out["raw"] = raw
    return out


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


def _split_binary_top_level(text: str, operator: str) -> Optional[Tuple[str, str]]:
    depth = 0
    i = 0

    while i <= len(text) - len(operator):
        char = text[i]
        if char == "(":
            depth += 1
            i += 1
            continue
        if char == ")":
            depth -= 1
            if depth < 0:
                return None
            i += 1
            continue
        if depth == 0 and text.startswith(operator, i):
            left = text[:i].strip()
            right = text[i + len(operator):].strip()
            return (left, right) if left and right else None
        i += 1

    return None


def _parse_option_reference(text: str) -> bool:
    return bool(_OPTION_LABEL_RE.fullmatch(text.strip()))


def _parse_option_status(function_name: str, args: List[str]) -> bool:
    if function_name not in {"sat", "unsat"} or len(args) != 1:
        return False

    arg = args[0].strip()
    if _parse_option_reference(arg):
        return True

    match = re.fullmatch(r"Not\((.*)\)", arg, flags=re.IGNORECASE | re.DOTALL)
    return bool(match and _parse_option_reference(match.group(1)))


def _parse_expr(expression: str, n_houses: int = 0) -> bool:
    """Validate the controlled Grouping DSL used by the Z3 validator."""
    del n_houses  # Retained only for API compatibility.

    expr = str(expression).strip().rstrip(".").strip()
    if not expr:
        return False

    if _ASSIGN_RE.fullmatch(expr):
        return True

    if _GROUP_REL_RE.fullmatch(expr):
        return True

    for operator in ("==", "!="):
        parts = _split_binary_top_level(expr, operator)
        if parts is not None:
            return _parse_expr(parts[0]) and _parse_expr(parts[1])

    match = re.fullmatch(
        rf"({_IDENT})\s*\((.*)\)",
        expr,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return False

    function_name = match.group(1).lower()
    args = _split_top_level_args(match.group(2))
    if args is None:
        return False

    if _parse_option_status(function_name, args):
        return True

    if function_name in {"and", "or"}:
        return len(args) >= 2 and all(_parse_expr(arg) for arg in args)

    if function_name == "not":
        return len(args) == 1 and _parse_expr(args[0])

    if function_name in {"implies", "xor"}:
        return len(args) == 2 and all(_parse_expr(arg) for arg in args)

    if function_name in {"atleast", "atmost", "exactly"}:
        if len(args) < 2 or not re.fullmatch(r"\d+", args[0].strip()):
            return False
        count = int(args[0].strip())
        predicates = args[1:]
        if count < 0 or count > len(predicates):
            return False
        return all(_parse_expr(arg) for arg in predicates)

    return False


def check_interleaved_reasoning_detailed(
    reasoning: Union[List[str], None],
    *,
    n_houses: int = 0,
    require_terminal_period: bool = True,
) -> Dict[str, Any]:
    """Return detailed diagnostics for an interleaved Grouping trace."""
    errors: List[Dict[str, Any]] = []

    if not isinstance(reasoning, list):
        return {
            "ok": False,
            "errors": [_error(None, "NOT_A_LIST", "reasoning must be a list of strings.")],
            "n_entries": 0,
            "n_pairs": 0,
        }

    if not reasoning:
        return {
            "ok": False,
            "errors": [_error(None, "EMPTY_REASONING", "reasoning must contain at least one NL/formal pair.")],
            "n_entries": 0,
            "n_pairs": 0,
        }

    if len(reasoning) % 2 != 0:
        errors.append(
            _error(
                len(reasoning) - 1,
                "UNPAIRED_ENTRY",
                "reasoning must contain complete natural-language/formal pairs; its length must be even.",
                reasoning[-1],
            )
        )

    expected_step = 1

    for index, item in enumerate(reasoning):
        if not isinstance(item, str):
            errors.append(_error(index, "NON_STRING_ENTRY", "Each reasoning entry must be a string.", item))
            continue

        text = item.strip()
        if not text:
            errors.append(_error(index, "EMPTY_ENTRY", "Reasoning entries cannot be empty.", item))
            continue

        step_match = _STEP_RE.fullmatch(text)

        if index % 2 == 0:
            if step_match:
                errors.append(
                    _error(
                        index,
                        "FORMAL_IN_NL_SLOT",
                        "Even-indexed entries must be natural-language explanations, not S<i> formal steps.",
                        item,
                    )
                )
            if require_terminal_period and not text.endswith("."):
                errors.append(
                    _error(index, "NL_MISSING_PERIOD", "Natural-language explanations must end with a period.", item)
                )
            continue

        if not step_match:
            errors.append(
                _error(
                    index,
                    "MISSING_STEP_PREFIX",
                    f"Formal entry must begin with S{expected_step}: and contain one expression.",
                    item,
                )
            )
            continue

        step_number = int(step_match.group(1))
        expression = step_match.group(2).strip()

        if step_number != expected_step:
            errors.append(
                _error(
                    index,
                    "NON_CONSECUTIVE_STEP",
                    f"Expected S{expected_step}, but found S{step_number}.",
                    item,
                )
            )

        if require_terminal_period and not text.endswith("."):
            errors.append(_error(index, "FORMAL_MISSING_PERIOD", "Formal steps must end with a period.", item))

        if not _parse_expr(expression, n_houses=n_houses):
            errors.append(
                _error(
                    index,
                    "INVALID_FORMAL_EXPRESSION",
                    "Formal step is not valid under the AR-LSAT Grouping DSL.",
                    expression,
                )
            )

        expected_step += 1

    return {
        "ok": not errors,
        "errors": errors,
        "n_entries": len(reasoning),
        "n_pairs": len(reasoning) // 2,
        "n_formal_expected": len(reasoning) // 2,
    }


def check_interleaved_reasoning(
    reasoning: Union[List[str], None],
    *,
    n_houses: int = 0,
    require_terminal_period: bool = True,
) -> bool:
    """Backward-compatible Boolean wrapper."""
    return bool(
        check_interleaved_reasoning_detailed(
            reasoning,
            n_houses=n_houses,
            require_terminal_period=require_terminal_period,
        )["ok"]
    )


def _demo() -> None:
    demos = {
        "valid_assign": [
            "A and B are fixed in different groups.",
            "S1: And(Assign(A, X), Assign(B, Y)).",
            "Option A is feasible.",
            "S2: Sat(Option_A).",
        ],
        "valid_group_relations": [
            "A and B must be assigned to different groups.",
            "S1: DifferentGroup(A, B).",
            "C and D must be together.",
            "S2: SameGroup(C, D).",
        ],
        "valid_nested": [
            "Exactly one of A, B, and C is assigned to X.",
            "S1: Exactly(1, Assign(A, X), Assign(B, X), Assign(C, X)).",
            "If A is in X, then B and C are apart.",
            "S2: Implies(Assign(A, X), DifferentGroup(B, C)).",
        ],
        "invalid_starts_formal": [
            "S1: Assign(A, X).",
            "A is assigned to X.",
        ],
        "invalid_unpaired": [
            "A is assigned to X.",
            "S1: Assign(A, X).",
            "This explanation has no formal partner.",
        ],
        "invalid_step_number": [
            "A is assigned to X.",
            "S2: Assign(A, X).",
        ],
        "invalid_group_syntax": [
            "A and B are together.",
            "S1: A and B are in the same group.",
        ],
        "invalid_count": [
            "Four of three entities are assigned to X.",
            "S1: Exactly(4, Assign(A, X), Assign(B, X), Assign(C, X)).",
        ],
    }

    for name, reasoning in demos.items():
        result = check_interleaved_reasoning_detailed(reasoning, n_houses=2)
        print(f"\n=== {name} ===")
        print(f"ok={result['ok']}")
        for error in result["errors"]:
            print(f"- {error['code']}: {error['message']}")


if __name__ == "__main__":
    _demo()
