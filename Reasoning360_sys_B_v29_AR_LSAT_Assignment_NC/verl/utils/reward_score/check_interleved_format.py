import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Keep the historical filename/API for drop-in compatibility.
_STEP_RE = re.compile(r"^\s*S(?P<sid>\d+)\s*:\s*(?P<expr>.+?)\s*$", re.IGNORECASE)
_STEP_PREFIX_RE = re.compile(r"^\s*S\d+\s*:", re.IGNORECASE)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASSIGN_RE = re.compile(
    r"^Assign\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*\)$",
    re.IGNORECASE,
)
_OPTION_REF_RE = re.compile(r"^(?:Option[_\-\s:]*)?([A-E])$", re.IGNORECASE)


def _split_top_level_args(s: str) -> Optional[List[str]]:
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
                return None
            buf.append(ch)
        elif ch == "," and depth == 0:
            part = "".join(buf).strip()
            if not part:
                return None
            args.append(part)
            buf = []
        else:
            buf.append(ch)

    if depth != 0:
        return None

    tail = "".join(buf).strip()
    if tail:
        args.append(tail)
    elif buf or s.rstrip().endswith(","):
        return None

    return args


def _split_binary_top_level(s: str, op: str) -> Optional[Tuple[str, str]]:
    depth = 0
    i = 0

    while i <= len(s) - len(op):
        ch = s[i]
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth -= 1
            if depth < 0:
                return None
            i += 1
            continue
        if depth == 0 and s.startswith(op, i):
            left = s[:i].strip()
            right = s[i + len(op):].strip()
            return (left, right) if left and right else None
        i += 1

    return None


def _is_option_ref(expr: str) -> bool:
    return _OPTION_REF_RE.fullmatch(expr.strip()) is not None


def _parse_expr(expr: str, n_houses: int = 0) -> bool:
    """Validate the controlled AR-LSAT Assignment expression grammar.

    This is a syntax/format check only. Semantic validation remains the job of
    the Z3 validator.
    """
    del n_houses  # Retained only for compatibility with existing callers.

    e = str(expr).strip()
    if not e:
        return False

    if e.endswith("."):
        e = e[:-1].rstrip()
    if not e:
        return False

    if e.lower() in {"true", "false"}:
        return True
    if _ASSIGN_RE.fullmatch(e):
        return True

    # Boolean equivalence/inequality between complete Boolean expressions.
    for op in ("==", "!="):
        parts = _split_binary_top_level(e, op)
        if parts is not None:
            return _parse_expr(parts[0]) and _parse_expr(parts[1])

    m = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", e, flags=re.DOTALL)
    if not m:
        return False

    fn = m.group(1).lower()
    args = _split_top_level_args(m.group(2))
    if args is None:
        return False

    if fn in {"and", "or"}:
        return len(args) >= 2 and all(_parse_expr(arg) for arg in args)

    if fn == "not":
        return len(args) == 1 and _parse_expr(args[0])

    if fn in {"implies", "xor"}:
        return len(args) == 2 and all(_parse_expr(arg) for arg in args)

    if fn in {"atleast", "atmost", "exactly"}:
        if len(args) < 2 or not re.fullmatch(r"\d+", args[0].strip()):
            return False
        k = int(args[0].strip())
        bool_args = args[1:]
        return 0 <= k <= len(bool_args) and all(_parse_expr(arg) for arg in bool_args)

    if fn in {"sat", "unsat"}:
        if len(args) != 1:
            return False
        inner = args[0].strip()
        if _is_option_ref(inner):
            return True
        not_match = re.fullmatch(r"Not\((.*)\)", inner, flags=re.IGNORECASE | re.DOTALL)
        return bool(not_match and _is_option_ref(not_match.group(1)))

    return False


def check_interleaved_reasoning_detailed(
    reasoning: Union[List[str], None],
    *,
    n_houses: int = 0,
    require_terminal_period: bool = True,
) -> Dict[str, Any]:
    """Return detailed diagnostics for the NL/formal alternation.

    Required layout:
        index 0: one natural-language sentence
        index 1: S1: <formal expression>.
        index 2: one natural-language sentence
        index 3: S2: <formal expression>.
        ...

    The list must therefore contain a positive, even number of entries.
    """
    errors: List[Dict[str, Any]] = []

    if not isinstance(reasoning, list):
        return {
            "ok": False,
            "errors": [{"index": None, "code": "NOT_A_LIST", "message": "reasoning must be a list[str]."}],
            "n_entries": 0,
            "n_pairs": 0,
        }

    if not reasoning:
        return {
            "ok": False,
            "errors": [{"index": None, "code": "EMPTY", "message": "reasoning must contain at least one NL/formal pair."}],
            "n_entries": 0,
            "n_pairs": 0,
        }

    if len(reasoning) % 2 != 0:
        errors.append({
            "index": len(reasoning) - 1,
            "code": "UNPAIRED_ENTRY",
            "message": "reasoning must contain complete NL/formal pairs; its length must be even.",
        })

    expected_step = 1

    for idx, item in enumerate(reasoning):
        role = "natural_language" if idx % 2 == 0 else "formal"

        if not isinstance(item, str) or not item.strip():
            errors.append({
                "index": idx,
                "role": role,
                "code": "EMPTY_OR_NON_STRING",
                "message": "Each reasoning entry must be a non-empty string.",
            })
            continue

        text = item.strip()
        step_match = _STEP_RE.fullmatch(text)

        if idx % 2 == 0:
            # Reject every S<number>: prefix, including malformed formal lines.
            if _STEP_PREFIX_RE.match(text):
                errors.append({
                    "index": idx,
                    "role": role,
                    "code": "FORMAL_IN_NL_SLOT",
                    "message": "Even-indexed entries must be natural language, not S<number>: formal steps.",
                })
            elif require_terminal_period and text[-1] not in ".!?":
                errors.append({
                    "index": idx,
                    "role": role,
                    "code": "NL_PUNCTUATION",
                    "message": "Natural-language entries must end with sentence punctuation.",
                })
            continue

        # Formal slot.
        if step_match is None:
            errors.append({
                "index": idx,
                "role": role,
                "code": "MISSING_STEP_PREFIX",
                "message": f"Expected S{expected_step}: <expression>.",
            })
            expected_step += 1
            continue

        step_number = int(step_match.group("sid"))
        raw_expr = step_match.group("expr").strip()

        if step_number != expected_step:
            errors.append({
                "index": idx,
                "role": role,
                "code": "NON_CONSECUTIVE_STEP",
                "message": f"Expected S{expected_step}, found S{step_number}.",
            })

        if require_terminal_period and not text.endswith("."):
            errors.append({
                "index": idx,
                "role": role,
                "code": "FORMAL_PUNCTUATION",
                "message": "Formal steps must end with a period.",
            })

        expr = raw_expr[:-1].rstrip() if raw_expr.endswith(".") else raw_expr
        if not _parse_expr(expr, n_houses):
            errors.append({
                "index": idx,
                "role": role,
                "code": "INVALID_FORMAL_EXPRESSION",
                "message": f"Unsupported or malformed formal expression: {expr!r}.",
            })

        expected_step += 1

    return {
        "ok": not errors,
        "errors": errors,
        "n_entries": len(reasoning),
        "n_pairs": len(reasoning) // 2,
    }


def check_interleaved_reasoning(
    reasoning: Union[List[str], None],
    *,
    n_houses: int = 0,
    require_terminal_period: bool = True,
) -> bool:
    """Backward-compatible Boolean wrapper used by reward scripts."""
    return bool(
        check_interleaved_reasoning_detailed(
            reasoning,
            n_houses=n_houses,
            require_terminal_period=require_terminal_period,
        )["ok"]
    )


def _run_demo() -> None:
    demos = {
        "valid_basic": [
            "A is not assigned to P1.",
            "S1: Not(Assign(A, P1)).",
            "Option A is feasible.",
            "S2: Sat(Option_A).",
        ],
        "valid_nested": [
            "Exactly one entity is assigned to P2.",
            "S1: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2)).",
            "B and C have the same P1-membership status.",
            "S2: Assign(B, P1) == Assign(C, P1).",
            "The negation of option A is impossible.",
            "S3: Unsat(Not(Option_A)).",
        ],
        "invalid_starts_with_formal": [
            "S1: Assign(A, P1).",
            "A is assigned to P1.",
        ],
        "invalid_odd_length": [
            "A is assigned to P1.",
            "S1: Assign(A, P1).",
            "This explanation has no paired formal step.",
        ],
        "invalid_step_number": [
            "A is assigned to P1.",
            "S2: Assign(A, P1).",
        ],
        "invalid_expression": [
            "A is assigned to P1.",
            "S1: A belongs to P1.",
        ],
        "invalid_missing_period": [
            "A is assigned to P1",
            "S1: Assign(A, P1)",
        ],
    }

    for name, reasoning in demos.items():
        result = check_interleaved_reasoning_detailed(reasoning)
        print(f"\n=== {name} ===")
        print(f"ok={result['ok']} pairs={result['n_pairs']}")
        for error in result["errors"]:
            print(f"- index={error.get('index')} code={error['code']}: {error['message']}")


if __name__ == "__main__":
    _run_demo()
