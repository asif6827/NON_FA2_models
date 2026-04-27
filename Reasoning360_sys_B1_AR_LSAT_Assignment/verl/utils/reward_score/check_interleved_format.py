import re
from typing import List, Union

_STEP_RE = re.compile(r"^\s*S(\d+):\s+(.+?)\s*$", re.IGNORECASE)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPTION_STATUS_RE = re.compile(r"^(Sat|Unsat)\(Option_[A-Z]\)$", re.IGNORECASE)


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


def _is_ident(x: str) -> bool:
    return bool(_IDENT_RE.match(x.strip()))


def _is_position_index(x: str, n_houses: int) -> bool:
    if not re.fullmatch(r"-?\d+", x.strip()):
        return False
    h = int(x)
    return 1 <= h <= n_houses


def _parse_atomic(expr: str, n_houses: int) -> bool:
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1].strip()

    if _OPTION_STATUS_RE.match(e):
        return True

    # A + d == B, A - d == B
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*([+-])\s*([1-9]\d*)\s*==\s*([A-Za-z_][A-Za-z0-9_]*)$", e)
    if m:
        return _is_ident(m.group(1)) and _is_ident(m.group(4))

    # A op B / A op integer / integer op A
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*|-?\d+)\s*(==|!=|<=|>=|<|>)\s*([A-Za-z_][A-Za-z0-9_]*|-?\d+)$", e)
    if not m:
        return False
    left, op, right = m.group(1), m.group(2), m.group(3)

    left_is_num = re.fullmatch(r"-?\d+", left) is not None
    right_is_num = re.fullmatch(r"-?\d+", right) is not None

    if left_is_num and right_is_num:
        return True
    if left_is_num:
        return _is_position_index(left, n_houses) and _is_ident(right)
    if right_is_num:
        return _is_ident(left) and _is_position_index(right, n_houses)
    return _is_ident(left) and _is_ident(right)


def _parse_expr(expr: str, n_houses: int) -> bool:
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1].strip()

    # Distinct(A, B, C)
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", e)
    if m:
        fn = m.group(1)
        args = _split_top_level_args(m.group(2))
        fn_l = fn.lower()
        if fn_l in {"and", "or"}:
            return len(args) >= 2 and all(_parse_expr(a, n_houses) for a in args)
        if fn_l == "not":
            return len(args) == 1 and _parse_expr(args[0], n_houses)
        if fn_l in {"implies", "xor"}:
            return len(args) == 2 and all(_parse_expr(a, n_houses) for a in args)
        if fn_l == "distinct":
            return len(args) >= 2 and all(_is_ident(a) for a in args)
        if fn_l in {"sat", "unsat"}:
            return len(args) == 1 and re.fullmatch(r"Option_[A-Z]", args[0].strip()) is not None
        return False

    return _parse_atomic(e, n_houses)


def check_interleaved_reasoning(reasoning: Union[List[str], None], *, n_houses: int) -> bool:
    """Validate AR-LSAT ordering reasoning format.

    Expected pattern: NL, S1, NL, S2, ... . The list may end with NL.
    Formal steps may use ordering expressions, Boolean operators, Distinct,
    and option-status steps Sat(Option_X)/Unsat(Option_X).
    """
    if not isinstance(n_houses, int) or n_houses <= 0:
        return False
    if not isinstance(reasoning, list) or len(reasoning) == 0:
        return False

    expected_k = 1
    for idx, item in enumerate(reasoning):
        if not isinstance(item, str) or not item.strip():
            return False
        m = _STEP_RE.match(item)
        is_syntactic_pos = idx % 2 == 1
        if is_syntactic_pos:
            if not m:
                return False
            k = int(m.group(1))
            expr = m.group(2).strip()
            if expr.endswith("."):
                expr = expr[:-1].strip()
            if k != expected_k:
                return False
            if not _parse_expr(expr, n_houses):
                return False
            expected_k += 1
        else:
            if m:
                return False
    return True


if __name__ == "__main__":
    valid = [
        "The question fixes B in fourth position.",
        "S1: B == 4.",
        "Option A is feasible.",
        "S2: Sat(Option_A).",
    ]
    print(check_interleaved_reasoning(valid, n_houses=4))
