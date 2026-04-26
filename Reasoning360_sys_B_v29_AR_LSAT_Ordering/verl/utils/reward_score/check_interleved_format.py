import re
from typing import List, Union

_STEP_RE = re.compile(r"^\s*S(\d+):\s+(.+?)\s*$", re.IGNORECASE)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPTION_STATUS_RE = re.compile(r"^(Sat|Unsat)\(Option_([A-Z])\)$", re.IGNORECASE)


def _split_top_level_args(s: str) -> List[str]:
    args, buf, depth = [], [], 0
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


def _is_pos_index(x: str, n_houses: int) -> bool:
    if not re.fullmatch(r"-?\d+", x.strip()):
        return False
    h = int(x)
    return 1 <= h <= n_houses


def _parse_atomic(expr: str, n_houses: int) -> bool:
    e = expr.strip()

    if _OPTION_STATUS_RE.match(e):
        return True

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*([+-])\s*([1-9]\d*)\s*==\s*([A-Za-z_][A-Za-z0-9_]*)$", e)
    if m:
        return _is_ident(m.group(1)) and _is_ident(m.group(4))

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<|>|<=|>=)\s*([A-Za-z_][A-Za-z0-9_]*|-?\d+)$", e)
    if not m:
        return False
    left, op, rhs = m.group(1), m.group(2), m.group(3)
    if not _is_ident(left):
        return False
    if rhs.lstrip("-").isdigit():
        return op in ("==", "!=", "<", ">", "<=", ">=") and _is_pos_index(rhs, n_houses)
    return _is_ident(rhs)


def _parse_expr(expr: str, n_houses: int) -> bool:
    e = expr.strip()
    if e.endswith("."):
        e = e[:-1].strip()

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", e)
    if m:
        fn = m.group(1)
        inner = m.group(2).strip()
        args = _split_top_level_args(inner)
        fn_l = fn.lower()
        if fn_l == "not":
            return len(args) == 1 and _parse_expr(args[0], n_houses)
        if fn_l in ("and", "or"):
            return len(args) >= 2 and all(_parse_expr(a, n_houses) for a in args)
        if fn_l in ("implies", "xor"):
            return len(args) == 2 and all(_parse_expr(a, n_houses) for a in args)
        if fn_l == "distinct":
            return len(args) >= 2 and all(_is_ident(a) for a in args)
        if fn_l in ("sat", "unsat"):
            return len(args) == 1 and re.fullmatch(r"Option_[A-Z]", args[0]) is not None
        return False

    return _parse_atomic(e, n_houses)


def check_interleaved_reasoning(reasoning: Union[List[str], None], *, n_houses: int) -> bool:
    if not isinstance(n_houses, int) or n_houses <= 0:
        return False
    if not isinstance(reasoning, list) or len(reasoning) == 0:
        return False

    expected_k = 1
    for idx, item in enumerate(reasoning):
        if not isinstance(item, str) or not item.strip():
            return False
        is_syntactic_pos = (idx % 2 == 1)
        m = _STEP_RE.match(item.strip())
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
    examples = [
        ["B is fourth.", "S1: B == 4.", "A is before B.", "S2: A < B.", "Option A works.", "S3: Sat(Option_A)."],
        ["Bad.", "S1: Foo(Eric)."],
    ]
    for e in examples:
        print(check_interleaved_reasoning(e, n_houses=4))
