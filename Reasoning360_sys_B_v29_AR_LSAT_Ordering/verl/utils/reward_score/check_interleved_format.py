import re
from typing import List, Union

_STEP_RE = re.compile(r"^\s*S(\d+):\s+(.+?)\s*$", re.IGNORECASE)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def _is_index(x: str, n_houses: int) -> bool:
    return x.strip().isdigit() and 1 <= int(x.strip()) <= int(n_houses)


def _parse_atomic(expr: str, n_houses: int) -> bool:
    e = expr.strip()

    # Sat(Option_A), Unsat(Option_B)
    if re.match(r"^(Sat|Unsat)\(Option_[A-Z]\)$", e, re.IGNORECASE):
        return True

    # A + d == B or A - d == B
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*([+\-])\s*([1-9]\d*)\s*==\s*([A-Za-z_][A-Za-z0-9_]*)$", e)
    if m:
        return _is_ident(m.group(1)) and _is_ident(m.group(4))

    # Distinct(A, B, ...)
    if e.startswith("Distinct(") and e.endswith(")"):
        args = _split_top_level_args(e[len("Distinct("):-1])
        return len(args) >= 2 and all(_is_ident(a) for a in args)

    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<|>|<=|>=)\s*([A-Za-z_][A-Za-z0-9_]*|\d+)$", e)
    if not m:
        return False
    left, op, rhs = m.group(1), m.group(2), m.group(3)
    if not _is_ident(left):
        return False
    if rhs.isdigit():
        return op in ("==", "!=", "<", ">", "<=", ">=") and _is_index(rhs, n_houses)
    return _is_ident(rhs)


def _parse_expr(expr: str, n_houses: int) -> bool:
    e = expr.strip()
    for fn in ("Not", "And", "Or", "Implies", "Xor"):
        prefix = fn + "("
        if e.startswith(prefix) and e.endswith(")"):
            inner = e[len(prefix):-1].strip()
            args = _split_top_level_args(inner)
            if fn == "Not":
                return len(args) == 1 and _parse_expr(args[0], n_houses)
            if fn in ("And", "Or"):
                return len(args) >= 2 and all(_parse_expr(a, n_houses) for a in args)
            if fn in ("Implies", "Xor"):
                return len(args) == 2 and all(_parse_expr(a, n_houses) for a in args)
    return _parse_atomic(e, n_houses)


def check_interleaved_reasoning(reasoning: Union[List[str], None], *, n_houses: int) -> bool:
    """Validate AR-LSAT ordering reasoning format: NL, S1, NL, S2, ... ."""
    if not isinstance(n_houses, int) or n_houses <= 0:
        return False
    if not isinstance(reasoning, list) or len(reasoning) == 0:
        return False

    expected_k = 1
    for idx, item in enumerate(reasoning):
        if not isinstance(item, str) or not item.strip():
            return False
        is_syntactic_pos = idx % 2 == 1
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
    r = ["B is fourth.", "S1: B == 4.", "Option A is feasible.", "S2: Sat(Option_A)."]
    print(check_interleaved_reasoning(r, n_houses=4))
