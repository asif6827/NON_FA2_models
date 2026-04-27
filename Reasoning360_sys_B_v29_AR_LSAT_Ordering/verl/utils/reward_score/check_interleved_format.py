import re
from typing import List, Union

_STEP_RE = re.compile(r"^\s*S(\d+):\s+(.+?)\s*$", re.IGNORECASE)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPTION_STATUS_RE = re.compile(r"^(Sat|Unsat)\(\s*(Not\()?\s*Option_[A-Z]\s*\)?\s*\)$", re.IGNORECASE)


def _split_top_level_args(s: str) -> List[str]:
    args, buf, depth = [], [], 0
    for ch in s:
        if ch == '(':
            depth += 1; buf.append(ch)
        elif ch == ')':
            depth -= 1
            if depth < 0: return []
            buf.append(ch)
        elif ch == ',' and depth == 0:
            part = ''.join(buf).strip()
            if not part: return []
            args.append(part); buf = []
        else:
            buf.append(ch)
    if depth != 0: return []
    tail = ''.join(buf).strip()
    if tail: args.append(tail)
    return args


def _is_ident(x: str) -> bool:
    return bool(_IDENT_RE.match(x.strip()))


def _is_position_index(x: str, n_houses: int) -> bool:
    if not re.fullmatch(r"-?\d+", x.strip()):
        return False
    h = int(x)
    return 1 <= h <= n_houses


def _parse_atomic(expr: str, n_houses: int) -> bool:
    e = expr.strip().rstrip('.')
    if _OPTION_STATUS_RE.match(e):
        return True
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*([+-])\s*([1-9]\d*)\s*==\s*([A-Za-z_][A-Za-z0-9_]*)$", e)
    if m:
        return _is_ident(m.group(1)) and _is_ident(m.group(4))
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*|-?\d+)\s*(==|!=|<=|>=|<|>)\s*([A-Za-z_][A-Za-z0-9_]*|-?\d+)$", e)
    if not m:
        return False
    left, _, right = m.group(1), m.group(2), m.group(3)
    left_num = re.fullmatch(r"-?\d+", left) is not None
    right_num = re.fullmatch(r"-?\d+", right) is not None
    if left_num and right_num:
        return True
    if left_num:
        return _is_position_index(left, n_houses) and _is_ident(right)
    if right_num:
        return _is_ident(left) and _is_position_index(right, n_houses)
    return _is_ident(left) and _is_ident(right)


def _parse_expr(expr: str, n_houses: int) -> bool:
    e = expr.strip().rstrip('.')
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", e)
    if m:
        fn = m.group(1).lower()
        args = _split_top_level_args(m.group(2))
        if fn in {'and', 'or'}:
            return len(args) >= 2 and all(_parse_expr(a, n_houses) for a in args)
        if fn == 'not':
            return len(args) == 1 and _parse_expr(args[0], n_houses)
        if fn in {'implies', 'xor'}:
            return len(args) == 2 and all(_parse_expr(a, n_houses) for a in args)
        if fn == 'distinct':
            return len(args) >= 2 and all(_is_ident(a) for a in args)
        if fn in {'sat', 'unsat'}:
            if len(args) != 1:
                return False
            a = args[0].strip()
            if re.fullmatch(r"Option_[A-Z]", a):
                return True
            return re.fullmatch(r"Not\(Option_[A-Z]\)", a) is not None
        return False
    return _parse_atomic(e, n_houses)


def check_interleaved_reasoning(reasoning: Union[List[str], None], *, n_houses: int) -> bool:
    if not isinstance(n_houses, int) or n_houses <= 0:
        return False
    if not isinstance(reasoning, list) or not reasoning:
        return False
    expected_k = 1
    for idx, item in enumerate(reasoning):
        if not isinstance(item, str) or not item.strip():
            return False
        m = _STEP_RE.match(item)
        if idx % 2 == 1:
            if not m:
                return False
            k = int(m.group(1))
            expr = m.group(2).strip().rstrip('.')
            if k != expected_k or not _parse_expr(expr, n_houses):
                return False
            expected_k += 1
        else:
            if m:
                return False
    return True


if __name__ == '__main__':
    valid = ['B is fourth.', 'S1: B == 4.', 'Option A is feasible.', 'S2: Sat(Option_A).']
    print(check_interleaved_reasoning(valid, n_houses=4))
