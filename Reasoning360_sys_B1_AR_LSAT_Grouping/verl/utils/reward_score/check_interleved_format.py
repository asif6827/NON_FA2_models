import re
from typing import List, Union

_STEP_RE = re.compile(r"^\s*S(\d+):\s+(.+?)\s*$", re.IGNORECASE)
_OPTION_STATUS_RE = re.compile(r"^(Sat|Unsat)\(\s*(Not\()?\s*Option_[A-Z]\s*\)?\s*\)$", re.IGNORECASE)
_ASSIGN_RE = re.compile(r"^Assign\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*\s*\)$", re.IGNORECASE)

def _split_top_level_args(s: str):
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

def _split_binary_top_level(s: str, op: str):
    depth = 0
    i = 0
    while i <= len(s)-len(op):
        ch = s[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and s[i:i+len(op)] == op:
            return s[:i].strip(), s[i+len(op):].strip()
        i += 1
    return None

def _parse_expr(expr: str, n_houses: int = 0) -> bool:
    e = expr.strip().rstrip('.')
    if _OPTION_STATUS_RE.match(e): return True
    if _ASSIGN_RE.match(e): return True
    for op in ('==', '!='):
        parts = _split_binary_top_level(e, op)
        if parts is not None:
            return _parse_expr(parts[0], n_houses) and _parse_expr(parts[1], n_houses)
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$", e)
    if not m: return False
    fn = m.group(1).lower(); args = _split_top_level_args(m.group(2))
    if fn in {'and','or'}: return len(args) >= 2 and all(_parse_expr(a, n_houses) for a in args)
    if fn == 'not': return len(args) == 1 and _parse_expr(args[0], n_houses)
    if fn in {'implies','xor'}: return len(args) == 2 and all(_parse_expr(a, n_houses) for a in args)
    if fn in {'atleast','atmost','exactly'}: return len(args) >= 2 and args[0].strip().isdigit() and all(_parse_expr(a, n_houses) for a in args[1:])
    if fn in {'sat','unsat'}:
        return len(args) == 1 and (re.fullmatch(r"Option_[A-Z]", args[0].strip()) is not None or re.fullmatch(r"Not\(Option_[A-Z]\)", args[0].strip()) is not None)
    return False

def check_interleaved_reasoning(reasoning: Union[List[str], None], *, n_houses: int = 0) -> bool:
    if not isinstance(reasoning, list) or not reasoning: return False
    expected_k = 1
    for idx, item in enumerate(reasoning):
        if not isinstance(item, str) or not item.strip(): return False
        m = _STEP_RE.match(item)
        if idx % 2 == 1:
            if not m: return False
            k = int(m.group(1)); expr = m.group(2).strip().rstrip('.')
            if k != expected_k or not _parse_expr(expr, n_houses): return False
            expected_k += 1
        else:
            if m: return False
    return True

if __name__ == '__main__':
    valid = ['D and F are fixed in X.', 'S1: And(Assign(D, X), Assign(F, X)).', 'Option D is feasible.', 'S2: Sat(Option_D).']
    print(check_interleaved_reasoning(valid))
