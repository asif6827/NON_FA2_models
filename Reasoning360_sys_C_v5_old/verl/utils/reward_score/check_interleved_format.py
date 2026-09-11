import re
from typing import List, Union


_STEP_RE = re.compile(r"^\s*S(\d+):\s+(.+?)\s*$")

# A conservative identifier: normalized tokens like Arnold, red, pall_mall, house3, etc.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _split_top_level_args(s: str) -> List[str]:
    """Split a comma-separated argument list, respecting nested parentheses."""
    args: List[str] = []
    buf: List[str] = []
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
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
            if part:
                args.append(part)
            else:
                return []
            buf = []
        else:
            buf.append(ch)
        i += 1

    if depth != 0:
        return []

    tail = "".join(buf).strip()
    if tail:
        args.append(tail)
    return args


def _is_ident(x: str) -> bool:
    return bool(_IDENT_RE.match(x.strip()))


def _is_house_index(x: str, n_houses: int) -> bool:
    if not x.isdigit():
        return False
    h = int(x)
    return 1 <= h <= n_houses


def _parse_atomic(expr: str, n_houses: int) -> bool:
    """
    Allowed atomic forms (whitespace flexible):

      A == B
      A != B
      A < B
      A > B
      A + d == B        (d is positive integer)
      A == H            (H integer in 1..n_houses)

    Notes:
      - For < and >, RHS must be an identifier (not a number).
      - For + d ==, both sides must be identifiers.
    """
    e = expr.strip()

    # Directed distance: A + d == B
    m = re.match(
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*([1-9]\d*)\s*==\s*([A-Za-z_][A-Za-z0-9_]*)$",
        e,
    )
    if m:
        left, d, right = m.group(1), m.group(2), m.group(3)
        return _is_ident(left) and int(d) > 0 and _is_ident(right)

    # Binary operator: A (op) RHS
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<|>)\s*([A-Za-z_][A-Za-z0-9_]*|\d+)$", e)
    if not m:
        return False

    left, op, rhs = m.group(1), m.group(2), m.group(3)
    if not _is_ident(left):
        return False

    if op in ("<", ">"):
        # ordering between two values only
        return _is_ident(rhs)

    if op in ("!=", "=="):
        if rhs.isdigit():
            # numeric RHS allowed only for ==
            return op == "==" and _is_house_index(rhs, n_houses)
        return _is_ident(rhs)

    return False


def _parse_expr(expr: str, n_houses: int) -> bool:
    """
    Allowed boolean forms:
      Not(e)          where e is an expression (atomic or boolean)
      And(e1, e2, ...)
      Or(e1, e2, ...)

    Atomic forms are handled by _parse_atomic().
    """
    e = expr.strip()

    # Not(...)
    if e.startswith("Not(") and e.endswith(")"):
        inner = e[len("Not(") : -1].strip()
        return bool(inner) and _parse_expr(inner, n_houses)

    # And(...)
    if e.startswith("And(") and e.endswith(")"):
        inner = e[len("And(") : -1].strip()
        args = _split_top_level_args(inner)
        return len(args) >= 2 and all(_parse_expr(a, n_houses) for a in args)

    # Or(...)
    if e.startswith("Or(") and e.endswith(")"):
        inner = e[len("Or(") : -1].strip()
        args = _split_top_level_args(inner)
        return len(args) >= 2 and all(_parse_expr(a, n_houses) for a in args)

    # Atomic
    return _parse_atomic(e, n_houses)


def check_interleaved_reasoning(
    reasoning: Union[List[str], None],
    *,
    n_houses: int,
) -> bool:
    """Return True iff `reasoning` satisfies the relaxed interleaving + syntax rules.

    Relaxations applied (per your request):
      - NL lines no longer need to end with a period.
      - The reasoning list may end with an NL line (i.e., odd length is allowed).

    Still enforced:
      1) `reasoning` must be a non-empty list[str]
      2) Interleaving pattern for as long as it exists: NL, S1, NL, S2, ...
         - even indices (0,2,4,...) must be NL (must NOT start with S<k>:)
         - odd indices (1,3,5,...) must be Si and correctly numbered
      3) Si lines must be consecutive starting at S1
      4) Si expressions must parse using ONLY allowed operators:
         - Atomic: ==, !=, <, >, "+ d ==", "== H" where H in 1..n_houses
         - Boolean: Not(e), And(...), Or(...)
    """
    if not isinstance(n_houses, int) or n_houses <= 0:
        return False
    if not isinstance(reasoning, list) or len(reasoning) == 0:
        return False

    expected_k = 1

    for idx, item in enumerate(reasoning):
        if not isinstance(item, str) or not item.strip():
            return False

        is_syntactic_pos = (idx % 2 == 1)  # 0-based: 1,3,5,... are syntactic
        m = _STEP_RE.match(item)

        if is_syntactic_pos:
            # Must be syntactic and correctly numbered
            if not m:
                return False
            k = int(m.group(1))
            expr = m.group(2).strip()

            # Allow an optional trailing period at end of line (common in datasets)
            if expr.endswith("."):
                expr = expr[:-1].strip()

            if k != expected_k:
                return False
            if not _parse_expr(expr, n_houses):
                return False

            expected_k += 1
        else:
            # Must be natural language (must NOT look like "S<number>: ...")
            if m:
                return False

    # NOTE: No longer require ending with a syntactic line.
    return True


if __name__ == "__main__":
    # ✅ VALID (ends with NL is now allowed; NL period is not required)
    reasoning_valid = [
        "I am NL",
        "S1: Or(eric == 1, Eric == 2, Eric == 3).",
        "I am NL.",
        "S2: Not(Eric == 1).",
        "I am NL.",
    ]

    # ❌ INVALID: two NL in a row
    reasoning_two_nl = [
        "The red color is fixed in the second house.",
        "Arnold is tied to the red color.",
        "S1: red == 2.",
    ]

    # ❌ INVALID: non-allowed operator / unparsable Si
    reasoning_bad_si = [
        "We combine two facts to constrain Eric.",
        "S1: Implies(Eric == 1, Eric == 2).",
    ]

    # ❌ INVALID: numeric RHS for < (not allowed)
    reasoning_bad_order_num = [
        "We compare positions with ordering, but use a number on the RHS incorrectly.",
        "S1: Fred < 2.",
    ]

    tests = [
        ("valid", reasoning_valid, 3),
        ("two_nl", reasoning_two_nl, 3),
        ("bad_si", reasoning_bad_si, 3),
        ("bad_order_num", reasoning_bad_order_num, 3),
    ]

    for name, r, n in tests:
        ok = check_interleaved_reasoning(r, n_houses=n)
        print(f"{name:20s} -> reward = {ok}")
