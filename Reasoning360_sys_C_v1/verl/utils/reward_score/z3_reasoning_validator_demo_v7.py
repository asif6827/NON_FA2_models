# -*- coding: utf-8 -*-
"""
z3_reasoning_validator_demo_v7.py

A tolerant Z3 validator for Zebra/Logic puzzles.

v7 Improvements focused on reducing base_sat=False:
- Conflict-tolerant clue adding (greedy Max-SAT style): skip only the clue that makes base UNSAT.
- Auto-detect 0-based house indexing and shift (+1) when patterns match.
- Treat numeric-only tokens as VALUES (not house constants) if they exist in the domain.
- Track additional skip reasons:
    - clue_skipped_conflict
    - clue_skipped_out_of_range
- Expose base_sat_raw (no conflict filtering) vs base_sat (after filtering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Abs, And, BoolRef, BoolVal, Distinct, Int, IntVal, Not, Solver, sat, unsat, unknown

import re
import json
import os
import sys


# ----------------------------- Normalization helpers -----------------------------

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}

def _norm_token(s: str) -> str:
    s = (s or "").strip()
    if (len(s) >= 2) and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1].strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def _as_int_maybe(s: str) -> Optional[int]:
    t = _norm_token(s)
    if t in _WORD_NUM:
        return _WORD_NUM[t]
    t2 = re.sub(r"[^\d\-]", "", t)
    if t2 == "" or t2 == "-":
        return None
    try:
        return int(t2)
    except Exception:
        return None

def _best_match(key: str, candidates: List[str]) -> Optional[str]:
    nk = _norm_token(key)
    for c in candidates:
        if _norm_token(c) == nk:
            return c
    return None

_ATTR_ALIASES = {
    "person": "Name",
    "people": "Name",
    "name": "Name",
    "job": "Occupation",
    "occupation": "Occupation",
    "cigar": "Smoke",
    "smoke": "Smoke",
    "drink": "Drink",
    "beverage": "Drink",
    "pet": "Pet",
    "animal": "Animal",
    "hair": "HairColor",
    "haircolor": "HairColor",
}

# ----------------------------- DSL parsing ---------------------------------

@dataclass(frozen=True)
class Term:
    kind: str                 # "attrval" | "house_const"
    attr: Optional[str] = None
    val: Optional[str] = None
    house: Optional[int] = None

@dataclass
class ParsedStmt:
    op: str
    args: Tuple[Any, ...]
    raw: str

_STMT_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)\s*\.?\s*$")


def _strip_prefixes(line: str) -> str:
    s = (line or "").strip()
    m = re.search(r"\]\s*(.*)$", s)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"=\s*(.*)$", s)
    if m2:
        return m2.group(1).strip()
    return s

def _split_args(arg_str: str) -> List[str]:
    if arg_str is None:
        return []
    parts = [p.strip() for p in arg_str.split(",")]
    return [p for p in parts if p != ""]

def _parse_kv(term: str) -> Tuple[str, str]:
    t = term.strip()
    if "=" not in t:
        raise ValueError(f"Expected Key=Value, got: {term!r}")
    k, v = t.split("=", 1)
    return k.strip(), v.strip()

def _infer_attr_from_value(val: str, attribute_values: Dict[str, List[str]]) -> Optional[str]:
    nv = _norm_token(val)
    for a, vals in attribute_values.items():
        for v in vals:
            if _norm_token(v) == nv:
                return a
    return None

def _coerce_attr(attr: str, attribute_values: Dict[str, List[str]], val_hint: Optional[str] = None) -> str:
    a_norm = _norm_token(attr)
    if a_norm in _ATTR_ALIASES:
        attr = _ATTR_ALIASES[a_norm]

    if attr in attribute_values:
        return attr

    m = _best_match(attr, list(attribute_values.keys()))
    if m is not None:
        return m

    if val_hint is not None:
        inferred = _infer_attr_from_value(val_hint, attribute_values)
        if inferred is not None:
            return inferred

    return attr

def _coerce_value(attr: str, val: str, attribute_values: Dict[str, List[str]]) -> str:
    if attr not in attribute_values:
        return val
    m = _best_match(val, attribute_values[attr])
    return m if m is not None else val

def _parse_term(tok: str, attribute_values: Dict[str, List[str]]) -> Term:
    """
    Parse a single term into:
      - House constant: House=1, 3, one, second, etc.
      - Attr=Val: Name=Eric
      - Value-only: Eric (infer its attribute from domain; commonly Name)

    v7 fix:
      If tok is numeric-only AND that numeric string exists as a domain VALUE,
      treat it as a value (attrval) rather than house_const.
    """
    t = tok.strip()
    if t == "":
        raise ValueError("Empty term")

    # numeric-only token?
    i = _as_int_maybe(t)
    if i is not None and ("=" not in t):
        # If "3" is a legitimate VALUE somewhere, treat as value (not house)
        inferred_attr = _infer_attr_from_value(t, attribute_values)
        if inferred_attr is not None:
            return Term(kind="attrval", attr=inferred_attr, val=t)
        return Term(kind="house_const", house=i)

    # Attr=Val
    if "=" in t:
        k, v = _parse_kv(t)
        k_norm = _norm_token(k)
        if k_norm == "house":
            hi = _as_int_maybe(v)
            if hi is not None:
                return Term(kind="house_const", house=hi)
            inferred = _infer_attr_from_value(v, attribute_values) or "Name"
            return Term(kind="attrval", attr=inferred, val=v)

        attr = _coerce_attr(k, attribute_values, val_hint=v)
        val2 = _coerce_value(attr, v, attribute_values)
        return Term(kind="attrval", attr=attr, val=val2)

    inferred = _infer_attr_from_value(t, attribute_values) or "Name"
    return Term(kind="attrval", attr=inferred, val=t)


def parse_dsl_statement(line: str, attribute_values: Optional[Dict[str, List[str]]] = None) -> Optional[ParsedStmt]:
    """
    Tolerant DSL parser.

    Supported ops / aliases:
      set, not, not_set,
      left_of, right_of, immediately_left_of, immediately_right_of,
      adjacent_to, adjacent, next_to,
      between,
      same_house, eq,
      plus equality lines using '=': "A = B = C."
    """
    s = _strip_prefixes(line)
    if not s:
        return None

    if "(" not in s and "=" in s:
        if attribute_values is None:
            raise ValueError("Equality clue requires attribute_values for inference")
        parts = [p.strip().strip(".") for p in s.split("=")]
        parts = [p for p in parts if p]
        if len(parts) < 2:
            return ParsedStmt("skip", ("UNDERCONSTRAINED_EQUALITY", s), line)

        terms = [_parse_term(p, attribute_values) for p in parts]
        conjuncts: List[ParsedStmt] = []
        for i in range(len(terms) - 1):
            t1, t2 = terms[i], terms[i + 1]
            conjuncts.append(ParsedStmt("same_house_term", (t1, t2), line))
        return ParsedStmt("conj", tuple(conjuncts), line)

    m = _STMT_RE.match(s)
    if not m:
        raise ValueError(f"Could not parse DSL statement: {line!r}")

    op = m.group(1).strip()
    args_raw = _split_args(m.group(2))

    op_norm = _norm_token(op).replace(" ", "_")
    if op_norm == "not_set":
        op_norm = "not"
    if op_norm in ("adjacent", "next_to", "adjacentto"):
        op_norm = "adjacent_to"
    if op_norm in ("eq", "equals", "equal"):
        op_norm = "same_house"

    if attribute_values is None:
        attribute_values = {}

    # SET
    if op_norm == "set":
        if len(args_raw) == 3:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" not in args_raw[1]:
                return ParsedStmt("set", (h_int, args_raw[1].strip(), args_raw[2].strip()), line)

            a1 = args_raw[0].strip()
            v1 = args_raw[1].strip()
            third = args_raw[2].strip()

            if "=" in third:
                t1 = _parse_term(f"{a1}={v1}", attribute_values)
                t2 = _parse_term(third, attribute_values)
                return ParsedStmt("same_house_term", (t1, t2), line)

            inferred = _infer_attr_from_value(third, attribute_values)
            if inferred is not None:
                t1 = _parse_term(f"{a1}={v1}", attribute_values)
                t2 = _parse_term(f"{inferred}={third}", attribute_values)
                return ParsedStmt("same_house_term", (t1, t2), line)

            return ParsedStmt("skip", ("UNDERCONSTRAINED_SET_3", args_raw), line)

        if len(args_raw) == 2:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" in args_raw[1]:
                k, v = _parse_kv(args_raw[1])
                return ParsedStmt("set", (h_int, k, v), line)

            if "=" in args_raw[0] and "=" in args_raw[1]:
                t1 = _parse_term(args_raw[0], attribute_values)
                t2 = _parse_term(args_raw[1], attribute_values)
                return ParsedStmt("same_house_term", (t1, t2), line)

            if "=" not in args_raw[0] and "=" not in args_raw[1]:
                return ParsedStmt("skip", ("UNDERCONSTRAINED_SET_2", args_raw), line)

        raise ValueError(f"Unsupported set(...) form: args={args_raw!r}")

    # NOT
    if op_norm == "not":
        if len(args_raw) == 3:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" not in args_raw[1]:
                return ParsedStmt("not", (h_int, args_raw[1].strip(), args_raw[2].strip()), line)

            if h_int is None:
                t1 = _parse_term(f"{args_raw[0]}={args_raw[1]}", attribute_values)
                t2 = _parse_term(args_raw[2], attribute_values)
                return ParsedStmt("not_same_house_term", (t1, t2), line)

        if len(args_raw) == 2:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" in args_raw[1]:
                k, v = _parse_kv(args_raw[1])
                return ParsedStmt("not", (h_int, k, v), line)

        raise ValueError(f"Unsupported not(...) form: args={args_raw!r}")

    # Relational ops
    if op_norm in ("left_of", "right_of", "immediately_left_of", "immediately_right_of", "adjacent_to", "same_house"):
        if len(args_raw) < 2:
            return ParsedStmt("skip", ("UNDERCONSTRAINED_REL", op_norm, args_raw), line)

        if len(args_raw) == 3 and _norm_token(args_raw[1]) == "house":
            t1 = _parse_term(args_raw[0], attribute_values)
            t2 = _parse_term(args_raw[2], attribute_values)
            return ParsedStmt(op_norm, (t1, t2), line)

        if len(args_raw) == 3 and _as_int_maybe(args_raw[0]) is not None and "=" not in args_raw[1]:
            t1 = _parse_term(args_raw[0], attribute_values)
            t2 = _parse_term(f"{args_raw[1]}={args_raw[2]}", attribute_values)
            return ParsedStmt(op_norm, (t1, t2), line)

        if len(args_raw) == 3 and "=" not in args_raw[0] and "=" not in args_raw[1]:
            t1 = _parse_term(args_raw[0], attribute_values)
            t2 = _parse_term(f"{args_raw[1]}={args_raw[2]}", attribute_values)
            return ParsedStmt(op_norm, (t1, t2), line)

        t1 = _parse_term(args_raw[0], attribute_values)
        t2 = _parse_term(args_raw[1], attribute_values)

        k = None
        if op_norm == "adjacent_to" and len(args_raw) >= 3:
            for extra in args_raw[2:]:
                if "=" in extra:
                    kk, vv = _parse_kv(extra)
                    if _norm_token(kk) == "k":
                        k = _as_int_maybe(vv)
                else:
                    maybe = _as_int_maybe(extra)
                    if maybe is not None:
                        k = maybe
            return ParsedStmt("adjacent_to", (t1, t2, int(k) if k is not None else 1), line)

        return ParsedStmt(op_norm, (t1, t2), line)

    # BETWEEN
    if op_norm == "between":
        if len(args_raw) < 2:
            return ParsedStmt("skip", ("UNDERCONSTRAINED_BETWEEN", args_raw), line)

        t1 = _parse_term(args_raw[0], attribute_values)
        t2 = _parse_term(args_raw[1], attribute_values)

        k_between = 1
        for extra in args_raw[2:]:
            if "=" in extra:
                kk, vv = _parse_kv(extra)
                if _norm_token(kk) in ("k", "gap", "between", "diff", "distance", "house"):
                    k_between = _as_int_maybe(vv) or k_between
            else:
                maybe = _as_int_maybe(extra)
                if maybe is not None:
                    k_between = maybe

        return ParsedStmt("between", (t1, t2, int(k_between)), line)

    raise ValueError(f"Unsupported op {op!r} in line: {line!r}")


# ----------------------------- House index handling -----------------------------

class HouseIndexOutOfRange(Exception):
    pass

def _extract_house_ints_from_stmt(stmt: ParsedStmt) -> List[int]:
    hs: List[int] = []

    def visit_term(t: Term) -> None:
        if isinstance(t, Term) and t.kind == "house_const" and t.house is not None:
            hs.append(int(t.house))

    def visit(x: Any) -> None:
        if isinstance(x, ParsedStmt):
            for y in x.args:
                visit(y)
        elif isinstance(x, Term):
            visit_term(x)
        elif isinstance(x, tuple) or isinstance(x, list):
            for y in x:
                visit(y)
        elif isinstance(x, int):
            # only count ints that are used as house positions (set/not use raw int)
            hs.append(int(x))

    visit(stmt)
    return hs

def detect_house_offset(parsed_clues: List[str], attribute_values: Dict[str, List[str]], n_houses: int) -> int:
    """
    If model uses 0..n-1 consistently, return +1 offset.
    Otherwise return 0.

    We only look at house indices found in parsed_clues (not reasoning) because base_sat is about clues.
    """
    found: List[int] = []
    for line in parsed_clues or []:
        try:
            st = parse_dsl_statement(line, attribute_values=attribute_values)
            if st is None or st.op == "skip":
                continue
            found.extend(_extract_house_ints_from_stmt(st))
        except Exception:
            continue

    found = [h for h in found if isinstance(h, int)]
    if not found:
        return 0

    mn = min(found)
    mx = max(found)

    # Classic 0-based pattern: min=0 and max<=n-1 and n not present.
    if mn == 0 and mx <= (n_houses - 1) and (n_houses not in found):
        return 1

    return 0


# ----------------------------- Z3 builder ----------------------------------

class Z3PuzzleModel:
    """
    Zebra-style encoding:
      var[attr][value] = Int house index (1..n_houses)
    Uniqueness:
      Distinct(values in each attribute)
    """

    def __init__(
        self,
        n_houses: int,
        attribute_values: Dict[str, List[str]],
        *,
        house_offset: int = 0,
    ):
        self.n = int(n_houses)
        if self.n <= 0:
            raise ValueError("n_houses must be > 0")
        if not isinstance(attribute_values, dict) or not attribute_values:
            raise ValueError("attribute_values must be a non-empty dict")

        self.attribute_values: Dict[str, List[str]] = {
            str(attr): [str(v) for v in vals]
            for attr, vals in attribute_values.items()
            if isinstance(vals, list) and len(vals) > 0
        }
        if not self.attribute_values:
            raise ValueError("attribute_values has no usable attribute lists")

        self.house_offset = int(house_offset)

        self.var: Dict[str, Dict[str, Any]] = {}
        self.solver = Solver()

        self._build_vars_and_domains()
        self._add_uniqueness_constraints()

    def _build_vars_and_domains(self) -> None:
        for attr, vals in self.attribute_values.items():
            self.var[attr] = {}
            for v in vals:
                x = Int(f"{attr}__{v}")
                self.var[attr][v] = x
                self.solver.add(And(x >= 1, x <= self.n))

    def _add_uniqueness_constraints(self) -> None:
        for attr, vals in self.attribute_values.items():
            xs = [self.var[attr][v] for v in vals]
            if len(xs) > 1:
                self.solver.add(Distinct(xs))

    def _adj_house(self, h: int) -> int:
        hh = int(h) + self.house_offset
        if hh < 1 or hh > self.n:
            raise HouseIndexOutOfRange(f"House index {h} (offset={self.house_offset}) -> {hh} outside 1..{self.n}")
        return hh

    def house_of(self, attr: str, val: str):
        attr = _coerce_attr(str(attr), self.attribute_values, val_hint=str(val))
        if attr not in self.var:
            raise KeyError(f"Unknown attribute {attr!r}. Available: {sorted(self.var.keys())}")
        val2 = _coerce_value(attr, str(val), self.attribute_values)
        if val2 not in self.var[attr]:
            raise KeyError(f"Unknown value {val2!r} for attribute {attr!r}.")
        return self.var[attr][val2]

    def _house_expr(self, term: Term) -> Any:
        if term.kind == "house_const":
            if term.house is None:
                raise ValueError("house_const without house")
            return IntVal(self._adj_house(int(term.house)))
        return self.house_of(term.attr, term.val)

    def stmt_to_z3(self, stmt: ParsedStmt) -> BoolRef:
        op = stmt.op
        a = stmt.args

        if op == "skip":
            return BoolVal(True)

        if op == "conj":
            inner: List[BoolRef] = []
            for st in a:
                inner.append(self.stmt_to_z3(st))
            return And(inner) if inner else BoolVal(True)

        if op == "set":
            h, attr, val = a
            hh = self._adj_house(int(h))
            if _norm_token(str(attr)) == "house":
                k = _as_int_maybe(str(val))
                if k is None:
                    return BoolVal(True)
                kk = self._adj_house(int(k))
                return BoolVal(hh == kk)
            return self.house_of(attr, val) == hh

        if op == "not":
            h, attr, val = a
            hh = self._adj_house(int(h))
            if _norm_token(str(attr)) == "house":
                k = _as_int_maybe(str(val))
                if k is None:
                    return BoolVal(True)
                kk = self._adj_house(int(k))
                return BoolVal(hh != kk)
            return self.house_of(attr, val) != hh

        if op in ("left_of", "right_of", "immediately_left_of", "immediately_right_of", "same_house"):
            t1, t2 = a
            h1 = self._house_expr(t1)
            h2 = self._house_expr(t2)

            if op == "left_of":
                return h1 < h2
            if op == "right_of":
                return h1 > h2
            if op == "same_house":
                return h1 == h2
            if op == "immediately_left_of":
                return h1 + 1 == h2
            if op == "immediately_right_of":
                return h2 + 1 == h1

        if op == "adjacent_to":
            t1, t2, k = a
            h1 = self._house_expr(t1)
            h2 = self._house_expr(t2)
            return Abs(h1 - h2) == int(k)

        if op == "same_house_term":
            t1, t2 = a
            return self._house_expr(t1) == self._house_expr(t2)

        if op == "not_same_house_term":
            t1, t2 = a
            return self._house_expr(t1) != self._house_expr(t2)

        if op == "between":
            t1, t2, k_between = a
            dist = int(k_between) + 1
            h1 = self._house_expr(t1)
            h2 = self._house_expr(t2)
            return Abs(h1 - h2) == dist

        raise ValueError(f"Unsupported stmt op: {op!r}")

    def _clue_priority(self, st: ParsedStmt) -> int:
        """
        Lower number = earlier addition (kept more often in conflict-tolerant mode).
        """
        if st.op in ("set", "same_house_term", "conj", "same_house"):
            return 10
        if st.op in ("immediately_left_of", "immediately_right_of", "left_of", "right_of", "adjacent_to", "between"):
            return 20
        if st.op in ("not", "not_same_house_term"):
            return 30
        return 40

    def add_parsed_clues(
        self,
        parsed_clues: List[str],
        *,
        conflict_tolerant: bool = True,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Add clue constraints.

        Returns dict with:
          - clue_parse_errors
          - clue_skipped_oov
          - clue_skipped_underconstrained
          - clue_skipped_out_of_range
          - clue_skipped_conflict
          - base_sat_raw
        """
        parse_errors: List[str] = []
        skipped_oov: List[str] = []
        skipped_under: List[str] = []
        skipped_oob: List[str] = []
        skipped_conflict: List[str] = []

        # Parse first
        parsed_items: List[Tuple[int, str, ParsedStmt]] = []
        for line in parsed_clues or []:
            try:
                st = parse_dsl_statement(line, attribute_values=self.attribute_values)
                if st is None:
                    continue
                if st.op == "skip":
                    skipped_under.append(f"{line!r} -> SKIPPED_UNDERCONSTRAINED")
                    continue
                parsed_items.append((self._clue_priority(st), line, st))
            except Exception as e:
                parse_errors.append(f"{line!r} -> {type(e).__name__}: {e}")

        # Raw SAT check (no conflict filtering)
        base_sat_raw: Optional[bool] = None
        try:
            raw = Solver()
            raw.append(self.solver.assertions())
            if timeout_ms is not None:
                raw.set(timeout=timeout_ms)
            for _, line, st in parsed_items:
                try:
                    raw.add(self.stmt_to_z3(st))
                except KeyError as e:
                    skipped_oov.append(f"{line!r} -> SKIPPED_OOV: {e}")
                except HouseIndexOutOfRange as e:
                    skipped_oob.append(f"{line!r} -> SKIPPED_OUT_OF_RANGE: {e}")
            r = raw.check()
            base_sat_raw = None if r == unknown else (r == sat)
        except Exception:
            base_sat_raw = None

        # Now actually add to self.solver, optionally conflict-tolerant
        parsed_items.sort(key=lambda x: x[0])  # stable heuristic ordering

        for _, line, st in parsed_items:
            try:
                phi = self.stmt_to_z3(st)
            except KeyError as e:
                skipped_oov.append(f"{line!r} -> SKIPPED_OOV: {e}")
                continue
            except HouseIndexOutOfRange as e:
                skipped_oob.append(f"{line!r} -> SKIPPED_OUT_OF_RANGE: {e}")
                continue
            except Exception as e:
                parse_errors.append(f"{line!r} -> {type(e).__name__}: {e}")
                continue

            if not conflict_tolerant:
                self.solver.add(phi)
                continue

            # conflict-tolerant: keep if doesn't make UNSAT
            self.solver.push()
            self.solver.add(phi)
            if timeout_ms is not None:
                self.solver.set(timeout=timeout_ms)
            chk = self.solver.check()
            if chk == unsat:
                self.solver.pop()
                skipped_conflict.append(f"{line!r} -> SKIPPED_CONFLICT")
            else:
                # keep it (pop the push frame but keep assertions) -> easiest: no pop; instead push/pop strategy:
                # We already added inside a push frame; to keep it, do nothing and just "commit" by not popping.
                # But Z3 only commits when we don't pop; that means we must not leave the stack growing.
                # So: pop, then add again permanently.
                self.solver.pop()
                self.solver.add(phi)

        return {
            "clue_parse_errors": parse_errors,
            "clue_skipped_oov": skipped_oov,
            "clue_skipped_underconstrained": skipped_under,
            "clue_skipped_out_of_range": skipped_oob,
            "clue_skipped_conflict": skipped_conflict,
            "base_sat_raw": base_sat_raw,
        }


# ----------------------- Reasoning validation logic -------------------------

@dataclass
class StepValidation:
    index: int
    raw: str
    parsed_ok: bool
    parse_error: Optional[str]
    strict_status: str   # ENTAILED / NOT_ENTAILED / CONTRADICTION / UNKNOWN / PARSE_ERROR / BASE_UNSAT
    chain_status: str
    strict_valid: Optional[bool]
    chain_valid: Optional[bool]


def _check_status(base_solver: Solver, phi: BoolRef, timeout_ms: Optional[int]) -> str:
    s1 = Solver()
    s1.append(base_solver.assertions())
    if timeout_ms is not None:
        s1.set(timeout=timeout_ms)
    s1.add(phi)
    r1 = s1.check()
    if r1 == unsat:
        return "CONTRADICTION"
    if r1 == unknown:
        return "UNKNOWN"

    s2 = Solver()
    s2.append(base_solver.assertions())
    if timeout_ms is not None:
        s2.set(timeout=timeout_ms)
    s2.add(Not(phi))
    r2 = s2.check()
    if r2 == unsat:
        return "ENTAILED"
    if r2 == unknown:
        return "UNKNOWN"
    return "NOT_ENTAILED"


def validate_reasoning_steps(
    *,
    n_houses: int,
    attribute_values: Dict[str, List[str]],
    parsed_clues: List[str],
    parsed_reasoning: List[str],
    timeout_s: float = 1.0,
    conflict_tolerant_clues: bool = True,
) -> Dict[str, Any]:
    timeout_ms = int(max(0.0, float(timeout_s)) * 1000)

    house_offset = detect_house_offset(parsed_clues or [], attribute_values, n_houses)

    model = Z3PuzzleModel(
        n_houses=n_houses,
        attribute_values=attribute_values,
        house_offset=house_offset,
    )

    clue_info = model.add_parsed_clues(
        parsed_clues or [],
        conflict_tolerant=conflict_tolerant_clues,
        timeout_ms=timeout_ms,
    )

    base_check = model.solver.check()
    base_sat: Optional[bool] = None if base_check == unknown else (base_check == sat)

    chain_solver = Solver()
    chain_solver.append(model.solver.assertions())
    chain_solver.set(timeout=timeout_ms)

    def _to_bool(status: str) -> Optional[bool]:
        if status == "ENTAILED":
            return True
        if status in ("NOT_ENTAILED", "CONTRADICTION"):
            return False
        return None

    steps_out: List[StepValidation] = []
    for i, step_raw in enumerate(parsed_reasoning or []):
        try:
            st = parse_dsl_statement(step_raw, attribute_values=model.attribute_values)
            if st is None or st.op == "skip":
                steps_out.append(StepValidation(
                    index=i + 1, raw=step_raw, parsed_ok=False,
                    parse_error="Empty/underconstrained step",
                    strict_status="PARSE_ERROR", chain_status="PARSE_ERROR",
                    strict_valid=None, chain_valid=None,
                ))
                continue

            phi = model.stmt_to_z3(st)

            if base_sat is False:
                strict_status = "BASE_UNSAT"
                chain_status = "BASE_UNSAT"
            else:
                strict_status = _check_status(model.solver, phi, timeout_ms=timeout_ms)
                chain_status = _check_status(chain_solver, phi, timeout_ms=timeout_ms)

            if chain_status not in ("CONTRADICTION", "UNKNOWN", "BASE_UNSAT", "PARSE_ERROR"):
                chain_solver.add(phi)

            steps_out.append(StepValidation(
                index=i + 1, raw=step_raw, parsed_ok=True, parse_error=None,
                strict_status=strict_status, chain_status=chain_status,
                strict_valid=_to_bool(strict_status), chain_valid=_to_bool(chain_status),
            ))
        except HouseIndexOutOfRange as e:
            steps_out.append(StepValidation(
                index=i + 1, raw=step_raw, parsed_ok=False,
                parse_error=f"HouseIndexOutOfRange: {e}",
                strict_status="PARSE_ERROR", chain_status="PARSE_ERROR",
                strict_valid=None, chain_valid=None,
            ))
        except Exception as e:
            steps_out.append(StepValidation(
                index=i + 1, raw=step_raw, parsed_ok=False,
                parse_error=f"{type(e).__name__}: {e}",
                strict_status="PARSE_ERROR", chain_status="PARSE_ERROR",
                strict_valid=None, chain_valid=None,
            ))

    return {
        "base_sat": base_sat,
        "base_sat_raw": clue_info.get("base_sat_raw"),
        "house_offset": house_offset,

        "clue_parse_errors": clue_info.get("clue_parse_errors", []),
        "clue_skipped_oov": clue_info.get("clue_skipped_oov", []),
        "clue_skipped_underconstrained": clue_info.get("clue_skipped_underconstrained", []),
        "clue_skipped_out_of_range": clue_info.get("clue_skipped_out_of_range", []),
        "clue_skipped_conflict": clue_info.get("clue_skipped_conflict", []),

        "steps": [sv.__dict__ for sv in steps_out],
    }


# ----------------------------- Optional JSONL helper -----------------------------

def extract_last_answer_json(text: str) -> Optional[dict]:
    ms = list(re.finditer(r"<answer>\s*(\{.*?\})\s*</answer>", text, flags=re.S))
    if not ms:
        return None
    blob = ms[-1].group(1)
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    try:
        return json.loads(blob)
    except Exception:
        return None


def main():

    fpath = '/home/asif/data3/Codes_QCRI/Reasoning360/evaluation_results/small_train_small_test_1_parsed_v2/qwen2515binstruct/'
    fname = 'jobid_243455/75.jsonl'

    #/export/home/asifali/Reasoning360/evaluation_results/small_train_small_test_1_parsed_v2/qwen2515binstruct/jobid_243455

    file_name = os.path.join(fpath, fname)


    with open(file_name, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    total = 0
    base_false = 0
    base_false_raw = 0

    for r in rows:
        ans = extract_last_answer_json(r.get("output", ""))
        if not ans:
            continue

        #print()

        total += 1
        out = validate_reasoning_steps(
            n_houses=ans["n_houses"],
            attribute_values=ans["attribute_values"],
            parsed_clues=ans.get("parsed_clues", []),
            parsed_reasoning=ans.get("parsed_reasoning", []),
            timeout_s=5.0,
            conflict_tolerant_clues=True,
        )
        print("Number of Valid reasoning steps = {}".format(len(out["steps"])))
        if out["base_sat"] is False:
            base_false += 1
        if out["base_sat_raw"] is False:
            base_false_raw += 1

    print(f"Rows evaluated: {total}")
    if total > 0:
        print(f"base_sat_raw False rate: {base_false_raw}/{total} = {100.0*base_false_raw/total:.2f}%")
        print(f"base_sat     False rate: {base_false}/{total} = {100.0*base_false/total:.2f}%")

if __name__ == "__main__":
    main()
