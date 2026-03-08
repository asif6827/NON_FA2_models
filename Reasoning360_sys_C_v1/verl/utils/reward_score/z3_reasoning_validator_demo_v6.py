# -*- coding: utf-8 -*-
"""
z3_reasoning_validator_demo_v6.py

A tolerant Z3 validator for Zebra/Logic puzzles.

This version specifically fixes the remaining failure patterns in your logs, including:
- Missing import of IntVal (caused NameError: IntVal not defined)
- House constants inside relational ops: immediately_left_of(Name=Peter,House=1), not(4,House,1), left_of(3,Name,Eric), immediately_left_of(1,House,2), etc.
- between(...) variants with house constants and extra metadata: between(Pet=dog,Name=Peter,House=1), between(1,Name,Alice,1), between(2,4,1), between(...,K=1)
- "A = B = C" equality clues like: Love=daffodils = Eric.
- set(attr,val,value) forms where the 3rd arg is a VALUE (infers its attribute) -> same_house(...)
- not_set(attr,val,Attr=Val) -> not_same_house(...)
- Underconstrained clues like immediately_left_of(Sport=soccer) are SKIPPED (not counted as parse errors).

Return schema includes:
- base_sat
- clue_parse_errors (true syntax/structural errors)
- clue_skipped_oov (unknown attr/value)
- clue_skipped_underconstrained (too little info to encode)
- steps (per reasoning-step validation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from z3 import Abs, And, BoolRef, BoolVal, Distinct, Int, IntVal, Not, Or, Solver, sat, unsat, unknown


import re
import os
import json
from our_puzzles_dataset import extract_reasoning_and_solution




# ----------------------------- Normalization helpers -----------------------------

_WORD_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
}

def _norm_token(s: str) -> str:
    s = (s or "").strip()
    # strip surrounding quotes
    if (len(s) >= 2) and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1].strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def _as_int_maybe(s: str) -> Optional[int]:
    t = _norm_token(s)
    if t in _WORD_NUM:
        return _WORD_NUM[t]
    # allow "3." or "3)" etc.
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
    m = re.search(r"\]\s*(.*)$", s)  # after [..]
    if m:
        return m.group(1).strip()
    m2 = re.search(r"=\s*(.*)$", s)  # after C1 =
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

def _coerce_attr(attr: str, attribute_values: Dict[str, List[str]], val_hint: Optional[str] = None) -> str:
    a_norm = _norm_token(attr)
    if a_norm in _ATTR_ALIASES:
        attr = _ATTR_ALIASES[a_norm]

    if attr in attribute_values:
        return attr

    m = _best_match(attr, list(attribute_values.keys()))
    if m is not None:
        return m

    # if unknown attribute but value exists somewhere, infer attribute from value
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

def _infer_attr_from_value(val: str, attribute_values: Dict[str, List[str]]) -> Optional[str]:
    nv = _norm_token(val)
    for a, vals in attribute_values.items():
        for v in vals:
            if _norm_token(v) == nv:
                return a
    return None

def _parse_term(tok: str, attribute_values: Dict[str, List[str]]) -> Term:
    """
    Parse a single term into:
      - House constant: House=1, 3, one, second, etc.
      - Attr=Val: Name=Eric
      - Value-only: Eric  (infer its attribute from domain; commonly Name)
    """
    t = tok.strip()
    if t == "":
        raise ValueError("Empty term")

    # numeric-only
    i = _as_int_maybe(t)
    if i is not None and ("=" not in t):
        return Term(kind="house_const", house=i)

    # Attr=Val
    if "=" in t:
        k, v = _parse_kv(t)
        k_norm = _norm_token(k)
        if k_norm == "house":
            hi = _as_int_maybe(v)
            if hi is not None:
                return Term(kind="house_const", house=hi)
            # House=Peter means value-only; infer attribute of "Peter" (usually Name)
            inferred = _infer_attr_from_value(v, attribute_values) or "Name"
            return Term(kind="attrval", attr=inferred, val=v)

        # normal attr=val
        attr = _coerce_attr(k, attribute_values, val_hint=v)
        val2 = _coerce_value(attr, v, attribute_values)
        return Term(kind="attrval", attr=attr, val=val2)

    # value-only (infer)
    inferred = _infer_attr_from_value(t, attribute_values) or "Name"
    return Term(kind="attrval", attr=inferred, val=t)


def _house_expr(term: Term, model: "Z3PuzzleModel") -> Any:
    if term.kind == "house_const":
        return IntVal(int(term.house))
    return model.house_of(term.attr, term.val)


def parse_dsl_statement(line: str, attribute_values: Optional[Dict[str, List[str]]] = None) -> Optional[ParsedStmt]:
    """
    Tolerant DSL parser.

    Supported ops / aliases:
      set, not, not_set,
      left_of, right_of, immediately_left_of, immediately_right_of,
      adjacent_to, adjacent, next_to,
      between,
      same_house, eq,
      plus equality lines using '=':  "A = B = C."  (returns conj(same_house(...), ...))
    """
    s = _strip_prefixes(line)
    if not s:
        return None

    # Equality clue without function wrapper: "A = B = C."
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
    if op_norm == "immediately_right_of":
        # normalize to immediately_left_of by swapping terms later in stmt_to_z3
        op_norm = "immediately_right_of"

    if attribute_values is None:
        # for strict parsing you can omit, but your use-case always has it
        attribute_values = {}

    # -------------- SET --------------
    if op_norm == "set":
        # set(h,Attr,Val)
        if len(args_raw) == 3:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" not in args_raw[1]:
                return ParsedStmt("set", (h_int, args_raw[1].strip(), args_raw[2].strip()), line)

            # set(Attr,Val,OtherValOrAttr)
            # Example from log: set(Name,Eric,pizza) -> same_house(Name=Eric, Lunch=pizza) (infer pizza)
            # Example: set(Lunch,stew,Eric) -> same_house(Lunch=stew, Name=Eric)
            a1 = args_raw[0].strip()
            v1 = args_raw[1].strip()
            third = args_raw[2].strip()

            # If third is "Attr=Val" -> same_house between first pair and third
            if "=" in third:
                t1 = _parse_term(f"{a1}={v1}", attribute_values)
                t2 = _parse_term(third, attribute_values)
                return ParsedStmt("same_house_term", (t1, t2), line)

            # If third is a VALUE -> infer its attribute -> same_house
            inferred = _infer_attr_from_value(third, attribute_values)
            if inferred is not None:
                t1 = _parse_term(f"{a1}={v1}", attribute_values)
                t2 = _parse_term(f"{inferred}={third}", attribute_values)
                return ParsedStmt("same_house_term", (t1, t2), line)

            # If third looks like an attribute name (no value), it's underconstrained
            return ParsedStmt("skip", ("UNDERCONSTRAINED_SET_3", args_raw), line)

        # set(h,Attr=Val)
        if len(args_raw) == 2:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" in args_raw[1]:
                k, v = _parse_kv(args_raw[1])
                return ParsedStmt("set", (h_int, k, v), line)

            # set(Attr=Val, Attr=Val) -> same_house
            if "=" in args_raw[0] and "=" in args_raw[1]:
                t1 = _parse_term(args_raw[0], attribute_values)
                t2 = _parse_term(args_raw[1], attribute_values)
                return ParsedStmt("same_house_term", (t1, t2), line)

            # set(Name,Arnold) -> underconstrained existence statement
            if "=" not in args_raw[0] and "=" not in args_raw[1]:
                return ParsedStmt("skip", ("UNDERCONSTRAINED_SET_2", args_raw), line)

        raise ValueError(f"Unsupported set(...) form: args={args_raw!r}")

    # -------------- NOT --------------
    if op_norm == "not":
        # not(h,Attr,Val)
        if len(args_raw) == 3:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" not in args_raw[1]:
                return ParsedStmt("not", (h_int, args_raw[1].strip(), args_raw[2].strip()), line)

            # not_set(Height,cat,HouseStyle=second)  => not_same_house(Height=cat, HouseStyle=second)
            if h_int is None:
                t1 = _parse_term(f"{args_raw[0]}={args_raw[1]}", attribute_values)
                t2 = _parse_term(args_raw[2], attribute_values)
                return ParsedStmt("not_same_house_term", (t1, t2), line)

        # not(h,Attr=Val)
        if len(args_raw) == 2:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" in args_raw[1]:
                k, v = _parse_kv(args_raw[1])
                return ParsedStmt("not", (h_int, k, v), line)

        # not(h, Attr=..., Val=...)
        if len(args_raw) >= 3 and _as_int_maybe(args_raw[0]) is not None:
            h_int = _as_int_maybe(args_raw[0])
            attr = None
            val = None
            for t in args_raw[1:]:
                if "=" not in t:
                    continue
                k, v = _parse_kv(t)
                if _norm:=_norm_token(k) in ("attr", "attribute"):
                    attr = v
                if _norm_token(k) == "val":
                    val = v
            if attr is not None and val is not None:
                return ParsedStmt("not", (h_int, attr, val), line)

        raise ValueError(f"Unsupported not(...) form: args={args_raw!r}")

    # -------------- Relational ops: left_of/right_of/immediately_left_of/... --------------
    if op_norm in ("left_of", "right_of", "immediately_left_of", "immediately_right_of", "adjacent_to", "same_house"):
        # Underconstrained: immediately_left_of(Sport=soccer)
        if len(args_raw) < 2:
            return ParsedStmt("skip", ("UNDERCONSTRAINED_REL", op_norm, args_raw), line)

        # Pattern: immediately_left_of(1,House,2)
        if len(args_raw) == 3 and _norm_token(args_raw[1]) == "house":
            t1 = _parse_term(args_raw[0], attribute_values)
            t2 = _parse_term(args_raw[2], attribute_values)
            return ParsedStmt(op_norm, (t1, t2), line)

        # Pattern: left_of(3,Name,Eric)
        if len(args_raw) == 3 and _as_int_maybe(args_raw[0]) is not None and "=" not in args_raw[1]:
            t1 = _parse_term(args_raw[0], attribute_values)              # house const
            t2 = _parse_term(f"{args_raw[1]}={args_raw[2]}", attribute_values)
            return ParsedStmt(op_norm, (t1, t2), line)

        # Pattern: immediately_left_of(Engineer,Name,Arnold)
        if len(args_raw) == 3 and "=" not in args_raw[0] and "=" not in args_raw[1]:
            # first is value-only, second+third is Attr,Val
            t1 = _parse_term(args_raw[0], attribute_values)
            t2 = _parse_term(f"{args_raw[1]}={args_raw[2]}", attribute_values)
            return ParsedStmt(op_norm, (t1, t2), line)

        # Standard: op(Attr=Val, Attr=Val) plus extra metadata like K=1 -> ignore extras
        t1 = _parse_term(args_raw[0], attribute_values)
        t2 = _parse_term(args_raw[1], attribute_values)
        # if extra K=... for adjacent_to, support distance K (default 1)
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
        if op_norm == "adjacent_to":
            return ParsedStmt("adjacent_to", (t1, t2, int(k) if k is not None else 1), line)

        return ParsedStmt(op_norm, (t1, t2), line)

    # -------------- BETWEEN --------------
    if op_norm == "between":
        # Highly variable in your logs. We'll extract up to two terms and a gap parameter K.
        if len(args_raw) < 2:
            return ParsedStmt("skip", ("UNDERCONSTRAINED_BETWEEN", args_raw), line)

        # Special: between(Lunch=pizza,House=2) => underconstrained (skip)
        if len(args_raw) == 2 and _norm_token(args_raw[1]).startswith("house="):
            return ParsedStmt("skip", ("UNDERCONSTRAINED_BETWEEN_2", args_raw), line)

        # Forms like between(1,Name,Alice,1) / between(2,4,1) / between(Animal=cat,Name=Alice,House=1,K=1)
        # Decide term1/term2
        t1: Optional[Term] = None
        t2: Optional[Term] = None
        k_between = 1

        # If first token is house const and next two look like Attr,Val
        if len(args_raw) >= 4 and _as_int_maybe(args_raw[0]) is not None and "=" not in args_raw[1]:
            t1 = _parse_term(args_raw[0], attribute_values)  # house const
            t2 = _parse_term(f"{args_raw[1]}={args_raw[2]}", attribute_values)
            # 4th is k
            k_between = _as_int_maybe(args_raw[3]) or 1
            return ParsedStmt("between", (t1, t2, int(k_between)), line)

        # Otherwise, first two are standard terms (or house consts)
        t1 = _parse_term(args_raw[0], attribute_values)
        t2 = _parse_term(args_raw[1], attribute_values)

        # Find K in remaining args (K=..., House=1 treated as K=1, bare int)
        for extra in args_raw[2:]:
            if "=" in extra:
                kk, vv = _parse_kv(extra)
                if _norm_token(kk) in ("k", "gap", "between", "diff", "distance"):
                    k_between = _as_int_maybe(vv) or k_between
                elif _norm_token(kk) == "house":
                    # many of your generations use House=1 as a proxy for "K=1"
                    k_between = _as_int_maybe(vv) or k_between
            else:
                maybe = _as_int_maybe(extra)
                if maybe is not None:
                    k_between = maybe

        return ParsedStmt("between", (t1, t2, int(k_between)), line)

    raise ValueError(f"Unsupported op {op!r} in line: {line!r}")


# ----------------------------- Z3 builder ----------------------------------

class Z3PuzzleModel:
    """
    Zebra-style encoding:
      var[attr][value] = Int house index (1..n_houses)
    Uniqueness:
      Distinct(values in each attribute)
    """

    def __init__(self, n_houses: int, attribute_values: Dict[str, List[str]]):
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

    def house_of(self, attr: str, val: str):
        attr = _coerce_attr(str(attr), self.attribute_values, val_hint=str(val))
        if attr not in self.var:
            raise KeyError(f"Unknown attribute {attr!r}. Available: {sorted(self.var.keys())}")
        val2 = _coerce_value(attr, str(val), self.attribute_values)
        if val2 not in self.var[attr]:
            raise KeyError(f"Unknown value {val2!r} for attribute {attr!r}.")
        return self.var[attr][val2]

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
            # Special handling: set(h,House,k) => BoolVal(h == k)
            if _norm_token(str(attr)) == "house":
                k = _as_int_maybe(str(val))
                if k is None:
                    return BoolVal(True)  # can't interpret
                return BoolVal(int(h) == int(k))
            return self.house_of(attr, val) == int(h)

        if op == "not":
            h, attr, val = a
            if _norm_token(str(attr)) == "house":
                k = _as_int_maybe(str(val))
                if k is None:
                    return BoolVal(True)
                return BoolVal(int(h) != int(k))
            return self.house_of(attr, val) != int(h)

        if op in ("left_of", "right_of", "immediately_left_of", "immediately_right_of", "same_house"):
            t1, t2 = a
            h1 = _house_expr(t1, self)
            h2 = _house_expr(t2, self)

            if op == "left_of":
                return h1 < h2
            if op == "right_of":
                return h1 > h2
            if op == "same_house":
                return h1 == h2
            if op == "immediately_left_of":
                return h1 + 1 == h2
            if op == "immediately_right_of":
                # A immediately right of B  <=>  B immediately left of A
                return h2 + 1 == h1

        if op == "adjacent_to":
            t1, t2, k = a
            h1 = _house_expr(t1, self)
            h2 = _house_expr(t2, self)
            return Abs(h1 - h2) == int(k)

        if op == "same_house_term":
            t1, t2 = a
            return _house_expr(t1, self) == _house_expr(t2, self)

        if op == "not_same_house_term":
            t1, t2 = a
            return _house_expr(t1, self) != _house_expr(t2, self)

        if op == "between":
            t1, t2, k_between = a
            # Interpret k_between as "houses between" => distance = k_between + 1
            dist = int(k_between) + 1
            h1 = _house_expr(t1, self)
            h2 = _house_expr(t2, self)
            return Abs(h1 - h2) == dist

        raise ValueError(f"Unsupported stmt op: {op!r}")

    def add_parsed_clues(self, parsed_clues: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Add clue constraints.
        Returns (parse_errors, skipped_oov, skipped_underconstrained).
        """
        parse_errors: List[str] = []
        skipped_oov: List[str] = []
        skipped_under: List[str] = []

        for line in parsed_clues or []:
            try:
                st = parse_dsl_statement(line, attribute_values=self.attribute_values)
                if st is None:
                    continue
                if st.op == "skip":
                    skipped_under.append(f"{line!r} -> SKIPPED_UNDERCONSTRAINED")
                    continue
                # Add constraint (may still raise KeyError -> OOV)
                self.solver.add(self.stmt_to_z3(st))
            except KeyError as e:
                skipped_oov.append(f"{line!r} -> SKIPPED_OOV: {e}")
            except Exception as e:
                parse_errors.append(f"{line!r} -> {type(e).__name__}: {e}")

        return parse_errors, skipped_oov, skipped_under


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
) -> Dict[str, Any]:
    timeout_ms = int(max(0.0, float(timeout_s)) * 1000)

    model = Z3PuzzleModel(n_houses=n_houses, attribute_values=attribute_values)
    clue_parse_errors, clue_skipped_oov, clue_skipped_under = model.add_parsed_clues(parsed_clues or [])

    base_check = model.solver.check()
    base_sat: Optional[bool] = None if base_check == unknown else (base_check == sat)

    chain_solver = Solver()
    chain_solver.append(model.solver.assertions())
    if timeout_ms is not None:
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
        except Exception as e:
            steps_out.append(StepValidation(
                index=i + 1, raw=step_raw, parsed_ok=False,
                parse_error=f"{type(e).__name__}: {e}",
                strict_status="PARSE_ERROR", chain_status="PARSE_ERROR",
                strict_valid=None, chain_valid=None,
            ))

    return {
        "base_sat": base_sat,
        "clue_parse_errors": clue_parse_errors,
        "clue_skipped_oov": clue_skipped_oov,
        "clue_skipped_underconstrained": clue_skipped_under,
        "steps": [sv.__dict__ for sv in steps_out],
    }


def main():
    #demo_1_entails_chain()
    #demo_2_not_entailed()
    #demo_3_contradiction()
    #demo_4_adjacent()
    #demo_5_chain_only()

    print("\n" + "=" * 88)
    #print("Integration snippet (paste into your compute_score after extracting parsed_*):\n")
    #print(
    #    "from z3_reasoning_validator_demo import validate_reasoning_steps\n\n"
    #    "reasoning_check = validate_reasoning_steps(\n"
    #    "    n_houses=n_houses,\n"
    #    "    attribute_values=attribute_values,\n"
    #    "    parsed_clues=parsed_clues,\n"
    #    "    parsed_reasoning=parsed_reasoning,\n"
    #    "    timeout_s=float(os.environ.get('Z3_TIMEOUT_S', '1.0')),\n"
    #    ")\n\n"
    #    "final_result['z3_base_sat'] = reasoning_check['base_sat']\n"
    #    "final_result['z3_clue_parse_errors'] = reasoning_check['clue_parse_errors']\n"
    #    "final_result['z3_reasoning_steps'] = reasoning_check['steps']\n"
    #)

    fpath = '/home/asif/data3/Codes_QCRI/Reasoning360/evaluation_results/small_train_small_test_1_parsed_v2/qwen2515binstruct/'
    fname = 'jobid_243455/75.jsonl'

    #/export/home/asifali/Reasoning360/evaluation_results/small_train_small_test_1_parsed_v2/qwen2515binstruct/jobid_243455

    file_name = os.path.join(fpath, fname)

    with open(file_name, "r", encoding="utf-8") as f:
        json_data = [json.loads(line) for line in f if line.strip()]
    print(f"Total data in file: {len(json_data)}")


    print("\n" + "=" * 88)
    idx1 = 0
    count_base_valid = 0

    for idx,line in enumerate(json_data):
        LLM_output = line['output']
        puzzle_acc = line['PUZZLE_ACCURACY']
        parsed_clues, parsed_reasoning, predicted_arrangement, attribute_values, n_houses, parse_status = extract_reasoning_and_solution(solution_str=LLM_output)

        valid_out = validate_reasoning_steps(
            n_houses=n_houses,
            attribute_values=attribute_values,
            parsed_clues=parsed_clues,
            parsed_reasoning=parsed_reasoning,
            timeout_s=5.0,
        )

        if valid_out is not None:
            if valid_out['base_sat'] is False:
                print(valid_out)
                count_base_valid += 1


        '''        
        if puzzle_acc == 1.0:
            idx1 +=1
            print(f"index = {idx1}; output = {json.dumps(valid_out, indent=1, ensure_ascii=False)}")
            print()
            print()
        if idx1 == 15:
            break
        '''
    print('Total Base Valid = {}'.format(count_base_valid))

if __name__ == "__main__":
    main()