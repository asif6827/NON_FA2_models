# -*- coding: utf-8 -*-
"""
z3_reasoning_validator_demo_v9.py

A tolerant Z3 validator for Zebra/Logic puzzles.

Adds (from v8):
- conflict_tolerant_clues (bool): if True, we will *not* allow a single contradictory clue
  to make the entire base system UNSAT. Instead, we add clues one-by-one and skip any clue
  that would make the solver unsat.

Why this helps:
- Many LLM outputs include 1-2 malformed or logically conflicting parsed_clues; previously
  those made base_sat=False for the whole sample, and all reasoning steps became BASE_UNSAT.
- With conflict_tolerant_clues=True, we keep the base model satisfiable and still validate
  as many reasoning steps as possible against the remaining consistent clue set.

Return schema includes:
- base_sat
- clue_parse_errors (true syntax/structural errors)
- clue_skipped_oov (unknown attr/value)
- clue_skipped_underconstrained (too little info to encode)
- clue_skipped_conflict (NEW): clues skipped because they would make the base UNSAT (only if conflict_tolerant_clues=True)
- steps (per reasoning-step validation)
- n_steps_total (NEW)
- n_steps_parsed_ok (NEW)
- n_steps_entailed_chain (NEW)
- n_steps_contradiction_chain (NEW)

Quick README (Output)
---------------------
base_sat:
  - True  -> the base constraints (uniqueness + accepted clues) are satisfiable
  - False -> UNSAT (only likely if conflict_tolerant_clues=False OR uniqueness/domain mismatch)
  - None  -> Z3 returned unknown (timeout/complexity)

clue_* lists:
  - parse_errors: parser couldn't understand a clue line
  - skipped_oov: attribute/value not found in attribute_values
  - skipped_underconstrained: clue too vague to encode (safe to ignore)
  - skipped_conflict: clue was syntactically valid but contradicts the existing clue set

steps:
  list of dicts; each dict has:
    index, raw, parsed_ok, parse_error,
    strict_status / chain_status in:
      ENTAILED, NOT_ENTAILED, CONTRADICTION, UNKNOWN, PARSE_ERROR, BASE_UNSAT

Counters:
  - n_steps_total: len(steps)
  - n_steps_parsed_ok: number with parsed_ok=True
  - n_steps_entailed_chain: chain_status == ENTAILED
  - n_steps_contradiction_chain: chain_status == CONTRADICTION
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Abs, And, BoolRef, BoolVal, Distinct, Int, IntVal, Not, Solver, sat, unsat, unknown
from pprint import pprint
import re
import os
import json

# NOTE: your project imports
#from our_puzzles_dataset import extract_reasoning_and_solution


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



# ----------------------------- Domain tolerance helpers -----------------------------

def _dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

def _extract_mentioned_values(lines: List[str]) -> List[str]:
    """
    Best-effort extraction of VALUE tokens mentioned in parsed_clues / parsed_reasoning,
    used to prioritize which domain values to keep if an attribute list is longer than n_houses.

    This is intentionally fuzzy (regex-based), because the DSL can be noisy.
    """
    mentioned: List[str] = []
    if not lines:
        return mentioned

    for line in lines:
        s = _strip_prefixes(line)
        if not s:
            continue

        # capture RHS of Attr=Val patterns
        for m in re.finditer(r"\b[A-Za-z_][\w]*\s*=\s*([^,\)\s]+)", s):
            val = m.group(1)
            if val:
                mentioned.append(_norm_token(val))

        # capture value-only args inside (...) (excluding obvious numerics and K=...)
        mm = _STMT_RE.match(s)
        if mm:
            args_raw = _split_args(mm.group(2))
            for a in args_raw:
                if "=" in a:
                    k, v = a.split("=", 1)
                    if _norm_token(k) in ("k", "gap", "between", "diff", "distance", "house", "attr", "attribute", "val"):
                        continue
                    mentioned.append(_norm_token(v))
                else:
                    if _as_int_maybe(a) is not None:
                        continue
                    t = _norm_token(a)
                    if t:
                        mentioned.append(t)

    return mentioned

def _prioritize_and_truncate_values(
    vals: List[str],
    *,
    n_houses: int,
    mentioned_norm_values: Optional[set] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Returns (new_vals, note). If truncation happened, note is a human-readable message.
    """
    vals2 = _dedupe_preserve_order([str(v) for v in vals])
    note = None

    if mentioned_norm_values:
        prioritized: List[str] = []
        rest: List[str] = []
        for v in vals2:
            if _norm_token(v) in mentioned_norm_values:
                prioritized.append(v)
            else:
                rest.append(v)
        vals2 = prioritized + rest

    if n_houses > 0 and len(vals2) > n_houses:
        note = f"TRUNCATED_DOMAIN: kept {n_houses}/{len(vals2)} values"
        vals2 = vals2[:n_houses]

    return vals2, note

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
            inferred = _infer_attr_from_value(v, attribute_values) or "Name"
            return Term(kind="attrval", attr=inferred, val=v)

        attr = _coerce_attr(k, attribute_values, val_hint=v)
        val2 = _coerce_value(attr, v, attribute_values)
        return Term(kind="attrval", attr=attr, val=val2)

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
            conjuncts.append(ParsedStmt("same_house_term", (terms[i], terms[i + 1]), line))
        return ParsedStmt("conj", tuple(conjuncts), line)

    m = _STMT_RE.match(s)
    if not m:
        raise ValueError(f"Could not parse DSL statement: {line!r}")

    op = m.group(1).strip()
    args_raw = _split_args(m.group(2))

    op_norm = _norm_token(op).replace(" ", "_")
    # keep not_set as its own canonical op (we also accept legacy op "not")
    if op_norm in ("adjacent", "next_to", "adjacentto"):
        op_norm = "adjacent_to"
    if op_norm in ("not_same_house", "not_samehouse", "not_same"):
        op_norm = "not_same_house"
    if op_norm in ("eq", "equals", "equal"):
        op_norm = "same_house"

    if attribute_values is None:
        attribute_values = {}

    # -------------- SET --------------
    if op_norm == "set":
        # set(h,Attr,Val)
        if len(args_raw) == 3:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" not in args_raw[1]:
                return ParsedStmt("set", (h_int, args_raw[1].strip(), args_raw[2].strip()), line)

            # set(Attr,Val,OtherValOrAttr) => same_house(...) if 3rd can be inferred as a value
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

        # set(h,Attr=Val)
        if len(args_raw) == 2:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" in args_raw[1]:
                k, v = _parse_kv(args_raw[1])
                return ParsedStmt("set", (h_int, k, v), line)

            if "=" in args_raw[0] and "=" in args_raw[1]:
                t1 = _parse_term(args_raw[0], attribute_values)
                t2 = _parse_term(args_raw[1], attribute_values)
                return ParsedStmt("same_house_term", (t1, t2), line)

            return ParsedStmt("skip", ("UNDERCONSTRAINED_SET_2", args_raw), line)

        raise ValueError(f"Unsupported set(...) form: args={args_raw!r}")

    # -------------- NOT --------------
    if op_norm in ("not", "not_set"):
        # not(h,Attr,Val)
        if len(args_raw) == 3:
            h_int = _as_int_maybe(args_raw[0])
            if h_int is not None and "=" not in args_raw[1]:
                return ParsedStmt("not", (h_int, args_raw[1].strip(), args_raw[2].strip()), line)

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

        raise ValueError(f"Unsupported not(...) form: args={args_raw!r}")

    # -------------- Relational ops --------------
    if op_norm in ("left_of", "right_of", "immediately_left_of", "immediately_right_of", "adjacent_to", "same_house", "not_same_house"):
        if len(args_raw) < 2:
            return ParsedStmt("skip", ("UNDERCONSTRAINED_REL", op_norm, args_raw), line)

        # immediately_left_of(1,House,2)
        if len(args_raw) == 3 and _norm_token(args_raw[1]) == "house":
            t1 = _parse_term(args_raw[0], attribute_values)
            t2 = _parse_term(args_raw[2], attribute_values)
            return ParsedStmt(op_norm, (t1, t2), line)

        # left_of(3,Name,Eric)
        if len(args_raw) == 3 and _as_int_maybe(args_raw[0]) is not None and "=" not in args_raw[1]:
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

    # -------------- BETWEEN --------------
    if op_norm == "between":
        if len(args_raw) < 2:
            return ParsedStmt("skip", ("UNDERCONSTRAINED_BETWEEN", args_raw), line)

        t1 = _parse_term(args_raw[0], attribute_values)
        t2 = _parse_term(args_raw[1], attribute_values)
        k_between = 1

        for extra in args_raw[2:]:
            if "=" in extra:
                kk, vv = _parse_kv(extra)
                if _norm_token(kk) in ("k", "gap", "between", "diff", "distance"):
                    k_between = _as_int_maybe(vv) or k_between
                elif _norm_token(kk) == "house":
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

    def __init__(self, n_houses: int, attribute_values: Dict[str, List[str]], mentioned_norm_values: Optional[List[str]] = None):
        self.n = int(n_houses)
        if self.n <= 0:
            raise ValueError("n_houses must be > 0")
        if not isinstance(attribute_values, dict) or not attribute_values:
            raise ValueError("attribute_values must be a non-empty dict")

        
        # Mentioned values help us keep the most relevant domain values when an attribute list is longer than n_houses.
        self.domain_truncations: List[str] = []
        mentioned_norm = set(mentioned_norm_values or [])

        self.attribute_values: Dict[str, List[str]] = {}
        for attr, vals in attribute_values.items():
            if not isinstance(vals, list) or len(vals) == 0:
                continue

            vals_clean, note = _prioritize_and_truncate_values(
                [str(v) for v in vals],
                n_houses=self.n,
                mentioned_norm_values=mentioned_norm if mentioned_norm else None,
            )
            if note is not None:
                self.domain_truncations.append(f"{attr}: {note}")
            if len(vals_clean) == 0:
                continue
            self.attribute_values[str(attr)] = vals_clean

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
            inner: List[BoolRef] = [self.stmt_to_z3(st) for st in a]
            return And(inner) if inner else BoolVal(True)

        if op == "set":
            h, attr, val = a
            if _norm_token(str(attr)) == "house":
                k = _as_int_maybe(str(val))
                if k is None:
                    return BoolVal(True)
                return BoolVal(int(h) == int(k))
            return self.house_of(attr, val) == int(h)

        if op in ("not", "not_set"):
            h, attr, val = a
            if _norm_token(str(attr)) == "house":
                k = _as_int_maybe(str(val))
                if k is None:
                    return BoolVal(True)
                return BoolVal(int(h) != int(k))
            return self.house_of(attr, val) != int(h)

        if op in ("left_of", "right_of", "immediately_left_of", "immediately_right_of", "same_house", "not_same_house"):
            t1, t2 = a
            h1 = _house_expr(t1, self)
            h2 = _house_expr(t2, self)

            if op == "left_of":
                return h1 < h2
            if op == "right_of":
                return h1 > h2
            if op == "same_house":
                return h1 == h2
            if op == "not_same_house":
                return h1 != h2
            if op == "immediately_left_of":
                return h1 + 1 == h2
            if op == "immediately_right_of":
                return h2 + 1 == h1

        if op in ("adjacent_to", "adjacent"):
            t1, t2, k = a
            h1 = _house_expr(t1, self)
            h2 = _house_expr(t2, self)
            return Abs(h1 - h2) == int(k)

        if op == "not_same_house":
            t1, t2 = a
            return _house_expr(t1, self) != _house_expr(t2, self)

        if op == "same_house_term":
            t1, t2 = a
            return _house_expr(t1, self) == _house_expr(t2, self)

        if op == "not_same_house_term":
            t1, t2 = a
            return _house_expr(t1, self) != _house_expr(t2, self)

        if op == "between":
            t1, t2, k_between = a
            dist = int(k_between) + 1
            h1 = _house_expr(t1, self)
            h2 = _house_expr(t2, self)
            return Abs(h1 - h2) == dist

        raise ValueError(f"Unsupported stmt op: {op!r}")

    def add_parsed_clues(
        self,
        parsed_clues: List[str],
        *,
        conflict_tolerant_clues: bool = False,
        conflict_tolerant: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Add clue constraints.

        If conflict_tolerant_clues=True:
          - Add clues one-by-one.
          - If adding a clue makes the system UNSAT, skip that clue and continue.

        Returns:
          (parse_errors, skipped_oov, skipped_underconstrained, skipped_conflict)
        """
        parse_errors: List[str] = []
        skipped_oov: List[str] = []
        skipped_under: List[str] = []
        skipped_conflict: List[str] = []

        if conflict_tolerant is not None:
            conflict_tolerant_clues = bool(conflict_tolerant)
        # Captures the (line, z3_constraint) pairs actually added to the solver.
        self.added_clue_constraints: List[Tuple[str, BoolRef]] = []

        for line in parsed_clues or []:
            try:
                st = parse_dsl_statement(line, attribute_values=self.attribute_values)
                if st is None:
                    continue
                if st.op == "skip":
                    skipped_under.append(f"{line!r} -> SKIPPED_UNDERCONSTRAINED")
                    continue

                phi = self.stmt_to_z3(st)

                if not conflict_tolerant_clues:
                    self.solver.add(phi)
                    self.added_clue_constraints.append((line, phi))
                    continue

                # Conflict-tolerant mode: check satisfiable before committing
                tmp = Solver()
                tmp.append(self.solver.assertions())
                if timeout_ms is not None:
                    tmp.set(timeout=timeout_ms)
                tmp.add(phi)
                r = tmp.check()
                if r == unsat:
                    skipped_conflict.append(f"{line!r} -> SKIPPED_CONFLICT: would make base UNSAT")
                    continue

                # sat or unknown -> keep
                self.solver.add(phi)
                self.added_clue_constraints.append((line, phi))

            except KeyError as e:
                skipped_oov.append(f"{line!r} -> SKIPPED_OOV: {e}")
            except Exception as e:
                parse_errors.append(f"{line!r} -> {type(e).__name__}: {e}")

        return parse_errors, skipped_oov, skipped_under, skipped_conflict


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



# ----------------------- Ground-truth solution validation -----------------------

def _solution_to_map(sol: Any) -> Optional[Dict[int, Dict[str, str]]]:
    """Convert solution table {header:[...], rows:[[...],...]} into {house:int -> {Attr: Val}}."""
    if not isinstance(sol, dict):
        return None
    header = sol.get("header")
    rows = sol.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list):
        return None
    # find House column
    try:
        h_idx = [str(x) for x in header].index("House")
    except Exception:
        try:
            h_idx = [str(x).lower() for x in header].index("house")
        except Exception:
            return None

    out: Dict[int, Dict[str, str]] = {}
    for r in rows:
        if not isinstance(r, list) or len(r) != len(header):
            continue
        h_raw = r[h_idx]
        h = _as_int_maybe(str(h_raw))
        if h is None:
            continue
        row_map: Dict[str, str] = {}
        for j, col in enumerate(header):
            if j == h_idx:
                continue
            row_map[str(col)] = str(r[j])
        out[int(h)] = row_map
    return out if out else None


def validate_solution_against_ground_truth(
    *,
    n_houses: int,
    predicted_solution: Any,
    ground_truth: Any,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Return (is_valid, details). is_valid is None if inputs missing/unusable."""
    pred_map = _solution_to_map(predicted_solution)
    gt_map = _solution_to_map(ground_truth)
    details: Dict[str, Any] = {
        "gt_check_available": bool(gt_map),
        "pred_check_available": bool(pred_map),
        "mismatches": [],
    }

    if pred_map is None or gt_map is None:
        return None, details

    mismatches = []
    for h in range(1, int(n_houses) + 1):
        gt_row = gt_map.get(h)
        pred_row = pred_map.get(h)
        if gt_row is None or pred_row is None:
            mismatches.append({
                "house": h,
                "type": "missing_house",
                "gt_present": gt_row is not None,
                "pred_present": pred_row is not None,
            })
            continue
        for attr, gt_val in gt_row.items():
            pred_val = pred_row.get(attr)
            if pred_val is None:
                mismatches.append({"house": h, "attr": attr, "type": "missing_attr", "gt": gt_val})
                continue
            if _norm_token(pred_val) != _norm_token(gt_val):
                mismatches.append({"house": h, "attr": attr, "type": "value_mismatch", "gt": gt_val, "pred": pred_val})

    details["mismatches"] = mismatches[:50]
    return (len(mismatches) == 0), details

def validate_reasoning_steps(
    *,
    n_houses: Optional[int],
    attribute_values: Optional[Dict[str, List[str]]],
    parsed_clues: Optional[List[str]],
    parsed_reasoning: Optional[List[str]],
    # new inputs
    predicted_solution: Optional[Dict[str, Any]] = None,
    ground_truth_solution: Optional[Dict[str, Any]] = None,

    timeout_s: float = 1.0,
    conflict_tolerant_clues: bool = False,
) -> Dict[str, Any]:
    out_fail: Dict[str, Any] = {
        "base_sat": None,
        "base_sat_raw": None,
        "base_sat_full": None,
        "base_unsat_debug": {},
        "clue_parse_errors": [],
        "clue_skipped_oov": [],
        "clue_skipped_underconstrained": [],
        "clue_skipped_conflict": [],
        "steps": [],
        "n_steps_total": 0,
        "n_steps_parsed_ok": 0,
        "n_steps_entailed_chain": 0,
        "n_steps_contradiction_chain": 0,
        "n_steps_entailed_strict": 0,

    }

    # ---- Deployment safety guards ----
    # Coerce / validate inputs early (prevents NoneType / bad int crashes)
    try:
        nh = int(n_houses)
    except Exception:
        out_fail["base_unsat_debug"] = {
            "fatal_error": "INVALID_N_HOUSES",
            "n_houses": n_houses,
            "n_houses_type": str(type(n_houses)),
        }
        return out_fail

    if attribute_values is None or (not isinstance(attribute_values, dict)) or len(attribute_values) == 0:
        out_fail["base_unsat_debug"] = {
            "fatal_error": "MISSING_ATTRIBUTE_VALUES",
            "attribute_values_type": str(type(attribute_values)),
            "attribute_values_len": (len(attribute_values) if isinstance(attribute_values, dict) else None),
        }
        return out_fail

    # Remove empty/invalid attribute lists (otherwise Z3PuzzleModel raises)
    av_clean = {}
    for a, vals in attribute_values.items():
        if isinstance(vals, list) and len(vals) > 0:
            av_clean[str(a)] = [str(v) for v in vals if v is not None and str(v).strip() != ""]
    if len(av_clean) == 0:
        out_fail["base_unsat_debug"] = {
            "fatal_error": "NO_USABLE_ATTRIBUTE_LISTS",
        }
        return out_fail
    try:
        timeout_ms = int(max(0.0, float(timeout_s)) * 1000)

        mentioned_norm_values = _extract_mentioned_values((parsed_clues or []) + (parsed_reasoning or []))

        model = Z3PuzzleModel(
            n_houses=nh,
            attribute_values=av_clean,
            mentioned_norm_values=mentioned_norm_values,
        )

        # RAW satisfiability: domains + uniqueness constraints only (no clues)
        raw_check = model.solver.check()
        raw_sat: Optional[bool] = None if raw_check == unknown else (raw_check == sat)

        # Add clue constraints (optionally skipping contradictions)
        clue_parse_errors, clue_skipped_oov, clue_skipped_under, clue_skipped_conflict = model.add_parsed_clues(
            parsed_clues or [],
            conflict_tolerant_clues=conflict_tolerant_clues,
            timeout_ms=timeout_ms,
        )

        # FULL satisfiability: after adding (accepted) clues
        full_check = model.solver.check()
        full_sat: Optional[bool] = None if full_check == unknown else (full_check == sat)

        # Debug / diagnostics for failing base satisfiability
        input_attr_stats: Dict[str, Dict[str, int]] = {}
        for a, vals in (av_clean or {}).items():
            if not isinstance(vals, list):
                continue
            vals_str = [str(v) for v in vals]
            vals_norm = [_norm_token(v) for v in vals_str]
            input_attr_stats[str(a)] = {
                "n_vals": len(vals_str),
                "n_unique": len(set(vals_norm)),
            }

        base_unsat_debug: Dict[str, Any] = {
            "n_houses": int(nh),
            "domain_truncations": list(getattr(model, "domain_truncations", [])),
            "input_attr_stats": input_attr_stats,
        }

        if raw_sat is False:
            base_unsat_debug["reason"] = "RAW_UNSAT"
        elif full_sat is False:
            base_unsat_debug["reason"] = "FULL_UNSAT"
            # Try to find the first accepted clue that makes the system UNSAT (best-effort).
            try:
                base_only = Z3PuzzleModel(
                    n_houses=n_houses,
                    attribute_values=attribute_values,
                    mentioned_norm_values=mentioned_norm_values,
                )
                s_tmp = Solver()
                s_tmp.append(base_only.solver.assertions())
                if timeout_ms is not None:
                    s_tmp.set(timeout=timeout_ms)

                culprit = None
                for j, (cl_line, cl_phi) in enumerate(getattr(model, "added_clue_constraints", []), start=1):
                    s_tmp.push()
                    s_tmp.add(cl_phi)
                    r = s_tmp.check()
                    if r == unsat:
                        culprit = {"index": j, "clue": cl_line}
                        s_tmp.pop()
                        break
                    s_tmp.pop()
                    # Now add permanently for prefix building
                    s_tmp.add(cl_phi)

                base_unsat_debug["first_unsat_clue"] = culprit
            except Exception as e:
                base_unsat_debug["first_unsat_clue_error"] = f"{type(e).__name__}: {e}"

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

                if raw_sat is False:
                    strict_status = "RAW_UNSAT"
                    chain_status = "RAW_UNSAT"
                elif full_sat is False:
                    strict_status = "FULL_UNSAT"
                    chain_status = "FULL_UNSAT"
                else:
                    strict_status = _check_status(model.solver, phi, timeout_ms=timeout_ms)
                    chain_status = _check_status(chain_solver, phi, timeout_ms=timeout_ms)

                if chain_status not in ("CONTRADICTION", "UNKNOWN", "RAW_UNSAT", "FULL_UNSAT", "PARSE_ERROR"):

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

        # ---- New counters
        steps_dicts = [sv.__dict__ for sv in steps_out]
        n_steps_total = len(steps_dicts)
        n_steps_parsed_ok = sum(1 for s in steps_dicts if s.get("parsed_ok") is True)
        n_steps_entailed_chain = sum(1 for s in steps_dicts if s.get("chain_status") == "ENTAILED")
        n_steps_contradiction_chain = sum(1 for s in steps_dicts if s.get("chain_status") == "CONTRADICTION")

        n_steps_entailed_strict = sum(1 for s in steps_dicts if s.get("strict_status") == "ENTAILED")

        gt_valid, gt_details = validate_solution_against_ground_truth(
            n_houses=int(n_houses) if n_houses is not None else 0,
            predicted_solution=predicted_solution,
            ground_truth=ground_truth_solution,
        )
        # Per spec: score = 1.0 if step passed AND gt_valid; 0.2 if step passed AND gt_valid is False; else 0.0
        if gt_valid is True:
            _gt_factor = 1.0
        elif gt_valid is False:
            _gt_factor = 0.2
        else:
            _gt_factor = 0.0

        _den = max(1, n_steps_total)
        pass_rate_strict = (float(n_steps_entailed_strict) / float(_den))
        pass_rate_chain = (float(n_steps_entailed_chain) / float(_den))

        pass_rate_strict_gt_validated = (float(n_steps_entailed_strict) / float(_den)) * _gt_factor
        pass_rate_chain_gt_validated = (float(n_steps_entailed_chain) / float(_den)) * _gt_factor


        return {
            "base_sat": full_sat,  # kept for backward compatibility
            "base_sat_raw": raw_sat,
            "base_sat_full": full_sat,
            "gt_solution_valid": gt_valid,
            "gt_solution_details": gt_details,
            "pass_rate_strict": pass_rate_strict,
            "pass_rate_chain": pass_rate_chain,
            "n_steps_entailed_strict": n_steps_entailed_strict,
            "pass_rate_strict_gt_validated": pass_rate_strict_gt_validated,
            "pass_rate_chain_gt_validated": pass_rate_chain_gt_validated,
            "base_unsat_debug": base_unsat_debug,
            "clue_parse_errors": clue_parse_errors,
            "clue_skipped_oov": clue_skipped_oov,
            "clue_skipped_underconstrained": clue_skipped_under,
            "clue_skipped_conflict": clue_skipped_conflict,  # NEW
            "steps": steps_dicts,
            "n_steps_total": n_steps_total,
            "n_steps_parsed_ok": n_steps_parsed_ok,
            "n_steps_entailed_chain": n_steps_entailed_chain,
            "n_steps_contradiction_chain": n_steps_contradiction_chain,
        }

    except Exception as e:
        out_fail["base_unsat_debug"] = {
            "fatal_error": f"{type(e).__name__}: {e}",
        }
        return out_fail


def main():
    # ----------------------------
    # Minimal toy puzzle (3 houses)
    # ----------------------------
    n_houses = 3
    attribute_values = {
        "Name": ["Arnold", "Peter", "Eric"],
        "Drink": ["tea", "water", "milk"],
    }

    # Clues:
    # 1) Peter is in house 2
    # 2) Arnold is immediately left of water drinker
    # 3) water is immediately left of milk
    parsed_clues = [
        "C1 = set(2,Name,Peter).",
        "C2 = immediately_left_of(Name=Arnold,Drink=water).",
        "C3 = immediately_left_of(Drink=water,Drink=milk).",
    ]

    # Reasoning steps (some are entailed)
    parsed_reasoning = [
        "S1 [C1] set(2,Name,Peter).",
        # from C3: water cannot be in house 3
        "S2 [C3] not_set(3,Drink,water).",
        # from C3: milk cannot be in house 1
        "S3 [C3] not_set(1,Drink,milk).",
        # from C2: Arnold cannot be in house 3
        "S4 [C2] not_set(3,Name,Arnold).",
        # from C2+C3 you can deduce Arnold must be in house 1
        "S5 [C2+C3] set(1,Name,Arnold).",
    ]

    # -----------------------------------------
    # Predicted solution + Ground truth solution
    # -----------------------------------------
    # (Both are in your "solution" table format)
    ground_truth_solution = {
        "header": ["House", "Name", "Drink"],
        "rows": [
            ["1", "Arnold", "tea"],
            ["2", "Peter", "water"],
            ["3", "Eric", "milk"],
        ],
    }

    # Try changing this to something wrong (e.g., swap drinks) to see GT penalty effects
    predicted_solution = {
        "header": ["House", "Name", "Drink"],
        "rows": [
            ["1", "Arnold", "tea"],
            ["2", "Peter", "water"],
            ["3", "Eric", "milk"],
        ],
    }

    # ----------------------------
    # Run validator
    # ----------------------------
    z3_out = validate_reasoning_steps(
        n_houses=n_houses,
        attribute_values=attribute_values,
        parsed_clues=parsed_clues,
        parsed_reasoning=parsed_reasoning,
        predicted_solution=predicted_solution,
        ground_truth_solution=ground_truth_solution,
        timeout_s=2.0,
        conflict_tolerant_clues=False,
    )

    # ----------------------------
    # Print key outputs
    # ----------------------------
    print("\n=== Key summary ===")
    print("base_sat_raw:", z3_out.get("base_sat_raw"))
    print("base_sat_full:", z3_out.get("base_sat_full"))
    print("gt_solution_valid:", z3_out.get("gt_solution_valid"))
    print("pass_rate_strict:", z3_out.get("pass_rate_strict"))
    print("pass_rate_chain:", z3_out.get("pass_rate_chain"))
    print("pass_rate_strict_gt_validated:", z3_out.get("pass_rate_strict_gt_validated"))
    print("pass_rate_chain_gt_validated:", z3_out.get("pass_rate_chain_gt_validated"))

    print("\n=== First 3 steps (if any) ===")
    for s in (z3_out.get("steps") or [])[:3]:
        pprint(
            {
                "index": s.get("index"),
                "parsed_ok": s.get("parsed_ok"),
                "strict_status": s.get("strict_status"),
                "chain_status": s.get("chain_status"),
                "parse_error": s.get("parse_error"),
                "raw": s.get("raw"),
            }
        )


if __name__ == "__main__":
    main()