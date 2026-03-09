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




def _try_parse_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Try parsing JSON object from text.
    - fast path: json.loads(text)
    - fallback: brute scan for the first valid {...} object
    """
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    starts = [m.start() for m in re.finditer(r'\{', text)]
    for st in starts:
        for ed in range(len(text), st, -1):
            if text[ed - 1] != '}':
                continue
            chunk = text[st:ed]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def find_last_answer_block(text: str) -> Optional[str]:
    """
    Returns the *last* <answer>...</answer> block found in `text`,
    or None if no block exists.

    - Case-insensitive tags: <answer> or <ANSWER>
    - Allows attributes in the opening tag: <answer id="x">
    - Dot matches newlines so multi-line blocks work
    """
    pattern = re.compile(
        r"<answer\b[^>]*>.*?</answer\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return None

    return matches[-1].group(0)


def extract_reasoning_and_solution(solution_str: str):
    """
    Extract both reasoning and solution.
    Returns (reasoning, solution, status)
    """
    answer_content = find_last_answer_block(solution_str)
    if answer_content:
        parsed = _try_parse_first_json_obj(answer_content)
        if parsed is not None:
            return parsed.get("parsed_clues", None), parsed.get("parsed_reasoning", None), parsed.get("solution", None), parsed.get("attribute_values", None), parsed.get("n_houses",
                                                                                                                                                                          None), "success_answer_tag"
        return None, None, None, None, None, "answer_tag_json_error"

    parsed = _try_parse_first_json_obj(solution_str)
    if parsed is not None:
        return parsed.get("parsed_clues", None), parsed.get("parsed_reasoning", None), parsed.get("solution", None), parsed.get("attribute_values", None), parsed.get("n_houses",
                                                                                                                                                                      None), "success_direct_json"

    return None, None, None, None, None, "parsing_failed"


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
    if op_norm == "not_set":
        op_norm = "not"
    if op_norm in ("adjacent", "next_to", "adjacentto"):
        op_norm = "adjacent_to"
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
    if op_norm == "not":
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
    if op_norm in ("left_of", "right_of", "immediately_left_of", "immediately_right_of", "adjacent_to", "same_house"):
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


def validate_reasoning_steps(
    n_houses: int,
    attribute_values: Dict[str, List[str]],
    parsed_clues: List[str],
    parsed_reasoning: List[str],
    timeout_s: float = 1.0,
    conflict_tolerant_clues: bool = False,   # NEW
) -> Dict[str, Any]:
    timeout_ms = int(max(0.0, float(timeout_s)) * 1000)

    mentioned_norm_values = _extract_mentioned_values((parsed_clues or []) + (parsed_reasoning or []))

    model = Z3PuzzleModel(
        n_houses=n_houses,
        attribute_values=attribute_values,
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
    for a, vals in (attribute_values or {}).items():
        if not isinstance(vals, list):
            continue
        vals_str = [str(v) for v in vals]
        vals_norm = [_norm_token(v) for v in vals_str]
        input_attr_stats[str(a)] = {
            "n_vals": len(vals_str),
            "n_unique": len(set(vals_norm)),
        }

    base_unsat_debug: Dict[str, Any] = {
        "n_houses": int(n_houses),
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

    return {
        "base_sat": full_sat,  # kept for backward compatibility
        "base_sat_raw": raw_sat,
        "base_sat_full": full_sat,
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



def main():
    print("\n" + "=" * 88)
    print("Z3 reasoning validator (v10 demo)\n")

    # Default to the uploaded demo file if it exists; otherwise keep your previous hardcoded path.
    default_file = os.environ.get("JSONL_PATH", "/mnt/data/75.jsonl")

    if os.path.exists(default_file):
        file_name = default_file
    else:
        # Fallback to your original path (edit as needed)
        fpath = "/home/asif/data3/Codes_QCRI/Reasoning360/evaluation_results/small_train_small_test_1_parsed_v2/qwen2515binstruct/"
        fname = "jobid_243455/75.jsonl"
        file_name = os.path.join(fpath, fname)

    with open(file_name, "r", encoding="utf-8") as f:
        json_data = [json.loads(line) for line in f if line.strip()]

    print(f"Input file: {file_name}")
    print(f"Total records: {len(json_data)}")
    print("\n" + "=" * 88)

    count_raw_sat = 0
    count_raw_unsat = 0
    count_raw_unknown = 0
    count_full_sat = 0
    count_full_unsat = 0
    count_full_unknown = 0

    n_fail_logs = 0
    MAX_FAIL_LOGS = int(os.environ.get("MAX_FAIL_LOGS", "25"))

    for idx, line in enumerate(json_data):
        llm_output = line.get("output", "")
        puzzle_acc = line.get("PUZZLE_ACCURACY", None)

        parsed_clues, parsed_reasoning, predicted_arrangement, attribute_values, n_houses, parse_status = extract_reasoning_and_solution(
            solution_str=llm_output
        )
        print(f"n_houses {n_houses}")
        valid_out = validate_reasoning_steps(
            n_houses=n_houses,
            attribute_values=attribute_values,
            parsed_clues=parsed_clues,
            parsed_reasoning=parsed_reasoning,
            timeout_s=5.0,
            conflict_tolerant_clues=True,
        )

        raw_sat = valid_out.get("base_sat_raw", None)
        full_sat = valid_out.get("base_sat_full", None)

        if raw_sat is True:
            count_raw_sat += 1
        elif raw_sat is False:
            count_raw_unsat += 1
        else:
            count_raw_unknown += 1

        if full_sat is True:
            count_full_sat += 1
        elif full_sat is False:
            count_full_unsat += 1
        else:
            count_full_unknown += 1

        # Log a few failing cases with diagnostics
        if n_fail_logs < MAX_FAIL_LOGS and (raw_sat is False or full_sat is False):
            n_fail_logs += 1
            dbg = valid_out.get("base_unsat_debug", {})

            print("\n" + "-" * 88)
            print(f"[FAIL {n_fail_logs}] idx={idx} puzzle_acc={puzzle_acc} raw_sat={raw_sat} full_sat={full_sat}")
            print(f"n_houses={dbg.get('n_houses')}")
            if dbg.get("domain_truncations"):
                print("domain_truncations:", dbg.get("domain_truncations"))
            if dbg.get("first_unsat_clue"):
                print("first_unsat_clue:", dbg.get("first_unsat_clue"))

            # Show suspicious attrs (n_vals != n_houses or duplicates)
            stats = dbg.get("input_attr_stats") or {}
            suspicious = []
            for a, st in stats.items():
                n_vals = st.get("n_vals", 0)
                n_unique = st.get("n_unique", 0)
                if (n_houses is not None and n_vals != int(n_houses)) or (n_unique != n_vals):
                    suspicious.append((a, st))
            if suspicious:
                print("suspicious_input_attr_stats:", suspicious[:12])

    print("\n" + "=" * 88)
    print(f"raw_sat=True   : {count_raw_sat}")
    print(f"raw_sat=False  : {count_raw_unsat}")
    print(f"raw_sat=None   : {count_raw_unknown}")
    print(f"full_sat=True  : {count_full_sat}")
    print(f"full_sat=False : {count_full_unsat}")
    print(f"full_sat=None  : {count_full_unknown}")


if __name__ == "__main__":
    main()
