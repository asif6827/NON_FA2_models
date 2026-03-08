# -*- coding: utf-8 -*-
"""
z3_reasoning_validator_demo_v8.py

A tolerant Z3 validator for Zebra/Logic puzzles.

This version adds summary counters for reasoning-step validation:

Added keys in validate_reasoning_steps(...) output:
- n_steps_total
- n_steps_parsed_ok
- n_steps_entailed_chain
- n_steps_contradiction_chain

Quick README (output schema)
----------------------------
validate_reasoning_steps(...) returns a dict with:

Core:
- base_sat: bool | None
    Whether the parsed_clues constraints are satisfiable under the base model.
    True  => SAT
    False => UNSAT
    None  => UNKNOWN (timeout/solver unknown)

Clue diagnostics:
- clue_parse_errors: List[str]
    True syntax/structural parsing errors for clue lines.
- clue_skipped_oov: List[str]
    Skipped due to unknown attribute/value (out-of-vocabulary).
- clue_skipped_underconstrained: List[str]
    Skipped because the clue didn't contain enough information to encode.

Per-step validation:
- steps: List[dict]
    One item per reasoning step in parsed_reasoning. Each item contains:
    - index: 1-based step index
    - raw: raw step string
    - parsed_ok: whether we parsed it into a supported DSL statement
    - parse_error: error message if parsed_ok is False
    - strict_status: status wrt ONLY base clues (ENTAILED / NOT_ENTAILED / CONTRADICTION / UNKNOWN / PARSE_ERROR / BASE_UNSAT)
    - chain_status: status wrt base clues + previously accepted chain steps
    - strict_valid: True/False/None mapping of strict_status (True only if ENTAILED)
    - chain_valid: True/False/None mapping of chain_status (True only if ENTAILED)

Summary counters:
- n_steps_total: int
    Total number of produced step records (== len(steps)).
- n_steps_parsed_ok: int
    How many steps were parsed successfully (parsed_ok == True).
- n_steps_entailed_chain: int
    How many steps are ENTAILED under the incremental chain context.
- n_steps_contradiction_chain: int
    How many steps are CONTRADICTION under the incremental chain context.

Notes:
- If base_sat is False, step statuses are reported as BASE_UNSAT (because the baseline constraint set itself is inconsistent).
- Underconstrained steps are treated as PARSE_ERROR entries in steps (parsed_ok=False), and therefore do not increase n_steps_parsed_ok.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from z3 import Abs, And, BoolRef, BoolVal, Distinct, Int, IntVal, Not, Solver, sat, unsat, unknown

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
    t = tok.strip()
    if t == "":
        raise ValueError("Empty term")

    i = _as_int_maybe(t)
    if i is not None and ("=" not in t):
        return Term(kind="house_const", house=i)

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

        t1 = _parse_term(args_raw[0], attribute_values)
        t2 = _parse_term(args_raw[1], attribute_values)

        if op_norm == "adjacent_to":
            k = 1
            for extra in args_raw[2:]:
                if "=" in extra:
                    kk, vv = _parse_kv(extra)
                    if _norm_token(kk) == "k":
                        k = _as_int_maybe(vv) or k
                else:
                    maybe = _as_int_maybe(extra)
                    if maybe is not None:
                        k = maybe
            return ParsedStmt("adjacent_to", (t1, t2, int(k)), line)

        return ParsedStmt(op_norm, (t1, t2), line)

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


# ----------------------------- Z3 builder ----------------------------------

class Z3PuzzleModel:
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

    def add_parsed_clues(self, parsed_clues: List[str]) -> Tuple[List[str], List[str], List[str]]:
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
    strict_status: str
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

    steps_dicts = [sv.__dict__ for sv in steps_out]

    # ---- NEW: summary counters ----
    n_steps_total = len(steps_dicts)
    n_steps_parsed_ok = sum(1 for s in steps_dicts if s.get("parsed_ok") is True)
    n_steps_entailed_chain = sum(1 for s in steps_dicts if s.get("chain_status") == "ENTAILED")
    n_steps_contradiction_chain = sum(1 for s in steps_dicts if s.get("chain_status") == "CONTRADICTION")

    return {
        "base_sat": base_sat,
        "clue_parse_errors": clue_parse_errors,
        "clue_skipped_oov": clue_skipped_oov,
        "clue_skipped_underconstrained": clue_skipped_under,
        "steps": steps_dicts,
        "n_steps_total": n_steps_total,
        "n_steps_parsed_ok": n_steps_parsed_ok,
        "n_steps_entailed_chain": n_steps_entailed_chain,
        "n_steps_contradiction_chain": n_steps_contradiction_chain,
    }


def main():
    print("\n" + "=" * 88)

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
