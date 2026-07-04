#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR-LSAT ASSIGNMENT Z3 validator for reward computation.

Supports assignment-style formulas:
  Assign(A, P1), Not(Assign(A, P1)), Assign(B, P1) == Assign(C, P1),
  Exactly/AtLeast/AtMost, Sat(Option_A), Unsat(Not(Option_A)).

Important: BASE_sat_full_GT is intentionally strict:
  base rules/facts SAT AND selected option parseable AND selected option satisfies
  question_type semantics AND normalized selected option == normalized ground truth.
"""
from __future__ import annotations

import json, logging, re, sys
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import z3
    from z3 import And, AtLeast, AtMost, Bool, BoolVal, Implies, Not, Or, PbEq, Solver, Xor, sat, unsat
except Exception:  # pragma: no cover
    z3 = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_FUNC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)


def normalize_header(data_sample): return data_sample

def normalize_months_in_rows(z3_solution: dict) -> dict: return z3_solution


def _norm_token(x: Any) -> str:
    s = str(x).strip().strip("`'\"“”‘’")
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return re.sub(r"_+", "_", s).strip("_")


def _norm_option_label(x: Any) -> Optional[str]:
    """Normalize selected/ground-truth/option labels to A-E.

    Handles: A, Option_A, option-a, Answer: B, selected_option C,
    and strings like "The correct answer is D". Prefer explicit/final
    option letters rather than the first letter in prose (e.g. Choice D
    should become D, not C).
    """
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None

    compact = s.replace("-", "_").replace(" ", "_").replace(":", "_")
    m = re.fullmatch(r"(?:SELECTED_)?(?:OPTION|ANSWER)?_?([A-E])", compact)
    if m:
        return m.group(1)

    m = re.search(r"(?:OPTION|ANSWER|CHOICE|SELECTED_OPTION)\s*[:_\-\s]*([A-E])\b", s)
    if m:
        return m.group(1)

    m = re.search(r"\b([A-E])\b\s*[\.)\]]?\s*$", s)
    if m:
        return m.group(1)

    letters = re.findall(r"\b([A-E])\b", s)
    if letters:
        return letters[-1]

    return None

def _split_top_level_args(s: str) -> List[str]:
    args, buf, depth = [], [], 0
    for ch in s:
        if ch == '(':
            depth += 1; buf.append(ch)
        elif ch == ')':
            depth -= 1
            if depth < 0: raise ValueError('Unbalanced parentheses')
            buf.append(ch)
        elif ch == ',' and depth == 0:
            part = ''.join(buf).strip()
            if part: args.append(part)
            buf = []
        else:
            buf.append(ch)
    if depth != 0: raise ValueError('Unbalanced parentheses')
    tail = ''.join(buf).strip()
    if tail: args.append(tail)
    return args


def _split_top_level_binary(expr: str, op: str) -> Optional[Tuple[str, str]]:
    depth = 0; i = 0
    while i <= len(expr) - len(op):
        ch = expr[i]
        if ch == '(':
            depth += 1; i += 1; continue
        if ch == ')':
            depth -= 1
            if depth < 0: raise ValueError('Unbalanced parentheses')
            i += 1; continue
        if depth == 0 and expr.startswith(op, i):
            left, right = expr[:i].strip(), expr[i+len(op):].strip()
            if left and right: return left, right
        i += 1
    return None


def _selected_from_ground_truth(gt: Any) -> Optional[str]:
    if isinstance(gt, str): return _norm_option_label(gt)
    if isinstance(gt, dict):
        for k in ('answer', 'selected_option', 'ground_truth_option'):
            if gt.get(k) is not None: return _norm_option_label(gt[k])
    return None


def _selected_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    sol = payload.get('solution') or {}
    if isinstance(sol, dict) and sol.get('selected_option') is not None:
        return _norm_option_label(sol['selected_option'])
    return None


def _solver_check(assertions: List[Any], timeout_s: float):
    s = Solver(); s.set('timeout', int(timeout_s * 1000)); s.add(assertions); return s.check()

def _is_sat(assertions: List[Any], timeout_s: float) -> bool: return _solver_check(assertions, timeout_s) == sat

def _is_unsat(assertions: List[Any], timeout_s: float) -> bool: return _solver_check(assertions, timeout_s) == unsat


def _status_under_gamma(gamma: List[Any], phi: Any, timeout_s: float) -> str:
    if not _is_sat(gamma, timeout_s): return 'PREMISES_UNSAT'
    if _is_unsat(gamma + [phi], timeout_s): return 'CONTRADICTION'
    if _is_unsat(gamma + [Not(phi)], timeout_s): return 'ENTAILED'
    return 'NOT_ENTAILED'


def _is_tautology(base: List[Any], phi: Any, timeout_s: float) -> bool:
    return _is_sat(base, timeout_s) and _is_unsat(base + [Not(phi)], timeout_s)


def _equiv_to_any_rule(base: List[Any], phi: Any, rules: List[Any], timeout_s: float) -> bool:
    for r in rules:
        try:
            if _is_unsat(base + [Xor(phi, r)], timeout_s): return True
        except Exception:
            pass
    return False


def _evaluate_option(question_type: str, option_phi: Any, gamma: List[Any], timeout_s: float) -> bool:
    qt = (question_type or '').strip().lower()
    if qt in {'could_be_true', 'acceptability', 'partial_acceptability', 'valid_complete_assignment'}:
        return _is_sat(gamma + [option_phi], timeout_s)
    if qt in {'must_be_true', 'must_follow'}:
        return _is_unsat(gamma + [Not(option_phi)], timeout_s)
    if qt in {'cannot_be_true', 'must_be_false'}:
        return _is_unsat(gamma + [option_phi], timeout_s)
    if qt == 'could_be_false':
        return _is_sat(gamma + [Not(option_phi)], timeout_s)
    return _is_sat(gamma + [option_phi], timeout_s)


def _extract_step_expr(line: str) -> Optional[Tuple[int, str]]:
    m = _STEP_RE.match((line or '').strip())
    if not m: return None
    expr = m.group(2).strip()
    if '[' in expr: expr = expr.split('[', 1)[0].strip()
    expr = expr.rstrip('.').strip()
    return (int(m.group(1)), expr) if expr else None


def _option_status_expr(expr: str) -> Optional[Tuple[str, bool, str]]:
    e = (expr or '').strip().rstrip('.')
    m = re.fullmatch(r"\s*(Sat|Unsat)\(\s*(Not\()?\s*Option[_\-\s:]*([A-Z])\s*\)?\s*\)\s*", e, flags=re.IGNORECASE)
    if not m: return None
    return m.group(1).lower(), bool(m.group(2)), m.group(3).upper()


def _option_status_is_true(status: str, is_negated: bool, opt_phi: Any, gamma: List[Any], timeout_s: float) -> bool:
    phi = Not(opt_phi) if is_negated else opt_phi
    return _is_sat(gamma + [phi], timeout_s) if status == 'sat' else _is_unsat(gamma + [phi], timeout_s)


def _supports_selected_solution(question_type: str, status: str, is_negated: bool, label: str, selected: Optional[str]) -> bool:
    if not selected or label != selected: return False
    qt = (question_type or '').strip().lower()
    if qt in {'could_be_true', 'acceptability', 'partial_acceptability', 'valid_complete_assignment'}: return status == 'sat' and not is_negated
    if qt in {'must_be_true', 'must_follow'}: return status == 'unsat' and is_negated
    if qt in {'cannot_be_true', 'must_be_false'}: return status == 'unsat' and not is_negated
    if qt == 'could_be_false': return status == 'sat' and is_negated
    return status == 'sat' and not is_negated


def _extract_entities_values(world_model: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    entities = [_norm_token(e) for e in (world_model.get('entities') or [])]
    domains = world_model.get('domains') or {}
    raw_values = []
    for key in ('values', 'assignments', 'projects', 'rooms', 'days', 'colors', 'groups', 'slots', 'tasks', 'courses', 'teams'):
        if isinstance(domains.get(key), list) and domains.get(key):
            raw_values = domains.get(key); break
    if not raw_values and isinstance(domains, dict):
        ent_set = set(entities)
        for v in domains.values():
            if isinstance(v, list) and v:
                cand = [_norm_token(x) for x in v]
                if set(cand) != ent_set:
                    raw_values = v; break
    return entities, [_norm_token(v) for v in raw_values]


def _assignment_value_exactly_one_required(world_model: Dict[str, Any]) -> bool:
    assumptions = world_model.get('structural_assumptions') or []
    if isinstance(assumptions, str): assumptions = [assumptions]
    text = ' '.join(str(x).lower() for x in assumptions)
    return bool('one-to-one' in text or 'bijective' in text or re.search(r"(each|every)\s+(value|project|room|day|slot|task|course|team|group)\b.*\bexactly\s+one\b", text))


def _make_assignment_base(world_model: Dict[str, Any], timeout_s: float):
    if z3 is None: raise RuntimeError('z3-solver is not installed')
    entities, values = _extract_entities_values(world_model)
    if not entities: raise ValueError('world_model.entities is empty')
    if not values: raise ValueError('world_model.domains.values/assignments/projects/etc. is empty')
    assign_vars: Dict[Tuple[str, str], Any] = {(e, v): Bool(f'Assign__{e}__{v}') for e in entities for v in values}
    base_assertions: List[Any] = []
    for e in entities:
        base_assertions.append(PbEq([(assign_vars[(e, v)], 1) for v in values], 1))
    if _assignment_value_exactly_one_required(world_model):
        for v in values:
            base_assertions.append(PbEq([(assign_vars[(e, v)], 1) for e in entities], 1))
    return assign_vars, base_assertions, len(values), len(entities)


def _assignment_var(entity_raw: str, value_raw: str, assign_vars: Dict[Tuple[str, str], Any]):
    e, v = _norm_token(entity_raw), _norm_token(value_raw)
    key = (e, v)
    if key not in assign_vars:
        known_e = sorted({x for x, _ in assign_vars.keys()}); known_v = sorted({y for _, y in assign_vars.keys()})
        raise KeyError(f'Unknown Assign({e}, {v}). Known entities={known_e}; known values={known_v}')
    return assign_vars[key]


def _parse_assignment_expr(expr: str, assign_vars: Dict[Tuple[str, str], Any]):
    e = str(expr).strip().rstrip('.')
    if e.lower() == 'true': return BoolVal(True)
    if e.lower() == 'false': return BoolVal(False)
    for op in ('==', '!='):
        split = _split_top_level_binary(e, op)
        if split:
            L = _parse_assignment_expr(split[0], assign_vars)
            R = _parse_assignment_expr(split[1], assign_vars)
            return (L == R) if op == '==' else (L != R)
    m = _FUNC_RE.match(e)
    if not m: raise ValueError(f'Unrecognized assignment expression: {expr!r}')
    fn, args = m.group(1).lower(), _split_top_level_args(m.group(2))
    if fn == 'assign':
        if len(args) != 2: raise ValueError('Assign expects exactly two arguments')
        return _assignment_var(args[0], args[1], assign_vars)
    if fn == 'not':
        if len(args) != 1: raise ValueError('Not expects one argument')
        return Not(_parse_assignment_expr(args[0], assign_vars))
    if fn == 'and': return And(*[_parse_assignment_expr(a, assign_vars) for a in args])
    if fn == 'or': return Or(*[_parse_assignment_expr(a, assign_vars) for a in args])
    if fn == 'implies':
        if len(args) != 2: raise ValueError('Implies expects two arguments')
        return Implies(_parse_assignment_expr(args[0], assign_vars), _parse_assignment_expr(args[1], assign_vars))
    if fn == 'xor':
        if len(args) != 2: raise ValueError('Xor expects two arguments')
        return Xor(_parse_assignment_expr(args[0], assign_vars), _parse_assignment_expr(args[1], assign_vars))
    if fn in {'exactly', 'atleast', 'atmost'}:
        if len(args) < 2: raise ValueError(f'{fn} expects count plus Boolean arguments')
        k = int(str(args[0]).strip())
        parsed = [_parse_assignment_expr(a, assign_vars) for a in args[1:]]
        if fn == 'exactly': return PbEq([(p, 1) for p in parsed], k)
        if fn == 'atleast': return AtLeast(*parsed, k)
        return AtMost(*parsed, k)
    raise ValueError(f'Unsupported assignment function: {fn}')


def _parse_constraints(lines: List[str], parser_fn: Callable[[str], Any]) -> Tuple[List[Any], List[Dict[str, str]]]:
    phis, errors = [], []
    for raw in lines or []:
        try: phis.append(parser_fn(str(raw)))
        except Exception as e: errors.append({'raw': str(raw), 'error': f'{type(e).__name__}: {e}'})
    return phis, errors


def _parse_options(options: Dict[str, str], parser_fn: Callable[[str], Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    out, errs = {}, []
    for label, expr in (options or {}).items():
        lab = _norm_option_label(label)
        if lab is None:
            errs.append({'label': str(label), 'raw': str(expr), 'error': 'Invalid option label'})
            continue
        try: out[lab] = parser_fn(str(expr))
        except Exception as e: errs.append({'label': lab, 'raw': str(expr), 'error': f'{type(e).__name__}: {e}'})
    return out, errs


def _validate_reasoning_steps(reasoning, *, parser_fn, base_assertions, rule_fact_phis, rule_phis, option_phis, question_type, selected_option, timeout_s):
    gamma_valid = base_assertions + rule_fact_phis; gamma_steps = list(base_assertions); seen = set()
    n_total = n_parsed = 0; valid_steps=[]; novel_steps=[]; non_valid=[]; parse_errors=[]; support=[]
    for line in reasoning or []:
        parsed = _extract_step_expr(line)
        if parsed is None: continue
        n_total += 1; k, expr = parsed
        st = _option_status_expr(expr)
        if st:
            n_parsed += 1; status, is_negated, label = st; opt_phi = option_phis.get(label)
            if opt_phi is None:
                non_valid.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': 'UNKNOWN_OPTION'}); continue
            option_valid = _option_status_is_true(status, is_negated, opt_phi, gamma_valid, timeout_s)
            supports = bool(option_valid and _supports_selected_solution(question_type, status, is_negated, label, selected_option))
            entry = {'k': k, 'raw': line, 'expr': expr, 'option_label': label, 'status_operator': status, 'is_negated': is_negated, 'supports_selected_solution': supports}
            if option_valid:
                entry['validity_status'] = 'OPTION_STATUS_VALID'; valid_steps.append(entry)
                if supports: support.append(entry)
            else:
                entry['validity_status'] = 'OPTION_STATUS_INVALID'; non_valid.append(entry)
            continue
        try:
            phi = parser_fn(expr); n_parsed += 1
        except Exception as e:
            err = {'k': k, 'raw': line, 'expr': expr, 'status': 'PARSE_ERROR', 'error': f'{type(e).__name__}: {e}'}
            parse_errors.append(err); non_valid.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': 'PARSE_ERROR', 'reason': err['error']}); continue
        sx = phi.sexpr()
        if sx in seen: continue
        seen.add(sx)
        if _is_tautology(base_assertions, phi, timeout_s):
            valid_steps.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': 'TAUTOLOGY'}); continue
        if _equiv_to_any_rule(base_assertions, phi, rule_phis, timeout_s):
            valid_steps.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': 'RESTATES_RULE'}); continue
        validity = _status_under_gamma(gamma_valid, phi, timeout_s)
        if validity == 'ENTAILED': valid_steps.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': validity})
        else: non_valid.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': validity})
        step_status = _status_under_gamma(gamma_steps, phi, timeout_s)
        if step_status != 'CONTRADICTION':
            if validity == 'ENTAILED' and step_status != 'ENTAILED': novel_steps.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': validity, 'steps_status': step_status})
            gamma_steps.append(phi)
    return {
        'n_steps_total': n_total,
        'n_steps_parsed_ok': n_parsed,
        'n_steps_valid': len(valid_steps),
        'n_steps_novel_inc_clues': len(novel_steps),
        'n_non_valid_contradiction': len([x for x in non_valid if x.get('validity_status') == 'CONTRADICTION']),
        'list_steps_valid': [x.get('expr') for x in valid_steps],
        'list_steps_non_valid': non_valid,
        'list_novel_steps_inc_clues': [x.get('expr') for x in novel_steps],
        'list_step_parse_errors': parse_errors,
        'consistency_score': 1.0 if support else 0.0,
        'solution_support_steps': support,
    }


def solve_and_validate_payload(payload: Dict[str, Any], *, timeout_s: float = 2.0, conflict_tolerant_clues: bool = False) -> Dict[str, Any]:
    report = {'base_sat_full_GT': False, 'parse_status': 'INIT', 'n_steps_total': 0, 'n_steps_parsed_ok': 0, 'n_steps_valid': 0, 'n_steps_novel_inc_clues': 0, 'n_non_valid_contradiction': 0, 'consistency_score': 0.0, 'solution_support_steps': []}
    try:
        if str(payload.get('problem_type') or '').strip().lower() != 'assignment':
            raise ValueError("This validator only supports problem_type='assignment'.")
        assign_vars, base_assertions, n_values, n_entities = _make_assignment_base(payload.get('world_model') or {}, timeout_s)
        parser_fn = lambda s: _parse_assignment_expr(s, assign_vars)
        rule_phis, rule_errors = _parse_constraints(payload.get('rules') or [], parser_fn)
        fact_phis, fact_errors = _parse_constraints(payload.get('facts') or [], parser_fn)
        option_phis, option_errors = _parse_options(payload.get('options') or {}, parser_fn)
        rule_fact_phis = rule_phis + fact_phis; gamma = base_assertions + rule_fact_phis; base_sat = _is_sat(gamma, timeout_s)
        selected = _selected_from_payload(payload); gt = _selected_from_ground_truth(payload.get('ground_truth'))
        question_type = ((payload.get('question_semantics') or {}).get('question_type') or payload.get('question_type') or 'could_be_true')
        selected_phi = option_phis.get(selected or '')
        solver_selected_ok = bool(selected_phi is not None and base_sat and _evaluate_option(question_type, selected_phi, gamma, timeout_s))
        gt_match = bool(selected and gt and selected == gt)
        report.update({'base_sat_full_GT': bool(base_sat and solver_selected_ok and gt_match), 'base_sat': bool(base_sat), 'solver_selected_ok': bool(solver_selected_ok), 'gt_match': bool(gt_match), 'selected_option': selected, 'ground_truth_option': gt, 'question_type': question_type, 'rule_parse_errors': rule_errors, 'fact_parse_errors': fact_errors, 'option_parse_errors': option_errors, 'n_rule_parse_errors': len(rule_errors), 'n_fact_parse_errors': len(fact_errors), 'n_option_parse_errors': len(option_errors), 'selected_option_parse_ok': bool(selected_phi is not None), 'n_values': n_values, 'n_entities': n_entities})
        report.update(_validate_reasoning_steps(payload.get('reasoning') or [], parser_fn=parser_fn, base_assertions=base_assertions, rule_fact_phis=rule_fact_phis, rule_phis=rule_phis, option_phis=option_phis, question_type=question_type, selected_option=selected, timeout_s=timeout_s))
        report['parse_status'] = 'AR_LSAT_ASSIGNMENT_SUCCESS'
        return report
    except Exception as e:
        report['parse_status'] = 'Z3_EXCEPTION'; report['error'] = f'{type(e).__name__}: {e}'
        return report


if __name__ == '__main__':
    sample = {'problem_type': 'assignment', 'world_model': {'entities': ['A','B','C'], 'domains': {'values': ['P1','P2','P3']}, 'structural_assumptions': ['each entity is assigned exactly one value']}, 'rules': ['Not(Assign(A, P1))', 'Assign(B, P1) == Assign(C, P1)', 'Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))'], 'facts': [], 'question_semantics': {'question_type': 'could_be_true'}, 'options': {'Option_A': 'Assign(A, P2)', 'B': 'Assign(A, P1)'}, 'reasoning': ['A is not assigned to P1.', 'S1: Not(Assign(A, P1)).', 'Option A is feasible.', 'S2: Sat(Option_A).'], 'solution': {'selected_option': 'Option_A'}, 'ground_truth': 'A'}
    print(json.dumps(solve_and_validate_payload(sample), indent=2, default=str))
