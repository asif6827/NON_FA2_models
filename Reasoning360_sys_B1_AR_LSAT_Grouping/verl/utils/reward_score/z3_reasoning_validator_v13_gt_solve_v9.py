#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AR-LSAT grouping Z3 validator for reward computation."""
from __future__ import annotations

import json, logging, re, sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import z3
    from z3 import And, If, Implies, Int, Not, Or, PbEq, PbGe, PbLe, Solver, Xor, sat, unsat
except Exception:  # pragma: no cover
    z3 = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"^\s*S(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
_FUNC_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_ASSIGN_RE = re.compile(r"^\s*Assign\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*\)\s*$", re.IGNORECASE)


def normalize_header(data_sample): return data_sample

def normalize_months_in_rows(z3_solution: dict) -> dict: return z3_solution

def _norm_token(x: Any) -> str:
    s = str(x).strip().strip("`'\"“”‘’")
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "", s)
    return re.sub(r"_+", "_", s).strip("_")

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

def _split_binary_top_level(s: str, op: str):
    depth = 0; i = 0
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

def _selected_from_ground_truth(gt: Any) -> Optional[str]:
    if isinstance(gt, str): return gt.strip().upper()
    if isinstance(gt, dict):
        for k in ('answer','selected_option','ground_truth_option'):
            if gt.get(k): return str(gt[k]).strip().upper()
    return None

def _selected_from_payload(payload: Dict[str, Any]) -> Optional[str]:
    sol = payload.get('solution') or {}
    return str(sol['selected_option']).strip().upper() if isinstance(sol, dict) and sol.get('selected_option') else None

def _extract_entities_groups(world_model: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    entities = [_norm_token(e) for e in (world_model.get('entities') or [])]
    domains = world_model.get('domains') or {}
    raw_groups = domains.get('groups') or domains.get('committees') or domains.get('values') or domains.get('group') or []
    groups = [_norm_token(g) for g in raw_groups]
    return entities, groups

def _make_base(world_model: Dict[str, Any], timeout_s: float):
    if z3 is None: raise RuntimeError('z3-solver is not installed')
    entities, groups = _extract_entities_groups(world_model)
    if not entities: raise ValueError('world_model.entities is empty')
    if not groups: raise ValueError('world_model.domains.groups is empty')
    group_map = {g: i+1 for i, g in enumerate(groups)}
    var_map = {e: Int(e) for e in entities}
    base_assertions = [And(v >= 1, v <= len(groups)) for v in var_map.values()]
    s = Solver(); s.set('timeout', int(timeout_s*1000)); s.add(base_assertions)
    return s, var_map, group_map, base_assertions, len(groups), len(entities)

def _parse_assign(expr: str, var_map: Dict[str, Any], group_map: Dict[str, int]):
    m = _ASSIGN_RE.match(expr.strip())
    if not m: raise ValueError(f'Expected Assign(entity, group), got {expr!r}')
    ent, grp = _norm_token(m.group(1)), _norm_token(m.group(2))
    if ent not in var_map: raise KeyError(f'Unknown entity token: {ent!r}')
    if grp not in group_map: raise KeyError(f'Unknown group token: {grp!r}')
    return var_map[ent] == group_map[grp]

def _parse_expr(expr: str, var_map: Dict[str, Any], group_map: Dict[str, int]):
    e = str(expr).strip().rstrip('.').strip()
    if _ASSIGN_RE.match(e): return _parse_assign(e, var_map, group_map)
    for op in ('==','!='):
        parts = _split_binary_top_level(e, op)
        if parts is not None:
            L = _parse_expr(parts[0], var_map, group_map); R = _parse_expr(parts[1], var_map, group_map)
            return (L == R) if op == '==' else (L != R)
    m = _FUNC_RE.match(e)
    if not m: raise ValueError(f'Unrecognized grouping expression: {expr!r}')
    fn, args = m.group(1).lower(), _split_top_level_args(m.group(2))
    if fn == 'and': return And(*[_parse_expr(a, var_map, group_map) for a in args])
    if fn == 'or': return Or(*[_parse_expr(a, var_map, group_map) for a in args])
    if fn == 'not':
        if len(args) != 1: raise ValueError('Not expects one argument')
        return Not(_parse_expr(args[0], var_map, group_map))
    if fn == 'implies':
        if len(args) != 2: raise ValueError('Implies expects two arguments')
        return Implies(_parse_expr(args[0], var_map, group_map), _parse_expr(args[1], var_map, group_map))
    if fn == 'xor':
        if len(args) != 2: raise ValueError('Xor expects two arguments')
        return Xor(_parse_expr(args[0], var_map, group_map), _parse_expr(args[1], var_map, group_map))
    if fn in {'atleast','atmost','exactly'}:
        if len(args) < 2: raise ValueError(f'{fn} expects count plus expressions')
        k = int(args[0]); pairs = [(_parse_expr(a, var_map, group_map), 1) for a in args[1:]]
        return PbGe(pairs, k) if fn == 'atleast' else PbLe(pairs, k) if fn == 'atmost' else PbEq(pairs, k)
    raise ValueError(f'Unsupported function: {fn}')

def _parse_constraints(lines, var_map, group_map):
    phis, errors = [], []
    for raw in lines or []:
        try: phis.append(_parse_expr(str(raw), var_map, group_map))
        except Exception as e: errors.append({'raw': str(raw), 'error': f'{type(e).__name__}: {e}'})
    return phis, errors

def _parse_options(options, var_map, group_map):
    out, errs = {}, []
    for label, expr in (options or {}).items():
        lab = str(label).strip().upper()
        try: out[lab] = _parse_expr(str(expr), var_map, group_map)
        except Exception as e: errs.append({'label': lab, 'raw': str(expr), 'error': f'{type(e).__name__}: {e}'})
    return out, errs

def _extract_step_expr(line: str):
    m = _STEP_RE.match((line or '').strip())
    if not m: return None
    expr = m.group(2).split('[',1)[0].rstrip('.').strip()
    return (int(m.group(1)), expr) if expr else None

def _solver_check(assertions, timeout_s):
    s = Solver(); s.set('timeout', int(timeout_s*1000)); s.add(assertions); return s.check()
def _is_sat(assertions, timeout_s): return _solver_check(assertions, timeout_s) == sat
def _is_unsat(assertions, timeout_s): return _solver_check(assertions, timeout_s) == unsat

def _status_under_gamma(gamma, phi, timeout_s):
    if not _is_sat(gamma, timeout_s): return 'PREMISES_UNSAT'
    if _is_unsat(gamma + [phi], timeout_s): return 'CONTRADICTION'
    if _is_unsat(gamma + [Not(phi)], timeout_s): return 'ENTAILED'
    return 'NOT_ENTAILED'

def _is_tautology(base, phi, timeout_s): return _is_sat(base, timeout_s) and _is_unsat(base + [Not(phi)], timeout_s)

def _equiv_to_any_rule(base, phi, rules, timeout_s):
    for r in rules:
        try:
            if _is_unsat(base + [Xor(phi, r)], timeout_s): return True
        except Exception: pass
    return False

def _evaluate_option(question_type, option_phi, gamma, timeout_s):
    qt = (question_type or '').strip().lower()
    if qt in {'could_be_true','acceptability','partial_acceptability','valid_complete_assignment'}: return _is_sat(gamma + [option_phi], timeout_s)
    if qt in {'must_be_true','must_follow'}: return _is_unsat(gamma + [Not(option_phi)], timeout_s)
    if qt in {'cannot_be_true','must_be_false'}: return _is_unsat(gamma + [option_phi], timeout_s)
    if qt == 'could_be_false': return _is_sat(gamma + [Not(option_phi)], timeout_s)
    return _is_sat(gamma + [option_phi], timeout_s)

def _option_status_expr(expr: str):
    m = re.fullmatch(r"\s*(Sat|Unsat)\(\s*(Not\()?\s*Option_([A-Z])\s*\)?\s*\)\s*", (expr or '').strip().rstrip('.'), flags=re.IGNORECASE)
    return (m.group(1).lower(), bool(m.group(2)), m.group(3).upper()) if m else None

def _option_status_is_true(status, is_negated, opt_phi, gamma, timeout_s):
    phi = Not(opt_phi) if is_negated else opt_phi
    return _is_sat(gamma + [phi], timeout_s) if status == 'sat' else _is_unsat(gamma + [phi], timeout_s)

def _supports_selected_solution(question_type, status, is_negated, label, selected):
    if not selected or label != selected: return False
    qt = (question_type or '').strip().lower()
    if qt in {'could_be_true','acceptability','partial_acceptability','valid_complete_assignment'}: return status == 'sat' and not is_negated
    if qt in {'must_be_true','must_follow'}: return status == 'unsat' and is_negated
    if qt in {'cannot_be_true','must_be_false'}: return status == 'unsat' and not is_negated
    if qt == 'could_be_false': return status == 'sat' and is_negated
    return status == 'sat' and not is_negated

def _validate_reasoning_steps(reasoning, *, var_map, group_map, base_assertions, rule_fact_phis, rule_phis, option_phis, question_type, selected_option, timeout_s):
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
            phi = _parse_expr(expr, var_map, group_map); n_parsed += 1
        except Exception as e:
            err = {'k': k, 'raw': line, 'expr': expr, 'status': 'PARSE_ERROR', 'error': f'{type(e).__name__}: {e}'}
            parse_errors.append(err); non_valid.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': 'PARSE_ERROR', 'reason': err['error']}); continue
        sexpr = phi.sexpr()
        if sexpr in seen: continue
        seen.add(sexpr)
        if _is_tautology(base_assertions, phi, timeout_s): valid_steps.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': 'TAUTOLOGY'}); continue
        if _equiv_to_any_rule(base_assertions, phi, rule_phis, timeout_s): valid_steps.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': 'RESTATES_RULE'}); continue
        validity = _status_under_gamma(gamma_valid, phi, timeout_s)
        if validity == 'ENTAILED': valid_steps.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': validity})
        else: non_valid.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': validity})
        step_status = _status_under_gamma(gamma_steps, phi, timeout_s)
        if step_status != 'CONTRADICTION':
            if validity == 'ENTAILED' and step_status != 'ENTAILED': novel_steps.append({'k': k, 'raw': line, 'expr': expr, 'validity_status': validity, 'steps_status': step_status})
            gamma_steps.append(phi)
    return {'n_steps_total': n_total, 'n_steps_parsed_ok': n_parsed, 'n_steps_valid': len(valid_steps), 'n_steps_novel_inc_clues': len(novel_steps), 'n_non_valid_contradiction': len([x for x in non_valid if x.get('validity_status') == 'CONTRADICTION']), 'list_steps_valid': [x.get('expr') for x in valid_steps], 'list_steps_non_valid': non_valid, 'list_novel_steps_inc_clues': [x.get('expr') for x in novel_steps], 'list_step_parse_errors': parse_errors, 'consistency_score': 1.0 if support else 0.0, 'solution_support_steps': support}

def solve_and_validate_payload(payload: Dict[str, Any], *, timeout_s: float = 2.0, conflict_tolerant_clues: bool = False) -> Dict[str, Any]:
    report = {'base_sat_full_GT': False, 'parse_status': 'INIT', 'n_steps_total':0, 'n_steps_parsed_ok':0, 'n_steps_valid':0, 'n_steps_novel_inc_clues':0, 'n_non_valid_contradiction':0, 'consistency_score':0.0, 'solution_support_steps':[]}
    try:
        if payload.get('problem_type') != 'grouping': raise ValueError("This validator only supports problem_type='grouping'.")
        _, var_map, group_map, base_assertions, n_groups, n_entities = _make_base(payload.get('world_model') or {}, timeout_s)
        rule_phis, rule_errors = _parse_constraints(payload.get('rules') or [], var_map, group_map)
        fact_phis, fact_errors = _parse_constraints(payload.get('facts') or [], var_map, group_map)
        option_phis, option_errors = _parse_options(payload.get('options') or {}, var_map, group_map)
        rule_fact_phis = rule_phis + fact_phis; gamma = base_assertions + rule_fact_phis; base_sat = _is_sat(gamma, timeout_s)
        selected = _selected_from_payload(payload); gt = _selected_from_ground_truth(payload.get('ground_truth'))
        question_type = ((payload.get('question_semantics') or {}).get('question_type') or payload.get('question_type') or 'could_be_true')
        selected_phi = option_phis.get(selected or ''); solver_selected_ok = bool(selected_phi is not None and base_sat and _evaluate_option(question_type, selected_phi, gamma, timeout_s))
        report.update({'base_sat_full_GT': bool(base_sat and solver_selected_ok and selected and gt and selected == gt), 'base_sat': bool(base_sat), 'solver_selected_ok': bool(solver_selected_ok), 'selected_option': selected, 'ground_truth_option': gt, 'question_type': question_type, 'rule_parse_errors': rule_errors, 'fact_parse_errors': fact_errors, 'option_parse_errors': option_errors, 'n_groups': n_groups, 'n_entities': n_entities})
        report.update(_validate_reasoning_steps(payload.get('reasoning') or [], var_map=var_map, group_map=group_map, base_assertions=base_assertions, rule_fact_phis=rule_fact_phis, rule_phis=rule_phis, option_phis=option_phis, question_type=question_type, selected_option=selected, timeout_s=timeout_s))
        report['parse_status'] = 'AR_LSAT_GROUPING_SUCCESS'
        return report
    except Exception as e:
        report['parse_status'] = 'Z3_EXCEPTION'; report['error'] = f'{type(e).__name__}: {e}'; return report

if __name__ == '__main__':
    sample = {'problem_type':'grouping','world_model':{'entities':['A','B','C','D','E','F','G'],'domains':{'groups':['X','Y']}},'rules':['Implies(Assign(A, X), Assign(B, Y))','Implies(Assign(C, X), And(Assign(D, Y), Assign(E, Y)))','Assign(F, X) != Assign(G, X)','Assign(E, X) != Assign(A, X)','Implies(Assign(G, X), Assign(B, X))'],'facts':['Assign(D, X)','Assign(F, X)'],'question_semantics':{'question_type':'could_be_true'},'options':{'D':'And(Assign(C, Y), Assign(E, Y))'},'reasoning':['D and F are fixed in X.','S1: And(Assign(D, X), Assign(F, X)).','F and G are different, so G is in Y.','S2: Assign(G, Y).','Option D is feasible.','S3: Sat(Option_D).'],'solution':{'selected_option':'D'},'ground_truth':'D'}
    print(json.dumps(solve_and_validate_payload(sample), indent=2))
