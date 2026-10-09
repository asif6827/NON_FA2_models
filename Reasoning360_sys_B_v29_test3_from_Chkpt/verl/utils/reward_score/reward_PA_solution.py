"""Evidence-grounded final-table partial-answer reward.

compute_pa_solution(payload) returns flags, diagnostic accuracy, and gated coverage.
compute_coverage(payload) returns only the gated coverage float.
Payload keys: ground_truth, reasoning, solution, n_houses, attribute_values,
z3_out. z3_out is accepted for compatibility but never supplies constraints.
House/Position columns are identifiers, excluded from all cell metrics.
No clues, bounds, or uniqueness constraints are added to the reasoning solver.
"""
import ast
import re
import z3


def _norm(value):
    return str(value).strip().lower().replace(' ', '_')


def _table(table, n):
    header = [_norm(x) for x in table['header']]
    if len(set(header)) != len(header):
        raise ValueError('Duplicate headers')
    ids = [x for x in header if x in ('house', 'position')]
    if len(ids) != 1:
        raise ValueError('Exactly one House/Position column required')
    hi = header.index(ids[0])
    columns = [x for x in header if x not in ids]
    rows = {}
    for row in table['rows']:
        if len(row) != len(header):
            raise ValueError('Invalid row width')
        house = int(str(row[hi]))
        if house in rows or not 1 <= house <= n:
            raise ValueError('Duplicate/out-of-range house')
        rows[house] = {c: _norm(row[header.index(c)]) for c in columns}
    if set(rows) != set(range(1, n + 1)):
        raise ValueError('Missing houses')
    return columns, rows


def _constraint(text, tokens):
    # Replace canonical entity tokens by safe identifiers before AST parsing.
    # No Python eval, attribute access, subscripting, or arbitrary calls allowed.
    lookup = {f'entity_{i}': v for i, v in enumerate(tokens.values())}
    aliases = {t: f'entity_{i}' for i, t in enumerate(tokens)}
    pattern = r'(?<!\w)(?:' + '|'.join(re.escape(t) for t in sorted(tokens, key=len, reverse=True)) + r')(?!\w)'
    text = re.sub(pattern, lambda m: aliases[m.group(0).lower()], text, flags=re.I)
    node = ast.parse(text, mode='eval').body

    def parse(x):
        if isinstance(x, ast.Name) and x.id in lookup:
            return lookup[x.id]
        if isinstance(x, ast.Constant) and type(x.value) is int:
            return x.value
        if isinstance(x, ast.BinOp) and isinstance(x.op, (ast.Add, ast.Sub)):
            a, b = parse(x.left), parse(x.right)
            if z3.is_bool(a) or z3.is_bool(b):
                raise ValueError('Boolean arithmetic')
            return a + b if isinstance(x.op, ast.Add) else a - b
        if isinstance(x, ast.Compare) and len(x.ops) == 1:
            a, b = parse(x.left), parse(x.comparators[0])
            if z3.is_bool(a) or z3.is_bool(b):
                raise ValueError('Boolean comparison')
            op = x.ops[0]
            if isinstance(op, ast.Eq): return a == b
            if isinstance(op, ast.NotEq): return a != b
            if isinstance(op, ast.Lt): return a < b
            if isinstance(op, ast.Gt): return a > b
        if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and not x.keywords:
            name = x.func.id
            args = [parse(a) for a in x.args]
            if not all(z3.is_bool(a) for a in args):
                raise ValueError('Non-constraint Boolean argument')
            if name == 'Not' and len(args) == 1: return z3.Not(args[0])
            if name == 'And' and args: return z3.And(*args)
            if name == 'Or' and args: return z3.Or(*args)
        raise ValueError('Unsupported constraint syntax')

    result = parse(node)
    if not z3.is_bool(result):
        raise ValueError('Expected a symbolic constraint')
    return result


def compute_pa_solution(payload, timeout_s=5.0):
    """Return zero coverage on inaccurate, unsupported, malformed, or unknown cases.

    Accuracy: all populated attribute cells match GT and at least one exists.
    Consistency: reasoning is SAT and reasoning AND NOT(all populated cells)
    is UNSAT. Every numbered syntactic step is used, not just novel steps.
    """
    out = dict(accuracy=False, consistency=False, pa_accuracy=0.0,
               coverage=0.0, raw_coverage=0.0, populated_cells=0,
               total_cells=0, status='invalid_payload')
    try:
        n = payload['n_houses']
        if isinstance(n, bool) or int(n) != n or int(n) <= 0:
            raise ValueError('Invalid n_houses')
        n = int(n)
        attrs = {_norm(k): [_norm(v) for v in vals]
                 for k, vals in payload['attribute_values'].items()}
        if not attrs or any(len(v) != n or len(set(v)) != n for v in attrs.values()):
            raise ValueError('Invalid attribute domain')
        columns, pred = _table(payload['solution'], n)
        gt_columns, gt = _table(payload['ground_truth'], n)
        if set(columns) != set(gt_columns) or set(columns) != set(attrs):
            raise ValueError('Attribute headers differ')
        cells = [(h, c, pred[h][c]) for h in pred for c in columns if pred[h][c] != '?']
        if any(gt[h][c] not in attrs[c] for h in gt for c in columns):
            raise ValueError('GT outside domain')
        if any(v not in attrs[c] for h, c, v in cells):
            raise ValueError('Solution outside domain')
        total = n * len(columns)
        correct = sum(v == gt[h][c] for h, c, v in cells)
        out.update(total_cells=total, populated_cells=len(cells),
                   pa_accuracy=correct / len(cells) if cells else 0.0,
                   raw_coverage=len(cells) / total,
                   accuracy=bool(cells) and correct == len(cells))
        if not cells:
            out['status'] = 'empty_solution'
            return out
        tokens = {}
        for c, vals in attrs.items():
            for v in vals:
                if v in tokens or v in ('?', 'and', 'or', 'not'):
                    raise ValueError('Ambiguous/reserved entity token')
                tokens[v] = z3.Int(f'{c}:{v}')
        reasoning = payload['reasoning']
        if not isinstance(reasoning, (list, tuple)) or not reasoning or len(reasoning) % 2:
            raise ValueError('Expected interleaved reasoning pairs')
        constraints = []
        for index, step in enumerate(reasoning):
            if not isinstance(step, str):
                raise ValueError('Non-string reasoning entry')
            match = re.fullmatch(r'\s*S(\d+):\s*(.+)\.\s*', step)
            if index % 2 == 0:
                if match:
                    raise ValueError('Expected natural-language entry')
                continue
            if not match or int(match[1]) != (index + 1) // 2:
                raise ValueError('Invalid syntactic step numbering/format')
            constraints.append(_constraint(match[2], tokens))
        solver = z3.Solver()
        solver.set(timeout=max(1, int(float(timeout_s) * 1000)))
        solver.add(*constraints)
        base = solver.check()
        if base != z3.sat:
            out['status'] = 'reasoning_unsat' if base == z3.unsat else 'reasoning_unknown'
            return out
        assertions = [tokens[v] == h for h, c, v in cells]
        solver.add(z3.Not(z3.And(*assertions)))
        result = solver.check()
        out['consistency'] = result == z3.unsat
        out['coverage'] = out['raw_coverage'] if out['accuracy'] and out['consistency'] else 0.0
        out['status'] = 'ok' if result != z3.unknown else 'entailment_unknown'
    except (KeyError, TypeError, ValueError, SyntaxError, OverflowError, z3.Z3Exception) as error:
        out.update(coverage=0.0, consistency=False, status='invalid_payload', error=str(error))
    return out


def compute_coverage(payload, timeout_s=5.0):
    """Convenience API returning gated coverage only."""
    return compute_pa_solution(payload, timeout_s=timeout_s)['coverage']


if __name__ == '__main__':
    from copy import deepcopy

    base_payload = {
        'n_houses': 2,
        'attribute_values': {'Name': ['Alice', 'Bob']},
        'ground_truth': {
            'header': ['House', 'Name'],
            'rows': [['1', 'Alice'], ['2', 'Bob']],
        },
        'solution': {
            'header': ['House', 'Name'],
            'rows': [['1', 'Alice'], ['2', '?']],
        },
        'reasoning': ['Alice occupies house 1.', 'S1: Alice == 1.'],
        'z3_out': {},
    }
    cases = []

    cases.append(('Supported partial answer', deepcopy(base_payload),
                  (1.0, True, True, 0.5)))

    full = deepcopy(base_payload)
    full['solution']['rows'][1][1] = 'Bob'
    full['reasoning'] += ['Bob occupies house 2.', 'S2: Bob == 2.']
    cases.append(('Supported complete answer', full, (1.0, True, True, 1.0)))

    wrong = deepcopy(base_payload)
    wrong['solution']['rows'][0][1] = 'Bob'
    wrong['reasoning'] = ['Bob occupies house 1.', 'S1: Bob == 1.']
    cases.append(('Incorrect but supported answer', wrong, (0.0, False, True, 0.0)))

    unsupported = deepcopy(base_payload)
    unsupported['solution']['rows'][1][1] = 'Bob'
    # One supported cell must not hide another unsupported cell.
    cases.append(('Correct but partly unsupported answer', unsupported,
                  (1.0, True, False, 0.0)))

    contradictory = deepcopy(base_payload)
    contradictory['reasoning'] += ['Alice is not in house 1.', 'S2: Alice != 1.']
    cases.append(('Contradictory reasoning', contradictory,
                  (1.0, True, False, 0.0)))

    empty = deepcopy(base_payload)
    empty['solution']['rows'][0][1] = '?'
    cases.append(('All cells unknown', empty, (0.0, False, False, 0.0)))

    mixed = deepcopy(full)
    mixed['solution']['rows'][1][1] = 'Alice'
    cases.append(('One correct and one incorrect cell', mixed,
                  (0.5, False, False, 0.0)))

    clues_only = deepcopy(base_payload)
    clues_only['reasoning'] = ['Alice may occupy either house.',
                              'S1: Or(Alice == 1, Alice == 2).']
    clues_only['syntactic_clues'] = ['C1: Alice == 1.']
    clues_only['z3_out'] = {'list_novel_steps_inc_clues': ['Alice == 1']}
    cases.append(('Clues and z3_out do not establish support', clues_only,
                  (1.0, True, False, 0.0)))

    for label, payload, expected in cases:
        result = compute_pa_solution(payload)
        actual = (result['pa_accuracy'], result['accuracy'],
                  result['consistency'], result['coverage'])
        if actual != expected:
            raise AssertionError(f'{label}: expected {expected}, got {result}')
        print(f'PASS: {label}: pa_accuracy={actual[0]}, '
              f'accuracy={actual[1]}, consistency={actual[2]}, coverage={actual[3]}')

    print(f'All {len(cases)} quick tests passed.')
