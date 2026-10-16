"""Final-table partial-answer reward using joint satisfiability.

Outputs: pa_accuracy_binary_pass, coverage, pa_consistency_binary_pass,
pa_accuracy, pa_reward. pa_accuracy is correct populated cells / ALL attribute
cells, not correct populated cells / populated cells.
Consistency checks SAT(reasoning AND all populated cell assignments).
No clues, bounds, uniqueness, or constraints from z3_out are added.
House/Position identifiers are excluded from cell counts. Empty attribute cells
are ?, blank strings, or None. An all-empty answer fails accuracy and earns zero
reward; its consistency still reflects the joint SAT check.
Reasoning is an interleaved list of natural-language / S<k>: constraint. pairs;
an empty list supplies no constraints. Malformed reasoning fails consistency.
Requires z3-solver. Payload fields are unchanged.
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
        rows[house] = {c: ('' if row[header.index(c)] is None else _norm(row[header.index(c)]))
                       for c in columns}
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
    """Compute ungated coverage, two binary gates, and the gated PA reward.

    SAT means compatibility, not entailment: an assignment absent from reasoning
    can pass consistency as long as it does not contradict the reasoning.
    Unknown/timeout results fail consistency. Invalid tables give zero metrics;
    after valid table scoring, reasoning errors preserve accuracy and coverage.
    """
    out = {
        'pa_accuracy_binary_pass': 0,
        'pa_accuracy': 0.0,
        'coverage': 0.0,
        'pa_consistency_binary_pass': 0,
        'pa_reward': 0.0,
    }
    try:
        if not isinstance(payload, dict):
            raise ValueError('Payload must be a dictionary')
        n = payload['n_houses']
        if isinstance(n, bool) or int(n) != n or int(n) <= 0:
            raise ValueError('Invalid n_houses')
        n = int(n)
        domains = payload['attribute_values']
        if not isinstance(domains, dict) or not domains:
            raise ValueError('attribute_values must be a nonempty dictionary')
        attrs = {}
        for key, values in domains.items():
            if not isinstance(values, (list, tuple)):
                raise ValueError('Attribute values must be lists')
            name = _norm(key)
            if name in attrs:
                raise ValueError('Duplicate normalized attribute name')
            attrs[name] = [_norm(value) for value in values]
        if any(len(values) != n or len(set(values)) != n for values in attrs.values()):
            raise ValueError('Invalid attribute domain')
        columns, pred = _table(payload['solution'], n)
        gt_columns, gt = _table(payload['ground_truth'], n)
        if set(columns) != set(gt_columns) or set(columns) != set(attrs):
            raise ValueError('Attribute headers differ')
        cells = [(house, column, pred[house][column])
                 for house in pred for column in columns
                 if pred[house][column] not in ('?', '')]
        if any(gt[h][c] not in attrs[c] for h in gt for c in columns):
            raise ValueError('GT outside domain')
        if any(value not in attrs[column] for h, column, value in cells):
            raise ValueError('Solution outside domain')
        total_cells = n * len(columns)
        correct_cells = sum(value == gt[h][c] for h, c, value in cells)

        out['coverage'] = len(cells) / total_cells
        out['pa_accuracy'] = correct_cells / total_cells
        out['pa_accuracy_binary_pass'] = int(bool(cells) and correct_cells == len(cells))

        tokens = {}

        for column, values in attrs.items():
            for value in values:
                if value in tokens or value in ('?', '', 'and', 'or', 'not'):
                    raise ValueError('Ambiguous/reserved entity token')
                tokens[value] = z3.Int(f'{column}:{value}')
        reasoning = payload['reasoning']
        if not isinstance(reasoning, (list, tuple)) or len(reasoning) % 2:
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
        # Positive assignments: no negation and no separate entailment check.
        solver.add(*[tokens[value] == h for h, c, value in cells])
        out['pa_consistency_binary_pass'] = int(solver.check() == z3.sat)
        if out['pa_accuracy_binary_pass'] and out['pa_consistency_binary_pass']:
            out['pa_reward'] = 1.0 + out['coverage']

    except (KeyError, TypeError, ValueError, AttributeError, SyntaxError,
            OverflowError, RecursionError, z3.Z3Exception):
        # Invalid input/solver failure must never create a positive PA reward.
        out['pa_consistency_binary_pass'] = 0
        out['pa_reward'] = 0.0
    return out


if __name__ == '__main__':
    from copy import deepcopy
    import math

    gt = {
        'header': ['House', 'Name', 'Color', 'Children'],
        'rows': [['1', 'Peter', 'yellow', 'Bella'],
                 ['2', 'Arnold', 'red', 'Fred'],
                 ['3', 'Eric', 'white', 'Meredith']],
    }
    base = {
        'n_houses': 3,
        'attribute_values': {
            'Name': ['Peter', 'Arnold', 'Eric'],
            'Color': ['yellow', 'red', 'white'],
            'Children': ['Bella', 'Fred', 'Meredith'],
        },
        'ground_truth': deepcopy(gt),
        'solution': deepcopy(gt),
        'reasoning': ['Red occupies house two.', 'S1: red == 2.',
                      'Arnold lives in the red house.', 'S2: Arnold == red.'],
        'z3_out': {},
    }
    cases = []
    def add(name, payload, accuracy, coverage, consistency, reward, pa_accuracy=None):
        if pa_accuracy is None:
            pa_accuracy = coverage
        cases.append((name, payload, (accuracy, coverage, consistency, pa_accuracy, reward)))

    add('Correct complete table compatible with sparse reasoning',
        deepcopy(base), 1, 1.0, 1, 2.0)
    p = deepcopy(base)
    for row in p['solution']['rows']:
        row[3] = '?'
    add('Correct partial table: six of nine cells', p, 1, 6/9, 1, 1+6/9)
    p = deepcopy(base)
    p['solution']['rows'][0][2], p['solution']['rows'][2][2] = 'white', 'yellow'
    add('Wrong colors compatible with reasoning', p, 0, 1.0, 1, 0.0, pa_accuracy=7/9)
    p = deepcopy(base)
    p['reasoning'] += ['Peter occupies house three.', 'S3: Peter == 3.']
    add('Correct table contradicts reasoning', p, 1, 1.0, 0, 0.0)
    p = deepcopy(base)
    p['solution']['rows'][0][1], p['solution']['rows'][1][1] = 'Arnold', 'Peter'
    add('Wrong names also contradict reasoning', p, 0, 1.0, 0, 0.0, pa_accuracy=7/9)
    p = deepcopy(base)
    p['reasoning'] += ['Red is not in house two.', 'S3: red != 2.']
    add('Internally contradictory reasoning', p, 1, 1.0, 0, 0.0)
    p = deepcopy(base)
    for row in p['solution']['rows']:
        row[1:] = ['?', '', None]
    add('All-empty table: no accuracy pass or reward', p, 0, 0.0, 1, 0.0)
    p = deepcopy(base)
    for row in p['solution']['rows']:
        row[1:] = ['?', '?', '?']
    p['solution']['rows'][0][1] = 'Peter'
    add('One correct cell absent from reasoning still passes SAT', p, 1, 1/9, 1, 1+1/9)
    p = deepcopy(base); p['reasoning'] = []
    add('Empty reasoning is compatible with a correct complete table', p, 1, 1.0, 1, 2.0)
    p = deepcopy(base); p['reasoning'][1] = 'S8: red == 2.'
    add('Malformed reasoning preserves ungated table metrics', p, 1, 1.0, 0, 0.0)
    p = deepcopy(base); p['attribute_values'] = None
    add('Missing attribute domain safely fails', p, 0, 0.0, 0, 0.0)
    p = deepcopy(base); p['solution']['rows'].reverse()
    add('Rows are aligned by house identifier', p, 1, 1.0, 1, 2.0)
    p = deepcopy(base); p['syntactic_clues'] = ['C1: red != 2.']
    p['z3_out'] = {'constraints': ['red != 2']}
    add('Clues and z3_out do not enter the SAT check', p, 1, 1.0, 1, 2.0)

    p = deepcopy(base)
    for row in p['solution']['rows']:
        row[1:] = ['?', '?', '?']
    p['solution']['rows'][0][1:] = ['Peter', 'white', 'Bella']
    add('Three populated cells, two correct: accuracy uses total denominator',
        p, 0, 3/9, 1, 0.0, pa_accuracy=2/9)

    keys = ('pa_accuracy_binary_pass', 'coverage',
            'pa_consistency_binary_pass', 'pa_accuracy', 'pa_reward')
    for name, payload, expected in cases:
        result = compute_pa_solution(payload)
        if set(result) != set(keys) or any(
                not math.isclose(result[key], target, abs_tol=1e-12)
                for key, target in zip(keys, expected)):
            raise AssertionError(f'{name}: expected {expected}, got {result}')
        print(f'PASS: {name}: {result}')
    print(f'All {len(cases)} tests passed.')
