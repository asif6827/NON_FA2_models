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
    import math

    # Five houses and five attributes: 25 scored cells, excluding House.
    # This synthetic Zebra-style fixture uses multi-step deductions rather
    # than relying on implicit bounds or uniqueness in the reward solver.
    header = ['House', 'Nationality', 'Color', 'Drink', 'Pet', 'Sport']
    rows = [
        ['1', 'Norwegian', 'yellow', 'water', 'cats', 'table_tennis'],
        ['2', 'Dane', 'blue', 'tea', 'horses', 'cycling'],
        ['3', 'Brit', 'red', 'milk', 'birds', 'football'],
        ['4', 'German', 'green', 'coffee', 'fish', 'swimming'],
        ['5', 'Swede', 'white', 'root_beer', 'dogs', 'running'],
    ]
    ground_truth = {'header': header, 'rows': rows}
    attribute_values = {
        name: [row[i] for row in rows]
        for i, name in enumerate(header) if i > 0
    }

    def trace(pairs):
        entries = []
        for i, (explanation, expression) in enumerate(pairs, 1):
            entries.extend([explanation, f'S{i}: {expression}.'])
        return entries

    deductions = [
        ('The Norwegian occupies the first house.', 'Norwegian == 1'),
        ('The Dane lives immediately right of the Norwegian.', 'Norwegian + 1 == Dane'),
        ('The Brit lives immediately right of the Dane.', 'Dane + 1 == Brit'),
        ('The German lives immediately right of the Brit.', 'Brit + 1 == German'),
        ('The Swede lives immediately right of the German.', 'German + 1 == Swede'),
        ('The Norwegian lives in the yellow house.', 'yellow == Norwegian'),
        ('The blue house is immediately right of the yellow house.', 'yellow + 1 == blue'),
        ('The Brit lives in the red house.', 'red == Brit'),
        ('The green house is two positions right of the blue house.', 'blue + 2 == green'),
        ('The white house is immediately right of the green house.', 'green + 1 == white'),
        ('The Norwegian drinks water.', 'water == Norwegian'),
        ('The Dane drinks tea.', 'tea == Dane'),
        ('Milk is drunk either in the second or third house.', 'Or(milk == 2, milk == 3)'),
        ('Milk is not drunk in the second house.', 'Not(milk == 2)'),
        ('Coffee is drunk in the green house.', 'coffee == green'),
        ('Root beer is drunk immediately right of coffee.', 'coffee + 1 == root_beer'),
        ('Cats live with the water drinker and horses live with the tea drinker.', 'And(cats == water, horses == tea)'),
        ('The bird keeper drinks milk.', 'birds == milk'),
        ('The cat keeper plays table tennis and the horse keeper enjoys cycling.', 'And(table_tennis == cats, cycling == horses)'),
        ('Football is played either beside cycling to its left or to its right.', 'Or(football == cycling + 1, football == cycling - 1)'),
        ('Football is played somewhere right of cycling.', 'football > cycling'),
        ('The green house lies left of the white house.', 'green < white'),
        ('Fish and dogs occupy houses four and five in an unresolved order.', 'Or(And(fish == 4, dogs == 5), And(fish == 5, dogs == 4))'),
        ('Swimming and running occupy houses four and five in an unresolved order.', 'Or(And(swimming == 4, running == 5), And(swimming == 5, running == 4))'),
    ]
    final_deductions = [
        ('The German keeps fish.', 'fish == German'),
        ('The Swede keeps dogs.', 'dogs == Swede'),
        ('Swimming is enjoyed in the green house.', 'swimming == green'),
        ('Running is enjoyed in the white house.', 'running == white'),
    ]
    partial = {
        'ground_truth': deepcopy(ground_truth),
        'solution': deepcopy(ground_truth),
        'n_houses': 5,
        'attribute_values': deepcopy(attribute_values),
        'reasoning': trace(deductions),
        'z3_out': {},
    }
    for row in partial['solution']['rows'][3:]:
        row[4] = row[5] = '?'
    complete = deepcopy(partial)
    complete['solution'] = deepcopy(ground_truth)
    complete['reasoning'] = trace(deductions + final_deductions)

    cases = []

    def add(label, payload, accuracy, consistency, pa_accuracy, coverage,
            populated, raw_coverage, status='ok'):
        cases.append((label, payload, dict(
            accuracy=accuracy, consistency=consistency,
            pa_accuracy=pa_accuracy, coverage=coverage,
            populated_cells=populated, total_cells=25,
            raw_coverage=raw_coverage, status=status)))

    add('21/25 cells supported by a long deduction chain', deepcopy(partial),
        True, True, 1.0, 21/25, 21, 21/25)
    add('25/25 cells supported after resolving both disjunctions', deepcopy(complete),
        True, True, 1.0, 1.0, 25, 1.0)

    q = deepcopy(partial)
    q['solution']['rows'][3][4] = 'fish'
    add('One correct but unsupported cell among 22 populated cells', q,
        True, False, 1.0, 0.0, 22, 22/25)

    q = deepcopy(partial)
    q['solution'] = deepcopy(ground_truth)
    add('Correct full table with four unresolved assignments', q,
        True, False, 1.0, 0.0, 25, 1.0)

    q = deepcopy(partial)
    q['solution']['rows'][0][2], q['solution']['rows'][1][2] = 'blue', 'yellow'
    add('Two wrong cells among otherwise correct supported cells', q,
        False, False, 19/21, 0.0, 21, 21/25)

    q = deepcopy(complete)
    q['solution']['rows'][3][4], q['solution']['rows'][4][4] = 'dogs', 'fish'
    alternate = deductions + [
        ('The German keeps dogs.', 'dogs == German'),
        ('The Swede keeps fish.', 'fish == Swede'),
    ] + final_deductions[2:]
    q['reasoning'] = trace(alternate)
    add('Wrong pet assignments fully supported by internally consistent reasoning', q,
        False, True, 23/25, 0.0, 25, 1.0)

    q = deepcopy(partial)
    q['reasoning'] += trace([('The Norwegian does not occupy house one.', 'Norwegian != 1')])
    q['reasoning'][-1] = 'S25: Norwegian != 1.'
    add('Contradictory reasoning must not produce vacuous support', q,
        True, False, 1.0, 0.0, 21, 21/25, 'reasoning_unsat')

    q = deepcopy(partial)
    for row in q['solution']['rows']:
        row[1:] = ['?'] * 5
    add('All 25 attribute cells unknown despite valid house indices', q,
        False, False, 0.0, 0.0, 0, 0.0, 'empty_solution')

    q = deepcopy(partial)
    for row in q['solution']['rows']:
        row[1:] = ['?'] * 5
    q['solution']['rows'][0][1] = 'Norwegian'
    add('Only one evidence-supported cell in a 25-cell table', q,
        True, True, 1.0, 1/25, 1, 1/25)

    q = deepcopy(partial)
    q['solution']['rows'].reverse()
    add('Rows aligned by house identifier rather than list order', q,
        True, True, 1.0, 21/25, 21, 21/25)

    q = deepcopy(partial)
    order = [4, 0, 5, 2, 1, 3]
    q['solution']['header'] = [header[i] for i in order]
    q['solution']['rows'] = [[row[i] for i in order] for row in q['solution']['rows']]
    add('Columns aligned by header rather than list position', q,
        True, True, 1.0, 21/25, 21, 21/25)

    q = deepcopy(partial)
    q['solution']['header'] = [h.upper() for h in header]
    q['solution']['rows'][0][5] = ' TABLE TENNIS '
    q['solution']['rows'][0][1] = ' norwegian '
    add('Case, surrounding whitespace, and space/underscore normalization', q,
        True, True, 1.0, 21/25, 21, 21/25)

    q = deepcopy(partial)
    q['solution']['header'][0] = 'Position'
    add('Position identifier is excluded from the denominator', q,
        True, True, 1.0, 21/25, 21, 21/25)

    q = deepcopy(partial)
    q['solution'] = deepcopy(ground_truth)
    q['syntactic_clues'] = ['C1: fish == 4.', 'C2: swimming == 4.']
    q['z3_out'] = {'list_novel_steps_inc_clues': ['fish == 4', 'dogs == 5',
                                                      'swimming == 4', 'running == 5']}
    add('Clues and cached validator output cannot support missing deductions', q,
        True, False, 1.0, 0.0, 25, 1.0)

    q = deepcopy(partial)
    q['reasoning'][27] = 'S14: milk != 2.'
    add('Inequality and Not forms have equivalent deductions', q,
        True, True, 1.0, 21/25, 21, 21/25)

    # Invalid inputs must return zero coverage instead of raising exceptions.
    invalid_cases = []
    q = deepcopy(partial); q['solution']['rows'].pop()
    invalid_cases.append(('Missing house row', q))
    q = deepcopy(partial); q['solution']['rows'][1][0] = '1'
    invalid_cases.append(('Duplicate house identifier', q))
    q = deepcopy(partial); q['solution']['rows'][0].pop()
    invalid_cases.append(('Ragged table row', q))
    q = deepcopy(partial); q['solution']['rows'][0][2] = 'purple'
    invalid_cases.append(('Out-of-domain color', q))
    q = deepcopy(partial); q['solution']['header'][2] = 'UnknownAttribute'
    invalid_cases.append(('Wrong attribute header', q))
    q = deepcopy(partial); q['reasoning'][1] = 'S2: Norwegian == 1.'
    invalid_cases.append(('Nonsequential syntactic step number', q))
    q = deepcopy(partial); q['reasoning'][1] = 'S1: Norwegian == 1'
    invalid_cases.append(('Missing syntactic terminator', q))
    q = deepcopy(partial); q['reasoning'][1] = 'S1: Norwegian == mystery_person.'
    invalid_cases.append(('Unknown reasoning token', q))
    q = deepcopy(partial); q['reasoning'].pop()
    invalid_cases.append(('Incomplete interleaved reasoning pair', q))
    q = deepcopy(partial); q['reasoning'][1] = 'S1: __import__("os").getpid() == 1.'
    invalid_cases.append(('Arbitrary Python calls are rejected', q))

    for label, payload, expected in cases:
        result = compute_pa_solution(payload)
        for key, value in expected.items():
            actual = result[key]
            valid = (math.isclose(actual, value, abs_tol=1e-12)
                     if isinstance(value, float) else actual == value)
            if not valid:
                raise AssertionError(f'{label}: {key}: expected {value}, got {result}')
        if not math.isclose(compute_coverage(payload), expected['coverage'], abs_tol=1e-12):
            raise AssertionError(f'{label}: compute_coverage disagrees')
        print(f"PASS: {label}: accuracy={result['accuracy']}, "
              f"consistency={result['consistency']}, pa_accuracy={result['pa_accuracy']:.4f}, "
              f"coverage={result['coverage']:.4f}, raw_coverage={result['raw_coverage']:.4f}")

    for label, payload in invalid_cases:
        result = compute_pa_solution(payload)
        if result['status'] != 'invalid_payload' or result['coverage'] != 0 or result['consistency']:
            raise AssertionError(f'{label}: expected safe failure, got {result}')
        print(f"PASS: {label}: {result['status']}: {result.get('error', '')}")

    print(f'All {len(cases) + len(invalid_cases)} realistic scenario checks passed.')
