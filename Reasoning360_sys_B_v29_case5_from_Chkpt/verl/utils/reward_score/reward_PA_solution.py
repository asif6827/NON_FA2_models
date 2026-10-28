"""Final-table partial-answer reward using joint satisfiability.

Outputs: pa_accuracy_binary_pass, coverage, pa_consistency_binary_pass,
pa_accuracy, pa_reward. pa_accuracy is correct populated cells / ALL attribute
cells, not correct populated cells / populated cells.
Consistency checks SAT(structural domain constraints AND reasoning AND
all populated cell assignments). Structural constraints enforce house bounds
and uniqueness within each attribute. No puzzle clues or constraints from
z3_out are added.

House/Position identifiers are excluded from cell counts. Empty attribute
cells are ?, blank strings, or None. An all-empty answer fails accuracy and
earns zero reward; its consistency reflects the SAT check.

Syntactic constraints are identified by their S<number>: prefix.
Natural-language entries are ignored. Empty reasoning adds no constraints;
malformed recognized syntactic constraints fail consistency.
DIAGNOSIS reports gate outcomes, mismatch details, and validation/solver failures.
Requires z3-solver. Payload fields are unchanged.
"""

import re
import z3
import ast
import math

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


def compute_pa_solution(payload, timeout_s=5.0,penalty_lambda=1.0):
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
        'n_wrong': 0,
        'n_wrong_available': 0,
        'wrong_cell_penalty': 0.0,
        'DIAGNOSIS': {
            'accuracy': {'status': 'not_evaluated', 'reason': 'Input validation pending'},
            'consistency': {'status': 'not_evaluated', 'reason': 'Input validation pending'},
        },
    }
    diagnosis = out['DIAGNOSIS']
    stage = 'input_validation'
    current_step = None
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
            diagnosis['invalid_ground_truth_cells'] = [
                {'house': h, 'attribute': c, 'value': gt[h][c]}
                for h in gt for c in columns if gt[h][c] not in attrs[c]
            ]
            raise ValueError('GT outside domain')
        if any(value not in attrs[column] for h, column, value in cells):
            diagnosis['invalid_solution_cells'] = [
                {'house': h, 'attribute': c, 'value': value}
                for h, c, value in cells if value not in attrs[c]
            ]
            raise ValueError('Solution outside domain')
        total_cells = n * len(columns)
        correct_cells = sum(value == gt[h][c] for h, c, value in cells)

        n_wrong = len(cells) - correct_cells
        out['n_wrong'] = n_wrong
        out['n_wrong_available'] = 1
        out['wrong_cell_penalty'] = float(penalty_lambda) * n_wrong


        out['coverage'] = len(cells) / total_cells
        out['pa_accuracy'] = correct_cells / total_cells
        out['pa_accuracy_binary_pass'] = int(bool(cells) and correct_cells == len(cells))
        diagnosis['accuracy'] = {
            'status': 'pass' if out['pa_accuracy_binary_pass'] else 'fail',
            'reason': ('All populated attribute cells match ground truth' if out['pa_accuracy_binary_pass']
                       else 'No populated attribute cells' if not cells
                       else 'Populated attribute cells differ from ground truth'),
            'wrong_cells': n_wrong,
            'total_cells': total_cells,
            'populated_cells': len(cells),
            'correct_cells': correct_cells,
            'mismatches': [
                {'house': h, 'attribute': c, 'predicted': value, 'expected': gt[h][c]}
                for h, c, value in cells if value != gt[h][c]
            ],
        }
        stage = 'entity_validation'


        tokens = {}

        for column, values in attrs.items():
            for value in values:
                if value in tokens or value in ('?', '', 'and', 'or', 'not'):
                    diagnosis['invalid_entity_token'] = {'attribute': column, 'value': value}
                    raise ValueError('Ambiguous/reserved entity token')
                tokens[value] = z3.Int(f'{column}:{value}')


        stage = 'reasoning_validation'
        reasoning = payload["reasoning"]

        if not isinstance(reasoning, (list, tuple)):
            raise ValueError("Reasoning must be a list or tuple")

        constraints = []

        for entry_index, step in enumerate(reasoning, 1):
            current_step = {'entry_index': entry_index, 'raw': step}
            if not isinstance(step, str):
                raise ValueError("Non-string reasoning entry")

            # Identify syntactic steps by prefix, not list position.
            match = re.match(r"^\s*S(\d+)\s*:\s*(.*)$", step)

            if match is None:
                # Ignore natural-language explanations and conclusions.
                continue

            current_step['step_id'] = 'S' + match.group(1)
            expression = match.group(2).strip()

            # Accept the prompt's terminating period.
            if expression.endswith("."):
                expression = expression[:-1].strip()

            if not expression:
                raise ValueError(
                    f"Empty constraint in syntactic step S{match.group(1)}")

            # Parse every recognized syntactic step.
            # Invalid constraints fail consistency; never silently skip them.
            constraints.append(_constraint(expression, tokens))




        current_step = None
        stage = 'solver_check'
        solver = z3.Solver()
        solver.set(timeout=max(1, int(float(timeout_s) * 1000)))
        # House bounds and uniqueness within each attribute.
        # Puzzle clues remain excluded.
        for values in attrs.values():
            variables = [tokens[value] for value in values]

            solver.add(*[
                z3.And(variable >= 1, variable <= n)
                for variable in variables
            ])
            solver.add(z3.Distinct(*variables))
        solver.add(*constraints)
        # Positive assignments: no negation and no separate entailment check.
        solver.add(*[tokens[value] == h for h, c, value in cells])
        solver_result = solver.check()
        out['pa_consistency_binary_pass'] = int(solver_result == z3.sat)
        diagnosis['consistency'] = {
            'status': 'pass' if solver_result == z3.sat else 'fail',
            'solver_result': str(solver_result),
            'reason': ('Domain rules, reasoning, and populated assignments are jointly SAT'
                       if solver_result == z3.sat else
                       'Domain rules, reasoning, and populated assignments are jointly UNSAT'
                       if solver_result == z3.unsat else solver.reason_unknown()),
            'n_constraints_parsed': len(constraints),
            'n_populated_assignments': len(cells),
        }

        # Case 5: penalize coverage whenever either binary gate fails.
        if (
                out['pa_accuracy_binary_pass']
                and out['pa_consistency_binary_pass']
        ):
            out['pa_reward'] = 1.0
        else:
            out['pa_reward'] = -out['coverage']

    except (KeyError, TypeError, ValueError, AttributeError, SyntaxError,
            OverflowError, RecursionError, z3.Z3Exception) as exc:
        error_details = {
            'status': 'fail', 'stage': stage,
            'error_type': type(exc).__name__, 'reason': str(exc),
        }
        if current_step is not None:
            error_details['reasoning_entry'] = current_step
        diagnosis['consistency'] = dict(error_details)
        if diagnosis['accuracy']['status'] == 'not_evaluated':
            diagnosis['accuracy'] = {
                **error_details,
                'reason': 'Accuracy not evaluated: ' + str(exc),
            }
        # Invalid input/solver failure must never create a positive PA reward.
        out['pa_consistency_binary_pass'] = 0
        out['pa_reward'] = 0.0
    diagnosis['pa_reward'] = {
        'value': out['pa_reward'],
        'reason': 'Both gates pass' if out['pa_reward'] > 0 else 'At least one gate did not pass',
    }
    return out


if __name__ == '__main__':
    from copy import deepcopy
    import math

    # ------------------------------------------------------------
    # Shared puzzle
    # ------------------------------------------------------------
    ground_truth = {
        'header': ['House', 'Name', 'Color', 'Children'],
        'rows': [
            ['1', 'Peter', 'yellow', 'Bella'],
            ['2', 'Arnold', 'red', 'Fred'],
            ['3', 'Eric', 'white', 'Meredith'],
        ],
    }

    base_payload = {
        'n_houses': 3,

        'attribute_values': {
            'Name': ['Peter', 'Arnold', 'Eric'],
            'Color': ['yellow', 'red', 'white'],
            'Children': ['Bella', 'Fred', 'Meredith'],
        },

        'ground_truth': deepcopy(ground_truth),
        'solution': deepcopy(ground_truth),

        'reasoning': [
            'Red occupies house two.',
            'S1: red == 2.',
            'Arnold lives in the red house.',
            'S2: Arnold == red.',
        ],

        # Included for payload compatibility. Its constraints are
        # not used in the PA consistency solver.
        'z3_out': {},
    }

    def empty_solution(payload):
        """Replace all attribute cells with '?'."""
        result = deepcopy(payload)

        for row in result['solution']['rows']:
            row[1:] = ['?'] * (len(row) - 1)

        return result

    def expected(
            accuracy_gate,
            pa_accuracy,
            coverage,
            consistency_gate,
            n_wrong,
            n_wrong_available,
            penalty,
            pa_reward,
    ):
        return {
            'pa_accuracy_binary_pass': accuracy_gate,
            'pa_accuracy': pa_accuracy,
            'coverage': coverage,
            'pa_consistency_binary_pass': consistency_gate,
            'n_wrong': n_wrong,
            'n_wrong_available': n_wrong_available,
            'wrong_cell_penalty': penalty,
            'pa_reward': pa_reward,
        }

    tests = []

    # ------------------------------------------------------------
    # 1. Correct complete solution, both gates pass.
    # ------------------------------------------------------------
    tests.append((
        'Correct complete solution',
        deepcopy(base_payload),
        1.0,
        expected(
            accuracy_gate=1,
            pa_accuracy=1.0,
            coverage=1.0,
            consistency_gate=1,
            n_wrong=0,
            n_wrong_available=1,
            penalty=0.0,
            pa_reward=1.0,
        ),
    ))

    # ------------------------------------------------------------
    # 2. Correct partial solution: six of nine cells populated.
    #
    # Case 4 uses a constant gate reward: coverage is logged but
    # does not increase the reward.
    # ------------------------------------------------------------
    p = deepcopy(base_payload)

    for row in p['solution']['rows']:
        row[3] = '?'

    tests.append((
        'Correct partial solution: six of nine cells',
        p,
        1.0,
        expected(
            accuracy_gate=1,
            pa_accuracy=6 / 9,
            coverage=6 / 9,
            consistency_gate=1,
            n_wrong=0,
            n_wrong_available=1,
            penalty=0.0,
            pa_reward=1.0,
        ),
    ))

    # ------------------------------------------------------------
    # 3. One incorrect cell compatible with sparse reasoning.
    #
    # White is placed in house 1, while its correct house is 3.
    # The other Color cells are unknown, so there is no duplicate
    # Color assignment in the predicted partial table.
    # ------------------------------------------------------------
    p = empty_solution(base_payload)

    p['solution']['rows'][0][2] = 'white'

    tests.append((
        'One wrong cell; consistency passes',
        p,
        1.0,
        expected(
            accuracy_gate=0,
            pa_accuracy=0.0,
            coverage=1 / 9,
            consistency_gate=1,
            n_wrong=1,
            n_wrong_available=1,
            penalty=1.0,
            pa_reward=-1.0,
        ),
    ))

    # ------------------------------------------------------------
    # 4. Two incorrect cells compatible with the reasoning.
    #
    # Swap yellow and white. The sparse reasoning fixes red in
    # house 2 but does not constrain yellow or white.
    # ------------------------------------------------------------
    p = deepcopy(base_payload)

    p['solution']['rows'][0][2] = 'white'
    p['solution']['rows'][2][2] = 'yellow'

    tests.append((
        'Two wrong cells; consistency passes',
        p,
        1.0,
        expected(
            accuracy_gate=0,
            pa_accuracy=7 / 9,
            coverage=1.0,
            consistency_gate=1,
            n_wrong=2,
            n_wrong_available=1,
            penalty=2.0,
            pa_reward=-2.0,
        ),
    ))

    # ------------------------------------------------------------
    # 5. Correct answer, but reasoning contradicts the answer.
    #
    # There are no wrong cells, so the penalty is zero. The gate
    # reward is also zero because consistency fails.
    # ------------------------------------------------------------
    p = deepcopy(base_payload)

    p['reasoning'].extend([
        'Peter occupies house three.',
        'S3: Peter == 3.',
    ])

    tests.append((
        'Correct answer; consistency fails',
        p,
        1.0,
        expected(
            accuracy_gate=1,
            pa_accuracy=1.0,
            coverage=1.0,
            consistency_gate=0,
            n_wrong=0,
            n_wrong_available=1,
            penalty=0.0,
            pa_reward=0.0,
        ),
    ))

    # ------------------------------------------------------------
    # 6. Two wrong cells and inconsistent reasoning.
    #
    # Arnold is moved to house 1 even though the reasoning implies
    # Arnold == red == 2.
    # ------------------------------------------------------------
    p = deepcopy(base_payload)

    p['solution']['rows'][0][1] = 'Arnold'
    p['solution']['rows'][1][1] = 'Peter'

    tests.append((
        'Two wrong cells; both gates fail',
        p,
        1.0,
        expected(
            accuracy_gate=0,
            pa_accuracy=7 / 9,
            coverage=1.0,
            consistency_gate=0,
            n_wrong=2,
            n_wrong_available=1,
            penalty=2.0,
            pa_reward=-2.0,
        ),
    ))

    # ------------------------------------------------------------
    # 7. All attribute cells unknown.
    #
    # Under the current policy, accuracy fails while consistency
    # reflects SAT of the domain and reasoning.
    # ------------------------------------------------------------
    p = empty_solution(base_payload)

    tests.append((
        'All cells unknown; reasoning is satisfiable',
        p,
        1.0,
        expected(
            accuracy_gate=0,
            pa_accuracy=0.0,
            coverage=0.0,
            consistency_gate=1,
            n_wrong=0,
            n_wrong_available=1,
            penalty=0.0,
            pa_reward=0.0,
        ),
    ))

    # ------------------------------------------------------------
    # 8. One wrong cell and a malformed reasoning constraint.
    #
    # The wrong-cell count is available, but malformed reasoning raises
    # an exception. Under the exception policy, pa_reward is zero.
    # ------------------------------------------------------------
    p = empty_solution(base_payload)

    p['solution']['rows'][0][2] = 'white'

    p['reasoning'].extend([
        'Use an undefined entity.',
        'S3: unknown_entity == 1.',
    ])

    tests.append((
        'Wrong cell and malformed reasoning',
        p,
        1.0,
        expected(
            accuracy_gate=0,
            pa_accuracy=0.0,
            coverage=1 / 9,
            consistency_gate=0,
            n_wrong=1,
            n_wrong_available=1,
            penalty=1.0,
            pa_reward=0.0,
        ),
    ))

    # ------------------------------------------------------------
    # 9. Configurable penalty strength.
    #
    # Two wrong cells with lambda = 0.5 incur a total penalty of 1.
    # ------------------------------------------------------------
    p = deepcopy(base_payload)

    p['solution']['rows'][0][2] = 'white'
    p['solution']['rows'][2][2] = 'yellow'

    tests.append((
        'Two wrong cells with penalty_lambda = 0.5',
        p,
        0.5,
        expected(
            accuracy_gate=0,
            pa_accuracy=7 / 9,
            coverage=1.0,
            consistency_gate=1,
            n_wrong=2,
            n_wrong_available=1,
            penalty=1.0,
            pa_reward=-1.0,
        ),
    ))

    # ------------------------------------------------------------
    # 10. Malformed table: wrong-cell count is unavailable.
    #
    # Current proposed policy returns zero reward when validation
    # fails before the table can be compared with ground truth.
    # ------------------------------------------------------------
    p = deepcopy(base_payload)

    # Header has four columns, but this row has only three cells.
    p['solution']['rows'][1] = ['2', 'Arnold', 'red']

    tests.append((
        'Malformed row: wrong-cell count unavailable',
        p,
        1.0,
        expected(
            accuracy_gate=0,
            pa_accuracy=0.0,
            coverage=0.0,
            consistency_gate=0,
            n_wrong=0,
            n_wrong_available=0,
            penalty=0.0,
            pa_reward=0.0,
        ),
    ))

    # ------------------------------------------------------------
    # Run tests
    # ------------------------------------------------------------
    failures = []

    for label, payload, penalty_lambda, target in tests:
        print(f'\n{"=" * 70}')
        print(label)

        try:
            result = compute_pa_solution(
                payload,
                penalty_lambda=penalty_lambda,
            )

            print('Result:', result)

            errors = []

            for key, expected_value in target.items():
                actual = result.get(key)

                if actual is None:
                    errors.append(f'{key}: missing')
                    continue

                if not math.isclose(
                    float(actual),
                    float(expected_value),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f'{key}: expected {expected_value}, got {actual}'
                    )

            if 'DIAGNOSIS' not in result:
                errors.append('DIAGNOSIS: missing')

            if errors:
                message = '; '.join(errors)
                failures.append((label, message))
                print('FAIL:', message)
            else:
                print('PASS')

        except Exception as exc:
            failures.append((label, repr(exc)))
            print('ERROR:', repr(exc))

    passed = len(tests) - len(failures)

    print(f'\nPassed {passed}/{len(tests)} Case 4 tests.')

    if failures:
        raise AssertionError(
            '\n'.join(
                f'{label}: {error}'
                for label, error in failures
            )
        )