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

        out['coverage'] = len(cells) / total_cells
        out['pa_accuracy'] = correct_cells / total_cells
        out['pa_accuracy_binary_pass'] = int(bool(cells) and correct_cells == len(cells))
        diagnosis['accuracy'] = {
            'status': 'pass' if out['pa_accuracy_binary_pass'] else 'fail',
            'reason': ('All populated attribute cells match ground truth' if out['pa_accuracy_binary_pass']
                       else 'No populated attribute cells' if not cells
                       else 'Populated attribute cells differ from ground truth'),
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
        if out['pa_accuracy_binary_pass'] and out['pa_consistency_binary_pass']:
            out['pa_reward'] = 1.0 + out['coverage']
        else:
            out['pa_reward'] = 0.0

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
    add('All-empty table: satisfiable reasoning passes consistency', p, 0, 0.0, 1, 0.0)
    p = deepcopy(base)
    for row in p['solution']['rows']:
        row[1:] = ['?', '?', '?']
    p['solution']['rows'][0][1] = 'Peter'
    add('One correct cell absent from reasoning still passes SAT', p, 1, 1/9, 1, 1+1/9)
    p = deepcopy(base); p['reasoning'] = []
    add('Empty reasoning is compatible with a correct complete table', p, 1, 1.0, 1, 2.0)
    p = deepcopy(base); p['reasoning'][1] = 'S8: red == 2.'
    add(        "Nonsequential numbering does not affect SAT consistency",
        p, 1, 1.0, 1, 2.0,)
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
        if set(result) != set(keys) | {'DIAGNOSIS'} or any(
                not math.isclose(result[key], target, abs_tol=1e-12)
                for key, target in zip(keys, expected)):
            raise AssertionError(f'{name}: expected {expected}, got {result}')
        print(f'PASS: {name}: {result}')
    print(f'All {len(cases)} tests passed.')

    # Regression: reasoning violates uniqueness within Height.
    payload = {
        "n_houses": 6,
        "attribute_values": {
            "Height": [
                "very_short", "short", "average",
                "tall", "very_tall", "extremely_tall",
            ],
            "Birthday": ["jan", "feb", "mar", "april", "may", "sept"],
        },
        "ground_truth": {
            "header": ["House", "Height", "Birthday"],
            "rows": [
                ["1", "short", "jan"],
                ["2", "average", "sept"],
                ["3", "tall", "feb"],
                ["4", "very_short", "mar"],
                ["5", "very_tall", "april"],
                ["6", "extremely_tall", "may"],
            ],
        },
        "reasoning": [
            "The very short person occupies house four.",
            "S1: very_short == 4.",
            "The March birthday belongs to the short person.",
            "S2: mar == short.",
            "The March birthday is in house four.",
            "S3: mar == 4.",
            "The September birthday is in house two.",
            "S4: sept == 2.",
        ],
        "solution": {
            "header": ["House", "Height", "Birthday"],
            "rows": [
                ["1", "?", "?"],
                ["2", "?", "sept"],
                ["3", "?", "?"],
                ["4", "very_short", "mar"],
                ["5", "?", "?"],
                ["6", "?", "?"],
            ],
        },
        "z3_out": {},
    }

    result = compute_pa_solution(payload)
    print("Height uniqueness contradiction:", result)

    expected = {
        "pa_accuracy_binary_pass": 1,
        "pa_accuracy": 3 / 12,
        "coverage": 3 / 12,
        "pa_consistency_binary_pass": 0,
        "pa_reward": 0.0,
    }

    import math

    for key, target in expected.items():
        actual = result.get(key)
        assert actual is not None and math.isclose(
            float(actual), float(target),
            rel_tol=1e-9, abs_tol=1e-12,
        ), f"{key}: expected {target}, got {actual}"

    print("PASS: conflicting Height assignments correctly fail consistency.")




    from copy import deepcopy
    import math

    def make_log_test(n_houses, attributes, header, rows, steps,
                      conclusion=None):
        table = {"header": header, "rows": rows}
        reasoning = []
        for step in steps:
            reasoning.extend(["Apply the following constraint.", step])
        if conclusion:
            reasoning.append(conclusion)

        return {
            "n_houses": n_houses,
            "attribute_values": attributes,
            "ground_truth": deepcopy(table),
            "solution": deepcopy(table),
            "reasoning": reasoning,
            "z3_out": {},
        }

    log_tests = []

    # 1. Correct table, but S6 and S10 contradict each other.
    log_tests.append((
        "lgp-test-3x3-30: contradictory reasoning",
        make_log_test(
            3,
            {
                "Name": ["Eric", "Peter", "Arnold"],
                "Children": ["Bella", "Fred", "Meredith"],
                "CarModel": [
                    "toyota_camry", "tesla_model_3", "ford_f150",
                ],
            },
            ["House", "Name", "Children", "CarModel"],
            [
                ["1", "Peter", "Meredith", "toyota_camry"],
                ["2", "Arnold", "Fred", "ford_f150"],
                ["3", "Eric", "Bella", "tesla_model_3"],
            ],
            [
                "S1: Meredith + 1 == Fred.",
                "S2: Peter + 2 == tesla_model_3.",
                "S3: Peter == 1.",
                "S4: Fred > toyota_camry.",
                "S5: Or(Meredith == 1, Meredith == 2).",
                "S6: Meredith == 2.",
                "S7: Fred == 3.",
                "S8: toyota_camry == 1.",
                "S9: ford_f150 == 2.",
                "S10: Meredith == 1.",
                "S11: Arnold == 2.",
                "S12: Eric == 3.",
                "S13: Bella == 3.",
            ],
        ),
        0,
    ))

    # 2. Correct table, but "flower" is not a declared entity token.
    log_tests.append((
        "lgp-test-2x5-5: invalid syntactic constraint",
        make_log_test(
            2,
            {
                "Name": ["Arnold", "Eric"],
                "Mother": ["Holly", "Aniya"],
                "Animal": ["horse", "cat"],
                "Birthday": ["april", "sept"],
                "Flower": ["carnations", "daffodils"],
            },
            ["House", "Name", "Mother", "Animal", "Birthday", "Flower"],
            [
                ["1", "Arnold", "Aniya", "cat", "sept", "carnations"],
                ["2", "Eric", "Holly", "horse", "april", "daffodils"],
            ],
            [
                "S1: april == Holly.",
                "S2: Arnold == Aniya.",
                "S3: Eric == Holly.",
                "S4: Eric == april.",
                "S5: Arnold == sept.",
                "S6: sept + 1 == horse.",
                "S7: horse == 2.",
                "S8: cat == 1.",
                "S9: cat == carnations.",
                "S10: flower == daffodils.",
            ],
        ),
        0,
    ))

    # 3. Compatible reasoning with an extra concluding sentence.
    # The odd-length list must not cause a consistency failure.
    log_tests.append((
        "lgp-test-3x2-2: compatible odd-length trace",
        make_log_test(
            3,
            {
                "Name": ["Eric", "Peter", "Arnold"],
                "Mother": ["Janelle", "Aniya", "Holly"],
            },
            ["House", "Name", "Mother"],
            [
                ["1", "Arnold", "Janelle"],
                ["2", "Peter", "Holly"],
                ["3", "Eric", "Aniya"],
            ],
            [
                "S1: Aniya == 3.",
                "S2: Eric == 3.",
                "S3: Janelle == Arnold.",
                "S4: Eric == Aniya.",
                "S5: Or(Holly == 2, Holly == 3).",
                "S6: Holly == 2.",
                "S7: Janelle == 1.",
                "S8: Arnold == 1.",
                "S9: Peter == 2.",
            ],
            conclusion=(
                "House 3 has name Eric and mother Aniya, "
                "so the solution is complete."
            ),
        ),
        1,
    ))

    # 4. Compatible And(...) constraint without a terminating period.
    # Period/interleaving requirements belong to the format validator.
    log_tests.append((
        "lgp-test-2x3-5: compatible compound constraint",
        make_log_test(
            2,
            {
                "Name": ["Arnold", "Eric"],
                "Drink": ["tea", "water"],
                "Cigar": ["prince", "pall_mall"],
            },
            ["House", "Name", "Drink", "Cigar"],
            [
                ["1", "Arnold", "tea", "pall_mall"],
                ["2", "Eric", "water", "prince"],
            ],
            [
                "S1: tea == 1.",
                "S2: Eric == 2.",
                "S3: Arnold == 1.",
                "S4: Arnold == tea.",
                "S5: water == 2.",
                "S6: pall_mall == 1.",
                "S7: prince == 2.",
                "S8: And(Arnold == 1, tea == 1)",
            ],
        ),
        1,
    ))

    failures = []

    for label, test_payload, expected_consistency in log_tests:
        result = compute_pa_solution(test_payload)

        expected = {
            "pa_accuracy_binary_pass": 1,
            "pa_accuracy": 1.0,
            "coverage": 1.0,
            "pa_consistency_binary_pass": expected_consistency,
            "pa_reward": 2.0 if expected_consistency else 0.0,
        }

        errors = []
        for key, target in expected.items():
            actual = result.get(key)
            if actual is None or not math.isclose(
                float(actual), float(target),
                rel_tol=1e-9, abs_tol=1e-12,
            ):
                errors.append(
                    f"{key}: expected {target}, got {actual}"
                )

        print(f"\n{label}\n{result}")

        if errors:
            failures.append(f"{label}: {'; '.join(errors)}")
            print("FAIL:", "; ".join(errors))
        else:
            print("PASS")

    print(
        f"\nPassed {len(log_tests) - len(failures)}"
        f"/{len(log_tests)} additional tests."
    )

    if failures:
        raise AssertionError("\n".join(failures))




    from copy import deepcopy
    import math

    def make_issue_payload(attributes, rows, steps):
        header = ["House"] + list(attributes)
        table = {"header": header, "rows": rows}
        reasoning = []
        for step in steps:
            reasoning.extend(["Apply this constraint.", step])

        return {
            "n_houses": len(rows),
            "attribute_values": attributes,
            "ground_truth": deepcopy(table),
            "solution": deepcopy(table),
            "reasoning": reasoning,
            "z3_out": {},
        }

    # Expected tuple:
    # (accuracy gate, pa_accuracy, coverage, consistency gate, pa_reward)
    issue_tests = []

    # 1. Different Name values cannot both occupy house 2.
    p = make_issue_payload(
        {"Name": ["Peter", "Alice", "Eric"]},
        [
            ["1", "Peter"],
            ["2", "Alice"],
            ["3", "Eric"],
        ],
        [
            "S1: Peter == 2.",
            "S2: Alice == 2.",
        ],
    )
    issue_tests.append((
        "Different names assigned to the same house",
        p,
        (1, 1.0, 1.0, 0, 0.0),
    ))

    # 2. Duplicate bare token across different attributes.
    # Both HairColor and Color contain "red".
    p = make_issue_payload(
        {
            "HairColor": ["red", "black"],
            "Color": ["blue", "red"],
        },
        [
            ["1", "red", "blue"],
            ["2", "black", "red"],
        ],
        ["S1: red == 1."],
    )
    issue_tests.append((
        "Ambiguous red token across attributes",
        p,
        (1, 1.0, 1.0, 0, 0.0),
    ))

    # 3. Unsupported alias: domain uses april, prediction uses apr.
    # Current implementation rejects the table before computing metrics.
    p = make_issue_payload(
        {"Birthday": ["jan", "april", "sept"]},
        [
            ["1", "jan"],
            ["2", "april"],
            ["3", "sept"],
        ],
        ["S1: april == 2."],
    )
    p["solution"]["rows"][1][1] = "apr"
    issue_tests.append((
        "Unsupported birthday alias apr",
        p,
        (0, 0.0, 0.0, 0, 0.0),
    ))

    # 4. Valid entity placed in the wrong attribute column.
    p = make_issue_payload(
        {
            "Smoothie": ["cherry", "lime"],
            "Food": ["spaghetti", "pizza"],
        },
        [
            ["1", "cherry", "spaghetti"],
            ["2", "lime", "pizza"],
        ],
        ["S1: spaghetti == 1."],
    )
    p["solution"]["rows"][0][1] = "spaghetti"
    issue_tests.append((
        "Food value incorrectly placed under Smoothie",
        p,
        (0, 0.0, 0.0, 0, 0.0),
    ))

    # 5. All-empty solution with satisfiable relational reasoning.
    # Under your latest policy, consistency passes but reward stays zero.
    p = make_issue_payload(
        {
            "Name": ["Eric", "Arnold", "Peter"],
            "Nationality": ["norwegian", "brit", "dane"],
            "MusicGenre": ["hip_hop", "rock", "pop"],
        },
        [
            ["1", "Eric", "norwegian", "hip_hop"],
            ["2", "Arnold", "brit", "rock"],
            ["3", "Peter", "dane", "pop"],
        ],
        [
            "S1: Eric == norwegian.",
            "S2: hip_hop == Eric.",
            "S3: Not(Eric == 2).",
        ],
    )
    for row in p["solution"]["rows"]:
        row[1:] = ["?", "?", "?"]

    issue_tests.append((
        "All-empty answer with satisfiable reasoning",
        p,
        (0, 0.0, 0.0, 1, 0.0),
    ))

    metric_keys = (
        "pa_accuracy_binary_pass",
        "pa_accuracy",
        "coverage",
        "pa_consistency_binary_pass",
        "pa_reward",
    )

    failures = []

    for label, test_payload, expected in issue_tests:
        result = compute_pa_solution(test_payload)
        errors = []

        for key, target in zip(metric_keys, expected):
            actual = result.get(key)
            if actual is None or not math.isclose(
                float(actual), float(target),
                rel_tol=1e-9, abs_tol=1e-12,
            ):
                errors.append(
                    f"{key}: expected {target}, got {actual}"
                )

        print(f"\n{label}\n{result}")

        if errors:
            failures.append(f"{label}: {'; '.join(errors)}")
            print("FAIL:", "; ".join(errors))
        else:
            print("PASS")

    print(
        f"\nPassed {len(issue_tests) - len(failures)}"
        f"/{len(issue_tests)} issue tests."
    )

    if failures:
        raise AssertionError("\n".join(failures))


    from copy import deepcopy
    import math

    def build_extra_payload(attributes, rows, constraints):
        table = {
            "header": ["House"] + list(attributes),
            "rows": rows,
        }
        return {
            "n_houses": len(rows),
            "attribute_values": attributes,
            "ground_truth": deepcopy(table),
            "solution": deepcopy(table),
            "reasoning": constraints,
            "z3_out": {},
        }

    # Expected order:
    # (accuracy gate, pa_accuracy, coverage, consistency gate, pa_reward)
    extra_tests = []

    # 1. Incorrect cell remains compatible with the reasoning.
    p = build_extra_payload(
        {
            "Name": ["Arnold", "Peter", "Alice", "Eric"],
            "Height": ["very_short", "tall", "short", "average"],
            "Hobby": ["painting", "photography", "gardening", "cooking"],
        },
        [
            ["1", "Arnold", "very_short", "painting"],
            ["2", "Peter", "tall", "photography"],
            ["3", "Alice", "short", "gardening"],
            ["4", "Eric", "average", "cooking"],
        ],
        [
            "S1: photography == Peter.",
            "S2: Peter + 1 == gardening.",
            "S3: cooking == Eric.",
            "S4: Not(Eric == 3).",
        ],
    )
    for row in p["solution"]["rows"]:
        row[1:] = ["?", "?", "?"]
    p["solution"]["rows"][0][3] = "photography"

    extra_tests.append((
        "Wrong cell compatible with reasoning",
        p,
        (0, 0.0, 1 / 12, 1, 0.0),
    ))

    # 2. Answer assignments violate a relative-position constraint.
    p = build_extra_payload(
        {
            "Name": ["Eric", "Alice", "Arnold", "Bob", "Peter"],
            "PhoneModel": [
                "samsung_galaxy_s21", "oneplus_9", "google_pixel_6",
                "huawei_p50", "iphone_13",
            ],
        },
        [
            ["1", "Eric", "samsung_galaxy_s21"],
            ["2", "Alice", "oneplus_9"],
            ["3", "Arnold", "google_pixel_6"],
            ["4", "Bob", "huawei_p50"],
            ["5", "Peter", "iphone_13"],
        ],
        ["S4: Arnold + 1 == huawei_p50."],
    )
    for row in p["solution"]["rows"]:
        row[1:] = ["?", "?"]
    p["solution"]["rows"][1][1] = "Arnold"       # Wrong: house 2.
    p["solution"]["rows"][3][2] = "huawei_p50"  # Correct: house 4.

    extra_tests.append((
        "Relative-position conflict: 2 + 1 != 4",
        p,
        (0, 1 / 10, 2 / 10, 0, 0.0),
    ))

    # 3. Raw header alias mismatch fails input validation.
    # GT uses Sport; prediction and attribute_values use FavoriteSport.
    p = build_extra_payload(
        {
            "Name": ["Eric", "Arnold"],
            "FavoriteSport": ["swimming", "soccer"],
        },
        [
            ["1", "Eric", "swimming"],
            ["2", "Arnold", "soccer"],
        ],
        ["S1: Eric == swimming."],
    )
    p["ground_truth"]["header"][2] = "Sport"

    extra_tests.append((
        "Unnormalized Sport / FavoriteSport headers",
        p,
        (0, 0.0, 0.0, 0, 0.0),
    ))

    # 4. A row with fewer cells than the header is rejected.
    p = build_extra_payload(
        {
            "Name": ["Alice", "Arnold", "Eric", "Peter"],
            "FavoriteSport": ["tennis", "basketball", "swimming", "soccer"],
            "CarModel": [
                "ford_f150", "honda_civic", "tesla_model_3", "toyota_camry",
            ],
            "Cigar": ["prince", "dunhill", "blue_master", "pall_mall"],
            "Occupation": ["teacher", "engineer", "artist", "doctor"],
            "Pet": ["fish", "dog", "bird", "cat"],
        },
        [
            ["1", "Alice", "tennis", "ford_f150",
             "prince", "teacher", "fish"],
            ["2", "Arnold", "basketball", "honda_civic",
             "dunhill", "engineer", "dog"],
            ["3", "Eric", "swimming", "tesla_model_3",
             "blue_master", "artist", "bird"],
            ["4", "Peter", "soccer", "toyota_camry",
             "pall_mall", "doctor", "cat"],
        ],
        [
            "S1: Eric == tesla_model_3.",
            "S2: Not(Eric == 4).",
        ],
    )
    p["solution"]["rows"][1] = ["2", "?", "?", "?", "dunhill", "?"]

    extra_tests.append((
        "Malformed row: six cells under seven headers",
        p,
        (0, 0.0, 0.0, 0, 0.0),
    ))

    extra_metric_keys = (
        "pa_accuracy_binary_pass",
        "pa_accuracy",
        "coverage",
        "pa_consistency_binary_pass",
        "pa_reward",
    )
    extra_failures = []

    for label, test_payload, expected in extra_tests:
        result = compute_pa_solution(test_payload)
        errors = []

        for key, target in zip(extra_metric_keys, expected):
            actual = result.get(key)
            if actual is None or not math.isclose(
                float(actual), float(target),
                rel_tol=1e-9, abs_tol=1e-12,
            ):
                errors.append(f"{key}: expected {target}, got {actual}")

        print(f"\n{label}\n{result}")

        if errors:
            extra_failures.append(f"{label}: {'; '.join(errors)}")
            print("FAIL:", "; ".join(errors))
        else:
            print("PASS")

    print(
        f"\nPassed {len(extra_tests) - len(extra_failures)}"
        f"/{len(extra_tests)} additional tests."
    )

    if extra_failures:
        raise AssertionError("\n".join(extra_failures))