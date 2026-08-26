import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from z3 import Solver, Int, Distinct, And, Or, Not, sat


# Number of blank lines between examples
EXAMPLE_GAP = 8


class Tee:
    """
    Print output to both terminal and a text file.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def separator(title=None, char="=", width=120):
    if title:
        print(f"\n{char * width}")
        print(title)
        print(char * width)
    else:
        print(char * width)


def print_dict(data, indent=0):
    """
    Recursively print dictionaries/lists in a human-readable form.
    """

    prefix = " " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}{key}:")
                print_dict(value, indent + 4)
            else:
                print(f"{prefix}{key}: {value}")

    elif isinstance(data, list):
        for i, value in enumerate(data, start=1):
            if isinstance(value, (dict, list)):
                print(f"{prefix}[{i}]")
                print_dict(value, indent + 4)
            else:
                print(f"{prefix}[{i}] {value}")

    else:
        print(f"{prefix}{data}")


def print_ground_truth(gt):
    """
    Pretty-print solution tables.
    """

    if not isinstance(gt, dict):
        print(gt)
        return

    header = gt.get("header", [])
    rows = gt.get("rows", [])

    if not header:
        print_dict(gt)
        return

    print(" | ".join(str(x) for x in header))
    print("-" * 120)

    for row in rows:
        print(" | ".join(str(x) for x in row))


# ============================================================
# Fresh Z3 reconstruction helpers
# ============================================================


def normalize_value(value):
    """
    Canonicalize table headers/values before semantic comparison.

    The logged Z3 solution commonly uses symbolic identifiers such as:
        high_school, tesla_model_3, pall_mall

    while ground_truth / processed_prediction may use display forms such as:
        high school, Tesla Model 3, pall mall

    These should be treated as the same value.
    """
    value = str(value).strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def make_identifier(value):
    """
    Convert an attribute value to the identifier form used in symbolic clues.

    Examples:
        "high school" -> "high_school"
        "Google Pixel 6" -> "Google_Pixel_6"
    """

    value = str(value).strip()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value)

    if not value:
        raise ValueError("Empty identifier after normalization")

    if value[0].isdigit():
        value = "v_" + value

    return value


def strip_symbolic_label(text):
    """
    Remove C1:/S1: labels and a final period.
    """

    text = str(text).strip()
    text = re.sub(
        r"^[CS]\d+\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip().rstrip(".")


class Z3PuzzleContext:
    """
    Fresh symbolic representation of one Zebra puzzle.

    Every attribute value is represented as an Int whose value is the
    corresponding house number. For every attribute/category, values are
    constrained to 1..n_houses and are Distinct.
    """

    def __init__(self, payload):
        self.n_houses = int(payload.get("n_houses"))
        self.attribute_values = payload.get("attribute_values", {}) or {}

        if not self.attribute_values:
            raise ValueError("payload.attribute_values is missing or empty")

        self.variables = {}
        self.category_variables = defaultdict(list)
        self.name_candidates = defaultdict(list)
        self.base_constraints = []

        for category, values in self.attribute_values.items():
            category_id = make_identifier(category)

            for value in values:
                value_id = make_identifier(value)

                # Namespace the underlying Z3 variable to avoid accidental
                # collisions when two categories share the same surface value.
                z3_name = f"{category_id}__{value_id}"
                variable = Int(z3_name)

                self.variables[(category, str(value))] = variable
                self.category_variables[category].append(variable)

                # Surface-name lookup used by C_i/S_i expressions.
                self.name_candidates[value_id].append(variable)
                self.name_candidates[value_id.lower()].append(variable)

                self.base_constraints.append(variable >= 1)
                self.base_constraints.append(variable <= self.n_houses)

            category_vars = self.category_variables[category]
            if len(category_vars) > 1:
                self.base_constraints.append(Distinct(*category_vars))

    def resolve_name(self, name):
        """
        Resolve a symbolic token appearing in a C_i/S_i expression.

        If the same surface token exists in multiple categories, the symbolic
        expression is ambiguous and we explicitly report it instead of silently
        choosing one variable.
        """

        candidates = self.name_candidates.get(name, [])

        if not candidates:
            candidates = self.name_candidates.get(name.lower(), [])

        # De-duplicate because both exact/lowercase aliases may point to the
        # same object.
        unique = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                unique.append(candidate)
                seen.add(key)

        if not unique:
            raise ValueError(f"Unknown symbolic token: {name}")

        if len(unique) > 1:
            raise ValueError(
                f"Ambiguous symbolic token '{name}' maps to multiple attributes"
            )

        return unique[0]


def ast_to_z3(node, context):
    """
    Convert the restricted expression grammar used in the logs into Z3.

    Supported examples include:
        Alice == 3
        Alice != 4
        Alice < Bob
        Alice + 1 == Bob
        dane + half == google_pixel_6
        Or(Alice == 1, Alice == 2)
        And(Alice == 1, red == 1)
        Not(Alice == 3)
        mountain == 3 == lilies
    """

    if isinstance(node, ast.Name):
        return context.resolve_name(node.id)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, bool)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.UnaryOp):
        operand = ast_to_z3(node.operand, context)

        if isinstance(node.op, ast.USub):
            return -operand

        if isinstance(node.op, ast.UAdd):
            return operand

        if isinstance(node.op, ast.Not):
            return Not(operand)

        raise ValueError(
            f"Unsupported unary operator: {type(node.op).__name__}"
        )

    if isinstance(node, ast.BinOp):
        left = ast_to_z3(node.left, context)
        right = ast_to_z3(node.right, context)

        if isinstance(node.op, ast.Add):
            return left + right

        if isinstance(node.op, ast.Sub):
            return left - right

        raise ValueError(
            f"Unsupported binary operator: {type(node.op).__name__}"
        )

    if isinstance(node, ast.Compare):
        left = ast_to_z3(node.left, context)
        comparisons = []

        for op, comparator_node in zip(node.ops, node.comparators):
            right = ast_to_z3(comparator_node, context)

            if isinstance(op, ast.Eq):
                comparisons.append(left == right)
            elif isinstance(op, ast.NotEq):
                comparisons.append(left != right)
            elif isinstance(op, ast.Lt):
                comparisons.append(left < right)
            elif isinstance(op, ast.LtE):
                comparisons.append(left <= right)
            elif isinstance(op, ast.Gt):
                comparisons.append(left > right)
            elif isinstance(op, ast.GtE):
                comparisons.append(left >= right)
            else:
                raise ValueError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )

            # Python chained comparison semantics:
            # A == B == C  ->  And(A == B, B == C)
            left = right

        if len(comparisons) == 1:
            return comparisons[0]

        return And(*comparisons)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")

        function_name = node.func.id
        arguments = [ast_to_z3(arg, context) for arg in node.args]

        if function_name == "Or":
            return Or(*arguments)

        if function_name == "And":
            return And(*arguments)

        if function_name == "Not":
            if len(arguments) != 1:
                raise ValueError("Not(...) expects exactly one argument")
            return Not(arguments[0])

        raise ValueError(f"Unsupported function: {function_name}")

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def parse_symbolic_expression(text, context):
    """Parse one C_i/S_i expression into a Z3 constraint."""

    expression = strip_symbolic_label(text)

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Python-expression parse error: {e}") from e

    return ast_to_z3(parsed.body, context)


def extract_symbolic_steps(reasoning):
    """
    Extract S_i reasoning statements from the interleaved reasoning and attach
    the natural-language explanation immediately preceding each S_i.

    Example:
        "From clue 3, Eric must be in house 2."
        "Combining this with clue 5 gives..."
        "S4: Eric == 2."

    The two NL lines above are stored as the NL description for S4.
    """

    symbolic_steps = []
    nl_buffer = []

    for reasoning_index, item in enumerate(reasoning, start=1):
        if not isinstance(item, str):
            continue

        item = item.strip()

        match = re.match(
            r"^\s*(S\d+)\s*:\s*(.+?)\s*$",
            item,
            flags=re.IGNORECASE,
        )

        if match:
            symbolic_steps.append(
                {
                    "label": match.group(1),
                    "expression": match.group(2).strip().rstrip("."),
                    "raw": item,
                    "reasoning_index": reasoning_index,
                    "nl_description": list(nl_buffer),
                }
            )

            # NL text after this S_i belongs to the next symbolic step.
            nl_buffer = []

        else:
            nl_buffer.append(item)

    return symbolic_steps


def table_signature(table):
    """
    Convert a Zebra table into a canonical category/value -> houses signature.

    A list of houses is kept for every value so malformed predictions with
    duplicate values are represented faithfully rather than silently collapsed.
    """

    if not isinstance(table, dict):
        return None

    header = table.get("header", [])
    rows = table.get("rows", [])

    if not header or not rows:
        return None

    house_index = None
    for i, column in enumerate(header):
        if normalize_value(column) == "house":
            house_index = i
            break

    if house_index is None:
        return None

    assignments = defaultdict(list)

    for row in rows:
        if not isinstance(row, list) or house_index >= len(row):
            return None

        try:
            house = int(row[house_index])
        except (TypeError, ValueError):
            return None

        for column_index, column_name in enumerate(header):
            if column_index == house_index:
                continue

            if column_index >= len(row):
                return None

            key = (
                normalize_value(column_name),
                normalize_value(row[column_index]),
            )
            assignments[key].append(house)

    return {
        key: tuple(sorted(houses))
        for key, houses in assignments.items()
    }


def tables_match(table_a, table_b):
    """
    Exact semantic table comparison independent of row/header order.

    Returns:
        True / False when both are valid table dictionaries
        None when either side cannot be represented as a table
    """

    signature_a = table_signature(table_a)
    signature_b = table_signature(table_b)

    if signature_a is None or signature_b is None:
        return None

    return signature_a == signature_b


def get_category_to_reference_header(attribute_values, reference_table):
    """
    Map payload category names to reference-table column names.

    This is needed because the payload and GT/prediction can use different
    labels for the same attribute, for example:

        payload: FavoriteSport
        GT:      Sport

    The mapping is inferred primarily from the set of possible values in each
    category. Raw category-name equality is used first when available.
    """

    mapping = {}

    if not isinstance(reference_table, dict):
        return {
            category: category
            for category in attribute_values
        }

    header = reference_table.get("header", []) or []
    rows = reference_table.get("rows", []) or []

    if not header or not rows:
        return {
            category: category
            for category in attribute_values
        }

    # Build the normalized value domain for each reference column.
    reference_domains = {}
    normalized_header_lookup = {}

    for column_index, column_name in enumerate(header):
        normalized_column = normalize_value(column_name)

        if normalized_column == "house":
            continue

        normalized_header_lookup[normalized_column] = column_name

        values = set()
        for row in rows:
            if not isinstance(row, list) or column_index >= len(row):
                continue

            values.add(
                normalize_value(row[column_index])
            )

        reference_domains[column_name] = values

    for category, values in attribute_values.items():
        normalized_category = normalize_value(category)

        # 1. Prefer direct normalized header-name matching.
        if normalized_category in normalized_header_lookup:
            mapping[category] = normalized_header_lookup[normalized_category]
            continue

        # 2. Otherwise infer the corresponding column from its value domain.
        payload_domain = {
            normalize_value(value)
            for value in values
        }

        matching_columns = [
            column_name
            for column_name, domain in reference_domains.items()
            if domain == payload_domain
        ]

        if len(matching_columns) == 1:
            mapping[category] = matching_columns[0]
        else:
            # Do not guess if zero/multiple domains match. The downstream
            # comparison will then expose the mismatch rather than silently
            # mapping the category incorrectly.
            mapping[category] = category

    return mapping


def model_to_table(context, model, reference_table=None):
    """
    Convert a Z3 model into table form.

    If reference_table is supplied, payload category names are mapped to the
    reference table's column names by semantic value domains. This handles
    cases such as FavoriteSport -> Sport while normalize_value() separately
    handles surface differences such as high_school -> high school.
    """

    categories = list(context.attribute_values.keys())

    if reference_table is not None:
        category_header_map = get_category_to_reference_header(
            context.attribute_values,
            reference_table,
        )
    else:
        category_header_map = {
            category: category
            for category in categories
        }

    header = ["House"] + [
        category_header_map[category]
        for category in categories
    ]

    rows = []

    # First materialize the position of every value.
    positions = {}

    for category, values in context.attribute_values.items():
        for value in values:
            variable = context.variables[(category, str(value))]
            evaluated = model.eval(variable, model_completion=True)

            try:
                positions[(category, str(value))] = evaluated.as_long()
            except Exception:
                positions[(category, str(value))] = None

    for house in range(1, context.n_houses + 1):
        row = [str(house)]

        for category in categories:
            matches = [
                str(value)
                for value in context.attribute_values[category]
                if positions.get((category, str(value))) == house
            ]

            if len(matches) == 1:
                row.append(matches[0])
            elif len(matches) == 0:
                row.append("<MISSING>")
            else:
                row.append("<AMBIGUOUS:" + "|".join(matches) + ">")

        rows.append(row)

    return {
        "header": header,
        "rows": rows,
    }


def build_fresh_solver(payload, extra_step=None):
    """
    Build a NEW solver from scratch.

    Base constraints:
        - each attribute value is in 1..n_houses
        - values inside each category are Distinct
        - all payload syntactic clues C_i

    If extra_step is provided, only that S_i is additionally inserted.
    Previous S-steps are intentionally NOT accumulated. This isolates the
    effect of the individual S_i as requested.
    """

    context = Z3PuzzleContext(payload)
    solver = Solver()
    solver.add(*context.base_constraints)

    parse_errors = []

    syntactic_clues = payload.get("syntactic_clues", []) or []

    for clue in syntactic_clues:
        try:
            solver.add(parse_symbolic_expression(clue, context))
        except Exception as e:
            parse_errors.append(
                {
                    "source": "clue",
                    "text": clue,
                    "error": str(e),
                }
            )

    if extra_step is not None:
        try:
            solver.add(parse_symbolic_expression(extra_step, context))
        except Exception as e:
            parse_errors.append(
                {
                    "source": "step",
                    "text": extra_step,
                    "error": str(e),
                }
            )

    return context, solver, parse_errors


def run_fresh_z3_check(
    payload,
    ground_truth,
    processed_prediction,
    extra_step=None,
):
    """
    Freshly rebuild and solve one constraint set.

    For base check:
        syntactic clues only

    For S_i check:
        syntactic clues + that single S_i

    Returned fields explicitly include:
        SAT
        Z3 model == GT
        Z3 model == processed prediction
    """

    result = {
        "sat": None,
        "solution": None,
        "solution_matches_gt": None,
        "solution_matches_prediction": None,
        "gt_status": "UNVERIFIED",
        "prediction_status": "UNVERIFIED",
        "parse_errors": [],
    }

    try:
        context, solver, parse_errors = build_fresh_solver(
            payload,
            extra_step=extra_step,
        )
    except Exception as e:
        result["parse_errors"] = [
            {
                "source": "solver_build",
                "text": extra_step,
                "error": str(e),
            }
        ]
        return result

    result["parse_errors"] = parse_errors

    # If a clue or the requested S_i could not be parsed, do not silently
    # solve a weakened constraint set and pretend the result is authoritative.
    if parse_errors:
        return result

    check_result = solver.check()

    if check_result != sat:
        result["sat"] = False
        result["gt_status"] = "GT-INCONSISTENT"

        # If there is a valid prediction table, UNSAT also means there is no
        # model equal to that prediction under this constraint set.
        if isinstance(processed_prediction, dict):
            result["prediction_status"] = "PREDICTION-INCONSISTENT"

        return result

    result["sat"] = True

    model = solver.model()
    # Render the fresh Z3 model using GT-compatible category headers before
    # comparing it with GT and processed_prediction.
    solution_table = model_to_table(
        context,
        model,
        reference_table=ground_truth,
    )
    result["solution"] = solution_table

    gt_match = tables_match(solution_table, ground_truth)
    prediction_match = tables_match(solution_table, processed_prediction)

    result["solution_matches_gt"] = gt_match
    result["solution_matches_prediction"] = prediction_match

    if gt_match is True:
        result["gt_status"] = "GT-CONSISTENT"
    elif gt_match is False:
        result["gt_status"] = "GT-INCONSISTENT"

    if prediction_match is True:
        result["prediction_status"] = "PREDICTION-CONSISTENT"
    elif prediction_match is False:
        result["prediction_status"] = "PREDICTION-INCONSISTENT"

    return result


def bool_text(value):
    if value is True:
        return "TRUE"
    if value is False:
        return "FALSE"
    return "UNVERIFIED"


def print_fresh_z3_result(title, result, print_solution=False):
    """Readable formatting for one fresh solver result."""

    print(f"\n{title}")
    print(f"SAT                              : {bool_text(result.get('sat'))}")
    print(
        "Z3 Solution == Ground Truth     : "
        f"{bool_text(result.get('solution_matches_gt'))}"
    )
    print(f"GT Status                        : {result.get('gt_status')}")
    print(
        "Z3 Solution == Prediction       : "
        f"{bool_text(result.get('solution_matches_prediction'))}"
    )
    print(
        "Prediction Status                : "
        f"{result.get('prediction_status')}"
    )

    parse_errors = result.get("parse_errors", []) or []
    if parse_errors:
        print("Parse Errors:")
        for error in parse_errors:
            print(
                f"  - [{error.get('source')}] {error.get('text')}"
                f" -> {error.get('error')}"
            )

    if print_solution and result.get("solution") is not None:
        print("\nFresh Z3 Solution:")
        print_ground_truth(result["solution"])


def run_per_step_z3_analysis(example, payload):
    """
    Run base + per-S_i fresh Z3 reconstruction for one example.
    """

    ground_truth = example.get("ground_truth", {})
    processed_prediction = example.get("processed_prediction")
    reasoning = payload.get("reasoning", []) or []

    print("\n\n===== Fresh Z3 Reconstruction =====")
    print(
        "Base solver = domain/distinct constraints + all syntactic clues C_i"
    )

    base_result = run_fresh_z3_check(
        payload=payload,
        ground_truth=ground_truth,
        processed_prediction=processed_prediction,
        extra_step=None,
    )

    print_fresh_z3_result(
        "\n[BASE: SYNTACTIC CLUES ONLY]",
        base_result,
        print_solution=False,
    )

    symbolic_steps = extract_symbolic_steps(reasoning)

    print("\n\n===== Per-Step Fresh Z3 Checks =====")
    print(
        "Each solver below is rebuilt independently as: "
        "syntactic clues + ONE S_i"
    )

    if not symbolic_steps:
        print("No S_i symbolic reasoning steps found.")
        return {
            "base_result": base_result,
            "step_results": [],
        }

    step_results = []

    gt_consistent_count = 0
    gt_inconsistent_count = 0
    gt_unverified_count = 0

    pred_consistent_count = 0
    pred_inconsistent_count = 0
    pred_unverified_count = 0

    sat_count = 0
    unsat_count = 0
    sat_unverified_count = 0

    for step in symbolic_steps:
        result = run_fresh_z3_check(
            payload=payload,
            ground_truth=ground_truth,
            processed_prediction=processed_prediction,
            extra_step=step["raw"],
        )

        print("\n" + "-" * 100)
        print(
            f"{step['label']} | reasoning item {step['reasoning_index']}"
        )

        nl_description = step.get("nl_description", []) or []
        print("NL Description:")
        if nl_description:
            for nl_line in nl_description:
                print(f"  {nl_line}")
        else:
            print("  NONE")

        print(f"Expression                       : {step['expression']}")
        print(f"SAT                              : {bool_text(result.get('sat'))}")
        print(
            "Z3 Solution == Ground Truth     : "
            f"{bool_text(result.get('solution_matches_gt'))}"
        )
        print(f"GT Step Status                   : {result.get('gt_status')}")
        print(
            "Z3 Solution == Prediction       : "
            f"{bool_text(result.get('solution_matches_prediction'))}"
        )
        print(
            "Prediction Step Status           : "
            f"{result.get('prediction_status')}"
        )

        parse_errors = result.get("parse_errors", []) or []
        if parse_errors:
            print("Parse Errors:")
            for error in parse_errors:
                print(
                    f"  - [{error.get('source')}] {error.get('text')}"
                    f" -> {error.get('error')}"
                )

        step_results.append(
            {
                "label": step["label"],
                "symbolic_index": len(step_results) + 1,
                "reasoning_index": step["reasoning_index"],
                "expression": step["expression"],
                "sat": result.get("sat"),
                "gt_status": result.get("gt_status"),
                "prediction_status": result.get("prediction_status"),
                "solution_matches_gt": result.get("solution_matches_gt"),
                "solution_matches_prediction": result.get("solution_matches_prediction"),
                "parse_errors": parse_errors,
            }
        )

        if result.get("sat") is True:
            sat_count += 1
        elif result.get("sat") is False:
            unsat_count += 1
        else:
            sat_unverified_count += 1

        gt_status = result.get("gt_status")
        if gt_status == "GT-CONSISTENT":
            gt_consistent_count += 1
        elif gt_status == "GT-INCONSISTENT":
            gt_inconsistent_count += 1
        else:
            gt_unverified_count += 1

        pred_status = result.get("prediction_status")
        if pred_status == "PREDICTION-CONSISTENT":
            pred_consistent_count += 1
        elif pred_status == "PREDICTION-INCONSISTENT":
            pred_inconsistent_count += 1
        else:
            pred_unverified_count += 1

    print("\n\n===== Fresh Per-Step Z3 Summary =====")
    print(f"Total S_i steps                 : {len(symbolic_steps)}")
    print(f"SAT                             : {sat_count}")
    print(f"UNSAT                           : {unsat_count}")
    print(f"SAT unverified                  : {sat_unverified_count}")
    print(f"GT-Consistent steps             : {gt_consistent_count}")
    print(f"GT-Inconsistent steps           : {gt_inconsistent_count}")
    print(f"GT-Unverified steps             : {gt_unverified_count}")
    print(f"Prediction-Consistent steps     : {pred_consistent_count}")
    print(f"Prediction-Inconsistent steps   : {pred_inconsistent_count}")
    print(f"Prediction-Unverified steps     : {pred_unverified_count}")

    return {
        "base_result": base_result,
        "step_results": step_results,
    }


def print_example(example, example_number):

    separator(
        f"EXAMPLE {example_number} | PID: {example.get('pid', 'UNKNOWN')}"
    )

    # =========================================================
    # Puzzle
    # =========================================================

    print("\n### PUZZLE TEXT ###\n")
    print(example.get("puzzle_text", ""))

    # =========================================================
    # Payload
    # =========================================================

    payload = example.get("payload", {})

    if not payload:
        print("LLM OUTPUT PARSING ERROR")
        return

    # =========================================================
    # Syntactic clues
    # =========================================================

    print("\nSyntactic clues:")

    syntactic_clues = payload.get("syntactic_clues", [])

    if syntactic_clues:
        for i, clue in enumerate(syntactic_clues, start=1):
            print(f"  [{i}] {clue}")
    else:
        print("  NONE")

    # =========================================================
    # Logged Z3 gate
    # =========================================================

    print("\n\n### Z3 Stats ###\n")
    z3_out = example.get("z3_out", {})
    base_sat_full_gt = z3_out.get("base_sat_full_GT")
    print(f"Logged Z3-SAT && Logged Z3-Solution == GT : {base_sat_full_gt}")

    # Preserve the existing workflow: only perform the detailed analysis
    # for examples whose logged base_sat_full_GT is True.
    if not base_sat_full_gt:
        return

    # =========================================================
    # NEW: fresh Z3 base + per-S_i reconstruction
    # =========================================================

    fresh_analysis = run_per_step_z3_analysis(
        example=example,
        payload=payload,
    )

    # ---------------------------------------------------------
    # Reasoning
    # ---------------------------------------------------------

    print("\nReasoning:")

    reasoning = payload.get("reasoning", [])

    if reasoning:
        for i, step in enumerate(reasoning, start=1):
            print(f"  [{i}] {step}")
    else:
        print("  NONE")

    print("\n===== Logged Reasoning Statistics =====")
    print(f"Total steps      : {z3_out.get('n_steps_total', 0)}")
    print(f"Parsed steps     : {z3_out.get('n_steps_parsed_ok', 0)}")
    print(f"Valid steps      : {z3_out.get('n_steps_valid', 0)}")
    print(f"Novel steps      : {z3_out.get('n_steps_novel_inc_clues', 0)}")

    print("\n===== Logged Novel Steps =====")
    novel_steps = z3_out.get("list_novel_steps_inc_clues", [])

    if novel_steps:
        for i, step in enumerate(novel_steps, 1):
            print(f"{i}. {step}")
    else:
        print("No novel steps found.")

    # =========================================================
    # Prediction -- use processed_prediction as requested
    # =========================================================

    print("\n\n### PREDICTION (processed_prediction) ###\n")

    processed = example.get("processed_prediction")

    if processed is None:
        print("NONE")
    elif isinstance(processed, dict):
        print_ground_truth(processed)
    else:
        print(processed)

    # =========================================================
    # Ground Truth
    # =========================================================

    print("\n\n### GROUND TRUTH ###\n")
    print_ground_truth(example.get("ground_truth", {}))

    # =========================================================
    # Basic Result
    # =========================================================

    print("\n\n### BASIC RESULT ###\n")
    print(f"Reward       : {example.get('reward')}")
    print(f"Format Check : {example.get('Format_Check')}")

    # =========================================================
    # Final Result
    # =========================================================

    print("\n\n### FINAL RESULT STATS ###\n")

    final_result = example.get("final_result", {})

    if final_result:
        for key, value in final_result.items():
            print(f"{key:35s}: {value}")
    else:
        print("EMPTY")

    separator(char="-")

    return fresh_analysis


def _step_number(step_result):
    """Return the numeric S_i index (e.g., S7 -> 7), with a safe fallback."""
    label = str(step_result.get("label", ""))
    match = re.fullmatch(r"S(\d+)", label, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return int(step_result.get("symbolic_index", 0) or 0)


def _print_dataset_z3_stats(stats):
    """
    Print aggregate case-level reasoning/answer consistency statistics.

    The A/B/C/D categories below are MUTUALLY EXCLUSIVE and are defined only
    over cases for which the logged base_sat_full_GT == True.

    Classification priority:
        A. At least one S_i is GT-INCONSISTENT.
        B. No GT-INCONSISTENT/GT-UNVERIFIED S_i; all S_i are GT-CONSISTENT,
           and at least one S_i is PREDICTION-INCONSISTENT.
        C. All S_i are GT-CONSISTENT and all S_i are PREDICTION-CONSISTENT.
        D. Remaining unresolved cases, e.g. no S_i, any GT-UNVERIFIED S_i,
           or prediction relation cannot be fully evaluated.
    """

    print("\n")
    separator("FRESH Z3 DATASET-LEVEL REASONING STATISTICS")

    base_true = stats["base_sat_full_gt_true"]
    first_fail_cases = stats["first_failing_reasoning_step_cases"]

    print(f"Cases with logged base_sat_full_GT == True : {base_true}")
    print(
        "Cases with a first GT-inconsistent S_i       : "
        f"{first_fail_cases}"
    )

    if base_true:
        print(
            "Percentage with first GT-inconsistent S_i    : "
            f"{100.0 * first_fail_cases / base_true:.2f}%"
        )

    # ========================================================
    # First failing reasoning-step position
    # ========================================================

    first_positions = stats["first_failing_step_positions"]

    print("\nFIRST FAILING REASONING STEP POSITION")

    if first_positions:
        print(f"Minimum first failing step : S{min(first_positions)}")
        print(f"Maximum first failing step : S{max(first_positions)}")
        print(
            "Mean first failing step    : "
            f"S{sum(first_positions) / len(first_positions):.2f}"
        )

        print("\nDistribution:")
        for step_number in sorted(stats["first_failing_step_distribution"]):
            count = stats["first_failing_step_distribution"][step_number]
            print(f"  S{step_number:<3d}: {count}")
    else:
        print("No GT-inconsistent S_i was detected.")

    # ========================================================
    # Mutually exclusive A/B/C/D breakdown
    # ========================================================

    a_count = stats["category_A_gt_inconsistent"]
    b_count = stats["category_B_gt_consistent_prediction_contradiction"]
    c_count = stats["category_C_gt_consistent_prediction_preserved"]
    d_count = stats["category_D_unresolved"]

    print("\n" + "=" * 80)
    print("BREAKDOWN OF base_sat_full_GT = TRUE CASES")
    print("=" * 80)
    print()
    print(f"Total cases                                             : {base_true}")
    print()
    print(
        "A. ANY GT-INCONSISTENT REASONING STEP                   : "
        f"{a_count}"
    )
    print("   └── At least one S_i is inconsistent with Ground Truth")
    print("   └── First failing S_i is recorded above")
    print()
    print("B. ALL REASONING GT-CONSISTENT,")
    print(
        "   BUT PREDICTION CONTRADICTS REASONING                 : "
        f"{b_count}"
    )
    print("   └── Every S_i is consistent with Ground Truth")
    print("   └── At least one S_i is inconsistent with prediction")
    print()
    print("C. ALL REASONING GT-CONSISTENT,")
    print(
        "   AND PREDICTION PRESERVES REASONING                   : "
        f"{c_count}"
    )
    print("   └── Every S_i is consistent with Ground Truth")
    print("   └── Every S_i is also consistent with prediction")
    print()
    print(
        "D. UNRESOLVED / UNVERIFIED REASONING                    : "
        f"{d_count}"
    )
    print("   └── No symbolic S_i, or one/more S_i cannot be reliably evaluated")
    print()
    print("-" * 80)

    classified_total = a_count + b_count + c_count + d_count
    print(
        "CHECK: A + B + C + D                                    : "
        f"{classified_total} / {base_true}"
    )

    # Helpful detail for D without introducing overlapping headline metrics.
    print("\nUNRESOLVED DETAILS")
    print(
        "Cases with no symbolic S_i steps                         : "
        f"{stats['no_symbolic_step_cases']}"
    )
    print(
        "Cases containing any GT-UNVERIFIED S_i                   : "
        f"{stats['gt_unverified_step_cases']}"
    )
    print(
        "Cases with GT-consistent S_i but prediction UNVERIFIED   : "
        f"{stats['prediction_unverified_only_cases']}"
    )

def read_jsonl(
    filename,
    start=1,
    limit=None,
    mode="all"
):

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {filename}"
        )

    printed = 0
    total_examples = 0

    dataset_stats = {
        "base_sat_full_gt_true": 0,

        # First GT-inconsistent reasoning-step statistics.
        "first_failing_reasoning_step_cases": 0,
        "first_failing_step_positions": [],
        "first_failing_step_distribution": defaultdict(int),

        # Mutually exclusive A/B/C/D case-level categories.
        "category_A_gt_inconsistent": 0,
        "category_B_gt_consistent_prediction_contradiction": 0,
        "category_C_gt_consistent_prediction_preserved": 0,
        "category_D_unresolved": 0,

        # Detail for unresolved category D.
        "gt_unverified_step_cases": 0,
        "prediction_unverified_only_cases": 0,
        "no_symbolic_step_cases": 0,
    }

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if line_number < start:
                continue

            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)

            except json.JSONDecodeError as e:
                print(
                    f"ERROR parsing line {line_number}: {e}"
                )
                continue

            total_examples += 1

            # =================================================
            # Get Puzzle Accuracy
            # =================================================

            final_result = example.get(
                "final_result",
                {}
            )

            puzzle_accuracy = final_result.get(
                "PUZZLE_ACCURACY"
            )

            # =================================================
            # FILTER MODE
            # =================================================

            if mode == "errors":
                if puzzle_accuracy != 0.0:
                    continue

            elif mode == "correct":
                if puzzle_accuracy != 1.0:
                    continue

            elif mode == "all":
                pass

            else:
                raise ValueError(
                    f"Unknown mode: {mode}. "
                    f"Use 'all', 'errors', or 'correct'."
                )

            # =================================================
            # Print Example
            # =================================================

            logged_base_sat_full_gt = (
                example.get("z3_out", {}) or {}
            ).get("base_sat_full_GT")

            if logged_base_sat_full_gt is True:
                dataset_stats["base_sat_full_gt_true"] += 1

            fresh_analysis = print_example(
                example,
                line_number
            )

            # =================================================
            # Aggregate fresh per-step Z3 statistics.
            #
            # IMPORTANT: A/B/C/D are mutually exclusive and only
            # apply to cases with logged base_sat_full_GT == True.
            # =================================================

            if logged_base_sat_full_gt is True:

                # If detailed fresh analysis could not be produced, classify
                # the case as unresolved so every base_sat_full_GT=True case
                # belongs to exactly one of A/B/C/D.
                if fresh_analysis is None:
                    dataset_stats["category_D_unresolved"] += 1

                else:
                    step_results = fresh_analysis.get("step_results", []) or []

                    if not step_results:
                        dataset_stats["no_symbolic_step_cases"] += 1
                        dataset_stats["category_D_unresolved"] += 1

                    else:
                        gt_statuses = [
                            step.get("gt_status")
                            for step in step_results
                        ]

                        pred_statuses = [
                            step.get("prediction_status")
                            for step in step_results
                        ]

                        # -----------------------------------------
                        # First S_i for which clues + S_i is
                        # GT-INCONSISTENT.
                        # -----------------------------------------

                        first_failing_step = next(
                            (
                                step
                                for step in step_results
                                if step.get("gt_status") == "GT-INCONSISTENT"
                            ),
                            None,
                        )

                        if first_failing_step is not None:
                            dataset_stats[
                                "first_failing_reasoning_step_cases"
                            ] += 1

                            step_number = _step_number(first_failing_step)
                            dataset_stats[
                                "first_failing_step_positions"
                            ].append(step_number)
                            dataset_stats[
                                "first_failing_step_distribution"
                            ][step_number] += 1

                        # -----------------------------------------
                        # A/B/C/D MUTUALLY EXCLUSIVE CLASSIFICATION
                        # -----------------------------------------
                        # Priority is intentional:
                        #
                        # A: Any definite GT-inconsistent S_i.
                        # D: Otherwise, if GT/prediction relation is not fully
                        #    evaluable, keep it unresolved.
                        # B: All S_i GT-consistent, with >=1 prediction conflict.
                        # C: All S_i GT-consistent and prediction-consistent.
                        # -----------------------------------------

                        has_gt_inconsistent = (
                            "GT-INCONSISTENT" in gt_statuses
                        )
                        has_gt_unverified = (
                            "UNVERIFIED" in gt_statuses
                        )
                        has_pred_inconsistent = (
                            "PREDICTION-INCONSISTENT" in pred_statuses
                        )
                        has_pred_unverified = (
                            "UNVERIFIED" in pred_statuses
                        )
                        all_gt_consistent = all(
                            status == "GT-CONSISTENT"
                            for status in gt_statuses
                        )
                        all_pred_consistent = all(
                            status == "PREDICTION-CONSISTENT"
                            for status in pred_statuses
                        )

                        if has_gt_inconsistent:
                            # A. At least one reasoning step is definitely
                            # inconsistent with GT.
                            dataset_stats[
                                "category_A_gt_inconsistent"
                            ] += 1

                        elif has_gt_unverified:
                            # D. No definite GT failure, but at least one step
                            # cannot be evaluated against GT.
                            dataset_stats[
                                "category_D_unresolved"
                            ] += 1
                            dataset_stats[
                                "gt_unverified_step_cases"
                            ] += 1

                        elif all_gt_consistent and has_pred_inconsistent:
                            # B. Reasoning remains GT-consistent, but the
                            # prediction contradicts at least one reasoning step.
                            dataset_stats[
                                "category_B_gt_consistent_prediction_contradiction"
                            ] += 1

                        elif all_gt_consistent and all_pred_consistent:
                            # C. Reasoning is GT-consistent and the prediction
                            # preserves every reasoning step.
                            dataset_stats[
                                "category_C_gt_consistent_prediction_preserved"
                            ] += 1

                        else:
                            # D. Typically all reasoning is GT-consistent but
                            # prediction consistency cannot be fully evaluated.
                            dataset_stats[
                                "category_D_unresolved"
                            ] += 1

                            if all_gt_consistent and has_pred_unverified:
                                dataset_stats[
                                    "prediction_unverified_only_cases"
                                ] += 1

            printed += 1

            print("\n" * EXAMPLE_GAP)

            if (
                limit is not None
                and printed >= limit
            ):
                break

    # =========================================================
    # Summary
    # =========================================================

    print("\n")
    separator("SUMMARY")

    print(f"Mode             : {mode}")
    print(f"Examples scanned : {total_examples}")
    print(f"Examples printed : {printed}")

    _print_dataset_z3_stats(dataset_stats)


def analyze_file(
    filename,
    start=1,
    limit=None,
    mode="all"
):

    input_path = Path(filename)
    output_folder = Path("./Outputs")
    output_folder.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Output filename based on selected mode
    # =========================================================

    output_filename = (
        output_folder
        / f"{input_path.stem}_{mode}_detailed_with_Z3_solver.txt"
    )

    print(f"Input file : {input_path}")
    print(f"Output file: {output_filename}")

    original_stdout = sys.stdout

    try:

        with output_filename.open(
            "w",
            encoding="utf-8"
        ) as output_file:

            sys.stdout = Tee(
                original_stdout,
                output_file
            )

            separator(
                "REASONING DRIFT ANALYSIS"
            )

            print(f"\nSource file : {input_path}")
            print(f"Mode        : {mode}")
            print(f"Start       : {start}")
            print(
                f"Limit       : "
                f"{limit if limit is not None else 'ALL'}"
            )

            print("\n")

            read_jsonl(
                filename,
                start=start,
                limit=limit,
                mode=mode
            )

    finally:

        sys.stdout = original_stdout

    print("\nReadable analysis saved to:")
    print(output_filename.resolve())


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    # =========================================================
    # PATH
    # =========================================================

    base_path = Path("./Input_Logs")

    # =========================================================
    # FILE
    # =========================================================

    file_name = "jobid_276924_epoch_28_valid_feedback.jsonl"

    filename = base_path / file_name

    # =========================================================
    # PRINT MODE
    # =========================================================

    # Options:
    #
    # "all"     -> print every example
    # "errors"  -> only PUZZLE_ACCURACY == 0.0
    # "correct" -> only PUZZLE_ACCURACY == 1.0

    mode = "errors"

    # =========================================================
    # SETTINGS
    # =========================================================

    start = 1
    limit = None

    # =========================================================
    # RUN
    # =========================================================

    analyze_file(
        filename,
        start=start,
        limit=limit,
        mode=mode
    )
