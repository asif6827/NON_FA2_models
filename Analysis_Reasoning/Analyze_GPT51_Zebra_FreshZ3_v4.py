import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from z3 import Solver, Int, Distinct, And, Or, Not, Abs, Implies, Xor, sat

# ============================================================
# Original SPRING/Zebra reasoning validator
# ============================================================
#
# Prefer the project import used by our_puzzle_dataset_v6.py.
# Fall back to a local module import if this analysis script is
# executed from the same directory as the validator.
try:
    from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import (
        solve_and_validate_payload as original_solve_and_validate_payload,
    )
    ORIGINAL_VALIDATOR_IMPORT_ERROR = None
except Exception as _project_import_error:
    try:
        from z3_reasoning_validator_v13_gt_solve_v9 import (
            solve_and_validate_payload as original_solve_and_validate_payload,
        )
        ORIGINAL_VALIDATOR_IMPORT_ERROR = None
    except Exception as _local_import_error:
        original_solve_and_validate_payload = None
        ORIGINAL_VALIDATOR_IMPORT_ERROR = (
            f"Project import failed: {_project_import_error}; "
            f"local import failed: {_local_import_error}"
        )


# ============================================================
# Configuration
# ============================================================

EXAMPLE_GAP = 8


# Canonicalization examples (documentation only):
#   "science fiction"  -> "science_fiction"
#   "Science-Fiction"  -> "science_fiction"
#   "science_fiction"  -> "science_fiction"
# These same canonical forms are used for GT, prediction, C_i, and S_i analysis.


class Tee:
    """Write print output to both terminal and a text file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ============================================================
# Pretty printing
# ============================================================


def separator(title=None, char="=", width=120):
    if title:
        print(f"\n{char * width}")
        print(title)
        print(char * width)
    else:
        print(char * width)


def print_dict(data, indent=0):
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


def print_table(table):
    """Pretty-print a Zebra solution table."""

    if not isinstance(table, dict):
        print(table)
        return

    header = table.get("header", []) or []
    rows = table.get("rows", []) or []

    if not header:
        print_dict(table)
        return

    print(" | ".join(str(x) for x in header))
    print("-" * 120)

    for row in rows:
        print(" | ".join(str(x) for x in row))


# ============================================================
# Result-log parsing
# ============================================================


def extract_answer_payload(llm_output):
    """
    Parse the JSON object inside:

        <answer>{ ... }</answer>

    Returns:
        (payload, error)
    """

    if not isinstance(llm_output, str):
        return None, "llm_output is not a string"

    match = re.search(
        r"<answer>\s*(\{.*\})\s*</answer>",
        llm_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match is None:
        return None, "No <answer>...</answer> JSON block found"

    json_text = match.group(1)

    try:
        return json.loads(json_text), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error inside <answer>: {e}"


# ============================================================
# Canonicalization / table matching
# ============================================================


def canonicalize_token(value):
    """
    ONE canonical normalization rule used throughout the whole analyzer.

    This function is used for:
        - Ground-truth comparison
        - Prediction comparison / puzzle accuracy / cell accuracy
        - Table-column/category matching
        - Z3 variable construction
        - Syntactic-clue token resolution
        - Reasoning-step S_i token resolution

    Surface forms such as the following are treated as equivalent:
        Science Fiction
        science fiction
        science_fiction
        science-fiction
        SCIENCE FICTION

    All become:
        science_fiction
    """

    value = str(value).strip().lower()

    # Treat whitespace, hyphens and underscores as the same separator.
    value = re.sub(r"[\s\-_]+", "_", value)

    # Remove remaining punctuation while preserving letters, digits, underscores.
    value = re.sub(r"[^a-z0-9_]", "", value)

    # Collapse repeated separators once more after punctuation removal.
    value = re.sub(r"_+", "_", value)

    return value.strip("_")


def normalize_value(value):
    """
    Backward-compatible wrapper for semantic table/value comparison.

    IMPORTANT: this intentionally delegates to canonicalize_token(), so table
    accuracy and reasoning/Z3 analysis use exactly the same normalization rule.
    """
    return canonicalize_token(value)


def make_identifier(value):
    """
    Convert a category/value into the canonical Python/Z3 identifier used by
    C_i and S_i expressions.

    The semantic part is produced by the SAME canonicalize_token() function
    used by GT/prediction matching. A leading ``v_`` is added only when the
    canonical token begins with a digit, because Python identifiers cannot.
    """

    value = canonicalize_token(value)

    if not value:
        raise ValueError("Empty identifier after canonicalization")

    if value[0].isdigit():
        value = "v_" + value

    return value


def _find_house_column(table):
    if not isinstance(table, dict):
        return None

    header = table.get("header", []) or []

    for index, column in enumerate(header):
        if normalize_value(column) == "house":
            return index

    return None


def map_table_columns_to_categories(table, attribute_values):
    """
    Map columns in GT/prediction to payload categories.

    First use normalized header equality. If names differ (e.g.
    FavoriteSport vs Sport), infer the mapping from category value domains.

    This also works reasonably well for wrong predictions containing duplicate
    or missing values by selecting the unused column with the highest overlap
    with the expected category domain.
    """

    if not isinstance(table, dict):
        return None

    header = table.get("header", []) or []
    rows = table.get("rows", []) or []

    if not header or not rows:
        return None

    house_index = _find_house_column(table)
    if house_index is None:
        return None

    candidate_columns = [
        i for i in range(len(header))
        if i != house_index
    ]

    mapping = {}
    used_columns = set()

    # --------------------------------------------------------
    # Pass 1: direct normalized header-name matching
    # --------------------------------------------------------

    for category in attribute_values:
        normalized_category = normalize_value(category)

        direct = [
            i for i in candidate_columns
            if i not in used_columns
            and normalize_value(header[i]) == normalized_category
        ]

        if len(direct) == 1:
            mapping[category] = direct[0]
            used_columns.add(direct[0])

    # --------------------------------------------------------
    # Pass 2: infer from value-domain overlap
    # --------------------------------------------------------

    for category, expected_values in attribute_values.items():
        if category in mapping:
            continue

        expected_domain = {
            normalize_value(v)
            for v in expected_values
        }

        scored = []

        for column_index in candidate_columns:
            if column_index in used_columns:
                continue

            observed_values = []
            valid_column = True

            for row in rows:
                if not isinstance(row, list) or column_index >= len(row):
                    valid_column = False
                    break
                observed_values.append(normalize_value(row[column_index]))

            if not valid_column:
                continue

            observed_domain = set(observed_values)
            overlap = len(expected_domain & observed_domain)
            unexpected = len(observed_domain - expected_domain)

            # Primary score: expected-domain overlap.
            # Tie-break: fewer unexpected values.
            scored.append((overlap, -unexpected, column_index))

        if not scored:
            return None

        scored.sort(reverse=True)
        best_overlap, best_unexpected, best_column = scored[0]

        if best_overlap == 0:
            return None

        # Do not silently choose between exact ties.
        if len(scored) > 1 and scored[1][:2] == scored[0][:2]:
            return None

        mapping[category] = best_column
        used_columns.add(best_column)

    return mapping


def table_signature(table, attribute_values):
    """
    Canonical semantic signature:
        (category, value) -> tuple(houses)

    Duplicate values in malformed predictions are preserved.
    """

    if not isinstance(table, dict):
        return None

    rows = table.get("rows", []) or []
    house_index = _find_house_column(table)
    category_columns = map_table_columns_to_categories(table, attribute_values)

    if house_index is None or category_columns is None:
        return None

    assignments = defaultdict(list)

    for row in rows:
        if not isinstance(row, list) or house_index >= len(row):
            return None

        try:
            house = int(row[house_index])
        except (TypeError, ValueError):
            return None

        for category, column_index in category_columns.items():
            if column_index >= len(row):
                return None

            key = (
                normalize_value(category),
                normalize_value(row[column_index]),
            )
            assignments[key].append(house)

    return {
        key: tuple(sorted(houses))
        for key, houses in assignments.items()
    }


def tables_match(table_a, table_b, attribute_values):
    signature_a = table_signature(table_a, attribute_values)
    signature_b = table_signature(table_b, attribute_values)

    if signature_a is None or signature_b is None:
        return None

    return signature_a == signature_b


def table_house_assignments(table, attribute_values):
    """Return house -> category -> normalized value for cell-accuracy checks."""

    if not isinstance(table, dict):
        return None

    rows = table.get("rows", []) or []
    house_index = _find_house_column(table)
    category_columns = map_table_columns_to_categories(table, attribute_values)

    if house_index is None or category_columns is None:
        return None

    output = {}

    for row in rows:
        if not isinstance(row, list) or house_index >= len(row):
            return None

        try:
            house = int(row[house_index])
        except (TypeError, ValueError):
            return None

        values = {}

        for category, column_index in category_columns.items():
            if column_index >= len(row):
                return None
            values[normalize_value(category)] = normalize_value(row[column_index])

        output[house] = values

    return output


def compute_prediction_metrics(ground_truth, prediction, attribute_values):
    """Compute puzzle/cell accuracy using the same canonicalization as Z3 reasoning analysis."""

    exact = tables_match(ground_truth, prediction, attribute_values)

    gt_assignments = table_house_assignments(ground_truth, attribute_values)
    pred_assignments = table_house_assignments(prediction, attribute_values)

    cell_accuracy = None

    if gt_assignments is not None and pred_assignments is not None:
        correct = 0
        total = 0

        for house, gt_values in gt_assignments.items():
            pred_values = pred_assignments.get(house, {})

            for category, gt_value in gt_values.items():
                total += 1
                if pred_values.get(category) == gt_value:
                    correct += 1

        if total:
            cell_accuracy = correct / total

    return {
        "puzzle_accuracy": (
            1.0 if exact is True
            else 0.0 if exact is False
            else None
        ),
        "cell_accuracy": cell_accuracy,
    }


# ============================================================
# Fresh Z3 puzzle construction
# ============================================================


def strip_symbolic_label(text):
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
    Represent every attribute value as an Int house position.

    Each category is a permutation of houses 1..n_houses.
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
                z3_name = f"{category_id}__{value_id}"
                variable = Int(z3_name)

                self.variables[(category, str(value))] = variable
                self.category_variables[category].append(variable)

                # Symbolic clues use value names, not category-qualified names.
                # value_id is already canonicalized by make_identifier().
                self.name_candidates[value_id].append(variable)

                self.base_constraints.append(variable >= 1)
                self.base_constraints.append(variable <= self.n_houses)

            category_vars = self.category_variables[category]
            if len(category_vars) > 1:
                self.base_constraints.append(Distinct(*category_vars))

    def resolve_name(self, name):
        # Canonicalize every symbolic token from C_i / S_i before lookup.
        # Example: Science_Fiction -> science_fiction.
        canonical_name = make_identifier(name)

        candidates = self.name_candidates.get(canonical_name, [])

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
    """Convert the restricted symbolic grammar in C_i/S_i to Z3."""

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
            # A == B == C -> And(A == B, B == C)
            left = right

        return comparisons[0] if len(comparisons) == 1 else And(*comparisons)

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
        if function_name == "Abs":
            if len(arguments) != 1:
                raise ValueError("Abs(...) expects exactly one argument")
            return Abs(arguments[0])
        if function_name == "Implies":
            if len(arguments) != 2:
                raise ValueError("Implies(...) expects exactly two arguments")
            return Implies(arguments[0], arguments[1])
        if function_name == "Xor":
            return Xor(*arguments)

        raise ValueError(f"Unsupported function: {function_name}")

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def parse_symbolic_expression(text, context):
    expression = strip_symbolic_label(text)

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Python-expression parse error: {e}") from e

    return ast_to_z3(parsed.body, context)


def extract_symbolic_steps(reasoning):
    """
    Extract S_i steps and attach the NL lines immediately preceding each step.
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
            nl_buffer = []
        else:
            nl_buffer.append(item)

    return symbolic_steps


def model_to_table(context, model):
    """Convert one Z3 model to a table using payload category names."""

    categories = list(context.attribute_values.keys())
    header = ["House"] + categories
    rows = []

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

    Base:
        domain/distinct constraints + all C_i

    Per step:
        domain/distinct constraints + all C_i + ONE S_i

    S-steps are intentionally not accumulated.
    """

    context = Z3PuzzleContext(payload)
    solver = Solver()
    solver.add(*context.base_constraints)

    parse_errors = []

    for clue in payload.get("syntactic_clues", []) or []:
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


def run_fresh_z3_check(payload, ground_truth, processed_prediction, extra_step=None):
    """Rebuild and solve one base or C_i + S_i constraint set."""

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

    # Do not solve a weakened puzzle if any intended constraint failed parsing.
    if parse_errors:
        return result

    check_result = solver.check()

    if check_result != sat:
        result["sat"] = False
        result["gt_status"] = "GT-INCONSISTENT"

        if isinstance(processed_prediction, dict):
            result["prediction_status"] = "PREDICTION-INCONSISTENT"

        return result

    result["sat"] = True

    model = solver.model()
    solution_table = model_to_table(context, model)
    result["solution"] = solution_table

    gt_match = tables_match(
        solution_table,
        ground_truth,
        context.attribute_values,
    )

    prediction_match = tables_match(
        solution_table,
        processed_prediction,
        context.attribute_values,
    )

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



# ============================================================
# Original validator reasoning/novelty statistics
# ============================================================

ORIGINAL_REASONING_STAT_KEYS = (
    "n_steps_total",
    "n_steps_parsed_ok",
    "n_steps_valid",
    "n_steps_novel_inc_clues",
    "n_non_valid_contradiction",
    "list_steps_valid",
    "list_steps_non_valid",
    "list_novel_steps_inc_clues",
)


def run_original_reasoning_validator(payload, ground_truth):
    """
    Run the SAME auxiliary validator used by our_puzzle_dataset_v6.py.

    This deliberately calls solve_and_validate_payload(), rather than
    re-implementing the validity/novelty definitions here.

    Important semantics inherited from the original validator:
      - base_sat_full_GT must succeed before novelty metrics are produced.
      - valid step: entailed by the clue model.
      - novel_inc_clues: valid under clues AND novel w.r.t. prior
        non-contradictory reasoning steps.
      - clue restatements / semantic clue-equivalents / tautologies are
        excluded from novel steps according to the original validator.
    """

    output = {
        "available": False,
        "error": None,
        "base_sat_full_GT": None,
        "parse_status": None,
    }

    for key in ORIGINAL_REASONING_STAT_KEYS:
        output[key] = [] if key.startswith("list_") else 0

    if original_solve_and_validate_payload is None:
        output["error"] = ORIGINAL_VALIDATOR_IMPORT_ERROR
        return output

    try:
        validator_payload = {
            "n_houses": payload.get("n_houses"),
            "attribute_values": payload.get("attribute_values", {}) or {},
            "syntactic_clues": payload.get("syntactic_clues", []) or [],
            "reasoning": payload.get("reasoning", []) or [],
            "ground_truth": ground_truth,
        }

        z3_out = original_solve_and_validate_payload(
            validator_payload,
            timeout_s=5.0,
            conflict_tolerant_clues=False,
        )

        output["available"] = True
        output["base_sat_full_GT"] = z3_out.get("base_sat_full_GT")
        output["parse_status"] = z3_out.get("parse_status")

        for key in ORIGINAL_REASONING_STAT_KEYS:
            default = [] if key.startswith("list_") else 0
            output[key] = z3_out.get(key, default)

        return output

    except Exception as e:
        output["error"] = f"{type(e).__name__}: {e}"
        return output


def print_original_reasoning_validator_stats(stats):
    """Print the requested original-validator metrics at the end of a case."""

    print("\n\n### ORIGINAL Z3 REASONING VALIDATOR STATISTICS ###\n")

    if not stats.get("available", False):
        print("Validator status                 : UNAVAILABLE / FAILED")
        print(f"Error                            : {stats.get('error')}")
        return

    print(
        "base_sat_full_GT                 : "
        f"{stats.get('base_sat_full_GT')}"
    )
    print(
        "parse_status                     : "
        f"{stats.get('parse_status')}"
    )
    print()
    print(
        "n_steps_total                    : "
        f"{stats.get('n_steps_total', 0)}"
    )
    print(
        "n_steps_parsed_ok                : "
        f"{stats.get('n_steps_parsed_ok', 0)}"
    )
    print(
        "n_steps_valid                    : "
        f"{stats.get('n_steps_valid', 0)}"
    )
    print(
        "n_steps_novel_inc_clues          : "
        f"{stats.get('n_steps_novel_inc_clues', 0)}"
    )
    print(
        "n_non_valid_contradiction        : "
        f"{stats.get('n_non_valid_contradiction', 0)}"
    )

    print("\nlist_steps_valid:")
    values = stats.get("list_steps_valid", []) or []
    if values:
        for i, value in enumerate(values, start=1):
            print(f"  [{i}] {value}")
    else:
        print("  NONE")

    print("\nlist_steps_non_valid:")
    values = stats.get("list_steps_non_valid", []) or []
    if values:
        for i, value in enumerate(values, start=1):
            if isinstance(value, dict):
                k = value.get("k")
                expr = value.get("expr")
                status = value.get("validity_status", value.get("status"))
                reason = value.get("reason", value.get("error"))
                print(
                    f"  [{i}] S{k}: {expr} | "
                    f"status={status} | reason={reason}"
                )
            else:
                print(f"  [{i}] {value}")
    else:
        print("  NONE")

    print("\nlist_novel_steps_inc_clues:")
    values = stats.get("list_novel_steps_inc_clues", []) or []
    if values:
        for i, value in enumerate(values, start=1):
            print(f"  [{i}] {value}")
    else:
        print("  NONE")


# ============================================================
# One-example analysis
# ============================================================


def analyze_one_example(result_example, puzzle_text, example_number):
    """Analyze one GPT result record matched to its Parquet puzzle."""

    pid = result_example.get("id", "UNKNOWN")
    ground_truth = result_example.get("ground_truth", {})
    llm_output = result_example.get("llm_output")

    separator(f"EXAMPLE {example_number} | PID: {pid}")

    print("\n### PUZZLE TEXT (FROM TEST PARQUET) ###\n")
    print(puzzle_text)

    payload, payload_error = extract_answer_payload(llm_output)

    if payload is None:
        print("\nLLM OUTPUT PARSING ERROR")
        print(payload_error)

        return {
            "index": result_example.get("index"),
            "id": pid,
            "status": result_example.get("status"),
            "payload_parse_ok": False,
            "payload_parse_error": payload_error,
        }

    processed_prediction = payload.get("solution")
    syntactic_clues = payload.get("syntactic_clues", []) or []
    reasoning = payload.get("reasoning", []) or []
    attribute_values = payload.get("attribute_values", {}) or {}

    # --------------------------------------------------------
    # Print syntactic clues
    # --------------------------------------------------------

    print("\n### SYNTACTIC CLUES ###\n")

    if syntactic_clues:
        for i, clue in enumerate(syntactic_clues, start=1):
            print(f"  [{i}] {clue}")
    else:
        print("  NONE")

    # --------------------------------------------------------
    # Fresh BASE Z3
    # --------------------------------------------------------

    print("\n\n===== FRESH Z3 RECONSTRUCTION =====")
    print("Base solver = domain/distinct constraints + all syntactic clues C_i")

    base_result = run_fresh_z3_check(
        payload=payload,
        ground_truth=ground_truth,
        processed_prediction=processed_prediction,
        extra_step=None,
    )

    print("\n[BASE: SYNTACTIC CLUES ONLY]")
    print(f"SAT                              : {bool_text(base_result['sat'])}")
    print(
        "Z3 Solution == Ground Truth     : "
        f"{bool_text(base_result['solution_matches_gt'])}"
    )
    print(f"GT Status                        : {base_result['gt_status']}")
    print(
        "Z3 Solution == Prediction       : "
        f"{bool_text(base_result['solution_matches_prediction'])}"
    )
    print(
        "Prediction Status                : "
        f"{base_result['prediction_status']}"
    )

    if base_result.get("parse_errors"):
        print("Parse Errors:")
        for error in base_result["parse_errors"]:
            print(
                f"  - [{error.get('source')}] {error.get('text')}"
                f" -> {error.get('error')}"
            )

    # --------------------------------------------------------
    # Fresh Z3 for each S_i
    # --------------------------------------------------------

    symbolic_steps = extract_symbolic_steps(reasoning)
    step_results = []

    print("\n\n===== PER-STEP FRESH Z3 CHECKS =====")
    print("Each solver is rebuilt independently as: C1...Cn + ONE S_i")

    if not symbolic_steps:
        print("No S_i symbolic reasoning steps found.")

    for symbolic_index, step in enumerate(symbolic_steps, start=1):
        step_result = run_fresh_z3_check(
            payload=payload,
            ground_truth=ground_truth,
            processed_prediction=processed_prediction,
            extra_step=step["raw"],
        )

        print("\n" + "-" * 100)
        print(f"{step['label']} | reasoning item {step['reasoning_index']}")

        print("NL Description:")
        if step.get("nl_description"):
            for nl_line in step["nl_description"]:
                print(f"  {nl_line}")
        else:
            print("  NONE")

        print(f"Expression                       : {step['expression']}")
        print(f"SAT                              : {bool_text(step_result['sat'])}")
        print(
            "Z3 Solution == Ground Truth     : "
            f"{bool_text(step_result['solution_matches_gt'])}"
        )
        print(f"GT Step Status                   : {step_result['gt_status']}")
        print(
            "Z3 Solution == Prediction       : "
            f"{bool_text(step_result['solution_matches_prediction'])}"
        )
        print(
            "Prediction Step Status           : "
            f"{step_result['prediction_status']}"
        )

        if step_result.get("parse_errors"):
            print("Parse Errors:")
            for error in step_result["parse_errors"]:
                print(
                    f"  - [{error.get('source')}] {error.get('text')}"
                    f" -> {error.get('error')}"
                )

        step_results.append(
            {
                "label": step["label"],
                "symbolic_index": symbolic_index,
                "reasoning_index": step["reasoning_index"],
                "nl_description": step.get("nl_description", []),
                "expression": step["expression"],
                "sat": step_result.get("sat"),
                "gt_status": step_result.get("gt_status"),
                "prediction_status": step_result.get("prediction_status"),
                "solution_matches_gt": step_result.get("solution_matches_gt"),
                "solution_matches_prediction": step_result.get(
                    "solution_matches_prediction"
                ),
                "parse_errors": step_result.get("parse_errors", []),
            }
        )

    # --------------------------------------------------------
    # Print complete reasoning
    # --------------------------------------------------------

    print("\n\n### REASONING ###\n")

    if reasoning:
        for i, item in enumerate(reasoning, start=1):
            print(f"  [{i}] {item}")
    else:
        print("  NONE")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\n\n### MODEL PREDICTION (solution from llm_output) ###\n")
    print_table(processed_prediction)

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    print("\n\n### GROUND TRUTH ###\n")
    print_table(ground_truth)

    # --------------------------------------------------------
    # Result metrics
    # --------------------------------------------------------

    metrics = compute_prediction_metrics(
        ground_truth,
        processed_prediction,
        attribute_values,
    )

    # --------------------------------------------------------
    # Original validator: validity / novelty metrics
    # --------------------------------------------------------

    original_validator_stats = run_original_reasoning_validator(
        payload=payload,
        ground_truth=ground_truth,
    )

    print("\n\n### RESULTS ###\n")
    print(f"Result-log status       : {result_example.get('status')}")
    print(f"Puzzle Accuracy         : {metrics['puzzle_accuracy']}")
    print(f"Cell Accuracy           : {metrics['cell_accuracy']}")
    print("Reward                  : N/A (not present in this GPT results log)")

    # Requested auxiliary-validator information is printed at the
    # END of each parsed instance.
    print_original_reasoning_validator_stats(
        original_validator_stats
    )

    separator(char="-")

    return {
        "index": result_example.get("index"),
        "id": pid,
        "size": result_example.get("size"),
        "status": result_example.get("status"),
        "payload_parse_ok": True,
        "base_result": {
            "sat": base_result.get("sat"),
            "gt_status": base_result.get("gt_status"),
            "prediction_status": base_result.get("prediction_status"),
            "solution_matches_gt": base_result.get("solution_matches_gt"),
            "solution_matches_prediction": base_result.get(
                "solution_matches_prediction"
            ),
            "parse_errors": base_result.get("parse_errors", []),
        },
        "step_results": step_results,
        "prediction_metrics": metrics,
        "original_reasoning_validator": original_validator_stats,
    }


# ============================================================
# Dataset processing
# ============================================================


def load_puzzle_lookup(parquet_file):
    """Load test Parquet and return id -> puzzle mapping."""

    df = pd.read_parquet(parquet_file)

    required_columns = {"id", "puzzle"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Test Parquet is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    if df["id"].astype(str).duplicated().any():
        duplicate_ids = (
            df.loc[df["id"].astype(str).duplicated(keep=False), "id"]
            .astype(str)
            .tolist()
        )
        raise ValueError(
            "Duplicate IDs found in test Parquet; ID join would be ambiguous: "
            f"{duplicate_ids[:20]}"
        )

    return {
        str(row["id"]): row["puzzle"]
        for _, row in df.iterrows()
    }



def _step_number(step_result):
    """Return numeric S_i position (e.g., S7 -> 7)."""
    label = str(step_result.get("label", ""))
    match = re.fullmatch(r"S(\d+)", label, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return int(step_result.get("symbolic_index", 0) or 0)


def _computed_puzzle_accuracy_for_filter(result_example):
    """
    Compute puzzle accuracy cheaply before printing/full Z3 analysis.

    For malformed/unparseable LLM output, treat the case as an error (0.0),
    matching the practical behavior of the earlier evaluation logs.
    """
    payload, payload_error = extract_answer_payload(result_example.get("llm_output"))

    if payload is None:
        return 0.0

    ground_truth = result_example.get("ground_truth", {})
    prediction = payload.get("solution")
    attribute_values = payload.get("attribute_values", {}) or {}

    if not attribute_values:
        return 0.0

    metrics = compute_prediction_metrics(
        ground_truth,
        prediction,
        attribute_values,
    )

    puzzle_accuracy = metrics.get("puzzle_accuracy")
    return 0.0 if puzzle_accuracy is None else puzzle_accuracy


def _print_dataset_z3_stats(stats):
    """Print fresh-Z3 dataset-level reasoning statistics."""

    print("\n")
    separator("FRESH Z3 DATASET-LEVEL REASONING STATISTICS")

    base_true = stats["base_sat_full_gt_true"]
    first_fail_cases = stats["first_failing_reasoning_step_cases"]

    print(f"Cases with fresh base_sat_full_GT == True  : {base_true}")
    print(
        "Cases with a first GT-inconsistent S_i       : "
        f"{first_fail_cases}"
    )

    if base_true:
        print(
            "Percentage with first GT-inconsistent S_i    : "
            f"{100.0 * first_fail_cases / base_true:.2f}%"
        )

    # --------------------------------------------------------
    # First failing reasoning-step position
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Mutually exclusive A/B/C/D breakdown
    # --------------------------------------------------------

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


def analyze_files(
    parquet_file,
    results_jsonl,
    output_dir,
    start=1,
    limit=None,
    mode="all",
):
    """
    Process GPT result records one-by-one, join each record to its puzzle by ID,
    print readable analysis, save structured JSONL, and compute dataset-level
    fresh-Z3 reasoning statistics.

    mode:
        "errors"  -> computed Puzzle Accuracy == 0.0
        "correct" -> computed Puzzle Accuracy == 1.0
        "all"     -> all result-log examples
    """

    parquet_file = Path(parquet_file)
    results_jsonl = Path(results_jsonl)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    puzzle_lookup = load_puzzle_lookup(parquet_file)

    output_txt = output_dir / f"{results_jsonl.stem}_fresh_z3_{mode}.txt"
    output_jsonl = output_dir / f"{results_jsonl.stem}_fresh_z3_{mode}_analysis.jsonl"

    original_stdout = sys.stdout

    printed = 0
    total_examples = 0
    missing_puzzle = 0
    payload_parse_errors = 0

    dataset_stats = {
        # Here base_sat_full_GT is reconstructed freshly:
        # BASE SAT == True AND fresh Z3 solution == GT.
        "base_sat_full_gt_true": 0,

        "first_failing_reasoning_step_cases": 0,
        "first_failing_step_positions": [],
        "first_failing_step_distribution": defaultdict(int),

        "category_A_gt_inconsistent": 0,
        "category_B_gt_consistent_prediction_contradiction": 0,
        "category_C_gt_consistent_prediction_preserved": 0,
        "category_D_unresolved": 0,

        "gt_unverified_step_cases": 0,
        "prediction_unverified_only_cases": 0,
        "no_symbolic_step_cases": 0,
    }

    try:
        with (
            results_jsonl.open("r", encoding="utf-8") as source,
            output_txt.open("w", encoding="utf-8") as readable,
            output_jsonl.open("w", encoding="utf-8") as structured,
        ):
            sys.stdout = Tee(original_stdout, readable)

            separator("GPT-5.1 ZEBRALOGIC FRESH-Z3 ANALYSIS")
            print(f"\nTest Parquet : {parquet_file}")
            print(f"Results JSONL: {results_jsonl}")
            print(f"Mode         : {mode}")
            print(f"Start        : {start}")
            print(f"Limit        : {limit if limit is not None else 'ALL'}")
            print()

            for line_number, line in enumerate(source, start=1):
                if line_number < start:
                    continue

                line = line.strip()
                if not line:
                    continue

                try:
                    result_example = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"ERROR parsing result-log line {line_number}: {e}")
                    continue

                total_examples += 1

                # -------------------------------------------------
                # Filter by computed Puzzle Accuracy
                # -------------------------------------------------

                puzzle_accuracy = _computed_puzzle_accuracy_for_filter(
                    result_example
                )

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
                        "Use 'all', 'errors', or 'correct'."
                    )

                pid = str(result_example.get("id", ""))

                if pid not in puzzle_lookup:
                    missing_puzzle += 1
                    print("\n" + "!" * 120)
                    print(
                        f"MISSING PUZZLE: result line {line_number}, id={pid!r} "
                        "was not found in the test Parquet"
                    )
                    print("!" * 120)
                    continue

                analysis = analyze_one_example(
                    result_example=result_example,
                    puzzle_text=puzzle_lookup[pid],
                    example_number=line_number,
                )

                structured.write(
                    json.dumps(analysis, ensure_ascii=False) + "\n"
                )
                structured.flush()

                printed += 1

                if not analysis.get("payload_parse_ok", False):
                    payload_parse_errors += 1
                else:
                    base_result = analysis.get("base_result", {}) or {}

                    # -------------------------------------------------
                    # Fresh equivalent of base_sat_full_GT == True:
                    # BASE solver is SAT AND its solution matches GT.
                    # -------------------------------------------------
                    base_sat_full_gt = (
                        base_result.get("sat") is True
                        and base_result.get("solution_matches_gt") is True
                    )

                    if base_sat_full_gt:
                        dataset_stats["base_sat_full_gt_true"] += 1

                        step_results = analysis.get("step_results", []) or []

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
                            # First GT-inconsistent S_i
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
                            # A/B/C/D mutually exclusive classification
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
                                dataset_stats[
                                    "category_A_gt_inconsistent"
                                ] += 1

                            elif has_gt_unverified:
                                dataset_stats[
                                    "category_D_unresolved"
                                ] += 1
                                dataset_stats[
                                    "gt_unverified_step_cases"
                                ] += 1

                            elif all_gt_consistent and has_pred_inconsistent:
                                dataset_stats[
                                    "category_B_gt_consistent_prediction_contradiction"
                                ] += 1

                            elif all_gt_consistent and all_pred_consistent:
                                dataset_stats[
                                    "category_C_gt_consistent_prediction_preserved"
                                ] += 1

                            else:
                                dataset_stats[
                                    "category_D_unresolved"
                                ] += 1

                                if all_gt_consistent and has_pred_unverified:
                                    dataset_stats[
                                        "prediction_unverified_only_cases"
                                    ] += 1

                print("\n" * EXAMPLE_GAP)

                if limit is not None and printed >= limit:
                    break

            # -----------------------------------------------------
            # Summary
            # -----------------------------------------------------

            print("\n")
            separator("SUMMARY")
            print(f"Mode             : {mode}")
            print(f"Examples scanned : {total_examples}")
            print(f"Examples printed : {printed}")

            _print_dataset_z3_stats(dataset_stats)

            print("\nOUTPUT FILES")
            print(f"Readable log        : {output_txt}")
            print(f"Structured analysis : {output_jsonl}")
            print(f"Missing puzzle IDs  : {missing_puzzle}")
            print(f"Payload parse errors: {payload_parse_errors}")

    finally:
        sys.stdout = original_stdout

    print("\nAnalysis complete.")
    print(f"Readable log       : {output_txt.resolve()}")
    print(f"Structured analysis: {output_jsonl.resolve()}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Input files
    # --------------------------------------------------------

    parquet_file = Path("./Input_Logs/mlxl_train_mlxl_test/test/logic_our_zebra_puzzle_new_reward_test_700.parquet")

    results_jsonl = Path("./Input_Logs/gpt51_outputs_test_700.jsonl")

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    output_dir = Path("./Outputs")

    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    # "all"     -> analyze all examples
    # "errors"  -> only computed Puzzle Accuracy == 0.0
    # "correct" -> only computed Puzzle Accuracy == 1.0
    mode = "errors"

    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    start = 1

    # None = process all result-log records.
    # Example: limit = 3
    limit = None

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    analyze_files(
        parquet_file=parquet_file,
        results_jsonl=results_jsonl,
        output_dir=output_dir,
        start=start,
        limit=limit,
        mode=mode,
    )
