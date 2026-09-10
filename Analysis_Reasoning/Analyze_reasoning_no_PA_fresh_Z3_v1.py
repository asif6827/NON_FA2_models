from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from z3 import Solver, Int, Distinct, And, Or, Not, Abs, Implies, Xor, sat


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("./Input_Logs")
OUTPUT_DIR = Path("./Output")
PUZZLE_DICTIONARY = "pid_to_puzzle_dic.json"

# Can be overridden with:
#   python SCRIPT.py --log YOUR_FILE.jsonl
DEFAULT_LOG_FILE = "gpt51_outputs_test_700_temp_0.jsonl"


# ============================================================
# Basic JSON / payload parsing
# ============================================================

def parse_outer_record(line: str) -> dict:
    """Parse one JSONL line, preserving the legacy {'prompt': {...}} wrapper."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as original_error:
        stripped = line.strip()
        if stripped.startswith('{"prompt": {'):
            try:
                record = json.loads(stripped + "}")
            except json.JSONDecodeError:
                raise original_error
        else:
            raise original_error

    if isinstance(record, dict) and isinstance(record.get("prompt"), dict):
        record = record["prompt"]

    if not isinstance(record, dict):
        raise ValueError("Outer JSONL record is not a JSON object.")

    return record


def extract_balanced_json_object(text: str, marker: str = '"solution"'):
    """Extract the balanced JSON object immediately following a marker."""
    if not isinstance(text, str):
        return None

    marker_pos = text.find(marker)
    if marker_pos < 0:
        return None

    colon_pos = text.find(":", marker_pos + len(marker))
    if colon_pos < 0:
        return None

    start = text.find("{", colon_pos)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                object_text = text[start:index + 1]
                try:
                    return json.loads(object_text)
                except json.JSONDecodeError:
                    return None

    return None


def extract_answer_payload(llm_output: Any):
    """
    Parse the JSON object inside <answer>...</answer>.

    Returns:
        payload, error
    """
    if isinstance(llm_output, dict):
        return llm_output, None

    if not isinstance(llm_output, str):
        return None, "llm_output is not a string"

    match = re.search(
        r"<answer>\s*(\{.*\})\s*</answer>",
        llm_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match is not None:
        try:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict):
                return payload, None
        except json.JSONDecodeError as e:
            answer_error = f"JSON parse error inside <answer>: {e}"
        else:
            answer_error = "Parsed <answer> payload is not a JSON object"
    else:
        answer_error = "No valid <answer>...</answer> JSON block found"

    # Secondary attempt: the whole output may itself be JSON.
    try:
        payload = json.loads(llm_output)
        if isinstance(payload, dict):
            return payload, None
    except Exception:
        pass

    return None, answer_error


def extract_prediction_for_filter(record: dict, payload: dict | None):
    """
    Get final solution for error filtering.

    Prefer parsed full payload. If that fails, use the same balanced 'solution'
    fallback used by the comparison script.
    """
    if isinstance(payload, dict) and isinstance(payload.get("solution"), dict):
        return payload["solution"]

    return extract_balanced_json_object(
        record.get("llm_output", ""),
        marker='"solution"',
    )


# ============================================================
# Canonicalization / table handling
# ============================================================

def canonicalize_token(value: Any) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[\s\-_]+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def make_identifier(value: Any) -> str:
    value = canonicalize_token(value)
    if not value:
        raise ValueError("Empty identifier after canonicalization")
    if value[0].isdigit():
        value = "v_" + value
    return value


def infer_attribute_values_from_gt(ground_truth: dict) -> dict:
    """Fallback when attribute_values is absent from the LLM payload."""
    if not isinstance(ground_truth, dict):
        return {}

    header = ground_truth.get("header", []) or []
    rows = ground_truth.get("rows", []) or []
    if not header or not rows:
        return {}

    out = {}
    for column_index, column in enumerate(header):
        if canonicalize_token(column) == "house":
            continue
        values = []
        for row in rows:
            if isinstance(row, list) and column_index < len(row):
                values.append(row[column_index])
        if values:
            out[str(column)] = values
    return out


def prepare_payload(payload: dict, ground_truth: dict) -> dict:
    """
    Make a shallow prepared copy and fill only structural fields required by Z3
    when they are absent. Syntactic clues and reasoning are NEVER invented.
    """
    prepared = dict(payload)

    if not prepared.get("attribute_values"):
        prepared["attribute_values"] = infer_attribute_values_from_gt(ground_truth)

    if prepared.get("n_houses") in (None, ""):
        rows = ground_truth.get("rows", []) if isinstance(ground_truth, dict) else []
        prepared["n_houses"] = len(rows)

    return prepared


def find_house_column(table: dict):
    if not isinstance(table, dict):
        return None
    for i, column in enumerate(table.get("header", []) or []):
        if canonicalize_token(column) == "house":
            return i
    return None


def map_table_columns_to_categories(table: dict, attribute_values: dict):
    """
    Map table columns to payload categories by:
      1. normalized header equality;
      2. otherwise value-domain overlap.
    """
    if not isinstance(table, dict):
        return None

    header = table.get("header", []) or []
    rows = table.get("rows", []) or []
    house_index = find_house_column(table)

    if not header or not rows or house_index is None:
        return None

    candidate_columns = [i for i in range(len(header)) if i != house_index]
    mapping = {}
    used_columns = set()

    for category in attribute_values:
        normalized_category = canonicalize_token(category)
        direct = [
            i for i in candidate_columns
            if i not in used_columns
            and canonicalize_token(header[i]) == normalized_category
        ]
        if len(direct) == 1:
            mapping[category] = direct[0]
            used_columns.add(direct[0])

    for category, expected_values in attribute_values.items():
        if category in mapping:
            continue

        expected_domain = {canonicalize_token(v) for v in expected_values}
        scored = []

        for column_index in candidate_columns:
            if column_index in used_columns:
                continue

            observed = []
            valid = True
            for row in rows:
                if not isinstance(row, list) or column_index >= len(row):
                    valid = False
                    break
                observed.append(canonicalize_token(row[column_index]))

            if not valid:
                continue

            observed_domain = set(observed)
            overlap = len(expected_domain & observed_domain)
            unexpected = len(observed_domain - expected_domain)
            scored.append((overlap, -unexpected, column_index))

        if not scored:
            return None

        scored.sort(reverse=True)
        if scored[0][0] == 0:
            return None
        if len(scored) > 1 and scored[1][:2] == scored[0][:2]:
            return None

        mapping[category] = scored[0][2]
        used_columns.add(scored[0][2])

    return mapping


def table_signature(table: dict, attribute_values: dict):
    if not isinstance(table, dict):
        return None

    rows = table.get("rows", []) or []
    house_index = find_house_column(table)
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
                canonicalize_token(category),
                canonicalize_token(row[column_index]),
            )
            assignments[key].append(house)

    return {
        key: tuple(sorted(houses))
        for key, houses in assignments.items()
    }


def tables_match(table_a: dict, table_b: dict, attribute_values: dict):
    sig_a = table_signature(table_a, attribute_values)
    sig_b = table_signature(table_b, attribute_values)
    if sig_a is None or sig_b is None:
        return None
    return sig_a == sig_b


def compact_table_text(table: Any) -> str:
    """
    Print tables in the requested compact JSON style:
      "header": [...],
      "rows": [
        [...],
        [...]
      ]
    """
    if not isinstance(table, dict):
        return json.dumps(table, ensure_ascii=False, indent=2)

    header = table.get("header", [])
    rows = table.get("rows", [])

    if not isinstance(header, list) or not isinstance(rows, list):
        return json.dumps(table, ensure_ascii=False, indent=2)

    lines = ["{"]
    lines.append(
        '  "header": '
        + json.dumps(header, ensure_ascii=False)
        + ","
    )
    lines.append('  "rows": [')

    for i, row in enumerate(rows):
        comma = "," if i < len(rows) - 1 else ""
        lines.append(
            "    "
            + json.dumps(row, ensure_ascii=False)
            + comma
        )

    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines)


# ============================================================
# Puzzle dictionary
# ============================================================

def load_puzzle_dictionary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Puzzle dictionary not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Direct pid -> info mapping.
        if not any(k in data for k in ("puzzles", "data", "items")):
            return {str(k): v for k, v in data.items()}

        # Wrapped mapping/list.
        for key in ("puzzles", "data", "items"):
            value = data.get(key)
            if isinstance(value, dict):
                return {str(k): v for k, v in value.items()}
            if isinstance(value, list):
                data = value
                break

    if isinstance(data, list):
        lookup = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            pid = (
                item.get("id")
                or item.get("pid")
                or item.get("puzzle_id")
            )
            if pid is not None:
                lookup[str(pid)] = item
        return lookup

    raise ValueError(
        "Unsupported pid_to_puzzle_dic.json structure. Expected a dict "
        "mapping puzzle IDs to puzzle information, or a list of records."
    )


def puzzle_text_and_clues(entry: Any):
    if entry is None:
        return "", []

    if isinstance(entry, str):
        return entry, []

    if not isinstance(entry, dict):
        return str(entry), []

    puzzle_text = (
        entry.get("puzzle")
        or entry.get("puzzle_text")
        or entry.get("text")
        or entry.get("raw_puzzle")
        or ""
    )

    clues = (
        entry.get("clues")
        or entry.get("natural_clues")
        or entry.get("puzzle_clues")
    )

    if clues is None and isinstance(entry.get("extra_info"), dict):
        clues = entry["extra_info"].get("clues")

    if clues is None:
        clues = []
    elif isinstance(clues, str):
        clues = [clues]
    elif not isinstance(clues, list):
        clues = [str(clues)]

    return str(puzzle_text), clues


# ============================================================
# Z3 expression parser
# ============================================================

def strip_symbolic_label(text: Any) -> str:
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
    Every attribute value is represented by an Int giving its house position.
    Each category is constrained to be a permutation of 1..n_houses.
    """

    def __init__(self, payload: dict):
        self.n_houses = int(payload.get("n_houses"))
        self.attribute_values = payload.get("attribute_values", {}) or {}

        if not self.attribute_values:
            raise ValueError("attribute_values is missing or empty")
        if self.n_houses <= 0:
            raise ValueError("n_houses must be positive")

        self.variables = {}
        self.category_variables = defaultdict(list)
        self.name_candidates = defaultdict(list)
        self.base_constraints = []

        for category, values in self.attribute_values.items():
            category_id = make_identifier(category)

            for value in values:
                value_id = make_identifier(value)
                variable = Int(f"{category_id}__{value_id}")

                self.variables[(str(category), str(value))] = variable
                self.category_variables[str(category)].append(variable)
                self.name_candidates[value_id].append(variable)

                self.base_constraints.append(variable >= 1)
                self.base_constraints.append(variable <= self.n_houses)

            category_vars = self.category_variables[str(category)]
            if len(category_vars) > 1:
                self.base_constraints.append(Distinct(*category_vars))

    def resolve_name(self, name: str):
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


def ast_to_z3(node, context: Z3PuzzleContext):
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


def parse_symbolic_expression(text: Any, context: Z3PuzzleContext):
    expression = strip_symbolic_label(text)
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Python-expression parse error: {e}") from e
    return ast_to_z3(parsed.body, context)


def iter_syntactic_clues(syntactic_clues: Any):
    """Yield clue strings from either list[str] or dict[str,str]."""
    if isinstance(syntactic_clues, dict):
        for key, value in syntactic_clues.items():
            if isinstance(value, str):
                if re.match(r"^\s*C\d+\s*:", value, flags=re.I):
                    yield value
                else:
                    yield f"{key}: {value}"
        return

    if isinstance(syntactic_clues, list):
        for clue in syntactic_clues:
            if isinstance(clue, str) and clue.strip():
                yield clue


def model_to_table(context: Z3PuzzleContext, model):
    categories = list(context.attribute_values.keys())
    header = ["House"] + categories
    rows = []

    positions = {}
    for category, values in context.attribute_values.items():
        for value in values:
            variable = context.variables[(str(category), str(value))]
            evaluated = model.eval(variable, model_completion=True)
            try:
                positions[(str(category), str(value))] = evaluated.as_long()
            except Exception:
                positions[(str(category), str(value))] = None

    for house in range(1, context.n_houses + 1):
        row = [str(house)]

        for category in categories:
            matches = [
                str(value)
                for value in context.attribute_values[category]
                if positions.get((str(category), str(value))) == house
            ]

            if len(matches) == 1:
                row.append(matches[0])
            elif len(matches) == 0:
                row.append("<MISSING>")
            else:
                row.append("<AMBIGUOUS:" + "|".join(matches) + ">")

        rows.append(row)

    return {"header": header, "rows": rows}


def build_fresh_solver(payload: dict, extra_step: str | None = None):
    """
    IMPORTANT: every call creates a NEW solver.

    Base solver:
        structural constraints + all C_i

    Step solver Z_i:
        structural constraints + all C_i + ONE S_i

    Earlier S steps are intentionally NOT accumulated.
    """
    context = Z3PuzzleContext(payload)
    solver = Solver()
    solver.add(*context.base_constraints)

    parse_errors = []

    for clue in iter_syntactic_clues(payload.get("syntactic_clues", [])):
        try:
            solver.add(parse_symbolic_expression(clue, context))
        except Exception as e:
            parse_errors.append({
                "source": "clue",
                "text": clue,
                "error": str(e),
            })

    if extra_step is not None:
        try:
            solver.add(parse_symbolic_expression(extra_step, context))
        except Exception as e:
            parse_errors.append({
                "source": "step",
                "text": extra_step,
                "error": str(e),
            })

    return context, solver, parse_errors


def run_fresh_z3_check(
    payload: dict,
    ground_truth: dict,
    extra_step: str | None = None,
):
    result = {
        "sat": None,
        "solution": None,
        "solution_matches_gt": None,
        "status": "UNVERIFIED",
        "parse_errors": [],
    }

    try:
        context, solver, parse_errors = build_fresh_solver(
            payload,
            extra_step=extra_step,
        )
    except Exception as e:
        result["parse_errors"] = [{
            "source": "solver_build",
            "text": extra_step,
            "error": str(e),
        }]
        return result

    result["parse_errors"] = parse_errors

    # Never solve a weakened constraint set.
    if parse_errors:
        return result

    check_result = solver.check()

    if check_result != sat:
        result["sat"] = False
        result["solution_matches_gt"] = False
        result["status"] = "GT-INCONSISTENT"
        return result

    result["sat"] = True
    solution_table = model_to_table(context, solver.model())
    result["solution"] = solution_table

    gt_match = tables_match(
        solution_table,
        ground_truth,
        context.attribute_values,
    )
    result["solution_matches_gt"] = gt_match

    if gt_match is True:
        result["status"] = "GT-CONSISTENT"
    elif gt_match is False:
        result["status"] = "GT-INCONSISTENT"

    return result


# ============================================================
# Reasoning extraction
# ============================================================

def extract_steps_and_pas(reasoning: Any):
    """
    Supports both:
      A) list-style reasoning:
         ["NL1 ...", "S1: Alice == 2.", ...]

      B) dict-style NL/S/PA reasoning:
         {
           "NL1": "...",
           "S1": "Alice == 2.",
           "PA1": {"header": ..., "rows": ...}
         }
    """
    steps = []
    pas = []
    nl_buffer = []

    if isinstance(reasoning, dict):
        items = list(reasoning.items())

        for order_index, (key, value) in enumerate(items, start=1):
            key_text = str(key)

            if re.fullmatch(r"NL\d+", key_text, flags=re.I):
                if isinstance(value, str):
                    nl_buffer.append(value)
                continue

            if re.fullmatch(r"S\d+", key_text, flags=re.I):
                if isinstance(value, str):
                    expression = strip_symbolic_label(value)
                    steps.append({
                        "label": key_text.upper(),
                        "expression": expression,
                        "raw": f"{key_text}: {expression}",
                        "reasoning_index": order_index,
                        "nl_description": list(nl_buffer),
                    })
                else:
                    steps.append({
                        "label": key_text.upper(),
                        "expression": "",
                        "raw": str(value),
                        "reasoning_index": order_index,
                        "nl_description": list(nl_buffer),
                        "extraction_error": "S_i value is not a string",
                    })
                nl_buffer = []
                continue

            if re.fullmatch(r"PA\d+", key_text, flags=re.I):
                pas.append({
                    "label": key_text.upper(),
                    "table": value,
                    "reasoning_index": order_index,
                })
                continue

        return steps, pas

    if isinstance(reasoning, list):
        for order_index, item in enumerate(reasoning, start=1):
            if isinstance(item, dict):
                # Tolerate [{"PA1": {...}}] or [{"S1": "..."}].
                sub_steps, sub_pas = extract_steps_and_pas(item)
                for step in sub_steps:
                    step["reasoning_index"] = order_index
                    if nl_buffer and not step.get("nl_description"):
                        step["nl_description"] = list(nl_buffer)
                        nl_buffer = []
                    steps.append(step)
                for pa in sub_pas:
                    pa["reasoning_index"] = order_index
                    pas.append(pa)
                continue

            if not isinstance(item, str):
                continue

            text = item.strip()

            match = re.match(
                r"^\s*(S\d+)\s*:\s*(.+?)\s*$",
                text,
                flags=re.I,
            )
            if match:
                steps.append({
                    "label": match.group(1).upper(),
                    "expression": match.group(2).strip().rstrip("."),
                    "raw": text,
                    "reasoning_index": order_index,
                    "nl_description": list(nl_buffer),
                })
                nl_buffer = []
            else:
                nl_buffer.append(text)

        return steps, pas

    return steps, pas


# ============================================================
# Output helpers
# ============================================================

def bool_text(value: Any) -> str:
    if value is True:
        return "TRUE"
    if value is False:
        return "FALSE"
    return "UNVERIFIED"


def print_parse_errors(errors: list[dict]):
    for error in errors or []:
        print(
            f"    [{error.get('source')}] {error.get('text')} "
            f"-> {error.get('error')}"
        )


def record_id(record: dict) -> str:
    return str(
        record.get("id")
        or record.get("puzzle_id")
        or record.get("pid")
        or "UNKNOWN"
    )


def output_path_for(log_path: Path) -> Path:
    return OUTPUT_DIR / f"{log_path.stem}_step-by-step-analyses.txt"


class Tee:
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
# No-PA analysis
# ============================================================

def classify_step_results(step_results: list[dict]) -> str:
    if not step_results:
        return "UNRESOLVED_NO_Si"

    statuses = [x.get("status") for x in step_results]

    if "GT-INCONSISTENT" in statuses:
        return "HAS_Si_NOT_CONSISTENT_WITH_GT"

    if any(status != "GT-CONSISTENT" for status in statuses):
        return "UNRESOLVED_Si"

    return "ALL_Si_CONSISTENT_WITH_GT"


def analyze_log(log_path: Path):
    puzzle_lookup = load_puzzle_dictionary(INPUT_DIR / PUZZLE_DICTIONARY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(log_path)

    stats = Counter()
    category_counts = Counter()
    step_status_counts = Counter()
    first_bad_step_counts = Counter()

    import sys
    original_stdout = sys.stdout

    try:
        with log_path.open("r", encoding="utf-8") as src, \
             output_path.open("w", encoding="utf-8") as out:

            sys.stdout = Tee(original_stdout, out)

            print("=" * 120)
            print("STEP-BY-STEP Z3 REASONING ANALYSIS -- LOG WITHOUT PARTIAL ANSWERS")
            print("=" * 120)
            print(f"Input log        : {log_path}")
            print(f"Puzzle dictionary: {INPUT_DIR / PUZZLE_DICTIONARY}")
            print(f"Output           : {output_path}")
            print()
            print("Workflow:")
            print("  1. Keep only final-answer error cases.")
            print("  2. Build fresh BASE Z3 solver from structural constraints + all syntactic clues.")
            print("  3. Ignore cases unless BASE Z3 Solution == Ground Truth.")
            print("  4. For every S_i, build fresh Z_i = clues + S_i only.")
            print("  5. Compare every Z_i solution with Ground Truth.")
            print()

            for line_number, line in enumerate(src, start=1):
                line = line.strip()
                if not line:
                    continue

                stats["records_seen"] += 1

                try:
                    record = parse_outer_record(line)
                except Exception as e:
                    stats["outer_parse_errors"] += 1
                    print(f"\nLINE {line_number}: OUTER JSON PARSE ERROR: {e}")
                    continue

                pid = record_id(record)
                gt = record.get("ground_truth", {}) or {}
                payload, payload_error = extract_answer_payload(
                    record.get("llm_output")
                )

                prepared_for_filter = (
                    prepare_payload(payload, gt)
                    if isinstance(payload, dict)
                    else {
                        "attribute_values": infer_attribute_values_from_gt(gt),
                        "n_houses": len(gt.get("rows", []) or [])
                    }
                )

                prediction = extract_prediction_for_filter(record, payload)
                attribute_values = (
                    prepared_for_filter.get("attribute_values", {}) or {}
                )

                final_match = tables_match(
                    prediction,
                    gt,
                    attribute_values,
                ) if attribute_values else None

                # Only error cases proceed.
                if final_match is True:
                    stats["final_correct"] += 1
                    continue

                stats["error_cases"] += 1

                print("\n\n" + "=" * 120)
                print(f"ERROR CASE {stats['error_cases']} | LINE={line_number} | PID={pid}")
                print("=" * 120)

                puzzle_entry = puzzle_lookup.get(pid)
                puzzle_text, natural_clues = puzzle_text_and_clues(puzzle_entry)

                print("\n### PUZZLE TEXT ###\n")
                print(puzzle_text if puzzle_text else "<NOT AVAILABLE>")

                print("\n### PUZZLE CLUES ###")
                if natural_clues:
                    for i, clue in enumerate(natural_clues, start=1):
                        print(f"  [{i}] {clue}")
                else:
                    print("  <NOT AVAILABLE IN PUZZLE DICTIONARY>")

                print("\n### GROUND TRUTH ###")
                print(compact_table_text(gt))

                print("\n### MODEL PREDICTION ###")
                print(compact_table_text(prediction))

                print("\nFinal Prediction == Ground Truth : "
                      + bool_text(final_match))

                if payload is None:
                    stats["payload_parse_errors"] += 1
                    print("\n### PAYLOAD PARSE ERROR ###")
                    print(payload_error)
                    print(
                        "\nBASE/STEP ANALYSIS: IGNORED because syntactic clues "
                        "and reasoning payload could not be recovered."
                    )
                    continue

                payload = prepare_payload(payload, gt)
                syntactic_clues = list(
                    iter_syntactic_clues(payload.get("syntactic_clues", []))
                )

                print("\n### SYNTACTIC CLUES USED FOR Z3 ###")
                if syntactic_clues:
                    for i, clue in enumerate(syntactic_clues, start=1):
                        print(f"  [{i}] {clue}")
                else:
                    print("  NONE")

                # ----------------------------------------------------
                # BASE gate
                # ----------------------------------------------------
                base_result = run_fresh_z3_check(
                    payload=payload,
                    ground_truth=gt,
                    extra_step=None,
                )

                print("\n### BASE Z3 GATE: CLUES ONLY ###")
                print(f"SAT                          : {bool_text(base_result['sat'])}")
                print(
                    "Z3 Solution == Ground Truth : "
                    + bool_text(base_result["solution_matches_gt"])
                )
                print(f"Status                       : {base_result['status']}")

                if base_result.get("parse_errors"):
                    print("Parse errors:")
                    print_parse_errors(base_result["parse_errors"])

                if base_result.get("solution") is not None:
                    print("\nBASE Z3 Solution:")
                    print(compact_table_text(base_result["solution"]))

                if base_result.get("solution_matches_gt") is not True:
                    if base_result.get("solution_matches_gt") is False:
                        stats["base_not_gt"] += 1
                    else:
                        stats["base_unverified"] += 1

                    print(
                        "\nCASE IGNORED FOR REASONING ANALYSIS: "
                        "BASE Z3 Solution != Ground Truth (or unverified)."
                    )
                    continue

                stats["base_gt_true"] += 1

                # ----------------------------------------------------
                # Per-S_i fresh checks
                # ----------------------------------------------------
                reasoning = payload.get("reasoning", [])
                steps, _ = extract_steps_and_pas(reasoning)

                print("\n### STEP-BY-STEP ANALYSIS ###")
                print(
                    "Each Z_i is a FRESH solver containing: "
                    "all clues + exactly ONE S_i."
                )

                step_results = []

                if not steps:
                    print("No S_i steps found.")
                else:
                    for step in steps:
                        result = run_fresh_z3_check(
                            payload=payload,
                            ground_truth=gt,
                            extra_step=step["raw"],
                        )

                        step_record = {
                            "label": step["label"],
                            "expression": step["expression"],
                            "status": result["status"],
                            "solution_matches_gt": result["solution_matches_gt"],
                        }
                        step_results.append(step_record)

                        print("\n" + "-" * 100)
                        print(f"{step['label']} | reasoning item {step['reasoning_index']}")

                        if step.get("nl_description"):
                            print("NL:")
                            for nl in step["nl_description"]:
                                print(f"  {nl}")

                        print(f"Expression                    : {step['expression']}")
                        print(f"Z_i SAT                       : {bool_text(result['sat'])}")
                        print(
                            "Z_i Solution == Ground Truth : "
                            + bool_text(result["solution_matches_gt"])
                        )
                        print(f"S_i Status                    : {result['status']}")

                        if result.get("parse_errors"):
                            print("Parse errors:")
                            print_parse_errors(result["parse_errors"])


                # Aggregate step-level results across all cases that passed
                # the BASE-Z3 == GT gate.
                for step_result in step_results:
                    step_status_counts[step_result.get("status", "UNVERIFIED")] += 1

                case_category = classify_step_results(step_results)
                category_counts[case_category] += 1

                print("\n### CASE REASONING CATEGORY ###")
                print(case_category)

                if step_results:
                    first_bad = next(
                        (
                            x for x in step_results
                            if x.get("status") == "GT-INCONSISTENT"
                        ),
                        None,
                    )
                    if first_bad:
                        first_bad_step_counts[first_bad["label"]] += 1
                        print(
                            "First S_i not consistent with GT: "
                            f"{first_bad['label']} ({first_bad['expression']})"
                        )

            # ========================================================
            # Summary
            # ========================================================
            print("\n\n" + "=" * 120)
            print("SUMMARY")
            print("=" * 120)
            print(f"Records seen                                : {stats['records_seen']}")
            print(f"Final-answer correct cases                   : {stats['final_correct']}")
            print(f"Final-answer ERROR cases                     : {stats['error_cases']}")
            print(f"Payload parse errors among error cases       : {stats['payload_parse_errors']}")
            print(f"BASE Z3 Solution == GT                       : {stats['base_gt_true']}")
            print(f"BASE Z3 Solution != GT                       : {stats['base_not_gt']}")
            print(f"BASE Z3 vs GT unverified                     : {stats['base_unverified']}")
            print()
            print("Reasoning categories after BASE-GT gate:")
            for category in (
                "ALL_Si_CONSISTENT_WITH_GT",
                "HAS_Si_NOT_CONSISTENT_WITH_GT",
                "UNRESOLVED_Si",
                "UNRESOLVED_NO_Si",
            ):
                print(f"  {category:<42}: {category_counts[category]}")

            # ========================================================
            # Compact key-results summary
            # ========================================================
            print("\n" + "=" * 120)
            print("KEY RESULTS")
            print("=" * 120)

            error_cases = stats["error_cases"]
            eligible_cases = stats["base_gt_true"]

            all_s_consistent = category_counts["ALL_Si_CONSISTENT_WITH_GT"]
            has_s_inconsistent = category_counts["HAS_Si_NOT_CONSISTENT_WITH_GT"]
            unresolved_s = (
                category_counts["UNRESOLVED_Si"]
                + category_counts["UNRESOLVED_NO_Si"]
            )

            print(f"Final-answer error cases                         : {error_cases}")

            if error_cases:
                print(
                    "Error cases passing BASE-Z3 == GT gate             : "
                    f"{eligible_cases} "
                    f"({100.0 * eligible_cases / error_cases:.2f}%)"
                )
            else:
                print(
                    "Error cases passing BASE-Z3 == GT gate             : "
                    f"{eligible_cases}"
                )

            if eligible_cases:
                print(
                    "All S_i consistent with GT                         : "
                    f"{all_s_consistent} "
                    f"({100.0 * all_s_consistent / eligible_cases:.2f}%)"
                )
                print(
                    "At least one S_i inconsistent with GT              : "
                    f"{has_s_inconsistent} "
                    f"({100.0 * has_s_inconsistent / eligible_cases:.2f}%)"
                )
                print(
                    "Unresolved / no-S_i cases                          : "
                    f"{unresolved_s} "
                    f"({100.0 * unresolved_s / eligible_cases:.2f}%)"
                )
            else:
                print("All S_i consistent with GT                         : 0")
                print("At least one S_i inconsistent with GT              : 0")
                print("Unresolved / no-S_i cases                          : 0")

            total_steps = sum(step_status_counts.values())
            gt_consistent_steps = step_status_counts["GT-CONSISTENT"]
            gt_inconsistent_steps = step_status_counts["GT-INCONSISTENT"]
            gt_unverified_steps = (
                total_steps - gt_consistent_steps - gt_inconsistent_steps
            )

            print("\nStep-level results:")
            print(f"  Total S_i checked                              : {total_steps}")
            if total_steps:
                print(
                    "  GT-consistent S_i                               : "
                    f"{gt_consistent_steps} "
                    f"({100.0 * gt_consistent_steps / total_steps:.2f}%)"
                )
                print(
                    "  GT-inconsistent S_i                             : "
                    f"{gt_inconsistent_steps} "
                    f"({100.0 * gt_inconsistent_steps / total_steps:.2f}%)"
                )
                print(
                    "  Unverified S_i                                  : "
                    f"{gt_unverified_steps} "
                    f"({100.0 * gt_unverified_steps / total_steps:.2f}%)"
                )

            print("\nFirst GT-inconsistent S_i distribution:")
            if first_bad_step_counts:
                def _s_num(label):
                    match = re.search(r"(\d+)", str(label))
                    return int(match.group(1)) if match else 10**9

                for label in sorted(first_bad_step_counts, key=_s_num):
                    print(
                        f"  {label:<8}: "
                        f"{first_bad_step_counts[label]}"
                    )
            else:
                print("  NONE")

            print("\nInterpretation-oriented counts:")
            print(
                "  Final answer wrong despite all verified S_i being GT-consistent : "
                f"{all_s_consistent}"
            )
            print(
                "  Final answer wrong with >=1 GT-inconsistent S_i                : "
                f"{has_s_inconsistent}"
            )

    finally:
        sys.stdout = original_stdout

    print(f"\nSaved: {output_path.resolve()}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Fresh-Z3 step analysis for a log WITHOUT Partial Answers."
    )
    parser.add_argument(
        "--log",
        default=DEFAULT_LOG_FILE,
        help="JSONL filename inside ./Input_Logs (or an explicit path).",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        log_path = INPUT_DIR / args.log

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    analyze_log(log_path)


if __name__ == "__main__":
    main()
