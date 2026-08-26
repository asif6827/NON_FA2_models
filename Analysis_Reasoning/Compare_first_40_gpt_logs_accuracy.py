import json
import re
from pathlib import Path


# ============================================================
# Canonicalization
# ============================================================

def canonicalize(value):
    """
    Canonicalize text before GT/prediction comparison.

    Examples:
        "hip hop"          -> "hip_hop"
        "hip_hop"          -> "hip_hop"
        "Hip-Hop"          -> "hip_hop"
        "science fiction"  -> "science_fiction"
        "blue master"      -> "blue_master"
    """

    value = str(value).strip().lower()

    # Spaces, hyphens, and underscores are equivalent.
    value = re.sub(r"[\s\-_]+", "_", value)

    # Remove remaining punctuation.
    value = re.sub(r"[^a-z0-9_]", "", value)

    # Collapse repeated underscores.
    value = re.sub(r"_+", "_", value)

    return value.strip("_")


# ============================================================
# Outer JSONL parsing
# ============================================================

def parse_outer_record(line):
    """
    Parse one JSONL record.

    Also handles the special first NSS record of the form:
        {"prompt": {...}
    where one final closing brace is missing.
    """

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

    # Optional wrapper used by the first NSS record.
    if (
        isinstance(record, dict)
        and isinstance(record.get("prompt"), dict)
    ):
        record = record["prompt"]

    return record


# ============================================================
# Robust solution extraction from llm_output
# ============================================================

def extract_balanced_json_object(text, marker='"solution"'):
    """
    Extract the JSON object immediately following a marker such as "solution".

    This fallback is useful when the full <answer> JSON is malformed because
    of an error inside the reasoning section, but the final solution table
    itself is still valid JSON.
    """

    if not isinstance(text, str):
        return None

    marker_pos = text.find(marker)

    if marker_pos < 0:
        return None

    colon_pos = text.find(
        ":",
        marker_pos + len(marker)
    )

    if colon_pos < 0:
        return None

    start = text.find(
        "{",
        colon_pos
    )

    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start,
        len(text)
    ):

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

                object_text = text[
                    start:index + 1
                ]

                try:
                    return json.loads(
                        object_text
                    )

                except json.JSONDecodeError:
                    return None

    return None


def extract_solution(record):
    """
    Extract the model's final solution table.

    First tries to parse the complete <answer> JSON.
    If that fails, extracts the "solution" object directly.
    """

    llm_output = record.get(
        "llm_output",
        ""
    )

    if isinstance(llm_output, str):

        match = re.search(
            r"<answer>\s*(\{.*\})\s*</answer>",
            llm_output,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if match is not None:

            try:
                payload = json.loads(
                    match.group(1)
                )

                solution = payload.get(
                    "solution"
                )

                if isinstance(
                    solution,
                    dict
                ):
                    return solution, None

            except json.JSONDecodeError:
                pass

    # --------------------------------------------------------
    # Fallback:
    # recover only the solution JSON object
    # --------------------------------------------------------

    solution = extract_balanced_json_object(
        llm_output,
        marker='"solution"'
    )

    if isinstance(
        solution,
        dict
    ):
        return solution, None

    return None, (
        "Could not extract a valid solution table"
    )


# ============================================================
# Normalize Zebra table by House
# ============================================================

def find_house_column(header):

    for index, column in enumerate(
        header
    ):

        if canonicalize(column) == "house":
            return index

    return None


def normalized_table_by_house(table):
    """
    Convert:

        House | Name | MusicGenre
        1       Alice  hip hop

    into:

        {
            1: {
                "name": "alice",
                "musicgenre": "hip_hop"
            }
        }

    House itself is NOT counted as an accuracy cell.
    """

    if not isinstance(table, dict):
        return None

    header = table.get(
        "header",
        []
    )

    rows = table.get(
        "rows",
        []
    )

    if (
        not isinstance(header, list)
        or not header
    ):
        return None

    if not isinstance(rows, list):
        return None

    house_index = find_house_column(
        header
    )

    if house_index is None:
        return None

    normalized_headers = [
        canonicalize(column)
        for column in header
    ]

    output = {}

    for row in rows:

        if (
            not isinstance(row, list)
            or house_index >= len(row)
        ):
            continue

        try:
            house = int(
                row[house_index]
            )

        except (
            TypeError,
            ValueError
        ):
            continue

        values = {}

        for column_index, column_name in enumerate(
            normalized_headers
        ):

            if column_index == house_index:
                continue

            if column_index >= len(row):
                values[column_name] = None
            else:
                values[column_name] = canonicalize(
                    row[column_index]
                )

        output[house] = values

    if not output:
        return None

    attribute_columns = [
        normalized_headers[i]
        for i in range(len(normalized_headers))
        if i != house_index
    ]

    return output, attribute_columns


# ============================================================
# Per-case Puzzle / Cell Accuracy
# ============================================================

def compute_case_accuracy(
    ground_truth,
    prediction
):
    """
    Puzzle Accuracy:
        1.0 only if the complete normalized prediction matches GT.

    Cell Accuracy:
        number of correct non-House cells / total GT non-House cells.

    If prediction cannot be parsed:
        Puzzle Accuracy = 0.0
        Cell Accuracy   = 0.0
    """

    gt_normalized = (
        normalized_table_by_house(
            ground_truth
        )
    )

    if gt_normalized is None:
        raise ValueError(
            "Ground-truth table is invalid"
        )

    (
        gt_by_house,
        gt_columns
    ) = gt_normalized

    total_cells = sum(
        len(values)
        for values in gt_by_house.values()
    )

    pred_normalized = (
        normalized_table_by_house(
            prediction
        )
        if prediction is not None
        else None
    )

    if pred_normalized is None:

        return {
            "puzzle_accuracy": 0.0,
            "cell_accuracy": 0.0,
            "correct_cells": 0,
            "total_cells": total_cells,
            "mismatches": total_cells,
        }

    (
        pred_by_house,
        pred_columns
    ) = pred_normalized

    correct_cells = 0

    for house, gt_values in (
        gt_by_house.items()
    ):

        pred_values = pred_by_house.get(
            house,
            {}
        )

        for column, gt_value in (
            gt_values.items()
        ):

            if (
                pred_values.get(column)
                == gt_value
            ):
                correct_cells += 1

    cell_accuracy = (
        correct_cells / total_cells
        if total_cells
        else 0.0
    )

    # Exact puzzle match requires:
    #   - every GT cell correct
    #   - same houses
    #   - same normalized attribute columns
    exact_match = (
        correct_cells == total_cells
        and set(pred_by_house.keys())
        == set(gt_by_house.keys())
        and set(pred_columns)
        == set(gt_columns)
    )

    return {
        "puzzle_accuracy":
            1.0 if exact_match else 0.0,

        "cell_accuracy":
            cell_accuracy,

        "correct_cells":
            correct_cells,

        "total_cells":
            total_cells,

        "mismatches":
            total_cells - correct_cells,
    }


# ============================================================
# Read first N cases
# ============================================================

def read_first_n_records(
    filename,
    n_cases
):

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"File not found: {filename}"
        )

    records = []

    with filename.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = parse_outer_record(
                    line
                )

            except json.JSONDecodeError as e:

                raise ValueError(
                    f"Could not parse outer JSONL "
                    f"record at line {line_number} "
                    f"in {filename}: {e}"
                ) from e

            records.append(
                record
            )

            if len(records) >= n_cases:
                break

    if len(records) < n_cases:

        raise ValueError(
            f"{filename} contains only "
            f"{len(records)} usable records; "
            f"requested {n_cases}."
        )

    return records


# ============================================================
# Analyze one log
# ============================================================

def analyze_log(
    filename,
    n_cases=40
):

    records = read_first_n_records(
        filename,
        n_cases
    )

    cases = []

    total_puzzle_correct = 0

    total_correct_cells = 0
    total_cells = 0

    sum_case_cell_accuracy = 0.0

    solution_parse_failures = 0

    for case_number, record in enumerate(
        records,
        start=1
    ):

        ground_truth = record.get(
            "ground_truth",
            {}
        )

        prediction, parse_error = (
            extract_solution(
                record
            )
        )

        if prediction is None:
            solution_parse_failures += 1

        metrics = compute_case_accuracy(
            ground_truth,
            prediction
        )

        total_puzzle_correct += int(
            metrics["puzzle_accuracy"]
        )

        total_correct_cells += (
            metrics["correct_cells"]
        )

        total_cells += (
            metrics["total_cells"]
        )

        sum_case_cell_accuracy += (
            metrics["cell_accuracy"]
        )

        cases.append(
            {
                "case_number":
                    case_number,

                "id":
                    str(
                        record.get(
                            "id",
                            "UNKNOWN"
                        )
                    ),

                "size":
                    record.get(
                        "size"
                    ),

                "puzzle_accuracy":
                    metrics[
                        "puzzle_accuracy"
                    ],

                "cell_accuracy":
                    metrics[
                        "cell_accuracy"
                    ],

                "correct_cells":
                    metrics[
                        "correct_cells"
                    ],

                "total_cells":
                    metrics[
                        "total_cells"
                    ],

                "solution_parse_error":
                    parse_error,
            }
        )

    puzzle_accuracy = (
        total_puzzle_correct
        / n_cases
    )

    # Macro:
    # every puzzle contributes equally.
    macro_cell_accuracy = (
        sum_case_cell_accuracy
        / n_cases
    )

    # Micro:
    # every individual cell contributes equally.
    micro_cell_accuracy = (
        total_correct_cells
        / total_cells
        if total_cells
        else 0.0
    )

    return {
        "filename":
            str(filename),

        "n_cases":
            n_cases,

        "cases":
            cases,

        "correct_puzzles":
            total_puzzle_correct,

        "puzzle_accuracy":
            puzzle_accuracy,

        "macro_cell_accuracy":
            macro_cell_accuracy,

        "micro_cell_accuracy":
            micro_cell_accuracy,

        "correct_cells":
            total_correct_cells,

        "total_cells":
            total_cells,

        "solution_parse_failures":
            solution_parse_failures,
    }


# ============================================================
# Compare two logs
# ============================================================

def compare_logs(
    file_a,
    file_b,
    n_cases=40,
    output_dir="./Outputs"
):

    result_a = analyze_log(
        file_a,
        n_cases=n_cases
    )

    result_b = analyze_log(
        file_b,
        n_cases=n_cases
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "first_40_accuracy_comparison.txt"
    )

    # ========================================================
    # Check ID alignment
    # ========================================================

    id_mismatches = []

    for case_a, case_b in zip(
        result_a["cases"],
        result_b["cases"]
    ):

        if case_a["id"] != case_b["id"]:

            id_mismatches.append(
                (
                    case_a["case_number"],
                    case_a["id"],
                    case_b["id"],
                )
            )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as output:

        def p(text=""):
            print(text)
            output.write(
                str(text) + "\n"
            )

        p("=" * 120)
        p(
            f"ACCURACY COMPARISON — FIRST "
            f"{n_cases} CASES"
        )
        p("=" * 120)

        p()
        p(f"LOG A: {file_a}")
        p(f"LOG B: {file_b}")

        p()
        p(
            "Normalization examples: "
            '"hip hop" == "hip_hop", '
            '"science fiction" == '
            '"science_fiction"'
        )

        p()

        # ====================================================
        # Per-case comparison
        # ====================================================

        p("=" * 120)
        p("PER-CASE RESULTS")
        p("=" * 120)

        header = (
            f"{'#':>3}  "
            f"{'ID':<24} "
            f"{'A PAcc':>7} "
            f"{'A CAcc':>9} "
            f"{'B PAcc':>7} "
            f"{'B CAcc':>9}"
        )

        p(header)
        p("-" * len(header))

        for case_a, case_b in zip(
            result_a["cases"],
            result_b["cases"]
        ):

            case_id = case_a["id"]

            if case_a["id"] != case_b["id"]:
                case_id = (
                    f"{case_a['id']} != "
                    f"{case_b['id']}"
                )

            p(
                f"{case_a['case_number']:>3}  "
                f"{case_id:<24} "
                f"{case_a['puzzle_accuracy']:>7.3f} "
                f"{case_a['cell_accuracy']:>9.4f} "
                f"{case_b['puzzle_accuracy']:>7.3f} "
                f"{case_b['cell_accuracy']:>9.4f}"
            )

        # ====================================================
        # Aggregate comparison
        # ====================================================

        p()
        p("=" * 120)
        p("AGGREGATE RESULTS")
        p("=" * 120)
        p()

        p(
            f"{'Metric':<38}"
            f"{'LOG A':>18}"
            f"{'LOG B':>18}"
            f"{'B - A':>18}"
        )

        p("-" * 92)

        p(
            f"{'Correct puzzles':<38}"
            f"{result_a['correct_puzzles']:>18}"
            f"{result_b['correct_puzzles']:>18}"
            f"{result_b['correct_puzzles'] - result_a['correct_puzzles']:>18}"
        )

        p(
            f"{'Puzzle Accuracy':<38}"
            f"{result_a['puzzle_accuracy']:>18.4f}"
            f"{result_b['puzzle_accuracy']:>18.4f}"
            f"{result_b['puzzle_accuracy'] - result_a['puzzle_accuracy']:>18.4f}"
        )

        p(
            f"{'Cell Accuracy (macro)':<38}"
            f"{result_a['macro_cell_accuracy']:>18.4f}"
            f"{result_b['macro_cell_accuracy']:>18.4f}"
            f"{result_b['macro_cell_accuracy'] - result_a['macro_cell_accuracy']:>18.4f}"
        )

        p(
            f"{'Cell Accuracy (micro)':<38}"
            f"{result_a['micro_cell_accuracy']:>18.4f}"
            f"{result_b['micro_cell_accuracy']:>18.4f}"
            f"{result_b['micro_cell_accuracy'] - result_a['micro_cell_accuracy']:>18.4f}"
        )

        p(
            f"{'Correct cells / total cells':<38}"
            f"{str(result_a['correct_cells']) + ' / ' + str(result_a['total_cells']):>18}"
            f"{str(result_b['correct_cells']) + ' / ' + str(result_b['total_cells']):>18}"
            f"{'':>18}"
        )

        p(
            f"{'Solution parse failures':<38}"
            f"{result_a['solution_parse_failures']:>18}"
            f"{result_b['solution_parse_failures']:>18}"
            f"{result_b['solution_parse_failures'] - result_a['solution_parse_failures']:>18}"
        )

        p()
        p("=" * 120)
        p("ID ALIGNMENT CHECK")
        p("=" * 120)

        if id_mismatches:

            p(
                f"WARNING: {len(id_mismatches)} "
                f"case positions have different IDs."
            )

            for item in id_mismatches:
                p(
                    f"Case {item[0]}: "
                    f"{item[1]} != {item[2]}"
                )

        else:

            p(
                f"All first {n_cases} cases have "
                f"matching IDs across both logs."
            )

    print()
    print(
        "Comparison saved to:"
    )
    print(
        output_file.resolve()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    base_path = Path(
        "./Input_Logs"
    )

    # ========================================================
    # Change filenames here if needed
    # ========================================================

    file_a_name = (
        "gpt51_outputs_test_50_mlxl_nss.jsonl"
    )

    file_b_name = (
        "gpt51_outputs_test_700.jsonl"
    )

    file_a = (
        base_path
        / file_a_name
    )

    file_b = (
        base_path
        / file_b_name
    )

    # Compare first 40 cases.
    n_cases = 40

    compare_logs(
        file_a=file_a,
        file_b=file_b,
        n_cases=n_cases,
        output_dir="./Outputs"
    )


if __name__ == "__main__":
    main()
