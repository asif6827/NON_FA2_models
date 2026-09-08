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

    # Treat spaces, hyphens, and underscores identically.
    value = re.sub(r"[\s\-_]+", "_", value)

    # Remove remaining punctuation.
    value = re.sub(r"[^a-z0-9_]", "", value)

    # Collapse repeated underscores.
    value = re.sub(r"_+", "_", value)

    return value.strip("_")


# ============================================================
# Pretty-print a Zebra table
# ============================================================

def print_table_to_file(output, table):

    if not isinstance(table, dict):
        output.write(str(table) + "\n")
        return

    header = table.get("header", [])
    rows = table.get("rows", [])

    if not header:
        output.write(
            json.dumps(
                table,
                indent=2,
                ensure_ascii=False
            )
            + "\n"
        )
        return

    output.write(
        " | ".join(str(x) for x in header)
        + "\n"
    )

    output.write("-" * 120 + "\n")

    for row in rows:
        output.write(
            " | ".join(str(x) for x in row)
            + "\n"
        )


# ============================================================
# Parse one outer JSONL line
# ============================================================

def parse_outer_record(line, line_number):
    """
    Handle both normal JSONL records and the optional:
        {"prompt": {...}}
    wrapper.
    """

    try:
        record = json.loads(line)

    except json.JSONDecodeError as original_error:

        stripped = line.strip()

        # Special repair for a record that starts with {"prompt": {
        # and is missing one final closing brace.
        if stripped.startswith('{"prompt": {'):
            try:
                record = json.loads(stripped + "}")
            except json.JSONDecodeError:
                raise original_error
        else:
            raise original_error

    if (
        isinstance(record, dict)
        and isinstance(record.get("prompt"), dict)
    ):
        record = record["prompt"]

    return record


# ============================================================
# Extract JSON inside <answer>...</answer>
# ============================================================

def extract_answer_payload(llm_output):

    if not isinstance(llm_output, str):
        return None, "llm_output is not a string"

    match = re.search(
        r"<answer>\s*(\{.*\})\s*</answer>",
        llm_output,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if match is None:
        return None, "No <answer>...</answer> block found"

    json_text = match.group(1)

    try:
        payload = json.loads(json_text)

    except json.JSONDecodeError as e:
        return None, (
            f"Could not parse JSON inside <answer>: {e}"
        )

    return payload, None


# ============================================================
# Print reasoning
# ============================================================

def print_reasoning(output, reasoning):

    if not reasoning:
        output.write("NONE\n")
        return

    # New GPT log format:
    # {
    #   "NL1": "...",
    #   "S1": "...",
    #   "PA1": {...}
    # }
    if isinstance(reasoning, dict):

        for key, value in reasoning.items():

            output.write(f"\n[{key}]\n")

            if isinstance(value, dict):
                print_table_to_file(output, value)

            elif isinstance(value, list):
                output.write(
                    json.dumps(
                        value,
                        indent=2,
                        ensure_ascii=False
                    )
                    + "\n"
                )

            else:
                output.write(str(value) + "\n")

        return

    # Older reasoning-list format.
    if isinstance(reasoning, list):

        for i, item in enumerate(reasoning, start=1):
            output.write(f"[{i}] {item}\n")

        return

    output.write(str(reasoning) + "\n")


# ============================================================
# GT / Prediction accuracy
# ============================================================

def find_house_column(header):

    for i, column in enumerate(header):
        if canonicalize(column) == "house":
            return i

    return None


def normalized_table_by_house(table):
    """
    Convert table into:

        {
            house_number: {
                normalized_column_name: normalized_value,
                ...
            }
        }

    House is used as the row key and is NOT counted as an
    attribute cell for Cell Accuracy.
    """

    if not isinstance(table, dict):
        return None

    header = table.get("header", [])
    rows = table.get("rows", [])

    if not isinstance(header, list) or not header:
        return None

    if not isinstance(rows, list) or not rows:
        return None

    house_index = find_house_column(header)

    if house_index is None:
        return None

    normalized_headers = [
        canonicalize(column)
        for column in header
    ]

    output = {}

    for row in rows:

        if not isinstance(row, list):
            return None

        if house_index >= len(row):
            return None

        try:
            house = int(row[house_index])
        except (TypeError, ValueError):
            return None

        values = {}

        for column_index, column_name in enumerate(
            normalized_headers
        ):

            if column_index == house_index:
                continue

            if column_index >= len(row):
                return None

            values[column_name] = canonicalize(
                row[column_index]
            )

        output[house] = values

    return output


def compare_gt_prediction(ground_truth, prediction):
    """
    Returns:
        {
            puzzle_accuracy: 0.0 / 1.0 / None,
            cell_accuracy: float / None,
            correct_cells: int,
            total_cells: int,
            mismatches: [...]
        }

    Puzzle Accuracy:
        1.0 only when every non-House cell matches after
        canonicalization.

    Cell Accuracy:
        correctly matched non-House cells / total GT non-House cells.

    Examples treated as equal:
        "hip hop" == "hip_hop"
        "science fiction" == "science_fiction"
        "blue master" == "blue_master"
    """

    gt = normalized_table_by_house(ground_truth)
    pred = normalized_table_by_house(prediction)

    if gt is None or pred is None:
        return {
            "puzzle_accuracy": None,
            "cell_accuracy": None,
            "correct_cells": 0,
            "total_cells": 0,
            "mismatches": [
                {
                    "type": "UNPARSEABLE_GT_OR_PREDICTION_TABLE"
                }
            ],
        }

    correct_cells = 0
    total_cells = 0
    mismatches = []

    # Compare every GT house and every GT attribute.
    for house in sorted(gt):

        gt_values = gt[house]
        pred_values = pred.get(house)

        if pred_values is None:

            for column, gt_value in gt_values.items():
                total_cells += 1

                mismatches.append(
                    {
                        "house": house,
                        "column": column,
                        "ground_truth": gt_value,
                        "prediction": "<MISSING_HOUSE>",
                    }
                )

            continue

        for column, gt_value in gt_values.items():

            total_cells += 1

            pred_value = pred_values.get(
                column,
                "<MISSING_COLUMN>"
            )

            if pred_value == gt_value:
                correct_cells += 1

            else:
                mismatches.append(
                    {
                        "house": house,
                        "column": column,
                        "ground_truth": gt_value,
                        "prediction": pred_value,
                    }
                )

    # Extra houses/columns should prevent exact puzzle accuracy.
    gt_houses = set(gt)
    pred_houses = set(pred)

    if pred_houses != gt_houses:

        for house in sorted(pred_houses - gt_houses):
            mismatches.append(
                {
                    "type": "EXTRA_PREDICTION_HOUSE",
                    "house": house,
                }
            )

    cell_accuracy = (
        correct_cells / total_cells
        if total_cells > 0
        else None
    )

    puzzle_accuracy = (
        1.0
        if len(mismatches) == 0
        else 0.0
    )

    return {
        "puzzle_accuracy": puzzle_accuracy,
        "cell_accuracy": cell_accuracy,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "mismatches": mismatches,
    }


# ============================================================
# Process complete JSONL file
# ============================================================

def read_log(filename):

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"File not found: {filename}"
        )

    output_dir = Path("./Outputs")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"{filename.stem}_readable_with_accuracy.txt"
    )

    # Dataset-level statistics.
    total_examples = 0
    successfully_compared = 0
    correct_puzzles = 0
    total_correct_cells = 0
    total_cells = 0

    with (
        filename.open(
            "r",
            encoding="utf-8"
        ) as source,

        output_file.open(
            "w",
            encoding="utf-8"
        ) as output
    ):

        for line_number, line in enumerate(
            source,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            total_examples += 1

            output.write("=" * 120 + "\n")
            output.write(f"EXAMPLE {line_number}\n")
            output.write("=" * 120 + "\n")

            # =================================================
            # Parse outer record
            # =================================================

            try:
                record = parse_outer_record(
                    line,
                    line_number
                )

            except json.JSONDecodeError as e:

                output.write(
                    "\nJSONL PARSING ERROR\n"
                )

                output.write(
                    f"{e}\n"
                )

                output.write("\n" * 8)
                continue

            # =================================================
            # ID / SIZE / STATUS
            # =================================================

            output.write(
                f"\nID     : {record.get('id')}\n"
            )
            output.write(
                f"SIZE   : {record.get('size')}\n"
            )
            output.write(
                f"STATUS : {record.get('status')}\n"
            )

            # =================================================
            # GROUND TRUTH
            # =================================================

            ground_truth = record.get(
                "ground_truth",
                {}
            )

            output.write(
                "\n\n### GROUND TRUTH ###\n\n"
            )

            print_table_to_file(
                output,
                ground_truth
            )

            # =================================================
            # Parse LLM answer payload first so MODEL SOLUTION
            # can appear before RAW LLM OUTPUT in the readable log.
            # =================================================

            llm_output = record.get(
                "llm_output"
            )

            payload, error = extract_answer_payload(
                llm_output
            )

            if payload is None:

                output.write(
                    "\n\n### MODEL SOLUTION ###\n\n"
                )
                output.write("UNAVAILABLE\n")

                output.write(
                    "\n\n### RAW LLM OUTPUT ###\n\n"
                )
                output.write(
                    str(llm_output) + "\n"
                )

                output.write(
                    "\n\n### LLM OUTPUT PARSING ERROR ###\n\n"
                )
                output.write(
                    str(error) + "\n"
                )

                output.write("\n" * 8)
                continue

            prediction = payload.get(
                "solution",
                {}
            )

            # =================================================
            # MODEL SOLUTION
            # =================================================

            output.write(
                "\n\n### MODEL SOLUTION ###\n\n"
            )

            print_table_to_file(
                output,
                prediction
            )

            # =================================================
            # Accuracy is still computed here, but not printed here,
            # so the requested section ordering remains exact.
            # =================================================

            metrics = compare_gt_prediction(
                ground_truth,
                prediction
            )

            if metrics["puzzle_accuracy"] is not None:

                successfully_compared += 1

                if metrics["puzzle_accuracy"] == 1.0:
                    correct_puzzles += 1

                total_correct_cells += (
                    metrics["correct_cells"]
                )

                total_cells += (
                    metrics["total_cells"]
                )

            # =================================================
            # RAW LLM OUTPUT
            # =================================================

            output.write(
                "\n\n### RAW LLM OUTPUT ###\n\n"
            )

            output.write(
                str(llm_output) + "\n"
            )

            # =================================================
            # NUMBER OF HOUSES
            # =================================================

            output.write(
                "\n\n### NUMBER OF HOUSES ###\n\n"
            )

            output.write(
                str(payload.get("n_houses")) + "\n"
            )

            # =================================================
            # ATTRIBUTE VALUES
            # =================================================

            output.write(
                "\n\n### ATTRIBUTE VALUES ###\n\n"
            )

            output.write(
                json.dumps(
                    payload.get(
                        "attribute_values",
                        {}
                    ),
                    indent=2,
                    ensure_ascii=False
                )
                + "\n"
            )

            # =================================================
            # SYNTACTIC CLUES
            # =================================================

            output.write(
                "\n\n### SYNTACTIC CLUES ###\n\n"
            )

            clues = payload.get(
                "syntactic_clues",
                []
            )

            if clues:

                if isinstance(clues, dict):
                    for key, clue in clues.items():
                        output.write(
                            f"[{key}] {clue}\n"
                        )
                else:
                    for i, clue in enumerate(
                        clues,
                        start=1
                    ):
                        output.write(
                            f"[{i}] {clue}\n"
                        )

            else:
                output.write("NONE\n")

            # =================================================
            # REASONING
            # =================================================

            output.write(
                "\n\n### REASONING ###\n"
            )

            reasoning = payload.get(
                "reasoning",
                {}
            )

            print_reasoning(
                output,
                reasoning
            )

            # =================================================
            # COMPLETE PARSED ANSWER JSON
            # =================================================

            output.write(
                "\n\n### COMPLETE PARSED ANSWER JSON ###\n\n"
            )

            output.write(
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False
                )
                + "\n"
            )

            output.write("\n" * 8)

        # =====================================================
        # Dataset-level accuracy summary
        # =====================================================

        output.write(
            "\n" + "=" * 120 + "\n"
        )
        output.write(
            "DATASET-LEVEL ACCURACY SUMMARY\n"
        )
        output.write(
            "=" * 120 + "\n\n"
        )

        output.write(
            f"Examples read                 : {total_examples}\n"
        )
        output.write(
            f"Examples successfully compared: {successfully_compared}\n"
        )
        output.write(
            f"Correct puzzles               : {correct_puzzles}\n"
        )

        puzzle_accuracy = (
            correct_puzzles / successfully_compared
            if successfully_compared
            else None
        )

        dataset_cell_accuracy = (
            total_correct_cells / total_cells
            if total_cells
            else None
        )

        output.write(
            f"Overall Puzzle Accuracy       : {puzzle_accuracy}\n"
        )
        output.write(
            f"Overall Cell Accuracy         : {dataset_cell_accuracy}\n"
        )
        output.write(
            f"Correct cells / total cells   : {total_correct_cells} / {total_cells}\n"
        )

    print(
        f"\nReadable log saved to:\n"
        f"{output_file.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    base_path = Path(
        "./Input_Logs"
    )

    file_name = (
        "gpt51_outputs_test_50_mlxl_nss.jsonl"
    )

    filename = (
        base_path
        / file_name
    )

    read_log(
        filename
    )


if __name__ == "__main__":
    main()
