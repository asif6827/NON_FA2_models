import json
import re
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_DIR = Path("./Input_Logs")
OUTPUT_DIR = Path("./Outputs")

ORIGINAL_FILE = "gpt51_outputs_test_700_temp_0.jsonl"
NSS_FILE = "gpt51_outputs_test_700_mlxl_nss_temp_0.jsonl"
PUZZLE_INFO_FILE = "pid_to_puzzle_dic.json"

# ZebraLogic difficulty groups supplied in the experiment setup.
DIFFICULTY_SIZES = {
    "Small": {
        (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 2), (3, 3), (4, 2),
    },
    "Medium": {
        (3, 4), (3, 5), (3, 6),
        (4, 3), (4, 4),
        (5, 2), (6, 2),
    },
    "Large": {
        (4, 5), (5, 3), (4, 6), (5, 4), (6, 3),
    },
    "XL": {
        (5, 5), (6, 4), (5, 6), (6, 5), (6, 6),
    },
}


# ============================================================
# Canonicalization / parsing
# ============================================================

def canonicalize(value):
    """Canonicalize text before GT/prediction comparison."""
    value = str(value).strip().lower()
    value = re.sub(r"[\s\-_]+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def parse_outer_record(line):
    """
    Parse one JSONL record.

    Keeps the legacy repair for an old NSS first-record wrapper:
        {"prompt": {...}
    with one missing final brace.
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

    if isinstance(record, dict) and isinstance(record.get("prompt"), dict):
        record = record["prompt"]

    if not isinstance(record, dict):
        raise ValueError("Outer JSONL record is not a JSON object.")

    return record


def extract_balanced_json_object(text, marker='"solution"'):
    """Extract the balanced JSON object immediately following marker."""
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


def extract_solution(record):
    """
    Extract final solution table.

    Returns:
        solution, parse_error, extraction_method

    extraction_method:
        "full_answer_json"
        "solution_fallback"
        "failure"
    """
    llm_output = record.get("llm_output", "")

    if isinstance(llm_output, str):
        match = re.search(
            r"<answer>\s*(\{.*\})\s*</answer>",
            llm_output,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if match is not None:
            try:
                payload = json.loads(match.group(1))
                solution = payload.get("solution")
                if isinstance(solution, dict):
                    return solution, None, "full_answer_json"
            except json.JSONDecodeError:
                pass

    solution = extract_balanced_json_object(
        llm_output,
        marker='"solution"',
    )

    if isinstance(solution, dict):
        return solution, None, "solution_fallback"

    return (
        None,
        "Could not extract a valid solution table",
        "failure",
    )


# ============================================================
# Puzzle size / difficulty
# ============================================================

def parse_size(size):
    """
    Convert values such as:
        "6*2", "6x2", "6×2", [6, 2], (6, 2)
    to:
        (6, 2)
    """
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            return int(size[0]), int(size[1])
        except (TypeError, ValueError):
            return None

    text = str(size).strip().lower()
    match = re.search(r"(\d+)\s*[*x×]\s*(\d+)", text)
    if not match:
        return None

    return int(match.group(1)), int(match.group(2))


def size_label(size):
    parsed = parse_size(size)
    if parsed is None:
        return str(size)
    return f"{parsed[0]}x{parsed[1]}"


def difficulty_from_size(size):
    parsed = parse_size(size)
    if parsed is None:
        return "Unknown"

    for difficulty, size_set in DIFFICULTY_SIZES.items():
        if parsed in size_set:
            return difficulty

    return "Unknown"


# ============================================================
# Strict Zebra table normalization / accuracy
# ============================================================

def find_house_column(header):
    for index, column in enumerate(header):
        if canonicalize(column) == "house":
            return index
    return None


def normalized_table_by_house(table):
    """
    Strict normalization:
      - requires a header and rows
      - requires House column
      - rejects duplicate canonicalized headers
      - rejects malformed row lengths
      - rejects invalid/duplicate House rows
    """
    if not isinstance(table, dict):
        return None

    header = table.get("header", [])
    rows = table.get("rows", [])

    if not isinstance(header, list) or not header:
        return None
    if not isinstance(rows, list) or not rows:
        return None

    normalized_headers = [canonicalize(column) for column in header]

    if (
        any(not h for h in normalized_headers)
        or len(normalized_headers) != len(set(normalized_headers))
    ):
        return None

    house_index = find_house_column(header)
    if house_index is None:
        return None

    output = {}

    for row in rows:
        if not isinstance(row, list):
            return None

        # Strictly require exactly one cell for every header.
        if len(row) != len(header):
            return None

        try:
            house = int(row[house_index])
        except (TypeError, ValueError):
            return None

        if house in output:
            return None

        values = {}
        for column_index, column_name in enumerate(normalized_headers):
            if column_index == house_index:
                continue
            values[column_name] = canonicalize(row[column_index])

        output[house] = values

    if not output:
        return None

    attribute_columns = [
        normalized_headers[i]
        for i in range(len(normalized_headers))
        if i != house_index
    ]

    return output, attribute_columns


def compute_case_accuracy(ground_truth, prediction):
    """
    Puzzle Accuracy:
        1 only for exact normalized GT match.

    Cell Accuracy:
        correct non-House cells / total GT non-House cells.
    """
    gt_normalized = normalized_table_by_house(ground_truth)
    if gt_normalized is None:
        raise ValueError("Ground-truth table is invalid.")

    gt_by_house, gt_columns = gt_normalized
    total_cells = sum(len(values) for values in gt_by_house.values())

    pred_normalized = (
        normalized_table_by_house(prediction)
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

    pred_by_house, pred_columns = pred_normalized

    correct_cells = 0

    for house, gt_values in gt_by_house.items():
        pred_values = pred_by_house.get(house, {})
        for column, gt_value in gt_values.items():
            if pred_values.get(column) == gt_value:
                correct_cells += 1

    cell_accuracy = (
        correct_cells / total_cells
        if total_cells
        else 0.0
    )

    exact_match = (
        correct_cells == total_cells
        and set(pred_by_house.keys()) == set(gt_by_house.keys())
        and set(pred_columns) == set(gt_columns)
    )

    return {
        "puzzle_accuracy": 1.0 if exact_match else 0.0,
        "cell_accuracy": cell_accuracy,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "mismatches": total_cells - correct_cells,
    }


# ============================================================
# Puzzle lookup / context
# ============================================================

def load_puzzle_dictionary(filename):
    """
    Load puzzle_id -> puzzle metadata from pid_to_puzzle_dic.json.

    Supported top-level shapes:
      1) {"puzzle_id": {...}, ...}
      2) {"pid_to_puzzle_dic": {"puzzle_id": {...}, ...}}
      3) [{"id": "puzzle_id", ...}, ...]

    Keys are normalized to strings so they align with log record IDs.
    """
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"Puzzle dictionary not found: {filename}")

    with filename.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if (
        isinstance(data, dict)
        and isinstance(data.get("pid_to_puzzle_dic"), (dict, list))
    ):
        data = data["pid_to_puzzle_dic"]

    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}

    if isinstance(data, list):
        lookup = {}
        for item in data:
            if not isinstance(item, dict):
                continue

            puzzle_id = None
            for key in ("id", "puzzle_id", "pid"):
                if item.get(key) is not None:
                    puzzle_id = item[key]
                    break

            if puzzle_id is not None:
                lookup[str(puzzle_id)] = item

        if lookup:
            return lookup

    raise ValueError(
        f"Unsupported puzzle dictionary format in {filename}. "
        "Expected a dict keyed by puzzle ID or a list of records with id/puzzle_id/pid."
    )


def normalize_clues(value):
    """Normalize clues into a printable list without changing their content."""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def extract_puzzle_context(entry):
    """
    Extract puzzle text and clues from one puzzle-dictionary entry.

    The loader is intentionally tolerant because datasets commonly store
    these fields as puzzle/puzzle_text/text and clues either at the top level
    or inside extra_info.
    """
    if entry is None:
        return None, []

    if isinstance(entry, str):
        return entry, []

    if not isinstance(entry, dict):
        return str(entry), []

    puzzle_text = None
    clues = []

    for key in ("puzzle_text", "puzzle", "text", "question"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            puzzle_text = value
            break
        if isinstance(value, dict):
            nested_text, nested_clues = extract_puzzle_context(value)
            if puzzle_text is None and nested_text:
                puzzle_text = nested_text
            if not clues and nested_clues:
                clues = nested_clues

    if not clues:
        for key in ("clues", "clue_list", "clue"):
            if key in entry:
                clues = normalize_clues(entry.get(key))
                if clues:
                    break

    extra_info = entry.get("extra_info")
    if isinstance(extra_info, dict):
        if not clues:
            for key in ("clues", "clue_list", "clue"):
                if key in extra_info:
                    clues = normalize_clues(extra_info.get(key))
                    if clues:
                        break

        if puzzle_text is None:
            for key in ("puzzle_text", "puzzle", "text", "question"):
                value = extra_info.get(key)
                if isinstance(value, str) and value.strip():
                    puzzle_text = value
                    break

    return puzzle_text, clues


def get_puzzle_context(puzzle_lookup, case_id):
    """Return printable puzzle text + clues for a log puzzle ID."""
    if puzzle_lookup is None:
        return None, []

    entry = puzzle_lookup.get(str(case_id))
    return extract_puzzle_context(entry)


# ============================================================
# Read / analyze logs
# ============================================================

def read_records(filename):
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    records = []

    with filename.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = parse_outer_record(line)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(
                    f"Could not parse outer JSONL record at "
                    f"line {line_number} in {filename}: {e}"
                ) from e

            record = dict(record)
            record["_source_line"] = line_number
            records.append(record)

    return records


def analyze_log(filename, puzzle_lookup=None):
    records = read_records(filename)

    cases = []
    seen_ids = set()

    for case_number, record in enumerate(records, start=1):
        raw_case_id = record.get("id", record.get("puzzle_id", "UNKNOWN"))
        case_id = str(raw_case_id)

        if case_id in seen_ids:
            raise ValueError(
                f"Duplicate puzzle ID {case_id!r} in {filename}."
            )
        seen_ids.add(case_id)

        ground_truth = record.get("ground_truth", {})
        prediction, parse_error, extraction_method = extract_solution(record)
        puzzle_text, clues = get_puzzle_context(puzzle_lookup, case_id)

        metrics = compute_case_accuracy(
            ground_truth,
            prediction,
        )

        cases.append(
            {
                "case_number": case_number,
                "id": case_id,
                "index": record.get("index"),
                "size_raw": record.get("size"),
                "size": size_label(record.get("size")),
                "difficulty": difficulty_from_size(record.get("size")),
                "status": record.get("status"),
                "puzzle_text": puzzle_text,
                "clues": clues,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "llm_output": record.get("llm_output", ""),
                "puzzle_accuracy": metrics["puzzle_accuracy"],
                "cell_accuracy": metrics["cell_accuracy"],
                "correct_cells": metrics["correct_cells"],
                "total_cells": metrics["total_cells"],
                "mismatches": metrics["mismatches"],
                "solution_parse_error": parse_error,
                "extraction_method": extraction_method,
                "source_line": record["_source_line"],
                "raw_record": {
                    k: v for k, v in record.items()
                    if k != "_source_line"
                },
            }
        )

    return {
        "filename": str(filename),
        "n_cases": len(cases),
        "cases": cases,
    }


# ============================================================
# Aggregate helpers
# ============================================================

def summarize_cases(cases):
    n = len(cases)

    correct_puzzles = sum(
        int(case["puzzle_accuracy"])
        for case in cases
    )

    sum_case_cell_accuracy = sum(
        case["cell_accuracy"]
        for case in cases
    )

    correct_cells = sum(
        case["correct_cells"]
        for case in cases
    )

    total_cells = sum(
        case["total_cells"]
        for case in cases
    )

    extraction_methods = Counter(
        case["extraction_method"]
        for case in cases
    )

    return {
        "n": n,
        "correct_puzzles": correct_puzzles,
        "puzzle_accuracy": (
            correct_puzzles / n if n else 0.0
        ),
        "macro_cell_accuracy": (
            sum_case_cell_accuracy / n if n else 0.0
        ),
        "micro_cell_accuracy": (
            correct_cells / total_cells
            if total_cells
            else 0.0
        ),
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "full_answer_json": extraction_methods["full_answer_json"],
        "solution_fallback": extraction_methods["solution_fallback"],
        "solution_failures": extraction_methods["failure"],
    }


def summarize_by_difficulty(cases):
    grouped = defaultdict(list)

    for case in cases:
        grouped[case["difficulty"]].append(case)

    result = {}
    for difficulty in ["Small", "Medium", "Large", "XL", "Unknown"]:
        if grouped[difficulty]:
            result[difficulty] = summarize_cases(grouped[difficulty])

    return result


def summarize_by_exact_size(cases):
    grouped = defaultdict(list)

    for case in cases:
        grouped[case["size"]].append(case)

    def sort_key(item):
        parsed = parse_size(item)
        return parsed if parsed is not None else (999, 999)

    return {
        size: summarize_cases(grouped[size])
        for size in sorted(grouped, key=sort_key)
    }


# ============================================================
# TXT export
# ============================================================

def pretty_json(obj):
    return json.dumps(
        obj,
        indent=2,
        ensure_ascii=False,
        sort_keys=False,
    )


def compact_table_json(table):
    """
    Print Zebra-style tables with the header on one line and each row on one line.

    Example:
        {
          "header": ["House", "Name", "Birthday"],
          "rows": [
            ["1", "Arnold", "april"],
            ["2", "Eric", "jan"]
          ]
        }

    Falls back to normal pretty JSON for non-table objects.
    """
    if table is None:
        return "null"

    if not isinstance(table, dict):
        return pretty_json(table)

    header = table.get("header")
    rows = table.get("rows")

    if not isinstance(header, list) or not isinstance(rows, list):
        return pretty_json(table)

    lines = [
        "{",
        f'  "header": {json.dumps(header, ensure_ascii=False)},',
        '  "rows": [',
    ]

    for index, row in enumerate(rows):
        suffix = "," if index < len(rows) - 1 else ""
        lines.append(
            f"    {json.dumps(row, ensure_ascii=False)}{suffix}"
        )

    lines.extend([
        "  ]",
        "}",
    ])

    return "\n".join(lines)


def printable_clue(clue):
    """Convert one clue to readable text while preserving structured values."""
    if isinstance(clue, str):
        return clue
    return json.dumps(clue, ensure_ascii=False, sort_keys=False)


def write_puzzle_context(out, case):
    """Write puzzle text and corresponding clues before result information."""
    out.write("PUZZLE TEXT\n")
    out.write("-" * 120 + "\n")

    puzzle_text = case.get("puzzle_text")
    if puzzle_text:
        out.write(str(puzzle_text).rstrip() + "\n")
    else:
        out.write(f"[Puzzle text not found for ID={case.get('id')}]\n")

    out.write("\nCLUES\n")
    out.write("-" * 120 + "\n")

    clues = case.get("clues") or []
    if clues:
        for clue_number, clue in enumerate(clues, start=1):
            out.write(f"{clue_number}. {printable_clue(clue)}\n")
    else:
        out.write("[No clues found in puzzle dictionary]\n")


def write_log_as_txt(result, output_dir):
    """
    Write a human-readable TXT version of one JSONL log.

    Same basename as original:
        X.jsonl -> ./Outputs/X.txt
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(result["filename"])
    output_path = output_dir / f"{source_path.stem}.txt"

    separator = "\n" + "=" * 120 + "\n"

    with output_path.open("w", encoding="utf-8") as out:
        for i, case in enumerate(result["cases"]):
            if i > 0:
                out.write("\n\n\n")

            out.write("=" * 120 + "\n")
            out.write(
                f"PUZZLE {case['case_number']} | "
                f"ID={case['id']} | "
                f"SIZE={case['size']} | "
                f"DIFFICULTY={case['difficulty']}\n"
            )
            out.write("=" * 120 + "\n\n")

            write_puzzle_context(out, case)

            out.write(f"\nStatus: {case['status']}\n")
            out.write(
                f"Puzzle Accuracy: {case['puzzle_accuracy']:.4f}\n"
            )
            out.write(
                f"Cell Accuracy:   {case['cell_accuracy']:.4f} "
                f"({case['correct_cells']}/{case['total_cells']})\n"
            )
            out.write(
                f"Solution extraction: {case['extraction_method']}\n"
            )

            if case["solution_parse_error"]:
                out.write(
                    f"Solution parse error: "
                    f"{case['solution_parse_error']}\n"
                )

            out.write("\nGROUND TRUTH\n")
            out.write("-" * 120 + "\n")
            out.write(compact_table_json(case["ground_truth"]) + "\n")

            out.write("\nEXTRACTED FINAL SOLUTION\n")
            out.write("-" * 120 + "\n")
            out.write(compact_table_json(case["prediction"]) + "\n")

            out.write("\nLLM OUTPUT\n")
            out.write("-" * 120 + "\n")
            out.write(str(case["llm_output"]).rstrip() + "\n")

    return output_path


def write_error_difference_file(
    output_path,
    cases,
    title,
    original_by_id,
    nss_by_id,
):
    """
    Write detailed paired cases for one error-set difference.

    Every entry includes GT + both model outputs.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        out.write("=" * 120 + "\n")
        out.write(title + "\n")
        out.write(f"N CASES: {len(cases)}\n")
        out.write("=" * 120 + "\n")

        for rank, case_id in enumerate(cases, start=1):
            original = original_by_id[case_id]
            nss = nss_by_id[case_id]

            out.write("\n\n\n")
            out.write("=" * 120 + "\n")
            out.write(
                f"DIFFERENCE CASE {rank} | ID={case_id} | "
                f"SIZE={original['size']} | "
                f"DIFFICULTY={original['difficulty']}\n"
            )
            out.write("=" * 120 + "\n\n")

            write_puzzle_context(out, original)

            out.write("\n")
            out.write(
                f"ORIGINAL: PAcc={original['puzzle_accuracy']:.0f}, "
                f"CAcc={original['cell_accuracy']:.4f}, "
                f"Extraction={original['extraction_method']}\n"
            )
            out.write(
                f"NSS:      PAcc={nss['puzzle_accuracy']:.0f}, "
                f"CAcc={nss['cell_accuracy']:.4f}, "
                f"Extraction={nss['extraction_method']}\n"
            )

            out.write("\nGROUND TRUTH\n")
            out.write("-" * 120 + "\n")
            out.write(compact_table_json(original["ground_truth"]) + "\n")

            out.write("\nORIGINAL EXTRACTED SOLUTION\n")
            out.write("-" * 120 + "\n")
            out.write(compact_table_json(original["prediction"]) + "\n")

            out.write("\nNSS EXTRACTED SOLUTION\n")
            out.write("-" * 120 + "\n")
            out.write(compact_table_json(nss["prediction"]) + "\n")

            out.write("\nORIGINAL LLM OUTPUT\n")
            out.write("-" * 120 + "\n")
            out.write(str(original["llm_output"]).rstrip() + "\n")

            out.write("\nNSS LLM OUTPUT\n")
            out.write("-" * 120 + "\n")
            out.write(str(nss["llm_output"]).rstrip() + "\n")


# ============================================================
# Comparison / reports
# ============================================================

def validate_pair_alignment(original_result, nss_result):
    original_by_id = {
        case["id"]: case
        for case in original_result["cases"]
    }
    nss_by_id = {
        case["id"]: case
        for case in nss_result["cases"]
    }

    original_ids = set(original_by_id)
    nss_ids = set(nss_by_id)

    missing_from_nss = sorted(original_ids - nss_ids)
    missing_from_original = sorted(nss_ids - original_ids)

    if missing_from_nss or missing_from_original:
        raise ValueError(
            "The two logs do not contain the same puzzle IDs.\n"
            f"Missing from NSS: {len(missing_from_nss)}\n"
            f"Missing from Original: {len(missing_from_original)}"
        )

    gt_mismatches = []
    size_mismatches = []

    for case_id in sorted(original_ids):
        original = original_by_id[case_id]
        nss = nss_by_id[case_id]

        if original["ground_truth"] != nss["ground_truth"]:
            gt_mismatches.append(case_id)

        if parse_size(original["size_raw"]) != parse_size(nss["size_raw"]):
            size_mismatches.append(case_id)

    if gt_mismatches:
        raise ValueError(
            f"Ground truth differs for {len(gt_mismatches)} aligned IDs. "
            f"Examples: {gt_mismatches[:10]}"
        )

    if size_mismatches:
        raise ValueError(
            f"Puzzle size differs for {len(size_mismatches)} aligned IDs. "
            f"Examples: {size_mismatches[:10]}"
        )

    return original_by_id, nss_by_id


def error_sets(original_by_id, nss_by_id):
    original_errors = {
        case_id
        for case_id, case in original_by_id.items()
        if case["puzzle_accuracy"] < 1.0
    }

    nss_errors = {
        case_id
        for case_id, case in nss_by_id.items()
        if case["puzzle_accuracy"] < 1.0
    }

    original_minus_nss = original_errors - nss_errors
    nss_minus_original = nss_errors - original_errors

    both_wrong = original_errors & nss_errors
    both_correct = (
        set(original_by_id)
        - original_errors
        - nss_errors
    )

    return {
        "original_errors": original_errors,
        "nss_errors": nss_errors,
        "original_minus_nss": original_minus_nss,
        "nss_minus_original": nss_minus_original,
        "both_wrong": both_wrong,
        "both_correct": both_correct,
    }


def sort_case_ids(case_ids, by_id):
    return sorted(
        case_ids,
        key=lambda cid: (
            parse_size(by_id[cid]["size_raw"]) or (999, 999),
            by_id[cid]["case_number"],
            cid,
        )
    )


def count_ids_by_difficulty(case_ids, by_id):
    counts = Counter(
        by_id[case_id]["difficulty"]
        for case_id in case_ids
    )
    return counts


def count_ids_by_size(case_ids, by_id):
    counts = Counter(
        by_id[case_id]["size"]
        for case_id in case_ids
    )
    return counts


def write_summary(
    output_path,
    original_result,
    nss_result,
    original_by_id,
    nss_by_id,
    errors,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_overall = summarize_cases(original_result["cases"])
    nss_overall = summarize_cases(nss_result["cases"])

    original_diff = summarize_by_difficulty(original_result["cases"])
    nss_diff = summarize_by_difficulty(nss_result["cases"])

    original_size = summarize_by_exact_size(original_result["cases"])
    nss_size = summarize_by_exact_size(nss_result["cases"])

    with output_path.open("w", encoding="utf-8") as out:
        def p(text=""):
            print(text)
            out.write(str(text) + "\n")

        p("=" * 140)
        p("GPT LOG COMPARISON: ORIGINAL vs NSS")
        p("=" * 140)
        p(f"Original: {original_result['filename']}")
        p(f"NSS:      {nss_result['filename']}")
        p(f"Aligned puzzle IDs: {len(original_by_id)}")
        p()

        p("ZebraLogic difficulty mapping used:")
        for difficulty in ["Small", "Medium", "Large", "XL"]:
            formatted = ", ".join(
                f"{a}x{b}"
                for a, b in sorted(DIFFICULTY_SIZES[difficulty])
            )
            p(f"  {difficulty:<6}: {formatted}")
        p()

        p("=" * 140)
        p("OVERALL ACCURACY")
        p("=" * 140)
        p(
            f"{'Metric':<34}"
            f"{'Original':>18}"
            f"{'NSS':>18}"
            f"{'NSS - Original':>20}"
        )
        p("-" * 90)
        p(
            f"{'Correct puzzles':<34}"
            f"{original_overall['correct_puzzles']:>18}"
            f"{nss_overall['correct_puzzles']:>18}"
            f"{nss_overall['correct_puzzles'] - original_overall['correct_puzzles']:>20}"
        )
        p(
            f"{'Puzzle Accuracy':<34}"
            f"{original_overall['puzzle_accuracy']:>18.4f}"
            f"{nss_overall['puzzle_accuracy']:>18.4f}"
            f"{nss_overall['puzzle_accuracy'] - original_overall['puzzle_accuracy']:>20.4f}"
        )
        p(
            f"{'Cell Accuracy (macro)':<34}"
            f"{original_overall['macro_cell_accuracy']:>18.4f}"
            f"{nss_overall['macro_cell_accuracy']:>18.4f}"
            f"{nss_overall['macro_cell_accuracy'] - original_overall['macro_cell_accuracy']:>20.4f}"
        )
        p(
            f"{'Cell Accuracy (micro)':<34}"
            f"{original_overall['micro_cell_accuracy']:>18.4f}"
            f"{nss_overall['micro_cell_accuracy']:>18.4f}"
            f"{nss_overall['micro_cell_accuracy'] - original_overall['micro_cell_accuracy']:>20.4f}"
        )
        p()

        p("Solution extraction:")
        p(
            f"  Original: full JSON={original_overall['full_answer_json']}, "
            f"fallback={original_overall['solution_fallback']}, "
            f"failure={original_overall['solution_failures']}"
        )
        p(
            f"  NSS:      full JSON={nss_overall['full_answer_json']}, "
            f"fallback={nss_overall['solution_fallback']}, "
            f"failure={nss_overall['solution_failures']}"
        )
        p()

        p("=" * 140)
        p("ACCURACY BY ZEBRALOGIC DIFFICULTY")
        p("=" * 140)
        p(
            f"{'Difficulty':<12}"
            f"{'N':>7}"
            f"{'Orig Correct':>14}"
            f"{'Orig PAcc':>12}"
            f"{'NSS Correct':>14}"
            f"{'NSS PAcc':>12}"
            f"{'Δ PAcc':>12}"
            f"{'Orig MacroC':>14}"
            f"{'NSS MacroC':>14}"
        )
        p("-" * 125)

        for difficulty in ["Small", "Medium", "Large", "XL", "Unknown"]:
            if difficulty not in original_diff and difficulty not in nss_diff:
                continue

            o = original_diff.get(difficulty, summarize_cases([]))
            n = nss_diff.get(difficulty, summarize_cases([]))

            if o["n"] != n["n"]:
                raise ValueError(
                    f"Different N for difficulty {difficulty}: "
                    f"{o['n']} vs {n['n']}"
                )

            p(
                f"{difficulty:<12}"
                f"{o['n']:>7}"
                f"{o['correct_puzzles']:>14}"
                f"{o['puzzle_accuracy']:>12.4f}"
                f"{n['correct_puzzles']:>14}"
                f"{n['puzzle_accuracy']:>12.4f}"
                f"{n['puzzle_accuracy'] - o['puzzle_accuracy']:>12.4f}"
                f"{o['macro_cell_accuracy']:>14.4f}"
                f"{n['macro_cell_accuracy']:>14.4f}"
            )

        p()
        p("=" * 140)
        p("ACCURACY BY EXACT PUZZLE SIZE")
        p("=" * 140)
        p(
            f"{'Size':<8}"
            f"{'Difficulty':<12}"
            f"{'N':>7}"
            f"{'Orig PAcc':>12}"
            f"{'NSS PAcc':>12}"
            f"{'Δ PAcc':>12}"
            f"{'Orig MacroC':>14}"
            f"{'NSS MacroC':>14}"
        )
        p("-" * 95)

        all_sizes = sorted(
            set(original_size) | set(nss_size),
            key=lambda s: parse_size(s) or (999, 999),
        )

        for size in all_sizes:
            o = original_size.get(size, summarize_cases([]))
            n = nss_size.get(size, summarize_cases([]))
            difficulty = difficulty_from_size(size)

            if o["n"] != n["n"]:
                raise ValueError(
                    f"Different N for size {size}: {o['n']} vs {n['n']}"
                )

            p(
                f"{size:<8}"
                f"{difficulty:<12}"
                f"{o['n']:>7}"
                f"{o['puzzle_accuracy']:>12.4f}"
                f"{n['puzzle_accuracy']:>12.4f}"
                f"{n['puzzle_accuracy'] - o['puzzle_accuracy']:>12.4f}"
                f"{o['macro_cell_accuracy']:>14.4f}"
                f"{n['macro_cell_accuracy']:>14.4f}"
            )

        p()
        p("=" * 140)
        p("PAIRED ERROR-DIFFERENCE SUMMARY")
        p("=" * 140)
        p(
            "Definition: original_minus_nss = errors(Original) - errors(NSS) "
            "= cases NSS fixes."
        )
        p(
            "Definition: nss_minus_original = errors(NSS) - errors(Original) "
            "= cases Original solves but NSS gets wrong."
        )
        p()

        n_both_correct = len(errors["both_correct"])
        n_both_wrong = len(errors["both_wrong"])
        n_nss_wins = len(errors["original_minus_nss"])
        n_original_wins = len(errors["nss_minus_original"])
        net = n_nss_wins - n_original_wins

        p(f"Both correct:                         {n_both_correct}")
        p(f"Both wrong:                           {n_both_wrong}")
        p(f"Original minus NSS / NSS wins:        {n_nss_wins}")
        p(f"NSS minus Original / Original wins:   {n_original_wins}")
        p(f"Net NSS advantage:                    {max(net, 0)}")
        p(f"Net Original advantage:               {max(-net, 0)}")
        if net > 0:
            p(f"Overall leader:                       NSS by {net} puzzles")
        elif net < 0:
            p(f"Overall leader:                       Original by {-net} puzzles")
        else:
            p("Overall leader:                       Tie")
        p()

        # ----------------------------------------------------
        # Full paired outcome statistics by ZebraLogic group
        # ----------------------------------------------------
        both_correct_by_diff = count_ids_by_difficulty(
            errors["both_correct"], original_by_id
        )
        both_wrong_by_diff = count_ids_by_difficulty(
            errors["both_wrong"], original_by_id
        )
        nss_wins_by_diff = count_ids_by_difficulty(
            errors["original_minus_nss"], original_by_id
        )
        original_wins_by_diff = count_ids_by_difficulty(
            errors["nss_minus_original"], original_by_id
        )

        p("PAIRED OUTCOMES BY ZEBRALOGIC DIFFICULTY")
        p(
            f"{'Difficulty':<12}"
            f"{'N':>7}"
            f"{'Both correct':>15}"
            f"{'Both wrong':>13}"
            f"{'NSS wins':>11}"
            f"{'Orig wins':>12}"
            f"{'Net NSS':>10}"
            f"{'Leader':>18}"
        )
        p("-" * 98)

        difficulty_order = ["Small", "Medium", "Large", "XL", "Unknown"]
        for difficulty in difficulty_order:
            o = original_diff.get(difficulty, summarize_cases([]))
            n_total = o["n"]
            both_correct = both_correct_by_diff[difficulty]
            both_wrong = both_wrong_by_diff[difficulty]
            nss_wins = nss_wins_by_diff[difficulty]
            original_wins = original_wins_by_diff[difficulty]

            if n_total == 0 and not (
                both_correct or both_wrong or nss_wins or original_wins
            ):
                continue

            # Strong internal consistency check: every aligned puzzle must
            # belong to exactly one of the four paired outcome categories.
            paired_total = (
                both_correct + both_wrong + nss_wins + original_wins
            )
            if paired_total != n_total:
                raise ValueError(
                    f"Paired outcome counts do not sum to N for "
                    f"{difficulty}: {paired_total} vs {n_total}"
                )

            net_nss = nss_wins - original_wins
            if net_nss > 0:
                leader = f"NSS +{net_nss}"
            elif net_nss < 0:
                leader = f"Original +{-net_nss}"
            else:
                leader = "Tie"

            p(
                f"{difficulty:<12}"
                f"{n_total:>7}"
                f"{both_correct:>15}"
                f"{both_wrong:>13}"
                f"{nss_wins:>11}"
                f"{original_wins:>12}"
                f"{net_nss:>+10}"
                f"{leader:>18}"
            )

        p()

        # ----------------------------------------------------
        # Full paired outcome statistics by exact puzzle size
        # ----------------------------------------------------
        both_correct_by_size = count_ids_by_size(
            errors["both_correct"], original_by_id
        )
        both_wrong_by_size = count_ids_by_size(
            errors["both_wrong"], original_by_id
        )
        nss_wins_by_size = count_ids_by_size(
            errors["original_minus_nss"], original_by_id
        )
        original_wins_by_size = count_ids_by_size(
            errors["nss_minus_original"], original_by_id
        )

        p("PAIRED OUTCOMES BY EXACT PUZZLE SIZE")
        p(
            f"{'Size':<8}"
            f"{'Difficulty':<12}"
            f"{'N':>7}"
            f"{'Both correct':>15}"
            f"{'Both wrong':>13}"
            f"{'NSS wins':>11}"
            f"{'Orig wins':>12}"
            f"{'Net NSS':>10}"
            f"{'Leader':>18}"
        )
        p("-" * 110)

        all_paired_sizes = sorted(
            set(original_size)
            | set(both_correct_by_size)
            | set(both_wrong_by_size)
            | set(nss_wins_by_size)
            | set(original_wins_by_size),
            key=lambda s: parse_size(s) or (999, 999),
        )

        for size in all_paired_sizes:
            n_total = original_size.get(size, summarize_cases([]))["n"]
            both_correct = both_correct_by_size[size]
            both_wrong = both_wrong_by_size[size]
            nss_wins = nss_wins_by_size[size]
            original_wins = original_wins_by_size[size]
            paired_total = (
                both_correct + both_wrong + nss_wins + original_wins
            )

            if paired_total != n_total:
                raise ValueError(
                    f"Paired outcome counts do not sum to N for size "
                    f"{size}: {paired_total} vs {n_total}"
                )

            net_nss = nss_wins - original_wins
            if net_nss > 0:
                leader = f"NSS +{net_nss}"
            elif net_nss < 0:
                leader = f"Original +{-net_nss}"
            else:
                leader = "Tie"

            p(
                f"{size:<8}"
                f"{difficulty_from_size(size):<12}"
                f"{n_total:>7}"
                f"{both_correct:>15}"
                f"{both_wrong:>13}"
                f"{nss_wins:>11}"
                f"{original_wins:>12}"
                f"{net_nss:>+10}"
                f"{leader:>18}"
            )

        p()
        p()
        p("Interpretation:")
        if net > 0:
            p(
                f"  NSS has a net advantage of {net} puzzles because it fixes "
                f"{len(errors['original_minus_nss'])} Original errors while "
                f"introducing {len(errors['nss_minus_original'])} regressions."
            )
        elif net < 0:
            p(
                f"  Original has a net advantage of {-net} puzzles because NSS "
                f"introduces more regressions than fixes."
            )
        else:
            p(
                "  The systems have equal total puzzle accuracy, although they "
                "may solve different individual cases."
            )

    return output_path


def compare_logs(
    original_file,
    nss_file,
    output_dir="./Outputs",
    puzzle_info_file=None,
):
    if puzzle_info_file is None:
        puzzle_info_file = Path(original_file).parent / PUZZLE_INFO_FILE

    puzzle_lookup = load_puzzle_dictionary(puzzle_info_file)

    original_result = analyze_log(original_file, puzzle_lookup=puzzle_lookup)
    nss_result = analyze_log(nss_file, puzzle_lookup=puzzle_lookup)

    original_by_id, nss_by_id = validate_pair_alignment(
        original_result,
        nss_result,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Human-readable TXT exports of complete logs.
    original_txt = write_log_as_txt(
        original_result,
        output_dir,
    )
    nss_txt = write_log_as_txt(
        nss_result,
        output_dir,
    )

    # 2/3/4. Accuracy + paired error differences.
    errors = error_sets(
        original_by_id,
        nss_by_id,
    )

    original_minus_nss_ids = sort_case_ids(
        errors["original_minus_nss"],
        original_by_id,
    )
    nss_minus_original_ids = sort_case_ids(
        errors["nss_minus_original"],
        original_by_id,
    )

    original_minus_nss_file = output_dir / "original_minus_nss.txt"
    nss_minus_original_file = output_dir / "nss_minus_original.txt"

    write_error_difference_file(
        original_minus_nss_file,
        original_minus_nss_ids,
        (
            "ORIGINAL MINUS NSS = errors(Original) - errors(NSS)\n"
            "These are cases the Original system gets WRONG and NSS gets CORRECT."
        ),
        original_by_id,
        nss_by_id,
    )

    write_error_difference_file(
        nss_minus_original_file,
        nss_minus_original_ids,
        (
            "NSS MINUS ORIGINAL = errors(NSS) - errors(Original)\n"
            "These are cases NSS gets WRONG and the Original system gets CORRECT."
        ),
        original_by_id,
        nss_by_id,
    )

    summary_file = output_dir / "accuracy_and_error_difference_summary.txt"

    write_summary(
        summary_file,
        original_result,
        nss_result,
        original_by_id,
        nss_by_id,
        errors,
    )

    print()
    print("=" * 100)
    print("OUTPUT FILES")
    print("=" * 100)
    print(f"Original TXT:           {original_txt.resolve()}")
    print(f"NSS TXT:                {nss_txt.resolve()}")
    print(f"Original minus NSS:     {original_minus_nss_file.resolve()}")
    print(f"NSS minus Original:     {nss_minus_original_file.resolve()}")
    print(f"Summary:                {summary_file.resolve()}")

    return {
        "original_txt": original_txt,
        "nss_txt": nss_txt,
        "original_minus_nss": original_minus_nss_file,
        "nss_minus_original": nss_minus_original_file,
        "summary": summary_file,
    }


# ============================================================
# Main
# ============================================================

def main():
    original_file = INPUT_DIR / ORIGINAL_FILE
    nss_file = INPUT_DIR / NSS_FILE

    compare_logs(
        original_file=original_file,
        nss_file=nss_file,
        output_dir=OUTPUT_DIR,
        puzzle_info_file=INPUT_DIR / PUZZLE_INFO_FILE,
    )


if __name__ == "__main__":
    main()
