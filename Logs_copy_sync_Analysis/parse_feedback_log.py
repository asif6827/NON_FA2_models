import json
import argparse
from pathlib import Path


def format_value(value):
    if isinstance(value, str):
        return value.replace("\n", "\\n")
    if value is None:
        return "None"
    return str(value)


def is_table_dict(value):
    """
    Detect structures like:
    {
        "header": [...],
        "rows": [[...], [...]]
    }
    """
    return (
        isinstance(value, dict)
        and "header" in value
        and "rows" in value
        and isinstance(value["header"], list)
        and isinstance(value["rows"], list)
    )


def print_clean_field(name, value, indent=0, write=print):
    space = " " * indent

    if is_table_dict(value):
        write(f"{space}{name}:")

        header = value.get("header", [])
        rows = value.get("rows", [])

        if header:
            write(f"{space}  header: {' | '.join(format_value(x) for x in header)}")

        write(f"{space}  rows:")

        for row in rows:
            if isinstance(row, list):
                write(f"{space}    {' | '.join(format_value(x) for x in row)}")
            else:
                write(f"{space}    {format_value(row)}")

        return

    if isinstance(value, dict):
        write(f"{space}{name}:")

        for key, val in value.items():
            print_clean_field(key, val, indent + 2, write)

        return

    if isinstance(value, list):
        write(f"{space}{name}:")

        for item in value:
            if isinstance(item, dict):
                write(f"{space}  -")
                for key, val in item.items():
                    print_clean_field(key, val, indent + 4, write)
            elif isinstance(item, list):
                write(f"{space}  - {' | '.join(format_value(x) for x in item)}")
            else:
                write(f"{space}  - {format_value(item)}")

        return

    write(f"{space}{name}: {format_value(value)}")


def parse_log_file(file_path, limit=None, output_file=None):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    out = open(output_file, "w", encoding="utf-8") if output_file else None

    def write(line=""):
        if out:
            out.write(line + "\n")
        else:
            print(line)

    total = 0

    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                write("=" * 120)
                write(f"INVALID JSON AT LINE {line_no}")
                write(f"Error: {e}")
                continue

            total += 1

            if limit is not None and total > limit:
                break

            pid = record.get("pid", "UNKNOWN_PID")

            write("=" * 120)
            write(f"PUZZLE #{total} | JSONL LINE: {line_no} | PID: {pid}")
            write("=" * 120)

            for key, value in record.items():
                print_clean_field(key, value, indent=0, write=write)

            write()

    if out:
        out.close()

    parsed_count = min(total, limit) if limit else total
    print(f"Parsed {parsed_count} puzzle records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse puzzle feedback JSONL log and print every field line by line.")
    parser.add_argument("--file", default="/home/asif/data3/Codes_QCRI/NON_FA2_models/Reasoning360_sys_B_v29_llama3_3B/evaluation_results/mlxl_train_mlxl_test_1_parsed_v6a_MLXL/llama323binstruct/jobid_298697/jobid_298697_epoch_34_valid_feedback.jsonl", help="Path to the JSONL log file")
    parser.add_argument("--limit", type=int, default=None, help="Only parse first N puzzle records")
    parser.add_argument("--out", default=None, help="Optional output text file path")
    args = parser.parse_args()
    parse_log_file(file_path=args.file, limit=args.limit, output_file=args.out)
