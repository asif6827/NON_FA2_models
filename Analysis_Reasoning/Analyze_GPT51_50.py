import json
import re
from pathlib import Path


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
        " | ".join(
            str(x)
            for x in header
        )
        + "\n"
    )

    output.write("-" * 120 + "\n")

    for row in rows:
        output.write(
            " | ".join(
                str(x)
                for x in row
            )
            + "\n"
        )


# ============================================================
# Parse one outer JSONL line
# ============================================================

def parse_outer_record(line, line_number):
    """
    Normal records look like:

        {
            "index": ...,
            "id": ...,
            "ground_truth": ...,
            "llm_output": ...,
            "status": ...
        }

    The first record in the supplied log instead starts with:

        {"prompt": {...}

    and appears to be missing one final closing brace.

    This function handles both formats.
    """

    try:
        record = json.loads(line)

    except json.JSONDecodeError as original_error:

        # ----------------------------------------------------
        # Repair special first-record format:
        #
        # {"prompt": {...}
        #
        # -> {"prompt": {...}}
        # ----------------------------------------------------

        stripped = line.strip()

        if stripped.startswith('{"prompt": {'):

            try:
                record = json.loads(
                    stripped + "}"
                )

            except json.JSONDecodeError:
                raise original_error

        else:
            raise original_error

    # --------------------------------------------------------
    # Unwrap optional "prompt" container
    # --------------------------------------------------------

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
            f"Could not parse JSON inside "
            f"<answer>: {e}"
        )

    return payload, None


# ============================================================
# Print reasoning
# ============================================================

def print_reasoning(output, reasoning):

    if not reasoning:
        output.write("NONE\n")
        return

    # --------------------------------------------------------
    # This GPT log stores reasoning as a DICTIONARY:
    #
    # {
    #     "NL1": "...",
    #     "S1": "...",
    #     "NL2": "...",
    #     "S2": "...",
    #     "PA1": {...}
    # }
    # --------------------------------------------------------

    if isinstance(reasoning, dict):

        for key, value in reasoning.items():

            output.write(
                f"\n[{key}]\n"
            )

            if isinstance(value, dict):

                # Intermediate partial-answer table
                print_table_to_file(
                    output,
                    value
                )

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
                output.write(
                    str(value)
                    + "\n"
                )

        return

    # --------------------------------------------------------
    # Support older reasoning-list format too
    # --------------------------------------------------------

    if isinstance(reasoning, list):

        for i, item in enumerate(
            reasoning,
            start=1
        ):

            output.write(
                f"[{i}] {item}\n"
            )

        return

    output.write(
        str(reasoning)
        + "\n"
    )


# ============================================================
# Process complete JSONL file
# ============================================================

def read_log(filename):

    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(
            f"File not found: {filename}"
        )

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = Path("./Outputs")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"{filename.stem}_readable.txt"
    )

    # ========================================================
    # Read line-by-line
    # ========================================================

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

            output.write(
                "=" * 120
                + "\n"
            )

            output.write(
                f"EXAMPLE {line_number}\n"
            )

            output.write(
                "=" * 120
                + "\n"
            )

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

                output.write(
                    "\n" * 8
                )

                continue

            # =================================================
            # Basic metadata
            # =================================================

            output.write(
                f"\nINDEX  : "
                f"{record.get('index')}\n"
            )

            output.write(
                f"ID     : "
                f"{record.get('id')}\n"
            )

            output.write(
                f"SIZE   : "
                f"{record.get('size')}\n"
            )

            output.write(
                f"STATUS : "
                f"{record.get('status')}\n"
            )

            # =================================================
            # Ground truth
            # =================================================

            output.write(
                "\n\n### GROUND TRUTH ###\n\n"
            )

            print_table_to_file(
                output,
                record.get(
                    "ground_truth",
                    {}
                )
            )

            # =================================================
            # Raw LLM output
            # =================================================

            llm_output = record.get(
                "llm_output"
            )

            output.write(
                "\n\n### RAW LLM OUTPUT ###\n\n"
            )

            output.write(
                str(llm_output)
                + "\n"
            )

            # =================================================
            # Parse <answer> payload
            # =================================================

            payload, error = (
                extract_answer_payload(
                    llm_output
                )
            )

            if payload is None:

                output.write(
                    "\n\n### LLM OUTPUT "
                    "PARSING ERROR ###\n\n"
                )

                output.write(
                    str(error)
                    + "\n"
                )

                output.write(
                    "\n" * 8
                )

                continue

            # =================================================
            # Number of houses
            # =================================================

            output.write(
                "\n\n### NUMBER OF HOUSES ###\n\n"
            )

            output.write(
                str(
                    payload.get(
                        "n_houses"
                    )
                )
                + "\n"
            )

            # =================================================
            # Attribute values
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
            # Syntactic clues
            # =================================================

            output.write(
                "\n\n### SYNTACTIC CLUES ###\n\n"
            )

            clues = payload.get(
                "syntactic_clues",
                []
            )

            if clues:

                for i, clue in enumerate(
                    clues,
                    start=1
                ):

                    output.write(
                        f"[{i}] {clue}\n"
                    )

            else:

                output.write(
                    "NONE\n"
                )

            # =================================================
            # Reasoning
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
            # Final model solution
            # =================================================

            output.write(
                "\n\n### MODEL SOLUTION ###\n\n"
            )

            print_table_to_file(
                output,
                payload.get(
                    "solution",
                    {}
                )
            )

            # =================================================
            # Parsed complete answer JSON
            # =================================================

            output.write(
                "\n\n### COMPLETE PARSED "
                "ANSWER JSON ###\n\n"
            )

            output.write(
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False
                )
                + "\n"
            )

            # Several blank lines between examples
            output.write(
                "\n" * 8
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