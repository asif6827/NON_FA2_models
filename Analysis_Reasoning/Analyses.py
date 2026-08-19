import json
import sys
from pathlib import Path


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

    print("\n\n### PAYLOAD ###")

    if not payload:

        print("EMPTY")

    else:

        print("\nNumber of houses:")
        print(payload.get("n_houses"))

        print("\nAttribute values:")
        print_dict(
            payload.get("attribute_values", {}),
            indent=4
        )

        print("\nSyntactic clues:")

        syntactic_clues = payload.get(
            "syntactic_clues",
            []
        )

        if syntactic_clues:

            for i, clue in enumerate(
                syntactic_clues,
                start=1
            ):
                print(f"  [{i}] {clue}")

        else:

            print("  NONE")


        # -----------------------------------------------------
        # Reasoning
        # -----------------------------------------------------

        print("\nReasoning:")

        reasoning = payload.get(
            "reasoning",
            []
        )

        if reasoning:

            for i, step in enumerate(
                reasoning,
                start=1
            ):

                print(f"\n  [{i}] {step}")

        else:

            print("  NONE")


    # =========================================================
    # Z3 Output
    # =========================================================

    print("\n\n### Z3 OUTPUT ###\n")

    z3_out = example.get(
        "z3_out",
        {}
    )

    if z3_out:

        print_dict(
            z3_out,
            indent=2
        )

    else:

        print("EMPTY")


    # =========================================================
    # Reasoning Validation
    # =========================================================

    print(
        "\n\n"
        "### REASONING VS SOLUTION VALIDATION ###"
        "\n"
    )

    validation = example.get(
        "reasoning_vs_sol_validate",
        {}
    )

    if validation:

        print_dict(
            validation,
            indent=2
        )

    else:

        print("EMPTY")


    # =========================================================
    # Ground Truth
    # =========================================================

    print("\n\n### GROUND TRUTH ###\n")

    print_ground_truth(
        example.get(
            "ground_truth",
            {}
        )
    )


    # =========================================================
    # Original Prediction
    # =========================================================

    print(
        "\n\n### ORIGINAL PREDICTION ###\n"
    )

    original = example.get(
        "original_prediction"
    )

    if original is None:

        print("NONE")

    elif isinstance(original, dict):

        print_ground_truth(original)

    else:

        print(original)


    # =========================================================
    # Processed Prediction
    # =========================================================

    print(
        "\n\n### PROCESSED PREDICTION ###\n"
    )

    processed = example.get(
        "processed_prediction"
    )

    if isinstance(processed, dict):

        print_ground_truth(processed)

    else:

        print(processed)


    # =========================================================
    # Basic Result
    # =========================================================

    print("\n\n### BASIC RESULT ###\n")

    print(
        f"Reward       : "
        f"{example.get('reward')}"
    )

    print(
        f"Format Check : "
        f"{example.get('Format_Check')}"
    )


    # =========================================================
    # Final Result
    # =========================================================

    print("\n\n### FINAL RESULT ###\n")

    final_result = example.get(
        "final_result",
        {}
    )

    if final_result:

        for key, value in final_result.items():

            print(
                f"{key:35s}: {value}"
            )

    else:

        print("EMPTY")


    separator(
        char="-"
    )


def read_jsonl(
    filename,
    start=1,
    limit=None
):

    path = Path(filename)

    if not path.exists():

        raise FileNotFoundError(
            f"File does not exist: {filename}"
        )

    printed = 0

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if line_number < start:
                continue

            line = line.strip()

            if not line:
                continue

            try:

                example = json.loads(line)

            except json.JSONDecodeError as e:

                print(
                    f"ERROR parsing line "
                    f"{line_number}: {e}"
                )

                continue


            print_example(
                example,
                line_number
            )

            printed += 1


            # =================================================
            # Large visual gap before next example
            # =================================================

            print("\n" * EXAMPLE_GAP)


            if (
                limit is not None
                and printed >= limit
            ):
                break


def analyze_file(
    filename,
    start=1,
    limit=None
):
    """
    Analyze JSONL and save readable output
    inside the Outputs folder.
    """

    input_path = Path(filename)


    # =========================================================
    # Create Outputs folder
    # =========================================================

    output_folder = Path("Outputs")

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    # =========================================================
    # Generate output filename automatically
    #
    # example:
    #
    # abc.jsonl
    #
    # becomes:
    #
    # Outputs/abc.txt
    # =========================================================

    output_filename = (
        output_folder
        / f"{input_path.stem}.txt"
    )


    print(
        f"Input file : {input_path}"
    )

    print(
        f"Output file: {output_filename}"
    )


    # =========================================================
    # Save original stdout
    # =========================================================

    original_stdout = sys.stdout


    try:

        with output_filename.open(
            "w",
            encoding="utf-8"
        ) as output_file:


            # -------------------------------------------------
            # Everything printed from here goes to BOTH:
            #
            # Terminal
            # Output TXT file
            # -------------------------------------------------

            sys.stdout = Tee(
                original_stdout,
                output_file
            )


            separator(
                "REASONING DRIFT ANALYSIS"
            )

            print(
                f"\nSource file : "
                f"{input_path}"
            )

            print(
                f"Start       : "
                f"{start}"
            )

            print(
                f"Limit       : "
                f"{limit if limit is not None else 'ALL'}"
            )

            print("\n")


            read_jsonl(
                filename,
                start=start,
                limit=limit
            )


    finally:

        # Always restore terminal stdout
        sys.stdout = original_stdout


    print(
        "\nReadable analysis saved to:"
    )

    print(
        output_filename.resolve()
    )


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":


    # =========================================================
    # FILE TO ANALYZE
    # =========================================================

    from pathlib import Path

    base_path = Path(
        "/home/asif/data3/Codes_QCRI/NON_FA2_models/"
        "Reasoning360_sys_B_v29/evaluation_results/"
        "mlxl_train_mlxl_test_1_parsed_v6a_MLXL/"
        "qwen34bthinking2507/jobid_276924"
    )

    file_name = "jobid_276924_epoch_28_train_feedback.jsonl"

    filename = base_path / file_name

    # For validation:
    #
    # filename = (
    #     "jobid_276924_epoch_28_valid_feedback.jsonl"
    # )


    # =========================================================
    # ANALYSIS SETTINGS
    # =========================================================

    # First example to analyze
    start = 1


    # None = analyze everything
    #
    # Example:
    # limit = 10
    #
    limit = None


    # =========================================================
    # RUN ANALYSIS
    # =========================================================

    analyze_file(
        filename,
        start=start,
        limit=limit
    )