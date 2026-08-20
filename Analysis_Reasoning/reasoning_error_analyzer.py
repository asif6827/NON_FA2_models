import json
import re
from pathlib import Path
from collections import defaultdict


# ============================================================
# Utilities
# ============================================================

def normalize(value):
    value = str(value).strip().lower()
    value = value.replace("-", "_")
    value = value.replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def strip_step_label(text):
    """
    S3: red == Eric.
        ->
    red == Eric
    """

    text = text.strip()

    match = re.match(
        r"^[SC]\d+\s*:\s*(.*)$",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        text = match.group(1)

    return text.strip().rstrip(".")


# ============================================================
# Convert GT / prediction table into positions
# ============================================================

def table_to_positions(table):
    """
    Example:

    House | Name | Drink | Color
    1       Eric   Tea     Red
    2       Bob    Coffee  Blue

    becomes:

    {
        "eric": 1,
        "tea": 1,
        "red": 1,
        "bob": 2,
        "coffee": 2,
        "blue": 2
    }
    """

    if not isinstance(table, dict):
        return {}

    header = table.get("header", [])
    rows = table.get("rows", [])

    if not header or not rows:
        return {}

    house_index = None

    for i, column in enumerate(header):
        if normalize(column) == "house":
            house_index = i
            break

    if house_index is None:
        return {}

    temporary = defaultdict(list)

    for row in rows:

        if house_index >= len(row):
            continue

        try:
            house = int(row[house_index])
        except Exception:
            continue

        for i, value in enumerate(row):

            if i == house_index:
                continue

            token = normalize(value)

            if token:
                temporary[token].append(house)

    positions = {}

    for token, houses in temporary.items():

        # Avoid ambiguous duplicate attribute names
        if len(set(houses)) == 1:
            positions[token] = houses[0]

    return positions


# ============================================================
# Basic symbolic expression evaluator
# ============================================================

def evaluate_term(term, positions):
    """
    Supports:

        Eric
        red
        3
        Eric + 1
        red - 2
    """

    term = term.strip()

    # Integer
    if re.fullmatch(r"-?\d+", term):
        return int(term)

    # variable +/- offset
    match = re.fullmatch(
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"\s*([+-])\s*(\d+)",
        term,
    )

    if match:

        token = normalize(match.group(1))
        operator = match.group(2)
        offset = int(match.group(3))

        if token not in positions:
            raise ValueError(
                f"Unknown token: {token}"
            )

        value = positions[token]

        if operator == "+":
            return value + offset

        return value - offset

    # Plain symbolic variable
    token = normalize(term)

    if token not in positions:
        raise ValueError(
            f"Unknown token: {token}"
        )

    return positions[token]


def split_arguments(text):
    arguments = []

    depth = 0
    start = 0

    for i, char in enumerate(text):

        if char == "(":
            depth += 1

        elif char == ")":
            depth -= 1

        elif char == "," and depth == 0:

            arguments.append(
                text[start:i].strip()
            )

            start = i + 1

    arguments.append(
        text[start:].strip()
    )

    return arguments


def evaluate_expression(expression, positions):
    """
    Evaluate symbolic step against a concrete table.

    Supported examples:

        Eric == 3
        Eric != 4
        Eric < Bob
        Eric + 1 == Bob
        red == tea
        Or(Eric == 1, Eric == 2)
        And(Eric > 1, Eric < 4)
        Not(Eric == 3)
    """

    expression = strip_step_label(expression)

    # --------------------------------------------------------
    # NOT
    # --------------------------------------------------------

    if (
        expression.startswith("Not(")
        and expression.endswith(")")
    ):

        inner = expression[4:-1]

        result = evaluate_expression(
            inner,
            positions,
        )

        return not result

    # --------------------------------------------------------
    # OR
    # --------------------------------------------------------

    if (
        expression.startswith("Or(")
        and expression.endswith(")")
    ):

        inside = expression[3:-1]

        arguments = split_arguments(
            inside
        )

        return any(
            evaluate_expression(
                argument,
                positions,
            )
            for argument in arguments
        )

    # --------------------------------------------------------
    # AND
    # --------------------------------------------------------

    if (
        expression.startswith("And(")
        and expression.endswith(")")
    ):

        inside = expression[4:-1]

        arguments = split_arguments(
            inside
        )

        return all(
            evaluate_expression(
                argument,
                positions,
            )
            for argument in arguments
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    match = re.fullmatch(
        r"(.+?)\s*(==|!=|<=|>=|<|>)\s*(.+)",
        expression,
    )

    if not match:

        raise ValueError(
            f"Unsupported expression: {expression}"
        )

    left_text = match.group(1).strip()
    operator = match.group(2)
    right_text = match.group(3).strip()

    left = evaluate_term(
        left_text,
        positions,
    )

    right = evaluate_term(
        right_text,
        positions,
    )

    if operator == "==":
        return left == right

    if operator == "!=":
        return left != right

    if operator == "<":
        return left < right

    if operator == ">":
        return left > right

    if operator == "<=":
        return left <= right

    if operator == ">=":
        return left >= right

    raise ValueError(
        f"Unknown operator: {operator}"
    )


# ============================================================
# Extract reasoning S-steps and preceding NL
# ============================================================

def extract_reasoning_steps(reasoning):
    """
    Input might look like:

    [
        "From clue 3, Eric must be...",
        "This places Eric...",
        "S1: Eric == 4.",
        "Now using clue 7...",
        "S2: tea == Eric."
    ]

    Result:

    S1 gets the preceding NL explanation.

    S2 gets the NL between S1 and S2.
    """

    steps = []

    nl_buffer = []

    for reasoning_item_index, item in enumerate(
        reasoning,
        start=1,
    ):

        if not isinstance(item, str):
            continue

        item = item.strip()

        match = re.match(
            r"^(S\d+)\s*:\s*(.+)$",
            item,
            flags=re.IGNORECASE,
        )

        if match:

            steps.append(
                {
                    "label":
                        match.group(1),

                    "expression":
                        match.group(2)
                        .strip()
                        .rstrip("."),

                    "raw":
                        item,

                    "reasoning_item_index":
                        reasoning_item_index,

                    "preceding_nl":
                        list(nl_buffer),
                }
            )

            nl_buffer = []

        else:

            nl_buffer.append(item)

    return steps


# ============================================================
# Analyze reasoning
# ============================================================

def analyze_reasoning(
    reasoning,
    ground_truth,
    final_prediction,
):

    gt_positions = table_to_positions(
        ground_truth
    )

    pred_positions = table_to_positions(
        final_prediction
    )

    steps = extract_reasoning_steps(
        reasoning
    )

    analyzed_steps = []

    first_error = None

    for index, step in enumerate(
        steps,
        start=1,
    ):

        result = {
            **step,

            "symbolic_index":
                index,

            "gt_status":
                "UNVERIFIED",

            "prediction_status":
                "UNVERIFIED",

            "gt_error":
                None,

            "prediction_error":
                None,
        }

        # ====================================================
        # Check against ground truth
        # ====================================================

        try:

            gt_result = evaluate_expression(
                step["expression"],
                gt_positions,
            )

            result["gt_status"] = (
                "CORRECT"
                if gt_result
                else "ERROR"
            )

        except Exception as e:

            result["gt_status"] = (
                "UNVERIFIED"
            )

            result["gt_error"] = str(e)

        # ====================================================
        # Check same reasoning claim against final prediction
        # ====================================================

        try:

            pred_result = evaluate_expression(
                step["expression"],
                pred_positions,
            )

            result[
                "prediction_status"
            ] = (
                "CONSISTENT"
                if pred_result
                else "INCONSISTENT"
            )

        except Exception as e:

            result[
                "prediction_status"
            ] = "UNVERIFIED"

            result[
                "prediction_error"
            ] = str(e)

        analyzed_steps.append(result)

        # ====================================================
        # FIRST HARD ERROR
        # ====================================================

        if (
            first_error is None
            and
            result["gt_status"] == "ERROR"
        ):

            first_error = result

    return {
        "steps":
            analyzed_steps,

        "first_error":
            first_error,
    }


# ============================================================
# Compare final table with GT
# ============================================================

def compare_predictions(
    ground_truth,
    prediction,
):

    if not isinstance(
        ground_truth,
        dict,
    ):

        return []

    if not isinstance(
        prediction,
        dict,
    ):

        return [
            {
                "type":
                    "INVALID_FINAL_PREDICTION"
            }
        ]

    gt_header = ground_truth.get(
        "header",
        [],
    )

    pred_header = prediction.get(
        "header",
        [],
    )

    gt_rows = ground_truth.get(
        "rows",
        [],
    )

    pred_rows = prediction.get(
        "rows",
        [],
    )

    if gt_header != pred_header:

        return [
            {
                "type":
                    "HEADER_MISMATCH",

                "ground_truth":
                    gt_header,

                "prediction":
                    pred_header,
            }
        ]

    mismatches = []

    for row_index, gt_row in enumerate(
        gt_rows
    ):

        if row_index >= len(pred_rows):

            mismatches.append(
                {
                    "row":
                        row_index + 1,

                    "type":
                        "MISSING_ROW",
                }
            )

            continue

        pred_row = pred_rows[row_index]

        for column_index, gt_value in enumerate(
            gt_row
        ):

            if column_index >= len(pred_row):
                continue

            pred_value = (
                pred_row[column_index]
            )

            if (
                normalize(gt_value)
                !=
                normalize(pred_value)
            ):

                mismatches.append(
                    {
                        "row":
                            row_index + 1,

                        "column":
                            gt_header[
                                column_index
                            ],

                        "ground_truth":
                            gt_value,

                        "prediction":
                            pred_value,
                    }
                )

    return mismatches


# ============================================================
# Analyze one example
# ============================================================

def diagnose_example(example):

    payload = (
        example.get(
            "payload",
            {}
        )
        or {}
    )

    reasoning = payload.get(
        "reasoning",
        [],
    ) or []

    ground_truth = example.get(
        "ground_truth",
        {},
    )

    final_prediction = example.get(
        "processed_prediction",
        {},
    )

    reasoning_analysis = (
        analyze_reasoning(
            reasoning,
            ground_truth,
            final_prediction,
        )
    )

    final_errors = compare_predictions(
        ground_truth,
        final_prediction,
    )

    first_error = (
        reasoning_analysis[
            "first_error"
        ]
    )

    # ========================================================
    # Reasoning -> final interaction
    # ========================================================

    if first_error is None:

        if final_errors:

            interaction = (
                "FINAL_ERROR_BUT_NO_SYMBOLIC_"
                "REASONING_ERROR_DETECTED"
            )

        else:

            interaction = "NO_ERROR"

    else:

        if (
            first_error[
                "prediction_status"
            ]
            ==
            "CONSISTENT"
        ):

            interaction = (
                "FIRST_REASONING_ERROR_"
                "PRESERVED_IN_FINAL"
            )

        elif (
            first_error[
                "prediction_status"
            ]
            ==
            "INCONSISTENT"
        ):

            interaction = (
                "FIRST_REASONING_ERROR_"
                "NOT_PRESERVED_IN_FINAL"
            )

        else:

            interaction = (
                "PROPAGATION_UNVERIFIED"
            )

    return {
        "pid":
            example.get("pid"),

        "puzzle_text":
            example.get(
                "puzzle_text"
            ),

        "syntactic_clues":
            payload.get(
                "syntactic_clues",
                [],
            ),

        "reasoning":
            reasoning,

        "reasoning_analysis":
            reasoning_analysis,

        "first_error":
            first_error,

        "ground_truth":
            ground_truth,

        "final_prediction":
            final_prediction,

        "final_prediction_errors":
            final_errors,

        "reasoning_final_interaction":
            interaction,

        "final_result":
            example.get(
                "final_result",
                {},
            ),
    }


# ============================================================
# Filtering
# ============================================================

def should_include(
    example,
    mode,
):

    final_result = (
        example.get(
            "final_result",
            {}
        )
        or {}
    )

    puzzle_accuracy = (
        final_result.get(
            "PUZZLE_ACCURACY"
        )
    )

    if mode == "all":
        return True

    if mode == "errors":
        return puzzle_accuracy == 0.0

    if mode == "correct":
        return puzzle_accuracy == 1.0

    raise ValueError(
        f"Unknown mode: {mode}"
    )


# ============================================================
# Human-readable report
# ============================================================

def write_case(
    output,
    diagnosis,
):

    def p(text=""):
        print(text)
        output.write(str(text) + "\n")

    p("=" * 120)

    p(
        f"PID: {diagnosis['pid']}"
    )

    p("=" * 120)

    # ========================================================
    # Original puzzle
    # ========================================================

    p("\n### PUZZLE ###\n")

    p(
        diagnosis[
            "puzzle_text"
        ]
    )

    # ========================================================
    # Syntactic clues
    # ========================================================

    p("\n\n### SYNTACTIC CLUES ###\n")

    for clue in (
        diagnosis[
            "syntactic_clues"
        ]
    ):

        p(clue)

    # ========================================================
    # Step audit
    # ========================================================

    p(
        "\n\n### REASONING STEP AUDIT ###"
    )

    for step in (
        diagnosis[
            "reasoning_analysis"
        ]["steps"]
    ):

        p("\n" + "-" * 100)

        p(
            f"{step['label']}: "
            f"{step['expression']}"
        )

        if step["preceding_nl"]:

            p("\nPreceding reasoning:")

            for line in (
                step["preceding_nl"]
            ):

                p(f"    {line}")

        p(
            "\nGround Truth Check : "
            f"{step['gt_status']}"
        )

        p(
            "Final Prediction    : "
            f"{step['prediction_status']}"
        )

        if step["gt_error"]:

            p(
                "GT parser note       : "
                f"{step['gt_error']}"
            )

    # ========================================================
    # First error
    # ========================================================

    p(
        "\n\n### FIRST ERRONEOUS "
        "SYMBOLIC STEP ###\n"
    )

    first_error = diagnosis[
        "first_error"
    ]

    if first_error is None:

        p(
            "NO HARD SYMBOLIC ERROR "
            "WAS DETECTED."
        )

    else:

        p(
            f"Step          : "
            f"{first_error['label']}"
        )

        p(
            f"Expression    : "
            f"{first_error['expression']}"
        )

        p(
            f"Reasoning item: "
            f"{first_error['reasoning_item_index']}"
        )

        p(
            f"GT check      : "
            f"{first_error['gt_status']}"
        )

        p(
            f"Final follows : "
            f"{first_error['prediction_status']}"
        )

        if first_error[
            "preceding_nl"
        ]:

            p(
                "\nNL reasoning immediately "
                "before first error:"
            )

            for line in (
                first_error[
                    "preceding_nl"
                ]
            ):

                p(
                    f"    {line}"
                )

    # ========================================================
    # Final errors
    # ========================================================

    p(
        "\n\n### FINAL PREDICTION "
        "MISMATCHES ###\n"
    )

    errors = diagnosis[
        "final_prediction_errors"
    ]

    p(
        f"Number of mismatches: "
        f"{len(errors)}"
    )

    for error in errors:

        p(
            json.dumps(
                error,
                ensure_ascii=False,
            )
        )

    # ========================================================
    # Interaction
    # ========================================================

    p(
        "\n\n### REASONING / FINAL "
        "INTERACTION ###\n"
    )

    p(
        diagnosis[
            "reasoning_final_interaction"
        ]
    )

    # ========================================================
    # ChatGPT diagnostic packet
    # ========================================================

    p(
        "\n\n### CHATGPT DIAGNOSTIC "
        "FOCUS ###\n"
    )

    if first_error:

        p(
            "Please investigate WHY the "
            "following is the first erroneous "
            "step:"
        )

        p(
            f"\n{first_error['label']}: "
            f"{first_error['expression']}"
        )

        p(
            "\nCheck:"
        )

        p(
            "1. Is there an earlier mistake "
            "in the natural-language reasoning?"
        )

        p(
            "2. Which clue or prior deduction "
            "was misunderstood?"
        )

        p(
            "3. What is the root cause of "
            "the error?"
        )

        p(
            "4. What should the correct "
            "reasoning have been?"
        )

        p(
            "5. Did this first error propagate "
            "into the final prediction?"
        )

    else:

        p(
            "The final prediction is wrong, "
            "but Python did not identify a "
            "false symbolic S-step."
        )

        p(
            "Inspect the NL reasoning for an "
            "unsupported inference, omitted "
            "deduction, or final-answer "
            "construction error."
        )

    p("\n" * 8)


# ============================================================
# Main JSONL processing
# ============================================================

def analyze_file(
    filename,
    mode="errors",
    start=1,
    limit=None,
):

    input_path = Path(filename)

    output_folder = (
        input_path.parent
        / "Outputs"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    txt_output = (
        output_folder
        /
        f"{input_path.stem}_"
        f"diagnostic_{mode}.txt"
    )

    jsonl_output = (
        output_folder
        /
        f"{input_path.stem}_"
        f"diagnostic_{mode}.jsonl"
    )

    analyzed = 0

    with (
        input_path.open(
            "r",
            encoding="utf-8",
        ) as source,

        txt_output.open(
            "w",
            encoding="utf-8",
        ) as txt,

        jsonl_output.open(
            "w",
            encoding="utf-8",
        ) as structured,
    ):

        for line_number, line in enumerate(
            source,
            start=1,
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
                    f"Invalid JSON line "
                    f"{line_number}: {e}"
                )

                continue

            if not should_include(
                example,
                mode,
            ):
                continue

            diagnosis = (
                diagnose_example(
                    example
                )
            )

            write_case(
                txt,
                diagnosis,
            )

            structured.write(
                json.dumps(
                    diagnosis,
                    ensure_ascii=False,
                )
                + "\n"
            )

            analyzed += 1

            if (
                limit is not None
                and analyzed >= limit
            ):
                break

    print()
    print(
        f"Analyzed cases : {analyzed}"
    )

    print(
        f"Readable output: {txt_output}"
    )

    print(
        f"JSONL output   : {jsonl_output}"
    )


from collections import Counter


def summarize_error_types(filename):

    counters = Counter()

    interaction_counters = Counter()

    first_error_propagation = Counter()

    input_path = Path(filename)

    with input_path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            example = json.loads(line)

            final_result = (
                example.get(
                    "final_result",
                    {}
                )
                or {}
            )

            # Only failed puzzles
            if (
                final_result.get(
                    "PUZZLE_ACCURACY"
                )
                != 0.0
            ):
                continue

            counters["TOTAL_ERRORS"] += 1

            diagnosis = diagnose_example(
                example
            )

            reasoning = (
                diagnosis.get(
                    "reasoning",
                    []
                )
                or []
            )

            first_error = (
                diagnosis.get(
                    "first_error"
                )
            )

            steps = (
                diagnosis
                .get(
                    "reasoning_analysis",
                    {}
                )
                .get(
                    "steps",
                    []
                )
            )

            # ================================================
            # 1. No reasoning / format failure
            # ================================================

            if not reasoning:

                counters[
                    "NO_REASONING_FORMAT_ERROR"
                ] += 1

                continue

            # ================================================
            # 2. First hard reasoning error detected
            # ================================================

            if first_error is not None:

                counters[
                    "FIRST_REASONING_ERROR_DETECTED"
                ] += 1

                status = (
                    first_error.get(
                        "prediction_status"
                    )
                )

                if status == "CONSISTENT":

                    first_error_propagation[
                        "PRESERVED_IN_FINAL"
                    ] += 1

                elif status == "INCONSISTENT":

                    first_error_propagation[
                        "NOT_PRESERVED_IN_FINAL"
                    ] += 1

                else:

                    first_error_propagation[
                        "PROPAGATION_UNVERIFIED"
                    ] += 1

                continue

            # ================================================
            # 3. Check whether any S-step was unverifiable
            # ================================================

            unverified = [
                step
                for step in steps
                if step.get(
                    "gt_status"
                ) == "UNVERIFIED"
            ]

            if unverified:

                counters[
                    "SYMBOLIC_UNVERIFIABLE"
                ] += 1

                continue

            # ================================================
            # 4. No hard error in explicit reasoning
            # ================================================

            counters[
                "NO_GT_CONTRADICTORY_REASONING_STEP"
            ] += 1

            prediction_statuses = [
                step.get(
                    "prediction_status"
                )
                for step in steps
            ]

            if "INCONSISTENT" in prediction_statuses:

                interaction_counters[
                    "FINAL_CONTRADICTS_REASONING"
                ] += 1

            elif "UNVERIFIED" in prediction_statuses:

                interaction_counters[
                    "FINAL_REASONING_RELATION_UNVERIFIED"
                ] += 1

            else:

                interaction_counters[
                    "FINAL_PRESERVES_REASONING_BUT_WRONG"
                ] += 1

    # ========================================================
    # PRINT
    # ========================================================

    total = counters[
        "TOTAL_ERRORS"
    ]

    print()
    print("=" * 80)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 80)

    print(
        f"\nTotal failed puzzles: {total}"
    )

    print(
        "\nFIRST-LEVEL CATEGORIES"
    )

    for key in [
        "FIRST_REASONING_ERROR_DETECTED",
        "NO_GT_CONTRADICTORY_REASONING_STEP",
        "NO_REASONING_FORMAT_ERROR",
        "SYMBOLIC_UNVERIFIABLE",
    ]:

        count = counters[key]

        percentage = (
            100.0 * count / total
            if total
            else 0
        )

        print(
            f"{key:40s} "
            f"{count:4d} "
            f"({percentage:6.2f}%)"
        )

    print(
        "\nREASONING-CORRECT / FINAL-WRONG SUBTYPES"
    )

    for key, count in (
        interaction_counters.items()
    ):

        print(
            f"{key:45s} "
            f"{count:4d}"
        )

    print(
        "\nFIRST REASONING ERROR PROPAGATION"
    )

    for key, count in (
        first_error_propagation.items()
    ):

        print(
            f"{key:45s} "
            f"{count:4d}"
        )

    print("=" * 80)



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    base_path = Path(
        "/home/asif/data3/Codes_QCRI/"
        "NON_FA2_models/"
        "Reasoning360_sys_B_v29/"
        "evaluation_results/"
        "mlxl_train_mlxl_test_1_parsed_v6a_MLXL/"
        "qwen34bthinking2507/"
        "jobid_276924"
    )

    file_name = (
        "jobid_276924_epoch_28_valid_feedback.jsonl"
    )

    filename = (
        base_path
        / file_name
    )

    # ========================================================
    # all
    # errors
    # correct
    # ========================================================

    mode = "errors"

    start = 1

    # None = everything
    limit = None

    analyze_file(
        filename=filename,
        mode=mode,
        start=start,
        limit=limit,
    )

    summarize_error_types(
        filename
    )