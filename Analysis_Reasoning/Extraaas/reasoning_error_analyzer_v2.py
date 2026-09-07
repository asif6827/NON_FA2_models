import json
import re
from pathlib import Path
from collections import defaultdict, Counter

# ============================================================
# Analysis output directory
# ============================================================

OUTPUT_DIR = Path(
    "/home/asif/data3/Codes_QCRI/NON_FA2_models/"
    "Analysis_Reasoning/Outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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
    # Ground truth
    # ========================================================

    def print_table(title, table):
        p(f"\n\n### {title} ###\n")

        if not isinstance(table, dict):
            p(str(table))
            return

        header = table.get("header", [])
        rows = table.get("rows", [])

        if not header or not rows:
            p(json.dumps(table, indent=2, ensure_ascii=False))
            return

        # Make a compact aligned text table.
        all_rows = [header] + rows
        widths = []

        for column_index in range(len(header)):
            widths.append(
                max(
                    len(str(row[column_index]))
                    if column_index < len(row)
                    else 0
                    for row in all_rows
                )
            )

        header_line = " | ".join(
            str(value).ljust(widths[i])
            for i, value in enumerate(header)
        )

        p(header_line)
        p("-" * len(header_line))

        for row in rows:
            p(
                " | ".join(
                    (
                        str(row[i])
                        if i < len(row)
                        else ""
                    ).ljust(widths[i])
                    for i in range(len(header))
                )
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
    # Ground truth and final prediction
    # ========================================================

    print_table(
        "GROUND TRUTH",
        diagnosis["ground_truth"],
    )

    print_table(
        "FINAL PREDICTION",
        diagnosis["final_prediction"],
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

def detect_structural_prediction_errors(
    ground_truth,
    prediction,
):
    """
    Detect duplicate/missing values in each prediction column.

    Zebra puzzle attributes should normally contain every
    attribute value exactly once.
    """

    errors = []

    if not isinstance(ground_truth, dict):
        return errors

    if not isinstance(prediction, dict):
        return errors

    gt_header = ground_truth.get(
        "header",
        []
    )

    gt_rows = ground_truth.get(
        "rows",
        []
    )

    pred_header = prediction.get(
        "header",
        []
    )

    pred_rows = prediction.get(
        "rows",
        []
    )

    if not gt_header or not pred_header:
        return errors

    if gt_header != pred_header:
        return errors

    for column_index, column_name in enumerate(
        gt_header
    ):

        # House itself is not an attribute
        if normalize(column_name) == "house":
            continue

        gt_values = []

        pred_values = []

        for row in gt_rows:

            if column_index < len(row):
                gt_values.append(
                    normalize(row[column_index])
                )

        for row in pred_rows:

            if column_index < len(row):
                pred_values.append(
                    normalize(row[column_index])
                )

        gt_set = set(gt_values)
        pred_set = set(pred_values)

        # ----------------------------------------------------
        # Duplicates
        # ----------------------------------------------------

        duplicates = sorted({
            value
            for value in pred_values
            if pred_values.count(value) > 1
        })

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        missing = sorted(
            gt_set - pred_set
        )

        # ----------------------------------------------------
        # Unexpected values
        # ----------------------------------------------------

        extra = sorted(
            pred_set - gt_set
        )

        if duplicates or missing or extra:

            errors.append(
                {
                    "column":
                        column_name,

                    "duplicates":
                        duplicates,

                    "missing":
                        missing,

                    "unexpected":
                        extra,
                }
            )

    return errors



def classify_error_case(example, diagnosis):
    """
    Assign one primary diagnostic category to a failed example.
    """

    reasoning = (
        diagnosis.get("reasoning", [])
        or []
    )

    first_error = diagnosis.get(
        "first_error"
    )

    steps = (
        diagnosis
        .get("reasoning_analysis", {})
        .get("steps", [])
    )

    final_prediction = diagnosis.get(
        "final_prediction"
    )

    # ========================================================
    # 1. No reasoning
    # ========================================================

    if not reasoning:
        return "NO_REASONING_FORMAT_ERROR"

    # ========================================================
    # 2. First erroneous symbolic reasoning step exists
    # ========================================================

    if first_error is not None:
        return "FIRST_REASONING_ERROR_DETECTED"

    # ========================================================
    # 3. Symbolic reasoning cannot be evaluated against GT
    # ========================================================

    if any(
        step.get("gt_status") == "UNVERIFIED"
        for step in steps
    ):
        return "SYMBOLIC_UNVERIFIABLE"

    # ========================================================
    # Everything below has:
    #
    # NO GT-contradictory explicit symbolic reasoning step
    # ========================================================

    prediction_statuses = [
        step.get("prediction_status")
        for step in steps
    ]

    # ========================================================
    # 4. Final answer directly contradicts explicit reasoning
    # ========================================================

    if "INCONSISTENT" in prediction_statuses:
        return "FINAL_CONTRADICTS_REASONING"

    # ========================================================
    # 5. Final prediction cannot be parsed
    # ========================================================

    if not isinstance(final_prediction, dict):
        return "FINAL_OUTPUT_UNPARSEABLE"

    # ========================================================
    # 6. Final table has duplicates / missing attribute values
    # ========================================================

    structural_errors = detect_structural_prediction_errors(
        diagnosis.get("ground_truth", {}),
        final_prediction,
    )

    if structural_errors:
        return "FINAL_STRUCTURAL_CONSTRAINT_VIOLATION"

    # ========================================================
    # 7. Reasoning preserved but final still wrong
    # ========================================================

    return "FINAL_PRESERVES_REASONING_BUT_WRONG"



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

    output_folder = OUTPUT_DIR

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



# ============================================================
# Final-output diagnostics for previously "unresolved" cases
# ============================================================

def is_parseable_final_table(prediction):
    """
    Return True only when processed_prediction contains a usable
    table with a non-empty header and rows.

    Examples classified as unparseable:
        None
        "WRONG OUTPUT FORMAT"
        {}
        {"header": [...]}           # no rows
        {"rows": [...]}             # no header
    """

    if not isinstance(prediction, dict):
        return False

    header = prediction.get("header")
    rows = prediction.get("rows")

    return (
        isinstance(header, list)
        and len(header) > 0
        and isinstance(rows, list)
        and len(rows) > 0
    )


def detect_final_table_structure_violations(
    ground_truth,
    prediction,
):
    """
    Detect Zebra-table structural violations in the final prediction.

    In a valid Zebra solution, every value of an attribute should
    occur exactly once.  Therefore, for each non-House column, the
    prediction should contain exactly the same multiset of values as
    the ground-truth column.

    This detects cases such as:

        Name:
            Bella
            Bella       <- duplicate
            Eric
            Alice

        while Timothy is missing.

    It also detects missing columns and row-count mismatches.

    IMPORTANT:
    A simple permutation of otherwise valid values is NOT considered
    a structural violation here.  That is a normal wrong assignment,
    not a uniqueness/completeness failure.
    """

    violations = []

    if not is_parseable_final_table(prediction):
        return violations

    if not isinstance(ground_truth, dict):
        return violations

    gt_header = ground_truth.get("header", [])
    gt_rows = ground_truth.get("rows", [])

    pred_header = prediction.get("header", [])
    pred_rows = prediction.get("rows", [])

    if not gt_header or not gt_rows:
        return violations

    gt_columns = {
        normalize(column): index
        for index, column in enumerate(gt_header)
    }

    pred_columns = {
        normalize(column): index
        for index, column in enumerate(pred_header)
    }

    # --------------------------------------------------------
    # Row-count mismatch
    # --------------------------------------------------------

    if len(pred_rows) != len(gt_rows):
        violations.append(
            {
                "type": "ROW_COUNT_MISMATCH",
                "expected": len(gt_rows),
                "actual": len(pred_rows),
            }
        )

    # --------------------------------------------------------
    # Check every attribute column
    # --------------------------------------------------------

    for column_name, gt_index in gt_columns.items():

        if column_name == "house":
            continue

        if column_name not in pred_columns:
            violations.append(
                {
                    "type": "MISSING_COLUMN",
                    "column": column_name,
                }
            )
            continue

        pred_index = pred_columns[column_name]

        expected_values = [
            normalize(row[gt_index])
            for row in gt_rows
            if gt_index < len(row)
        ]

        actual_values = [
            normalize(row[pred_index])
            for row in pred_rows
            if pred_index < len(row)
        ]

        expected_counter = Counter(
            expected_values
        )

        actual_counter = Counter(
            actual_values
        )

        duplicates = sorted(
            value
            for value, count
            in actual_counter.items()
            if count > 1
        )

        missing = sorted(
            list(
                (
                    expected_counter
                    - actual_counter
                ).elements()
            )
        )

        extra = sorted(
            list(
                (
                    actual_counter
                    - expected_counter
                ).elements()
            )
        )

        if duplicates or missing or extra:

            violations.append(
                {
                    "type":
                        "ATTRIBUTE_VALUE_SET_VIOLATION",

                    "column":
                        column_name,

                    "duplicates":
                        duplicates,

                    "missing":
                        missing,

                    "extra":
                        extra,
                }
            )

    return violations


def z3_solution_matches_ground_truth(example):
    """
    Determine whether the Z3 solution stored in z3_out matches GT.

    Preferred signal:
        z3_out["gt_solution_details"]["ok"]

    Fallback:
        compare z3_out["z3_solution"] directly against ground_truth.

    Returns:
        True   -> Z3 solution matches GT
        False  -> Z3 solution does not match GT
        None   -> no usable Z3-vs-GT information
    """

    z3_out = (
        example.get(
            "z3_out",
            {}
        )
        or {}
    )

    details = z3_out.get(
        "gt_solution_details"
    )

    if (
        isinstance(details, dict)
        and "ok" in details
    ):
        return bool(details["ok"])

    z3_solution = z3_out.get(
        "z3_solution"
    )

    ground_truth = example.get(
        "ground_truth"
    )

    if (
        isinstance(z3_solution, dict)
        and isinstance(ground_truth, dict)
    ):

        mismatches = compare_predictions(
            ground_truth,
            z3_solution,
        )

        return len(mismatches) == 0

    return None


def z3_status_label(status):
    if status is True:
        return "Z3_CORRECT"

    if status is False:
        return "Z3_WRONG"

    return "Z3_UNKNOWN"


def classify_unverified_reasoning_final_relation(
    example,
):
    """
    Replace the old generic:

        FINAL_REASONING_RELATION_UNVERIFIED

    with a concrete explanation whenever possible.

    Returns:
        category
        z3_status
        details
    """

    final_prediction = example.get(
        "processed_prediction"
    )

    z3_status = (
        z3_solution_matches_ground_truth(
            example
        )
    )

    # --------------------------------------------------------
    # Case A: no usable final table exists
    # --------------------------------------------------------

    if not is_parseable_final_table(
        final_prediction
    ):

        return (
            "FINAL_OUTPUT_UNPARSEABLE",
            z3_status,
            {
                "processed_prediction":
                    final_prediction,
            },
        )

    # --------------------------------------------------------
    # Case B: final table exists but violates basic Zebra
    # uniqueness/completeness constraints
    # --------------------------------------------------------

    structural_violations = (
        detect_final_table_structure_violations(
            example.get(
                "ground_truth",
                {},
            ),
            final_prediction,
        )
    )

    if structural_violations:

        return (
            "FINAL_STRUCTURAL_CONSTRAINT_VIOLATION",
            z3_status,
            {
                "violations":
                    structural_violations,
            },
        )

    # --------------------------------------------------------
    # Keep a safe fallback for future epochs/models.
    # We should not force every future case into one of the
    # two categories above.
    # --------------------------------------------------------

    return (
        "FINAL_RELATION_OTHER_UNRESOLVED",
        z3_status,
        {},
    )


def summarize_error_types(
    filename,
    show_case_ids=True,
):

    counters = Counter()

    interaction_counters = Counter()

    first_error_propagation = Counter()

    # Z3 breakdown specifically for reclassified
    # reasoning/final-output cases.
    z3_breakdown = Counter()

    # Optional case-level information for debugging.
    categorized_cases = defaultdict(list)

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

            counters[
                "TOTAL_ERRORS"
            ] += 1

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

            pid = example.get(
                "pid",
                "UNKNOWN"
            )

            # ================================================
            # 1. No reasoning / format failure
            # ================================================

            if not reasoning:

                counters[
                    "NO_REASONING_FORMAT_ERROR"
                ] += 1

                categorized_cases[
                    "NO_REASONING_FORMAT_ERROR"
                ].append(pid)

                continue

            # ================================================
            # 2. First hard reasoning error detected
            # ================================================

            if first_error is not None:

                counters[
                    "FIRST_REASONING_ERROR_DETECTED"
                ] += 1

                categorized_cases[
                    "FIRST_REASONING_ERROR_DETECTED"
                ].append(pid)

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
            # 3. Reasoning step itself cannot be checked
            #    against GT
            # ================================================

            unverified = [
                step
                for step in steps
                if step.get(
                    "gt_status"
                )
                == "UNVERIFIED"
            ]

            if unverified:

                counters[
                    "SYMBOLIC_UNVERIFIABLE"
                ] += 1

                categorized_cases[
                    "SYMBOLIC_UNVERIFIABLE"
                ].append(pid)

                continue

            # ================================================
            # 4. No GT-contradictory explicit reasoning step
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

            # ------------------------------------------------
            # A. Final directly contradicts at least one
            #    explicit GT-consistent reasoning statement.
            # ------------------------------------------------

            if (
                "INCONSISTENT"
                in prediction_statuses
            ):

                category = (
                    "FINAL_CONTRADICTS_REASONING"
                )

                interaction_counters[
                    category
                ] += 1

                categorized_cases[
                    category
                ].append(pid)

            # ------------------------------------------------
            # B. Current evaluator cannot directly compare
            #    one or more reasoning statements with final.
            #
            #    Instead of leaving these as one generic
            #    "UNVERIFIED" bucket, diagnose WHY.
            # ------------------------------------------------

            elif (
                "UNVERIFIED"
                in prediction_statuses
            ):

                (
                    category,
                    z3_status,
                    details,
                ) = (
                    classify_unverified_reasoning_final_relation(
                        example
                    )
                )

                interaction_counters[
                    category
                ] += 1

                categorized_cases[
                    category
                ].append(
                    {
                        "pid": pid,
                        "z3_status":
                            z3_status_label(
                                z3_status
                            ),
                        "details":
                            details,
                    }
                )

                z3_breakdown[
                    (
                        category,
                        z3_status_label(
                            z3_status
                        ),
                    )
                ] += 1

            # ------------------------------------------------
            # C. Every evaluable explicit reasoning statement
            #    is also respected by the final answer, but
            #    the final puzzle is nevertheless wrong.
            # ------------------------------------------------

            else:

                category = (
                    "FINAL_PRESERVES_REASONING_BUT_WRONG"
                )

                interaction_counters[
                    category
                ] += 1

                categorized_cases[
                    category
                ].append(pid)

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    total = counters[
        "TOTAL_ERRORS"
    ]

    print()
    print("=" * 90)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 90)

    print(
        f"\nTotal failed puzzles: {total}"
    )

    # ========================================================
    # First-level categories
    # ========================================================

    print(
        "\nFIRST-LEVEL CATEGORIES"
    )

    first_level_keys = [
        "FIRST_REASONING_ERROR_DETECTED",
        "NO_GT_CONTRADICTORY_REASONING_STEP",
        "NO_REASONING_FORMAT_ERROR",
        "SYMBOLIC_UNVERIFIABLE",
    ]

    for key in first_level_keys:

        count = counters[key]

        percentage = (
            100.0 * count / total
            if total
            else 0.0
        )

        print(
            f"{key:45s} "
            f"{count:4d} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # No-GT-error reasoning / final-answer subtypes
    # ========================================================

    print(
        "\nNO-GT-CONTRADICTORY-REASONING / "
        "FINAL-WRONG SUBTYPES"
    )

    subtype_keys = [
        "FINAL_CONTRADICTS_REASONING",
        "FINAL_OUTPUT_UNPARSEABLE",
        "FINAL_STRUCTURAL_CONSTRAINT_VIOLATION",
        "FINAL_PRESERVES_REASONING_BUT_WRONG",
        "FINAL_RELATION_OTHER_UNRESOLVED",
    ]

    for key in subtype_keys:

        count = interaction_counters[
            key
        ]

        # Do not clutter output with a zero fallback.
        if (
            key
            == "FINAL_RELATION_OTHER_UNRESOLVED"
            and count == 0
        ):
            continue

        print(
            f"{key:50s} "
            f"{count:4d}"
        )

        # ----------------------------------------------------
        # Print Z3 breakdown for categories that came from
        # the old "UNVERIFIED" bucket.
        # ----------------------------------------------------

        if key in {
            "FINAL_OUTPUT_UNPARSEABLE",
            "FINAL_STRUCTURAL_CONSTRAINT_VIOLATION",
            "FINAL_RELATION_OTHER_UNRESOLVED",
        }:

            for z3_label in [
                "Z3_CORRECT",
                "Z3_WRONG",
                "Z3_UNKNOWN",
            ]:

                z3_count = (
                    z3_breakdown[
                        (
                            key,
                            z3_label,
                        )
                    ]
                )

                if z3_count:

                    print(
                        f"    └── {z3_label:41s} "
                        f"{z3_count:4d}"
                    )

    # ========================================================
    # First reasoning error propagation
    # ========================================================

    print(
        "\nFIRST REASONING ERROR PROPAGATION"
    )

    propagation_keys = [
        "PRESERVED_IN_FINAL",
        "NOT_PRESERVED_IN_FINAL",
        "PROPAGATION_UNVERIFIED",
    ]

    for key in propagation_keys:

        print(
            f"{key:50s} "
            f"{first_error_propagation[key]:4d}"
        )

    # ========================================================
    # Sanity checks
    # ========================================================

    first_level_sum = sum(
        counters[key]
        for key in first_level_keys
    )

    subtype_sum = sum(
        interaction_counters[key]
        for key in subtype_keys
    )

    print(
        "\nSANITY CHECKS"
    )

    print(
        f"First-level sum: "
        f"{first_level_sum} / {total}"
    )

    print(
        "No-GT-contradictory subtype sum: "
        f"{subtype_sum} / "
        f"{counters['NO_GT_CONTRADICTORY_REASONING_STEP']}"
    )

    # ========================================================
    # Optional case IDs for the newly reclassified groups
    # ========================================================

    if show_case_ids:

        print(
            "\nRECLASSIFIED CASE DETAILS"
        )

        for key in [
            "FINAL_OUTPUT_UNPARSEABLE",
            "FINAL_STRUCTURAL_CONSTRAINT_VIOLATION",
            "FINAL_RELATION_OTHER_UNRESOLVED",
        ]:

            cases = categorized_cases[
                key
            ]

            if not cases:
                continue

            print(
                f"\n{key} ({len(cases)} cases)"
            )

            for case in cases:

                if isinstance(case, dict):

                    print(
                        f"  {case['pid']} "
                        f"[{case['z3_status']}]"
                    )

                    violations = (
                        case
                        .get(
                            "details",
                            {}
                        )
                        .get(
                            "violations",
                            []
                        )
                    )

                    for violation in violations:

                        if (
                            violation.get(
                                "type"
                            )
                            ==
                            "ATTRIBUTE_VALUE_SET_VIOLATION"
                        ):

                            print(
                                "      "
                                f"{violation['column']}: "
                                f"duplicates="
                                f"{violation['duplicates']}, "
                                f"missing="
                                f"{violation['missing']}, "
                                f"extra="
                                f"{violation['extra']}"
                            )

                        else:

                            print(
                                "      "
                                f"{violation}"
                            )

                else:

                    print(
                        f"  {case}"
                    )

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    base_path = Path(
        "/home/asif/data3/Codes_QCRI/"
        "NON_FA2_models/"
        "Reasoning360_sys_B_v29_FR/"
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