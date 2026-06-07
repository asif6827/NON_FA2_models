import os
import sys
import json
import datasets
import random
import argparse
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset

SOLUTION_SYSTEM_PROMPT = """
You are an expert logic puzzle solver.

You are given:
(i) one logic puzzle_text written in plain English,
(ii) solution_header that lists the attribute names used in the puzzle, and
(iii) a dictionary of attribute_values specifying the complete and exclusive set of allowed values for each attribute.

All values appearing in syntactic_clues, reasoning, and the final solution MUST be drawn from attribute_values and interpreted as entity tokens representing unknown house positions.

Your task is to construct a fully consistent, solver-verifiable solution by generating the following FIVE fields:
1) n_houses — the total number of houses in the puzzle.
2) attribute_values — returned exactly as given, without modification.
3) syntactic_clues — a normalized, Z3-style textual encoding of each clue.
4) reasoning — interleaved reasoning consisting of natural-language explanations and syntactic (solver-checkable) deduction steps.
5) solution — the final house-by-house assignment derived exclusively from syntactic_clues, and syntactic reasoning steps (S1..Sk).


You MUST return the result STRICTLY as a single valid JSON object wrapped inside:
<answer>...</answer>

No additional text, commentary, or formatting outside the <answer> block is permitted.


================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Do NOT include extra text, markdown, explanations, or code fences.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly FIVE top-level keys, spelled EXACTLY:
    "n_houses",
    "attribute_values",
    "syntactic_clues",
    "reasoning",
    "solution"
- Do NOT add any other keys.

================================================================================
NORMALIZATION RULES
================================================================================
- Use underscores instead of spaces in VALUES (e.g., grilled_cheese, very_short).
- Attribute names MUST match the solution_header exactly (case-sensitive), e.g., Name, Animal, Occupation, Sport, Height, etc.
- House numbers are integers 1..N.
- Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6, etc.
- Do not invent values. Every value must be mapped to its canonical token (in attribute_values) and selected from the list of allowed attribute_values (after normalization).
 - Example: If puzzle says “september” and attribute_values contains "sept", output "sept" (not september).
 - Example: If puzzle says “sept” and attribute_values contains "september", output "september"
- If the clue mentions a bare person name (e.g., "Arnold"), treat it as Name=Arnold.
- If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching token in attribute_values.

================================================================================
1) DOMAIN OUTPUT (MANDATORY)
================================================================================
- "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
attribute_values immutability rule:
- The "attribute_values" object MUST be returned exactly as provided in the input.
- It must be identical:
  - Same attribute keys
  - Same ordering of keys
  - Same ordering of values within each list
  - Same casing and spelling
- Do NOT normalize, rename, reorder, add, or remove anything in "attribute_values".
- Normalization rules apply ONLY to syntactic_clues, reasoning, and solution — NOT to attribute_values.

================================================================================
2) syntactic_clues (MANDATORY, TEXTUAL CONSTRAINTS — NOT PREDICATES)
================================================================================
We do NOT use predicate-style DSL for clues.
Instead, each clue MUST be rewritten as a single-line *syntactic constraint statement* in a Z3-like textual form.

Rules:
- "syntactic_clues" MUST be a list of strings.
- For each clue, the selected tokens must be mapped to one of the values defined in attribute_values.
 - Example: If the clue says “sept” and attribute_values contains "september", use "september"; if attribute_values contains "sept", use "sept".
- There MUST be exactly one entry per clue, in the same order as the clues.
- Each entry MUST be exactly 1 line and end with a period.
- Each entry MUST start with the clue id prefix: "C<i>: ".
- Use ONLY these syntactic operators in the clue text:
    ==   (same house / equivalence)
    !=   (not same house)
    <    (somewhere left of)
    >    (somewhere right of)
    + k == (k is a positive integer, e.g., 1 for immediately left, 2 for one house between, 3 for two houses between)
    == H  (fixed house index, where H is an integer)
- Use bare normalized tokens (no quotes) for values (e.g., Arnold, engineer, very_short).
- When a clue states a specific house like "in the fifth house", encode as: <token> == 5
  Example: "The lawyer is in the fifth house." -> "C9: lawyer == 5."
- When a clue states "directly left of", encode as: A + 1 == B
  Example: "baseball is directly left of engineer" -> "C12: baseball + 1 == engineer."
- When a clue states "one house between", encode as: A + 2 == B
  Example: "There is one house between Eric and the bird keeper" -> "C12: Eric + 2 == bird_keeper."
  Example: "There is one house between Arnold and Peter" -> "C12: Arnold + 2 == Peter."
- When a clue states "two houses between", encode as: A + 3 == B
  Example: "There are two houses between Eric and Arnold" -> "C12: Eric + 3 == Arnold."
- When a clue states "person who has", encode as: A == B
  Example: "The person whose mother's name is Holly is the person who has black hair" -> "C12: Holly == black."
- When a clue states "one house between the person who has", encode as: A + 2 == B
  Example: "There is one house between the person who has black hair and Eric" -> "C12: black + 2 == Eric."
- When a clue states "next to each other", encode it as: Or(A == B + 1, A == B - 1)
  Example: "The person who prefers city breaks and Alice are next to each other" -> C12: "Or(city_breaks == Alice + 1, city_breaks == Alice - 1)."
- When a clue states "somewhere to the left of", encode as: A < B
- When a clue states "somewhere to the right of", encode as: A > B
- When a clue states "X is the Y", encode as: X == Y

IMPORTANT:
- The goal is to produce constraints that resemble:
  s.add(<left> <op> <right>)
  but you must NOT write "s.add(...)".
  Only output the inner constraint as text.

================================================================================
3) reasoning (MANDATORY — INTERLEAVED NATURAL + SYNTACTIC)
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly 1 sentence and end with a period.
- Reasoning MUST be interleaved:
    Odd-numbered entries: Natural-language reasoning.
    Even-numbered entries: Syntactic reasoning step (Z3-like statement).
- Natural-language entries should explain the deduction in plain English.
- Syntactic entries should encode the *newly deduced fact* as a Z3-like statement.
- Tokens in Syntactic entries should encode the *mapped* to values in "attribute_values".

Syntactic entry format:
- Every syntactic entry MUST start with "S<k>: " and MUST end with a period.
- <k> starts at 1 and increments by 1 for each syntactic step only (S1, S2, S3, ...).
- The syntactic constraint MUST be solver-verifiable and may use ONLY:
  ==, !=, <, >, + d ==, Not(...), And(...), Or(...)

- Each syntactic step MUST be written in the exact form: S<k>

  Atomic operators:
    ==        (same house / equivalence)
    !=        (not the same house)
    <         (somewhere to the left of)
    >         (somewhere to the right of)
    + d ==    (directed distance; d is a positive integer)
    == H      (fixed house index; H is an integer in 1..n_houses)

  Boolean operators:
    Not(e)    (negation of a single atomic expression)
    And(e1, e2, ..., en)
    Or(e1, e2, ..., en)

- Boolean operators may ONLY be applied to valid atomic expressions.
- Nested Boolean expressions are allowed but MUST remain solver-verifiable.

Examples of valid INTERLEVED reasoning steps:
    The engineer is assigned to house 2.
    S1: engineer == 2.

    Since the engineer occupies house 2, the dog cannot also be in house 2.
    S2: dog != 2.

    The cat is immediately to the left of the coffee, so the cat’s house index plus one equals the coffee’s house index.
    S3: cat + 1 == coffee.

    The green house appears somewhere to the left of the white house.
    S4: green < white.

    The dog is not in the first house.
    S5: Not(dog == 1).

    The cat cannot be in house 1 or house 3.
    S6: And(cat != 1, cat != 3).

    The milk is located either in house 1 or in house 5.
    S7: Or(milk == 1, milk == 5).

Logical validity requirement:
- Every syntactic step MUST be logically entailed by the syntactic_clues plus any earlier syntactic steps.
- Do NOT output syntactic steps that merely restate a clue unless they are required as part of the deduction chain.

================================================================================
4) solution (MANDATORY TABLE)
================================================================================
- "solution" MUST be in tabular form with:
  - "header": a list of column names
  - "rows": a list of rows, each row being a list of strings matching the header order
- The header MUST include "House" and then all attribute columns from the puzzle text.
- The rows MUST list houses in increasing order from 1..N.
- All solution values MUST be normalized with underscores.
"""

FEWSHOT_PUZZLE = """
There are 3 houses, numbered 1 to 3 from left to right. Each house is occupied by a different person.
Each house has a unique attribute for each of the following characteristics:

- Each person has a unique name: Peter, Eric, Arnold
- The people like unique colors: red, white, yellow
- The people have children named: Fred, Meredith, Bella

Clues:
1. Arnold is the person whose favorite color is red.
2. The person whose child is named Fred is somewhere to the left of Eric.
3. The person whose favorite color is red is in the second house.
4. The person whose child is named Bella is in the first house.
5. The person who loves white is the person whose child is named Meredith.
"""

FEWSHOT_HEADER = ["House", "Name", "Color", "Children"]

FEWSHOT_ATTRIBUTE_VALUES = {
    "Name": ["Peter", "Eric", "Arnold"],
    "Color": ["red", "white", "yellow"],
    "Children": ["Fred", "Meredith", "Bella"]
}

FEWSHOT_ASSISTANT_ANSWER = """
<answer> {
  "n_houses": 3,
  "attribute_values": {
  "Name": ["Peter", "Eric", "Arnold"],
  "Color": ["red", "white", "yellow"],
  "Children": ["Fred", "Meredith", "Bella"]
  },
  "syntactic_clues": [
    "C1: Arnold == red.",
    "C2: Fred < Eric.",
    "C3: red == 2.",
    "C4: Bella == 1.",
    "C5: white == Meredith."
  ],
  "reasoning": [
    "Clue 3 immediately anchors the red favorite color in the second house, which is a very strong positional fact to start from.",
    "S1: red == 2.",
    "Clue 1 then ties Arnold directly to the red color, so Arnold must be in that same second house.",
    "S2: Arnold == red.",
    "Putting those two together, Arnold is fixed in house 2. At this point, house 2 is completely identified as “Arnold’s house,” and we know it has the red color.",
    "S3: Arnold == 2.",
    "Clue 4 gives us another concrete placement: the child Bella is in the first house. So whatever person lives in house 1, their child must be Bella.",
    "S4: Bella == 1.",
    "So far, we know: House 1 has child Bella, House 2 has Arnold and the color red, House 3 is still entirely open. Now we look at Clue 2, which introduces a relative ordering: the person whose child is Fred is somewhere to the left of Eric. This doesn’t give a house yet, but it constrains the ordering.",
    "S5: Fred < Eric.",
    "Since houses are only 1 through 3, Eric cannot be in the first house (there would be nothing to the left of him). That means Eric must be in house 2 or house 3.",
    "S6: Or(Eric == 2, Eric == 3).",
    "But we already know Arnold occupies house 2, and all people are distinct. So Eric cannot be in house 2 and must therefore be in house 3.",
    "S7: Eric == 3.",
    "This is a good point to pause and take stock again. House 1: unknown person, child Bella. House 2: Arnold, red. House 3: Eric. Now, going back to the same ordering constraint (Fred < Eric), if Eric is in house 3, then Fred must be in house 1 or house 2.",
    "S8: Or(Fred == 1, Fred == 2).",
    "But we already know the child in house 1 is Bella, and children are unique. So Fred cannot be in house 1. That forces Fred into house 2.",
    "S9: Fred == 2.",
    "This tells us that Arnold, who is in house 2, is also the parent of Fred. At this stage, all people except Peter are placed: Arnold is in house 2 and Eric is in house 3. Since each house has exactly one person, Peter must be in the remaining house, house 1.",
    "S10: Peter == 1.",
    "Let’s summarize again. House 1: Peter, child Bella. House 2: Arnold, red, child Fred. House 3: Eric. Now consider the children again. Bella is in house 1 and Fred is in house 2, so the only remaining child, Meredith, must be in house 3.",
    "S11: Meredith == 3.",
    "Clue 5 connects the color white to Meredith’s parent, meaning the white color must be in the same house as Meredith.",
    "S12: white == Meredith.",
    "Since Meredith is in house 3, white must be in house 3 as well.",
    "S13: white == 3.",
    "At this point, two colors are fixed: red in house 2 and white in house 3. Colors are unique, so the only remaining color, yellow, must belong to house 1.",
    "S14: yellow == 1.",
    "With that, everything is now determined: House 1: Peter, yellow, child Bella. House 2: Arnold, red, child Fred. House 3: Eric, white, child Meredith. All clues are satisfied, and no attributes remain unassigned."
  ],
  "solution": {
    "header": ["House", "Name", "Color", "Children"],
    "rows": [
      ["1", "Peter", "yellow", "Bella"],
      ["2", "Arnold", "red", "Fred"],
      ["3", "Eric", "white", "Meredith"]
    ]
  }
}</answer>
"""

PUZZLE_USER_PROMPT_PHI = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

puzzle = {puzzle}

solution_header = {solution_header}

attribute_values = {attribute_values}

Solve the puzzle above and provide n_houses, attribute_values, syntactic_clues, reasoning, and solution.

Your response MUST begin with the exact characters:
<answer>{{

Your JSON MUST contain the fields in this exact order:
1. "n_houses"
2. "attribute_values"
3. "syntactic_clues"
4. "reasoning"
5. "solution"

The "reasoning" list must be concise and must contain at most 12 strings.
After the final reasoning string, immediately write the "solution" field.
The "solution" field must be the final top-level key and must not be omitted.

After the complete solution field, close the JSON object and end with:
}}</answer>

Return only one complete <answer>...</answer> block with no additional text.
"""


def serialize_phi_messages(messages):
    prompt = ""

    for message in messages:
        role = message["role"]
        content = message["content"].strip()

        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")

        prompt += f"<|{role}|>\n{content}\n<|end|>\n"

    prompt += "<|assistant|>"
    return prompt


def extract_clues_from_puzzle(puzzle_text):
    """Extract clues from the puzzle text."""
    if "## Clues:" in puzzle_text:
        clues_part = puzzle_text.split("## Clues:")[1]
        # Extract each clue line
        clues = []
        for line in clues_part.splitlines():
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                # Remove the numbering and keep the clue text
                clue_text = line.split(".", 1)[1].strip()
                clues.append(clue_text)
        return clues
    else:
        return []


def attribute_values_from_solution(solution: dict) -> dict:
    """
    Convert a solution table into attribute_values:
      solution = {"header": [...], "rows": [[...], ...]}
    Returns:
      {"Name": [...], "CarModel": [...], ...}   (excludes "House")
    """
    header = solution.get("header", [])
    rows = solution.get("rows", [])

    # column indices, skipping "House"
    col_indices = [(i, col) for i, col in enumerate(header) if col != "House"]

    values = {col: [] for _, col in col_indices}
    seen = {col: set() for _, col in col_indices}

    for row in rows:
        if not isinstance(row, list):
            continue
        for i, col in col_indices:
            if i >= len(row):
                continue
            v = "_".join(row[i].split(" "))
            # v = row[i]
            if v not in seen[col]:
                seen[col].add(v)
                values[col].append(v)

    for key in values:
        random.shuffle(values[key])
    return values


def normalize_solution_grid(solution):
    header = solution["header"]
    rows = []

    for row in solution["rows"]:
        norm_row = []
        for col, value in zip(header, row):
            if col == "House":
                norm_row.append(str(value))
            else:
                norm_row.append("_".join(str(value).split()))
        rows.append(norm_row)

    return {
        "header": header,
        "rows": rows,
    }


def make_map_fn_1_shot(split, data_source):
    def process_fn_1_shot(example, idx):
        final_grid = normalize_solution_grid(example["solution"])
        clues = extract_clues_from_puzzle(puzzle_text=example["puzzle"])

        # Important: compute once so the same attribute_values are used in the target prompt.
        target_attribute_values = attribute_values_from_solution(example["solution"])
        target_solution_header = final_grid["header"]

        messages = [
            {
                "role": "system",
                "content": SOLUTION_SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": PUZZLE_USER_PROMPT_PHI.format(
                    puzzle=FEWSHOT_PUZZLE.strip(),
                    solution_header=json.dumps(FEWSHOT_HEADER, ensure_ascii=False),
                    attribute_values=json.dumps(FEWSHOT_ATTRIBUTE_VALUES, ensure_ascii=False),
                ).strip(),
            },
            {
                "role": "assistant",
                "content": FEWSHOT_ASSISTANT_ANSWER.strip(),
            },
            {
                "role": "user",
                "content": PUZZLE_USER_PROMPT_PHI.format(
                    puzzle=example["puzzle"],
                    solution_header=json.dumps(target_solution_header, ensure_ascii=False),
                    attribute_values=json.dumps(target_attribute_values, ensure_ascii=False),
                ).strip(),
            },
        ]

        phi_prompt = serialize_phi_messages(messages)

        data = {
            "data_source": data_source,
            "prompt": phi_prompt,
            "raw_prompt": phi_prompt,
            "ability": "logical_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": final_grid,
            },
            "apply_chat_template": False,
            "extra_info": {
                "id": example["id"] if "id" in example else str(idx),
                "split": split,
                "clues": clues,
            },
        }

        if idx != 0:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            print("\n\n")

        return data

    return process_fn_1_shot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/asif/data3/HF_cache/ZebraLogic/', help='Path to json file')
    parser.add_argument('--data_setting', default='mlxl_train_mlxl_test', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/ZebraPuzzle_to_guru_parsed_v6a_MLXL', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.3, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.7, help='Proportion of data for test set')
    parser.add_argument('--data_source_train', default='our_zebra_puzzle_new_reward_phi4', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_zebra_puzzle_new_reward_test_phi4', help='Name of data source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

    random.seed(args.seed)

    if args.data_setting == 'small_train_med_test':
        args.train_file = os.path.join(args.data_path, 'Zebra_Puzzle_small_320.json')
        args.test_file = os.path.join(args.data_path, 'Zebra_Puzzle_medium_280.json')
    elif args.data_setting == 'med_train_small_test':
        args.train_file = os.path.join(args.data_path, 'Zebra_Puzzle_medium_280.json')
        args.test_file = os.path.join(args.data_path, 'Zebra_Puzzle_small_320.json')
    elif args.data_setting == 'med_train_large_test':
        args.train_file = os.path.join(args.data_path, 'Zebra_Puzzle_medium_280.json')
        args.test_file = os.path.join(args.data_path, 'Zebra_Puzzle_large_200.json')
    elif args.data_setting == 'med_train_med_test':
        args.data_file = os.path.join(args.data_path, 'Zebra_Puzzle_medium_280.json')
    elif args.data_setting == 'mlxl_train_mlxl_test':
        args.data_file = os.path.join(args.data_path, 'Zebra_Puzzle_complete_1000.json')
    else:
        raise ValueError('Invalid data_setting')
    args.output_dir = os.path.join(args.output_dir, args.data_setting)

    if args.data_setting == 'mlxl_train_mlxl_test':
        # Load dataset from JSON or Parquet based on file extension
        file_extension = os.path.splitext(args.data_file)[1].lower()
        if file_extension in ['.json', '.jsonl']:
            dataset = datasets.load_dataset('json', data_files=args.data_file)['train']
        elif file_extension == '.parquet':
            dataset = datasets.load_dataset('parquet', data_files=args.data_file)['train']
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only JSON, JSONL, and Parquet are supported.")

        train_indices, test_indices = train_test_split(
            range(len(dataset)),
            train_size=args.train_size,
            test_size=args.test_size,
            random_state=args.seed
        )

        # Create train and test datasets
        train_dataset = dataset.select(train_indices)
        test_dataset = dataset.select(test_indices)

        # Transform dataset
        process_train_fn = make_map_fn_1_shot('train', args.data_source_train)
        train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)

        process_test_fn = make_map_fn_1_shot('test', args.data_source_test)
        test_dataset = test_dataset.map(function=process_test_fn, with_indices=True)

        if args.train_size + args.test_size > 1.0:
            raise ValueError(f"The sum of train_size ({args.train_size}) and test_size ({args.test_size}) cannot exceed 1.0")

        # Split dataset into train and test

    else:
        # Load dataset from JSON or Parquet based on file extension
        file_extension = os.path.splitext(args.train_file)[1].lower()
        if file_extension in ['.json', '.jsonl']:
            train_dataset = datasets.load_dataset('json', data_files=args.train_file)['train']
        elif file_extension == '.parquet':
            train_dataset = datasets.load_dataset('parquet', data_files=args.train_file)['train']
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only JSON, JSONL, and Parquet are supported.")

        file_extension = os.path.splitext(args.test_file)[1].lower()
        if file_extension in ['.json', '.jsonl']:
            test_dataset = datasets.load_dataset('json', data_files=args.test_file)['train']
        elif file_extension == '.parquet':
            test_dataset = datasets.load_dataset('parquet', data_files=args.test_file)['train']
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only JSON, JSONL, and Parquet are supported.")

        # Transform dataset
        process_train_fn = make_map_fn_1_shot('train', args.data_source_train)
        train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)

        process_test_fn = make_map_fn_1_shot('test', args.data_source_test)
        test_dataset = test_dataset.map(function=process_test_fn, with_indices=True)

    # Store the original training dataset size
    original_train_size = len(train_dataset)

    # Sample the training dataset if needed
    train_dataset = sample_dataset(train_dataset, args.train_sample_size)

    # Create output directories
    train_output_dir = os.path.join(args.output_dir, "train")
    test_output_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Save train dataset
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=f"logic_{args.data_source_train}",
        sample_size=args.train_sample_size if args.train_sample_size else len(train_dataset)
    )

    # Save test dataset
    test_output_path = save_dataset(
        dataset=test_dataset,
        output_dir=test_output_dir,
        filename_prefix=f"logic_{args.data_source_test}",
        sample_size=len(test_dataset)
    )

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir)
            copy(src=args.output_dir, dst=args.hdfs_dir)
            print(f"Data copied to HDFS: {args.hdfs_dir}")
        except ImportError:
            print("HDFS utilities not available. Install verl package for HDFS support.")

    print(f"Done! \n"
          f"Train data saved to {train_output_path}\n"
          f"Test data saved to {test_output_path}")
    print(f"Original train set size: {original_train_size} examples")
    print(f"Final train set size: {len(train_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")