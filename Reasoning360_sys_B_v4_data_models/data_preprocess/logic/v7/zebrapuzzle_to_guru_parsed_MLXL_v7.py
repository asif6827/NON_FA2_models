import os
import sys
import datasets
import random
import argparse
import json
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset

SOLUTION_PROMPT_1_SHOT_SYS = """
You are an expert logic puzzle solver.

You are given:
(i) one logic puzzle_text written in plain English,
(ii) solution_header that lists the attribute names used in the puzzle, and
(iii) a dictionary of attribute_values specifying the complete and exclusive set of allowed values for each attribute.

All ENTITY TOKENS appearing in syntactic_clues, syntactic reasoning values under S_i keys, PA_i resolved cells, and the final solution MUST be drawn from attribute_values and interpreted as entity tokens representing house positions. Natural-language reasoning values under NL_i keys may use ordinary English, and PA_i may use "?" only for unresolved attribute cells.

Your task is to construct a fully consistent, solver-verifiable solution by generating the following FIVE fields:
1) n_houses — the total number of houses in the puzzle.
2) attribute_values — returned exactly as given, without modification.
3) syntactic_clues — a normalized, Z3-style textual encoding of each clue.
4) reasoning — a JSON object whose EMITTED KEY ORDER follows strictly interleaved natural-language keys NL_i and syntactic keys S_i, with occasional structured Partial Answer keys PA_k inserted after meaningful reasoning milestones.
5) solution — the final house-by-house assignment derived from the puzzle clues and the complete interleaved reasoning trajectory.


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
    + k == (directed distance; k is a positive integer)
    == H  (fixed house index, where H is an integer)
    Not(...)
    And(...)
    Or(...)
- Use bare normalized tokens (no quotes) for values (e.g., Arnold, engineer, very_short).
- When a clue states a specific house like "in the fifth house", encode as: <token> == 5
  Example: "The lawyer is in the fifth house." -> "C9: lawyer == 5."
- When a clue states "directly left of", encode as: A + 1 == B
  Example: "baseball is directly left of engineer" -> "C12: baseball + 1 == engineer."
- When a clue states "one house between" WITHOUT specifying which entity is left/right, encode symmetrically as: Or(A + 2 == B, B + 2 == A)
  Example: "There is one house between Eric and the bird keeper" -> "C12: Or(Eric + 2 == bird_keeper, bird_keeper + 2 == Eric)."
  If the clue explicitly specifies direction, use the corresponding directed form A + 2 == B.
- When a clue states "two houses between" WITHOUT specifying which entity is left/right, encode symmetrically as: Or(A + 3 == B, B + 3 == A)
  Example: "There are two houses between Eric and Arnold" -> "C12: Or(Eric + 3 == Arnold, Arnold + 3 == Eric)."
  If the clue explicitly specifies direction, use the corresponding directed form A + 3 == B.
- When a clue states "person who has", encode as: A == B
  Example: "The person whose mother's name is Holly is the person who has black hair" -> "C12: Holly == black."
- When a clue states "one house between the person who has" WITHOUT direction, encode symmetrically as: Or(A + 2 == B, B + 2 == A)
  Example: "There is one house between the person who has black hair and Eric" -> "C12: Or(black + 2 == Eric, Eric + 2 == black)."
- When a clue states "next to each other", encode it symmetrically using only the allowed +1 form: Or(A + 1 == B, B + 1 == A)
  Example: "The person who prefers city breaks and Alice are next to each other" -> "C12: Or(city_breaks + 1 == Alice, Alice + 1 == city_breaks)."
- When a clue states "somewhere to the left of", encode as: A < B
- When a clue states "somewhere to the right of", encode as: A > B
- When a clue states "X is the Y", encode as: X == Y

IMPORTANT:
- The goal is to produce constraints that resemble:
  s.add(<left> <op> <right>)
  but you must NOT write "s.add(...)".
  Only output the inner constraint as text.

================================================================================
3) reasoning (MANDATORY — EMITTED KEY ORDER: NL_i + S_i + STRUCTURED PA_i)
================================================================================
- "reasoning" MUST be a JSON object, NOT a list.
- JSON objects are formally unordered, so this prompt explicitly constrains the SERIALIZED/EMITTED textual key order; the keys MUST be written in the trajectory order specified below.
- The keys of "reasoning" MUST consist ONLY of:
    "NL1", "NL2", ..., "NLz"
    "S1",  "S2",  ..., "Sz"
    and optional
    "PA1", "PA2", ..., "PAk"
- Do NOT add any other keys inside "reasoning".

Required NL/S structure:
- Every reasoning deduction MUST be represented by a STRICT key pair:
    "NL<i>": "<natural-language deduction>"
    immediately followed by
    "S<i>": "<solver-checkable syntactic deduction>"
- NL and S numbering MUST start at 1 and increase consecutively with no gaps.
- NL and S indices MUST match exactly:
    NL1 -> S1
    NL2 -> S2
    ...
    NLz -> Sz
- The textual key order in the emitted JSON object MUST preserve the reasoning trajectory.
- Do NOT output two NL keys consecutively.
- Do NOT output two S keys consecutively.
- Do NOT place a PA key between NL<i> and S<i>.

Partial Answer insertion:
- OPTIONAL Partial Answer keys PA_k may be inserted ONLY after a completed NL<i>/S<i> pair.
- PA numbering MUST start at 1 and increase consecutively with no gaps.
- A PA does NOT replace an NL/S pair and does NOT itself introduce a new deduction.

Required overall pattern:
    "NL1": ...,
    "S1": ...,
    "NL2": ...,
    "S2": ...,
    ...
    "PA1": {...},
    "NLi": ...,
    "Si": ...,
    ...
    "PA2": {...},
    ...
    "NLz": ...,
    "Sz": ...

Example valid structure:
    "reasoning": {
      "NL1": "Clues 1 and 3 together place Arnold in house 2.",
      "S1": "Arnold == 2.",
      "NL2": "Because Fred is somewhere to the left of Eric, Eric cannot occupy house 1.",
      "S2": "Eric != 1.",
      "PA1": {
        "header": ["House", "Name", "Color", "Children"],
        "rows": [
          ["1", "?", "?", "Bella"],
          ["2", "Arnold", "red", "?"],
          ["3", "?", "?", "?"]
        ]
      },
      "NL3": "Arnold already occupies house 2, so Eric must occupy house 3.",
      "S3": "Eric == 3."
    }

Invalid structures include:
- NL1 followed by NL2 without S1.
- S1 without an immediately preceding NL1.
- PA1 between NL1 and S1.
- PA1 before the first completed NL/S pair.
- Missing or skipped indices such as NL1, S1, NL3, S3.
- A PA represented as a quoted/escaped JSON string instead of a JSON object.

Natural-language value rules:
- Each NL<i> VALUE MUST be a single natural-language sentence ending with a period.
- Each NL<i> should describe one logically meaningful deduction.
- Each NL<i> must explain why the immediately following S<i> deduction is justified.
- Avoid state summaries inside NL<i>; use PA_k for intermediate state snapshots.

Syntactic value rules:
- Each S<i> VALUE MUST contain ONLY the syntactic constraint itself.
- Do NOT repeat the key inside the value.
  Correct:
      "S1": "Arnold == 2."
  Incorrect:
      "S1": "S1: Arnold == 2."
- Each S<i> value MUST end with a period.
- Each S<i> MUST encode the deduction explained by the immediately preceding NL<i>.
- Tokens in S<i> MUST map to values in "attribute_values".
- Each S<i> MUST be solver-verifiable and may use ONLY:
    ==, !=, <, >, + d ==, Not(...), And(...), Or(...)

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

Partial Answer (PA_k) format:
- PA_k is an OPTIONAL structured snapshot of the partially solved puzzle.
- PA_k MUST be a JSON object directly associated with the "PA<k>" key.
- Do NOT encode PA_k as a string.
- PA_k MUST have exactly TWO keys:
    "header"
    "rows"
- PA_k MUST use the SAME internal schema as the final "solution":

    "PA1": {
      "header": ["House", "Name", "Color", "Children"],
      "rows": [
        ["1", "?", "?", "Bella"],
        ["2", "Arnold", "red", "?"],
        ["3", "?", "?", "?"]
      ]
    }

Structural constraints for every PA_k:
- "header" MUST be exactly identical to the final solution header and solution_header.
- "rows" MUST contain exactly n_houses rows.
- Rows MUST appear in increasing house order from 1 to n_houses.
- Every row MUST contain exactly the same number of cells as the header.
- The House cell MUST contain the corresponding house number and MUST NEVER be "?".
- Every resolved attribute cell MUST contain a normalized value drawn from the corresponding attribute_values list.
- Every unresolved attribute cell MUST contain exactly "?".
- Do NOT use blank cells, null, None, empty strings, candidate lists, or free-form explanations inside PA cells.
- For each attribute column, every known non-"?" value MUST be unique across houses.
- PA_k MUST obey the same header/row/domain/uniqueness constraints as the final solution, except that unresolved attribute cells may contain "?".

Reasoning-state constraints for PA_k:
- PA_k may appear ONLY after a completed NL<i>/S<i> pair.
- Every non-"?" cell in PA_k MUST already be supported by the syntactic_clues, preceding S1..Si deductions, and the puzzle's domain/uniqueness constraints.
- PA_k MUST NOT introduce a speculative assignment or a new assumption.
- PA_k represents the explicitly established state at that checkpoint; it is not required to perform future reasoning in advance.
- A cell not yet established at that checkpoint SHOULD remain "?".
- Later partial answers MUST be monotonic:
    * a previously resolved cell MUST keep the same value;
    * a previously unresolved "?" cell may become resolved;
    * a resolved cell MUST NEVER revert to "?";
    * a resolved cell MUST NEVER change to another value.
- Emit PA_k only at meaningful reasoning milestones, not after every NL/S pair.

Logical validity requirements:
- Every S<i> MUST be logically entailed by syntactic_clues plus earlier S deductions and the puzzle's domain/uniqueness constraints.
- Every S<i> MUST be supported by its immediately preceding NL<i>.
- Every non-"?" PA_k cell MUST be consistent with all syntactic_clues and preceding S deductions.
- The final solution MUST satisfy all syntactic_clues and all S1..Sz deductions.

================================================================================
4) solution (MANDATORY TABLE)
================================================================================
- "solution" MUST be an object with exactly:
  - "header": a list of column names
  - "rows": a list of rows, each row being a list of strings matching the header order
- "header" MUST be exactly identical to solution_header.
- "rows" MUST contain exactly n_houses rows.
- Rows MUST appear in increasing house order from 1 to n_houses.
- Every row MUST contain exactly the same number of cells as the header.
- The House cell MUST contain the corresponding house number and MUST NEVER be "?".
- Every attribute cell MUST contain a normalized value drawn from the corresponding attribute_values list.
- For each attribute column, every allowed value MUST appear exactly once across the n_houses rows.
- The final solution MUST NOT contain "?", blank cells, null, None, empty strings, or candidate lists.
- Every PA_i object uses these SAME header/row/domain/uniqueness constraints, with the single relaxation that unresolved attribute cells may contain "?".

================================================================================
ONE-SHOT EXAMPLE (3 HOUSES, 3 ATTRIBUTES)
================================================================================

Example Puzzle:
There are 3 houses, numbered 1 to 3 from left to right. Each house is occupied by a different person.
Each house has a unique attribute for each of the following characteristics:

- Each person has a unique name: Peter, Eric, Arnold
- The people like unique colors: red, white, yellow
- The people have childern named: Fred, Meredith, Bella

Clues:
1. Arnold is the person whose favorite color is red.
2. The person whose child is named Fred is somewhere to the left of Eric.
3. The person whose favorite color is red is in the second house.
4. The person whose child is named Bella is in the first house.
5. The person who loves white is the person whose child is named Meredith.

solution_header = ["House", "Name", "Color", "Children"]

attribute_values = {
  "Name": ["Peter", "Eric", "Arnold"],
  "Color": ["red", "white", "yellow"],
  "Children": ["Fred", "Meredith", "Bella"]
}


Correct Example Output:
<answer>{
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
  "reasoning": {
    "NL1": "Clues 1 and 3 together show that Arnold, who has the red favorite color, must occupy house 2.",
    "S1": "Arnold == 2.",
    "NL2": "Because Fred is somewhere to the left of Eric, Eric cannot occupy house 1.",
    "S2": "Eric != 1.",
    "NL3": "Arnold already occupies house 2, so Eric cannot occupy house 2 and therefore must occupy house 3.",
    "S3": "Eric == 3.",
    "PA1": {
      "header": ["House", "Name", "Color", "Children"],
      "rows": [
        ["1", "?", "?", "Bella"],
        ["2", "Arnold", "red", "?"],
        ["3", "Eric", "?", "?"]
      ]
    },
    "NL4": "With Arnold in house 2 and Eric in house 3, the remaining person Peter must occupy house 1.",
    "S4": "Peter == 1.",
    "NL5": "Since Eric is in house 3 and Fred must be somewhere to his left, Fred can only be in house 1 or house 2.",
    "S5": "Or(Fred == 1, Fred == 2).",
    "NL6": "Bella is fixed in house 1 by Clue 4, so child uniqueness forces Fred into house 2.",
    "S6": "Fred == 2.",
    "NL7": "With Bella in house 1 and Fred in house 2, the remaining child Meredith must occupy house 3.",
    "S7": "Meredith == 3.",
    "PA2": {
      "header": ["House", "Name", "Color", "Children"],
      "rows": [
        ["1", "Peter", "?", "Bella"],
        ["2", "Arnold", "red", "Fred"],
        ["3", "Eric", "?", "Meredith"]
      ]
    },
    "NL8": "Clue 5 places white in the same house as Meredith, so white must occupy house 3.",
    "S8": "white == 3.",
    "NL9": "Red is already in house 2 and white is in house 3, so color uniqueness forces yellow into house 1.",
    "S9": "yellow == 1."
  },
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


SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

puzzle = {puzzle}

solution_header = {solution_header}

attribute_values = {attribute_values}

Solve the puzzle above and provide n_houses, attribute_values, syntactic_clues, reasoning, and solution for this puzzle in the <answer> </answer> block, with no additional text.
"""


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
            v = "_".join(str(row[i]).split())
            # v = row[i]
            if v not in seen[col]:
                seen[col].add(v)
                values[col].append(v)

    for key in values:
        random.shuffle(values[key])
    return values

def make_map_fn_1_shot(split, data_source):
    def process_fn_1_shot(example, idx):
        # The source ZebraLogic example stores the reference grid under 'solution'.
        final_grid = example['solution']
        # Extract clue text from the puzzle string for extra_info.
        clues = extract_clues_from_puzzle(puzzle_text=example['puzzle'])
        # user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        attribute_values = attribute_values_from_solution(example['solution'])
        user_prompt = SOLUTION_PROMPT_1_SHOT_SYS + SOLUTION_PROMPT_1_SHOT_USER.format(
            puzzle=example['puzzle'],
            solution_header=json.dumps(final_grid['header'], ensure_ascii=False),
            attribute_values=json.dumps(attribute_values, ensure_ascii=False),
        )

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": user_prompt
                }],
            'raw_prompt': [
                {
                    "role": "user",
                    "content": user_prompt
                }],
            "ability": "logical_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": final_grid,
            },
            "apply_chat_template": False,
            "extra_info": {
                'id': example['id'] if 'id' in example else str(idx),
                'split': split,
                'clues': clues
            }
        }

        if idx == 0:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            print("\n\n")
        return data

    return process_fn_1_shot


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/asif/data3/HF_cache/ZebraLogic/', help='Path to json file')
    parser.add_argument('--data_setting', default='mlxl_nss_mlxl_nss', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/Codes_QCRI/OpenAI_test_ZebraPuzzles/mlxl_train_mlxl_test', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.3, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.7, help='Proportion of data for test set')
    parser.add_argument('--data_source_train', default='our_zebra_puzzle_new_reward', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_zebra_puzzle_new_reward_test', help='Name of data source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

    random.seed(args.seed)

    if not (0.0 < args.train_size <= 1.0):
        raise ValueError(f"train_size must be in (0, 1], got {args.train_size}")
    if not (0.0 < args.test_size <= 1.0):
        raise ValueError(f"test_size must be in (0, 1], got {args.test_size}")
    if args.train_size + args.test_size > 1.0:
        raise ValueError(f"The sum of train_size ({args.train_size}) and test_size ({args.test_size}) cannot exceed 1.0")

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
    elif args.data_setting == 'mlxl_nss_mlxl_nss':
        args.data_file = os.path.join(args.data_path, 'Zebra_Puzzle_complete_1000.json')
    else:
        raise ValueError('Invalid data_setting')
    args.output_dir = os.path.join(args.output_dir, args.data_setting)

    if args.data_setting in ('mlxl_nss_mlxl_nss', 'med_train_med_test'):
        # Load one dataset and split it into train/test sets
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
    train_output_dir = os.path.join(args.output_dir ,"train")
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
