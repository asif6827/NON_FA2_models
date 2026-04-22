import os
import sys
import datasets
import random
import argparse
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset



SOLUTION_PROMPT_1_SHOT_SYS = """
You are an expert AR-LSAT logical reasoning solver.

You are given:
(i) one AR-LSAT logic game passage written in plain English,
(ii) a question about that passage,
(iii) a list of answer options,
and optionally
(iv) structured metadata such as game_type, entity sets, slot sets, or candidate value vocabularies if provided.

Your task is to construct a fully consistent, solver-verifiable formalization of the AR-LSAT problem and determine the correct answer by generating the following SIX fields:
1) problem_type — the inferred game structure.
2) world_model — a normalized structural representation of the entities, slots, groups, domains, and uniqueness/cardinality assumptions.
3) syntactic_clues — a normalized, Z3-style textual encoding of each passage clue.
4) question_semantics — a normalized formal interpretation of the question type and any temporary question-specific assumption.
5) reasoning — interleaved reasoning consisting of natural-language explanations and syntactic (solver-checkable) deduction/testing steps.
6) answer — the final selected option, justified exclusively from syntactic_clues, question_semantics, and syntactic reasoning steps (S1..Sk).

You MUST return the result STRICTLY as a single valid JSON object wrapped inside:
<answer>...</answer>

No additional text, commentary, or formatting outside the <answer> block is permitted.


================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Do NOT include extra text, markdown, explanations, or code fences.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly SIX top-level keys, spelled EXACTLY:
    "problem_type",
    "world_model",
    "syntactic_clues",
    "question_semantics",
    "reasoning",
    "answer"
- Do NOT add any other keys.

================================================================================
CORE MODELING PRINCIPLE
================================================================================
- Do NOT assume the problem is a ZebraPuzzle-style house puzzle.
- Do NOT assume fixed houses unless the passage explicitly defines positions/houses/seats/days/ranks/etc.
- First infer the game structure from the passage.
- AR-LSAT problems may involve:
    - ordering
    - grouping
    - assignment
    - selection
    - matching
    - hybrid combinations of the above
- The goal is not always to recover one complete global table.
- The goal is to determine which answer option is logically correct under the passage constraints and question semantics.

================================================================================
NORMALIZATION RULES
================================================================================
- Use underscores instead of spaces in normalized symbolic VALUES (e.g., grilled_cheese, very_short, group_1).
- Do not invent entities, slots, or values not licensed by the passage or explicit metadata.
- Convert ordinals to integers when applicable:
    first=1, second=2, third=3, fourth=4, fifth=5, sixth=6, etc.
- If the passage uses names, treat them as canonical entity tokens.
- If the passage uses phrases such as “the violinist”, “the red book”, “the Tuesday speaker”, or “the student in group F”, map them to concise normalized tokens.
- Keep normalization faithful to the passage meaning.
- If metadata provides canonical tokens, use them.
- If no metadata is provided, create faithful normalized tokens directly from the passage.

================================================================================
1) problem_type (MANDATORY)
================================================================================
- "problem_type" MUST be one of:
    "ordering"
    "grouping"
    "assignment"
    "selection"
    "matching"
    "hybrid"
- Use "hybrid" if more than one structure is essential.

================================================================================
2) world_model (MANDATORY)
================================================================================
- "world_model" MUST be a JSON object.
- It MUST describe the inferred formal structure of the puzzle.
- Include exactly these keys:
    "entities"
    "domains"
    "structural_assumptions"

Rules:
- "entities" = list of core items being assigned/arranged/selected.
- "domains" = object describing target domains such as positions, groups, categories, days, ranks, or Boolean selection status.
- "structural_assumptions" = list of short strings describing the formal assumptions required by the passage, such as:
    - "each entity occupies exactly one position"
    - "positions are unique"
    - "each selected committee member is distinct"
    - "each group has exactly two members"
    - "an entity may belong to more than one category"
- Only include assumptions supported by the passage.
- Do NOT force one-to-one mappings unless the passage clearly implies them.

================================================================================
3) syntactic_clues (MANDATORY, TEXTUAL CONSTRAINTS — NOT PREDICATES)
================================================================================
We do NOT use predicate-style DSL for clues.
Instead, each clue MUST be rewritten as a single-line syntactic constraint statement in a Z3-like textual form.

Rules:
- "syntactic_clues" MUST be a list of strings.
- There MUST be exactly one entry per passage clue, in the same order as the clues.
- Each entry MUST be exactly 1 line and end with a period.
- Each entry MUST start with the clue id prefix: "C<i>: ".

Allowed operator families:
- Equality / co-reference:
    ==
    !=
- Ordering:
    <
    >
    + k ==
- Boolean composition:
    Not(...)
    And(...)
    Or(...)
    Xor(...)
    Implies(A, B)
- Counting:
    Count(...)
    Exactly(k, ...)
    AtLeast(k, ...)
    AtMost(k, ...)
- Membership / assignment style:
    In(entity, group)
    Assign(entity, slot)
    Select(entity)

Use the simplest faithful representation.

Examples:
- "A is earlier than B" -> "C1: A < B."
- "A is immediately before B" -> "C2: A + 1 == B."
- "A and B are in the same group" -> "C3: group_of(A) == group_of(B)."
- "A and B are not both selected" -> "C4: Not(And(Select(A), Select(B)))."
- "If F is selected, G is selected" -> "C5: Implies(Select(F), Select(G))."
- "Exactly two of A, B, C are selected" -> "C6: Exactly(2, Select(A), Select(B), Select(C))."

IMPORTANT:
- The goal is to produce constraints that resemble solver-checkable expressions,
  but you must NOT write "s.add(...)".
- Only output the inner formal constraint as text.
- Use the representation that best matches the puzzle type.
- Do not force every clue into house-index form.

================================================================================
4) question_semantics (MANDATORY)
================================================================================
- "question_semantics" MUST be a JSON object with exactly these keys:
    "question_type"
    "question_condition"
    "option_interpretation_rule"

Rules:
- "question_type" MUST be one of:
    "could_be_true"
    "must_be_true"
    "cannot_be_true"
    "could_be_false"
    "must_follow"
    "valid_complete_assignment"
    "other"
- "question_condition" MUST be either:
    - a normalized formal string representing an extra temporary assumption from the question stem, or
    - "None" if there is no extra assumption.
- "option_interpretation_rule" MUST be a short string describing the solver semantics:

Required semantics:
- could_be_true => choose the option whose formalization is satisfiable with base constraints and question condition.
- must_be_true => choose the option whose negation is unsatisfiable with base constraints and question condition.
- cannot_be_true => choose the option whose formalization is unsatisfiable with base constraints and question condition.
- could_be_false => choose the option whose negation is satisfiable with base constraints and question condition.
- must_follow => choose the option whose negation is unsatisfiable with base constraints and question condition.
- valid_complete_assignment => choose the option that is satisfiable and satisfies all uniqueness/cardinality rules.

================================================================================
5) reasoning (MANDATORY — INTERLEAVED NATURAL + SYNTACTIC)
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly 1 sentence and end with a period.
- Reasoning MUST be interleaved:
    Odd-numbered entries: Natural-language reasoning.
    Even-numbered entries: Syntactic reasoning/testing step.
- Natural-language entries should explain either:
    - the extracted structure,
    - a deduction from the clues,
    - or the logic of testing an option.
- Syntactic entries should encode newly deduced facts, structural consequences, or option-test results.

Syntactic entry format:
- Every syntactic entry MUST start with "S<k>: " and MUST end with a period.
- <k> starts at 1 and increments by 1 for each syntactic step only.

Allowed syntactic step forms:
- atomic structural facts
- derived constraints
- question-condition constraints
- option test outcomes such as:
    Sat(Option_A)
    Unsat(Option_B)
    Sat(Base_And_Q_And_Option_C)
    Unsat(Base_And_Q_And_Not_Option_D)

Allowed operators:
  ==, !=, <, >, + d ==, Not(...), And(...), Or(...), Xor(...), Implies(...),
  Count(...), Exactly(...), AtLeast(...), AtMost(...), In(...), Assign(...), Select(...),
  Sat(...), Unsat(...)

Logical validity requirement:
- Every syntactic step MUST be logically supported by the passage clues, the question condition, and earlier syntactic steps.
- Do NOT invent deductions.
- You may include option-testing steps when the question requires checking answer choices.

================================================================================
6) answer (MANDATORY)
================================================================================
- "answer" MUST be a JSON object with exactly these keys:
    "selected_option"
    "justification"
- "selected_option" MUST be one of:
    "A", "B", "C", "D", "E"
  unless the provided problem has a different labeled option set, in which case use the exact labels given.
- "justification" MUST be a short string explaining why that option satisfies the question semantics.

================================================================================
OPTION HANDLING
================================================================================
- Treat the passage clues as the base constraint system.
- Treat any hypothetical condition in the question stem as a temporary extra condition.
- Treat each answer option as a temporary candidate constraint or candidate full assignment.
- Test options independently.
- Do not let one option contaminate another.
- Conceptually use push/pop style reasoning.

================================================================================
SOLVING BEHAVIOR
================================================================================
When solving:
1. Infer the game structure.
2. Build the world model.
3. Normalize each clue into solver-style syntax.
4. Interpret the exact question semantics.
5. Normalize each answer option into a solver-checkable candidate.
6. Use satisfiability-style reasoning appropriate to the question type.
7. Return the final answer.

Important:
- Be faithful to the passage.
- Do not force ZebraPuzzle assumptions onto AR-LSAT problems.
- Do not require a full final world unless the question itself requires it.
- Prefer general formal correctness over stylistic elegance.
"""

SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
AR-LSAT PROBLEM TO SOLVE
--------------------------------

passage = {passage}

question = {question}

options = {options}

optional_metadata = {optional_metadata}

Solve the AR-LSAT problem above and provide problem_type, world_model, syntactic_clues, question_semantics, reasoning, and answer in the <answer> </answer> block, with no additional text.
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
            v = "_".join(row[i].split(" "))
            #v = row[i]
            if v not in seen[col]:
                seen[col].add(v)
                values[col].append(v)

    for key in values:
        random.shuffle(values[key])
    return values

def make_map_fn_1_shot(split, data_source):
    def process_fn_1_shot(example, idx):
        # Use 'ground_truth' instead of 'solution' since that's what the input data has
        final_grid = example['solution']
        # Use the 'clues' field directly from the input data
        clues = extract_clues_from_puzzle(puzzle_text=example['puzzle'])
        # user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        user_prompt = SOLUTION_PROMPT_1_SHOT_SYS + SOLUTION_PROMPT_1_SHOT_USER.format(
            puzzle=example['puzzle'],solution_header=final_grid['header'], attribute_values=attribute_values_from_solution(example['solution']))

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
    parser.add_argument('--data_setting', default=None, help='Path to json file')
    parser.add_argument('--output_dir', default=None, help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.3, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.7, help='Proportion of data for test set')
    parser.add_argument('--data_source_train', default='our_zebra_puzzle_new_reward', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_zebra_puzzle_new_reward_test', help='Name of data source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

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
