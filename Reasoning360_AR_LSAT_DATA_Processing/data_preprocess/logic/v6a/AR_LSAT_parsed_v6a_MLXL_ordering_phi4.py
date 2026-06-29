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



ORDERING_PROMPT_1_SHOT_SYS = """
You are an expert AR-LSAT ordering-game solver.

You are given:
(i) one AR-LSAT ordering passage written in plain English,
(ii) one question about that passage,
(iii) a question_type label,
(iv) a dictionary of answer options,
and optionally
(v) metadata such as tags or entity hints if available.

This prompt is ONLY for ORDERING problems.

Your task is to parse the ordering problem into a solver-oriented logical representation and determine the correct answer by generating the following EIGHT fields:
1) problem_type — must be "ordering".
2) world_model — ordered entities, position domain, and structural assumptions.
3) rules — formalized passage rules only.
4) facts — question-specific temporary conditions only.
5) question_semantics — how the options must be evaluated using the provided question_type.
6) options — formalized answer options.
7) reasoning — interleaved natural-language reasoning and formal solver-oriented steps.
8) solution — the final selected answer option.

You MUST return the result STRICTLY as a single valid JSON object wrapped inside:
<answer>...</answer>

No additional text, commentary, markdown, or formatting outside the <answer> block is permitted.

================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly EIGHT top-level keys, spelled EXACTLY:
    "problem_type",
    "world_model",
    "rules",
    "facts",
    "question_semantics",
    "options",
    "reasoning",
    "solution"
- "problem_type" MUST be exactly "ordering".
- Do NOT add any other top-level keys.
- Do NOT output Python code.
- Do NOT output executable Z3 code.
- Do NOT use markdown or code fences.
- All formal expressions MUST be strings.

================================================================================
NORMALIZATION RULES FOR ORDERING
================================================================================
- Use concise symbolic tokens for entities.
- Preserve single-letter entity labels exactly when used, e.g., A, B, C, D.
- Use underscores instead of spaces in multi-word tokens, e.g., first_shift, red_book.
- Convert ordinals to integers when used as positions:
    first=1, second=2, third=3, fourth=4, fifth=5, sixth=6, seventh=7, eighth=8.
- Positions, ranks, seats, slots, days, shelves in a sequence, bays, folders, hangers, or time slots must be represented with integers.
- Do not invent entities, positions, rules, facts, or assumptions not supported by the passage or metadata.
- If metadata provides canonical names or tags, use them when consistent with the passage.

================================================================================
PARSING INSTRUCTIONS FOR ORDERING
================================================================================
The problem is already classified as ordering.

Construct the world model:
- Extract all ordered entities.
- Extract the position domain, usually 1..N.
- Include structural assumptions such as:
    each entity occupies exactly one position,
    each position is occupied by exactly one entity,
    positions are ordered from 1 to N.
- Include Distinct(...) in rules when the passage states or implies unique positions.

Parse passage rules:
- "rules" MUST contain only permanent passage constraints.
- Do NOT include question-specific facts in "rules".
- Each rule must be a single solver-oriented logical expression.

Parse question facts:
- "facts" MUST contain only temporary assumptions introduced by the question stem.
- If there are no question-specific facts, output [].

Parse question semantics:
- Use the provided question_type input.
- "question_type" MUST be copied from the input question_type.
- Use the correct solver interpretation:
    could_be_true => option is satisfiable with rules + facts.
    must_be_true => Not(option) is unsatisfiable with rules + facts.
    cannot_be_true => option is unsatisfiable with rules + facts.
    could_be_false => Not(option) is satisfiable with rules + facts.
    acceptability => option is a complete or partial arrangement that is satisfiable with rules + facts.
    rule_substitution => option preserves the same solution space as the replaced rule.
    other => infer the closest solver interpretation from the question text.

Parse answer options:
- "options" MUST be a JSON object mapping option labels to formal expressions.
- Each option must be formalized independently.
- Do not let one option affect another option.
- Use the exact option labels from the input.

Allowed formal operators:
- Logical: And(...), Or(...), Not(...), Xor(...), Implies(...)
- Equality: ==, !=
- Ordering: <, >, <=, >=, + k ==, - k ==
- Distinctness: Distinct(...)
- Counting if needed: AtLeast(k, ...), AtMost(k, ...), Exactly(k, ...)
- Solver-status steps in reasoning only: Sat(...), Unsat(...)

Ordering expression guide:
- A before B => A < B.
- A after B => A > B.
- A immediately before B => A + 1 == B.
- A immediately after B => A - 1 == B.
- A is in position 3 => A == 3.
- A is not first => A != 1.
- A and B are adjacent => Or(A + 1 == B, B + 1 == A).
- Exactly one item per position => Distinct(A, B, C, ...).

================================================================================
REASONING REQUIREMENTS FOR ORDERING PROBLEMS
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly one sentence and end with a period.
- Reasoning MUST be interleaved:
    Odd-numbered entries: natural-language reasoning.
    Even-numbered entries: formal solver-oriented step.
- Natural-language entries must explain why the next formal step follows from rules, facts, earlier steps, or option testing.
- Formal entries must encode a newly derived ordering fact, domain restriction, option feasibility result, or forced/impossible position.

Formal step format:
- Every formal step MUST start with "S<k>: " and MUST end with a period.
- <k> starts at 1 and increments by 1 for each formal step only.
- Formal steps must be solver-verifiable and may use ONLY:
    ==, !=, <, >, <=, >=, + d ==, - d ==, Not(...), And(...), Or(...), Implies(...), Distinct(...), Sat(...), Unsat(...)

Atomic ordering operators:
    A == H          entity A is in position H.
    A != H          entity A is not in position H.
    A < B           A is before B.
    A > B           A is after B.
    A + d == B      A is exactly d positions before B.
    A - d == B      A is exactly d positions after B.
    A <= B          A is not after B.
    A >= B          A is not before B.

Boolean operators:
    Not(e)
    And(e1, e2, ..., en)
    Or(e1, e2, ..., en)
    Implies(e1, e2)

Option-testing operators:
    Sat(Option_A)
    Unsat(Option_A)

Allowed reasoning step types:
- Direct question facts:
    If the question says "If B is fourth", a valid step is:
    S1: B == 4.

- Derived forced positions:
    If rules and facts force A to be second, a valid step is:
    S2: A == 2.

- Derived impossibilities:
    If A cannot be first, a valid step is:
    S3: Not(A == 1).

- Relative-order deductions:
    If A must occur before B, a valid step is:
    S4: A < B.

- Immediate-order deductions:
    If C must be immediately after A, a valid step is:
    S5: C == A + 1.

- Disjunctive placement deductions:
    If A can only be first or third, a valid step is:
    S6: Or(A == 1, A == 3).

- Combined deductions:
    If multiple position exclusions are derived together, a valid step is:
    S7: And(A != 1, A != 4).

- Option feasibility checks:
    If an option can be extended to at least one full valid ordering, use:
    S8: Sat(Option_C).

- Option impossibility checks:
    If an option cannot be extended to any full valid ordering, use:
    S9: Unsat(Option_D).

Logical validity requirements:
- Every formal step MUST be entailed by rules + facts + earlier accepted formal steps, unless it is an option feasibility step.
- For option feasibility steps:
    Sat(Option_X) means rules + facts + Option_X is satisfiable.
    Unsat(Option_X) means rules + facts + Option_X is unsatisfiable.
- Do NOT output unsupported guesses.
- Do NOT output contradictory steps.
- Do NOT output tautologies such as Or(A == 1, A != 1).
- Do NOT merely restate every passage rule unless the restatement is needed to connect a deduction.
- Do NOT use house-based language unless the ordering problem is actually about houses.
- Do NOT use Assign(entity, value) in ordering reasoning unless the problem explicitly uses assignment-like values; prefer integer position expressions.

Examples of valid interleaved ordering reasoning:
    The question condition places B in the fourth position.
    S1: B == 4.

    Since C is immediately after A, C's position must be exactly one greater than A's position.
    S2: C == A + 1.

    Because B is already fourth and all positions are distinct, C cannot also be fourth.
    S3: C != 4.

    Since C must be immediately after A and C cannot be fourth, A cannot be third.
    S4: Not(A == 3).

    Since D is not first, D must occupy one of positions 2, 3, or 4.
    S5: Or(D == 2, D == 3, D == 4).

    Option A can be extended to a complete valid ordering.
    S6: Sat(Option_A).

    Option B forces C into the fourth position, which conflicts with B already being fourth.
    S7: Unsat(Option_B).
    
    
================================================================================
SOLUTION REQUIREMENTS
================================================================================
- "solution" MUST be a JSON object with exactly this key:
    "selected_option"
- "selected_option" MUST use the exact option label from the input.
- Do NOT include a final table unless the question asks for a complete ordering.
- Do NOT include extra explanation inside "solution".

================================================================================
OUTPUT SCHEMA
================================================================================
The output MUST follow this exact structure:

<answer>{
  "problem_type": "ordering",
  "world_model": {
    "entities": [],
    "domains": {
      "positions": []
    },
    "structural_assumptions": []
  },
  "rules": [],
  "facts": [],
  "question_semantics": {
    "question_type": "",
    "option_interpretation_rule": ""
  },
  "options": {},
  "reasoning": [],
  "solution": {
    "selected_option": ""
  }
}</answer>

================================================================================
ONE-SHOT EXAMPLE: ORDERING
================================================================================

Example Passage:
Four speakers A, B, C, and D speak in positions 1 through 4, with exactly one speaker in each position.

Rules:
1. A speaks before B.
2. C speaks immediately after A.
3. D does not speak first.

Example Question:
If B speaks fourth, which one of the following could be true?

Example question_type:
could_be_true

Example Options:
A. A speaks second.
B. C speaks fourth.
C. D speaks second.
D. A speaks third.
E. C speaks first.

Correct Example Output:
<answer>{
  "problem_type": "ordering",
  "world_model": {
    "entities": ["A", "B", "C", "D"],
    "domains": {
      "positions": ["1", "2", "3", "4"]
    },
    "structural_assumptions": [
      "each speaker occupies exactly one position",
      "each position is occupied by exactly one speaker",
      "positions are ordered from 1 to 4"
    ]
  },
  "rules": [
    "Distinct(A, B, C, D)",
    "A < B",
    "C == A + 1",
    "D != 1"
  ],
  "facts": [
    "B == 4"
  ],
  "question_semantics": {
    "question_type": "could_be_true",
    "option_interpretation_rule": "choose the option whose formalization is satisfiable with rules and facts"
  },
  "options": {
    "A": "A == 2",
    "B": "C == 4",
    "C": "D == 2",
    "D": "A == 3",
    "E": "C == 1"
  },
  "reasoning": [
    "The question condition fixes B in the fourth position.",
    "S1: B == 4.",
    "Since C is immediately after A, A cannot be fourth and C cannot be first.",
    "S2: C == A + 1.",
    "Option A is satisfiable because A can be second, C third, B fourth, and D first is disallowed by the rule.",
    "S3: Sat(Option_A).",
    "Option B places C fourth, which would force A third and conflict with B already being fourth.",
    "S4: Unsat(Option_B).",
    "Option C places D second, leaving no valid consecutive placement for A and C before B.",
    "S5: Unsat(Option_C).",
    "Option D places A third, which forces C fourth and conflicts with B fourth.",
    "S6: Unsat(Option_D).",
    "Option E places C first, which is impossible because C must be immediately after A.",
    "S7: Unsat(Option_E)."
  ],
  "solution": {
    "selected_option": "A"
  }
}</answer>
"""

ORDERING_PROMPT_1_SHOT_USER = """
--------------------------------
AR-LSAT ORDERING PROBLEM TO SOLVE
--------------------------------

passage = {passage}

question = {question}

question_type = {question_type}

options = {options}

metadata = {metadata}

Solve the AR-LSAT ordering problem above and return problem_type, world_model, rules, facts, question_semantics, options, reasoning, and solution inside a single <answer>...</answer> block, with no additional text.
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
        final_grid = example['answer']
        # user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        user_prompt = ORDERING_PROMPT_1_SHOT_SYS + ORDERING_PROMPT_1_SHOT_USER.format(
            passage=example['passage'],
            question=example['question'],
            question_type=example['question_type'],
            options=example['options'],
            metadata=None,
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
                'split': split
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
    parser.add_argument('--data_path', default='/home/asif/data3/Codes_QCRI/AR-LSAT/processed_ar_lsat/', help='Path to json file')
    parser.add_argument('--data_setting', default='mlxl_train_mlxl_test', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/AR_LSAT_to_guru/', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--data_source_train', default='our_ar_lsat_ordering_new_reward_phi4', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_ar_lsat_ordering_new_reward_test_phi4', help='Name of data source')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

    if args.data_setting == 'mlxl_train_mlxl_test':
        args.train_data_file = os.path.join(args.data_path, 'AR_LSAT_train_ordering_300.json')
        args.test_data_file = os.path.join(args.data_path, 'AR_LSAT_test_ordering_112.json')
    else:
        raise ValueError('Invalid data_setting')
    args.output_dir = os.path.join(args.output_dir, args.data_setting)



    if args.data_setting == 'mlxl_train_mlxl_test':
        # Load dataset from JSON or Parquet based on file extension
        train_dataset = datasets.load_dataset('json', data_files=args.train_data_file)['train']
        test_dataset = datasets.load_dataset('json', data_files=args.test_data_file)['train']



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
