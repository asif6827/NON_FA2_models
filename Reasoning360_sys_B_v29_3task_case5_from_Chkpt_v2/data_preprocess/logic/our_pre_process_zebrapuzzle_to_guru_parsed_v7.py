import os
import sys
import datasets
import argparse
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset



SOLUTION_PROMPT_1_SHOT_SYS = """
You are an expert logic puzzle solver.

You will be given one logic puzzle written in plain English, along with a solution header that lists the attribute names used in the puzzle.

Your job is to generate the following FIVE fields:
1) n_houses
2) attribute_values
3) syntactic_clues
4) reasoning (INTERLEAVED natural-language + syntactic)
5) solution

You MUST return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

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
- Merge spaces in VALUES (e.g., "grilled cheese" ~ grilledcheese, "very short" ~ veryshort).
- Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Animal, Occupation, Sport, Height, etc.
- House numbers are integers 1..N.
- Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6, etc.
- 
- Do NOT invent values. Every value MUST be one of the allowed values listed in the puzzle text (after normalization).
- If the clue mentions a bare person name (e.g., "Arnold"), treat it as Name=Arnold.
- If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching attribute/value from the puzzle text.

================================================================================
GLOBAL REASONING REQUIREMENT (MANDATORY)
================================================================================
Let:
- N = n_houses
- M = number of attributes (i.e., number of keys in attribute_values)

You MUST produce **at least N × M × M reasoning steps** in total.

Reasoning steps MUST:
- Be logically valid
- NOT be repeating
- Progressively narrow the solution space
- Include eliminations (not_equal / positional exclusions), not only final placements

================================================================================
1) DOMAIN OUTPUT (MANDATORY)
================================================================================
- "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
- "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values from the puzzle text.
- Each attribute list MUST contain exactly M unique values (after normalization).
- Include every attribute listed in the puzzle text, and only those attributes.
- Do NOT infer extra attributes that are not explicitly listed in the puzzle text.

================================================================================
2) syntactic_clues (MANDATORY, TEXTUAL CONSTRAINTS — NOT PREDICATES)
================================================================================
We do NOT use predicate-style DSL for clues.
Instead, each clue MUST be rewritten as a single-line *syntactic constraint statement* in a Z3-like textual form.

Rules:
- "syntactic_clues" MUST be a list of strings.
- There MUST be exactly one entry per clue, in the same order as the clues.
- Each entry MUST be exactly 1 line and end with a period.
- Each entry MUST start with the clue id prefix: "C<i>: ".
- Use ONLY these syntactic operators in the clue text:
    ==   (same house / equivalence)
    !=   (not same house)
    <    (somewhere left of)
    >    (somewhere right of)
    + 1 == (immediately left of)
    == H  (fixed house index, where H is an integer)
- Use bare normalized tokens (no quotes) for values (e.g., Arnold, engineer, very_short).
- When a clue states a specific house like "in the fifth house", encode as: <token> == 5
  Example: "The lawyer is in the fifth house." -> "C9: lawyer == 5."
- When a clue states "directly left of", encode as: A + 1 == B
  Example: "baseball is directly left of engineer" -> "C12: baseball + 1 == engineer."
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

Syntactic reasoning step format:
  S<k>: <constraint>.

Where:
- <k> starts at 1 and increments by 1 for each syntactic step only (S1, S2, S3...).
- <constraint> must follow the same operator rules as syntactic_clues (==, !=, <, >, + 1 ==, == H).
- Each syntactic step MUST cite its evidence at the end in square brackets using clue ids and/or prior steps:
  Example:
    "S1: engineer == 2. [C12+C18]"
    "S2: dog == 2. [C1+S1]"

Evidence rules:
- Evidence MUST be included for EVERY syntactic step.
- Evidence may reference: C<i> and S<k> only.
- Use "+" to join multiple evidence references, e.g., [C3+C11+S2].

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

================================================================================
ONE-SHOT EXAMPLE (3 HOUSES, 3 ATTRIBUTES)
================================================================================

Example Puzzle:
There are 3 houses, numbered 1 to 3 from left to right.
Each house is occupied by a different person.
Each house has a unique attribute for each category:
- Name: Alice, Bob, Carol
- Pet: cat, dog, bird
- Color: red, blue, green

Clues:
1. Alice is the cat owner.
2. The cat owner is somewhere to the left of the bird owner.
3. The blue house is directly left of the dog owner.
4. Bob lives in the green house.
5. Carol does not own the dog.

solution_header = ["House", "Name", "Pet", "Color"]
    
Correct Example Output:
<answer>{
  "n_houses": 3,
  "attribute_values": {
    "Name": ["Alice", "Bob", "Carol"],
    "Pet": ["cat", "dog", "bird"],
    "Color": ["red", "blue", "green"]
  },
  "syntactic_clues": [
    "C1: Alice == cat.",
    "C2: cat < bird.",
    "C3: blue + 1 == dog.",
    "C4: Bob == green.",
    "C5: Carol != dog."
  ],
  "reasoning": [
    "Alice is the cat owner, so the cat and Alice must be in the same house.",
    "S1: cat == Alice. [C1]",
    "Because the cat is somewhere to the left of the bird, the cat cannot be in the rightmost house.",
    "S2: cat != 3. [C2]",
    "Since Alice owns the cat, Alice cannot be in house 3.",
    "S3: Alice != 3. [S1+S2]",
    "The blue house is directly left of the dog owner, so blue cannot be in the rightmost house.",
    "S4: blue != 3. [C3]",
    "Because blue must have a house to its right, the dog cannot be in the first house.",
    "S5: dog != 1. [C3]",
    "Carol does not own the dog, so Carol cannot be in the same house as the dog.",
    "S6: Carol != dog. [C5]",
    "Bob lives in the green house, so Bob and green share the same house.",
    "S7: Bob == green. [C4]",
    "Since Bob is green, Bob cannot live in a blue house.",
    "S8: Bob != blue. [S7]",
    "Alice cannot be in house 3, so Alice must be in house 1 or 2.",
    "S9: Alice != 3. [S3]",
    "If the cat were in house 2, the bird would have to be in house 3.",
    "S10: bird == 3. [C2]",
    "Because the cat cannot be in house 3 or 2, the cat must be in house 1.",
    "S11: cat == 1. [S2]",
    "Since Alice owns the cat, Alice must be in house 1.",
    "S12: Alice == 1. [S1+S11]",
    "Because blue is not in house 3 and house 1 is occupied by Alice, blue must be in house 1 or 2.",
    "S13: blue != 3. [S4]",
    "Blue cannot be in house 1 because Alice already occupies house 1.",
    "S14: blue != 1. [S12]",
    "Therefore blue must be in house 2.",
    "S15: blue == 2. [S13+S14]",
    "Since blue is in house 2, the dog must be in house 3.",
    "S16: dog == 3. [C3+S15]",
    "Because Carol does not own the dog, Carol cannot be in house 3.",
    "S17: Carol != 3. [S6+S16]",
    "Bob cannot be in house 1 because Alice is already there.",
    "S18: Bob != 1. [S12]",
    "Bob cannot be in house 2 because that house is blue and Bob is green.",
    "S19: Bob != 2. [S7+S15]",
    "Therefore Bob must be in house 3.",
    "S20: Bob == 3. [S18+S19]",
    "Since Bob lives in the green house, green must be in house 3.",
    "S21: green == 3. [S7+S20]",
    "Carol cannot be in house 3 and cannot be in house 1, so Carol must be in house 2.",
    "S22: Carol == 2. [S12+S17]",
    "The remaining color for house 1 is red.",
    "S23: red == 1. [S15+S21]",
    "The remaining color for house 2 is blue, which is already known.",
    "S24: blue == 2. [S15]",
    "The remaining pet for house 2 cannot be cat or dog.",
    "S25: bird != 2. [S11+S16]",
    "Since bird is not in house 2 and not in house 1, it must be in house 3.",
    "S26: bird == 3. [S11+S25]",
    "All remaining assignments are now fully determined and consistent.",
    "S27: Carol == 2. [S22]"
  ],
  "solution": {
    "header": ["House", "Name", "Pet", "Color"],
    "rows": [
      ["1", "Alice", "cat", "red"],
      ["2", "Carol", "bird", "blue"],
      ["3", "Bob", "dog", "green"]
    ]
  }
}</answer>
"""


SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

{puzzle}


solution_header = {solution_header}

Solve the puzzle above and provide n_houses, attribute_values, parsed_clues, parsed_reasoning and solution for this puzzle in the <answer> </answer> block, with no additional text.
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



def make_map_fn_1_shot(split, data_source):
    def process_fn_1_shot(example, idx):
        # Use 'ground_truth' instead of 'solution' since that's what the input data has
        final_grid = example['solution']
        # Use the 'clues' field directly from the input data
        clues = extract_clues_from_puzzle(puzzle_text=example['puzzle'])
        # user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        user_prompt = SOLUTION_PROMPT_1_SHOT_SYS + SOLUTION_PROMPT_1_SHOT_USER.format(
            puzzle=example['puzzle'],solution_header=final_grid['header'])

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
    parser.add_argument('--data_setting', default='small_train_small_test', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/ZebraLogic/', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.6, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.4, help='Proportion of data for test set')
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
    elif args.data_setting == 'small_train_small_test':
        args.data_file = os.path.join(args.data_path, 'Zebra_Puzzle_small_320.json')
        pass
    else:
        raise ValueError('Invalid data_setting')
    args.output_dir = os.path.join(args.output_dir, args.data_setting)



    if args.data_setting == 'small_train_small_test':
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
