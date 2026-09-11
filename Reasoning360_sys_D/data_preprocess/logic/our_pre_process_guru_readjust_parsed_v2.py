import os
import sys
import datasets
import argparse
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset

# Import prompt templates directly
SOLUTION_PROMPT_1_SHOT_SYS = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
    - Extract the domain (N houses + all attribute values).
    - Parse each clue into a canonical, machine-checkable form.
    - Perform step-by-step deductions using only canonical atoms.
    - Derive a correct final solution.
    - Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
    - Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
    - Do NOT include extra text, markdown, explanations, or code fences.
    - Inside <answer>...</answer>, the content MUST be a single valid JSON object.
    - The JSON object MUST have exactly FIVE top-level keys:
        "n_houses", "attribute_values", "parsed_clues", "parsed_reasoning", "solution".
    - Do NOT add any other keys.

NORMALIZATION RULES:
    - Use underscores instead of spaces in VALUES (e.g., grilled_cheese, root_beer, bmw_3_series).
    - Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Drink, Pet, HairColor, Lunch, Nationality, PhoneModel, etc.
    - House numbers are integers 1..N.
    - Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6.
    - Do NOT invent values. Every <Val> must be one of the allowed values listed in the puzzle text (after normalization).
    - If the clue mentions a bare person name (e.g., "Bob"), treat it as Name=Bob.
    - If the clue mentions a bare demonym (e.g., "The German"), map it to Nationality=german (or the matching attribute in the puzzle text).
    - If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching attribute/value from the puzzle text (e.g., Pet=cat, Drink=coffee), choosing the closest listed value.

D) DOMAIN OUTPUT (MANDATORY)
    - "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
    - "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values from the puzzle text.
    - Each attribute list MUST contain exactly N unique values (after normalization).
    - Include every attribute listed in the puzzle text, and only those attributes.
    - Do NOT infer extra attributes that are not explicitly listed in the puzzle text.

A) parsed_clues (MANDATORY, PARSABLE)
    - "parsed_clues" MUST be a list of strings.
    - Each string must be exactly 1 sentence and end with a period.
    - There MUST be exactly one entry per clue, in the same order as the clues.
    - Each parsed clue MUST follow this exact DSL format:

    C<i> = <predicate>.

Allowed <predicate> forms (use exactly these):
    - set(<H>,<Attr>,<Val>)
    - not_set(<H>,<Attr>,<Val>)
    - immediately_left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - right_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - adjacent(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - same_house(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - between(<AttrA>=<ValA>,<AttrB>=<ValB>,<K>)

Semantics:
    - immediately_left_of(A,B): A is exactly 1 house left of B.
    - left_of(A,B): A is somewhere left of B (strictly smaller house index).
    - right_of(A,B): A is somewhere right of B (strictly larger house index).
    - adjacent(A,B): houses differ by exactly 1.
    - between(A,B,K): there are exactly K houses strictly between A and B.
      (So K=1 => positions differ by 2, and K=2 => positions differ by 3.)

B) parsed_reasoning (MANDATORY, PARSABLE)
    - "parsed_reasoning" MUST be a list of strings.
    - Each string must be exactly 1 sentence and end with a period.
    - There is NO LIMIT on the number of entries.
    - Each entry MUST follow this exact DSL format:

    S<k> [C<i>(+C<j>...)] <op>(<H>,<Attr>,<Val>).

Where:
    - <k> is a step number starting at 1 and increasing by 1 each step.
    - Evidence inside [...] must reference clue ids, e.g. [C1] or [C1+C3].
    - <op> is either set or not.
    - <H> is a house number integer (1..N).
    - <Attr> and <Val> must come from the puzzle text (normalized with underscores for values).

LOGICAL VALIDITY REQUIREMENT:
    - Every step in "parsed_reasoning" MUST be logically entailed by the parsed clues plus any earlier reasoning steps.
    - If you cannot deduce a set(...) fact with certainty, output a not(...) fact that is guaranteed true.

C) solution (MANDATORY TABLE)
    - "solution" MUST be in tabular form with:
      - "header": a list of column names
      - "rows": a list of rows, each row being a list of strings matching the header order.
    - The header MUST include "House" and then all attribute columns from the puzzle text.
    - The rows MUST list houses in increasing order from 1..N.
    - All solution VALUES must be normalized with underscores (same as above).

EXAMPLE DEMONSTRATION (illustration only)

Example Puzzle:
There are 3 houses, numbered 1 to 3 from left to right.
Each person has a unique name: Peter, Eric, Arnold.
Each person has a unique drink: tea, water, milk.

Clues:
1. Peter is in the second house.
2. Arnold is directly left of the one who drinks water.
3. The water drinker is directly left of the milk drinker.

Correct Example Output:
<answer>{
  "n_houses": 3,
  "attribute_values": {
    "Name": ["Peter", "Eric", "Arnold"],
    "Drink": ["tea", "water", "milk"]
  },
  "parsed_clues": [
    "C1 = set(2,Name,Peter).",
    "C2 = immediately_left_of(Name=Arnold,Drink=water).",
    "C3 = immediately_left_of(Drink=water,Drink=milk)."
  ],
  "parsed_reasoning": [
    "S1 [C1] set(2,Name,Peter).",
    "S2 [C3] not(3,Drink,water).",
    "S3 [C3] not(1,Drink,milk).",
    "S4 [C2] not(3,Name,Arnold).",
    "S5 [C2+C3] set(1,Name,Arnold)."
  ],
  "solution": {
    "header": ["House", "Name", "Drink"],
    "rows": [
      ["1", "Arnold", "tea"],
      ["2", "Peter", "water"],
      ["3", "Eric", "milk"]
    ]
  }
}</answer>
"""

SOLUTION_PROMPT_ORDERED_V2 = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
    - Extract the domain (N houses + all attribute values).
    - Parse each clue into a canonical, machine-checkable form (ONE parsed clue per clue, SAME ORDER).
    - Perform step-by-step deductions using only canonical atoms.
    - Derive a correct final solution.
    - Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
    - Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
    - Do NOT include extra text, markdown, explanations, or code fences.
    - Inside <answer>...</answer>, the content MUST be a single valid JSON object.
    - The JSON object MUST have exactly FIVE top-level keys:
      "n_houses", "attribute_values", "parsed_clues", "parsed_reasoning", "solution".
    - Do NOT add any other keys.

NORMALIZATION RULES:
    - Use underscores instead of spaces in VALUES (e.g., grilled_cheese, root_beer, bmw_3_series).
    - Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Drink, Pet, HairColor, Lunch, Nationality, PhoneModel, etc.
    - House numbers are integers 1..N.
    - Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6.
    - Do NOT invent values. Every <Val> must be one of the allowed values listed in the puzzle text (after normalization).
    - If the clue mentions a bare person name (e.g., "Bob"), treat it as Name=Bob.
    - If the clue mentions a bare demonym (e.g., "The German"), map it to Nationality=german (or the matching attribute in the puzzle text).
    - If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching attribute/value from the puzzle text
      (e.g., Pet=cat, Drink=coffee), choosing the closest listed value.

D) DOMAIN OUTPUT (MANDATORY)
    - "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
    - "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values from the puzzle text.
    - Each attribute list MUST contain exactly N unique values (after normalization).
    - Include every attribute listed in the puzzle text, and only those attributes.
    - Do NOT infer extra attributes that are not explicitly listed in the puzzle text.

A) parsed_clues (MANDATORY, PARSABLE, ORDERED, FAITHFUL)
    - "parsed_clues" MUST be a list of strings.
    - Each string must be exactly 1 sentence and end with a period.
    - There MUST be exactly one entry per clue, in the same order as the clues:
      - parsed_clues[0] is ONLY for Clue #1
      - parsed_clues[1] is ONLY for Clue #2
      - ...
    - Each parsed clue MUST follow this exact DSL format:

      C<i> = <predicate>.

    Allowed <predicate> forms (use exactly these):
    - set(<H>,<Attr>,<Val>)
    - not_set(<H>,<Attr>,<Val>)
    - immediately_left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - right_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - adjacent(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - same_house(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - between(<AttrA>=<ValA>,<AttrB>=<ValB>,<K>)

    Semantics (DO NOT MIX THESE UP):
    - immediately_left_of(A,B): A is exactly 1 house left of B. ("directly left of", "immediately left of")
    - left_of(A,B): A is somewhere left of B (strictly smaller index). ("somewhere to the left of", "to the left of")
    - right_of(A,B): A is somewhere right of B. ("somewhere to the right of", "to the right of")
    - adjacent(A,B): houses differ by exactly 1. ("next to", "adjacent to")
    - same_house(A,B): A and B belong to the same person/house. ("is", "has", "the X is Y", "X is the Y")
    - between(A,B,K): exactly K houses strictly between A and B (K=1 => distance 2).

    CLUE-TO-DSL FIDELITY RULES (VERY IMPORTANT):
    - For each clue i, ONLY use entities that appear in that clue:
      - If clue i does NOT mention Arnold, do NOT include Name=Arnold in C<i>.
      - Do NOT swap in other values from the domain.
    - Preserve the relation type from the text:
      - "somewhere to the left" => left_of(...) (NOT immediately_left_of)
      - "directly/immediately left" => immediately_left_of(...)
      - "in the second house" => set(2,...)
      - "X is Y" / "The X is Y" => same_house(...)
    - parsed_clues are NOT a place for deductions:
      - Do NOT set a house number unless the clue explicitly provides it.
      - Do NOT invent intermediate constraints.
    - If a clue states an identity between two attributes (e.g., "The doctor is Eric"):
      - Use same_house(Occupation=doctor,Name=Eric) (or matching attribute names from the puzzle).
    - If a clue states a positional relation between two described entities:
      - Use left_of / immediately_left_of / right_of / adjacent / between with the two Attr=Val terms.

    SILENT SELF-CHECK (DO NOT OUTPUT THIS):
    Before finalizing "parsed_clues", re-read each original clue and confirm:
    - C<i> uses ONLY entities mentioned in clue i.
    - The predicate type matches the clue wording (left_of vs immediately_left_of etc.).
    - All Attr and Val tokens exist in attribute_values (after underscore normalization).
    If any of the above fails, fix the parsed clue.

B) parsed_reasoning (MANDATORY, PARSABLE)
    - "parsed_reasoning" MUST be a list of strings.
    - Each string must be exactly 1 sentence and end with a period.
    - There is NO LIMIT on the number of entries.
    - Each entry MUST follow this exact DSL format:

      S<k> [C<i>(+C<j>...)] <op>(<H>,<Attr>,<Val>).

    Where:
    - <k> is a step number starting at 1 and increasing by 1 each step.
    - Evidence inside [...] must reference clue ids, e.g. [C1] or [C1+C3].
    - <op> is either set or not.
    - <H> is a house number integer (1..N).
    - <Attr> and <Val> must come from the puzzle text (normalized).

    LOGICAL VALIDITY REQUIREMENT:
    - Every step in "parsed_reasoning" MUST be logically entailed by the parsed clues plus any earlier reasoning steps.
    - If you cannot deduce a set(...) fact with certainty, output a not(...) fact that is guaranteed true.

C) solution (MANDATORY TABLE)
    - "solution" MUST be in tabular form with:
      - "header": a list of column names
      - "rows": a list of rows, each row being a list of strings matching the header order.
    - The header MUST include "House" and then all attribute columns from the puzzle text.
    - The rows MUST list houses in increasing order from 1..N.
    - All solution VALUES must be normalized with underscores.

EXAMPLE DEMONSTRATION (illustration only)

Example Puzzle:
    There are 3 houses, numbered 1 to 3 from left to right.
    Each person has a unique name: Peter, Eric, Arnold.
    Each person has a unique drink: tea, water, milk.

Clues:
    1. Peter is in the second house.
    2. Arnold is directly left of the one who drinks water.
    3. The water drinker is directly left of the milk drinker.

Correct Example Output:
<answer>{
  "n_houses": 3,
  "attribute_values": {
    "Name": ["Peter", "Eric", "Arnold"],
    "Drink": ["tea", "water", "milk"]
  },
  "parsed_clues": [
    "C1 = set(2,Name,Peter).",
    "C2 = immediately_left_of(Name=Arnold,Drink=water).",
    "C3 = immediately_left_of(Drink=water,Drink=milk)."
  ],
  "parsed_reasoning": [
    "S1 [C1] set(2,Name,Peter).",
    "S2 [C3] not(3,Drink,water).",
    "S3 [C3] not(1,Drink,milk).",
    "S4 [C2] not(3,Name,Arnold).",
    "S5 [C2+C3] set(1,Name,Arnold)."
  ],
  "solution": {
    "header": ["House", "Name", "Drink"],
    "rows": [
      ["1", "Arnold", "tea"],
      ["2", "Peter", "water"],
      ["3", "Eric", "milk"]
    ]
  }
}</answer>
"""

SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

{puzzle}

Solve the puzzle above and provide parsed_reasoning parsed_clues and solution by returning ONLY the <answer>...</answer> block, with no additional text.
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
        final_grid = example['ground_truth']
        # Use the 'clues' field directly from the input data
        clues = example['clues']
        # user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        user_prompt = SOLUTION_PROMPT_ORDERED_V2 + SOLUTION_PROMPT_1_SHOT_USER.format(puzzle=example['instruction'])

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
    parser.add_argument('--input_path', default='/home/asif/data3/HF_cache/guru_data', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/guru_data_adjusted_1shot', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set')
    parser.add_argument('--data_source_train', default='our_zebra_puzzle_new_reward', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_zebra_puzzle_new_reward_test', help='Name of data source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)

    args.train_file = os.path.join(args.input_path,'train/logic__zebra_puzzle_1.3k.parquet')
    args.test_file = os.path.join(args.input_path,'online_eval/logic__zebra_puzzle_dataset_200.parquet')

    #args.output_dir = os.path.join(args.output_dir, 'guru_data_readjusted')


    # Load Training data set from JSON or Parquet based on file extension
    file_extension = os.path.splitext(args.train_file)[1].lower()
    if file_extension in ['.json', '.jsonl']:
        train_dataset = datasets.load_dataset('json', data_files=args.train_file)['train']
    elif file_extension == '.parquet':
        train_dataset = datasets.load_dataset('parquet', data_files=args.train_file)['train']
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only JSON, JSONL, and Parquet are supported.")


    # Load Test data set from JSON or Parquet based on file extension
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

    # Transform dataset
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

