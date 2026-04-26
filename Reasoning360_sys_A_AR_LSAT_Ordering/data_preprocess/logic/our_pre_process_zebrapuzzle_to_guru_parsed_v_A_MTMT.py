import os
import sys
import random
import datasets
import argparse
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset

SOLUTION_PROMPT_1_SHOT_SYS = """
You are an expert logic puzzle solver.

You are given (i) one logic puzzle written in plain English, (ii) solution_header: that lists the attribute names used in the puzzle, and (iii) attribute_values: a dictionary of attribute values specifying the allowed values for each attribute.


Your job is to generate the following TWO fields:

1) reasoning
2) solution

You MUST return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Do NOT include extra text, markdown, explanations, or code fences.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly TWO top-level keys, spelled EXACTLY:
    "reasoning",
    "solution"
- Do NOT add any other keys.

================================================================================
NORMALIZATION RULES
================================================================================
- Use underscores instead of spaces in VALUES (e.g., grilled_cheese, very_short).
- Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Animal, Occupation, Sport, Height, etc.
- House numbers are integers 1..N.
- Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6, etc.
- Do NOT invent values. Every value MUST be one of the allowed values listed in the puzzle text (after normalization).
- If the clue mentions a bare person name (e.g., "Arnold"), treat it as Name=Arnold.
- If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching attribute/value from the puzzle text.

================================================================================
1) reasoning
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly 1 sentence and end with a period.


================================================================================
2) solution (MANDATORY TABLE)
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
    "Name": ["Arnold", "Peter", "Eric"],
    "Color": ["white", "red", "yellow"],
    "Children": ["Fred", "Bella", "Meredith"]
  }

Correct Example Output:
<answer>{
  "reasoning": [
    "Clue 3 immediately anchors the red favorite color in the second house, which is a very strong positional fact to start from.",
    "Clue 1 then ties Arnold directly to the red color, so Arnold must be in that same second house.",
    "Putting those two together, Arnold is fixed in house 2. At this point, house 2 is completely identified as “Arnold’s house,” and we know it has the red color.",
    "Clue 4 gives us another concrete placement: the child Bella is in the first house. So whatever person lives in house 1, their child must be Bella.",
    "So far, we know: House 1 has child Bella, House 2 has Arnold and the color red, House 3 is still entirely open. Now we look at Clue 2, which introduces a relative ordering: the person whose child is Fred is somewhere to the left of Eric. This doesn’t give a house yet, but it constrains the ordering.",
    "Since houses are only 1 through 3, Eric cannot be in the first house (there would be nothing to the left of him). That means Eric must be in house 2 or house 3.",
    "But we already know Arnold occupies house 2, and all people are distinct. So Eric cannot be in house 2 and must therefore be in house 3.",
    "This is a good point to pause and take stock again. House 1: unknown person, child Bella. House 2: Arnold, red. House 3: Eric. Now, going back to the same ordering constraint (Fred < Eric), if Eric is in house 3, then Fred must be in house 1 or house 2.",
    "But we already know the child in house 1 is Bella, and children are unique. So Fred cannot be in house 1. That forces Fred into house 2.",
    "This tells us that Arnold, who is in house 2, is also the parent of Fred. At this stage, all people except Peter are placed: Arnold is in house 2 and Eric is in house 3. Since each house has exactly one person, Peter must be in the remaining house, house 1.",
    "Let’s summarize again. House 1: Peter, child Bella. House 2: Arnold, red, child Fred. House 3: Eric. Now consider the children again. Bella is in house 1 and Fred is in house 2, so the only remaining child, Meredith, must be in house 3.",
    "Clue 5 connects the color white to Meredith’s parent, meaning the white color must be in the same house as Meredith.",
    "Since Meredith is in house 3, white must be in house 3 as well.",
    "At this point, two colors are fixed: red in house 2 and white in house 3. Colors are unique, so the only remaining color, yellow, must belong to house 1.",
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



SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

{puzzle}

solution_header = {solution_header}

attribute_values = {attribute_values}

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
            #v = "_".join(row[i].split(" "))
            v = row[i]
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
    elif args.data_setting == 'med_train_med_test':
        args.data_file = os.path.join(args.data_path, 'Zebra_Puzzle_medium_280.json')
        pass
    else:
        raise ValueError('Invalid data_setting')
    args.output_dir = os.path.join(args.output_dir, args.data_setting)



    if args.data_setting == 'med_train_med_test':
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
