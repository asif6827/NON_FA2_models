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
SOLUTION_PROMPT_SYSTEM_SOLUTION_BASED = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
1. Analyze the clues step by step.
2. Derive a correct final solution.
3. Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE (no extra text, no markdown, no code fences).
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly two top-level keys: "reasoning" and "solution".
- "reasoning" MUST be a SHORT English explanation (1–5 sentences, not more).
- "solution" MUST be an object with:
  - "header": a list of column names (e.g. ["House", "Name", "Pet", "..."])
  - "rows": a list of rows, where each row is a list of strings, one per column.

EXACT OUTPUT TEMPLATE (this is ONLY a template, not the answer):

<answer>{
  "reasoning": "Your step-by-step logic here, but concise.",
  "solution": {
    "header": ["House", "Name", "Pet", "..."],
    "rows": [
      ["1", "Eric", "cat", "..."],
      ["2", "Arnold", "dog", "..."]
    ]
  }
}</answer>

Do NOT include any text before or after the <answer>...</answer> block. \n\n"""

SOLUTION_PROMPT_USER_SOLUTION_BASED = """PUZZLE:
{puzzle}

Please provide your reasoning and solution:"""

## Instruction for zebra puzzle answers - 使用final_code-1中的格式要求
#InstructionFollow = """Please output ONLY <answer>{...}</answer>, where {...} is a single valid JSON object with exactly two keys "reasoning" and "solution"; "solution" must be a grid with "header" and "rows" (all strings)."""



# Import prompt templates directly
SOLUTION_PROMPT_1_SHOT_SYS = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
1. Analyze the clues step by step.
2. Derive a correct final solution.
3. Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Do NOT include extra text, markdown, explanations, or code fences.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly two top-level keys: "reasoning" and "solution".
- "reasoning" MUST be a LIST of SHORT English strings (each exactly 1 sentence).
- The list MUST contain between 1 and 5 items.
- "solution" MUST be in tabular form with:
  - "header": a list of column names
  - "rows": a list of rows, each row being a list of strings matching the header order.

EXACT OUTPUT TEMPLATE (this is ONLY a template, not the answer):

<answer>{
  "reasoning": [
    "First logical deduction.",
    "Second logical deduction.",
    "Final conclusion."
  ],
  "solution": {
    "header": ["House", "Name", "Drink", "Hobby"],
    "rows": [
      ["1", "Eric", "milk", "photography"],
      ["2", "Peter", "water", "cooking"],
      ["3", "Arnold", "tea", "gardening"]
    ]
  }
}</answer>

--------------------------------
EXAMPLE (for illustration only)
--------------------------------

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
  "reasoning": [
    "Peter is fixed in House 2 by Clue 1.",
    "Arnold must be in House 1 to be directly left of the water drinker.",
    "This places water in House 2 and milk in House 3, leaving tea for Arnold."
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

Solve the puzzle above and provide your reasoning and solution by returning ONLY the <answer>...</answer> block, with no additional text.
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


def make_map_fn(split, data_source):
    def process_fn(example, idx):
        # Use 'ground_truth' instead of 'solution' since that's what the input data has
        final_grid = example['solution']
        
        # Use the 'clues' field directly from the input data
        clues = extract_clues_from_puzzle(puzzle_text=example['puzzle'])
        
        system_prompt = SOLUTION_PROMPT_SYSTEM_SOLUTION_BASED
        #user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        user_prompt = SOLUTION_PROMPT_SYSTEM_SOLUTION_BASED + SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        
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
        
    return process_fn


def make_map_fn_1_shot(split, data_source):
    def process_fn_1_shot(example, idx):
        # Use 'ground_truth' instead of 'solution' since that's what the input data has
        final_grid = example['solution']
        # Use the 'clues' field directly from the input data
        clues = extract_clues_from_puzzle(puzzle_text=example['puzzle'])
        # user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        user_prompt = SOLUTION_PROMPT_1_SHOT_SYS + SOLUTION_PROMPT_1_SHOT_USER.format(puzzle=example['puzzle'])

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
    parser.add_argument('--data_setting', default=None, help='Path to json file')
    parser.add_argument('--output_dir', default=None, help='Directory to save processed data')
    parser.add_argument('--shot', default='zero', help='Directory to save processed data')
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

    if args.shot == 'zero':
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
            process_train_fn = make_map_fn('train', args.data_source_train)
            train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)

            process_test_fn = make_map_fn('train', args.data_source_test)
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
            process_train_fn = make_map_fn('train', args.data_source_train)
            process_test_fn = make_map_fn('test', args.data_source_test)
            train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)
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


    elif args.shot=='one':
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
    else:
        print("NOT WORKING")