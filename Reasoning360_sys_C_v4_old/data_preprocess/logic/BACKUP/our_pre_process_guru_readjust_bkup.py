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
3. Return the result STRICTLY as a single valid JSON object.

CRITICAL FORMAT REQUIREMENTS:
- Output ONLY a JSON object, NO natural language, NO markdown, NO code fences.
- The top-level JSON MUST have exactly two keys: "reasoning" and "solution".
- "reasoning" MUST be a SHORT English explanation (1–5 sentences, not more).
- "solution" MUST be an object with:
  - "header": a list of column names (e.g. ["House", "Name", "Pet", "..."])
  - "rows": a list of rows, where each row is a list of strings, one per column.

Example of the REQUIRED SHAPE (this is ONLY an example, not the answer):

{
  "reasoning": "Your step-by-step logic here, but concise.",
  "solution": {
    "header": ["House", "Name", "Pet", "..."],
    "rows": [
      ["1", "Eric", "cat", "..."],
      ["2", "Arnold", "dog", "..."]
    ]
  }
}

Do NOT include any text before or after the JSON.
"""

SOLUTION_PROMPT_USER_SOLUTION_BASED = """PUZZLE:
{puzzle}

Please provide your reasoning and solution:"""

# Instruction for zebra puzzle answers - 使用final_code-1中的格式要求
InstructionFollow = """Please provide your reasoning and solution in the required JSON format. The solution should be a grid with header and rows."""

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
        final_grid = example['ground_truth']
        
        # Use the 'clues' field directly from the input data
        clues = example['clues']
        
        # 使用系统prompt和用户prompt模板，使用'instruction'作为puzzle文本
        system_prompt = SOLUTION_PROMPT_SYSTEM_SOLUTION_BASED
        user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['instruction'])
        
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            'raw_prompt': [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
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
        return data
        
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='/home/asif/data3/HF_cache/guru_data', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/guru_data_adjusted', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--train_size', type=float, default=0.8, help='Proportion of data for train set')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for test set')
    parser.add_argument('--data_source', default='wwq_zebra_puzzle_new_reward', help='Name of data source')
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
    process_train_fn = make_map_fn('train', args.data_source)
    train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)

    # Transform dataset
    process_test_fn = make_map_fn('test', args.data_source)
    test_dataset = test_dataset.map(function=process_train_fn, with_indices=True)
    


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
        filename_prefix=f"logic__{args.data_source}",
        sample_size=args.train_sample_size if args.train_sample_size else len(train_dataset)
    )
    
    # Save test dataset
    test_output_path = save_dataset(
        dataset=test_dataset,
        output_dir=test_output_dir,
        filename_prefix=f"logic__{args.data_source}",
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
