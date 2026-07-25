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
You are an expert solver of Knights and Knaves logic puzzles.

You are given (i) one Knights and Knaves puzzle written in plain English, and
(ii) people: the complete ordered list of people appearing in the puzzle.

Every person is exactly one of the following:

- Knight: always makes a true statement.
- Knave: always makes a false statement.

To reason about a statement made by person P, remember that P is a Knight if
and only if the complete statement uttered by P is true, while P is a Knave if
and only if that statement is false.

Your job is to generate the following TWO fields:

1) reasoning
2) solution

You MUST return the result STRICTLY as a single valid JSON object wrapped inside
<answer>...</answer>.

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
- Person names MUST be drawn exclusively from the supplied people list.
- Preserve every person's name exactly as supplied, including spelling, casing,
  punctuation, and spaces.
- Do NOT invent, omit, rename, merge, or reorder people.
- Use only the identity labels "Knight" and "Knave", with exactly this casing.
- Interpret every person's entire spoken statement, including conjunctions,
  disjunctions, negations, implications, equivalences, and references to self.
- A speaker's identity must agree with the truth value of the speaker's complete
  statement: a Knight's statement is true and a Knave's statement is false.
- Do NOT treat a quoted claim as an independent fact without accounting for the
  identity of the person who said it.
- The final classification must satisfy all statements simultaneously.

================================================================================
1) reasoning
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly 1 sentence and end with a period.
- Explain the deductions step by step in clear natural language.
- Explicitly account for how each speaker's identity is linked to the truth or
  falsity of that speaker's statement.
- Use contradiction or case analysis when necessary.
- Do NOT include raw Python, Z3 code, solver commands, JSON fragments, or formal
  labels such as C1 or S1 in the reasoning.
- Do NOT merely restate the puzzle; every sentence should advance or verify the
  solution.
- The reasoning must support the complete final classification.

================================================================================
2) solution (MANDATORY TABLE)
================================================================================
- "solution" MUST be in tabular form with exactly:
  - "header": a list of column names
  - "rows": a list of rows, each row being a list of strings matching the
    header order
- The header MUST be exactly:
    ["Person", "Identity"]
- Each row MUST have exactly two entries:
    [person_name, identity]
- identity MUST be exactly "Knight" or "Knave".
- Include every supplied person exactly once.
- List rows in exactly the same order as the supplied people list.
- Do NOT include additional columns, summaries, confidence scores, Boolean
  labels, or explanations in the solution table.

================================================================================
ONE-SHOT EXAMPLE
================================================================================

Example Puzzle:
Three inhabitants—Alice, Bob, and Charlie—live on an island where every person
is either a Knight, who always tells the truth, or a Knave, who always lies.

Alice says, "Bob is a Knight."
Bob says, "Alice and Charlie are of different types."
Charlie says, "Bob is a Knave."

people = ["Alice", "Bob", "Charlie"]

Correct Example Output:
<answer>{
  "reasoning": [
    "Alice's statement means that Alice and Bob must have the same identity because Alice is truthful exactly when Bob is a Knight.",
    "Charlie's statement means that Charlie and Bob must have opposite identities because Charlie is truthful exactly when Bob is a Knave.",
    "Assuming Bob is a Knave forces Alice to be a Knave and Charlie to be a Knight.",
    "Under that assumption Alice and Charlie have different identities, which makes Bob's statement true and contradicts Bob being a Knave.",
    "Therefore Bob must be a Knight.",
    "Since Alice has the same identity as Bob, Alice must also be a Knight.",
    "Since Charlie has the opposite identity from Bob, Charlie must be a Knave.",
    "The resulting identities satisfy all three statements simultaneously."
  ],
  "solution": {
    "header": ["Person", "Identity"],
    "rows": [
      ["Alice", "Knight"],
      ["Bob", "Knight"],
      ["Charlie", "Knave"]
    ]
  }
}</answer>
"""


SOLUTION_PROMPT_1_SHOT_USER = r"""
--------------------------------
PUZZLE TO SOLVE
--------------------------------

{puzzle}

people = {people}

Solve the puzzle above and provide reasoning and solution for this puzzle in the
<answer>...</answer> block, with no additional text.
"""


from typing import Any


def solution_table_from_knights_and_knaves(
    names: list[str],
    labels: list[bool],
) -> dict[str, list]:
    """
    Convert the dataset's Boolean solution into the output table expected
    by the Knights-and-Knaves prompt.

    True  -> Knight
    False -> Knave
    """
    if len(names) != len(labels):
        raise ValueError(
            f"Number of names ({len(names)}) does not match "
            f"number of labels ({len(labels)})."
        )

    rows = [
        [name, "Knight" if label else "Knave"]
        for name, label in zip(names, labels)
    ]

    return {
        "header": ["Person", "Identity"],
        "rows": rows,
    }


def make_map_fn_1_shot(split: str, data_source: str):
    def process_fn_1_shot(
        example: dict[str, Any],
        idx: int,
    ) -> dict[str, Any]:

        # Dataset fields:
        #   quiz: puzzle text
        #   names: ordered list of people
        #   solution: ordered Boolean labels
        puzzle = example["quiz"]
        people = example["names"]
        labels = example["solution"]

        final_solution = solution_table_from_knights_and_knaves(
            names=people,
            labels=labels,
        )

        user_prompt = (
            SOLUTION_PROMPT_1_SHOT_SYS
            + SOLUTION_PROMPT_1_SHOT_USER.format(
                puzzle=puzzle,
                people=people,
            )
        )

        example_id = example.get(
            "id",
            example.get("index", str(idx)),
        )

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            "raw_prompt": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            "ability": "logical_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": final_solution,
            },
            "apply_chat_template": False,
            "extra_info": {
                "id": example_id,
                "split": split,
                "people": people,
                "boolean_solution": labels,
                "solution_text": example.get("solution_text"),
                "n_people": len(people),
            },
        }
        if idx == 0:
            print(
                f"data_source: {data_source}, "
                f"split: {split}, idx: {idx}"
            )
            print(
                "\n"
                + "=" * 100
                + f"\n{data_source} | {split} | {idx}\n"
                + "=" * 100
            )
            print(data)
            print("\n")

        return data

    return process_fn_1_shot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/asif/data3/HF_cache/knights_and_knaves_300_train/', help='Path to json file')
    parser.add_argument('--data_setting', default='mlxl_train_mlxl_test', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/K_K_to_guru/', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--data_source_train', default='our_k_k_new_reward', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_k_k_new_reward_test', help='Name of data source')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

    if args.data_setting == 'mlxl_train_mlxl_test':
        args.train_data_file = os.path.join(args.data_path, 'train_all_num300.jsonl')
        args.test_data_file = os.path.join(args.data_path, 'test_all_num700.jsonl')
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
