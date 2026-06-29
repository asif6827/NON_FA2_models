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


GROUPING_SYSTEM_PROMPT = """
You are an expert AR-LSAT grouping-game solver.

This prompt is ONLY for AR-LSAT GROUPING problems.

You are given:
(i) one AR-LSAT grouping passage written in plain English,
(ii) one question about that passage,
(iii) a question_type label,
(iv) a dictionary of answer options, and
(v) optional metadata such as tags or entity hints if available.

Your task is to solve the grouping problem and produce the following TWO fields:
1) reasoning — natural-language reasoning steps that justify the answer.
2) solution — the final selected answer option.

Your final answer must contain exactly one <answer>...</answer> block.
The content inside <answer>...</answer> must be a single valid JSON object.

Any text outside <answer>...</answer>, including <think>...</think>, is ignored by the grader and receives zero reasoning credit.

Do not rely on <think> for the solution proof.
All graded reasoning must be repeated inside the JSON "reasoning" field.

If a <think> block is generated, keep it brief. The answer justification must be inside "reasoning".
Do not put thinking markers, markdown, comments, or explanations inside the <answer> block.

The grading system will evaluate only the first complete <answer>...</answer> block.

================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- The final graded output MUST contain exactly one <answer>...</answer> block.
- Anything outside the answer block is ignored by the grader, but the answer block itself must contain only valid JSON.
- Do NOT include extra text, markdown, explanations, or code fences.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly TWO top-level keys, spelled EXACTLY:
    "reasoning",
    "solution"
- Do NOT add any other keys.
- All formal expressions, if used, MUST be strings.

================================================================================
REASONING REQUIREMENTS FOR GROUPING
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly one sentence and end with a period.
- Do not write the reasoning as one paragraph summary.
- The reasoning must contain enough explicit deductions to justify the selected option.
- If the question asks what must be true, explain why the selected option is forced.
- If the question asks what could be true, explain why the selected option is consistent with the grouping rules.
- If the question asks what cannot be true, explain why the selected option violates the grouping rules.
- If the question asks for a complete and accurate list, explain why included cases are possible and excluded cases are impossible.
- When testing options, explicitly state which options are ruled out and why.
- The model-specific <think> block is ignored by the grader.
- Only the JSON "reasoning" list inside <answer>...</answer> is graded for reasoning quality.

================================================================================
SOLUTION REQUIREMENTS
================================================================================
- "solution" MUST be an object with exactly one key:
    "selected_option"
- The selected option MUST be the option letter only, e.g., "A", "B", "C", "D", or "E".
- Do not include option text inside "selected_option".

================================================================================
OUTPUT SCHEMA
================================================================================
<answer>{
  "reasoning": [],
  "solution": {
    "selected_option": ""
  }
}</answer>
"""


FEWSHOT_PASSAGE = """
Seven directors A, B, C, D, E, F, and G serve on either committee X or committee Y.

Rules:
1. If A is on X, then B is on Y.
2. If C is on X, then D and E are on Y.
3. F is on a different committee from G.
4. E is on a different committee from A.
5. If G is on X, then B is on X.
"""

FEWSHOT_QUESTION = "If D and F are both on X, which could be true?"

FEWSHOT_QUESTION_TYPE = "could_be_true"

FEWSHOT_OPTIONS = {
    "A": "A and C are on X",
    "B": "A and E are on Y",
    "C": "B and G are on X",
    "D": "C and E are on Y",
    "E": "G and E are on X",
}

FEWSHOT_ASSISTANT_ANSWER = """
<answer>{
  "reasoning": [
    "The question fixes D on committee X and F on committee X.",
    "Since F and G must be on different committees, G must be on committee Y.",
    "If C were on committee X, rule 2 would force D onto committee Y, which contradicts D being on committee X.",
    "Therefore C must be on committee Y.",
    "Option A is impossible because it places C on committee X.",
    "Option B is impossible because it places both A and E on committee Y even though A and E must be on different committees.",
    "Option C is impossible because it places G on committee X even though G must be on committee Y.",
    "Option E is impossible because it also places G on committee X.",
    "Option D is consistent because C can be on committee Y and E can also be on committee Y while A is on committee X and B is on committee Y.",
    "Thus option D could be true under all the given conditions."
  ],
  "solution": {
    "selected_option": "D"
  }
}</answer>
"""


GROUPING_USER_PROMPT_PHI = """
--------------------------------
AR-LSAT GROUPING PROBLEM TO SOLVE
--------------------------------

passage = {passage}

question = {question}

question_type = {question_type}

options = {options}

metadata = {metadata}

Solve the AR-LSAT grouping problem above and provide reasoning and solution.

Your final answer block MUST begin with the exact characters:
<answer>{{

Your JSON MUST contain the fields in this exact order:
1. "reasoning"
2. "solution"

Important reasoning-field rule:
The "reasoning" field must NOT be a paragraph summary.
It must be a list of explicit one-sentence deductions.
Every reasoning string must end with a period.
The model-specific <think> block is ignored by the grader.
Only the "reasoning" field inside <answer> is evaluated for reasoning quality.
Therefore, repeat the answer justification inside the "reasoning" field.

After the final reasoning string, immediately write the "solution" field.
The "solution" field must be the final top-level key and must not be omitted.
The "solution" object must contain only "selected_option".

After the complete solution field, close the JSON object and end with:
}}</answer>

Return only one complete <answer>...</answer> block with no additional text.

Reminder:
The grader ignores <think> and any text outside <answer>.
The only graded reasoning is the JSON "reasoning" list.
If the "reasoning" list is a summary without explicit deductions, the reasoning score is zero.
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


def to_json_text(value):
    """Serialize dictionaries/lists as JSON while leaving plain strings readable."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def make_metadata(example):
    """Collect optional metadata fields when they exist in the AR-LSAT examples."""
    metadata = {}
    for key in ("metadata", "tags", "tag", "entities", "entity_hints", "game_type"):
        if key in example and example[key] is not None:
            metadata[key] = example[key]
    return metadata if metadata else None


def make_map_fn_1_shot(split, data_source):
    def process_fn_1_shot(example, idx):
        # Use 'answer' as ground truth since that is what the input AR-LSAT data uses.
        final_grid = example["answer"]

        target_metadata = make_metadata(example)

        messages = [
            {
                "role": "system",
                "content": GROUPING_SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": GROUPING_USER_PROMPT_PHI.format(
                    passage=FEWSHOT_PASSAGE.strip(),
                    question=FEWSHOT_QUESTION,
                    question_type=FEWSHOT_QUESTION_TYPE,
                    options=json.dumps(FEWSHOT_OPTIONS, ensure_ascii=False),
                    metadata="null",
                ).strip(),
            },
            {
                "role": "assistant",
                "content": FEWSHOT_ASSISTANT_ANSWER.strip(),
            },
            {
                "role": "user",
                "content": GROUPING_USER_PROMPT_PHI.format(
                    passage=example["passage"],
                    question=example["question"],
                    question_type=example["question_type"],
                    options=to_json_text(example["options"]),
                    metadata=to_json_text(target_metadata),
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
    parser.add_argument('--data_path', default='/home/asif/data3/Codes_QCRI/AR-LSAT/processed_ar_lsat/', help='Path to json file')
    parser.add_argument('--data_setting', default='mlxl_train_mlxl_test', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/AR_LSAT_to_guru/', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--data_source_train', default='our_ar_lsat_grouping_new_reward_phi4', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_ar_lsat_grouping_new_reward_test_phi4', help='Name of data source')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

    if args.data_setting == 'mlxl_train_mlxl_test':
        args.train_data_file = os.path.join(args.data_path, 'AR_LSAT_train_grouping_300.json')
        args.test_data_file = os.path.join(args.data_path, 'AR_LSAT_test_grouping_49.json')
    else:
        raise ValueError('Invalid data_setting')
    args.output_dir = os.path.join(args.output_dir, args.data_setting)

    if args.data_setting == 'mlxl_train_mlxl_test':
        # Load dataset from JSON.
        train_dataset = datasets.load_dataset('json', data_files=args.train_data_file)['train']
        test_dataset = datasets.load_dataset('json', data_files=args.test_data_file)['train']

        # Transform dataset.
        process_train_fn = make_map_fn_1_shot('train', args.data_source_train)
        train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)

        process_test_fn = make_map_fn_1_shot('test', args.data_source_test)
        test_dataset = test_dataset.map(function=process_test_fn, with_indices=True)

    # Store the original training dataset size.
    original_train_size = len(train_dataset)

    # Sample the training dataset if needed.
    train_dataset = sample_dataset(train_dataset, args.train_sample_size)

    # Create output directories.
    train_output_dir = os.path.join(args.output_dir, "train")
    test_output_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Save train dataset.
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=f"logic_{args.data_source_train}",
        sample_size=args.train_sample_size if args.train_sample_size else len(train_dataset)
    )

    # Save test dataset.
    test_output_path = save_dataset(
        dataset=test_dataset,
        output_dir=test_output_dir,
        filename_prefix=f"logic_{args.data_source_test}",
        sample_size=len(test_dataset)
    )

    # Copy to HDFS if specified.
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
