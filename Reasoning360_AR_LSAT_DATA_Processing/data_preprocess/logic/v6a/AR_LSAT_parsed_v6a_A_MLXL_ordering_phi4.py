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


ORDERING_SYSTEM_PROMPT = """
You are an expert AR-LSAT ordering-game solver.

This prompt is ONLY for AR-LSAT ORDERING problems.

You are given:
(i) one AR-LSAT ordering passage written in plain English,
(ii) one question about that passage,
(iii) a question_type label,
(iv) a dictionary of answer options, and
(v) optional metadata such as tags or entity hints if available.

Your task is to solve the ordering problem and produce the following TWO fields:
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
REASONING REQUIREMENTS FOR ORDERING
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly one sentence and end with a period.
- Do not write the reasoning as one paragraph summary.
- The reasoning must contain enough explicit deductions to justify the selected option.
- If the question asks what must be true, explain why the selected option is forced.
- If the question asks what could be true, explain why the selected option is consistent with the ordering rules.
- If the question asks what cannot be true, explain why the selected option violates the ordering rules.
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
- Do not include a final table unless the question explicitly asks for a complete ordering.

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
Four speakers A, B, C, and D speak in positions 1 through 4, with exactly one speaker in each position.

Rules:
1. A speaks before B.
2. C speaks immediately after A.
3. D does not speak first.
"""

FEWSHOT_QUESTION = "If B speaks fourth, which one of the following could be true?"

FEWSHOT_QUESTION_TYPE = "could_be_true"

FEWSHOT_OPTIONS = {
    "A": "A speaks second",
    "B": "C speaks fourth",
    "C": "D speaks third",
    "D": "A speaks third",
    "E": "C speaks first",
}

FEWSHOT_ASSISTANT_ANSWER = """
<answer>{
  "reasoning": [
    "The question condition fixes B in the fourth position.",
    "Since C must speak immediately after A, A and C must occupy two consecutive positions before B.",
    "The consecutive pair A and C cannot be positions 2 and 3 because that would force D into position 1, which violates the rule that D does not speak first.",
    "Therefore A must be in position 1 and C must be in position 2.",
    "With B fixed in position 4, the only remaining position for D is position 3.",
    "Option A is impossible because A must be first rather than second.",
    "Option B is impossible because C must be second rather than fourth.",
    "Option C is consistent because D is forced to speak third.",
    "Option D is impossible because A cannot be third.",
    "Option E is impossible because C cannot be first.",
    "Thus option C could be true."
  ],
  "solution": {
    "selected_option": "C"
  }
}</answer>
"""


ORDERING_USER_PROMPT_PHI = """
--------------------------------
AR-LSAT ORDERING PROBLEM TO SOLVE
--------------------------------

passage = {passage}

question = {question}

question_type = {question_type}

options = {options}

metadata = {metadata}

Solve the AR-LSAT ordering problem above and provide reasoning and solution.

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
                "content": ORDERING_SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": ORDERING_USER_PROMPT_PHI.format(
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
                "content": ORDERING_USER_PROMPT_PHI.format(
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
