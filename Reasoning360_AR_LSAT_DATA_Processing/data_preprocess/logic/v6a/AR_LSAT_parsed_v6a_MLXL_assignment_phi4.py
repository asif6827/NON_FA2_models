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


ASSIGNMENT_SYSTEM_PROMPT = """
You are an expert AR-LSAT assignment-game solver.

All AR-LSAT assignment problems in this dataset are complete, consistent, and solvable under their answer choices.
Never say that the problem is incomplete, impossible, too complex, or cannot be solved.
Always produce the required final answer block.

Your final answer must contain exactly one <answer>...</answer> block.
The content inside <answer>...</answer> must be a single valid JSON object.

Any text outside <answer>...</answer>, including <think>...</think>, is ignored by the grader and receives zero reasoning credit.
Do not rely on <think> for the solution proof.
All graded deduction steps must be repeated inside the JSON "reasoning" field.

If a <think> block is generated, keep it brief.
The formal proof must be inside "reasoning".
Do not put thinking markers, markdown, comments, or explanations inside the <answer> block.

The grading system will evaluate only the first complete <answer>...</answer> block.

You are given:
(i) one AR-LSAT assignment passage written in plain English,
(ii) one question about that passage,
(iii) a question_type label,
(iv) a dictionary of answer options,
and optionally
(v) metadata such as tags or entity hints if available.

This prompt is ONLY for ASSIGNMENT problems.

Your task is to parse the assignment problem into a solver-oriented logical representation and determine the correct answer by generating the following EIGHT fields:
1) problem_type — must be "assignment".
2) world_model — entities, attribute domains, and structural assumptions.
3) rules — formalized passage rules only.
4) facts — question-specific temporary conditions only.
5) question_semantics — how the options must be evaluated using the provided question_type.
6) options — formalized answer options.
7) reasoning — interleaved natural-language reasoning and formal solver-oriented steps.
8) solution — the final selected answer option.

You MUST return the result STRICTLY as a single valid JSON object wrapped inside:
<answer>...</answer>

The graded content must be inside exactly one <answer>...</answer> block.
Anything outside the answer block is ignored by the grader.
Inside the answer block, output only a single valid JSON object.

================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- The final graded output MUST contain exactly one <answer>...</answer> block.
- Anything outside the answer block is ignored by the grader, but the answer block itself must contain only valid JSON.
- JSON MUST contain EXACTLY the 8 required keys.
- The JSON object MUST have exactly EIGHT top-level keys, spelled EXACTLY:
    "problem_type",
    "world_model",
    "rules",
    "facts",
    "question_semantics",
    "options",
    "reasoning",
    "solution"
- "problem_type" MUST be exactly "assignment".
- All formal expressions MUST be strings.
- Do NOT include extra text, markdown, explanations, or code fences.
- Do NOT add any other keys.

CRITICAL:
- "problem_type" is NOT the question type.
- "problem_type" MUST always be exactly "assignment".
- The input question_type must appear only here:
  "question_semantics": {
    "question_type": "<input question_type>",
    "option_interpretation_rule": "..."
  }
- Never set "problem_type" to "must_be_true", "could_be_true", "cannot_be_true", "acceptability",

================================================================================
NORMALIZATION RULES FOR ASSIGNMENT
================================================================================
- Use concise symbolic tokens.
- Preserve entity names exactly (A, B, C, etc.).
- Represent assignments using:

    Assign(entity, value)

- Each entity must be assigned exactly one value per attribute.
- Do NOT use positional operators like <, > unless explicitly required.

================================================================================
PARSING INSTRUCTIONS FOR ASSIGNMENT
================================================================================

Construct world_model:
- Extract entities.
- Extract attribute domains (e.g., floors, cities, colors).
- Add assumptions:
    each entity is assigned exactly one value,
    assignments are consistent across attributes.

Parse rules:
- Include ONLY passage constraints.
- Do NOT include question facts.

Parse facts:
- Include temporary assumptions from question.

Parse question semantics:
- Use question_type mapping exactly as provided.

Parse options:
- Represent using Assign(...) expressions.

================================================================================
ALLOWED FORMAL OPERATORS FOR ASSIGNMENT
================================================================================

Assignment:
    Assign(A, X)

Equality:
    Assign(A, X) == Assign(B, Y)
    Assign(A, X) != Assign(B, Y)

Boolean:
    And(...)
    Or(...)
    Not(...)
    Implies(...)
    Xor(...)

Counting:
    AtLeast(k, ...)
    AtMost(k, ...)
    Exactly(k, ...)

Solver:
    Sat(...)
    Unsat(...)

================================================================================
ASSIGNMENT EXPRESSION GUIDE
================================================================================

A is assigned to X:
    Assign(A, X)

A and B share same attribute:
    Assign(A, X) == Assign(B, X)

A and B have different attributes:
    Assign(A, X) != Assign(B, X)

If A has X then B has Y:
    Implies(Assign(A, X), Assign(B, Y))

Exactly one assignment:
    Exactly(1, Assign(A, X1), Assign(A, X2), ...)

================================================================================
REASONING REQUIREMENTS FOR ASSIGNMENT
================================================================================

- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly one sentence and end with a period.
- Reasoning MUST be interleaved:
    Odd-numbered entries: natural-language reasoning.
    Even-numbered entries: formal solver-oriented step.

Formal steps MUST:
- Start with "S<k>: "
- End with a period.
- Be logically valid and solver-verifiable.

Allowed step types:

- Direct facts:
    S1: Assign(A, X).

- Derived assignments:
    S2: Assign(B, Y).

- Exclusions:
    S3: Not(Assign(C, Z)).

- Equality relations:
    S4: Assign(A, X) == Assign(B, X).

- Inequality relations:
    S5: Assign(A, X) != Assign(B, X).

- Conditional rules:
    S6: Implies(Assign(A, X), Assign(B, Y)).

- Counting constraints:
    S7: Exactly(1, Assign(A, X), Assign(A, Y)).

- Option feasibility:
    S8: Sat(Option_C).

- Option contradiction:
    S9: Unsat(Option_A).

The "reasoning" field MUST follow this exact pair structure:
    "reasoning": [
    "Natural-language explanation.",
    "S1: formal_step.",
    "Natural-language explanation.",
    "S2: formal_step.",
    "Natural-language explanation.",
    "S3: formal_step."
    ]

FORMAL STEP HARD REQUIREMENT:
Every S-step MUST contain at least one allowed formal operator:
Assign(...), Not(...), And(...), Or(...), Implies(...), Exactly(...), AtLeast(...), AtMost(...), Sat(...), Unsat(...).

Invalid S-steps:
- S1: French novels = 1, Russian novels = 2.
- S2: Option A satisfies all rules.
- S3: A is assigned to P2.
- S4: French novels = 1 ≥ Russian novels = 2.

Valid S-steps:
- S1: Assign(A, P2).
- S2: Not(Assign(B, P2)).
- S3: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2)).
- S4: Sat(Option_C).
- S5: Unsat(Not(Option_A)).

Rules:
* Natural-language entries must NOT start with any label.
* Natural-language entries must be plain explanatory sentences.
* S entries must start with S1:, S2:, S3:, ...
* The list must strictly alternate:
  natural-language sentence, S-step, natural-language sentence, S-step, ...
* Every S entry must be solver-checkable.
* Every S entry must end with a period.
* Do not write paragraph summaries.
* Do not use unsupported notation such as arrows, quotes around formal expressions, or table-row summaries.
* The model-specific <think> block is ignored by the grader.
* Only the JSON "reasoning" list inside <answer>...</answer> is graded for reasoning quality.
* Therefore, any formal deduction written in <think> must be repeated inside the JSON "reasoning" field.
* Each S-step must be exactly one sentence.
* Do not put explanations inside S-steps.
* Put explanations only in the natural-language sentence immediately before the S-step.

Logical validity requirements:
- Each step must follow from rules + facts + prior steps.
- No contradictions.
- No tautologies.
- No hallucinated assumptions.
- No ordering operators unless explicitly required.

================================================================================
SOLUTION REQUIREMENTS
================================================================================

"solution": {
  "selected_option": "X"
}

================================================================================
OUTPUT SCHEMA
================================================================================

<answer>{
  "problem_type": "assignment",
  "world_model": {
    "entities": [],
    "domains": {
      "values": []
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
"""

ASSIGNMENT_FEWSHOT_USER_PROMPT = """
--------------------------------
AR-LSAT ASSIGNMENT EXAMPLE
--------------------------------

Example Passage:
Three employees A, B, and C are each assigned to one of three projects P1, P2, and P3.

Rules:
1. A is not assigned to P1.
2. B is assigned to the same project as C.
3. Exactly one employee is assigned to P2.

Example Question:
Which of the following must be true?

question_type:
must_be_true

Options:
A. A is assigned to P2
B. B is assigned to P3
C. C is assigned to P3
D. B is assigned to P2
E. A is assigned to P3

Solve the AR-LSAT assignment example and return problem_type, world_model, rules, facts, question_semantics, options, reasoning, and solution inside a single <answer>...</answer> block, with no additional text.
"""

ASSIGNMENT_FEWSHOT_ASSISTANT_ANSWER = """
<answer>{
  "problem_type": "assignment",
  "world_model": {
    "entities": ["A","B","C"],
    "domains": {"values": ["P1","P2","P3"]},
    "structural_assumptions": [
      "each entity is assigned exactly one value",
      "assignments are mutually exclusive"
    ]
  },
  "rules": [
    "Not(Assign(A, P1))",
    "Assign(B, P1) == Assign(C, P1)",
    "Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))"
  ],
  "facts": [],
  "question_semantics": {
    "question_type": "must_be_true",
    "option_interpretation_rule": "choose option whose negation is unsatisfiable"
  },
  "options": {
    "A": "Assign(A, P2)",
    "B": "Assign(B, P3)",
    "C": "Assign(C, P3)",
    "D": "Assign(B, P2)",
    "E": "Assign(A, P3)"
  },
  "reasoning": [
  "Exactly one employee is assigned to P2.",
  "S1: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2)).",
  "B and C must share the same project, so they cannot be the unique single employee assigned to P2.",
  "S2: Not(Assign(B, P2)).",
  "Since B and C share the same project, C also cannot be assigned to P2.",
  "S3: Not(Assign(C, P2)).",
  "Therefore A must be the one employee assigned to P2.",
  "S4: Assign(A, P2).",
  "The negation of Option A is impossible under the rules.",
  "S5: Unsat(Not(Option_A))."
  ],
  "solution": {
    "selected_option": "A"
  }
}</answer>
"""

ASSIGNMENT_USER_PROMPT_PHI = """
--------------------------------
AR-LSAT ASSIGNMENT PROBLEM TO SOLVE
--------------------------------

passage = {passage}

question = {question}

question_type = {question_type}

options = {options}

metadata = {metadata}

Solve the AR-LSAT assignment problem above and provide problem_type, world_model, rules, facts, question_semantics, options, reasoning, and solution.

Your final answer block MUST begin with the exact characters:
<answer>{{

Your JSON MUST contain the fields in this exact order:
1. "problem_type"
2. "world_model"
3. "rules"
4. "facts"
5. "question_semantics"
6. "options"
7. "reasoning"
8. "solution"

Important reasoning-field rule:
The "reasoning" field must NOT be a paragraph summary.
It must be an alternating list:
natural-language sentence, syntactic/formal step, natural-language sentence, syntactic/formal step, ...

Every formal step must start with S1:, S2:, S3:, etc.
The model-specific <think> block is ignored by the grader.
Only the "reasoning" field inside <answer> is evaluated for reasoning quality.
Therefore, repeat the formal deduction steps inside the "reasoning" field.

After the final reasoning string, immediately write the "solution" field.
The "solution" field must be the final top-level key and must not be omitted.

After the complete solution field, close the JSON object and end with:
}}</answer>

Return only one complete <answer>...</answer> block with no additional text.

Reminder:
The grader ignores <think> and any text outside <answer>.
The only graded reasoning is the JSON "reasoning" list.
If the "reasoning" list is a summary without S1:, S2:, S3: steps, the reasoning score is zero.
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

def to_json_text(value):
    """Serialize prompt fields deterministically and safely for model input."""
    return json.dumps(value, ensure_ascii=False)


def make_metadata(example):
    """Keep optional metadata if present, without changing the core AR-LSAT fields."""
    metadata = {}
    for key in ("tags", "entities", "entity_hints", "game_type", "source", "difficulty"):
        if key in example and example[key] is not None:
            metadata[key] = example[key]
    return metadata if metadata else None


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


def make_map_fn_1_shot(split, data_source):
    def process_fn_1_shot(example, idx):
        # Use 'answer' as the rule-based ground truth for AR-LSAT.
        final_grid = example['answer']

        metadata = make_metadata(example)

        messages = [
            {
                "role": "system",
                "content": ASSIGNMENT_SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": ASSIGNMENT_FEWSHOT_USER_PROMPT.strip(),
            },
            {
                "role": "assistant",
                "content": ASSIGNMENT_FEWSHOT_ASSISTANT_ANSWER.strip(),
            },
            {
                "role": "user",
                "content": ASSIGNMENT_USER_PROMPT_PHI.format(
                    passage=example['passage'],
                    question=example['question'],
                    question_type=example['question_type'],
                    options=to_json_text(example['options']),
                    metadata=to_json_text(metadata),
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
    parser.add_argument('--data_source_train', default='our_ar_lsat_assignment_new_reward_phi4', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_ar_lsat_assignment_new_reward_test_phi4', help='Name of data source')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

    if args.data_setting == 'mlxl_train_mlxl_test':
        args.train_data_file = os.path.join(args.data_path, 'AR_LSAT_train_assignment_300.json')
        args.test_data_file = os.path.join(args.data_path, 'AR_LSAT_test_assignment_69.json')
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