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

All AR-LSAT grouping problems in this dataset are complete, consistent, and solvable under their answer choices.
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
(i) one AR-LSAT grouping passage written in plain English,
(ii) one question about that passage,
(iii) a question_type label,
(iv) a dictionary of answer options,
and optionally
(v) metadata such as tags or entity hints if available.

This prompt is ONLY for GROUPING problems.

Your task is to parse the grouping problem into a solver-oriented logical representation and determine the correct answer by generating the following EIGHT fields:
1) problem_type — must be "grouping".
2) world_model — entities, groups, and structural assumptions.
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
- "problem_type" MUST be exactly "grouping".
- All formal expressions MUST be strings.
- Do NOT include extra text, markdown, explanations, or code fences.
- Do NOT add any other keys.

================================================================================
NORMALIZATION RULES FOR GROUPING
================================================================================
- Use concise symbolic tokens.
- Preserve entity names exactly (A, B, C, etc.).
- Use group labels exactly as defined (X, Y, Z, Shelf1, Shelf2, etc.).
- Represent assignments using:

    Assign(entity, group)

- Do NOT use numeric positions unless explicitly required.
- Each entity must belong to exactly one group.

================================================================================
PARSING INSTRUCTIONS FOR GROUPING
================================================================================

Construct world_model:
- Extract entities.
- Extract groups.
- Add assumptions:
    each entity belongs to exactly one group,
    groups are mutually exclusive.

Parse rules:
- Use ONLY passage rules.
- Do NOT include question facts here.

Parse facts:
- Add temporary conditions from question.

Parse question semantics:
- Use provided question_type.

Parse options:
- Represent using Assign(...) expressions.

================================================================================
ALLOWED FORMAL OPERATORS FOR GROUPING
================================================================================

Assignment:
    Assign(A, X)

Equality:
    Assign(A, X) == Assign(B, X)
    Assign(A, X) != Assign(B, X)

Boolean:
    And(...)
    Or(...)
    Not(...)
    Implies(...)

Counting:
    AtLeast(k, ...)
    AtMost(k, ...)
    Exactly(k, ...)

Solver:
    Sat(...)
    Unsat(...)

================================================================================
GROUPING EXPRESSION GUIDE
================================================================================

A is in group X:
    Assign(A, X)

A and B are in same group:
    Assign(A, X) == Assign(B, X)

A and B are in different groups:
    Assign(A, X) != Assign(B, X)

If A is in X then B is in Y:
    Implies(Assign(A, X), Assign(B, Y))

Exactly k elements in X:
    Exactly(k, Assign(A, X), Assign(B, X), ...)

================================================================================
REASONING REQUIREMENTS FOR GROUPING PROBLEMS
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly one sentence and end with a period.
- Reasoning MUST be interleaved:
    Odd-numbered entries: natural-language reasoning.
    Even-numbered entries: formal solver-oriented step.
- Natural-language entries must explain why the next formal step follows from rules, facts, earlier steps, or option testing.
- Formal entries must encode a newly derived grouping fact, membership restriction, counting restriction, option feasibility result, or forced/impossible group assignment.

Formal step format:
- Every formal step MUST start with "S<k>: " and MUST end with a period.
- <k> starts at 1 and increments by 1 for each formal step only.
- Formal steps must be solver-verifiable and may use ONLY:
    Assign(entity, group), ==, !=, Not(...), And(...), Or(...), Xor(...), Implies(...), AtLeast(k, ...), AtMost(k, ...), Exactly(k, ...), Sat(...), Unsat(...)

Atomic grouping expressions:
    Assign(A, X)               A belongs to group X.
    Not(Assign(A, X))          A does not belong to group X.
    Assign(A, X) == Assign(B, X)   A and B have the same X-membership status.
    Assign(A, X) != Assign(B, X)   A and B have different X-membership status.
    Implies(Assign(A, X), Assign(B, Y))   If A is in X, then B is in Y.

Boolean operators:
    Not(e)
    And(e1, e2, ..., en)
    Or(e1, e2, ..., en)
    Xor(e1, e2)
    Implies(e1, e2)

Counting operators:
    AtLeast(k, Assign(A, X), Assign(B, X), ...)
    AtMost(k, Assign(A, X), Assign(B, X), ...)
    Exactly(k, Assign(A, X), Assign(B, X), ...)

Option-testing operators:
    Sat(Option_A)
    Unsat(Option_A)

Allowed reasoning step types:
- Direct question facts:
    If the question says "If D and F are both on X", a valid step is:
    S1: And(Assign(D, X), Assign(F, X)).

- Forced group membership:
    If rules and facts force G to be in Y, a valid step is:
    S2: Assign(G, Y).

- Group exclusion:
    If A cannot be in X, a valid step is:
    S3: Not(Assign(A, X)).

- Same-group deductions:
    If A and B must be in the same group, a valid step is:
    S4: Assign(A, X) == Assign(B, X).

- Different-group deductions:
    If A and B must be in different groups, a valid step is:
    S5: Assign(A, X) != Assign(B, X).

- Conditional deductions:
    If A being in X would force B into Y, a valid step is:
    S6: Implies(Assign(A, X), Assign(B, Y)).

- Exclusive-choice deductions:
    If exactly one of A or B must be in X, a valid step is:
    S7: Xor(Assign(A, X), Assign(B, X)).

- Capacity or counting deductions:
    If exactly two of A, B, and C must be in X, a valid step is:
    S8: Exactly(2, Assign(A, X), Assign(B, X), Assign(C, X)).

- Option feasibility checks:
    If an option can be extended to at least one full valid grouping, use:
    S9: Sat(Option_D).

- Option impossibility checks:
    If an option cannot be extended to any full valid grouping, use:
    S10: Unsat(Option_A).

The "reasoning" field MUST follow this exact pair structure:
    "reasoning": [
    "Natural-language explanation.",
    "S1: formal_step.",
    "Natural-language explanation.",
    "S2: formal_step.",
    "Natural-language explanation.",
    "S3: formal_step."
    ]

Rules:
* Natural-language entries must NOT start with any label.
* Natural-language entries must be plain explanatory sentences.
* S entries must start with S1:, S2:, S3:, ...
* The list must strictly alternate:
  natural-language sentence, S-step, natural-language sentence, S-step, ...
* Every S entry must be solver-checkable.
* Every S entry must end with a period.
* Do not write paragraph summaries.
* Do not use unsupported notation such as arrows, quoted formal expressions, table-row summaries, or ordering notation unless the grouping problem explicitly includes ordered groups.
* The model-specific <think> block is ignored by the grader.
* Only the JSON "reasoning" list inside <answer>...</answer> is graded for reasoning quality.
* Therefore, any formal deduction written in <think> must be repeated inside the JSON "reasoning" field.

Logical validity requirements:
- Every formal step MUST be entailed by rules + facts + earlier accepted formal steps, unless it is an option feasibility step.
- For option feasibility steps:
    Sat(Option_X) means rules + facts + Option_X is satisfiable.
    Unsat(Option_X) means rules + facts + Option_X is unsatisfiable.
- Do NOT output unsupported guesses.
- Do NOT output contradictory steps.
- Do NOT output tautologies such as Or(Assign(A, X), Not(Assign(A, X))).
- Do NOT merely restate every passage rule unless the restatement is needed to connect a deduction.
- Do NOT use ordering operators such as <, >, +1 ==, or position numbers unless the grouping problem explicitly includes ordered groups.
- Do NOT use final answer text as a reasoning step; formal steps must remain solver-oriented.

Examples of valid interleaved grouping reasoning:
    The question condition places both D and F in group X.
    S1: And(Assign(D, X), Assign(F, X)).

    Since F and G must be in different groups and F is in X, G must be in Y.
    S2: Assign(G, Y).

    Since C in X would force D to be in Y, C cannot be in X because D is already in X.
    S3: Not(Assign(C, X)).

    Since E and A must be in different groups, they cannot both be in Y.
    S4: Not(And(Assign(E, Y), Assign(A, Y))).

    Option A forces C into X, which contradicts the derived restriction that C cannot be in X.
    S5: Unsat(Option_A).

    Option D can be extended to a complete valid grouping.
    S6: Sat(Option_D).

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
  "problem_type": "grouping",
  "world_model": {
    "entities": [],
    "domains": {
      "groups": []
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

GROUPING_FEWSHOT_USER_PROMPT = """
--------------------------------
AR-LSAT GROUPING EXAMPLE
--------------------------------

Example Passage:
Seven directors A, B, C, D, E, F, and G serve on either committee X or committee Y.

Rules:
1. If A is on X, then B is on Y.
2. If C is on X, then D and E are on Y.
3. F is on a different committee from G.
4. E is on a different committee from A.
5. If G is on X, then B is on X.

Example Question:
If D and F are both on X, which could be true?

question_type:
could_be_true

Options:
A. A and C are on X
B. A and E are on Y
C. B and G are on X
D. C and E are on Y
E. G and E are on X

Solve the AR-LSAT grouping example and return problem_type, world_model, rules, facts, question_semantics, options, reasoning, and solution inside a single <answer>...</answer> block, with no additional text.
"""

GROUPING_FEWSHOT_ASSISTANT_ANSWER = """
<answer>{
  "problem_type": "grouping",
  "world_model": {
    "entities": ["A","B","C","D","E","F","G"],
    "domains": {"groups": ["X","Y"]},
    "structural_assumptions": [
      "each entity belongs to exactly one group",
      "groups are mutually exclusive"
    ]
  },
  "rules": [
    "Implies(Assign(A, X), Assign(B, Y))",
    "Implies(Assign(C, X), And(Assign(D, Y), Assign(E, Y)))",
    "Assign(F, X) != Assign(G, X)",
    "Assign(E, X) != Assign(A, X)",
    "Implies(Assign(G, X), Assign(B, X))"
  ],
  "facts": [
    "Assign(D, X)",
    "Assign(F, X)"
  ],
  "question_semantics": {
    "question_type": "could_be_true",
    "option_interpretation_rule": "choose satisfiable option"
  },
  "options": {
    "A": "And(Assign(A, X), Assign(C, X))",
    "B": "And(Assign(A, Y), Assign(E, Y))",
    "C": "And(Assign(B, X), Assign(G, X))",
    "D": "And(Assign(C, Y), Assign(E, Y))",
    "E": "And(Assign(G, X), Assign(E, X))"
  },
  "reasoning": [
    "D and F are fixed in X.",
    "S1: And(Assign(D, X), Assign(F, X)).",
    "F and G must be in different groups, so G must be in Y.",
    "S2: Assign(G, Y).",
    "Option D satisfies all constraints.",
    "S3: Sat(Option_D).",
    "Other options violate at least one rule.",
    "S4: Unsat(Option_A)."
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

Solve the AR-LSAT grouping problem above and provide problem_type, world_model, rules, facts, question_semantics, options, reasoning, and solution.

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
                "content": GROUPING_SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": GROUPING_FEWSHOT_USER_PROMPT.strip(),
            },
            {
                "role": "assistant",
                "content": GROUPING_FEWSHOT_ASSISTANT_ANSWER.strip(),
            },
            {
                "role": "user",
                "content": GROUPING_USER_PROMPT_PHI.format(
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
