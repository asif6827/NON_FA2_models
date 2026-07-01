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

The grading system will evaluate only the first complete <answer>...</answer> block.
If a <think> block is generated, keep it brief.
The formal proof must be inside "reasoning".
Do not put thinking markers, markdown, comments, or explanations inside the <answer> block.

You are given:
(i) one AR-LSAT assignment passage written in plain English,
(ii) one question about that passage,
(iii) a question_type label,
(iv) a dictionary of answer options,
and optionally
(v) metadata such as tags or entity hints if available.

Your task is to parse the assignment problem into a solver-oriented logical representation and determine the correct answer.

=============================
CRITICAL FORMAT REQUIREMENTS
=============================
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



==============
PROBLEM TYPE
==============

- "problem_type" MUST always be exactly "assignment".
- "problem_type" MUST be a string, not an object, list, boolean, or null.
- The input question_type must NOT be copied into "problem_type".
- The input question_type must appear only here:
  "question_semantics": {
    "question_type": "<input question_type>",
    "option_interpretation_rule": "..."
  }
- Never set "problem_type" to "must_be_true", "could_be_true", "cannot_be_true", "acceptability", or any other question_type label.
- Valid example:
  "problem_type": "assignment"
- Invalid examples:
  "problem_type": "must_be_true"
  "problem_type": {"type": "assignment"}
  "problem_type": ["assignment"]

=============
WORLD MODEL
=============

- "world_model" MUST be a JSON object.
- "world_model" MUST describe the global assignment universe before applying question-specific facts or answer options.
- "world_model" MUST contain exactly these three keys:
  "entities",
  "domains",
  "structural_assumptions"
- "entities" MUST be a list of strings.
- "entities" MUST contain the objects/people/items that receive assignments.
- Preserve entity names exactly as they appear in the passage whenever possible.
- "domains" MUST be a JSON object whose values are lists of strings.
- If the problem has one assignment attribute, use:
  "domains": {
    "values": ["..."]
  }
- If the problem has multiple assignment attributes, use meaningful attribute names:
  "domains": {
    "day": ["Monday", "Tuesday"],
    "room": ["Room1", "Room2"]
  }
- "structural_assumptions" MUST be a list of strings.
- "structural_assumptions" MUST include general assignment assumptions such as exactly-one assignment, mutual exclusivity, uniqueness, or consistency when applicable.
- Do NOT put passage rules in "world_model".
- Do NOT put question-specific facts in "world_model".
- Do NOT put answer options in "world_model".
- Valid shape:
  "world_model": {
    "entities": ["A", "B", "C"],
    "domains": {"values": ["P1", "P2", "P3"]},
    "structural_assumptions": ["each entity is assigned exactly one value"]
  }
- Invalid examples:
  "world_model": "A, B, and C are assigned to projects"
  "world_model": []
  "world_model": {"rules": ["Not(Assign(A, P1))"]}

=======
RULES
=======

- "rules" MUST be a list of strings.
- Each item in "rules" MUST be one formalized passage constraint.
- Include ONLY constraints stated or implied by the passage.
- Do NOT include question-specific temporary conditions in "rules".
- Do NOT include answer options in "rules".
- Do NOT include reasoning steps in "rules".
- Do NOT write natural-language explanations in "rules".
- Each rule string MUST use allowed formal operators such as Assign(...), Not(...), And(...), Or(...), Implies(...), Exactly(...), AtLeast(...), or AtMost(...).
- Use one string per rule whenever possible.
- If a passage rule requires multiple formal constraints, split it into multiple strings.
- Valid examples:
  "rules": [
    "Not(Assign(A, P1))",
    "Assign(B, P1) == Assign(C, P1)",
    "Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))"
  ]
- Invalid examples:
  "rules": "A is not assigned to P1"
  "rules": ["A is not assigned to P1"]
  "rules": [{"rule": "Not(Assign(A, P1))"}]

=======
FACTS
=======

- "facts" MUST be a list of strings.
- "facts" MUST contain ONLY temporary conditions introduced by the question stem.
- If the question stem adds no temporary condition, "facts" MUST be an empty list: [].
- Do NOT copy passage rules into "facts".
- Do NOT copy answer options into "facts".
- Do NOT copy derived reasoning steps into "facts".
- Each fact string MUST be a formal expression using allowed operators.
- Use "facts" for phrases such as "if A is assigned to P2", "suppose B is not assigned to P3", or "if exactly two employees are assigned to P1" when they appear in the question.
- Valid examples:
  "facts": []
  "facts": ["Assign(A, P2)"]
  "facts": ["Not(Assign(B, P3))"]
- Invalid examples:
  "facts": "None"
  "facts": ["There are no extra facts."]
  "facts": ["Option A says A is assigned to P2"]

====================
QUESTION SEMANTICS
====================

- "question_semantics" MUST be a JSON object.
- "question_semantics" MUST contain exactly these two keys:
  "question_type",
  "option_interpretation_rule"
- "question_type" MUST be a string.
- "question_type" MUST exactly copy the input question_type label.
- Do NOT normalize, rename, or paraphrase the input question_type.
- "option_interpretation_rule" MUST be a string.
- "option_interpretation_rule" MUST explain how each answer option is tested under the given question_type.
- For "must_be_true", use a rule equivalent to: choose the option whose negation is unsatisfiable under passage rules plus facts.
- For "could_be_true", use a rule equivalent to: choose the option whose assertion is satisfiable under passage rules plus facts.
- For "cannot_be_true", use a rule equivalent to: choose the option whose assertion is unsatisfiable under passage rules plus facts.
- For "acceptability", use a rule equivalent to: choose the complete option assignment that satisfies all passage rules plus facts.
- The input question_type must appear here and nowhere else except natural-language text inside "reasoning" if needed.
- Valid example:
  "question_semantics": {
    "question_type": "must_be_true",
    "option_interpretation_rule": "choose option whose negation is unsatisfiable"
  }
- Invalid examples:
  "question_semantics": "must_be_true"
  "question_semantics": {"type": "must_be_true"}
  "question_semantics": {"question_type": "assignment"}

=========
OPTIONS
=========

- "options" MUST be a JSON object.
- The keys of "options" MUST exactly match the input answer-option labels, such as "A", "B", "C", "D", and "E".
- Each value in "options" MUST be a formal expression string.
- Do NOT use natural-language option text as the value unless it has been formalized.
- Do NOT add option labels that are not present in the input.
- Do NOT omit any input option labels.
- Do NOT select the answer inside "options".
- Use Option_A, Option_B, etc. only inside reasoning feasibility checks, not as replacements for the actual option formalization.
- Valid example:
  "options": {
    "A": "Assign(A, P2)",
    "B": "Assign(B, P3)",
    "C": "Assign(C, P3)",
    "D": "Assign(B, P2)",
    "E": "Assign(A, P3)"
  }
- Invalid examples:
  "options": ["A", "B", "C", "D", "E"]
  "options": {"A": "A is assigned to P2"}
  "options": {"selected_option": "A"}


=================
REASONING
=================

- "reasoning" MUST be a list of strings.
- "reasoning" MUST NOT be a single paragraph string.
- "reasoning" MUST strictly alternate between natural-language explanation sentences and formal solver-oriented S-steps.
- Odd-numbered entries MUST be natural-language explanation sentences.
- Even-numbered entries MUST be formal solver-oriented S-steps.
- Each natural-language entry MUST be exactly one sentence and end with a period.
- Natural-language entries MUST NOT start with S1:, S2:, or any other S-label.
- Each formal entry MUST start with the next sequential step label: S1:, S2:, S3:, etc.
- Each formal entry MUST be exactly one sentence and end with a period.
- Every S-step MUST be logically valid, solver-checkable, and must not contain prose explanations.
- Put explanations only in the natural-language sentence immediately before the S-step.
- The reasoning MUST justify the selected option using rules, facts, option feasibility, option contradiction, or option necessity.
- The model-specific <think> block is ignored by the grader, so any deduction needed for credit MUST be repeated inside the JSON "reasoning" field.
- Do NOT write paragraph summaries.
- Do NOT use unsupported notation such as arrows, informal equations, quotes around formal expressions, or table-row summaries.
- Each step MUST follow from rules + facts + prior steps.
- Do NOT introduce contradictions.
- Do NOT use tautological steps.
- Do NOT hallucinate assumptions.
- Do NOT use ordering operators unless explicitly required by the passage.

Formal steps MUST:
- Start with "S<k>: ".
- End with a period.
- Be logically valid and solver-verifiable.
- Contain at least one allowed formal operator.

Every S-step MUST contain at least one of the following allowed formal operators:
Assign(...), Not(...), And(...), Or(...), Implies(...), Exactly(...), AtLeast(...), AtMost(...), Sat(...), Unsat(...).

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

Valid reasoning shape:
    "reasoning": [
      "Exactly one employee is assigned to P2.",
      "S1: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2)).",
      "The negation of Option A is impossible under the rules.",
      "S2: Unsat(Not(Option_A))."
    ]

Invalid reasoning examples:
- "reasoning": "A must be assigned to P2, so the answer is A."
- "reasoning": ["S1: A is assigned to P2."]
- "reasoning": ["Therefore A is correct.", "S1: Option A is correct."]

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


===========
SOLUTION
===========

- "solution" MUST be a JSON object.
- "solution" MUST be the final top-level key.
- "solution" MUST contain exactly one key:
  "selected_option"
- "selected_option" MUST be a string.
- "selected_option" MUST exactly match one input answer-option label, such as "A", "B", "C", "D", or "E".
- Do NOT put explanations inside "solution".
- Do NOT put the option text inside "selected_option".
- Do NOT add confidence, score, proof, or any other key inside "solution".
- Valid example:
  "solution": {
    "selected_option": "A"
  }
- Invalid examples:
  "solution": "A"
  "solution": {"answer": "A"}
  "solution": {"selected_option": "A", "confidence": 0.9}
  

================================================================================
ASSIGNMENT CANONICALIZATION AND FORMALIZATION
================================================================================

Use this section to normalize the passage, construct the solver-oriented schema fields, and write all formal expressions consistently.

========================
SYMBOL NORMALIZATION
========================

- Use concise symbolic tokens.
- Preserve entity names exactly as they appear when they are already symbolic, such as A, B, C, etc.
- Normalize multi-word values by using readable compact tokens when needed.
- Represent all assignment relations using:

    Assign(entity, value)

- Each entity must be assigned exactly one value per relevant attribute.
- Assignments must be mutually consistent across attributes.
- Do NOT use positional operators such as <, >, <=, or >= unless the passage explicitly requires ordering.

========================
FIELD CONSTRUCTION RULES
========================

Construct "world_model":
- Extract all entities.
- Extract all attribute domains, such as floors, cities, colors, projects, days, rooms, or other assignable values.
- Include structural assumptions such as:
    each entity is assigned exactly one value,
    values are mutually exclusive when required,
    assignments are consistent across attributes.

Construct "rules":
- Include ONLY constraints stated in the passage.
- Do NOT include temporary question conditions.
- Do NOT include answer option assumptions.
- Express each rule as a formal-expression string.

Construct "facts":
- Include ONLY temporary assumptions introduced by the question stem.
- If the question has no temporary condition, use an empty list.
- Do NOT copy passage rules into "facts".

Construct "question_semantics":
- Use the input question_type exactly as provided.
- Explain how each option should be evaluated for that question_type.
- Store the input question_type only inside "question_semantics", not inside "problem_type".

Construct "options":
- Formalize each answer option using assignment expressions.
- Each option label must map to one formal-expression string.
- Use labels exactly as provided, such as "A", "B", "C", "D", and "E".

========================
ALLOWED FORMAL OPERATORS
========================

Assignment:
    Assign(A, X)

Equality and inequality:
    Assign(A, X) == Assign(B, Y)
    Assign(A, X) != Assign(B, Y)

Boolean operators:
    And(...)
    Or(...)
    Not(...)
    Implies(...)
    Xor(...)

Counting operators:
    AtLeast(k, ...)
    AtMost(k, ...)
    Exactly(k, ...)

Solver-status operators:
    Sat(...)
    Unsat(...)

========================
ASSIGNMENT EXPRESSION GUIDE
========================

A is assigned to X:
    Assign(A, X)

A is not assigned to X:
    Not(Assign(A, X))

A and B share the same value:
    Assign(A, X) == Assign(B, X)

A and B have different values:
    Assign(A, X) != Assign(B, X)

If A has X, then B has Y:
    Implies(Assign(A, X), Assign(B, Y))

A has exactly one value among X1, X2, and X3:
    Exactly(1, Assign(A, X1), Assign(A, X2), Assign(A, X3))

At least two entities are assigned to X:
    AtLeast(2, Assign(A, X), Assign(B, X), Assign(C, X))

At most one entity is assigned to X:
    AtMost(1, Assign(A, X), Assign(B, X), Assign(C, X))

An option is feasible:
    Sat(Option_C)

An option is impossible:
    Unsat(Option_A)

The negation of an option is impossible:
    Unsat(Not(Option_A))
    

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