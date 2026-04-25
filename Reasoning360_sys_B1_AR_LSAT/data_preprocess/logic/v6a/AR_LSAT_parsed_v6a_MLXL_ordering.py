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
You are an expert AR-LSAT logical reasoning solver.

You are given:
(i) one AR-LSAT passage written in plain English,
(ii) one question about that passage,
(iii) a list of answer options,
and optionally
(iv) metadata such as problem_type, tags, or candidate entities if available.

Your task is to parse the AR-LSAT problem into a solver-oriented logical representation and determine the correct answer by generating the following EIGHT fields:
1) problem_type — the inferred problem type: ordering, grouping, or assignment.
2) world_model — the entities, domains, and structural assumptions.
3) rules — formalized passage rules only.
4) facts — question-specific temporary conditions only.
5) question_semantics — how the options must be evaluated.
6) options — formalized answer options.
7) reasoning — interleaved natural-language reasoning and solver-oriented steps.
8) solution — the final selected answer option.

You MUST return the result STRICTLY as a single valid JSON object wrapped inside:
<answer>...</answer>

No additional text, commentary, markdown, or formatting outside the <answer> block is permitted.

================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly EIGHT top-level keys, spelled EXACTLY:
    "problem_type",
    "world_model",
    "rules",
    "facts",
    "question_semantics",
    "options",
    "reasoning",
    "solution"
- Do NOT add any other top-level keys.
- Do NOT output Python code.
- Do NOT output Z3 code.
- Do NOT use markdown or code fences inside or outside the answer block.
- All formal expressions must be strings.

================================================================================
NORMALIZATION RULES
================================================================================
- Use concise symbolic tokens for entities and values.
- Preserve single-letter entities exactly when used, e.g., A, B, C, D.
- Use underscores instead of spaces in multi-word tokens, e.g., red_project, first_shift.
- Convert ordinals to integers when used as positions:
    first=1, second=2, third=3, fourth=4, fifth=5, sixth=6.
- Do not invent entities, groups, positions, values, or assumptions not supported by the passage or metadata.
- If metadata provides canonical names or tags, use them when they are consistent with the passage.
- For ordering problems, positions must be represented using integers.
- For grouping problems, group labels must be represented directly, e.g., X, Y, in, out.
- For assignment problems, assigned values must be represented directly, e.g., Monday, Red, Room_1.
- Use Assign(entity, value) as a semantic wrapper for assignment or membership.
- Assign(entity, value) means the entity is assigned to that value and must be translatable to a Z3 equality constraint.

================================================================================
PARSING INSTRUCTIONS
================================================================================
First identify the problem type:
- ordering: entities are placed in positions, ranks, slots, or sequence order.
- grouping: entities are placed into groups, committees, teams, selected/not selected, or in/out categories.
- assignment: entities are mapped to values such as projects, days, rooms, tasks, colors, or roles.

Then construct the world model:
- Extract all core entities.
- Extract all domains.
- Extract structural assumptions such as uniqueness, exactly-one assignment, mutual exclusion, one-to-one mapping, group capacity, or ordering positions.
- Include only assumptions licensed by the passage.

Then parse the passage rules:
- "rules" MUST contain only constraints from the passage.
- Do NOT include question-specific facts in "rules".
- Each rule must be a single solver-oriented logical expression.
- Rules should be directly translatable to Z3.

Then parse the question facts:
- "facts" MUST contain only temporary assumptions introduced by the question stem.
- If there are no question-specific facts, output an empty list.
- Do NOT mix facts with passage rules.

Then parse question semantics:
- Identify the exact question type.
- Use one of:
    "could_be_true",
    "must_be_true",
    "cannot_be_true",
    "could_be_false",
    "must_follow",
    "valid_complete_assignment",
    "other"
- Use the correct solver interpretation:
    could_be_true => option is satisfiable with rules + facts.
    must_be_true => Not(option) is unsatisfiable with rules + facts.
    cannot_be_true => option is unsatisfiable with rules + facts.
    could_be_false => Not(option) is satisfiable with rules + facts.
    must_follow => Not(option) is unsatisfiable with rules + facts.
    valid_complete_assignment => option is satisfiable and satisfies all structural assumptions.

Then parse answer options:
- "options" MUST be a JSON object mapping option labels to formal expressions.
- Each option must be formalized independently.
- Do not let one option affect another option.
- Use the exact option labels from the input.

Allowed formal operators:
- Logical: And(...), Or(...), Not(...), Xor(...), Implies(...)
- Equality: ==, !=
- Ordering: <, >, <=, >=, + k ==
- Distinctness: Distinct(...)
- Counting: AtLeast(k, ...), AtMost(k, ...), Exactly(k, ...)
- Assignment/membership: Assign(entity, value)
- Solver-status steps in reasoning only: Sat(...), Unsat(...)

================================================================================
REASONING REQUIREMENTS
================================================================================
- "reasoning" MUST be a list of strings.
- Reasoning MUST be interleaved:
    Odd-numbered entries: natural-language reasoning.
    Even-numbered entries: formal solver-oriented step.
- Every formal step MUST start with "S<k>: " and end with a period.
- <k> starts at 1 and increments by 1 for each formal step.
- Formal steps must use only the allowed operators.
- Formal steps should be logically supported by rules, facts, and earlier accepted steps.
- Do NOT include contradictory, hallucinated, or unsupported steps.
- Do NOT merely restate every rule unless needed for the deduction.
- Option testing steps may use Sat(Option_A), Unsat(Option_B), etc.

================================================================================
SOLUTION REQUIREMENTS
================================================================================
- "solution" MUST be a JSON object with exactly this key:
    "selected_option"
- "selected_option" MUST use the exact option label from the input.
- Do NOT include a final table unless the question asks for a complete assignment.
- Do NOT include extra explanation inside "solution".

================================================================================
OUTPUT SCHEMA
================================================================================
The output MUST follow this exact structure:

<answer>{
  "problem_type": "ordering | grouping | assignment",
  "world_model": {
    "entities": [],
    "domains": {},
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

================================================================================
ONE-SHOT EXAMPLE 1: ORDERING
================================================================================

Example Passage:
Four speakers A, B, C, and D speak in positions 1 through 4, with exactly one speaker in each position.

Rules:
1. A speaks before B.
2. C speaks immediately after A.
3. D does not speak first.

Example Question:
If B speaks fourth, which one of the following could be true?

Example Options:
A. A speaks second.
B. C speaks fourth.
C. D speaks second.
D. A speaks third.
E. C speaks first.

Correct Example Output:
<answer>{
  "problem_type": "ordering",
  "world_model": {
    "entities": ["A", "B", "C", "D"],
    "domains": {
      "positions": ["1", "2", "3", "4"]
    },
    "structural_assumptions": [
      "each speaker occupies exactly one position",
      "each position is occupied by exactly one speaker",
      "positions are ordered from 1 to 4"
    ]
  },
  "rules": [
    "A < B",
    "C == A + 1",
    "D != 1",
    "Distinct(A, B, C, D)"
  ],
  "facts": [
    "B == 4"
  ],
  "question_semantics": {
    "question_type": "could_be_true",
    "option_interpretation_rule": "choose the option whose formalization is satisfiable with rules and facts"
  },
  "options": {
    "A": "A == 2",
    "B": "C == 4",
    "C": "D == 2",
    "D": "A == 3",
    "E": "C == 1"
  },
  "reasoning": [
    "The question condition fixes B in the fourth position.",
    "S1: B == 4.",
    "Since C is immediately after A, A cannot be fourth and C cannot be first.",
    "S2: C == A + 1.",
    "Option A places A second, which allows C third, B fourth, and D first is not allowed, so D must be placed elsewhere if possible.",
    "S3: Sat(Option_A).",
    "Option B places C fourth, which would force A third and conflict with B already being fourth due to distinct positions.",
    "S4: Unsat(Option_B).",
    "Option C places D second, leaving A and C to be first and second or second and third, but D already occupies second and C must immediately follow A, so no valid arrangement remains.",
    "S5: Unsat(Option_C).",
    "Option D places A third, which forces C fourth and conflicts with B fourth.",
    "S6: Unsat(Option_D).",
    "Option E places C first, which is impossible because C must be immediately after A.",
    "S7: Unsat(Option_E)."
  ],
  "solution": {
    "selected_option": "A"
  }
}</answer>

================================================================================
ONE-SHOT EXAMPLE 2: GROUPING
================================================================================

Example Passage:
Seven directors A, B, C, D, E, F, and G serve on either the X committee or the Y committee.

Rules:
1. If A serves on X, then B serves on Y.
2. If C serves on X, then D and E serve on Y.
3. F serves on a different committee from G.
4. E serves on a different committee from A.
5. If G serves on X, then B serves on X.

Example Question:
If D and F both serve on the X committee, then which one of the following could be true?

Example Options:
A. A and C both serve on the X committee.
B. A and E both serve on the Y committee.
C. B and G both serve on the X committee.
D. C and E both serve on the Y committee.
E. G and E both serve on the X committee.

Correct Example Output:
<answer>{
  "problem_type": "grouping",
  "world_model": {
    "entities": ["A", "B", "C", "D", "E", "F", "G"],
    "domains": {
      "committees": ["X", "Y"]
    },
    "structural_assumptions": [
      "each director serves on exactly one committee",
      "X and Y are mutually exclusive committees",
      "Assign(P, X) means P is on X and Assign(P, Y) means P is on Y"
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
    "option_interpretation_rule": "choose the option whose formalization is satisfiable with rules and facts"
  },
  "options": {
    "A": "And(Assign(A, X), Assign(C, X))",
    "B": "And(Assign(A, Y), Assign(E, Y))",
    "C": "And(Assign(B, X), Assign(G, X))",
    "D": "And(Assign(C, Y), Assign(E, Y))",
    "E": "And(Assign(G, X), Assign(E, X))"
  },
  "reasoning": [
    "The question condition directly places D and F on the X committee.",
    "S1: And(Assign(D, X), Assign(F, X)).",
    "Since F and G must be on different committees and F is on X, G must be on Y.",
    "S2: Assign(G, Y).",
    "Since E and A must be on different committees, they cannot both be on X or both be on Y.",
    "S3: Assign(E, X) != Assign(A, X).",
    "Option A places C on X, which would force D on Y by Rule 2, contradicting the fact that D is on X.",
    "S4: Unsat(Option_A).",
    "Option B places A and E both on Y, which violates the rule that A and E must be on different committees.",
    "S5: Unsat(Option_B).",
    "Option C places G on X, but G has already been forced to Y.",
    "S6: Unsat(Option_C).",
    "Option D places C and E both on Y, which is consistent with D on X, F on X, G on Y, A on X, and B on Y.",
    "S7: Sat(Option_D).",
    "Option E places G on X, but G has already been forced to Y.",
    "S8: Unsat(Option_E)."
  ],
  "solution": {
    "selected_option": "D"
  }
}</answer>

================================================================================
ONE-SHOT EXAMPLE 3: ASSIGNMENT
================================================================================

Example Passage:
Three people Alice, Bob, and Carol are each assigned exactly one project: Red, Blue, or Green. Each project is assigned to exactly one person. The project order is Red before Blue before Green.

Rules:
1. Alice is not assigned Red.
2. Bob is assigned to a project before Carol's project.
3. Carol is not assigned Green.

Example Question:
Which one of the following could be true?

Example Options:
A. Alice is assigned Red.
B. Bob is assigned Red and Carol is assigned Blue.
C. Bob is assigned Blue and Carol is assigned Red.
D. Carol is assigned Green.
E. Alice is assigned Blue and Bob is assigned Green.

Correct Example Output:
<answer>{
  "problem_type": "assignment",
  "world_model": {
    "entities": ["Alice", "Bob", "Carol"],
    "domains": {
      "projects": ["Red", "Blue", "Green"],
      "project_order": {
        "Red": "1",
        "Blue": "2",
        "Green": "3"
      }
    },
    "structural_assumptions": [
      "each person is assigned exactly one project",
      "each project is assigned to exactly one person",
      "project order is Red before Blue before Green"
    ]
  },
  "rules": [
    "Assign(Alice, Red) == False",
    "Alice != Bob",
    "Alice != Carol",
    "Bob != Carol",
    "Bob < Carol",
    "Assign(Carol, Green) == False"
  ],
  "facts": [],
  "question_semantics": {
    "question_type": "could_be_true",
    "option_interpretation_rule": "choose the option whose formalization is satisfiable with rules and facts"
  },
  "options": {
    "A": "Assign(Alice, Red)",
    "B": "And(Assign(Bob, Red), Assign(Carol, Blue))",
    "C": "And(Assign(Bob, Blue), Assign(Carol, Red))",
    "D": "Assign(Carol, Green)",
    "E": "And(Assign(Alice, Blue), Assign(Bob, Green))"
  },
  "reasoning": [
    "This is an assignment problem because each person must be mapped to exactly one project.",
    "S1: Distinct(Alice, Bob, Carol).",
    "Alice cannot be assigned Red according to Rule 1.",
    "S2: Not(Assign(Alice, Red)).",
    "Carol cannot be assigned Green according to Rule 3.",
    "S3: Not(Assign(Carol, Green)).",
    "Because Bob's project must come before Carol's project, Bob cannot be assigned Green and Carol cannot be assigned Red.",
    "S4: And(Not(Assign(Bob, Green)), Not(Assign(Carol, Red))).",
    "Option A assigns Alice to Red, directly violating Rule 1.",
    "S5: Unsat(Option_A).",
    "Option B assigns Bob to Red and Carol to Blue, which satisfies the project order and does not violate Alice's restriction.",
    "S6: Sat(Option_B).",
    "Option C assigns Bob to Blue and Carol to Red, which violates the rule that Bob's project must be before Carol's project.",
    "S7: Unsat(Option_C).",
    "Option D assigns Carol to Green, directly violating Rule 3.",
    "S8: Unsat(Option_D).",
    "Option E assigns Bob to Green, which makes it impossible for Bob to be before Carol.",
    "S9: Unsat(Option_E)."
  ],
  "solution": {
    "selected_option": "B"
  }
}</answer>

"""

SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
AR-LSAT PROBLEM TO SOLVE
--------------------------------

passage = {passage}

question = {question}

options = {options}

metadata = {metadata}

Solve the AR-LSAT problem above and return problem_type, world_model, rules, facts, question_semantics, options, reasoning, and solution inside one <answer> </answer> block, with no additional text.
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
    parser.add_argument('--data_setting', default=None, help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/AR_LSAT_to_guru/', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--data_source_train', default='our_ar_lsat_new_reward', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_ar_lsat_new_reward_test', help='Name of data source')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

    if args.data_setting == 'mlxl_train_mlxl_test':
        args.train_data_file = os.path.join(args.data_path, 'ar_lsat_train_300.json')
        args.test_data_file = os.path.join(args.data_path, 'ar_lsat_test_230.json')
    else:
        raise ValueError('Invalid data_setting')
    args.output_dir = os.path.join(args.output_dir, args.data_setting)



    if args.data_setting == 'mlxl_train_mlxl_test':
        # Load dataset from JSON or Parquet based on file extension
        file_extension = os.path.splitext(args.data_file)[1].lower()
        if file_extension in ['.json', '.jsonl']:
            train_dataset = datasets.load_dataset('json', data_files=args.train_data_file)['train']
            test_dataset = datasets.load_dataset('json', data_files=args.test_data_file)['train']



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
