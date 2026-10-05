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

You are given:
(i) one logic puzzle_text written in plain English,
(ii) solution_header that lists the attribute names used in the puzzle, and
(iii) a dictionary of attribute_values specifying the complete and exclusive set of allowed values for each attribute.

All values appearing in syntactic_clues, reasoning, and the final solution MUST be drawn from attribute_values and interpreted as entity tokens representing unknown house positions.

Your task is to construct a fully consistent, solver-verifiable solution by generating the following SIX fields:
1) n_houses — the total number of houses in the puzzle.
2) attribute_values — returned exactly as given, without modification.
3) syntactic_clues — a normalized, Z3-style textual encoding of each clue.
4) reasoning — categorized reasoning consisting of interleaved natural-language explanations and syntactic (solver-checkable) deduction steps.
5) solution — the final house-by-house assignment derived exclusively from syntactic_clues, reasoning.


You MUST return the result STRICTLY as a single valid JSON object wrapped inside:
<answer>...</answer>

No additional text, commentary, or formatting outside the <answer> block is permitted.


================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Do NOT include extra text, markdown, explanations, or code fences.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly FIVE top-level keys, spelled EXACTLY:
    "n_houses",
    "attribute_values",
    "syntactic_clues",
    "reasoning",
    "solution"
- Do NOT add any other keys.

================================================================================
NORMALIZATION RULES
================================================================================
- Use underscores instead of spaces in VALUES (e.g., grilled_cheese, very_short).
- Attribute names MUST match the solution_header exactly (case-sensitive), e.g., Name, Animal, Occupation, Sport, Height, etc.
- House numbers are integers 1..N.
- Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6, etc.
- Do not invent values. Every value must be mapped to its acronym (in attribute_values) and selected from the list of allowed attribute_values (after normalization).
 - Example: If the puzzle text says “sept” and the allowed attribute value is “september,” use “september.”
- If the clue mentions a bare person name (e.g., "Arnold"), treat it as Name=Arnold.
- If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching token in attribute_values.

================================================================================
1) DOMAIN OUTPUT (MANDATORY)
================================================================================
- "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
- "attribute_values" a JSON object. You MUST return the same attribute_values as those passed in the input.
- For a given attribute name, the values in "attribute_values" MUST be non-repeating."
- Do NOT infer extra attributes that are not explicitly listed in the "attribute_values".

================================================================================
2) syntactic_clues (MANDATORY, TEXTUAL CONSTRAINTS — NOT PREDICATES)
================================================================================
We do NOT use predicate-style DSL for clues.
Instead, each clue MUST be rewritten as a single-line *syntactic constraint statement* in a Z3-like textual form.

Rules:
- "syntactic_clues" MUST be a list of strings.
- For each clue, the selected tokens must be mapped to one of the values defined in attribute_values.
 - Example: If the clue says “sept” and the allowed attribute value is “september,” use “september.”
- There MUST be exactly one entry per clue, in the same order as the clues.
- Each entry MUST be exactly 1 line and end with a period.
- Each entry MUST start with the clue id prefix: "C<i>: ".
- Use ONLY these syntactic operators in the clue text:
    ==   (same house / equivalence)
    !=   (not same house)
    <    (somewhere left of)
    >    (somewhere right of)
    + 1 == (immediately left of)
    + 2 == (one house between)
    + 3 == (two house between)
    == H  (fixed house index, where H is an integer)
- Use bare normalized tokens (no quotes) for values (e.g., Arnold, engineer, very_short).
- When a clue states a specific house like "in the fifth house", encode as: <token> == 5
  Example: "The lawyer is in the fifth house." -> "C9: lawyer == 5."
- When a clue states "directly left of", encode as: A + 1 == B
  Example: "baseball is directly left of engineer" -> "C12: baseball + 1 == engineer."
- When a clue states "one house between", encode as: A + 2 == B
  Example: "There is one house between Eric and the bird keeper" -> "C12: Eric + 2 == bird_keeper."
  Example: "There is one house between Arnold and Peter" -> "C12: Arnold + 2 == Peter."
- When a clue states "two houses between", encode as: A + 3 == B
  Example: "There are two houses between Eric and Arnold" -> "C12: Eric + 3 == Arnold."
- When a clue states "person who has", encode as: A == B
  Example: "The person whose mother's name is Holly is the person who has black hair" -> "C12: Holly == black."
- When a clue states "one house between the person who has", encode as: A + 2 == B
  Example: "There is one house between the person who has black hair and Eric" -> "C12: black + 2 == Eric."
- When a clue states "somewhere to the left of", encode as: A < B
- When a clue states "somewhere to the right of", encode as: A > B
- When a clue states "X is the Y", encode as: X == Y

IMPORTANT:
- The goal is to produce constraints that resemble:
  s.add(<left> <op> <right>)
  but you must NOT write "s.add(...)".
  Only output the inner constraint as text.

================================================================================
3) reasoning (MANDATORY — CATEGORIZED, INTERLEAVED NATURAL + SYNTACTIC)
================================================================================

Overall structure:
- "reasoning" MUST be a structured JSON object grouped by reasoning category.
- Each category contains a list of reasoning entries.
- Categories MAY be empty but MUST NOT be omitted.

- The reasoning JSON object MUST have exactly TEN top-level keys, spelled EXACTLY:

- "Abs_Placement"
- "Direct_Equality"
- "Directed_Adjacency"
- "Structural_Positioning"
- "Domain_Restriction"
- "Exclusion"
- "Propagation"
- "Forced_Resolution"
- "Disjunction"
- "Case_Split"
- "Self-Verification"

Entry formatting rules:
- Each category’s value MUST be a list of strings.
- Each entry MUST be exactly 1 sentence and MUST end with a period.
- Reasoning MUST be interleaved within each category:
    Odd-numbered entries: Natural-language reasoning.
    Even-numbered entries: Syntactic reasoning step (Z3-like statement).

Natural-language entries:
- Must explain the deduction in plain English.
- Must justify why the immediately following syntactic step is valid.
- Must not introduce assumptions or unstated facts.
- When puzzle surface forms differ from attribute_values, ALWAYS use the normalized tokens in attribute_values
  (e.g., “September” → sept, “pop song” → pop).

Syntactic reasoning entries:
- Must encode only the newly deduced fact.
- Must not merely restate a clue unless required for dependency chaining.
- Must be written in Z3-like constraint form.
- Tokens MUST map directly to values in "attribute_values".

Syntactic reasoning step format:
  S<k>: <constraint>. [<evidence>]

Where:
- <k> starts at 1 and increments by 1 for each syntactic step only (S1, S2, S3, ...).
- <constraint> is a valid Z3-like Boolean expression.
- <evidence> lists the clues and/or earlier steps that justify the deduction.
- House positions are integers from 1 to n_houses.

Allowed constraint forms (Z3-style):

1) Atomic placement / ordering
  - Equality: x == y
      Example: "S4: lawyer == 5. [C9]"
  - Inequality: x != y
      Example: "S5: teacher != 5. [S4]"
  - Ordering: x < y  |  x > y
      Example: "S6: Arnold < very_short. [C4]"
  - Directed adjacency: x + 1 == y
      Example: "S7: baseball + 1 == engineer. [C12]"

2) Boolean connectives (disjunction + case analysis)
  - And(e1, e2, ..., en)
      Example: "S8: And(teacher != 2, teacher != 5). [S2+S4]"
  - Or(e1, e2, ..., en)
      Example: "S9: Or(teacher == 3, teacher == 4). [S6+S7]"
  - Not(e)
      Example: "S10: Not(teacher == 1). [S9]"
  - Implies(e1, e2)
      Example: "S11: Implies(teacher == 3, soccer == 4). [C15]"

3) Case elimination after contradiction
  - Use boolean forms to reject a case.
      Example: "S12: teacher != 3. [S11+S10]"

Evidence rules:
- Evidence MUST be included for EVERY syntactic step.
- Evidence may reference C<i> and earlier S<j> only (j < k).
- Use "+" to join multiple evidence refs, e.g., [C3+C11+S2].

Logical validity requirement:
- Every syntactic step MUST be logically entailed by the syntactic_clues plus all earlier syntactic steps.
- Forward references are forbidden.
- Do NOT output syntactic steps that merely restate a clue unless needed for the deduction chain.
- All steps must be verifiable by a constraint solver.

Category semantics (binding):
- "Abs_Placement": fixes a variable to a specific house.
- "Direct_Equality": binds two entities/attributes.
- "Directed_Adjacency": exact adjacency with direction.
- "Structural_Positioning": relative ordering constraints.
- "Domain_Restriction": shrinks remaining possibilities.
- "Exclusion": forbids a value/pairing.
- "Propagation": immediate consequences from existing constraints.
- "Forced_Resolution": assigns a value because only one option remains.
- "Disjunction": maintain remaining possibilities; MUST use solver-valid Or(...).
- "Case_Split": resolve a disjunction by contradiction/elimination.
- "Self-Verification": Verifies that each reasoning step is justified by the given clues or earlier steps, and that all derived constraints jointly satisfy and reproduce the final solution.


Global ordering rule:
- Categories do not impose order; S<k> is the only authoritative ordering.
- All dependencies must point from lower S<k> to higher S<k>.

Final invariant:
- Given syntactic_clues + all syntactic reasoning steps, a solver must verify every step with no ambiguity.

================================================================================
4) solution (MANDATORY — DERIVED FROM SYNTACTIC CLUES AND REASONING)
================================================================================

Purpose:
- The "solution" represents the final, complete assignment of all attributes to houses.
- The solution MUST be derived exclusively from the syntactic_clues and reasoning steps.
- The solution MUST NOT be independently guessed, inferred, or reasoned in natural language.

Derivation rule (binding invariant):
- The solution MUST be fully determined by:
    1) The complete set of syntactic_clues (C1..Cm), and
    2) The complete set of syntactic reasoning steps (S1..Sk).
- If the syntactic_clues and syntactic reasoning steps do not uniquely determine a
  complete assignment, the solution MUST be omitted or explicitly marked incomplete.

Structure:
- "solution" MUST be expressed in tabular form with:
  - "header": a list of column names.
  - "rows": a list of rows, each row being a list of strings matching the header order.

Header requirements:
- The header MUST include "House" as the first column.
- The remaining columns MUST appear in the same order as defined in "solution_header".

Row requirements:
- Each row corresponds to exactly one house.
- Rows MUST be listed in strictly increasing house order from 1..n_houses.
- Every attribute value MUST appear exactly once across all rows.

Normalization requirements:
- All solution values MUST be normalized using underscores.
- Values MUST exactly match tokens defined in "attribute_values".

Consistency requirements:
- Every value in the solution table MUST be logically entailed by the syntactic_clues
  and syntactic reasoning steps.
- The solution MUST be consistent with every syntactic reasoning step.
- No value may appear in the solution unless it is justified by the reasoning.

Prohibited behavior:
- The solution MUST NOT introduce new facts not present or implied in the syntactic
  reasoning steps.
- The solution MUST NOT contradict any syntactic reasoning step.
- The solution MUST NOT rely on natural-language reasoning text.

Final invariant:
- Given only the syntactic_clues and syntactic reasoning steps, the solution table
  must be uniquely reconstructible with no ambiguity.


================================================================================
ONE-SHOT EXAMPLE — Demonstration of REQUIRED OUTPUT
================================================================================

Example Puzzle:
There are 6 houses, numbered 1 to 6 from left to right.
Each house is occupied by a different person.
Each house has a unique attribute for each of the following characteristics:
- Each person has a unique name: Arnold, Peter, Bob, Eric, Carol, Alice
- The people keep unique animals: horse, rabbit, fish, cat, bird, dog
- Each person has an occupation: engineer, nurse, lawyer, teacher, artist, doctor
- People have unique favorite sports: basketball, volleyball, soccer, tennis, baseball, swimming
- People have unique heights: average, tall, short, very_short, very_tall, super_tall

Clues: 
1. The person who is an engineer is the dog owner. 
2. The person who has an average height is somewhere to the left of the person who is short. 
3. The person who has an average height is directly left of the rabbit owner. 
4. The person who is tall is somewhere to the left of the person who is very short. 
5. Arnold is the cat lover. 
6. The person who keeps horses is the person who is a teacher. 
7. Carol is the person who loves soccer. 
8. The person who is tall is the person who loves volleyball. 
9. The person who is a lawyer is in the fifth house. 
10. The person who loves tennis is the person who is a teacher. 
11. The person who has an average height is the person who loves swimming. 
12. The person who loves baseball is directly left of the person who is an engineer. 
13. Peter is the person who is a nurse. 
14. Bob is somewhere to the right of the person who is an artist. 
15. The person who is a teacher is directly left of the person who loves soccer. 
16. The rabbit owner is Alice. 
17. The fish enthusiast is Carol. 
18. The person who loves baseball is in the first house. 
19. The cat lover is somewhere to the right of the person who is very short. 
20. The person who is super tall is in the fifth house.

Correct Example Output:
<answer>{
  "n_houses": 6,
  "attribute_values": {
    "Name": ["Arnold", "Peter", "Bob", "Eric", "Carol", "Alice"],
    "Animal": ["horse", "rabbit", "fish", "cat", "bird", "dog"],
    "Occupation": ["engineer", "nurse", "lawyer", "teacher", "artist", "doctor"],
    "Sport": ["basketball", "volleyball", "soccer", "tennis", "baseball", "swimming"],
    "Height": ["average", "tall", "short", "very_short", "very_tall", "super_tall"]
  },
  "syntactic_clues": [
    "C1: engineer == dog.",
    "C2: average < short.",
    "C3: average + 1 == rabbit.",
    "C4: tall < very_short.",
    "C5: Arnold == cat.",
    "C6: horse == teacher.",
    "C7: Carol == soccer.",
    "C8: tall == volleyball.",
    "C9: lawyer == 5.",
    "C10: tennis == teacher.",
    "C11: average == swimming.",
    "C12: baseball + 1 == engineer.",
    "C13: Peter == nurse.",
    "C14: Bob > artist.",
    "C15: teacher + 1 == soccer.",
    "C16: rabbit == Alice.",
    "C17: fish == Carol.",
    "C18: baseball == 1.",
    "C19: cat > very_short.",
    "C20: super_tall == 5."
  ],
  "reasoning": {
    "Abs_Placement": [
      "The clue explicitly fixes the baseball fan to the leftmost position, anchoring the entire adjacency chain.",
      "S1: baseball == 1. [C18]",
      "The lawyer is directly fixed to a specific house by the clue, independent of other attributes.",
      "S2: lawyer == 5. [C9]",
      "The tallest possible height category is fixed to house 5, constraining all other height placements.",
      "S3: super_tall == 5. [C20]"
    ],
    "Direct_Equality": [
      "Engineering and dog ownership are defined as the same individual, allowing occupation-based propagation.",
      "S4: engineer == dog. [C1]",
      "Carol is explicitly identified as the soccer enthusiast, linking name and sport.",
      "S5: Carol == soccer. [C7]",
      "Fish ownership is directly tied to Carol, extending the binding across categories.",
      "S6: fish == Carol. [C17]",
      "Teaching is equivalent to horse ownership, forming a multi-attribute identity.",
      "S7: horse == teacher. [C6]",
      "Tennis is directly associated with the teacher, further strengthening the teacher identity.",
      "S8: tennis == teacher. [C10]",
      "Average height and swimming preference belong to the same person by definition.",
      "S9: average == swimming. [C11]",
      "The rabbit owner is Alice, fixing the animal-name pairing.",
      "S10: rabbit == Alice. [C16]",
      "Volleyball preference is uniquely associated with tall height.",
      "S11: tall == volleyball. [C8]",
      "Peter is explicitly stated to be the nurse, binding name and occupation.",
      "S12: Peter == nurse. [C13]",
      "Arnold is directly identified as the cat owner.",
      "S13: Arnold == cat. [C5]"
    ],
    "Directed_Adjacency": [
      "The baseball fan must be immediately left of the engineer, forming a strict adjacency constraint.",
      "S14: baseball + 1 == engineer. [C12]",
      "The teacher must be immediately left of the soccer fan, creating another adjacency chain.",
      "S15: teacher + 1 == soccer. [C15]",
      "Average height must be immediately left of the rabbit owner, tightly restricting both positions.",
      "S16: average + 1 == rabbit. [C3]"
    ],
    "Structural_Positioning": [
      "Average height is constrained to appear somewhere to the left of short height.",
      "S17: average < short. [C2]",
      "Tall height must occur strictly to the left of very short height.",
      "S18: tall < very_short. [C4]",
      "Bob must be positioned somewhere to the right of the artist, introducing a directional dependency.",
      "S19: Bob > artist. [C14]",
      "The cat owner must be located to the right of the very short person.",
      "S20: cat > very_short. [C19]"
    ],
    "Domain_Restriction": [
      "Since baseball is fixed to house 1, all other houses are excluded from hosting baseball.",
      "S21: And(baseball != 2, baseball != 3, baseball != 4, baseball != 5, baseball != 6). [S1]"
    ],
    "Exclusion": [
      "Because super_tall occupies house 5, no other height category can appear in that house.",
      "S22: And(tall != 5, very_short != 5, average != 5, short != 5, very_tall != 5). [S3]"
    ],
    "Propagation": [
      "Since baseball is in house 1 and must be immediately left of engineer, engineer is forced into house 2.",
      "S23: engineer == 2. [S1+S14]",
      "Because engineer and dog are the same person, dog ownership propagates to house 2.",
      "S24: dog == 2. [S4+S23]",
      "If soccer were in house 6, teacher would be forced into house 5, contradicting the lawyer placement.",
      "S25: Implies(soccer == 6, teacher == 5). [S2+S15]",
      "Since teacher cannot be in house 5, soccer cannot be in house 6.",
      "S26: soccer != 6. [S25]",
      "Given adjacency constraints, soccer must therefore be in house 5.",
      "S27: soccer == 5. [S5+S26]"
    ],
    "Forced_Resolution": [
      "Because teacher must be immediately left of soccer and soccer is in house 5, teacher is forced into house 4.",
      "S28: teacher == 4. [S15+S27]",
      "All remaining height and animal assignments now collapse to a single consistent configuration.",
      "S29: And(horse == 4, tennis == 4, fish == 5). [S7+S8+S6+S28]"
    ],
    "Disjunction": [
      "From tall < very_short and tall == volleyball, volleyball could initially be in house 3 or 4.",
      "S30: Or(volleyball == 3, volleyball == 4). [S11+S18]"
    ],
    "Case_Split": [
      "If volleyball were in house 4 then tall would be in house 4, but house 4 is already teacher, leading to a conflict.",
      "S31: Not(volleyball == 4). [S28+S11]",
      "Therefore volleyball must be in house 3.",
      "S32: volleyball == 3. [S30+S31]"
    ]
  },
  "solution": {
    "header": ["House", "Name", "Animal", "Occupation", "Sport", "Height"],
    "rows": [
      ["1", "Peter", "bird", "nurse", "baseball", "very_tall"],
      ["2", "Eric", "dog", "engineer", "swimming", "average"],
      ["3", "Alice", "rabbit", "artist", "volleyball", "tall"],
      ["4", "Bob", "horse", "teacher", "tennis", "very_short"],
      ["5", "Carol", "fish", "lawyer", "soccer", "super_tall"],
      ["6", "Arnold", "cat", "doctor", "basketball", "short"]
    ]
  }
}</answer>
"""


SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

puzzle_text: {puzzle}

solution_header: {solution_header}

attribute_values: {attribute_values}

Solve the puzzle above and provide n_houses, attribute_values, parsed_clues, reasoning, and solution for this puzzle in the <answer> </answer> block, with no additional text.
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
    elif args.data_setting == 'small_train_small_test':
        args.data_file = os.path.join(args.data_path, 'Zebra_Puzzle_small_320.json')
        pass
    else:
        raise ValueError('Invalid data_setting')
    args.output_dir = os.path.join(args.output_dir, args.data_setting)



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
