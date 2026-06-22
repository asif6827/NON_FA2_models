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

Your task is to construct a fully consistent, solver-verifiable solution by generating the following FIVE fields:
1) n_houses — the total number of houses in the puzzle.
2) attribute_values — returned exactly as given, without modification.
3) syntactic_clues — a normalized, Z3-style textual encoding of each clue.
4) reasoning — categorized reasoning consisting of interleaved natural-language explanations and syntactic (solver-checkable) deduction steps.
5) solution — the final house-by-house assignment derived exclusively from syntactic_clues, and syntactic reasoning steps (S1..Sk).


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
- Do not invent values. Every value must be mapped to its canonical token (in attribute_values) and selected from the list of allowed attribute_values (after normalization).
 - Example: If puzzle says “september” and attribute_values contains "sept", output "sept" (not september).
 - Example: If puzzle says “sept” and attribute_values contains "september", output "september"
- If the clue mentions a bare person name (e.g., "Arnold"), treat it as Name=Arnold.
- If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching token in attribute_values.

================================================================================
1) DOMAIN OUTPUT (MANDATORY)
================================================================================
- "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
attribute_values immutability rule:
- The "attribute_values" object MUST be returned exactly as provided in the input.
- It must be identical:
  - Same attribute keys
  - Same ordering of keys
  - Same ordering of values within each list
  - Same casing and spelling
- Do NOT normalize, rename, reorder, add, or remove anything in "attribute_values".
- Normalization rules apply ONLY to syntactic_clues, reasoning, and solution — NOT to attribute_values.

================================================================================
2) syntactic_clues (MANDATORY, TEXTUAL CONSTRAINTS — NOT PREDICATES)
================================================================================
We do NOT use predicate-style DSL for clues.
Instead, each clue MUST be rewritten as a single-line *syntactic constraint statement* in a Z3-like textual form.

Rules:
- "syntactic_clues" MUST be a list of strings.
- For each clue, the selected tokens must be mapped to one of the values defined in attribute_values.
 - Example: If the clue says “sept” and attribute_values contains "september", use "september"; if attribute_values contains "sept", use "sept".
- There MUST be exactly one entry per clue, in the same order as the clues.
- Each entry MUST be exactly 1 line and end with a period.
- Each entry MUST start with the clue id prefix: "C<i>: ".
- Allowed syntactic operators in "syntactic_clues":
    ==   (same house / equivalence)
    !=   (not same house)
    <    (somewhere left of)
    >    (somewhere right of)
    + k == (k is a positive integer, e.g., 1 for immediately left, 2 for one house between, 3 for two houses between)
    == H  (fixed house index, where H is an integer)
- Examples:
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


Entry formatting rules:
- Each category’s value MUST be a list of strings.
- Each entry MUST be exactly 1 sentence and MUST end with a period.
- Reasoning MUST be interleaved within each category:
    Odd-numbered entries: Natural-language reasoning.
    Even-numbered entries: Syntactic reasoning step (Z3-like statement).
- All 10 keys MUST appear in reasoning, even if the value is an empty list []

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


Logical Validity Requirements:
    - Every syntactic step MUST be logically entailed by the syntactic_clues together with all preceding syntactic steps.
    - Forward references are forbidden: a syntactic step may only depend on clues or steps that appear earlier.
    - Do NOT output syntactic steps that merely restate a clue unless the restatement is required as an explicit anchor in a deduction chain.
    - Every syntactic step MUST be verifiable by a constraint solver (e.g., Z3). Informal, probabilistic, or unverifiable deductions are not allowed.
    
    
Category Semantics (Strict Binding)

Each syntactic reasoning step MUST belong to exactly one category below and MUST respect its semantics.
For every category, the natural-language explanation MUST justify the immediately following syntactic step.

    --------------------------------------------------
    Abs_Placement
    Fixes an entity or attribute to a specific house index.
    
    Natural-language example:
    "The tea drinker is explicitly stated to live in the third house."
    
    Syntactic example:
    S1: tea == 3.
    --------------------------------------------------
    
    --------------------------------------------------
    Direct_Equality
    Binds two entities or attributes to the same house or value.
    
    Natural-language example:
    "The clue states that Alice is the tea drinker, so Alice and tea must share the same house."
    
    Syntactic example:
    S2: Alice == tea.
    --------------------------------------------------
    
    --------------------------------------------------
    Directed_Adjacency
    Enforces exact adjacency with direction.
    
    Natural-language example:
    "Alice lives immediately to the left of Bob, which fixes their house positions relative to each other."
    
    Syntactic example:
    S3: Alice + 1 == Bob.
    --------------------------------------------------
    
    --------------------------------------------------
    Structural_Positioning
    Encodes relative ordering without fixing exact positions.
    
    Natural-language example:
    "The green house is somewhere to the left of the white house, but the exact distance is not specified."
    
    Syntactic example:
    S4: green < white.
    --------------------------------------------------
    
    --------------------------------------------------
    Domain_Restriction
    Removes one or more remaining candidate positions or values without fixing a unique assignment.
    
    Natural-language example:
    "The dog cannot be in the first house, so that position is excluded from its domain."
    
    Syntactic example:
    S5: Not(dog == 1).
    --------------------------------------------------
    
    --------------------------------------------------
    Exclusion
    Forbids a specific pairing or assignment between two entities or attributes.
    
    Natural-language example:
    "Alice does not own the dog, so Alice and dog cannot refer to the same house."
    
    Syntactic example:
    S6: Alice != dog.
    --------------------------------------------------
    
    --------------------------------------------------
    Propagation
    States an immediate logical consequence implied by existing constraints.
    
    Natural-language example:
    "Since tea is in house 3 and Alice is the tea drinker, Alice must also be in house 3."
    
    Syntactic example:
    S7: Alice == 3.
    --------------------------------------------------
    
    --------------------------------------------------
    Forced_Resolution
    Assigns a value because all other possibilities have been eliminated.
    
    Natural-language example:
    "All houses except house 2 have been ruled out for the cat, so the cat must be in house 2."
    
    Syntactic example:
    S8: cat == 2.
    --------------------------------------------------
    
    --------------------------------------------------
    Disjunction
    Represents unresolved alternatives and MUST use a solver-valid Or(...) expression.
    
    Natural-language example:
    "Given the remaining constraints, milk can only be in house 1 or house 5, but it is not yet determined which."
    
    Syntactic example:
    S9: Or(milk == 1, milk == 5).
    --------------------------------------------------
    
    --------------------------------------------------
    Case_Split
    Resolves a prior Disjunction by contradiction or systematic elimination.
    The resolved disjunction MUST be explicitly referenced.
    
    Natural-language example:
    "If milk were in house 1, it would contradict the adjacency constraint, so that case is eliminated."
    
    Syntactic example:
    S10: milk == 5.
    --------------------------------------------------


Global ordering rule:
- Categories do not impose order; S<k> is the only authoritative ordering.
- All dependencies must point from lower S<k> to higher S<k>.

Final invariant:
- Given syntactic_clues + all syntactic reasoning steps, a solver must verify every step with no ambiguity.

================================================================================
4) solution (MANDATORY — DERIVED FROM SYNTACTIC CLUES AND SYNTACTIC REASONING STEPS)
================================================================================

Purpose:
- "solution" is the final house-by-house assignment.
- Every placement in the solution MUST be derivable from syntactic_clues and
  syntactic reasoning steps (S1..Sk), under the background axioms below.
- Natural-language reasoning MUST NOT be used to justify solution placements.

--------------------------------------------------------------------------------
Background axioms (ASSUMED)
--------------------------------------------------------------------------------
- Each house has exactly one value per attribute (excluding House).
- Each attribute value appears in exactly one house.
- Values within the same attribute category are all-different.
- House indices are integers 1..n_houses.

--------------------------------------------------------------------------------
How syntactic facts determine solution placement (GENERIC RULES + EXAMPLES)
--------------------------------------------------------------------------------
All syntactic expressions denote HOUSE INDICES.
Let x, y be value tokens and H, H1, H2 be house indices.

1) Direct placement
   - S: x == H
     ⇒ Place x in solution row ["H", ...].

2) Equality propagation
   - S: x == y
   - S': y == H
     ⇒ Place both x and y in solution row ["H", ...].

3) Directed adjacency
   - S: x + 1 == y
   - S': x == H
     ⇒ y == H + 1
     ⇒ Place y in solution row ["H+1", ...].

4) Relative ordering
   - S: x > y
   - S': y == H
     ⇒ x ∈ {H+1, ..., n_houses}
     ⇒ Place x only in rows with index > H.

5) Disjunction resolution
   - S: Or(x == H1, x == H2)
   - S': x != H1
     ⇒ x == H2
     ⇒ Place x in solution row ["H2", ...].

6) Forced resolution (domain exhaustion)
   - If all houses except H are excluded for x
     ⇒ x == H
     ⇒ Place x in solution row ["H", ...].

7) Row interpretation
   - A solution row ["H", v1, v2, ..., vn] is interpreted as:
       v1 == H ∧ v2 == H ∧ ... ∧ vn == H

--------------------------------------------------------------------------------
Derivation rule (BINDING)
--------------------------------------------------------------------------------
- Every value in the solution MUST satisfy:
    vi == H is logically ENTAILED by
    (syntactic_clues + all syntactic reasoning steps + background axioms).

- Entailment may be:
    - Explicit (a reasoning step states vi == H), or
    - Implicit via solver-verifiable Forced_Resolution or domain exhaustion.

- If a complete assignment is NOT uniquely determined, the solution MUST still
  be present with the correct header and an empty "rows" list.

--------------------------------------------------------------------------------
Structure and consistency (STRICT)
--------------------------------------------------------------------------------
- "solution" MUST be a table with keys "header" and "rows".
- "solution.header" MUST match solution_header exactly.
- Rows MUST be ordered from house 1 to n_houses.
- A COMPLETE solution has exactly n_houses rows.
- All values MUST exactly match tokens in attribute_values.
- No solution placement may contradict any syntactic reasoning step.

--------------------------------------------------------------------------------
Final invariant
--------------------------------------------------------------------------------
Given only syntactic_clues, syntactic reasoning steps, and background axioms,
the solution table must be uniquely reconstructible.

================================================================================
ONE-SHOT EXAMPLE — Demonstration of REQUIRED OUTPUT (3 × 3 ZebraPuzzle)
================================================================================

There are 3 houses, numbered 1 to 3 from left to right.
Each house is occupied by a different person.
Each house has a unique attribute for each of the following characteristics:
- Each person has a unique name: Alice, Bob, Carol
- The people keep unique pets: cat, dog, fish
- Each person has a unique favorite drink: tea, coffee, milk

Clues:
1. Alice is the cat owner.
2. Bob keeps the dog.
3. The coffee drinker lives immediately to the right of the cat owner.
4. The fish owner lives somewhere to the right of the coffee drinker.
5. Carol drinks milk.

solution_header = ["House", "Name", "Pet", "Drink"]

attribute_values = {
  "Name": ["Alice", "Bob", "Carol"],
  "Pet": ["cat", "dog", "fish"],
  "Drink": ["tea", "coffee", "milk"]
}

Correct Example Output:
<answer>{
  "n_houses": 3,
  "attribute_values": {
    "Name": ["Alice", "Bob", "Carol"],
    "Pet": ["cat", "dog", "fish"],
    "Drink": ["tea", "coffee", "milk"]
  },
  "syntactic_clues": [
    "C1: Alice == cat.",
    "C2: Bob == dog.",
    "C3: cat + 1 == coffee.",
    "C4: fish > coffee.",
    "C5: Carol == milk."
  ],
  "reasoning": {
    "Abs_Placement": [
      "Because coffee must have a house to its right for fish, coffee cannot be in the last house.",
      "S1: coffee != 3. [C4]",
      "In a 3-house line, coffee must therefore be in house 1 or house 2.",
      "S2: Or(coffee == 1, coffee == 2). [S1]",
      "Since coffee is immediately right of cat, coffee cannot be in house 1.",
      "S3: coffee != 1. [C3]",
      "With coffee restricted to house 1 or 2 and excluded from house 1, coffee must be in house 2.",
      "S4: coffee == 2. [S2+S3]"
    ],
    "Direct_Equality": [
      "Clue 1 ties Alice to the cat house, so they share the same index.",
      "S5: Alice == cat. [C1]",
      "Clue 2 ties Bob to the dog house, so they share the same index.",
      "S6: Bob == dog. [C2]",
      "Clue 5 ties Carol to the milk house, so they share the same index.",
      "S7: Carol == milk. [C5]"
    ],
    "Directed_Adjacency": [
      "Clue 3 is a directed adjacency relating cat and coffee.",
      "S8: cat + 1 == coffee. [C3]"
    ],
    "Structural_Positioning": [
      "Clue 4 is a strict ordering: fish lies somewhere to the right of coffee.",
      "S9: fish > coffee. [C4]"
    ],
    "Domain_Restriction": [
      "If cat were in house 2 then coffee would be in house 3 by adjacency, which would contradict coffee == 2.",
      "S10: Implies(cat == 2, coffee == 3). [S8]",
      "Given coffee == 2, cat cannot be in house 2 and must be in house 1.",
      "S11: cat == 1. [S4+S10]"
    ],
    "Exclusion": [
      "Since cat is in house 1, coffee cannot be in house 3 and must remain consistent with coffee == 2.",
      "S12: And(cat != 3, coffee != 3). [S11+S1]"
    ],
    "Propagation": [
      "From cat == 1 and cat + 1 == coffee, coffee must be in house 2, confirming the placement.",
      "S13: Implies(cat == 1, coffee == 2). [S8+S11]"
    ],
    "Forced_Resolution": [
      "Because Alice shares the cat house and cat is in house 1, Alice is in house 1.",
      "S14: Alice == 1. [S5+S11]",
      "Coffee is in house 2, so fish must be to the right of house 2 and is forced into house 3.",
      "S15: fish == 3. [S4+S9]",
      "Since fish is in house 3, the remaining pet for house 2 is dog.",
      "S16: dog == 2. [S11+S15]",
      "Bob shares the dog house, so Bob is in house 2.",
      "S17: Bob == 2. [S6+S16]",
      "With Alice in house 1 and Bob in house 2, the remaining name Carol is in house 3.",
      "S18: Carol == 3. [S14+S17]",
      "Carol shares the milk house, so milk is in house 3.",
      "S19: milk == 3. [S7+S18]",
      "With coffee in house 2 and milk in house 3, the remaining drink tea is in house 1.",
      "S20: tea == 1. [S4+S19]"
    ],
    "Disjunction": [
      "Before excluding coffee == 1, coffee had exactly two candidate houses under the ordering constraint.",
      "S21: Or(coffee == 1, coffee == 2). [S1]"
    ],
    "Case_Split": [
      "If coffee were in house 1 then cat would need to be in house 0 by adjacency, which is impossible, so coffee cannot be 1.",
      "S22: coffee != 1. [S8]"
    ]
  },
  "solution": {
    "header": ["House", "Name", "Pet", "Drink"],
    "rows": [
      ["1", "Alice", "cat", "tea"],
      ["2", "Bob", "dog", "coffee"],
      ["3", "Carol", "fish", "milk"]
    ]
  }
}</answer>

"""

SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

puzzle = {puzzle}

solution_header = {solution_header}

attribute_values = {attribute_values}

Solve the puzzle above and provide n_houses, attribute_values, syntactic_clues, reasoning, and solution for this puzzle in the <answer> </answer> block, with no additional text.
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
            # v = row[i]
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
            puzzle=example['puzzle'], solution_header=final_grid['header'], attribute_values=attribute_values_from_solution(example['solution']))

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
            print(5 * "\n\n")
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
    train_output_dir = os.path.join(args.output_dir, "train")
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
