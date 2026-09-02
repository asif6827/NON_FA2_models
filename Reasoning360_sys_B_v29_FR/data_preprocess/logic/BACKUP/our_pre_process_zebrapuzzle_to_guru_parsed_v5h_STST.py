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
- Use ONLY these syntactic operators in the clue text:
    ==   (same house / equivalence)
    !=   (not same house)
    <    (somewhere left of)
    >    (somewhere right of)
    + k == (k is a positive integer, e.g., 1 for immediately left, 2 for one house between, 3 for two houses between)
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

- The reasoning JSON object MUST have exactly ELEVEN top-level keys, spelled EXACTLY:

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
- "Self_Verification"


Entry formatting rules:
- Each category’s value MUST be a list of strings.
- Each entry MUST be exactly 1 sentence and MUST end with a period.
- Reasoning MUST be interleaved within each category:
    Odd-numbered entries: Natural-language reasoning.
    Even-numbered entries: Syntactic reasoning step (Z3-like statement).
    Exception: Self_Verification is NL-only and does not require interleaving with syntactic steps.
- All 11 keys MUST appear in reasoning, even if the value is an empty list []

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
- "Self_Verification": Verifies that each reasoning step is justified by the given clues or earlier steps, and that all derived constraints jointly satisfy and reproduce the final solution.


Global ordering rule:
- Categories do not impose order; S<k> is the only authoritative ordering.
- All dependencies must point from lower S<k> to higher S<k>.

Final invariant:
- Given syntactic_clues + all syntactic reasoning steps, a solver must verify every step with no ambiguity.

================================================================================
4) solution (MANDATORY — DERIVED FROM SYNTACTIC CLUES AND SYNTACTIC REASONING STEPS)
================================================================================

Purpose:
- The "solution" represents the final, complete assignment of all attributes to houses.
- The solution MUST be derived exclusively from:
    (a) syntactic_clues, and
    (b) syntactic reasoning steps (S1..Sk).
- The solution MUST NOT be independently guessed, inferred, or reasoned in natural language.

Derivation rule (binding invariant):
- The solution MUST be fully determined by:
    1) The complete set of syntactic_clues (C1..Cm), and
    2) The complete set of syntactic reasoning steps (S1..Sk).

- If the syntactic_clues and syntactic reasoning steps do not uniquely determine a
  complete assignment, the solution MUST be omitted or explicitly marked incomplete.

--------------------------------------------------------------------------------
Syntactic grounding requirement (MANDATORY)
--------------------------------------------------------------------------------
- For every row ["H", v1, v2, ..., vn] in the solution table,
  and for every value vi in that row (excluding "H"),
  there MUST exist at least one syntactic reasoning step S<k>
  that explicitly entails vi == H, using one of the following forms:
    - vi == H
    - And(..., vi == H, ...)
    - A conjunction or implication whose resolved branch entails vi == H

- Pure uniqueness-based or implicit assignment is NOT allowed.
- Every house assignment MUST be traceable to at least one syntactic reasoning step.
- Natural-language explanations MUST NOT be used to justify solution assignments.

--------------------------------------------------------------------------------
Structure and house semantics (STRICT)
--------------------------------------------------------------------------------
- "solution" MUST be expressed in tabular form with:
  - "header": a list of column names.
  - "rows": a list of rows, each row being a list of strings matching the header order.

Header requirements:
- "solution.header" MUST match "solution_header" exactly.

Row requirements:
- Each row represents exactly ONE house.
- The FIRST element of each row MUST be the house index as a string
  (e.g., "1", "2", ..., "n_houses").
- The house index in a row defines the house H for ALL remaining values in that row.

- A row of the form:
    ["H", v1, v2, ..., vn]
  MUST be interpreted as the conjunction:
    v1 == H
    v2 == H
    ...
    vn == H

- Therefore, every non-house value appearing in a row for house H
  MUST be syntactically entailed to be in house H by the reasoning steps.

- Rows MUST be listed in strictly increasing house index order from 1..n_houses.
- There MUST be exactly n_houses rows.

--------------------------------------------------------------------------------
Normalization and consistency requirements
--------------------------------------------------------------------------------
- All solution values MUST be normalized using underscores.
- Values MUST exactly match tokens defined in "attribute_values".

- Every value in the solution table MUST be entailed by at least one syntactic reasoning step.
- The solution MUST be consistent with every syntactic reasoning step.
- No value may appear in the solution unless it is justified by the syntactic reasoning.

--------------------------------------------------------------------------------
Prohibited behavior
--------------------------------------------------------------------------------
- The solution MUST NOT introduce new facts not present or implied in the syntactic reasoning steps.
- The solution MUST NOT contradict any syntactic reasoning step.
- The solution MUST NOT rely on natural-language reasoning text.
- The solution MUST NOT assign a value to a house unless that assignment is explicitly entailed.

--------------------------------------------------------------------------------
Final invariant
--------------------------------------------------------------------------------
- Given only the syntactic_clues and syntactic reasoning steps,
  the solution table must be uniquely reconstructible with no ambiguity.

--------------------------------------------------------------------------------
Examples (Illustrative, NOT additional output)
--------------------------------------------------------------------------------

VALID:
- Row ["2", "Alice", "cat", "tea"] means:
    Alice == 2, cat == 2, tea == 2
  and each of these MUST be justified by syntactic reasoning steps.

VALID:
- If coffee appears in row ["3", ..., "coffee"],
  then at least one syntactic step MUST entail coffee == 3
  (directly or via a resolved conjunction or implication).

VALID:
- If Bob appears in row ["1", "Bob", ...] because Bob == milk and milk == 1,
  then BOTH syntactic steps MUST exist.

INVALID:
- Assigning a value to a house without any syntactic step that entails value == house.
- Using uniqueness alone to justify a house assignment.
- Introducing a house index in the solution that never appears in any syntactic reasoning step.

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
2. Bob drinks milk.
3. The coffee drinker lives immediately to the right of the cat owner.
4. The fish owner lives somewhere to the right of the milk drinker.
5. Carol does not drink coffee.

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
    "C2: Bob == milk.",
    "C3: cat + 1 == coffee.",
    "C4: fish > milk.",
    "C5: Carol != coffee."
  ],
  "reasoning": {
    "Abs_Placement": [
      "Since coffee must be immediately right of cat and there are only 3 houses, coffee cannot be in house 1, so coffee must be in house 2 or 3.",
      "S1: Or(coffee == 2, coffee == 3). [C3]"
    ],
    "Direct_Equality": [
      "Clue 1 binds Alice and cat, so wherever Alice is placed, cat is placed in the same house.",
      "S2: Alice == cat. [C1]",
      "Clue 2 binds Bob and milk, so wherever Bob is placed, milk is placed in the same house.",
      "S3: Bob == milk. [C2]"
    ],
    "Directed_Adjacency": [
      "The adjacency clue directly constrains cat to be exactly one house left of coffee.",
      "S4: cat + 1 == coffee. [C3]"
    ],
    "Structural_Positioning": [
      "Fish must be strictly to the right of milk, creating an ordering constraint between their house indices.",
      "S5: fish > milk. [C4]"
    ],
    "Domain_Restriction": [
      "Because Carol is not coffee, if coffee were in house 2 then Carol would be forced into house 3, otherwise Carol could still be in house 1 or 3.",
      "S6: Implies(coffee == 2, Carol == 3). [C5]"
    ],
    "Exclusion": [
      "Since coffee is immediately right of cat, cat cannot be in house 3 because there is no house 4 for coffee.",
      "S7: cat != 3. [S4]",
      "Since Bob drinks milk, Bob cannot drink tea or coffee by drink uniqueness.",
      "S8: And(Bob != tea, Bob != coffee). [S3]"
    ],
    "Propagation": [
      "From cat != 3 and cat + 1 == coffee, if cat were in house 1 then coffee would be in house 2, and if cat were in house 2 then coffee would be in house 3.",
      "S9: Or(And(cat == 1, coffee == 2), And(cat == 2, coffee == 3)). [S4+S7]",
      "If cat were in house 1 then coffee would be in house 2, forcing Carol to be in house 3 by the Carol != coffee restriction.",
      "S10: Implies(cat == 1, Carol == 3). [S6+S9]",
      "If Carol were in house 3, then Alice cannot be in house 3 and must be the cat owner, so cat cannot be in house 1 if it would force a contradiction with fish placement.",
      "S11: Implies(Carol == 3, Alice != 3). [S2]"
    ],
    "Forced_Resolution": [
      "If milk were in house 3 then fish would need to be to the right of house 3 which is impossible, so milk cannot be in house 3.",
      "S12: milk != 3. [S5]",
      "Since milk cannot be in house 3 and Bob == milk, Bob also cannot be in house 3.",
      "S13: Bob != 3. [S3+S12]",
      "Given Bob cannot be in house 3 and Alice is tied to cat which cannot be in house 3, the only consistent placement is Bob in house 1 and Alice in house 2.",
      "S14: And(Bob == 1, Alice == 2). [S2+S3+S7+S13]",
      "With Alice in house 2 and Alice == cat, cat is in house 2, and thus coffee is in house 3 by adjacency.",
      "S15: And(cat == 2, coffee == 3). [S2+S4+S14]",
      "Since fish must be to the right of milk and milk is in house 1, fish is forced into house 3.",
      "S16: fish == 3. [S5+S14]"
    ],
    "Disjunction": [
      "Given milk != 3 and the uniqueness of house indices, milk must be in house 1 or house 2.",
      "S17: Or(milk == 1, milk == 2). [S12]"
    ],
    "Case_Split": [
      "If milk were in house 2 then fish would have to be in house 3, but coffee is already in house 3 and Carol cannot be coffee, forcing Carol into house 1 and leaving no place for Bob == milk, so milk cannot be in house 2.",
      "S18: milk != 2. [S3+S5+S15+C5]",
      "Since milk is either in house 1 or 2 and it is not in house 2, milk must be in house 1.",
      "S19: milk == 1. [S17+S18]"
    ],
    "Self_Verification": [
      "S1 (Or(coffee==2,coffee==3)) is satisfied because in the final solution coffee is in house 3. — verified by the final solution.",
      "S2 (Alice==cat) is satisfied because Alice and cat both appear in house 2. — verified by the final solution.",
      "S3 (Bob==milk) is satisfied because Bob and milk both appear in house 1. — verified by the final solution.",
      "S4 (cat+1==coffee) is satisfied because cat is in house 2 and coffee is in house 3, and 2+1=3. — verified by the final solution.",
      "S5 (fish>milk) is satisfied because fish is in house 3 and milk is in house 1, and 3>1. — verified by the final solution.",
      "S6 (Implies(coffee==2,Carol==3)) is satisfied because coffee is not in house 2 in the final solution, so the implication holds. — verified by the final solution.",
      "S7 (cat!=3) is satisfied because cat appears in house 2, not house 3. — verified by the final solution.",
      "S8 (And(Bob!=tea,Bob!=coffee)) is satisfied because Bob appears in house 1 with milk, not tea or coffee. — verified by the final solution.",
      "S9 (Or(And(cat==1,coffee==2),And(cat==2,coffee==3))) is satisfied by the second conjunct because cat is in house 2 and coffee is in house 3. — verified by the final solution.",
      "S10 (Implies(cat==1,Carol==3)) is satisfied because cat is not in house 1 in the final solution, so the implication holds. — verified by the final solution.",
      "S11 (Implies(Carol==3,Alice!=3)) is satisfied because Carol is in house 3 and Alice is in house 2. — verified by the final solution.",
      "S12 (milk!=3) is satisfied because milk appears in house 1, not house 3. — verified by the final solution.",
      "S13 (Bob!=3) is satisfied because Bob appears in house 1, not house 3. — verified by the final solution.",
      "S14 (And(Bob==1,Alice==2)) is satisfied because Bob is in house 1 and Alice is in house 2. — verified by the final solution.",
      "S15 (And(cat==2,coffee==3)) is satisfied because cat is in house 2 and coffee is in house 3. — verified by the final solution.",
      "S16 (fish==3) is satisfied because fish appears in house 3. — verified by the final solution.",
      "S17 (Or(milk==1,milk==2)) is satisfied because milk appears in house 1. — verified by the final solution.",
      "S18 (milk!=2) is satisfied because milk appears in house 1, not house 2. — verified by the final solution.",
      "S19 (milk==1) is satisfied because milk appears in house 1. — verified by the final solution.",
      "All steps S1–S19 are consistent with the final solution table, and the solution satisfies all constraints derived from the clues and reasoning. — verified by the final solution."
    ]
  },
  "solution": {
    "header": ["House", "Name", "Pet", "Drink"],
    "rows": [
      ["1", "Bob", "dog", "milk"],
      ["2", "Alice", "cat", "tea"],
      ["3", "Carol", "fish", "coffee"]
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
