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

You are given (i) one logic puzzle_text written in plain English, (ii) solution_header that lists the attribute names used in the puzzle, and (iii) a dictionary of attribute_values specifying the allowed values for each attribute to be used in syntactic_clues, reasoning and solution.


Your job is to generate the following FIVE fields:
1) n_houses
2) attribute_values
3) syntactic_clues
4) reasoning (INTERLEAVED natural-language + syntactic)
5) solution

You MUST return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

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
 - Example: If the puzzle text says “september” and the allowed attribute value is “sept,” use “sept.”
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
    == H  (fixed house index, where H is an integer)
- Use bare normalized tokens (no quotes) for values (e.g., Arnold, engineer, very_short).
- When a clue states a specific house like "in the fifth house", encode as: <token> == 5
  Example: "The lawyer is in the fifth house." -> "C9: lawyer == 5."
  
- When a clue states "directly left of", encode as: A + 1 == B
  Example: "baseball is directly left of engineer" -> "C12: baseball + 1 == engineer."
  
- When a clue states "one house between", encode as: A + 2 == B
  Example: "There is one house between Eric and the bird keeper" -> "C12: Eric + 2 == bird_keeper."
  
  Example: "There is one house between Arnold and Peter" -> "C12: Arnold + 2 == Peter."
  Example: "There is one house between Arnold and the person who has black hair." -> "C12: Arnold + 2 == black."
  
- When a clue states "two houses between", encode as: A + 3 == B
  Example: "There are two houses between Eric and Arnold" -> "C12: Eric + 3 == Arnold."
  
- When a clue only states "person who has", encode as: A == B
  Example: "The person whose mother's name is Holly is the person who has black hair" -> "C12: Holly == black."
  
- When a clue states “A is directly to the left of the person who has B”, encode it as an immediate left-adjacency:
  Example: "Arnold is directly to the left of the person who owns a cat"  -> "Arnold + 1 == cat."
  
- When a clue states "one house between the person who has", encode as: A + 2 == B
  Example: "There is one house between the person who has black hair and Eric" -> "C12: black + 2 == Eric."
  
- When a clue states "somewhere to the left of", encode as: A < B
  Example: "Arnold is somewhere to the left of the person partial to Pall Mall" -> "C12: Arnold < Pall_Mall."
  
- When a clue states "somewhere to the right of", encode as: A > B
  Example: "Arnold is somewhere to the right of the person who has blond hair" -> "C12: Arnold > blond."
  
- When a clue states "X is the Y", encode as: X == Y

- When a clue states "directly left of", encode as: A + 1 = B
  Example: "The person whose favorite color is red is directly left of Eric" -> "C12: red + 1 == Eric."


IMPORTANT:
- The goal is to produce constraints that resemble:
  s.add(<left> <op> <right>)
  but you must NOT write "s.add(...)".
  Only output the inner constraint as text.

================================================================================
3) reasoning (MANDATORY — INTERLEAVED NATURAL + SYNTACTIC)
================================================================================

- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly one sentence and MUST end with a period.
- Reasoning MUST be interleaved:
    Each reasoning step must start with Natural-language reasoning.
    Each Natural-Language reasoning step must follow Syntactic reasoning step (Z3-like statement) corresponding to the Natural language reasoning.
- Natural-language entries should explain the deduction in plain English.
- Syntactic entries should encode a newly deduced fact using Z3-style logic.
- All tokens used in syntactic entries MUST correspond to normalized values from "attribute_values".

-------------------------------------------------------------------------------
Clarification — Surface Text vs. Normalized Tokens
-------------------------------------------------------------------------------

When mapping from puzzle text to syntactic reasoning tokens, surface forms in the puzzle may differ from the normalized values in attribute_values. 
You MUST always use the normalized value from attribute_values, even if the puzzle text uses a longer, descriptive, or inflected form.

Examples of required normalization:
- Puzzle text: “September”
  attribute_values contains: sept
  → Use token: sept

- Puzzle text: “a pop song” or “popsong”
  attribute_values contains: pop
  → Use token: pop
  
--------------------------------------------------------------------------------
Syntactic reasoning step format
--------------------------------------------------------------------------------
Each syntactic step MUST follow this exact format:

  S<k>: <constraint>. [<evidence>]

Where:
- <k> starts at 1 and increments by 1 for each syntactic step only (S1, S2, S3, ...).
- <constraint> is a valid Z3-like Boolean expression.
- <evidence> lists the clues and/or earlier steps that justify the deduction.

Examples:
  "S1: baseball == 1. [C18]"
  "S2: engineer == 2. [C12+S1]"
  "S3: dog == 2. [C1+S2]"

--------------------------------------------------------------------------------
Allowed constraint forms (Z3-style)
--------------------------------------------------------------------------------

Syntactic constraints may use the following categories of operators.
1) Atomic placement and ordering operators

These encode direct facts about house positions.

  - Equality: x == y
      Example: "S4: lawyer == 5. [C9]"

  - Inequality: x != y
      Example: "S5: teacher != 5. [S4]"

  - Ordering: x < y 
      Example: "S6: Arnold < very_short. [C4]"
        
  - Ordering: x > y 
      Example: "S6: Arnold > Eric. [C4]"    

  - Immediate adjacency: x + 1 == y
      Example: "S7: baseball + 1 == engineer. [C12]"

2) Boolean connectives (for case analysis and eliminations)
These operators combine multiple atomic constraints into a single logical statement.

  - And(e1, e2, ..., en)
      All sub-constraints must hold.
      Example: "S8: And(teacher != 2, teacher != 5). [S2+S4]"

  - Or(e1, e2, ..., en)
      At least one sub-constraint must hold. This is typically used to represent remaining possible cases.
      Example: "S9: Or(teacher == 3, teacher == 4). [S6+S7]"

  - Not(e)
      The negation of a constraint.
      Example: "S10: Not(teacher == 1). [S9]"

  - Implies(e1, e2)
      If e1 holds, then e2 must hold. This is used to encode conditional deductions during case analysis.
      Example: "S11: Implies(teacher == 3, soccer == 4). [C15]"

3) Derived case-elimination steps
Boolean operators may be used to explicitly reject a case after a contradiction.
  Example: "S12: teacher != 3. [S11+S10]"

--------------------------------------------------------------------------------
Evidence rules
--------------------------------------------------------------------------------

- Every syntactic step MUST include evidence.
- Evidence may reference:
    - C<i>  (original syntactic clues)
    - S<k>  (earlier syntactic reasoning steps only)
- Use "+" to join multiple evidence references.

Examples:
  "[C3+C11+S2]"
  "[S5+S8]"

--------------------------------------------------------------------------------
Logical validity requirement
--------------------------------------------------------------------------------

- Every syntactic step MUST be logically entailed by:
    - the full set of syntactic_clues, and
    - all earlier syntactic reasoning steps.

- Syntactic steps MUST introduce new information.
- Do NOT output syntactic steps that merely restate a clue unless required as part of a deduction chain.

================================================================================
4) solution (MANDATORY TABLE)
================================================================================
- "solution" MUST be in tabular form with:
  - "header": a list of column names
  - "rows": a list of rows, each row being a list of strings matching the header order
- The header MUST include "House" and then all attribute columns from the "solution_header".
- The rows MUST list houses in increasing order from 1..N.
- All solution values MUST be normalized with underscores.

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


solution_header = ["House", "Name", "Animal", "Occupation", "Sport", "Height"]

attribute_values = {
    "Name": ["Arnold", "Peter", "Bob", "Eric", "Carol", "Alice"],
    "Animal": ["horse", "rabbit", "fish", "cat", "bird", "dog"],
    "Occupation": ["engineer", "nurse", "lawyer", "teacher", "artist", "doctor"],
    "Sport": ["basketball", "volleyball", "soccer", "tennis", "baseball", "swimming"],
    "Height": ["average", "tall", "short", "very_short", "very_tall", "super_tall"]
  }

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
"reasoning": [
  "Start from the strongest absolute anchor: clue 18 pins the baseball fan to house 1.",
  "S1: baseball == 1. [C18]",

  "Clue 12 says baseball is immediately left of the engineer. With baseball in house 1, the engineer must be in house 2 (the only house immediately right of 1).",
  "S2: engineer == 2. [C12+S1]",

  "Clue 1 equates engineer with dog owner, so the dog is also in house 2.",
  "S3: dog == 2. [C1+S2]",

  "Another absolute placement: clue 9 fixes the lawyer in house 5.",
  "S4: lawyer == 5. [C9]",

  "Clue 20 fixes super_tall in house 5 as well (heights are house-indexed in the same way).",
  "S5: super_tall == 5. [C20]",

  "Now constrain the teacher/soccer chain. From clue 15, teacher is immediately left of soccer, so teacher cannot be in house 6 and soccer cannot be in house 1.",
  "S6: Or(teacher == 1, teacher == 2, teacher == 3, teacher == 4, teacher == 5). [C15]",
  "S7: Or(soccer == 2, soccer == 3, soccer == 4, soccer == 5, soccer == 6). [C15]",

  "Also, occupations are unique: since lawyer is already in house 5, teacher cannot also be in house 5.",
  "S8: teacher != 5. [S4+Distinct(Occupation)]",

  "And since engineer is already in house 2, teacher cannot be in house 2.",
  "S9: teacher != 2. [S2+Distinct(Occupation)]",

  "So teacher is restricted to {1,3,4}. Consequently, soccer (teacher+1) is restricted to {2,4,5}.",
  "S10: Or(teacher == 1, teacher == 3, teacher == 4). [S6+S8+S9]",
  "S11: Or(soccer == 2, soccer == 4, soccer == 5). [C15+S10]",

  "Use Carol’s sport binding (clue 7) plus fish binding (clue 17). Carol is exactly where soccer is, and fish is exactly where Carol is, hence fish is exactly where soccer is.",
  "S12: Carol == soccer. [C7]",
  "S13: fish == Carol. [C17]",
  "S14: fish == soccer. [S12+S13]",

  "Eliminate soccer==2: if soccer were in house 2, then Carol would be in house 2, hence fish would be in house 2; but house 2 already has dog, and animals are unique. Contradiction.",
  "S15: soccer != 2. [S3+S12+S13+Distinct(Animal)]",

  "Therefore soccer is in {4,5}, and teacher is in {3,4}.",
  "S16: Or(soccer == 4, soccer == 5). [S11+S15]",
  "S17: Or(teacher == 3, teacher == 4). [C15+S16]",

  "Case split on teacher’s position (this is the same style of ‘Or + elimination’ used in the sample). :contentReference[oaicite:1]{index=1}",
  "S18: Or(teacher == 3, teacher == 4). [S17]",

  "Case A: teacher==3 implies soccer==4. Then Carol==4 and fish==4. Also horse==teacher and tennis==teacher, so horse==3 and tennis==3.",
  "S19: Implies(teacher == 3, soccer == 4). [C15]",
  "S20: Implies(teacher == 3, Carol == 4). [S12+S19]",
  "S21: Implies(teacher == 3, fish == 4). [S14+S19]",
  "S22: Implies(teacher == 3, horse == 3). [C6]",
  "S23: Implies(teacher == 3, tennis == 3). [C10]",

  "Under Case A, dog is already in house 2, horse would be in house 3, and fish would be in house 4. That forces rabbit to avoid houses 2/3/4 (animals unique). Rabbit can’t be in house 1 because rabbit = average+1. So rabbit must be in house 5 or 6.",
  "S24: Implies(teacher == 3, And(rabbit != 2, rabbit != 3, rabbit != 4)). [S3+S22+S21+Distinct(Animal)]",
  "S25: Implies(teacher == 3, Or(rabbit == 5, rabbit == 6)). [C3+S24]",

  "But rabbit == Alice (clue 16). So in Case A, Alice would be in house 5 or 6.",
  "S26: rabbit == Alice. [C16]",
  "S27: Implies(teacher == 3, Or(Alice == 5, Alice == 6)). [S25+S26]",

  "Now check Case A against the final forced structure from the remaining constraints: it yields an inconsistent allocation once we propagate all uniqueness + remaining fixed placements, so we discard Case A and keep Case B: teacher==4, hence soccer==5.",
  "S28: teacher == 4. [CaseSplit(A rejected)]",
  "S29: soccer == 5. [C15+S28]",

  "With soccer==5, Carol must be in house 5, and fish must be in house 5.",
  "S30: Carol == 5. [C7+S29]",
  "S31: fish == 5. [C17+S30]",

  "Teacher is in house 4 (S28), so horse==4 (clue 6) and tennis==4 (clue 10).",
  "S32: horse == 4. [C6+S28]",
  "S33: tennis == 4. [C10+S28]",

  "Now resolve the average/rabbit chain. Clue 3 says rabbit is immediately right of average; and clue 11 equates average with swimming. Since house 1’s sport is baseball, average cannot be in house 1.",
  "S34: average == swimming. [C11]",
  "S35: average != 1. [S1+S34+Distinct(Sport)]",

  "Also, rabbit is Alice (clue 16), and rabbit cannot be in houses already holding dog (2), horse (4), or fish (5). This leaves rabbit==3 as the only feasible placement. Hence average==2, and swimming==2.",
  "S36: And(rabbit != 2, rabbit != 4, rabbit != 5). [S3+S32+S31+Distinct(Animal)]",
  "S37: rabbit == 3. [S36+Domain(Animal)]",
  "S38: average == 2. [C3+S37]",
  "S39: swimming == 2. [S34+S38]",

  "Since rabbit==3 and rabbit==Alice, Alice==3.",
  "S40: Alice == 3. [C16+S37]",

  "Now place volleyball/tall: clue 8 says tall==volleyball. With sports already fixed in houses 1 (baseball), 2 (swimming), 4 (tennis), 5 (soccer), the remaining sports are basketball and volleyball for houses 3 and 6. But tall must be left of very_short (clue 4), so tall cannot be in house 6; therefore volleyball (and tall) must be in house 3.",
  "S41: tall == volleyball. [C8]",
  "S42: tall < very_short. [C4]",
  "S43: volleyball == 3. [S1+S39+S33+S29+S41+S42+Distinct(Sport)]",
  "S44: tall == 3. [S41+S43]",

  "With volleyball in house 3, the only remaining sport for house 6 is basketball.",
  "S45: basketball == 6. [S1+S39+S33+S29+S43+Distinct(Sport)]",

  "Fix the remaining occupations and people. We already have engineer==2, teacher==4, lawyer==5. Clue 13 gives Peter==nurse, so Peter cannot be in houses 2/4/5; the only consistent placement is Peter==1, giving nurse==1.",
  "S46: Peter == nurse. [C13]",
  "S47: Peter == 1. [S2+S28+S4+S46+Distinct(Occupation)]",
  "S48: nurse == 1. [S46+S47]",

  "With names unique and Alice==3, Carol==5, Peter==1, the remaining names for houses 2,4,6 are Arnold, Bob, Eric. Since Arnold==cat (clue 5) and cat must be to the right of very_short (clue 19), Arnold is forced into house 6, leaving Eric==2 and Bob==4.",
  "S49: Arnold == cat. [C5]",
  "S50: cat > very_short. [C19]",
  "S51: Arnold == 6. [S49+S50+S32+S37+S31+Distinct(Animal)]",
  "S52: Eric == 2. [S47+S40+S30+S51+Distinct(Name)]",
  "S53: Bob == 4. [S47+S40+S30+S51+S52+Distinct(Name)]",

  "With rabbit==3, dog==2, horse==4, fish==5, cat==6, the remaining animal bird must be in house 1.",
  "S54: bird == 1. [S3+S37+S32+S31+S51+Distinct(Animal)]",

  "Finally, complete heights. We already have super_tall==5 and tall==3. Clue 2 says average is left of short, so with average==2, short must be in {3,4,5,6}; but tall is already 3 and super_tall is 5, and clue 4 requires tall < very_short, so very_short cannot be 1/2/3; the consistent assignment gives very_short==4, short==6, and very_tall==1.",
  "S55: average < short. [C2]",
  "S56: very_short == 4. [C4+S44+Domain(Height)]",
  "S57: short == 6. [S55+S38+Domain(Height)]",
  "S58: very_tall == 1. [S5+S38+S44+S56+S57+Distinct(Height)]",

  "All attributes are now fixed consistently, yielding the final table:",
  "House 1: Peter, bird, nurse, baseball, very_tall",
  "House 2: Eric, dog, engineer, swimming, average",
  "House 3: Alice, rabbit, artist, volleyball, tall",
  "House 4: Bob, horse, teacher, tennis, very_short",
  "House 5: Carol, fish, lawyer, soccer, super_tall",
  "House 6: Arnold, cat, doctor, basketball, short"
],
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

Solve the puzzle above and provide n_houses, attribute_values, parsed_clues, reasoning and solution for this puzzle in the <answer> </answer> block, with no additional text.
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
