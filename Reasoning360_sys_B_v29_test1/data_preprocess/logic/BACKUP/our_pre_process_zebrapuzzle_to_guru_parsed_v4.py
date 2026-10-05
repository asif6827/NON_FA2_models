import os
import sys
import datasets
import argparse
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset


# Import prompt templates directly
SOLUTION_PROMPT_1_SHOT_SYS_BKP = """You are an expert logic puzzle solver. You will be given ONE logic puzzle in plain English.

Your task is to:
  - Extract the domain (N houses + all attribute values).
  - Parse each clue into a canonical, machine-checkable form (DSL).
  - Produce step-by-step deductions using only the DSL steps format.
  - Provide the final solution table.
  - Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
  - Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
  - Do NOT include extra text, markdown, explanations, or code fences.
  - Inside <answer>...</answer>, the content MUST be a single valid JSON object.
  - The JSON object MUST have exactly FIVE top-level keys:
      "n_houses", "attribute_values", "parsed_clues", "parsed_reasoning", "solution".
  - Do NOT add any other keys.

NORMALIZATION RULES:
  - Use underscores instead of spaces in VALUES (e.g., grilled_cheese, root_beer, very_short).
  - Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Animal, Occupation, Sport, Height, etc.
  - House numbers are integers 1..N.
  - Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6.
  - Do NOT invent values. Every <Val> must be one of the allowed values listed in the puzzle text (after normalization).
  - If the clue mentions a bare person name (e.g., "Arnold"), treat it as Name=Arnold.
  - IMPORTANT (fix for “Arnold is the person whose favorite color is red”):
      Statements of the form “X is the person who has Y” MUST be encoded as:
      same_house(Name=X, <Attr>=<Val>)
      Example: “Arnold is the person whose favorite color is red.” =>
               same_house(Name=Arnold, FavoriteColor=red)

D) DOMAIN OUTPUT (MANDATORY)
  - "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
  - "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values from the puzzle text.
  - Each attribute list MUST contain exactly N unique values (after normalization).
  - Include every attribute listed in the puzzle text, and only those attributes.
  - Do NOT infer extra attributes that are not explicitly listed in the puzzle text.

A) parsed_clues (MANDATORY, PARSABLE)
  - "parsed_clues" MUST be a list of strings.
  - Each string must be exactly 1 sentence and end with a period.
  - There MUST be exactly one entry per clue, in the same order as the clues.
  - Each parsed clue MUST follow this exact DSL format:

    C<i> = <predicate>.

Allowed <predicate> forms (use exactly these):
  - set(<H>,<Attr>,<Val>)
  - not_set(<H>,<Attr>,<Val>)
  - immediately_left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - right_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - adjacent(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - same_house(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - between(<AttrA>=<ValA>,<AttrB>=<ValB>,<K>)

Semantics:
  - immediately_left_of(A,B): A is exactly 1 house left of B.
  - left_of(A,B): A is somewhere left of B (strictly smaller house index).
  - right_of(A,B): A is somewhere right of B (strictly larger house index).
  - adjacent(A,B): houses differ by exactly 1.
  - between(A,B,K): there are exactly K houses strictly between A and B.
    (So K=1 => positions differ by 2, and K=2 => positions differ by 3.)

B) parsed_reasoning (MANDATORY, PARSABLE)
  - "parsed_reasoning" MUST be a list of strings.
  - Each string must be exactly 1 sentence and end with a period.
  - There is NO LIMIT on the number of entries.
  - Each entry MUST follow this exact DSL format:

    S<k> [C<i>(+C<j>...)] <op>(<H>,<Attr>,<Val>).

Where:
  - <k> is a step number starting at 1 and increasing by 1 each step.
  - Evidence inside [...] must reference clue ids, e.g. [C1] or [C1+C3].
  - <op> is either set or not.
  - <H> is a house number integer (1..N).
  - <Attr> and <Val> must come from the puzzle text (normalized).

LOGICAL VALIDITY REQUIREMENT:
  - Every step in "parsed_reasoning" MUST be logically entailed by the parsed clues plus any earlier reasoning steps.
  - If you cannot deduce a set(...) fact with certainty, output a not(...) fact that is guaranteed true.

C) solution (MANDATORY TABLE)
  - "solution" MUST be in tabular form with:
    - "header": a list of column names
    - "rows": a list of rows, each row being a list of strings matching the header order.
  - The header MUST include "House" and then all attribute columns from the puzzle text.
  - The rows MUST list houses in increasing order from 1..N.
  - All solution VALUES must be normalized with underscores.

================================================================================
ONE-SHOT EXAMPLE (20 clues) — Demonstration of the REQUIRED JSON OUTPUT FORMAT
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
  "parsed_clues": [
    "C1 = same_house(Occupation=engineer,Animal=dog).",
    "C2 = left_of(Height=average,Height=short).",
    "C3 = immediately_left_of(Height=average,Animal=rabbit).",
    "C4 = left_of(Height=tall,Height=very_short).",
    "C5 = same_house(Name=Arnold,Animal=cat).",
    "C6 = same_house(Animal=horse,Occupation=teacher).",
    "C7 = same_house(Name=Carol,Sport=soccer).",
    "C8 = same_house(Height=tall,Sport=volleyball).",
    "C9 = set(5,Occupation,lawyer).",
    "C10 = same_house(Sport=tennis,Occupation=teacher).",
    "C11 = same_house(Height=average,Sport=swimming).",
    "C12 = immediately_left_of(Sport=baseball,Occupation=engineer).",
    "C13 = same_house(Name=Peter,Occupation=nurse).",
    "C14 = right_of(Name=Bob,Occupation=artist).",
    "C15 = immediately_left_of(Occupation=teacher,Sport=soccer).",
    "C16 = same_house(Animal=rabbit,Name=Alice).",
    "C17 = same_house(Animal=fish,Name=Carol).",
    "C18 = set(1,Sport,baseball).",
    "C19 = right_of(Animal=cat,Height=very_short).",
    "C20 = set(5,Height,super_tall)."
  ],
  "parsed_reasoning": [
    "S1 [C18] set(1,Sport,baseball).",
    "S2 [C12+C18] set(2,Occupation,engineer).",
    "S3 [C1+S2] set(2,Animal,dog).",
    "S4 [C9] set(5,Occupation,lawyer).",
    "S5 [C20] set(5,Height,super_tall).",
    "S6 [C16+C3+C11+C2+C8+C4+C19+C5+C15+C7+C17+C9+C20] set(3,Name,Alice).",
    "S7 [C16] set(3,Animal,rabbit).",
    "S8 [C3+S7] set(2,Height,average).",
    "S9 [C11+S8] set(2,Sport,swimming).",
    "S10 [C8+C4+C19+C5+C15+C7+C17+C9+C20+C2] set(3,Sport,volleyball).",
    "S11 [C8+S10] set(3,Height,tall).",
    "S12 [C14+C6+C10+C15+C7+C17+C9+C20+C5+C19+C4] set(4,Name,Bob).",
    "S13 [C15+C7+C17+C9+C20] set(4,Occupation,teacher).",
    "S14 [C6+S13] set(4,Animal,horse).",
    "S15 [C10+S13] set(4,Sport,tennis).",
    "S16 [C4+S11] set(4,Height,very_short).",
    "S17 [C7+C17+C9+C20+C15] set(5,Name,Carol).",
    "S18 [C17+S17] set(5,Animal,fish).",
    "S19 [C7+S17] set(5,Sport,soccer).",
    "S20 [C5+C19+C4+C8+C2+C3+C15+C7+C17+C9+C20] set(6,Name,Arnold).",
    "S21 [C5+S20] set(6,Animal,cat).",
    "S22 [C8+C4+C19+C5+C7+C10+C6+C12+C18+C11] set(6,Sport,basketball).",
    "S23 [C2+S8] set(6,Height,short).",
    "S24 [C1+C6+C9+C10+C13+C14+C15+C16+C17+C18+C20] set(6,Occupation,doctor).",
    "S25 [C13+C1+C6+C9+C10+C14+C15+C16+C17+C18+C20] set(1,Name,Peter).",
    "S26 [C13+S25] set(1,Occupation,nurse).",
    "S27 [C1+C6+C9+C10+C16+C17+C18+C20] set(1,Animal,bird).",
    "S28 [C2+C4+C8+C20+C19] set(1,Height,very_tall).",
    "S29 [C1+S2] set(2,Name,Eric).",
    "S30 [C14+C16+C6+C10+C9+C20] set(3,Occupation,artist)."
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


SOLUTION_PROMPT_1_SHOT_SYS = """
You are an expert logic puzzle solver.

You will be given ONE logic puzzle in plain English.

Your task is to:
  - Extract the domain (N houses + all attribute values).
  - Parse each clue into a canonical, machine-checkable form (DSL).
  - Perform step-by-step deductions using only the allowed step format.
  - Derive a complete final solution.
  - Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
  - Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
  - Do NOT include extra text, markdown, explanations, or code fences.
  - Inside <answer>...</answer>, the content MUST be a single valid JSON object.
  - The JSON object MUST have exactly FIVE top-level keys:
      "n_houses", "attribute_values", "parsed_clues", "parsed_reasoning", "solution".
  - Do NOT add any other keys.

NORMALIZATION RULES:
  - Use underscores instead of spaces in VALUES (e.g., grilled_cheese, root_beer, very_short).
  - Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Animal, Occupation, Sport, Height, Color, Drink, etc.
  - House numbers are integers 1..N.
  - Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6.
  - Do NOT invent values. Every <Val> must be one of the allowed values listed in the puzzle text (after normalization).
  - If the clue mentions a bare person name (e.g., "Arnold"), treat it as Name=Arnold.
  - If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching attribute/value from the puzzle text
    (e.g., Animal=cat or Pet=cat; Drink=coffee), choosing the closest listed value.

D) DOMAIN OUTPUT (MANDATORY)
  - "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
  - "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values from the puzzle text.
  - Each attribute list MUST contain exactly N unique values (after normalization).
  - Include every attribute listed in the puzzle text, and only those attributes.
  - Do NOT infer extra attributes that are not explicitly listed in the puzzle text.

A) parsed_clues (MANDATORY, PARSABLE)
  - "parsed_clues" MUST be a list of strings.
  - Each string must be exactly 1 sentence and end with a period.
  - There MUST be exactly one entry per clue, in the same order as the clues.
  - Each parsed clue MUST follow this exact DSL format:

    C<i> = <predicate>.

Allowed predicates (use only these):
  - set(<H>,<Attr>,<Val>)
  - not_set(<H>,<Attr>,<Val>)
  - same_house(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - not_same_house(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - right_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - immediately_left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - immediately_right_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - adjacent(<AttrA>=<ValA>,<AttrB>=<ValB>)
  - between(<AttrA>=<ValA>,<AttrB>=<ValB>,<K>)

Semantics:
  - immediately_left_of(A,B): A is exactly 1 house left of B.
  - immediately_right_of(A,B): A is exactly 1 house right of B.
  - left_of(A,B): A is somewhere left of B (strictly smaller house index).
  - right_of(A,B): A is somewhere right of B (strictly larger house index).
  - adjacent(A,B): houses differ by exactly 1.
  - between(A,B,K): there are exactly K houses strictly between A and B.
    (So K=1 => positions differ by 2, and K=2 => positions differ by 3.)

IMPORTANT PARSE MAPPINGS (MUST FOLLOW):
  - "X is the person whose A is v" => same_house(Name=X, A=v).
    Example: "Arnold is the person whose favorite color is red." =>
             same_house(Name=Arnold, Color=red).
  - "The A=v person is in house H" => set(H, A, v).
  - "X is not in house H" => not_set(H, Name, X).
  - "X is not the A=v person" => not_same_house(Name=X, A=v).
  - "A=v is directly left of B=w" => immediately_left_of(A=v, B=w).
  - "A=v is directly right of B=w" => immediately_right_of(A=v, B=w).
  - "A=v is somewhere left of B=w" => left_of(A=v, B=w).
  - "A=v is somewhere right of B=w" => right_of(A=v, B=w).
  - "A=v is next to B=w" => adjacent(A=v, B=w).
  - "Exactly K houses between A=v and B=w" => between(A=v, B=w, K).

B) parsed_reasoning (MANDATORY, PARSABLE)
  - "parsed_reasoning" MUST be a list of strings.
  - Each string must be exactly 1 sentence and end with a period.
  - There is NO LIMIT on the number of entries.
  - Each entry MUST follow this exact DSL format:

    S<k> [C<i>(+C<j>...)] <op>(<H>,<Attr>,<Val>).

Where:
  - <k> is a step number starting at 1 and increasing by 1 each step.
  - Evidence inside [...] must reference clue ids, e.g. [C1] or [C1+C3].
  - <op> is either set or not_set (use only these two in reasoning steps).
  - <H> is a house number integer (1..N).
  - <Attr> and <Val> must come from the puzzle text (normalized).

LOGICAL VALIDITY REQUIREMENT:
  - Every step in "parsed_reasoning" MUST be logically entailed by the parsed clues plus any earlier reasoning steps.
  - Prefer producing many safe not_set(...) deductions if uncertain.
  - Do NOT output a step that is merely a restatement of a clue unless it is in set/not_set form and is useful for deduction chains.

C) solution (MANDATORY TABLE)
  - "solution" MUST be in tabular form with:
    - "header": a list of column names
    - "rows": a list of rows, each row being a list of strings matching the header order.
  - The header MUST include "House" and then all attribute columns from the puzzle text.
  - The rows MUST list houses in increasing order from 1..N.
  - All solution VALUES must be normalized with underscores.

================================================================================
ONE-SHOT EXAMPLE (20 clues) — Demonstration of the REQUIRED JSON OUTPUT FORMAT
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
  "parsed_clues": [
    "C1 = same_house(Occupation=engineer,Animal=dog).",
    "C2 = left_of(Height=average,Height=short).",
    "C3 = immediately_left_of(Height=average,Animal=rabbit).",
    "C4 = left_of(Height=tall,Height=very_short).",
    "C5 = same_house(Name=Arnold,Animal=cat).",
    "C6 = same_house(Animal=horse,Occupation=teacher).",
    "C7 = same_house(Name=Carol,Sport=soccer).",
    "C8 = same_house(Height=tall,Sport=volleyball).",
    "C9 = set(5,Occupation,lawyer).",
    "C10 = same_house(Sport=tennis,Occupation=teacher).",
    "C11 = same_house(Height=average,Sport=swimming).",
    "C12 = immediately_left_of(Sport=baseball,Occupation=engineer).",
    "C13 = same_house(Name=Peter,Occupation=nurse).",
    "C14 = right_of(Name=Bob,Occupation=artist).",
    "C15 = immediately_left_of(Occupation=teacher,Sport=soccer).",
    "C16 = same_house(Animal=rabbit,Name=Alice).",
    "C17 = same_house(Animal=fish,Name=Carol).",
    "C18 = set(1,Sport,baseball).",
    "C19 = right_of(Animal=cat,Height=very_short).",
    "C20 = set(5,Height,super_tall)."
  ],
  "parsed_reasoning": [
    "S1 [C18] set(1,Sport,baseball).",
    "S2 [C12+C18] set(2,Occupation,engineer).",
    "S3 [C1+S2] set(2,Animal,dog).",
    "S4 [C9] set(5,Occupation,lawyer).",
    "S5 [C20] set(5,Height,super_tall).",
    "S6 [C16] not_set(1,Animal,rabbit).",
    "S7 [C16] not_set(2,Animal,rabbit).",
    "S8 [C16] not_set(4,Animal,rabbit).",
    "S9 [C16] not_set(5,Animal,rabbit).",
    "S10 [C16] not_set(6,Animal,rabbit).",
    "S11 [C16] set(3,Animal,rabbit).",
    "S12 [C16] set(3,Name,Alice).",
    "S13 [C3+S11] set(2,Height,average).",
    "S14 [C11+S13] set(2,Sport,swimming).",
    "S15 [C15+C7] set(4,Occupation,teacher).",
    "S16 [C15+C7] set(5,Sport,soccer).",
    "S17 [C7] set(5,Name,Carol).",
    "S18 [C17+S17] set(5,Animal,fish).",
    "S19 [C10+S15] set(4,Sport,tennis).",
    "S20 [C6+S15] set(4,Animal,horse).",
    "S21 [C14+C9] set(3,Occupation,artist).",
    "S22 [C14+S21] set(4,Name,Bob).",
    "S23 [C8+C4+C19+C5] set(4,Height,very_short).",
    "S24 [C19+C5] set(6,Animal,cat).",
    "S25 [C5+C24] set(6,Name,Arnold).",
    "S26 [C13] set(1,Name,Peter).",
    "S27 [C13+S26] set(1,Occupation,nurse).",
    "S28 [C2+C3+C4+C8+C19] set(6,Height,short).",
    "S29 [C8] set(3,Sport,volleyball).",
    "S30 [C8+S29] set(3,Height,tall).",
    "S31 [C1+S2] set(2,Name,Eric).",
    "S32 [C2] set(1,Height,very_tall).",
    "S33 [C4] set(1,Sport,basketball).",
    "S34 [C1] set(6,Occupation,doctor).",
    "S35 [C2] set(1,Animal,bird)."
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

END OF ONE-SHOT EXAMPLE.
"""



SOLUTION_PROMPT_1_SHOT_USER = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

{puzzle}

Solve the puzzle above and provide n_houses, attribute_values, parsed_clues, parsed_reasoning and solution for this puzzle in the <answer> </answer> block, with no additional text.
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



def make_map_fn_1_shot(split, data_source):
    def process_fn_1_shot(example, idx):
        # Use 'ground_truth' instead of 'solution' since that's what the input data has
        final_grid = example['solution']
        # Use the 'clues' field directly from the input data
        clues = extract_clues_from_puzzle(puzzle_text=example['puzzle'])
        # user_prompt = SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=example['puzzle'])
        user_prompt = SOLUTION_PROMPT_1_SHOT_SYS + SOLUTION_PROMPT_1_SHOT_USER.format(puzzle=example['puzzle'])

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
    parser.add_argument('--data_setting', default=None, help='Path to json file')
    parser.add_argument('--output_dir', default=None, help='Directory to save processed data')
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
