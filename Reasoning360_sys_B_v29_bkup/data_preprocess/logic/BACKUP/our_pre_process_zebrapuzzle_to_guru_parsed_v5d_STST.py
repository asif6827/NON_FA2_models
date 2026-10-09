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
3) reasoning (MANDATORY — INTERLEAVED NATURAL + SYNTACTIC)
================================================================================
- "reasoning" MUST be a list of strings.
- Each entry MUST be exactly 1 sentence and end with a period.
- Reasoning MUST be interleaved:
    Odd-numbered entries: Natural-language reasoning.
    Even-numbered entries: Syntactic reasoning step (Z3-like statement).
- Natural-language entries should explain the deduction in plain English.
- Syntactic entries should encode the *newly deduced fact* as a Z3-like statement.
- Tokens in Syntactic entries should encode the *mapped* to values in "attribute_values".

Syntactic reasoning step format:
  S<k>: <constraint>.

Where:
- <k> starts at 1 and increments by 1 for each syntactic step only (S1, S2, S3...).
- <constraint> must follow the same operator rules as syntactic_clues (==, !=, <, >, + 1 ==, == H).
- Each syntactic step MUST cite its evidence at the end in square brackets using clue ids and/or prior steps:
  Example:
    "S1: engineer == 2. [C12+C18]"
    "S2: dog == 2. [C1+S1]"

Evidence rules:
- Evidence MUST be included for EVERY syntactic step.
- Evidence may reference: C<i> and S<k> only.
- Use "+" to join multiple evidence references, e.g., [C3+C11+S2].

Logical validity requirement:
- Every syntactic step MUST be logically entailed by the syntactic_clues plus any earlier syntactic steps.
- Do NOT output syntactic steps that merely restate a clue unless they are required as part of the deduction chain.

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
   "The baseball fan is fixed in the first house by clue 18, giving us a strong anchor to begin.",
    "S1: baseball == 1. [C18]",
    "Because baseball is directly left of the engineer and baseball is in house 1, the engineer must be in house 2.",
    "S2: engineer == 2. [C12+S1]",
    "Since the engineer is the dog owner, the dog must also be placed in house 2.",
    "S3: dog == 2. [C1+S2]",
    "Clue 9 directly places the lawyer in the fifth house.",
    "S4: lawyer == 5. [C9]",
    "Clue 20 fixes the super tall person in the fifth house, strengthening the profile of house 5.",
    "S5: super_tall == 5. [C20]",
    "Using the teacher–soccer relation together with Carol loving soccer, the only consistent placement is soccer in house 5.",
    "S6: soccer == 5. [C15+C7+S4]",
    "Because the teacher is immediately left of the soccer fan, the teacher must be in house 4.",
    "S7: teacher == 4. [C15+S6]",
    "Since Carol is the soccer fan and soccer is in house 5, Carol must live in house 5.",
    "S8: Carol == 5. [C7+S6]",
    "Because the fish enthusiast is Carol, the fish must also be in house 5.",
    "S9: fish == 5. [C17+S8]",
    "Since the horse owner is the teacher and the teacher is in house 4, the horse must be in house 4.",
    "S10: horse == 4. [C6+S7]",
    "Because the tennis fan is the teacher, tennis must also be in house 4.",
    "S11: tennis == 4. [C10+S7]",
    "Using the average–rabbit adjacency and eliminating all other houses due to existing animal placements, average must be in house 2.",
    "S12: average == 2. [C3+S3+S9+S10]",
    "Since average is in house 2 and is directly left of the rabbit owner, the rabbit must be in house 3.",
    "S13: rabbit == 3. [C3+S12]",
    "Because average height corresponds to swimming, swimming must also be in house 2.",
    "S14: swimming == 2. [C11+S12]",
    "Since the rabbit owner is Alice and the rabbit is in house 3, Alice must be in house 3.",
    "S15: Alice == 3. [C16+S13]",
    "Placing volleyball in house 6 would violate the tall–very short ordering, so volleyball must be in house 3.",
    "S16: volleyball == 3. [C8+C4+S5+S15]",
    "Because tall is equivalent to volleyball, tall must also be in house 3.",
    "S17: tall == 3. [C8+S16]",
    "With all other sports placed, the only remaining house for basketball is house 6.",
    "S18: basketball == 6. [S6+S11+S14+S16]",
    "To allow Bob to be to the right of the artist and with Alice fixed in house 3, the artist must be in house 3.",
    "S19: artist == 3. [C14+S15]",
    "With artist fixed in house 3, the only remaining occupation for house 6 is doctor.",
    "S20: doctor == 6. [S19]",
    "House 4 is already identified as the teacher’s house, and Bob is the only remaining person who can occupy it.",
    "S21: Bob == 4. [S7]",
    "Because Arnold is the cat lover and the cat must be to the right of the very short person, Arnold must be in house 6.",
    "S22: Arnold == 6. [C5+C19+S10+S13]",
    "Since Peter is the nurse and house 2 is already occupied by the engineer, Peter must be in house 1.",
    "S23: Peter == 1. [C13+S2]",
    "With Peter in house 1 and Arnold in house 6, the remaining person Eric must be in house 2.",
    "S24: Eric == 2. [S23]",
    "All other animals are assigned, so the remaining animal bird must belong to house 1.",
    "S25: bird == 1. [S3+S9+S10+S13+S22]",
    "Because very short must be to the right of tall and tall is in house 3, very short must be in house 4.",
    "S26: very_short == 4. [C4+S17]",
    "Since average is in house 2 and must be left of short, the only feasible placement for short is house 6.",
    "S27: short == 6. [C2+S12]",
    "The only remaining height value, very_tall, goes to house 1.",
    "S28: very_tall == 1. [S5+S12+S17+S26+S27]",    
    "At this stage every attribute is assigned consistently, yielding the final table:",
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
