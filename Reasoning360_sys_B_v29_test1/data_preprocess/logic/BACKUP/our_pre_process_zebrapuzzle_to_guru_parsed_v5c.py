import os
import sys
import datasets
import argparse
from sklearn.model_selection import train_test_split

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset

SOLUTION_PROMPT_1_SHOT_SYS = """
You are an expert logic puzzle solver.

You are given (i) one logic puzzle written in plain English, (ii) solution header that lists the attribute names used in the puzzle, and (iii) a dictionary of attribute values specifying the allowed values for each attribute.


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
- Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Animal, Occupation, Sport, Height, etc.
- House numbers are integers 1..N.
- Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6, etc.
- Do NOT invent values. Every value MUST be one of the allowed values listed in the puzzle text (after normalization).
- If the clue mentions a bare person name (e.g., "Arnold"), treat it as Name=Arnold.
- If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching attribute/value from the puzzle text.

================================================================================
1) DOMAIN OUTPUT (MANDATORY)
================================================================================
- "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
- "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values from the puzzle text.
- Each attribute list MUST contain exactly N unique values (after normalization).
- Include every attribute listed in the puzzle text, and only those attributes.
- Do NOT infer extra attributes that are not explicitly listed in the puzzle text.

================================================================================
2) syntactic_clues (MANDATORY, TEXTUAL CONSTRAINTS — NOT PREDICATES)
================================================================================
We do NOT use predicate-style DSL for clues.
Instead, each clue MUST be rewritten as a single-line *syntactic constraint statement* in a Z3-like textual form.

Rules:
- "syntactic_clues" MUST be a list of strings.
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
- When a clue states "two houses between", encode as: A + 3 == B
  Example: "There are two houses between Eric and Arnold" -> "C12: Eric + 3 == Arnold."
- When a clue states "person who has", encode as: A == B
  Example: "The person whose mother's name is Holly is the person who has a dog" -> "C12: Holly == dog."
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
- The header MUST include "House" and then all attribute columns from the puzzle text.
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
    "We start by turning the clearest absolute clue into a Z3 anchor. Clue 18 says the baseball fan is in the first house.",
    "S1:baseball == 1",
    "Clue 12 says baseball is directly left of the engineer, so the engineer must be in house 2.",
    "S2:baseball + 1 == engineer",
    "S3:engineer == 2",
    "Clue 1 equates engineer with the dog owner, so the dog is also placed in house 2.",
    "S4:engineer == dog",
    "S5:dog == 2",
    "Next we use the teacher/soccer/Carol chain. Clue 15 says the teacher is directly left of the soccer fan, and clue 7 says Carol loves soccer.",
    "S6:teacher + 1 == soccer",
    "S7:Carol == soccer",
    "Clue 9 fixes the lawyer in house 5. Since each house has exactly one sport, if soccer were not in house 5 then Carol would not be able to align cleanly with other fixed facts later. Clue 17 also says the fish enthusiast is Carol, so Carol’s house must be consistent with a single animal assignment.",
    "Clue 20 fixes super_tall in house 5, giving us a strong profile for house 5. With clue 9, house 5 is both lawyer and super_tall.",
    "S8:lawyer == 5",
    "S9:super_tall == 5",
    "Now combine clue 15 (teacher immediately left of soccer) with the fact that house 5 is already strongly identified; the clean placement is soccer in house 5, which forces teacher in house 4.",
    "S10:soccer == 5",
    "S11:teacher == 4",
    "Then clue 7 places Carol in the soccer house, so Carol is in house 5.",
    "S12:Carol == 5",
    "Clue 17 says fish == Carol, so fish is also in house 5.",
    "S13:fish == Carol",
    "S14:fish == 5",
    "Clue 6 says horse == teacher, so horse is in house 4.",
    "S15:horse == teacher",
    "S16:horse == 4",
    "Clue 10 says tennis == teacher, so tennis is also in house 4.",
    "S17:tennis == teacher",
    "S18:tennis == 4",
    "At this point, several houses are partially pinned: house 1 has baseball; house 2 has engineer and dog; house 4 has teacher, horse, and tennis; house 5 has lawyer, super_tall, Carol, and fish.",
    "We now use the ‘average next to rabbit’ structure. Clue 3 says average is directly left of rabbit.",
    "S19:average + 1 == rabbit",
    "Because house 2 already has the dog (animal is fixed there), rabbit cannot be in house 2. That rules out average being in house 1.",
    "S20:Not(average == 1)",
    "Also, rabbit cannot be in house 5 because house 5 already has fish (clue 17). That rules out average being in house 4.",
    "S21:rabbit != 5",
    "S22:Not(average == 4)",
    "And rabbit cannot be in house 4 because house 4 already has horse (clue 6). That rules out average being in house 3.",
    "S23:rabbit != 4",
    "S24:Not(average == 3)",
    "If average were in house 5, rabbit would be in house 6. Then clue 16 (rabbit == Alice) would force Alice into house 6, and clue 14 (Bob is to the right of the artist) would become impossible because the artist would have to be left of Bob but only house 6 would remain for Alice. So average cannot be 5.",
    "S25:Not(average == 5)",
    "The only remaining possibility is average in house 2, which forces rabbit in house 3.",
    "S26:average == 2",
    "S27:rabbit == 3",
    "Clue 11 says average == swimming, so swimming is also in house 2.",
    "S28:average == swimming",
    "S29:swimming == 2",
    "Clue 16 says the rabbit owner is Alice, so Alice must be in house 3.",
    "S30:rabbit == Alice",
    "S31:Alice == 3",
    "Now use clue 8: tall == volleyball. Since volleyball is not yet placed, we pair this with the remaining sports after placing baseball (house 1), swimming (house 2), tennis (house 4), and soccer (house 5). The only sports left are volleyball and basketball for houses 3 and 6.",
    "S32:Distinct(volleyball, basketball, baseball, tennis, soccer, swimming)",
    "S33:Or(volleyball == 3, volleyball == 6)",
    "Clue 8 links tall to volleyball, and clue 4 says tall is somewhere left of very_short. If volleyball were in house 6 then tall would be in house 6, which is impossible because nothing is to the right for very_short. Therefore volleyball (and tall) must be in house 3.",
    "S34:tall == volleyball",
    "S35:tall < very_short",
    "S36:volleyball == 3",
    "S37:tall == 3",
    "So basketball must be the remaining sport in house 6.",
    "S38:basketball == 6",
    "We now lock down the remaining occupations. We already have engineer (2), teacher (4), lawyer (5), and nurse from clue 13 belongs to Peter.",
    "S39:Peter == nurse",
    "The remaining occupations to place are doctor and artist in houses 3 and 6 (since houses 1,2,4,5 are already taken).",
    "Clue 14 says Bob is somewhere to the right of the artist. Since Alice is in house 3, the artist cannot be in house 6 (there would be no place for Bob to the right). So the artist must be in house 3.",
    "S40:Bob > artist",
    "S41:artist == 3",
    "That forces doctor in house 6.",
    "S42:doctor == 6",
    "With the artist in house 3 and Bob to the right of the artist, the only remaining unassigned name that cleanly fits the fixed teacher house 4 is Bob. So Bob is in house 4.",
    "S43:Bob == 4",
    "Now the remaining names for houses 1,2,6 are Peter, Eric, and Arnold, but clue 5 tells us Arnold is the cat lover.",
    "S44:Arnold == cat",
    "Since house 2 has the dog and house 5 has the fish and house 3 has the rabbit and house 4 has the horse, the only houses where cat can go are 1 or 6. Clue 19 says cat is somewhere to the right of very_short, so cat cannot be in house 1. Therefore cat (and Arnold) must be in house 6.",
    "S45:cat > very_short",
    "S46:cat == 6",
    "S47:Arnold == 6",
    "That leaves Peter and Eric for houses 1 and 2. But clue 13 says Peter is the nurse; house 2 is the engineer, so Peter cannot be in house 2. Therefore Peter is in house 1, and Eric is in house 2.",
    "S48:Peter == 1",
    "S49:Eric == 2",
    "We now finish animals: house 1 cannot be dog (house 2), rabbit (house 3), horse (house 4), fish (house 5), or cat (house 6). So the remaining animal bird must be in house 1.",
    "S50:bird == 1",
    "Finally, we complete heights using the remaining constraints. We already have average in house 2 and tall in house 3 and super_tall in house 5.",
    "S51:average == 2",
    "S52:tall == 3",
    "S53:super_tall == 5",
    "Clue 2 says average is to the left of short, so short must be in house 6 (since houses 4 and 5 already have fixed profiles and house 3 is tall).",
    "S54:average < short",
    "S55:short == 6",
    "Clue 4 says tall is left of very_short, and since tall is in house 3, very_short must be in house 4.",
    "S56:very_short == 4",
    "The only remaining height value, very_tall, goes to house 1.",
    "S57:very_tall == 1",
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

{puzzle}

solution_header = {solution_header}

attribute_values = {attribute_values}

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
            if v not in seen[col]:
                seen[col].add(v)
                values[col].append(v)

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
