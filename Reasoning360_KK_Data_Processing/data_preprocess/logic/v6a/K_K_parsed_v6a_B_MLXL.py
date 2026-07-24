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
You are an expert solver of Knights and Knaves logic puzzles.

You are given:

(i) a Knights and Knaves puzzle written in natural language, and
(ii) a list named people containing the complete and exclusive set of people
     appearing in the puzzle.

Every person is exactly one of the following:

- Knight: always makes a true statement.
- Knave: always makes a false statement.

For each person P, use a Boolean variable with the same normalized name:

- P = True means that P is a Knight.
- P = False means that P is a Knave.

The formal expressions generated in this task use a restricted textual logic
language. The equality operator "=" in this language must be interpreted as
Boolean equality and translated internally to the Z3 operator "==".

If person P utters a statement represented by the Boolean formula phi_P, the
corresponding clue constraint is:

P = phi_P

This equivalence captures both cases:

- If P is a Knight, phi_P must be true.
- If P is a Knave, phi_P must be false.

Your task is to construct a fully consistent and solver-verifiable solution by
generating exactly the following FIVE fields:

1) n_people
   The total number of people appearing in the puzzle.

2) people
   The complete list of people, returned exactly as supplied.

3) syntactic_clues
   A normalized solver-oriented encoding of every statement in the puzzle.

4) reasoning
   Interleaved reasoning containing alternating natural-language explanations
   and formal solver-checkable deductions.

5) solution
   The final Knight-or-Knave classification for every person.

You MUST return the result strictly as a single valid JSON object enclosed in:

<answer>...</answer>

No text, explanation, markdown, or formatting is permitted outside the
<answer> block.


================================================================================
CRITICAL OUTPUT REQUIREMENTS
================================================================================

- Output exactly ONE <answer>...</answer> block.
- Do not output markdown code fences.
- The content inside <answer>...</answer> must be one valid JSON object.
- The JSON object must contain exactly these five top-level keys:

    "n_people",
    "people",
    "syntactic_clues",
    "reasoning",
    "solution"

- Do not add, remove, or rename any top-level key.


================================================================================
NORMALIZATION RULES
================================================================================

- Person names must be drawn exclusively from the supplied people list.
- The "people" list must be returned exactly as supplied:
  - same names,
  - same spelling,
  - same casing,
  - same ordering.
- In syntactic_clues and reasoning, replace spaces and punctuation inside
  multi-word names with underscores when necessary.
- Do not invent additional people.
- Do not place quotation marks around person variables in formal expressions.
- Use True and False exactly with this capitalization.
- A person variable evaluates to True when that person is a Knight and False
  when that person is a Knave.
- Do not use a bare person variable as a complete formal statement.
- Always express a person's final identity explicitly:
    P = True
    P = False


================================================================================
1) DOMAIN OUTPUT
================================================================================

- "n_people" must be an integer equal to the number of supplied people.
- "people" must be returned exactly as supplied.
- Every person appearing in syntactic_clues, reasoning, and solution must occur
  in the supplied people list.


================================================================================
2) SYNTACTIC CLUES
================================================================================

"syntactic_clues" must be a list of strings.

Each entry must:

- encode exactly one person's complete spoken statement;
- occur in the same order as the statements in the puzzle;
- begin with "C<i>: ";
- contain exactly one line;
- end with a period.

For a speaker P and the logical content phi_P of P's statement, encode the clue
as:

P = phi_P

The speaker's truthfulness must always be linked to the truth value of the
statement.

Example:

Alice says, "Bob is a Knight."

Correct:

C1: Alice = Bob.

Incorrect:

C1: Bob = True.

The correct form expresses that Alice is a Knight exactly when Bob is a Knight.
The incorrect form encodes only the content of Alice's statement and omits the
truthfulness condition associated with Alice.


================================================================================
PERMITTED FORMAL EXPRESSIONS
================================================================================

Use only the following restricted syntax.

Atomic identity assignments:

    P = True
    P = False

Identity relations:

    P = Q
    P != Q

Boolean operators:

    Not(e)
    And(e1, e2, ..., en)
    Or(e1, e2, ..., en)
    Xor(e1, e2)
    Implies(e1, e2)

Boolean equivalence:

    e1 = e2

All operands must be valid Boolean expressions.

Nested Boolean expressions are permitted.

Examples:

    Alice = True
    Bob = False
    Alice = Bob
    Alice != Bob
    Not(Alice = True)
    And(Alice = True, Bob = False)
    Or(Alice = True, Bob = True)
    Xor(Alice = True, Bob = True)
    Implies(Alice = True, Bob = False)
    Charlie = And(Alice = True, Bob = False)

Do not use:

- bare variables as complete statements, such as "Alice";
- Python assignment statements;
- Python control flow;
- quantifiers;
- arithmetic operators;
- house indices;
- list membership;
- set notation;
- natural-language predicates;
- s.add(...);
- Solver(...);
- model evaluation syntax.


================================================================================
INTERNAL Z3 TRANSLATION
================================================================================

The generated formal expressions use "=" as Boolean equality.

Before sending an expression to Z3, translate:

    P = True        -> P == True
    P = False       -> P == False
    P = Q           -> P == Q
    e1 = e2         -> e1 == e2

The operator "!=" remains unchanged.

The Boolean operators map directly to Z3:

    Not(e)          -> Not(e)
    And(...)        -> And(...)
    Or(...)         -> Or(...)
    Xor(e1, e2)     -> Xor(e1, e2)
    Implies(e1, e2) -> Implies(e1, e2)


================================================================================
TRANSLATION RULES FOR COMMON STATEMENTS
================================================================================

1. "P is a Knight"
    P = True

2. "P is a Knave"
    P = False

3. "P and Q are both Knights"
    And(P = True, Q = True)

4. "P and Q are both Knaves"
    And(P = False, Q = False)

5. "P and Q are of the same type"
    P = Q

6. "P and Q are of different types"
    P != Q

7. "At least one of P and Q is a Knight"
    Or(P = True, Q = True)

8. "At least one of P and Q is a Knave"
    Or(P = False, Q = False)

9. "Exactly one of P and Q is a Knight"
    Xor(P = True, Q = True)

10. "Exactly one of P and Q is a Knave"
    Xor(P = False, Q = False)

11. "Neither P nor Q is a Knight"
    And(P = False, Q = False)

12. "Neither P nor Q is a Knave"
    And(P = True, Q = True)

13. "P is a Knight if and only if Q is a Knight"
    P = Q

14. "If P is a Knight, then Q is a Knight"
    Implies(P = True, Q = True)

15. "If P is a Knight, then Q is a Knave"
    Implies(P = True, Q = False)


16. "P is a Knight or Q is a Knave"
    Or(P = True, Q = False)

17. "P is not a Knight"
    P = False

18. "P is not a Knave"
    P = True

19. "P and Q have opposite identities"
    P != Q

20. "Either P or Q is a Knight, but not both"
    Xor(P = True, Q = True)


================================================================================
SELF-REFERENTIAL STATEMENTS
================================================================================

If Alice says, "I am a Knight," the spoken content is:
    Alice = True

The full clue constraint is therefore:
    Alice = (Alice = True)

This clue is tautological and does not determine Alice's identity by itself.


If Alice says, "I am a Knave," the spoken content is:
    Alice = False

The full clue constraint is therefore:
    Alice = (Alice = False)

This constraint is unsatisfiable in a standard Knights and Knaves setting.

If Alice says, "Bob and I are of the same type," encode:
    Alice = (Alice = Bob)


If Alice says, "Bob and I are of different types," encode:
    Alice = (Alice != Bob)


================================================================================
EXAMPLES OF COMPLETE CLUE ENCODING
================================================================================

Alice says, "Bob is a Knight."
    C1: Alice = (Bob = True).

The shorter equivalent form below is also permitted:
    C1: Alice = Bob.


Bob says, "Alice is a Knave."
    C2: Bob = (Alice = False).


Charlie says, "Alice and Bob are of the same type."
    C3: Charlie = (Alice = Bob).


David says, "Exactly one of Alice and Charlie is a Knight."
    C4: David = Xor(Alice = True, Charlie = True).


Eve says, "If Alice is a Knight, then Bob is a Knave."
    C5: Eve = Implies(Alice = True, Bob = False).


Frank says, "Alice is a Knight and Bob is a Knave."
    C6: Frank = And(Alice = True, Bob = False).


Grace says, "Alice or Bob is a Knight."
    C7: Grace = Or(Alice = True, Bob = True).


================================================================================
3) REASONING
================================================================================

"reasoning" must be a list of strings.

The entries must alternate strictly:

- Odd-numbered entries:
  one natural-language explanation.

- Even-numbered entries:
  one formal solver-checkable deduction.

The required structure is:

    natural-language explanation,
    S1 formal deduction,
    natural-language explanation,
    S2 formal deduction,
    ...

Every natural-language entry must:

- contain exactly one sentence;
- explain the corresponding deduction;
- end with a period.

Every formal reasoning entry must:

- contain exactly one solver-checkable Boolean expression;
- begin with "S<k>: ";
- use consecutive identifiers S1, S2, S3, ...;
- end with a period;
- use only the permitted formal operators;
- express individual identities explicitly as P = True or P = False.

Examples:

    "Alice and Bob must have the same identity because Alice claims that Bob is a Knight."
    "S1: Alice = Bob."

    "If Alice is a Knight, then Bob must also be a Knight."
    "S2: Implies(Alice = True, Bob = True)."

    "Bob cannot be a Knight because that assignment would contradict his statement."
    "S3: Bob = False."

    "Since Bob is a Knave and Alice has the same identity as Bob, Alice must also be a Knave."
    "S4: Alice = False."

    "Charlie must be a Knight."
    "S5: Charlie = True."

    "Alice and Charlie must have different identities."
    "S6: Alice != Charlie."

    "At least one of Alice and Bob must be a Knave."
    "S7: Or(Alice = False, Bob = False)."


================================================================================
LOGICAL REQUIREMENTS FOR REASONING STEPS
================================================================================

Each formal step S_k must satisfy all of the following:

1. Parseable
   It must follow the permitted formal grammar.

2. Valid
   It must be logically entailed by the complete puzzle theory consisting of
   all syntactic clues.

3. Satisfiable
   It must be jointly satisfiable with the complete puzzle theory.

4. Non-contradictory
   Adding the step to the puzzle theory must not make the theory unsatisfiable.

5. Preferably novel
   It should contribute information not already implied by the previous
   non-contradictory reasoning steps alone.

6. Non-repetitive
   It should not merely repeat an individual clue unless that clue is needed to
   establish a clear deduction chain.

7. Non-tautological
   Avoid formulas that are true independently of the puzzle clues, such as:

       Or(Alice = True, Alice = False)

8. Explicit
   Do not use a bare variable such as:
       S4: Alice.

   Instead write:
       S4: Alice = True.

   Similarly, do not write:
       S5: Not(Bob).

   Instead write:
       S5: Bob = False.


================================================================================
4) SOLUTION
================================================================================

"solution" must be a JSON object with exactly two keys:

    "header"
    "rows"

The header must be:

    ["Person", "Identity"]

Each row must contain:

    [person_name, identity]

where identity is exactly one of:

    "Knight"
    "Knave"

The rows must:

- include every supplied person exactly once;
- follow the same ordering as the supplied people list;
- contain no additional people;
- agree with all syntactic clues;
- agree with all formal reasoning steps;
- represent a complete satisfying assignment.


================================================================================
ONE-SHOT EXAMPLE
================================================================================

Example puzzle:

Three inhabitants—Alice, Bob, and Charlie—live on an island where every person
is either a Knight, who always tells the truth, or a Knave, who always lies.

Alice says, "Bob is a Knight."
Bob says, "Alice and Charlie are of different types."
Charlie says, "Bob is a Knave."

people = ["Alice", "Bob", "Charlie"]


Correct example output:

<answer>{
  "n_people": 3,
  "people": [
    "Alice",
    "Bob",
    "Charlie"
  ],
  "syntactic_clues": [
    "C1: Alice = Bob.",
    "C2: Bob = (Alice != Charlie).",
    "C3: Charlie = (Bob = False)."
  ],
  "reasoning": [
    "Alice's statement requires Alice and Bob to have the same identity.",
    "S1: Alice = Bob.",
    "Charlie's statement requires Charlie and Bob to have opposite identities.",
    "S2: Charlie != Bob.",
    "If Bob were a Knave, then Alice would also be a Knave and Charlie would be a Knight.",
    "S3: Implies(Bob = False, And(Alice = False, Charlie = True)).",
    "Under that assumption Alice and Charlie would have different identities, making Bob's statement true and contradicting Bob being a Knave.",
    "S4: Not(And(Bob = False, Alice != Charlie)).",
    "Therefore Bob must be a Knight.",
    "S5: Bob = True.",
    "Since Alice has the same identity as Bob, Alice must also be a Knight.",
    "S6: Alice = True.",
    "Since Charlie has the opposite identity from Bob, Charlie must be a Knave.",
    "S7: Charlie = False."
  ],
  "solution": {
    "header": [
      "Person",
      "Identity"
    ],
    "rows": [
      [
        "Alice",
        "Knight"
      ],
      [
        "Bob",
        "Knight"
      ],
      [
        "Charlie",
        "Knave"
      ]
    ]
  }
}</answer>
"""


SOLUTION_PROMPT_1_SHOT_USER = r"""
--------------------------------
KNIGHTS AND KNAVES PUZZLE
--------------------------------

puzzle = {puzzle}

people = {people}

Solve the puzzle and return exactly the following five fields:

1. n_people
2. people
3. syntactic_clues
4. reasoning
5. solution

Return the result as one valid JSON object inside a single
<answer>...</answer> block, with no additional text.
"""


from typing import Any


def solution_table_from_knights_and_knaves(
    names: list[str],
    labels: list[bool],
) -> dict[str, list]:
    """
    Convert the dataset's Boolean solution into the output table expected
    by the Knights-and-Knaves prompt.

    True  -> Knight
    False -> Knave
    """
    if len(names) != len(labels):
        raise ValueError(
            f"Number of names ({len(names)}) does not match "
            f"number of labels ({len(labels)})."
        )

    rows = [
        [name, "Knight" if label else "Knave"]
        for name, label in zip(names, labels)
    ]

    return {
        "header": ["Person", "Identity"],
        "rows": rows,
    }


def make_map_fn_1_shot(split: str, data_source: str):
    def process_fn_1_shot(
        example: dict[str, Any],
        idx: int,
    ) -> dict[str, Any]:

        # Dataset fields:
        #   quiz: puzzle text
        #   names: ordered list of people
        #   solution: ordered Boolean labels
        puzzle = example["quiz"]
        people = example["names"]
        labels = example["solution"]

        final_solution = solution_table_from_knights_and_knaves(
            names=people,
            labels=labels,
        )

        user_prompt = (
            SOLUTION_PROMPT_1_SHOT_SYS
            + SOLUTION_PROMPT_1_SHOT_USER.format(
                puzzle=puzzle,
                people=people,
            )
        )

        example_id = example.get(
            "id",
            example.get("index", str(idx)),
        )

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            "raw_prompt": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            "ability": "logical_reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": final_solution,
            },
            "apply_chat_template": False,
            "extra_info": {
                "id": example_id,
                "split": split,
                "people": people,
                "boolean_solution": labels,
                "solution_text": example.get("solution_text"),
                "n_people": len(people),
            },
        }
        print("Within Loop")
        # Usually print only the first example, not every example except zero.
        #if idx == 0:
        print(
            f"data_source: {data_source}, "
            f"split: {split}, idx: {idx}"
        )
        print(
            "\n"
            + "=" * 100
            + f"\n{data_source} | {split} | {idx}\n"
            + "=" * 100
        )
        print(data)
        print("\n")

        return data

    return process_fn_1_shot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/asif/data3/HF_cache/knights_and_knaves_300_train/', help='Path to json file')
    parser.add_argument('--data_setting', default='mlxl_train_mlxl_test', help='Path to json file')
    parser.add_argument('--output_dir', default='/home/asif/data3/HF_cache/K_K_to_guru/', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--data_source_train', default='our_k_k_new_reward', help='Name of data source')
    parser.add_argument('--data_source_test', default='our_k_k_new_reward_test', help='Name of data source')
    parser.add_argument('--train_sample_size', type=int, default=None, help='Number of samples to use from train. If None, use all.')
    args = parser.parse_args()

    if args.data_setting == 'mlxl_train_mlxl_test':
        args.train_data_file = os.path.join(args.data_path, 'train_all_num300.jsonl')
        args.test_data_file = os.path.join(args.data_path, 'test_all_num700.jsonl')
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
