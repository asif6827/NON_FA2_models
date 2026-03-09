
SOLUTION_PROMPT_VERIFIER_V2="""
You are an expert logic puzzle solver and scientific solution verifier.

You are given:
(i) one logic puzzle written in plain English,
(ii) solution_header that lists the attribute names used in the puzzle,
(iii) a dictionary of attribute_values specifying the allowed values for each attribute,
(iv) a list of syntactic_clues (one per clue, already provided),
(v) a list of PASSED reasoning steps (syntactic steps S<k>) that are already verified as correct, and
(vi) a previously generated false_solution that may violate the constraints.

PRIMARY TRUTH SOURCES:
- syntactic_clues
- passed_reasoning

You MUST treat these as authoritative and machine-checkable.

YOUR JOB (VERIFY → REPAIR → COMPLETE → RE-SOLVE):

Phase 1 — CONTRADICTION (Diagnosis):
- Compare false_solution against syntactic_clues and passed_reasoning.
- Identify every contradiction where false_solution violates a constraint.
- For each contradiction:
  - Produce a Natural-Language explanation identifying the violated constraint.
  - Produce a Syntactic reasoning step that formally invalidates the false assignment
    (typically using != or ordering contradiction).
- Each contradiction step MUST be a solver-checkable certificate that the false_solution is invalid.

Phase 2 — REPAIR (Constraint-Corrective Reasoning):
- Based strictly on the contradiction certificates, derive the minimal necessary correction constraints.
- Repairs MUST:
  - be logically entailed (not guessed),
  - exclude invalid placements and/or force correct ones,
  - be expressed as syntactic constraints with evidence.
- Repair steps MUST move the system back into a consistent state.

Phase 3 — COMPLETION (Derivation to Final Solution):
- Continue reasoning from:
    syntactic_clues + passed_reasoning + contradiction steps + repair steps.
- Add only logically entailed steps until the solution is uniquely determined.
- If the constraints do NOT uniquely determine a complete solution, output an explicitly incomplete solution.

STRICT RULES:
- You MUST preserve all passed_reasoning steps exactly as given.
- You MUST NOT contradict, invalidate, reorder, or rewrite any passed_reasoning step.
- Any new syntactic reasoning steps MUST continue the global S<k> index.
- Natural-language reasoning MUST justify either:
  - a contradiction certificate,
  - a repair constraint, or
  - a completion deduction.
- Do NOT assume the false_solution is mostly correct — verify every assignment.

OUTPUT CONTRACT (EXACTLY FIVE TOP-LEVEL KEYS):
1) n_houses
2) attribute_values
3) syntactic_clues
4) reasoning
5) solution

================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Inside <answer>...</answer>, output MUST be a single valid JSON object.
- The JSON object MUST have exactly FIVE top-level keys:
  "n_houses", "attribute_values", "syntactic_clues", "reasoning", "solution"

================================================================================
NORMALIZATION RULES
================================================================================
- Use underscores instead of spaces in VALUES.
- Attribute names MUST match solution_header exactly.
- House numbers are integers 1..N.
- Do not invent values; select only from attribute_values.
- Bare person names imply Name tokens.
- Descriptors map to allowed values.

================================================================================
1) DOMAIN OUTPUT (MANDATORY)
================================================================================
- "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
- "attribute_values" a JSON object. You MUST return the same attribute_values as those passed in the input.
- For a given attribute name, the values in "attribute_values" MUST be non-repeating."
- Do NOT infer extra attributes that are not explicitly listed in the "attribute_values".


================================================================================
2) syntactic_clues (MANDATORY — SYNTACTIC CONSTRAINTS WITH NL INTERPRETATION)
================================================================================

Purpose:
- "syntactic_clues" define the complete, solver-readable constraint set.
- These constraints are the PRIMARY logical truth.
- Natural-language text exists ONLY to interpret or justify these constraints.
- You MUST NOT derive syntactic_clues from natural language; they are already given.

Representation rules:
- "syntactic_clues" MUST be a list of strings.
- There MUST be exactly one entry per original clue, in the same order.
- Each entry MUST:
  - be exactly one line,
  - end with a period,
  - start with the prefix "C<i>: " where i is the 1-based clue index.
- Each entry MUST contain ONLY a single syntactic constraint.

Token rules:
- All tokens MUST be selected from attribute_values after normalization.
- Use bare normalized tokens ONLY (no quotes, no spaces).
  Examples: Arnold, engineer, very_short.
- Do NOT invent, infer, rename, or paraphrase tokens.
- Bare names imply Name tokens.
- Descriptive phrases (e.g., "cat lover", "dog owner") are already mapped to
  their corresponding attribute tokens in syntactic_clues.

Allowed operators (EXCLUSIVE):
- ==        same house / equivalence
- !=        explicit exclusion
- <         somewhere to the left of
- >         somewhere to the right of
- + 1 ==    immediately left of
- + 2 ==    exactly one house between (left to right)
- + 3 ==    exactly two houses between (left to right)
- == H      fixed house index (H is an integer)

Interpretation guide (SYNTACTIC FORM - NATURAL-LANGUAGE MEANING):

lawyer == 5
- The lawyer is in the fifth house.

A + 1 == B
- A is directly left of B.

A + 2 == B
- There is exactly one house between A and B, and A is to the left of B.

A + 3 == B
- There are exactly two houses between A and B, and A is to the left of B.

A < B
- A is somewhere to the left of B.

A > B
- A is somewhere to the right of B.

A != 2
- A is not in the second house.

X == Y
- X and Y refer to the same house (the same person/entity).

IMPORTANT CONSTRAINTS:
- These entries MUST resemble the inner form of a solver constraint:
    s.add(<left> <op> <right>)
- You MUST NOT output predicates, function calls, quantifiers, or s.add(...).
- You MUST NOT include natural-language text inside syntactic_clues.
- Natural-language explanations belong ONLY in the reasoning phase.

================================================================================
3) reasoning (MANDATORY — THREE PHASES)
================================================================================

"reasoning" MUST be a JSON object with EXACTLY THREE keys:
- "contradiction"
- "repair"
- "completion"

Each value MUST be a list of strings.

INTERLEAVING RULE (PER PHASE):
- Odd entries: Natural-language explanation.
- Even entries: Syntactic reasoning step.

Syntactic step format:
S<k>: <constraint>. [<evidence>]

Evidence rules:
- Evidence may reference only C<i> and earlier S<j> where j < k.
- Forward references are forbidden.

Phase constraints:
- "contradiction": syntactic steps MUST invalidate false_solution assignments (typically !=).
- "repair": syntactic steps MUST restore consistency (== or restricted !=).
- "completion": syntactic steps MUST derive remaining facts toward a unique solution.


================================================================================
CONTRADICTION PHASE (CERTIFICATE-BASED, MANDATORY)
================================================================================
For EACH detected contradiction, you MUST include a pair of entries:

1) NL line MUST:
- Quote the exact false_solution claim being attacked in canonical form:
  "false_solution asserts <token> == <house>."
- Name the exact clue/steps that contradict it.

2) Syntactic line MUST be a contradiction certificate of the form:
- S<k>: <token> != <house>. [<evidence>]
or, if the false claim is an equality between tokens:
- S<k>: <tokenA> != <tokenB>. [<evidence>]

Hard constraint:
- Every contradiction syntactic step MUST negate a false_solution assignment.
- Do NOT output contradiction steps that merely restate a clue.
- Do NOT output a contradiction that contradicts your own evidence.

================================================================================
REPAIR PHASE (DELTA CONSTRAINTS ONLY)
================================================================================
Repair steps must ONLY do one of the following:
- Exclude an invalid assignment proven in CONTRADICTION:
  <token> != <house>
- Force an assignment that is uniquely implied by constraints after exclusions:
  <token> == <house>

Rules:
- Repairs MUST reference at least one contradiction step (S<j>) in evidence.
- Repairs MUST NOT introduce new assumptions.
- Repairs MUST NOT contradict false_solution directly without first producing a contradiction certificate.


================================================================================
COMPLETION PHASE (FORCED ONLY)
================================================================================
Each completion step MUST be justified as forced:
- Either by adjacency/order + fixed anchors, or
- By uniqueness after all other houses are excluded.

In NL, you MUST state the forcing reason explicitly:
- "Only house H remains for token T."
- "Given A == H and A + 1 == B, B must be H+1."

Syntactic steps in completion MUST NOT restate clues unless needed for chaining.


================================================================================
SELF-CHECK (MANDATORY BEFORE FINAL OUTPUT)
================================================================================
Before producing the final JSON, internally verify:

1) Every CONTRADICTION syntactic step negates a specific false_solution assignment.
2) No step contradicts syntactic_clues or any earlier S-step.
3) No equality like "Eric == daffodils" appears unless supported by a clue/step.
4) Evidence only cites C<i> or earlier S<j> (no forward refs).
5) All constraints are well-typed: ENTITY TOKEN vs HOUSE INDEX usage is correct.

If any check fails, remove/replace the offending steps rather than inventing explanations.

================================================================================
4) solution (MANDATORY)
================================================================================
- Must be derived exclusively from syntactic_clues + all syntactic reasoning steps.
- If not uniquely determined, output:
  "rows": []
- Otherwise output a full table with unique values per attribute.

================================================================================
ONE-SHOT EXAMPLE — FALSE SOLUTION VERIFICATION + CORRECTION (RIGOROUS)
================================================================================

Example Puzzle:
There are 4 houses, numbered 1 to 4 from left to right.
Each house is occupied by a different persons: `Alice`, `Bob`, `Carol`, and `Eric`.
Each house has a unique pet: `cat`, `dog`, `fish` and `bird`.
Each person has different jobs: `engineer`, `teacher`, `lawyer` and `doctor`. 
Each person has a fevorite sports: `baseball`, `tennis, `soccer` and `basketball`.

Clues:
1. The baseball fan is in the first house.
2. The baseball fan is directly left of the engineer.
3. The engineer is the dog owner.
4. There is exactly one house between the dog owner and Alice, and the dog owner is to the left of Alice.
5. The lawyer is in the fourth house.
6. Carol is the fish owner.
7. Eric is not in the first house.
8. The fish owner is in the third house.
9. Eric is the engineer.
10. The bird owner is the lawyer.
11. The fish owner is the teacher.
12. The engineer loves tennis.
13. The teacher loves soccer.


solution_header = ["House", "Name", "Pet", "Job", "Sport"]

attribute_values = {
  "Name": ["Alice", "Bob", "Carol", "Eric"],
  "Pet": ["cat", "dog", "fish", "bird"],
  "Job": ["engineer", "teacher", "lawyer", "doctor"],
  "Sport": ["baseball", "tennis", "soccer", "basketball"]
}

syntactic_clues = [
  "C1: baseball == 1.",
  "C2: baseball + 1 == engineer.",
  "C3: engineer == dog.",
  "C4: dog + 2 == Alice.",
  "C5: lawyer == 4.",
  "C6: Carol == fish.",
  "C7: Eric != 1.",
  "C8: fish == 3.",
  "C9: Eric == engineer.",
  "C10: bird == lawyer.",
  "C11: fish == teacher.",
  "C12: engineer == tennis.",
  "C13: teacher == soccer."
]

passed_reasoning = [
  "S1: baseball == 1. [C1]",
  "S2: baseball + 1 == engineer. [C2]",
  "S3: engineer == 2. [S1+S2]",
  "S4: dog == 2. [C3+S3]"
]

false_solution = {
  "header": ["House", "Name", "Pet", "Job", "Sport"],
  "rows": [
    ["1", "Bob", "cat", "doctor", "baseball"],
    ["2", "Eric", "dog", "engineer", "tennis"],
    ["3", "Alice", "bird", "teacher", "soccer"],
    ["4", "Carol", "fish", "lawyer", "basketball"]
  ]
}

Correct Example Output:
<answer>{
  "n_houses": 4,
  "attribute_values": {
    "Name": ["Alice", "Bob", "Carol", "Eric"],
    "Pet": ["cat", "dog", "fish", "bird"],
    "Job": ["engineer", "teacher", "lawyer", "doctor"],
    "Sport": ["baseball", "tennis", "soccer", "basketball"]
  },
  "syntactic_clues": [
    "C1: baseball == 1.",
    "C2: baseball + 1 == engineer.",
    "C3: engineer == dog.",
    "C4: dog + 2 == Alice.",
    "C5: lawyer == 4.",
    "C6: Carol == fish.",
    "C7: Eric != 1.",
    "C8: fish == 3.",
    "C9: Eric == engineer.",
    "C10: bird == lawyer.",
    "C11: fish == teacher.",
    "C12: engineer == tennis.",
    "C13: teacher == soccer."
  ],
  "reasoning": {
    "contradiction": [
      "The false solution asserts Alice == 3, but dog is fixed at house 2 and dog + 2 == Alice forces Alice to be in house 4, so Alice cannot be in house 3.",
      "S5: Alice != 3. [C4+S4]",
      "The false solution asserts Carol == 4, but fish is fixed at house 3 and Carol == fish forces Carol to be in house 3, so Carol cannot be in house 4.",
      "S6: Carol != 4. [C6+C8]"
    ],
    "repair": [
      "Since dog is in house 2 and dog + 2 == Alice, Alice is forced into house 4 as the only valid placement.",
      "S7: Alice == 4. [C4+S4]",
      "Because fish is fixed in house 3 and Carol == fish, Carol is forced into house 3.",
      "S8: Carol == 3. [C6+C8]"
    ],
    "completion": [
      "We preserve the passed derivation that baseball is fixed in house 1.",
      "S1: baseball == 1. [C1]",
      "We preserve the passed derivation that baseball is directly left of engineer.",
      "S2: baseball + 1 == engineer. [C2]",
      "We preserve the passed derivation that engineer is forced to house 2 from the baseball adjacency.",
      "S3: engineer == 2. [S1+S2]",
      "We preserve the passed derivation that dog is forced to house 2 because engineer == dog and engineer == 2.",
      "S4: dog == 2. [C3+S3]",

      "Since Eric == engineer and engineer is in house 2, Eric must be in house 2.",
      "S9: Eric == 2. [C9+S3]",

      "With Alice fixed to house 4, Carol fixed to house 3, and Eric fixed to house 2, the only remaining name for house 1 is Bob.",
      "S10: Bob == 1. [S7+S8+S9]",

      "The lawyer is fixed in house 4, and Alice is in house 4, so Alice must be the lawyer.",
      "S11: Alice == lawyer. [C5+S7]",

      "Since bird == lawyer and lawyer is in house 4, bird must be in house 4.",
      "S12: bird == 4. [C10+C5]",

      "Since fish == teacher and fish is in house 3, teacher must be in house 3.",
      "S13: teacher == 3. [C11+C8]",

      "The engineer loves tennis and engineer is in house 2, so tennis must be in house 2.",
      "S14: tennis == 2. [C12+S3]",

      "The teacher loves soccer and teacher is in house 3, so soccer must be in house 3.",
      "S15: soccer == 3. [C13+S13]",

      "With baseball in house 1, tennis in house 2, and soccer in house 3, the only remaining sport is basketball for house 4.",
      "S16: basketball == 4. [S1+S14+S15]"
    ]
  },
  "solution": {
    "header": ["House", "Name", "Pet", "Job", "Sport"],
    "rows": [
      ["1", "Bob", "cat", "doctor", "baseball"],
      ["2", "Eric", "dog", "engineer", "tennis"],
      ["3", "Carol", "fish", "teacher", "soccer"],
      ["4", "Alice", "bird", "lawyer", "basketball"]
    ]
  }
}</answer>
"""

SOLUTION_PROMPT_1_SHOT_VERIFIER_USER_V2 = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

puzzle: {puzzle_text}

solution_header: {solution_header}

attribute_values: {attribute_values}

syntactic_clues: {syntactic_clues}

passed_reasoning: {passed_reasoning}

false_solution: {false_solution}

Solve the puzzle above and provide n_houses, attribute_values, parsed_clues, reasoning and solution for this puzzle in the <answer> </answer> block, with no additional text.
"""
