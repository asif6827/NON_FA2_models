
SOLUTION_PROMPT_VERIFIER_V2="""
You are an expert logic puzzle solver.

You are given:
(i) one logic puzzle_text written in plain English,
(ii) solution_header that lists the attribute names used in the puzzle,
(iii) a dictionary of attribute_values specifying the allowed values for each attribute, and
(iv) a list of PASSED reasoning steps that are already verified as correct.

IMPORTANT CONTEXT:
- The passed reasoning steps represent a valid partial derivation.
- You MUST CONTINUE reasoning from these steps.
- You are REQUIRED to preserve all passed reasoning steps and build upon them.
- You MUST NOT contradict, invalidate, or rewrite any passed reasoning step.
- Your task is to extend the reasoning, add new valid deductions, and re-generate
  the final solution using:
    (a) the syntactic_clues,
    (b) the passed reasoning steps, and
    (c) your newly derived reasoning steps.

Your job is to generate the following FIVE fields:
1) n_houses
2) attribute_values
3) syntactic_clues
4) reasoning (Categorized + INTERLEAVED natural-language + syntactic)
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

Natural-language entries:
- Must explain the deduction in plain English.
- Must justify why the immediately following syntactic step is valid.
- Must not introduce assumptions or unstated facts.

Syntactic reasoning entries:
- Must encode the newly deduced fact only.
- Must not merely restate a clue unless required for dependency chaining.
- Must be written in Z3-like constraint form.
- Tokens MUST map directly to values in "attribute_values".

Syntactic reasoning step format:
  S<k>: <constraint>. [<evidence>]

Where:
- <k> starts at 1 and increments globally across all categories.
- Only syntactic steps increment the counter.
- <constraint> must follow the same operator rules as syntactic_clues (==, !=, <, >, + 1 ==).
- House positions are encoded as integers from 1 to n_houses.

Evidence rules:
- Evidence MUST be included for EVERY syntactic step.
- Evidence may reference: C<i> and earlier S<j> only, where j < k.
- Use "+" to join multiple evidence references, e.g., [C3+C11+S2].

Examples:
- "S1: baseball == 1. [C18]"
- "S2: engineer == 2. [C12+S1]"
- "S3: dog == 2. [C1+S2]"

Logical validity requirement:
- Every syntactic step MUST be logically entailed by the syntactic_clues plus all earlier syntactic steps.
- Forward references are forbidden.
- Do NOT output syntactic steps that merely restate a clue unless required as part of the deduction chain.
- All steps must be verifiable by a constraint solver.

Category semantics (binding):
- "Abs_Placement" fixes a variable to a specific house.
- "Direct_Equality" binds two attributes or entities together.
- "Directed_Adjacency" enforces exact positional adjacency with direction.
- "Structural_Positioning" expresses relative ordering constraints.
- "Domain_Restriction" shrinks a variable’s allowed value set.
- "Exclusion" explicitly forbids a value or pairing.
- "Propagation" derives immediate consequences from existing constraints.
- "Forced_Resolution" assigns a value because only one option remains.
- "Disjunction" explicitly maintains multiple remaining possibilities and MUST be expressed in solver-valid logical OR form.
- "Case_Split" resolves a disjunction by contradiction or elimination.

Global ordering rule:
- Categories do not impose solver ordering, but reasoning steps MUST respect
  logical preconditions implied by their category semantics.
- The syntactic step index S<k> is the only authoritative ordering mechanism.
- All dependencies must point from lower S<k> to higher S<k>.

Final invariant:
- If all syntactic_clues and all syntactic reasoning steps are given to a solver,
  every step must be provably valid with no ambiguity.

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
ONE-SHOT EXAMPLE — Demonstration of REQUIRED OUTPUT (CONTINUATION SETTING)
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

passed_reasoning = [
  "S1: baseball == 1. [C18]",
  "S2: baseball + 1 == engineer. [C12]",
  "S3: engineer == 2. [S1+S2]",
  "S4: engineer == dog. [C1]",
  "S5: dog == 2. [S3+S4]",
  "S6: lawyer == 5. [C9]",
  "S7: super_tall == 5. [C20]",
  "S8: teacher + 1 == soccer. [C15]",
  "S9: Carol == soccer. [C7]",
  "S10: Carol == fish. [C17]"
]

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
      "The passed steps already anchor baseball in house 1, so we keep that fixed point.",
      "S1: baseball == 1. [C18]",
      "The passed steps already place the lawyer in house 5, so house 5 is constrained early.",
      "S6: lawyer == 5. [C9]",
      "The passed steps already fix super_tall in house 5, strengthening the profile of house 5.",
      "S7: super_tall == 5. [C20]"
    ],

    "Direct_Equality": [
      "The passed steps already link baseball being directly left of engineer, so we retain that adjacency constraint.",
      "S2: baseball + 1 == engineer. [C12]",
      "The passed steps already bind engineer and dog to the same house.",
      "S4: engineer == dog. [C1]",
      "The passed steps already bind Carol to the soccer fan.",
      "S9: Carol == soccer. [C7]",
      "The passed steps already bind Carol to the fish enthusiast.",
      "S10: Carol == fish. [C17]"
    ],

    "Directed_Adjacency": [
      "The passed steps already include the teacher-to-soccer adjacency constraint, which we must preserve.",
      "S8: teacher + 1 == soccer. [C15]"
    ],

    "Structural_Positioning": [
      "The cat lover must be to the right of very_short, so cat is constrained to occur after very_short in house order.",
      "S11: cat > very_short. [C19]"
    ],

    "Domain_Restriction": [
      "Because baseball is fixed in house 1, baseball is excluded from house 2 by uniqueness.",
      "S12: baseball != 2. [S1]"
    ],

    "Exclusion": [
      "Since super_tall is in house 5, tall cannot be in house 5 under height uniqueness.",
      "S13: tall != 5. [S7]"
    ],

    "Propagation": [
      "Since baseball is in house 1 and is directly left of engineer, engineer remains forced into house 2 as already established by the passed steps.",
      "S3: engineer == 2. [S1+S2]",
      "Since engineer equals dog and engineer is in house 2, dog remains forced into house 2 as already established by the passed steps.",
      "S5: dog == 2. [S3+S4]",
      "Since Carol equals soccer and the teacher is directly left of soccer, placing soccer in house 6 would force teacher in house 5 which conflicts with lawyer in house 5, so soccer must be in house 5.",
      "S14: soccer == 5. [S6+S8+S9]",
      "Because teacher is directly left of soccer and soccer is in house 5, the teacher must be in house 4.",
      "S15: teacher == 4. [S8+S14]",
      "Since Carol is the soccer fan and soccer is in house 5, Carol must live in house 5.",
      "S16: Carol == 5. [S9+S14]",
      "Because Carol is the fish enthusiast and Carol is in house 5, fish must be in house 5.",
      "S17: fish == 5. [S10+S16]",
      "Since tennis equals teacher and teacher is in house 4, tennis must be in house 4.",
      "S18: tennis == 4. [C10+S15]",
      "Since horse equals teacher and teacher is in house 4, horse must be in house 4.",
      "S19: horse == 4. [C6+S15]",
      "Since average is directly left of rabbit and dog is already in house 2, average cannot be in house 2, so we shift average to house 1 to allow rabbit in house 2.",
      "S20: average == 1. [C3+S5]",
      "Since average is directly left of rabbit and average is in house 1, rabbit must be in house 2.",
      "S21: rabbit == 2. [C3+S20]",
      "Because rabbit equals Alice and rabbit is in house 2, Alice must be in house 2.",
      "S22: Alice == 2. [C16+S21]",
      "Because average equals swimming and average is in house 1, swimming must be in house 1.",
      "S23: swimming == 1. [C11+S20]"
    ],

    "Forced_Resolution": [
      "With soccer fixed in house 5 and teacher fixed in house 4, the occupation slots and sport slots narrow and force remaining placements step by step.",
      "S24: tennis != 5. [S18+S14]"
    ],

    "Disjunction": [
      "Before final pruning, tall could tentatively fall in either house 3 or house 6 while still allowing a very_short somewhere to its right.",
      "S25: Or(tall == 3, tall == 6). [C4]"
    ],

    "Case_Split": [
      "If tall were in house 6 then very_short would need to be to its right, which is impossible, so tall cannot be in house 6.",
      "S26: tall != 6. [C4]",
      "With tall restricted to houses 3 or 6 and tall not equal to 6, tall must be in house 3.",
      "S27: tall == 3. [S25+S26]"
    ]
  },
  "solution": {
    "header": ["House", "Name", "Animal", "Occupation", "Sport", "Height"],
    "rows": [
      ["1", "Peter", "bird", "nurse", "baseball", "average"],
      ["2", "Alice", "rabbit", "engineer", "swimming", "short"],
      ["3", "Bob", "dog", "artist", "volleyball", "tall"],
      ["4", "Eric", "horse", "teacher", "tennis", "very_short"],
      ["5", "Carol", "fish", "lawyer", "soccer", "super_tall"],
      ["6", "Arnold", "cat", "doctor", "basketball", "very_tall"]
    ]
  }
}</answer>
"""


SOLUTION_PROMPT_1_SHOT_VERIFIER_USER_V2 = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

puzzle_text: {puzzle_text}

solution_header: {solution_header}

attribute_values: {attribute_values}

passed_reasoning: {passed_reasoning}

Solve the puzzle above and provide n_houses, attribute_values, parsed_clues, reasoning and solution for this puzzle in the <answer> </answer> block, with no additional text.
"""
