SOLUTION_PROMPT_VERIFIER_V2 = """
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
4) reasoning — interleaved reasoning consisting of natural-language explanations and syntactic (solver-checkable) deduction steps.
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
- When a clue states "next to each other", encode it as: Or(A == B + 1, A == B - 1)
  Example: "The person who prefers city breaks and Alice are next to each other" -> C12: "Or(city_breaks == Alice + 1, city_breaks == Alice - 1)."
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

Syntactic entry format:
- Every syntactic entry MUST start with "S<k>: " and MUST end with a period.
- <k> starts at 1 and increments by 1 for each syntactic step only (S1, S2, S3, ...).
- The syntactic constraint MUST be solver-verifiable and may use ONLY:
  ==, !=, <, >, + d ==, Not(...), And(...), Or(...)

- Each syntactic step MUST be written in the exact form: S<k>

  Atomic operators:
    ==        (same house / equivalence)
    !=        (not the same house)
    <         (somewhere to the left of)
    >         (somewhere to the right of)
    + d ==    (directed distance; d is a positive integer)
    == H      (fixed house index; H is an integer in 1..n_houses)

  Boolean operators:
    Not(e)    (negation of a single atomic expression)
    And(e1, e2, ..., en)
    Or(e1, e2, ..., en)

- Boolean operators may ONLY be applied to valid atomic expressions.
- Nested Boolean expressions are allowed but MUST remain solver-verifiable.

Examples of valid INTERLEVED reasoning steps:
    The engineer is assigned to house 2.
    S1: engineer == 2.

    Since the engineer occupies house 2, the dog cannot also be in house 2.
    S2: dog != 2.

    The cat is immediately to the left of the coffee, so the cat’s house index plus one equals the coffee’s house index.
    S3: cat + 1 == coffee.

    The green house appears somewhere to the left of the white house.
    S4: green < white.

    The dog is not in the first house.
    S5: Not(dog == 1).

    The cat cannot be in house 1 or house 3.
    S6: And(cat != 1, cat != 3).

    The milk is located either in house 1 or in house 5.
    S7: Or(milk == 1, milk == 5).

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
ONE-SHOT EXAMPLE (3 HOUSES, 3 ATTRIBUTES)
================================================================================

Example Puzzle:
There are 3 houses, numbered 1 to 3 from left to right. Each house is occupied by a different person.
Each house has a unique attribute for each of the following characteristics:

- Each person has a unique name: Peter, Eric, Arnold
- The people like unique colors: red, white, yellow
- The people have childern named: Fred, Meredith, Bella

Clues:
1. Arnold is the person whose favorite color is red.
2. The person whose child is named Fred is somewhere to the left of Eric.
3. The person whose favorite color is red is in the second house.
4. The person whose child is named Bella is in the first house.
5. The person who loves white is the person whose child is named Meredith.

solution_header = ["House", "Name", "Color", "Children"]

attribute_values = {
  "Name": ["Peter", "Eric", "Arnold"],
  "Color": ["red", "white", "yellow"],
  "Children": ["Fred", "Meredith", "Bella"]
}


Correct Example Output:
<answer>{
  "n_houses": 3,
  "attribute_values": {
  "Name": ["Peter", "Eric", "Arnold"],
  "Color": ["red", "white", "yellow"],
  "Children": ["Fred", "Meredith", "Bella"]
  },
  "syntactic_clues": [
    "C1: Arnold == red.",
    "C2: Fred < Eric.",
    "C3: red == 2.",
    "C4: Bella == 1.",
    "C5: white == Meredith."
  ],
  "reasoning": [
    "Clue 3 immediately anchors the red favorite color in the second house, which is a very strong positional fact to start from.",
    "S1: red == 2.",
    "Clue 1 then ties Arnold directly to the red color, so Arnold must be in that same second house."
    "S2: Arnold == red.",
    "Putting those two together, Arnold is fixed in house 2. At this point, house 2 is completely identified as “Arnold’s house,” and we know it has the red color.",
    "S3: Arnold == 2.",
    "Clue 4 gives us another concrete placement: the child Bella is in the first house. So whatever person lives in house 1, their child must be Bella.",
    "S4: Bella == 1.",
    "So far, we know: House 1 has child Bella, House 2 has Arnold and the color red, House 3 is still entirely open. Now we look at Clue 2, which introduces a relative ordering: the person whose child is Fred is somewhere to the left of Eric. This doesn’t give a house yet, but it constrains the ordering.",
    "S5: Fred < Eric.",
    "Since houses are only 1 through 3, Eric cannot be in the first house (there would be nothing to the left of him). That means Eric must be in house 2 or house 3.",
    "S6: Or(Eric == 2, Eric == 3).",
    "But we already know Arnold occupies house 2, and all people are distinct. So Eric cannot be in house 2 and must therefore be in house 3.",
    "S7: Eric == 3",
    "This is a good point to pause and take stock again. House 1: unknown person, child Bella. House 2: Arnold, red. House 3: Eric. Now, going back to the same ordering constraint (Fred < Eric), if Eric is in house 3, then Fred must be in house 1 or house 2.",
    "S8: Or(Fred == 1, Fred == 2).",
    "But we already know the child in house 1 is Bella, and children are unique. So Fred cannot be in house 1. That forces Fred into house 2.",
    "S9: Fred == 2.",
    "This tells us that Arnold, who is in house 2, is also the parent of Fred. At this stage, all people except Peter are placed: Arnold is in house 2 and Eric is in house 3. Since each house has exactly one person, Peter must be in the remaining house, house 1.",
    "S10: Peter == 1.",
    "Let’s summarize again. House 1: Peter, child Bella. House 2: Arnold, red, child Fred. House 3: Eric. Now consider the children again. Bella is in house 1 and Fred is in house 2, so the only remaining child, Meredith, must be in house 3.",
    "S11: Meredith == 3.",
    "Clue 5 connects the color white to Meredith’s parent, meaning the white color must be in the same house as Meredith.",
    "S12: white == Meredith.",
    "Since Meredith is in house 3, white must be in house 3 as well.",
    "S13: white == 3.",
    "At this point, two colors are fixed: red in house 2 and white in house 3. Colors are unique, so the only remaining color, yellow, must belong to house 1.",
    "S14: yellow == 1.",
    "With that, everything is now determined: House 1: Peter, yellow, child Bella. House 2: Arnold, red, child Fred. House 3: Eric, white, child Meredith. All clues are satisfied, and no attributes remain unassigned.",
  ],
  "solution": {
    "header": ["House", "Name", "Color", "Children"],
    "rows": [
      ["1", "Peter", "yellow", "Bella"],
      ["2", "Arnold", "red", "Fred"],
      ["3", "Eric", "white", "Meredith"]
    ]
  }
}</answer>
"""

SOLUTION_PROMPT_1_SHOT_VERIFIER_USER_V2 = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

puzzle = {puzzle}

solution_header = {solution_header}

attribute_values = {attribute_values}

passed_reasoning = {passed_reasoning}

Solve the puzzle above and provide n_houses, attribute_values, parsed_clues, reasoning, and solution for this puzzle in the <answer> </answer> block, with no additional text.
"""

