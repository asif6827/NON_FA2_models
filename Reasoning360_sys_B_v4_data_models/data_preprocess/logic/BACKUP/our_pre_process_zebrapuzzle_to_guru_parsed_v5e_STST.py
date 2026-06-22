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

All values appearing in syntactic_clues, reasoning, self_verification, and the final solution MUST be drawn from attribute_values and interpreted as entity tokens representing unknown house positions.

Your task is to construct a fully consistent, solver-verifiable solution by generating the following SIX fields:
1) n_houses — the total number of houses in the puzzle.
2) attribute_values — returned exactly as given, without modification.
3) syntactic_clues — a normalized, Z3-style textual encoding of each clue.
4) reasoning — categorized reasoning consisting of interleaved natural-language explanations and syntactic (solver-checkable) deduction steps.
5) self_verification — a linear, interleaved list of natural-language explanations and syntactic checks that verify the reasoning and final solution against the syntactic_clues.
6) solution — the final house-by-house assignment derived exclusively from syntactic_clues, reasoning, and validated by self_verification.

The purpose of self_verification is strictly to confirm correctness:
- It MUST NOT introduce new deductions or assignments.
- It MUST only re-derive or check constraints already implied by syntactic_clues and established reasoning steps.
- Each syntactic verification step MUST be solver-checkable and reference previously derived steps.

You MUST return the result STRICTLY as a single valid JSON object wrapped inside:
<answer>...</answer>

No additional text, commentary, or formatting outside the <answer> block is permitted.


================================================================================
CRITICAL FORMAT REQUIREMENTS
================================================================================
- Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
- Do NOT include extra text, markdown, explanations, or code fences.
- Inside <answer>...</answer>, the content MUST be a single valid JSON object.
- The JSON object MUST have exactly SIX top-level keys, spelled EXACTLY:
    "n_houses",
    "attribute_values",
    "syntactic_clues",
    "reasoning",
    "self_verification",
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
- "Disjunction" explicitly maintains multiple remaining possibilities and MUST be
  expressed in solver-valid logical OR form.
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
4) SELF_VERIFICATION
================================================================================

Purpose:
- self_verification is a VALIDATION phase, not a reasoning phase.
- Its sole purpose is to confirm that the derived reasoning and final solution
  satisfy the syntactic_clues.

Format:
- "self_verification" MUST be a list of strings.
- Entries MUST be INTERLEAVED:
    Odd entries: Natural-language statement of what is being verified.
    Even entries: Syntactic verification step.

Syntactic step format:
  S<k>: <constraint>. [<evidence>]

Rules:
- self_verification MUST NOT introduce new facts, assignments, exclusions,
  disjunctions, or case splits.
- Every syntactic verification step MUST be solver-checkable.
- Evidence may reference only syntactic_clues (C<i>) and earlier reasoning steps (S<j>).
- No forward references.

What to verify:
- All load-bearing syntactic_clues must be explicitly verified.
- Verification must connect derived house assignments to the final solution rows.
- Ordering (<, >), adjacency (+k ==), and equality (==) constraints must be checked
  using concrete house indices.

Gate condition:
- The final solution MUST NOT be output unless self_verification confirms that the
  syntactic_clues are satisfied by the derived assignments.

Invariant:
- Given syntactic_clues, reasoning, and self_verification, a constraint solver
  must be able to independently confirm the solution with no ambiguity.
================================================================================
5) solution (MANDATORY — DERIVED FROM SYNTACTIC CLUES AND REASONING)
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
ONE-SHOT EXAMPLE — Demonstration of REQUIRED OUTPUT (WITH SELF-VERIFICATION)
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
  "reasoning": {
    "Abs_Placement": [
      "Clue 18 fixes baseball in house 1.",
      "S1: baseball == 1. [C18]",
      "Clue 9 fixes lawyer in house 5.",
      "S2: lawyer == 5. [C9]",
      "Clue 20 fixes super_tall in house 5.",
      "S3: super_tall == 5. [C20]"
    ],
    "Direct_Equality": [
      "Clue 1 binds engineer and dog.",
      "S4: engineer == dog. [C1]",
      "Clue 7 binds Carol and soccer.",
      "S5: Carol == soccer. [C7]",
      "Clue 17 binds fish and Carol.",
      "S6: fish == Carol. [C17]",
      "Clue 6 binds horse and teacher.",
      "S7: horse == teacher. [C6]",
      "Clue 10 binds tennis and teacher.",
      "S8: tennis == teacher. [C10]",
      "Clue 11 binds average and swimming.",
      "S9: average == swimming. [C11]",
      "Clue 16 binds rabbit and Alice.",
      "S10: rabbit == Alice. [C16]",
      "Clue 8 binds tall and volleyball.",
      "S11: tall == volleyball. [C8]",
      "Clue 13 binds Peter and nurse.",
      "S12: Peter == nurse. [C13]",
      "Clue 5 binds Arnold and cat.",
      "S13: Arnold == cat. [C5]"
    ],
    "Directed_Adjacency": [
      "Clue 12 states baseball is directly left of engineer.",
      "S14: baseball + 1 == engineer. [C12]",
      "Clue 15 states teacher is directly left of soccer.",
      "S15: teacher + 1 == soccer. [C15]",
      "Clue 3 states average is directly left of rabbit.",
      "S16: average + 1 == rabbit. [C3]"
    ],
    "Structural_Positioning": [
      "Clue 2 states average is somewhere left of short.",
      "S17: average < short. [C2]",
      "Clue 4 states tall is somewhere left of very_short.",
      "S18: tall < very_short. [C4]",
      "Clue 14 states Bob is somewhere right of artist.",
      "S19: Bob > artist. [C14]",
      "Clue 19 states cat is somewhere right of very_short.",
      "S20: cat > very_short. [C19]"
    ],
    "Domain_Restriction": [
      "Because baseball is fixed at house 1, it cannot be in any other house.",
      "S21: baseball != 2. [S1]",
      "Because baseball is fixed at house 1, it cannot be in house 3.",
      "S22: baseball != 3. [S1]",
      "Because baseball is fixed at house 1, it cannot be in house 4.",
      "S23: baseball != 4. [S1]",
      "Because baseball is fixed at house 1, it cannot be in house 5.",
      "S24: baseball != 5. [S1]",
      "Because baseball is fixed at house 1, it cannot be in house 6.",
      "S25: baseball != 6. [S1]"
    ],
    "Exclusion": [
      "Since super_tall is in house 5, very_short cannot be in house 5 by height uniqueness.",
      "S26: very_short != 5. [S3]",
      "Since super_tall is in house 5, tall cannot be in house 5 by height uniqueness.",
      "S27: tall != 5. [S3]"
    ],
    "Propagation": [
      "Baseball is in house 1 and is directly left of engineer, so engineer must be in house 2.",
      "S28: engineer == 2. [S1+S14]",
      "Engineer equals dog, so dog must be in house 2.",
      "S29: dog == 2. [S4+S28]",
      "If soccer were in house 6, teacher would be in house 5 which conflicts with lawyer in house 5, so soccer must be in house 5.",
      "S30: soccer == 5. [S2+S15+S5]",
      "Teacher is directly left of soccer, so teacher must be in house 4.",
      "S31: teacher == 4. [S15+S30]",
      "Since Carol equals soccer and soccer is in house 5, Carol must be in house 5.",
      "S32: Carol == 5. [S5+S30]",
      "Since fish equals Carol and Carol is in house 5, fish must be in house 5.",
      "S33: fish == 5. [S6+S32]",
      "Horse equals teacher and teacher is in house 4, so horse must be in house 4.",
      "S34: horse == 4. [S7+S31]",
      "Tennis equals teacher and teacher is in house 4, so tennis must be in house 4.",
      "S35: tennis == 4. [S8+S31]",
      "Average is directly left of rabbit, and house 2 is already occupied by dog/engineer, so average must be in house 2 to keep rabbit in house 3.",
      "S36: average == 2. [S16+S29]",
      "Average is directly left of rabbit, so rabbit must be in house 3.",
      "S37: rabbit == 3. [S16+S36]",
      "Rabbit equals Alice, so Alice must be in house 3.",
      "S38: Alice == 3. [S10+S37]",
      "Average equals swimming, so swimming must be in house 2.",
      "S39: swimming == 2. [S9+S36]"
    ],
    "Forced_Resolution": [
      "Tall must be left of very_short, so tall cannot be in house 6.",
      "S40: tall != 6. [S18]",
      "Tall is not allowed in house 5 and cannot be in house 6, and house 4 is teacher/horse/tennis, so tall is forced into house 3.",
      "S41: tall == 3. [S27+S31+S40]",
      "Tall equals volleyball, so volleyball must be in house 3.",
      "S42: volleyball == 3. [S11+S41]",
      "With average in 2 and tall in 3, very_short must be to the right of tall and cannot be in 5, so very_short is forced into house 4.",
      "S43: very_short == 4. [S18+S26+S41]",
      "Cat must be to the right of very_short, so cat cannot be in houses 1..4 and cannot be in house 5 (fish/Carol), so cat is forced into house 6.",
      "S44: cat == 6. [S20+S33+S43]",
      "Arnold equals cat, so Arnold must be in house 6.",
      "S45: Arnold == 6. [S13+S44]",
      "Bob must be to the right of artist, and the only available position to the right of house 3 is house 4 among remaining Name slots, so Bob is forced into house 4.",
      "S46: Bob == 4. [S19+S31+S32+S38+S45]",
      "With Carol in 5, Alice in 3, Arnold in 6, and Bob in 4, the only remaining name positions are house 1 and 2 for Peter and Eric, and engineer is already in house 2 so Eric is forced into house 2.",
      "S47: Eric == 2. [S28+S32+S38+S45+S46]",
      "With Eric fixed in house 2, the remaining name Peter must be in house 1.",
      "S48: Peter == 1. [S12+S47]",
      "With dog in 2, rabbit in 3, horse in 4, fish in 5, and cat in 6, the remaining animal bird must be in house 1.",
      "S49: bird == 1. [S29+S37+S34+S33+S44]",
      "Lawyer is fixed in house 5, so Carol in house 5 must be the lawyer.",
      "S50: Carol == lawyer. [S2+S32]",
      "Teacher is fixed in house 4, so Bob in house 4 must be the teacher.",
      "S51: Bob == teacher. [S31+S46]",
      "With lawyer in 5, teacher in 4, and engineer in 2, the remaining occupations for houses 1, 3, 6 are nurse, artist, doctor, and Alice in 3 is forced to be artist to satisfy Bob > artist.",
      "S52: artist == 3. [S19+S46+S38]",
      "Since Peter == nurse and Peter is in house 1, nurse must be in house 1.",
      "S53: nurse == 1. [S12+S48]",
      "The remaining occupation for house 6 is doctor.",
      "S54: doctor == 6. [S2+S31+S28+S52+S53]",
      "Since super_tall is in house 5, and tall is in 3 and very_short is in 4 and average is in 2, the remaining heights for houses 1 and 6 are very_tall and short, and short must be to the right of average so short is in house 6.",
      "S55: short == 6. [S17+S36+S3+S41+S43]",
      "Therefore very_tall must be in house 1.",
      "S56: very_tall == 1. [S3+S36+S41+S43+S55]",
      "With baseball in 1, swimming in 2, volleyball in 3, tennis in 4, soccer in 5, the remaining sport basketball must be in 6.",
      "S57: basketball == 6. [S1+S39+S42+S35+S30]"
    ],
    "Disjunction": [
      "Before forcing volleyball, tall == volleyball allows an initial disjunction where volleyball could have been placed in house 3 or house 6 under tall < very_short.",
      "S58: Or(volleyball == 3, volleyball == 6). [S11+S18]"
    ],
    "Case_Split": [
      "If volleyball were in house 6 then tall would be in house 6, but tall < very_short would become impossible, so volleyball cannot be in house 6.",
      "S59: volleyball != 6. [S18+S11]",
      "Since volleyball is either in house 3 or house 6 and it is not in house 6, volleyball must be in house 3.",
      "S60: volleyball == 3. [S58+S59]"
    ]
  },
  "self_verification": [
    "We now self-verify that the derived house assignments satisfy the clue constraints and justify the final solution placements.",
    "S61: engineer == dog. [S28+S29]",
    "We verify the ordering constraint average < short using the derived placements average == 2 and short == 6.",
    "S62: average < short. [S36+S55]",
    "We verify the adjacency constraint average + 1 == rabbit using average == 2 and rabbit == 3.",
    "S63: average + 1 == rabbit. [S36+S37]",
    "We verify the ordering constraint tall < very_short using tall == 3 and very_short == 4.",
    "S64: tall < very_short. [S41+S43]",
    "We verify Arnold == cat using Arnold == 6 and cat == 6.",
    "S65: Arnold == cat. [S45+S44]",
    "We verify horse == teacher using horse == 4 and teacher == 4.",
    "S66: horse == teacher. [S34+S31]",
    "We verify Carol == soccer using Carol == 5 and soccer == 5.",
    "S67: Carol == soccer. [S32+S30]",
    "We verify tall == volleyball using tall == 3 and volleyball == 3.",
    "S68: tall == volleyball. [S41+S42]",
    "We verify lawyer == 5 using the established placement.",
    "S69: lawyer == 5. [S2]",
    "We verify tennis == teacher using tennis == 4 and teacher == 4.",
    "S70: tennis == teacher. [S35+S31]",
    "We verify average == swimming using average == 2 and swimming == 2.",
    "S71: average == swimming. [S36+S39]",
    "We verify baseball + 1 == engineer using baseball == 1 and engineer == 2.",
    "S72: baseball + 1 == engineer. [S1+S28]",
    "We verify Peter == nurse using Peter == 1 and nurse == 1.",
    "S73: Peter == nurse. [S48+S53]",
    "We verify Bob > artist using Bob == 4 and artist == 3.",
    "S74: Bob > artist. [S46+S52]",
    "We verify teacher + 1 == soccer using teacher == 4 and soccer == 5.",
    "S75: teacher + 1 == soccer. [S31+S30]",
    "We verify rabbit == Alice using rabbit == 3 and Alice == 3.",
    "S76: rabbit == Alice. [S37+S38]",
    "We verify fish == Carol using fish == 5 and Carol == 5.",
    "S77: fish == Carol. [S33+S32]",
    "We verify baseball == 1 using the established placement.",
    "S78: baseball == 1. [S1]",
    "We verify cat > very_short using cat == 6 and very_short == 4.",
    "S79: cat > very_short. [S44+S43]",
    "We verify super_tall == 5 using the established placement.",
    "S80: super_tall == 5. [S3]",
    "Since all self-verification constraints are satisfied, the following solution table is justified by the derived placements.",
    "S81: True == True. [S61+S62+S63+S64+S65+S66+S67+S68+S69+S70+S71+S72+S73+S74+S75+S76+S77+S78+S79+S80]"
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

Solve the puzzle above and provide n_houses, attribute_values, parsed_clues, parsed_reasoning, self_verification and solution for this puzzle in the <answer> </answer> block, with no additional text.
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
