SOLUTION_PROMPT_VERIFIER_V2 = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
    - Extract the domain (N houses + all attribute values).
    - Parse each clue into a canonical, machine-checkable form (ONE parsed clue per clue, SAME ORDER).
    - Perform step-by-step deductions using only canonical atoms.
    - Derive a correct final solution.
    - Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
    - Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
    - Do NOT include extra text, markdown, explanations, or code fences.
    - Inside <answer>...</answer>, the content MUST be a single valid JSON object.
    - The JSON object MUST have exactly FIVE top-level keys:
      "n_houses", "attribute_values", "parsed_clues", "parsed_reasoning", "solution".
    - Do NOT add any other keys.

NORMALIZATION RULES:
    - Use underscores instead of spaces in VALUES (e.g., grilled_cheese, root_beer, bmw_3_series).
    - Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Drink, Pet, HairColor, Lunch, Nationality, PhoneModel, etc.
    - House numbers are integers 1..N.
    - Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6.
    - Do NOT invent values. Every <Val> must be one of the allowed values listed in the puzzle text (after normalization).
    - If the clue mentions a bare person name (e.g., "Bob"), treat it as Name=Bob.
    - If the clue mentions a bare demonym (e.g., "The German"), map it to Nationality=german (or the matching attribute in the puzzle text).
    - If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching attribute/value from the puzzle text
      (e.g., Pet=cat, Drink=coffee), choosing the closest listed value.

D) DOMAIN OUTPUT (MANDATORY)
    - "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
    - "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values from the puzzle text.
    - Each attribute list MUST contain exactly N unique values (after normalization).
    - Include every attribute listed in the puzzle text, and only those attributes.
    - Do NOT infer extra attributes that are not explicitly listed in the puzzle text.

A) parsed_clues (MANDATORY, PARSABLE, ORDERED, FAITHFUL)
    - "parsed_clues" MUST be a list of strings.
    - Each string must be exactly 1 sentence and end with a period.
    - There MUST be exactly one entry per clue, in the same order as the clues:
      - parsed_clues[0] is ONLY for Clue #1
      - parsed_clues[1] is ONLY for Clue #2
      - ...
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

    Semantics (DO NOT MIX THESE UP):
    - immediately_left_of(A,B): A is exactly 1 house left of B. ("directly left of", "immediately left of")
    - left_of(A,B): A is somewhere left of B (strictly smaller index). ("somewhere to the left of", "to the left of")
    - right_of(A,B): A is somewhere right of B. ("somewhere to the right of", "to the right of")
    - adjacent(A,B): houses differ by exactly 1. ("next to", "adjacent to")
    - same_house(A,B): A and B belong to the same person/house. ("is", "has", "the X is Y", "X is the Y")
    - between(A,B,K): exactly K houses strictly between A and B (K=1 => distance 2).

    CLUE-TO-DSL FIDELITY RULES (VERY IMPORTANT):
    - For each clue i, ONLY use entities that appear in that clue:
      - If clue i does NOT mention Arnold, do NOT include Name=Arnold in C<i>.
      - Do NOT swap in other values from the domain.
    - Preserve the relation type from the text:
      - "somewhere to the left" => left_of(...) (NOT immediately_left_of)
      - "directly/immediately left" => immediately_left_of(...)
      - "in the second house" => set(2,...)
      - "X is Y" / "The X is Y" => same_house(...)
    - parsed_clues are NOT a place for deductions:
      - Do NOT set a house number unless the clue explicitly provides it.
      - Do NOT invent intermediate constraints.
    - If a clue states an identity between two attributes (e.g., "The doctor is Eric"):
      - Use same_house(Occupation=doctor,Name=Eric) (or matching attribute names from the puzzle).
    - If a clue states a positional relation between two described entities:
      - Use left_of / immediately_left_of / right_of / adjacent / between with the two Attr=Val terms.

    SILENT SELF-CHECK (DO NOT OUTPUT THIS):
    Before finalizing "parsed_clues", re-read each original clue and confirm:
    - C<i> uses ONLY entities mentioned in clue i.
    - The predicate type matches the clue wording (left_of vs immediately_left_of etc.).
    - All Attr and Val tokens exist in attribute_values (after underscore normalization).
    If any of the above fails, fix the parsed clue.

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

EXAMPLE DEMONSTRATION (illustration only)

Example Puzzle:
    There are 3 houses, numbered 1 to 3 from left to right.
    Each person has a unique name: Peter, Eric, Arnold.
    Each person has a unique drink: tea, water, milk.

Clues:
    1. Peter is in the second house.
    2. Arnold is directly left of the one who drinks water.
    3. The water drinker is directly left of the milk drinker.

Correct Example Output:
<answer>{
  "n_houses": 3,
  "attribute_values": {
    "Name": ["Peter", "Eric", "Arnold"],
    "Drink": ["tea", "water", "milk"]
  },
  "parsed_clues": [
    "C1 = set(2,Name,Peter).",
    "C2 = immediately_left_of(Name=Arnold,Drink=water).",
    "C3 = immediately_left_of(Drink=water,Drink=milk)."
  ],
  "parsed_reasoning": [
    "S1 [C1] set(2,Name,Peter).",
    "S2 [C3] not(3,Drink,water).",
    "S3 [C3] not(1,Drink,milk).",
    "S4 [C2] not(3,Name,Arnold).",
    "S5 [C2+C3] set(1,Name,Arnold)."
  ],
  "solution": {
    "header": ["House", "Name", "Drink"],
    "rows": [
      ["1", "Arnold", "tea"],
      ["2", "Peter", "water"],
      ["3", "Eric", "milk"]
    ]
  }
}</answer>
"""

SOLUTION_PROMPT_1_SHOT_VERIFIER_USER_V2 = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

{puzzle}

Solve the puzzle above and provide parsed_reasoning parsed_clues and solution by returning ONLY the <answer>...</answer> block, with no additional text.
Previous correctly generated reasoning steps:

{reasoning_steps}
"""



SOLUTION_PROMPT_RESUME_WITH_VERIFIER_V1 = """You are an expert logic puzzle solver. You are provided with:
  (1) The original logic puzzle text.
  (2) A prior model attempt (Step-1) that produced:
      - n_houses, attribute_values, parsed_clues, parsed_reasoning, solution
  (3) Step-by-step Z3 verification feedback over the prior parsed_reasoning.

Your task is to:
  - Keep the domain and clue parsing consistent and ordered.
  - Treat Z3-VERIFIED (PASSED) reasoning steps as trusted facts.
  - Remove or correct FAILED reasoning steps.
  - Continue generating logically valid step-by-step deductions from the verified state.
  - Produce a correct final solution.
  - Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
  - Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
  - Do NOT include extra text, markdown, explanations, or code fences.
  - Inside <answer>...</answer>, the content MUST be a single valid JSON object.
  - The JSON object MUST have exactly FIVE top-level keys:
    "n_houses", "attribute_values", "parsed_clues", "parsed_reasoning", "solution".
  - Do NOT add any other keys (do NOT include verifier feedback in the output).

NORMALIZATION RULES (SAME AS STEP-1):
  - Use underscores instead of spaces in VALUES (e.g., grilled_cheese, root_beer, bmw_3_series).
  - Attribute names MUST match the puzzle text exactly (case-sensitive).
  - House numbers are integers 1..N.
  - Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6.
  - Do NOT invent values. Every <Val> must be one of the allowed values listed in the puzzle text (after normalization).

INPUTS YOU WILL RECEIVE (DO NOT OUTPUT THEM VERBATIM):

(1) PUZZLE TEXT:
{PUZZLE_TEXT}

(2) STEP-1 ATTEMPT (JSON):
{STEP1_JSON}

(3) Z3 STEP VERIFICATION FEEDBACK (JSON):
The verifier checks each reasoning step in order using parsed_clues + all previously ACCEPTED steps.
It provides:
  - passed_steps: list of reasoning step strings that were verified as entailed (trusted).
  - failed_steps: list of objects {{ "step": <reasoning_step_string>, "error": <short_reason> }}
  - optional: notes (string)

{VERIFIER_FEEDBACK_JSON}

IMPORTANT: HOW TO USE VERIFIER FEEDBACK
  - You MUST keep all PASSED steps as the foundation.
  - You MUST NOT include any FAILED step as-is.
  - Any later deductions that depend on a FAILED step must be re-derived without that invalid step.
  - If the verifier says a step FAILED, assume it is NOT entailed by the constraints and do not reuse it.
  - PASSED steps are considered logically sound under the current parsed_clues/domain.

D) DOMAIN OUTPUT (MANDATORY)
  - "n_houses" MUST match the puzzle N.
  - "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values.
  - Each attribute list MUST contain exactly N unique values (after normalization).
  - Keep "attribute_values" exactly as in STEP-1 unless STEP-1 clearly violated the puzzle text.
  - Do NOT add new attributes not in the puzzle.

A) parsed_clues (MANDATORY, PARSABLE, ORDERED, FAITHFUL)
  - "parsed_clues" MUST be a list of strings.
  - There MUST be exactly one entry per clue, in the same order as the clues.
  - Keep "parsed_clues" exactly as in STEP-1 unless STEP-1 clearly violated the clue text.
  - Each parsed clue MUST follow the same DSL format:

    C<i> = <predicate>.

  - Preserve relation types exactly (left_of vs immediately_left_of, etc.).
  - Do NOT add deductions into parsed_clues.

B) parsed_reasoning (MANDATORY, PARSABLE, VERIFIED-FIRST)
  - "parsed_reasoning" MUST be a list of strings.
  - Each string must be exactly 1 sentence and end with a period.
  - Each entry MUST follow this exact DSL format:

    S<k> [C<i>(+C<j>...)] <op>(<H>,<Attr>,<Val>).

  - <op> is either set or not.
  - <k> MUST start at 1 and increase by 1 each step (NO GAPS).

  VERIFIED-FIRST CONSTRUCTION RULE:
  - Begin your new "parsed_reasoning" by rewriting ALL verifier passed_steps in the SAME ORDER,
    but with step indices renumbered to be S1, S2, S3, ... (keep the rest identical).
  - Do NOT include any failed step.
  - After you place all passed steps, continue adding new steps S<k>+1 ... until the puzzle is solved.

  LOGICAL VALIDITY REQUIREMENT:
  - Every NEW step you add MUST be logically entailed by:
      parsed_clues + all earlier steps in this NEW parsed_reasoning list.
  - If you cannot deduce a set(...) fact with certainty, output a not(...) fact that is guaranteed true.
  - Do NOT invent probabilistic or "likely" steps.

C) solution (MANDATORY TABLE)
  - "solution" MUST be in tabular form with:
    - "header": a list of column names
    - "rows": a list of rows, each row being a list of strings matching the header order.
  - The header MUST include "House" and then all attribute columns from the puzzle text.
  - Rows MUST list houses in increasing order from 1..N.
  - All solution VALUES must be normalized with underscores.
  - The final solution MUST satisfy parsed_clues and MUST be consistent with all PASSED steps.

SILENT SELF-CHECK (DO NOT OUTPUT THIS):
  - Ensure parsed_clues remain ordered and faithful.
  - Ensure parsed_reasoning starts with all passed steps (renumbered), contains no failed steps,
    and every added step is entailed.
  - Ensure solution is complete and consistent.

Return ONLY the final JSON object wrapped in <answer>...</answer>.
"""