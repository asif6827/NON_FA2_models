# Prompt templates for puzzle solving system
# 
# This file contains all prompt templates used in the system, organized by type:
# 1. Solution-based verification prompts (from first original file)
# 2. Constraint-based verification prompts (from second original file)

# =============================================================================
# SOLUTION-BASED VERIFICATION PROMPTS
# =============================================================================

# --- Solution Generation Prompts (Solution-based) --- 
SOLUTION_PROMPT_SYSTEM_SOLUTION_BASED = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
1. Analyze the clues step by step.
2. Derive a correct final solution.
3. Return the result STRICTLY as a single valid JSON object.

CRITICAL FORMAT REQUIREMENTS:
- Output ONLY a JSON object, NO natural language, NO markdown, NO code fences.
- The top-level JSON MUST have exactly two keys: "reasoning" and "solution".
- "reasoning" MUST be a SHORT English explanation (1–5 sentences, not more).
- "solution" MUST be an object with:
  - "header": a list of column names (e.g. ["House", "Name", "Pet", "..."])
  - "rows": a list of rows, where each row is a list of strings, one per column.

Example of the REQUIRED SHAPE (this is ONLY an example, not the answer):

{
  "reasoning": "Your step-by-step logic here, but concise.",
  "solution": {
    "header": ["House", "Name", "Pet", "..."],
    "rows": [
      ["1", "Eric", "cat", "..."],
      ["2", "Arnold", "dog", "..."]
    ]
  }
}

Do NOT include any text before or after the JSON.
"""

SOLUTION_PROMPT_USER_SOLUTION_BASED = """PUZZLE:
{puzzle}

Please provide your reasoning and solution:"""

# --- Verification Prompts (Solution-based) --- 
VERIFICATION_PROMPT_SYSTEM_SOLUTION_BASED = """You are an expert logic puzzle solver. I need you to verify if a given solution satisfies all the clues in a logic puzzle."""

VERIFICATION_PROMPT_USER_SOLUTION_BASED = """Problem ID: {problem_id}

CLUES:
{clues_text}

PROPOSED SOLUTION:
{solution_text}

Please check if the proposed solution satisfies ALL the clues. For each clue, first reason about whether it is satisfied or violated by the solution, and then state your final answer.

Respond with a JSON object in the following format:
{{
  "clue_analysis": [
    {{ "clue_number": 1, "reasoning": "work out if clue is satisfied", "satisfied": true }},
    {{ "clue_number": 2, "reasoning": "work out if clue is satisfied", "satisfied": false }}
  ],
  "violated_clues": [1, 3],
  "all_clues_satisfied": false
}}
"""

# --- Refinement Prompts (Solution-based) --- 
REFINEMENT_PROMPT_SYSTEM_SOLUTION_BASED = """You are an expert logic puzzle solver. You are provided with a logic puzzle and a previous reasoning and solution to it that is wrong and violates some of the constraints.

Your task is to:

1. Carefully read the FEEDBACK_JSON about why the previous solution is wrong. This feedback is a single JSON object with the following fields:
   - "z3_analysis": the output of a Z3-based constraint checker, with keys such as "valid" (bool), "feedback" (string), and "issues" (list of strings).
   - "accuracy": the ground-truth accuracy information, with keys "score" (0–1 float), "correct_cells" (int), and "total_cells" (int).
   - "verification": LLM-based clue verification feedback, with keys "all_clues_satisfied" (bool or null), "violated_clues" (list of integers), and "raw" (the full JSON returned by the verification model, or null if parsing failed).

2. Based on this FEEDBACK_JSON, identify the specific errors in the previous reasoning and solution. In particular, you MUST:
   - Fix any structural issues (e.g., if an attribute 'Name' has illegal values ['Doctor', 'Engineer'] and the allowed values are ['Eric', 'Arnold', 'Alice', 'Peter'], you MUST only use allowed values for that attribute in your new solution).
   - Ensure that each attribute's values form a valid permutation of the allowed set (each value appears exactly once, no missing and no duplicates).
   - Respect the logical constraints from the clues, especially any clues that the verification feedback says are violated (those listed in "verification.violated_clues").
   - Try to increase the number of correct cells compared to the previous solution (using 'accuracy.score' and the cell counts as a guide).

3. Provide:
   (a) A brief analysis of what was wrong in the previous reasoning, explicitly referring to the FEEDBACK_JSON (Z3 structural issues, violated clues, and low accuracy).
   (b) A new reasoning that corrects these errors and leads to a better solution.
   (c) A NEW SOLUTION that:
       - Strictly follows the same table format as before.
       - Uses the SAME COLUMN NAMES as the previous solution.
       - Uses only allowed values for each attribute (as implied by the feedback).
       - Has the following JSON structure:

       "new_solution": {
         "header": ["House", "Name", "Pet", "..."],
         "rows": [
           ["1", "Eric", "cat", "..."],
           ["2", "Arnold", "dog", "..."],
           ["3", "Alice", "bird", "..."]
         ]
       }

CRITICAL FORMAT REQUIREMENTS:

- You MUST output ONLY a single JSON object, with EXACTLY the following top-level keys:
  { "previous_reasoning_error_analysis": "...", "new_reasoning": "...", "new_solution": { ... } }

- The "new_solution" object MUST directly contain:
  - "header": a list of column names (strings).
  - "rows": a list of rows, where each row is a list of strings, one per column.

- Do NOT wrap the new solution inside another "solution" field.
- Do NOT include any text before or after the JSON.
"""

REFINEMENT_PROMPT_USER_SOLUTION_BASED = """PUZZLE:
{puzzle}

PREVIOUS_REASONING:
{previous_reasoning}

PREVIOUS_SOLUTION:
{previous_solution}

FEEDBACK_JSON:
{failed_clues}

ANALYSIS_AND_NEW_REASONING_SOLUTION:"""

# =============================================================================
# CONSTRAINT-BASED VERIFICATION PROMPTS
# =============================================================================

# --- Solution Generation Prompts (Constraint-based) --- 
SOLUTION_PROMPT_SYSTEM_CONSTRAINT_BASED = """You are an expert logic puzzle solver and formalizer.

Your task:
1. Read the given logic puzzle carefully.
2. Extract META information from the puzzle text:
   - entity_count: number of houses/people (from the first sentence like "There are N houses")
   - attributes: list of distinct attributes (like name, pet, color, book, etc.)
   - attribute_values: dictionary mapping each attribute to its possible values (extracted from lines with values in backticks)
3. Use consistent attribute names throughout all constraints.
4. Convert ALL clues into a list of JSON constraints in a small DSL.
5. For EACH constraint, ADD a "source_clue" field indicating which clue it came from (e.g., source_clue: 3 for the 3rd clue).

CRITICAL RULES:
- Output MUST be a single valid JSON object with two keys: "meta" and "constraints".
- The "meta" key contains the extracted META information.
- The "constraints" key contains the list of constraint objects.
- DO NOT include any natural language explanation.
- DO NOT wrap the JSON in code fences.

META INFORMATION EXAMPLE:
{
  "entity_count": 5,
  "attributes": ["name", "pet", "color", "book"],
  "attribute_values": {
    "name": ["Alice", "Bob", "Charlie"],
    "pet": ["cat", "dog", "fish"],
    "color": ["red", "blue", "green"],
    "book": ["fiction", "non-fiction", "science"]
  }
}

DSL SCHEMA:

1) AttributeRef
- Used whenever you refer to a specific attribute-value pair.
- Format:
  { "attr": "<attribute_name>", "value": "<value_string>" }

2) Constraint objects:

(1) same_person:
- Meaning: the person with attribute A=value_a is the same person as the person with attribute B=value_b.
- Example: "Alice is the person who loves jazz music."
  ->
  {
    "op": "same_person",
    "a": { "attr": "name", "value": "Alice" },
    "b": { "attr": "music", "value": "jazz" },
    "source_clue": 3
  }

(2) house_is:
- Meaning: the person with attribute A=value_a is in a specific house index (1-based).
- Example: "The person who loves carnations is in the first house."
  ->
  {
    "op": "house_is",
    "a": { "attr": "flower", "value": "carnations" },
    "house": 1,
    "source_clue": 1
  }

(3) not_house:
- Meaning: the person with attribute A=value_a is NOT in a specific house.
- Example 1: "The person who loves beach vacations is not in the first house."
  ->
  {
    "op": "not_house",
    "a": { "attr": "vacation", "value": "beach" },
    "house": 1,
    "source_clue": 2
  }
- Example 2: "The person who owns a dog is not in the first house."
  ->
  {
    "op": "not_house",
    "a": { "attr": "pet", "value": "dog" },
    "house": 1,
    "source_clue": 3
  }

(4) left_of:
- Meaning: the person with A=value_a is somewhere to the LEFT of the person with B=value_b (not necessarily adjacent).
  {
    "op": "left_of",
    "a": { "attr": "book", "value": "romance" },
    "b": { "attr": "name", "value": "Alice" },
    "source_clue": 4
  }

(5) right_of:
- Meaning: the person with A=value_a is somewhere to the RIGHT of the person with B=value_b.
  {
    "op": "right_of",
    "a": { "attr": "name", "value": "Alice" },
    "b": { "attr": "name", "value": "Peter" },
    "source_clue": 5
  }

(6) next_to:
- Meaning: the person with A=value_a and the person with B=value_b are in adjacent houses.
  {
    "op": "next_to",
    "a": { "attr": "vacation", "value": "city" },
    "b": { "attr": "music", "value": "classical" },
    "source_clue": 6
  }

(7) distance:
- Meaning: the houses of A=value_a and B=value_b have exactly K houses in between.
- Example: "There is one house between the Dane and the pizza lover."
  ->
  {
    "op": "distance",
    "a": { "attr": "nationality", "value": "dane" },
    "b": { "attr": "food", "value": "pizza" },
    "distance": 1,
    "source_clue": 7
  }

IMPORTANT:
- Use consistent attribute names throughout all constraints.
- Houses are numbered from 1 (leftmost) to N (rightmost).
- Use as many constraints as necessary to encode ALL clues.
- Only use attribute names that you have extracted in the META information.
- Only use values that you have extracted in the META information.
- The "source_clue" field is REQUIRED for every constraint.
"""

SOLUTION_PROMPT_USER_CONSTRAINT_BASED = """PUZZLE:
{puzzle}

Please generate the META information and convert ALL clues of this puzzle into JSON constraints using the DSL.
Return ONLY the JSON object with "meta" and "constraints" keys.
"""

# --- Verification Prompts (Constraint-based) --- 
VERIFICATION_PROMPT_SYSTEM_CONSTRAINT_BASED = """You are an expert logic puzzle formalizer.

Your job in this task is NOT to directly solve the puzzle, but to ANALYZE a previously generated DSL specification for a Zebra-style puzzle and identify which constraints are correct or incorrect.

You are given:
1. The original puzzle text (natural language).
2. META information extracted from the puzzle.
3. A list of DSL constraints that are supposed to encode ALL clues. Each constraint has a "source_clue" field indicating which clue it came from (e.g., source_clue: 3 for the 3rd clue).
4. Optionally, a decoded solution table and/or a ground-truth solution table.

DSL SCHEMA (FULL DEFINITION):

1) AttributeRef
- Used whenever you refer to a specific attribute-value pair.
- Format:
  { "attr": "<attribute_name>", "value": "<value_string>" }

2) Constraint objects:

(1) same_person:
- Meaning: the person with attribute A=value_a is the same person as the person with attribute B=value_b.
- Example: "Alice is the person who loves jazz music."
  ->
  {
    "op": "same_person",
    "a": { "attr": "name", "value": "Alice" },
    "b": { "attr": "music", "value": "jazz" },
    "source_clue": 3
  }

(2) house_is:
- Meaning: the person with attribute A=value_a is in a specific house index (1-based).
- Example: "The person who loves carnations is in the first house."
  ->
  {
    "op": "house_is",
    "a": { "attr": "flower", "value": "carnations" },
    "house": 1,
    "source_clue": 1
  }

(3) not_house:
- Meaning: the person with attribute A=value_a is NOT in a specific house.
- Example 1: "The person who loves beach vacations is not in the first house."
  ->
  {
    "op": "not_house",
    "a": { "attr": "vacation", "value": "beach" },
    "house": 1,
    "source_clue": 2
  }
- Example 2: "The person who owns a dog is not in the first house."
  ->
  {
    "op": "not_house",
    "a": { "attr": "pet", "value": "dog" },
    "house": 1,
    "source_clue": 3
  }

(4) left_of:
- Meaning: the person with A=value_a is somewhere to the LEFT of the person with B=value_b (not necessarily adjacent).
  {
    "op": "left_of",
    "a": { "attr": "book", "value": "romance" },
    "b": { "attr": "name", "value": "Alice" },
    "source_clue": 4
  }

(5) right_of:
- Meaning: the person with A=value_a is somewhere to the RIGHT of the person with B=value_b.
  {
    "op": "right_of",
    "a": { "attr": "name", "value": "Alice" },
    "b": { "attr": "name", "value": "Peter" },
    "source_clue": 5
  }

(6) next_to:
- Meaning: the person with A=value_a and the person with B=value_b are in adjacent houses.
  {
    "op": "next_to",
    "a": { "attr": "vacation", "value": "city" },
    "b": { "attr": "music", "value": "classical" },
    "source_clue": 6
  }

(7) distance:
- Meaning: the houses of A=value_a and B=value_b have exactly K houses in between.
- Example: "There is one house between the Dane and the pizza lover."
  ->
  {
    "op": "distance",
    "a": { "attr": "nationality", "value": "dane" },
    "b": { "attr": "food", "value": "pizza" },
    "distance": 1,
    "source_clue": 7
  }

IMPORTANT DSL RULES:
- Use consistent attribute names throughout all constraints.
- Houses are numbered from 1 (leftmost) to N (rightmost).
- The "source_clue" field is REQUIRED for every constraint.

Your tasks:
1. Read the puzzle and META carefully to understand the intended structure (entities, attributes, values).
2. For EACH constraint:
   - Use the "source_clue" field to identify which clue this constraint is trying to encode
   - Locate the specific clue in the puzzle text that matches the source_clue number
   - Compare the constraint's logical meaning with the actual clue content
   - Decide whether it correctly encodes the core logic of the clue
   - Be VERY FLEXIBLE: only mark a constraint as invalid if it clearly misrepresents the clue's meaning
3. DO NOT generate a new full DSL here. Only diagnose correctness of the existing constraints.

OUTPUT FORMAT (CRITICAL):
- You MUST output a SINGLE valid JSON object with the following keys:
  - "constraint_analysis": a list of objects, one per constraint, in the SAME ORDER as the input list:
    [
      {
        "index": 0,
        "constraint": { ... the original DSL constraint ... },
        "valid": true/false,
        "error_type": "<short string, e.g. 'wrong_relation' or 'wrong_direction' or 'wrong_attribute' or 'ambiguous'>",
        "reason": "<short explanation in English>",
       },
      ...
    ]

"""

VERIFICATION_PROMPT_USER_CONSTRAINT_BASED = """Problem ID: {problem_id}

PUZZLE:
{puzzle_text}

META (already extracted):
{meta_json}

CURRENT DSL CONSTRAINTS (previous iteration):
{constraints_json}

OPTIONAL: CURRENT DECODED SOLUTION TABLE (may be empty if unsat):
{solution_text}

OPTIONAL: GROUND TRUTH SOLUTION TABLE (may be empty if not provided):
{ground_truth_text}

Please analyze EACH constraint in order and output a JSON object with the fields described in the system message.
"""

# --- Refinement Prompts (Constraint-based) --- 
REFINEMENT_PROMPT_SYSTEM_CONSTRAINT_BASED = """You are an expert logic puzzle formalizer.

You are given:
1. The original puzzle text (natural language).
2. META information for a Zebra-style puzzle (entity_count, attributes, attribute_values).
3. A previous DSL constraint list that tries to encode all clues.
4. A verification report that marks each constraint as valid/invalid and may provide suggested fixes.

Your task:
1. Use the verification report to decide which existing constraints can be kept as-is and which must be modified or discarded.
2. Fix as many issues mentioned in the verification report as possible.
3. Produce a NEW, self-consistent DSL specification that correctly encodes ALL clues of the puzzle.

CRITICAL OUTPUT FORMAT:
- You MUST output a SINGLE valid JSON object of the form:
  {
    "meta": { ... possibly updated META, but usually same as input ... },
    "constraints": [ ... a COMPLETE list of DSL constraints ... ]
  }

CONSTRAINT DSL SCHEMA (MUST FOLLOW STRICTLY):
- Each constraint is a JSON object with at least the key "op".
- "op" must be one of:
  "same_person", "house_is", "not_house", "left_of", "right_of", "next_to", "distance".
- AttributeRef objects must be of the form:
  { "attr": "<attribute_name>", "value": "<value_string>" }.
- Examples:
  {
    "op": "same_person",
    "a": { "attr": "child", "value": "Fred" },
    "b": { "attr": "sport", "value": "soccer" }
  }
  {
    "op": "house_is",
    "a": { "attr": "vacation", "value": "beach" },
    "house": 1
  }

RULES:
- Do NOT include any fields other than: "op", "a", "b", "house", "distance" inside a constraint object.
- Do NOT include any natural language outside of the JSON.
- The resulting constraints list must be sufficient for a Z3 solver to derive a UNIQUE solution.
"""

REFINEMENT_PROMPT_USER_CONSTRAINT_BASED = """PUZZLE:
{puzzle}

META (current):
{meta_json}

PREVIOUS DSL CONSTRAINTS:
{constraints_json}

VERIFICATION REPORT:
{verification_json}

Please generate a NEW, corrected DSL specification as a JSON object with "meta" and "constraints".
"""
