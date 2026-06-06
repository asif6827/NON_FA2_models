import re

import os
import sys
import time
import json
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from vllm import LLM, SamplingParams

from typing import Dict, List, Optional, Any, Tuple
from  z3 import Int, Solver, Distinct, And, Or, sat, unsat, unknown


#job_id = os.getenv("SLURM_JOB_ID")
#print("SLURM Job ID:", job_id)
# ==========================================
# 0. Global variables
# ==========================================
job_id = "Fake"
hp = True
panther = False
debug = True

n_samples=1
temperature=0.7
top_p=0.9
tokenizer_mode="auto"
limit=10
max_steps=10

hp_LLM_path = '/home/asif/data3/HF_cache/Qwen2.5-3B-Instruct'
hp_data_path = '/home/asif/data3/HF_cache/ZebraLogic/Zebra_Puzzle_small_320.json'
hp_out_path = '/home/asif/data3/Codes_QCRI/Reasoning360/Prompts_Results/ZebraPuzzle_1000_debug/'

panther_LLM_path = '/export/home/asifali/HF_cache/Qwen2.5-3B-Instruct'
panther_data_path = '/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_small_320.json'
panther_out_path = '/export/home/asifali/Reasoning360/Prompts_Results/ZebraPuzzle_1000_debug/'




# Z3 Constraint Checker Class
class Z3ConstraintChecker:
    def __init__(self):
        self.solver = Solver()
        self.constraints = []
        self.var_map: Dict[str, Dict[str, Any]] = {}
        self.attr_values: Dict[str, List[str]] = {}
        self.house_count: int = 0
        self.attributes: List[str] = []

    def set_house_count(self, count: int):
        self.house_count = count

    def set_attributes(self, attributes: List[str]):
        self.attributes = attributes

    def set_attribute_values(self, attr_values: Dict[str, List[str]]):
        self.attr_values = attr_values
        self.var_map = {}
        for attr, values in attr_values.items():
            self.var_map[attr] = {}
            for val in values:
                self.var_map[attr][val] = Int(f"{attr}_{val}")

    def add_base_constraints(self):
        for attr, var_dict in self.var_map.items():
            for val, var in var_dict.items():
                self.solver.add(And(var >= 1, var <= self.house_count))

        for attr, var_dict in self.var_map.items():
            if var_dict:
                self.solver.add(Distinct(list(var_dict.values())))

    # def add_constraint(self, constraint_type: str, *args, **kwargs):
    #     if constraint_type == "same_house":
    #         attr1, val1, attr2, val2 = args
    #         self.solver.add(self.var_map[attr1][val1] == self.var_map[attr2][val2])
    #     elif constraint_type == "left_of":
    #         attr1, val1, attr2, val2 = args
    #         self.solver.add(self.var_map[attr1][val1] < self.var_map[attr2][val2])
    #     elif constraint_type == "next_to":

    #         attr1, val1, attr2, val2 = args
    #         self.solver.add(
    #             Or(
    #                 self.var_map[attr1][val1] == self.var_map[attr2][val2] + 1,
    #                 self.var_map[attr1][val1] == self.var_map[attr2][val2] - 1
    #             )
    #         )
    #     elif constraint_type == "not_same_house":

    #         attr1, val1, attr2, val2 = args
    #         self.solver.add(self.var_map[attr1][val1] != self.var_map[attr2][val2])
    #     elif constraint_type == "house_is":

    #         attr, val, house = args
    #         self.solver.add(self.var_map[attr][val] == house)
    #     elif constraint_type == "not_house":

    #         attr, val, house = args
    #         self.solver.add(self.var_map[attr][val] != house)

    def check_solution(self, solution: Dict[str, Any]) -> Tuple[bool, str]:

        temp_solver = Solver()
        temp_solver.add(self.solver.assertions())

        header = solution.get("header", [])
        rows = solution.get("rows", [])

        if not header or not rows:
            return False, "Solution missing header or rows"

        house_mapping: Dict[int, Dict[str, str]] = {}
        for row in rows:
            if len(row) < len(header):
                continue
            house_num = row[0]
            try:
                house_idx = int(house_num)
            except Exception:
                continue
            house_mapping[house_idx] = {}
            for i, attr in enumerate(header[1:], 1):
                house_mapping[house_idx][attr] = row[i]

        for house_idx, attr_values in house_mapping.items():
            for attr, val in attr_values.items():
                if attr in self.var_map and val in self.var_map[attr]:
                    temp_solver.add(self.var_map[attr][val] == house_idx)

        result = temp_solver.check()
        if result == sat:
            return True, "All Z3 constraints are satisfied"
        else:
            return False, self._generate_feedback(temp_solver, solution)

    def _generate_feedback(self, solver, solution: Dict[str, Any]) -> str:
        feedback = "Constraint violations found by Z3.\n"

        if solver.check() == unsat:
            core = solver.unsat_core()
            if core:
                feedback += "Unsatisfiable core constraints:\n"
                for i, constraint in enumerate(core):
                    feedback += f"{i+1}. {constraint}\n"

        return feedback

    def analyze_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:

        header = solution.get("header", [])
        rows = solution.get("rows", [])

        issues: List[str] = []

        if not header or not rows:
            feedback = "Solution missing header or rows"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": ["Missing header or rows"]
            }

        expected_len = len(header)

        # 1)
        for i, row in enumerate(rows):
            if len(row) != expected_len:
                feedback = f"Row {i+1} has incorrect length. Expected {expected_len}, got {len(row)}"
                return {
                    "valid": False,
                    "feedback": feedback,
                    "issues": [feedback]
                }

        # 2)
        try:
            house_nums = [int(row[0]) for row in rows]
        except Exception:
            feedback = "House numbers (first column) contain non-integer values"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": [feedback]
            }

        if sorted(house_nums) != list(range(1, len(rows) + 1)):
            feedback = "House numbers are not consecutive starting from 1"
            return {
                "valid": False,
                "feedback": feedback,
                "issues": [feedback]
            }

        # 3) Z3
        z3_valid, z3_feedback = self.check_solution(solution)

        # 4)
        attributes_in_header = header[1:]  # House

        #  attribute -> (value -> [houses])
        for attr_idx, attr in enumerate(attributes_in_header, start=1):
            seen_vals: Dict[str, List[int]] = {}
            for row in rows:
                try:
                    house_idx = int(row[0])
                except Exception:
                    continue
                if len(row) <= attr_idx:
                    continue
                val = row[attr_idx]
                seen_vals.setdefault(val, []).append(house_idx)

            # a)  house
            for val, houses in seen_vals.items():
                if len(houses) > 1:
                    issues.append(
                        f"Value '{val}' of attribute '{attr}' appears in multiple houses: {houses}"
                    )

            # b)   ground_truth
            expected_vals = self.attr_values.get(attr)
            if expected_vals:
                #
                missing = [v for v in expected_vals if v not in seen_vals]
                if missing:
                    issues.append(
                        f"Attribute '{attr}' is missing values: {missing}"
                    )
                #
                illegal = [v for v in seen_vals if v not in expected_vals]
                if illegal:
                    issues.append(
                        f"Attribute '{attr}' has illegal values: {illegal} (expected: {expected_vals})"
                    )


        valid = z3_valid and not issues

        if issues:
            feedback = "; ".join(issues)
            if z3_feedback and z3_feedback != "All Z3 constraints are satisfied":
                feedback = z3_feedback + " " + feedback
        else:

            feedback = z3_feedback or "All constraints are satisfied"

        return {
            "valid": valid,
            "feedback": feedback,
            "issues": issues
        }



def extract_solution_info(solution: Dict[str, Any]) -> Tuple[List[str], Dict[str, List[str]]]:

    header = solution.get("header", [])
    rows = solution.get("rows", [])

    attributes = header[1:] if header else []
    attr_values: Dict[str, Any] = {attr: set() for attr in attributes}

    for row in rows:
        if len(row) >= len(header):
            for i, attr in enumerate(attributes):
                attr_values[attr].add(row[i + 1])

    for attr in attr_values:
        attr_values[attr] = list(attr_values[attr])

    return attributes, attr_values


# ==========================================
# 1.  (PROMPTS)
# ==========================================

# --- SolutionPrompt ---
# Straightforward initial solution generation
SolutionPrompt_System = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

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



# --- Prompt Decompose ---
# Straightforward Prompt Decomposition
DecomposePrompt_System = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
1. Analyze the clues step by step.
2. Derive all attributes, and their possible values.
3. Return the result STRICTLY as a single valid JSON object.

CRITICAL FORMAT REQUIREMENTS:
- Output ONLY a JSON object, NO natural language, NO markdown, NO code fences.
- The top-level JSON MUST have exactly three keys: "reasoning" and "solution".
- "attributes" a list of all attribute names.
- "attribute_values": an object mapping each attribute name to its list of possible values.
- Every attribute listed in "attributes" MUST appear as a key inside "attribute_values".
- Every value MUST be a string.
- No additional fields are allowed.

Example of the REQUIRED SHAPE (this is ONLY an example, not the answer):

{
  "attributes" : ["Name", "Pet", "..."],
  "attribute_values": {
    "Name": ["Eric", "Arnold", "..."],
    "Pet": ["cat", "dog", "..."],
    "..."
  }
}

Do NOT include any text before or after the JSON.
"""



DecomposePrompt_User = """PUZZLE:
{puzzle}

Please provide attributes and attribute_values:"""




SolutionPrompt_User = """PUZZLE:
{puzzle}

Please provide your reasoning and solution:"""


# --- VerificationPrompt ---
# Based on the text provided: Verifies if clues are satisfied using LLM reasoning.
VerificationPrompt_System = """You are an expert logic puzzle solver. I need you to verify if a given solution satisfies all the clues in a logic puzzle."""

VerificationPrompt_User = """Problem ID: {problem_id}

CLUES:
{clues_text}

PROPOSED SOLUTION:
{solution_text}

Please check if the proposed solution satisfies ALL the clues. For each clue, first reason about whether it is satisfied or violated by the solution, and then state your final answer.

Respond with a JSON object in the following format:
{
"clue_analysis": [
{ "clue_number": 1, "reasoning": "work out if clue is satisfied", "satisfied": true/false },
{ "clue_number": 2, "reasoning": "work out if clue is satisfied", "satisfied": true/false }
],
"violated_clues": [list of clue numbers that are violated, e.g., [1, 3]],
"all_clues_satisfied": true/false
}
"""


# --- RefinementPrompt ---
# Based on the text provided: Takes failed clues and previous reasoning to generate a fix.
RefinementPrompt_System = """You are an expert logic puzzle solver. You are provided with a logic puzzle and a previous reasoning and solution to it that is wrong and violates some of the constraints.

Your task is to:

1. Carefully read the FEEDBACK about why the previous solution is wrong. This feedback includes:
   - Structural / constraint feedback from a Z3-based constraint checker (e.g., duplicate or missing values, illegal values that are not allowed for a given attribute).
   - The ground-truth accuracy (how many cells are currently correct).

2. Based on this feedback, identify the specific errors in the previous reasoning and solution. In particular, you MUST:
   - Fix any structural issues (e.g., if an attribute 'Name' has illegal values ['Doctor', 'Engineer'] and the allowed values are ['Eric', 'Arnold', 'Alice', 'Peter'], you MUST only use allowed values for that attribute in your new solution).
   - Ensure that each attribute's values form a valid permutation of the allowed set (each value appears exactly once, no missing and no duplicates).
   - Try to increase the number of correct cells compared to the previous solution.

3. Provide:
   (a) A brief analysis of what was wrong in the previous reasoning.
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

RefinementPrompt_User = """PUZZLE:
{puzzle}

PREVIOUS_REASONING:
{previous_reasoning}

PREVIOUS_SOLUTION:
{previous_solution}

FAILED_CLUES:
{failed_clues}

ANALYSIS_AND_NEW_REASONING_SOLUTION:"""


class ZebraVerificationSystem:
    def __init__(self, model_path: str, max_attempts: int, temperature: float, top_p: float, tokenizer_mode: str = "auto"):
        self.max_attempts = max_attempts
        print(f"[System] Loading local model from: {model_path}")
        print(f"[System] Config: Max Attempts={max_attempts}, Temp={temperature}, Top_P={top_p}")

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=28672,
            stop=None
        )

    def _interpret_verification_result(self, ver_json: Dict[str, Any]) -> Tuple[bool, List[int]]:
        if not ver_json or not isinstance(ver_json, dict):
            return False, []

        clue_analysis = ver_json.get("clue_analysis")
        violated_clues = ver_json.get("violated_clues")
        all_flag = bool(ver_json.get("all_clues_satisfied", False))

        if isinstance(clue_analysis, list) and len(clue_analysis) > 0:
            violated = []
            for entry in clue_analysis:
                raw_sat = entry.get("satisfied", False)
                if isinstance(raw_sat, bool):
                    sat = raw_sat
                elif isinstance(raw_sat, (int, float)):
                    sat = (raw_sat != 0)
                elif isinstance(raw_sat, str):
                    sat = raw_sat.strip().lower() in ["true", "yes", "y", "1"]
                else:
                    sat = False
                if not sat:
                    num = entry.get("clue_number")
                    if isinstance(num, int):
                        violated.append(num)
            per_all_ok = (len(violated) == 0)

            is_verified = per_all_ok

            if not isinstance(violated_clues, list) or len(violated_clues) == 0:
                violated_clues = violated
        else:
            is_verified = all_flag
            if not isinstance(violated_clues, list):
                violated_clues = []

        return is_verified, violated_clues

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Robust JSON extraction handling markdown blocks and raw text"""
        if not text:
            return None
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = text[start:end + 1]
                return json.loads(json_str)
        except Exception:
            pass
        return None


    def _normalize_atom(self, x: Any) -> str:
        s = str(x).strip().lower()
        return s

    def _try_extract_grid(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "header" in data and "rows" in data:
            return data

        sol = data.get("solution")
        if isinstance(sol, dict) and "header" in sol and "rows" in sol:
            return sol

        return data

    def _normalize_grid(self, data: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(data, dict):
            return None
        if "header" not in data or "rows" not in data:
            return None

        try:
            header = [self._normalize_atom(h) for h in data["header"]]

            ignore_cols = {"house"}
            keep_indices = [i for i, h in enumerate(header) if h not in ignore_cols]
            if not keep_indices:
                return None  #

            rows_norm = []
            for row in data["rows"]:
                row_norm_full = [self._normalize_atom(x) for x in row]
                row_norm = [row_norm_full[i] for i in keep_indices]
                rows_norm.append(row_norm)

            header_kept = [header[i] for i in keep_indices]
            rows_norm_sorted = sorted(rows_norm)
            return {"header": header_kept, "rows": rows_norm_sorted}
        except Exception:
            return None

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._normalize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._normalize_value(item) for item in value]
        else:
            return self._normalize_atom(value)

    def _score_with_ground_truth(self, current_solution: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        try:
            gt_for_grid = self._try_extract_grid(ground_truth)
            cur_for_grid = self._try_extract_grid(current_solution)
            norm_gt = self._normalize_grid(gt_for_grid)
            norm_cur = self._normalize_grid(cur_for_grid)
            if norm_gt is not None and norm_cur is not None:
                rows_gt = norm_gt["rows"]
                rows_cur = norm_cur["rows"]
                cols = len(norm_gt["header"]) if isinstance(norm_gt.get("header"), list) else 0
                total_rows = len(rows_gt)
                total_cells = total_rows * cols
                correct = 0
                if total_cells > 0:
                    for i in range(total_rows):
                        gt_row = rows_gt[i] if i < len(rows_gt) else []
                        cur_row = rows_cur[i] if i < len(rows_cur) else []
                        for j in range(cols):
                            gt_val = gt_row[j] if j < len(gt_row) else None
                            cur_val = cur_row[j] if j < len(cur_row) else None
                            if gt_val == cur_val:
                                correct += 1
                    score = correct / total_cells
                else:
                    score = 0.0
                return {"score": score, "correct_cells": correct, "total_cells": total_cells}
            normalized_gt = self._normalize_value(ground_truth)
            normalized_sol = self._normalize_value(current_solution)
            ok = 1.0 if normalized_gt == normalized_sol else 0.0
            return {"score": ok, "correct_cells": int(ok), "total_cells": 1}
        except Exception:
            return {"score": 0.0, "correct_cells": 0, "total_cells": 0}

    def _check_with_ground_truth(
        self,
        current_solution: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> Optional[bool]:

        try:
            gt_for_grid = self._try_extract_grid(ground_truth)
            cur_for_grid = self._try_extract_grid(current_solution)

            norm_gt = self._normalize_grid(gt_for_grid)
            norm_cur = self._normalize_grid(cur_for_grid)

            if norm_gt is not None and norm_cur is not None:
                return norm_gt == norm_cur

            normalized_gt = self._normalize_value(ground_truth)
            normalized_sol = self._normalize_value(current_solution)
            return normalized_gt == normalized_sol

        except Exception:
            return None

    def solve_puzzle(self, problem_id: str, puzzle_text: str, ground_truth: Optional[Dict[str, Any]] = None, n_samples: int = 1) -> Dict[str, Any]:
        start_time = time.time()
        all_samples_results: List[Dict[str, Any]] = []

        # Initialize Z3 constraint checker for each sample
        for sample_idx in range(n_samples):
            print(f"  > [Sample {sample_idx+1}/{n_samples}] Generating solution using SolutionPrompt...")
            trace_steps: List[Dict[str, Any]] = []

            # Initialize Z3 constraint checker
            z3_checker = Z3ConstraintChecker()
            if debug:
                print("\n" * 5)
                print("Z3_checker = {}".format(z3_checker))

            # Extract attributes and values from ground truth if available
            if ground_truth:
                gt_solution = self._try_extract_grid(ground_truth)
                if debug:
                    print("\n" * 5)
                    print("GT solution = {}".format(gt_solution))
                if gt_solution:
                    attributes, attr_values = extract_solution_info(gt_solution)
                    if debug:
                        print("\n" * 5)
                        print("Attributes = {}".format(attributes))
                        print("\n" * 5)
                        print("Attr Values = {}".format(attr_values))
                    if attributes and attr_values:
                        # Determine house count from ground truth
                        house_count = len(gt_solution.get("rows", []))
                        z3_checker.set_house_count(house_count)
                        z3_checker.set_attributes(attributes)
                        z3_checker.set_attribute_values(attr_values)
                        z3_checker.add_base_constraints()

            user_prompt_1 = SolutionPrompt_User.format(puzzle=puzzle_text)
            if debug:
                print("\n" * 5)
                print("SolutionPrompt_System = {}".format(SolutionPrompt_System))
                print("\n" * 5)
                print("User_prompt_1 = {}".format(user_prompt_1))
                print("\n" * 5)

            response_1 = {
                "reasoning": "From clue 1, Eric is to the left of Arnold. From clue 2, the dog owner cannot be in the first house. Therefore, the dog owner must be in the second house, and Eric must be in the first house. Since Eric is in the first house, the dog must be in the second house, which is Arnold's house.",
                "solution": {
                    "header": ["House", "Name", "Pet"],
                    "rows": [
                    ["1", "Eric", "dog"],
                    ["2", "Arnold", "cat"]
                    ]
                  }
                }
 # ()._generate(SolutionPrompt_System, user_prompt_1)
            if debug:
                print("\n" * 5)
                print("Response_1 = {}".format(response_1))
                print("\n" * 5)
            parsed_1 = self._extract_json(response_1)
            if debug:
                print("\n" * 5)
                print("Parsed_1 = {}".format(parsed_1))
                print("\n" * 5)

            current_solution: Dict[str, Any] = {}
            current_reasoning: str = response_1

            if isinstance(parsed_1, dict):
                # reasoning
                if isinstance(parsed_1.get("reasoning"), str):
                    current_reasoning = parsed_1["reasoning"]
                sol = parsed_1.get("solution")
                if isinstance(sol, dict) and len(sol) > 0:
                    current_solution = sol

                if debug:
                    print("\n" * 5)
                    print("current_reasoning = {}".format(current_reasoning))
                    print("\n" * 5)
                    print("current_solution= {}".format(current_solution))

            trace_steps.append({
                "stage": "SolutionPrompt",
                "user_prompt": user_prompt_1,
                "response": response_1,
                "parsed_json": parsed_1
            })

            if not current_solution:
                if debug:
                    print("\n" * 5)
                    print("No solution found returning Loop outside for max_steps.")
                sample_result = {
                    "sample_idx": sample_idx,
                    "status": "no_initial_solution",
                    "final_solution": current_solution,
                    "attempts_used": 0,
                    "trace": trace_steps,
                    "round_scores": [],
                    "best_score": 0.0,
                    "is_correct": False
                }
                all_samples_results.append(sample_result)
                continue

            status = "max_attempts_reached"
            final_solution: Dict[str, Any] = current_solution

            if debug:
                print("\n" * 5)
                print("Entering Max Attempts Loop")
                print("\n" * 5)

            round_scores: List[float] = []

            for i in range(self.max_attempts):
                if debug:
                    print("\n" * 5)
                    print("Current iteration No= {}/{}".format(i + 1, self.max_attempts))
                hard_verified: Optional[bool] = None
                hard_score_info: Optional[Dict[str, Any]] = None

                # Perform Z3 constraint checking
                if debug:
                    print("Perform Z3 Constraint Check in round {}".format(i+1))
                z3_feedback = ""
                z3_valid = False
                if hasattr(z3_checker, 'house_count') and z3_checker.house_count > 0:
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Checking solution with Z3 constraint checker...")
                    if debug:
                        print("\n" * 5)
                        print("Analyzing current solution with z3, idx = {}, n_sample = {}; Solution= {}".format(i+1, sample_idx, current_solution))
                    z3_analysis = z3_checker.analyze_solution(current_solution)
                    if debug:
                        print("\n" * 5)
                        print("Iter = {}; Sample = {}; Z3 Analysis = {}".format(i+1, sample_idx, z3_analysis))

                    z3_valid = z3_analysis.get("valid", False)
                    z3_feedback = z3_analysis.get("feedback", "")
                    trace_steps.append({
                        "stage": f"Z3Check_Attempt_{i+1}",
                        "z3_analysis": z3_analysis
                    })

                    if z3_valid:
                        print(f"    ✓ Z3: All constraints satisfied")
                    else:
                        print(f"    ✗ Z3: Constraint violations / issues found: {z3_feedback[:200]}...")

                # Ground truth & Accuracy
                if ground_truth is not None:
                    if debug:
                        print("\n" * 5)
                        print("Checking solution against ground truth, idx = {}, n_sample = {}".format(i+1, sample_idx))
                        print("\n" * 5)
                        print("Ground truth = {}".format(ground_truth))
                        print("\n" * 5)
                        print("Current solution = {}".format(current_solution))
                        print("\n" * 5)

                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Verifying using ground_truth...")
                    hard_score_info = self._score_with_ground_truth(current_solution, ground_truth)
                    hard_verified = hard_score_info["score"] == 1.0
                    trace_steps.append({
                        "stage": f"GroundTruthCheck_Attempt_{i+1}",
                        "score_info": hard_score_info
                    })
                    print(f"    Accuracy: {hard_score_info['correct_cells']}/{hard_score_info['total_cells']} = {hard_score_info['score']:.4f}")
                    round_scores.append(hard_score_info["score"])

                    if hard_verified:
                        if debug:
                            print("\n" * 5)
                            print("Hard Verified = True at sample = {}; Iteration={}/{}".format(sample_idx, i + 1, self.max_attempts))
                        status = "success"
                        final_solution = current_solution
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Verified by ground_truth. All attributes correct.")
                        break
                    else:
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Ground Truth] Solution does not match ground_truth (score: {hard_score_info['score']:.4f}).")
                else:
                    if debug:
                        print('\n' * 5)
                        print('Ground Truth is None only checking Z3_valid..!')
                    # ground_truth;  Z3
                    hard_score_info = {"score": 0.0, "correct_cells": 0, "total_cells": 0}
                    round_scores.append(0.0)
                    if z3_valid:
                        status = "success"
                        final_solution = current_solution
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Z3 constraints satisfied (no ground_truth).")
                        break
                    else:
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Z3 Only] Solution violates constraints. Refinement will try to fix it.")

                if i < self.max_attempts - 1:
                    if debug:
                        print("\n" * 5)
                        print(f"  > Start Verifying the puzzle at [Attempt {i + 1}, Sample idx = {sample_idx}] ...!")
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Solution incorrect. Refining...")

                    # Prepare Z3 feedback + accuracy refinement prompt
                    z3_feedback_text = ""
                    if z3_feedback:
                        z3_feedback_text = f"Z3 STRUCTURAL / CONSTRAINT FEEDBACK:\n{z3_feedback}\n\n"

                    if hard_score_info is not None:
                        accuracy_text = (
                            f"Ground truth accuracy: {hard_score_info['score']:.4f} "
                            f"({hard_score_info['correct_cells']}/{hard_score_info['total_cells']} cells correct)."
                        )
                    else:
                        accuracy_text = "Ground truth accuracy is not available."

                    ref_user_prompt = RefinementPrompt_User.format(
                        puzzle=puzzle_text,
                        previous_reasoning=current_reasoning,
                        previous_solution=json.dumps(current_solution),
                        failed_clues=f"{z3_feedback_text}{accuracy_text}"
                    )
                    if debug:
                        print("\n" * 5)
                        print("Attempt = {}, RefinementPrompt_System = {}".format(i + 1, RefinementPrompt_System))
                        print("\n" * 5)
                        print("Attempt = {}, refine_user_Prompt = {}".format(i + 1, ref_user_prompt))

                    ref_response = "" #self._generate(RefinementPrompt_System, ref_user_prompt)
                    if debug:
                        print("\n" * 5)
                        print("Attempt = {}, ref_response = {}".format(i + 1, ref_response))
                    ref_json = self._extract_json(ref_response)
                    if debug:
                        print("\n" * 5)
                        print("Attempt = {}, ref_json = {}".format(i + 1, ref_json))

                    trace_steps.append({
                        "stage": f"RefinementPrompt_Attempt_{i+1}",
                        "user_prompt": ref_user_prompt,
                        "response": ref_response,
                        "parsed_json": ref_json
                    })

                    if not ref_json:
                        print(f"    [Sample {sample_idx+1}/{n_samples}] [Warning] Failed to parse Refinement JSON at all. Loop continues with old solution.")
                    else:
                        # 1) new_reasoning
                        new_reas = ref_json.get("new_reasoning")
                        if isinstance(new_reas, str) and new_reas.strip():
                            current_reasoning = new_reas

                        # 2)  new_solution
                        raw_new_sol = ref_json.get("new_solution")
                        candidate_grid = None

                        if debug:
                            print("\n" * 5)
                            print("Attempt = {}, new_sol = {}".format(i + 1, raw_new_sol))
                            print("\n" * 5)
                            print("Attempt = {}, new_reas = {}".format(i + 1, new_reas))



                        if isinstance(raw_new_sol, dict):
                            # a)  header/rows
                            if "header" in raw_new_sol and "rows" in raw_new_sol:
                                candidate_grid = raw_new_sol
                            # b) {"solution": {...}}
                            elif "solution" in raw_new_sol and isinstance(raw_new_sol["solution"], dict):
                                inner = raw_new_sol["solution"]
                                if "header" in inner and "rows" in inner:
                                    candidate_grid = inner
                        else:
                            # c)  new_solution，"solution"
                            if "solution" in ref_json and isinstance(ref_json["solution"], dict):
                                inner = ref_json["solution"]
                                if "header" in inner and "rows" in inner:
                                    candidate_grid = inner

                        if candidate_grid is not None:
                            current_solution = candidate_grid
                            final_solution = candidate_grid
                            print(f"    [Sample {sample_idx+1}/{n_samples}] [Info] Updated solution from refinement.")
                        else:
                            # print
                            print(f"    [Sample {sample_idx+1}/{n_samples}] [Warning] Refinement JSON parsed but no valid 'new_solution' grid found. Keeping previous solution.")
                else:
                    # print a log
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Reached max attempts without success.")
                    break

            attempts_used = i + 1 if 'i' in locals() else 0
            best_score = max(round_scores) if round_scores else 0.0
            is_correct = best_score == 1.0

            sample_result = {
                "sample_idx": sample_idx,
                "status": status,
                "final_solution": final_solution,
                "attempts_used": attempts_used,
                "trace": trace_steps,
                "round_scores": round_scores,
                "best_score": best_score,
                "is_correct": is_correct
            }
            all_samples_results.append(sample_result)

        # Calculate correct samples count
        correct_samples = sum(1 for sample in all_samples_results if sample["is_correct"])

        # Calculate pass@k metrics using standard definition
        pass_at_k = {}
        for k in range(1, n_samples + 1):
            # Pass @ k
            has_correct = any(sample["is_correct"] for sample in all_samples_results[:k])
            pass_at_k[k] = 1 if has_correct else 0

        # Determine overall status
        overall_status = "success" if correct_samples >= 1 else "max_attempts_reached"

        return {
            "id": problem_id,
            "puzzle_snippet": puzzle_text,
            "status": overall_status,
            "attempts_used": max(sample["attempts_used"] for sample in all_samples_results),
            "all_samples_results": all_samples_results,
            "correct_samples": correct_samples,
            "total_samples": n_samples,
            "pass_at_k": pass_at_k,
            "round_scores": [sample["best_score"] for sample in all_samples_results],
            "best_score": max(sample["best_score"] for sample in all_samples_results)
        }


def load_local_dataset(file_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    print(f"[Data] Loading data from {file_path}...")
    try:
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
            data = df.to_dict('records')
        else:
            print(f"[Warning] Unsupported file format: {os.path.splitext(file_path)[1]}")
            return []
        print(f"[Data] Loaded {len(data)} samples.")
        return data
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return []


def get_puzzle_text(item: Dict[str, Any]) -> str:
    puzzle = item.get("puzzle")
    if isinstance(puzzle, str):
        return puzzle
    return ""


def main():
    parser = argparse.ArgumentParser()
    if hp:
        parser.add_argument("--model_path", type=str, default=hp_LLM_path, help="LLM to use")
        parser.add_argument("--data_path", type=str, default=hp_data_path, help=" Use (.json, .jsonl, .parquet)")
        parser.add_argument("--output_dir", type=str, default=hp_out_path, help="Output directory")
    elif panther:
        parser.add_argument("--model_path", type=str, default=panther_LLM_path, help="LLM to use")
        parser.add_argument("--data_path", type=str,   default=panther_data_path, help=" Use (.json, .jsonl, .parquet)")
        parser.add_argument("--output_dir", type=str,  default=panther_out_path, help="Output directory")

    parser.add_argument("--n_samples", type=int, default=n_samples, help="Number of samples per prompt (default 4 for zebra_puzzle)")
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--top_p", type=float, default=top_p)
    parser.add_argument("--tokenizer_mode", type=str, default=tokenizer_mode)
    parser.add_argument("--limit", type=int, default=limit, help="Only the first K data , -1 mean the full data")
    parser.add_argument("--max_steps", type=int, default=max_steps, help="Number of steps for Prompting rounds")

    args = parser.parse_args()
    print("arguments = {}".format(args))

    os.makedirs(args.output_dir, exist_ok=True)

    if args.temperature == 0.0:
        assert args.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert args.n_samples >= 1, "n_samples should always >= 1"

    dataset = load_local_dataset(args.data_path)
    if not dataset:
        return
    if args.limit > 0:
        dataset = dataset[:args.limit]


    try:
        system = ZebraVerificationSystem(
            model_path=args.model_path,
            max_attempts=args.max_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer_mode=args.tokenizer_mode
        )
    except Exception:
        sys.exit(1)

    print(f"\n" + "=" * 50)
    print(f"[System] Evaluating with max_n={args.max_steps}")
    print("=" * 50)

    all_results: List[Dict[str, Any]] = []
    print(f"[System] Starting Verification Loop...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    limit_str = f"_{args.limit}" if args.limit > 0 else "_full"
    output_file = os.path.join(args.output_dir, f"results_all_z3_small_n_{timestamp}_jobid_{job_id}_limit_{limit_str}.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing puzzles"):

            if debug:
                print("\n" * 5)
                print("=" * 50)
                print("Processing data point = {}".format(idx))
                print("=" * 50)
                print("\n" * 5)

            pid = item.get("id", f"sample_{idx}")
            puzzle_text = get_puzzle_text(item)
            if not puzzle_text:
                continue

            # ground_truth
            ground_truth = item.get("solution")

            tqdm.write(f"Processing idx = {idx}, pid = {pid} ... ")
            result = system.solve_puzzle(pid, puzzle_text, ground_truth=ground_truth, n_samples=args.n_samples)
            tqdm.write(f"Status: {result['status']}, Attempts: {result['attempts_used']}, Correct Samples: {result['correct_samples']}/{result['total_samples']}")
            all_results.append(result)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    print(f"[System] All results saved to {output_file}")

    print("\n" + "=" * 60)
    print("ALL N VALUES SUMMARY")
    print("=" * 60)

    n_stats = {}
    for n in range(0, args.max_steps + 1):
        n_stats[n] = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "total_score": 0.0
        }

    # Calculate pass@k statistics
    pass_at_k_stats = {}
    max_k = args.n_samples
    for k in range(1, max_k + 1):
        pass_at_k_stats[k] = {
            "total_tasks": 0,
            "pass_tasks": 0
        }

    for result in all_results:
        sample_results = result.get("all_samples_results", [])

        sample_scores = []
        for s in sample_results:
            scores = s.get("round_scores", [])

            if not isinstance(scores, list):
                scores = []
            sample_scores.append(scores)

        for n in range(0, args.max_steps + 1):
            n_stats[n]["total_tasks"] += 1
            success_n = False

            if n == 0:
                for scores in sample_scores:
                    if len(scores) > 0 and scores[0] == 1.0:
                        success_n = True
                        break
            else:
                for s, scores in zip(sample_results, sample_scores):
                    upto_n = scores[:min(n, len(scores))]
                    if any(sc == 1.0 for sc in upto_n):
                        success_n = True
                        break

            if success_n:
                n_stats[n]["successful_tasks"] += 1

            best_score_for_n = 0.0
            for scores in sample_scores:
                if not scores:
                    continue
                if n == 0:
                    score_n = scores[0]
                else:
                    upto_n = scores[:min(n, len(scores))]
                    if not upto_n:
                        continue
                    score_n = max(upto_n)
                if score_n > best_score_for_n:
                    best_score_for_n = score_n

            n_stats[n]["total_score"] += best_score_for_n

            # Update pass@k statistics
            for k in range(1, max_k + 1):
                pass_at_k_stats[k]["total_tasks"] += 1
                if result["pass_at_k"].get(k, 0) == 1:
                    pass_at_k_stats[k]["pass_tasks"] += 1

    # Print N values summary
    print(f"{'N':<5} {'Accuracy':<15} {'Cell Avg':<15} {'Success Tasks':<15} {'Total Tasks':<15}")
    print("-" * 60)

    detailed_stats = {}

    for n in range(0, args.max_steps + 1):
        stats = n_stats[n]
        accuracy = (stats["successful_tasks"] / stats["total_tasks"] * 100) if stats["total_tasks"] > 0 else 0.0
        cell_avg = (stats["total_score"] / stats["total_tasks"]) if stats["total_tasks"] > 0 else 0.0

        print(f"{n:<5} {accuracy:<15.2f} {cell_avg:<15.4f} {stats['successful_tasks']:<15} {stats['total_tasks']:<15}")

        detailed_stats[n] = {
            "n": n,
            "accuracy": accuracy,
            "cell_average_score": cell_avg,
            "successful_tasks": stats["successful_tasks"],
            "total_tasks": stats["total_tasks"]
        }

    # Print pass@k summary
    print("\n" + "=" * 60)
    print("PASS@K SUMMARY")
    print("=" * 60)
    print(f"{'K':<5} {'pass@k':<15} {'Pass Tasks':<15} {'Total Tasks':<15}")
    print("-" * 60)

    for k in range(1, max_k + 1):
        stats = pass_at_k_stats[k]
        pass_rate = (stats["pass_tasks"] / stats["total_tasks"] * 100) if stats["total_tasks"] > 0 else 0.0
        print(f"{k:<5} {pass_rate:<15.2f} {stats['pass_tasks']:<15} {stats['total_tasks']:<15}")
        detailed_stats[f"pass@{k}"] = {
            "k": k,
            "pass_rate": pass_rate,
            "pass_tasks": stats["pass_tasks"],
            "total_tasks": stats["total_tasks"]
        }

    # Save detailed statistics to file
    limit_str = f"_{args.limit}" if args.limit > 0 else "_full"
    stats_output = os.path.join(args.output_dir, f"stats_all_z3_small_n{timestamp}_jobid_{job_id}_limit_{limit_str}.json")
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed statistics saved to: {stats_output}")


if __name__ == "__main__":
    main()