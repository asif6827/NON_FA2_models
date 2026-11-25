import json
import re
import argparse
import os
import time
import sys
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Tuple
from vllm import LLM, SamplingParams
from typing import Any, Dict, Optional
import pandas as pd
from z3 import Int, Solver, Distinct, And, Or, sat, unsat, unknown

job_id = os.getenv("SLURM_JOB_ID")
print("SLURM Job ID:", job_id)
# ==========================================
# 0. Global variables
# ==========================================



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

    def add_constraint(self, constraint_type: str, *args, **kwargs):
        if constraint_type == "same_house":
            attr1, val1, attr2, val2 = args
            self.solver.add(self.var_map[attr1][val1] == self.var_map[attr2][val2])
        elif constraint_type == "left_of":
            attr1, val1, attr2, val2 = args
            self.solver.add(self.var_map[attr1][val1] < self.var_map[attr2][val2])
        elif constraint_type == "next_to":
            attr1, val1, attr2, val2 = args
            self.solver.add(
                Or(
                    self.var_map[attr1][val1] == self.var_map[attr2][val2] + 1,
                    self.var_map[attr1][val1] == self.var_map[attr2][val2] - 1
                )
            )
        elif constraint_type == "not_same_house":
            attr1, val1, attr2, val2 = args
            self.solver.add(self.var_map[attr1][val1] != self.var_map[attr2][val2])
        elif constraint_type == "house_is":
            attr, val, house = args
            self.solver.add(self.var_map[attr][val] == house)
        elif constraint_type == "not_house":
            attr, val, house = args
            self.solver.add(self.var_map[attr][val] != house)

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

        for i, row in enumerate(rows):
            if len(row) != expected_len:
                feedback = f"Row {i+1} has incorrect length. Expected {expected_len}, got {len(row)}"
                return {
                    "valid": False,
                    "feedback": feedback,
                    "issues": [feedback]
                }

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

        z3_valid, z3_feedback = self.check_solution(solution)

        attributes_in_header = header[1:]  


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


            for val, houses in seen_vals.items():
                if len(houses) > 1:
                    issues.append(
                        f"Value '{val}' of attribute '{attr}' appears in multiple houses: {houses}"
                    )


            expected_vals = self.attr_values.get(attr)
            if expected_vals:

                missing = [v for v in expected_vals if v not in seen_vals]
                if missing:
                    issues.append(
                        f"Attribute '{attr}' is missing values: {missing}"
                    )
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


# --- SolutionPrompt ---
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

SolutionPrompt_User = """PUZZLE:
{puzzle}

Please provide your reasoning and solution:"""


# --- VerificationPrompt ---
VerificationPrompt_System = """You are an expert logic puzzle solver. I need you to verify if a given solution satisfies all the clues in a logic puzzle."""

VerificationPrompt_User = """Problem ID: {problem_id}

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


# --- RefinementPrompt ---
RefinementPrompt_System = """You are an expert logic puzzle solver. You are provided with a logic puzzle and a previous reasoning and solution to it that is wrong and violates some of the constraints.

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

RefinementPrompt_User = """PUZZLE:
{puzzle}

PREVIOUS_REASONING:
{previous_reasoning}

PREVIOUS_SOLUTION:
{previous_solution}

FEEDBACK_JSON:
{failed_clues}

ANALYSIS_AND_NEW_REASONING_SOLUTION:"""


class ZebraVerificationSystem:
    def __init__(self, model_path: str, max_attempts: int, temperature: float, top_p: float, tokenizer_mode: str = "auto", 
                 refinement_include_z3: bool = False, refinement_include_accuracy: bool = False, refinement_include_verification: bool = True):
        self.max_attempts = max_attempts
        print(f"[System] Loading local model from: {model_path}")
        print(f"[System] Config: Max Attempts={max_attempts}, Temp={temperature}, Top_P={top_p}")
        print(f"[System] Refinement Feedback Config: include_z3={refinement_include_z3}, include_accuracy={refinement_include_accuracy}, include_verification={refinement_include_verification}")

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=28672,
            stop=None
        )
        
        # Refinement feedback control parameters
        self.refinement_include_z3 = refinement_include_z3
        self.refinement_include_accuracy = refinement_include_accuracy
        self.refinement_include_verification = refinement_include_verification

        try:
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                tokenizer_mode=tokenizer_mode
            )
        except Exception as e:
            print(f"\n[Critical Error] Failed to initialize vLLM: {e}")
            sys.exit(1)

    def _interpret_verification_result(self, ver_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret verification result and return detailed statistics.
        Returns: Dict with verification status, violated clues, total clues, correct clues, and ratio.
        """
        result = {
            "is_verified": False,
            "violated_clues": [],
            "total_clues": 0,
            "correct_clues": 0,
            "verification_ratio": 0.0
        }

        if not ver_json or not isinstance(ver_json, dict):
            return result

        clue_analysis = ver_json.get("clue_analysis")
        violated_clues = ver_json.get("violated_clues")
        all_flag = bool(ver_json.get("all_clues_satisfied", False))

        if isinstance(clue_analysis, list) and len(clue_analysis) > 0:
            result["total_clues"] = len(clue_analysis)
            violated = []
            correct_count = 0
            
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
                
                if sat:
                    correct_count += 1
                else:
                    num = entry.get("clue_number")
                    if isinstance(num, int):
                        violated.append(num)
            
            result["correct_clues"] = correct_count
            result["violated_clues"] = violated
            result["is_verified"] = (len(violated) == 0)
            
            if result["total_clues"] > 0:
                result["verification_ratio"] = correct_count / result["total_clues"]
            
            if not isinstance(violated_clues, list) or len(violated_clues) == 0:
                result["violated_clues"] = violated
        else:
            result["is_verified"] = all_flag
            if isinstance(violated_clues, list):
                result["violated_clues"] = violated_clues
                result["total_clues"] = len(violated_clues)  # Best guess if no clue_analysis
                result["correct_clues"] = 0

        return result

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

    def _generate(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

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
                return None

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

        for sample_idx in range(n_samples):
            print(f"  > [Sample {sample_idx+1}/{n_samples}] Generating solution using SolutionPrompt...")
            trace_steps: List[Dict[str, Any]] = []

            # Initialize Z3 constraint checker
            z3_checker = Z3ConstraintChecker()

            # 从 ground_truth 提取属性信息并构建 Z3 变量
            if ground_truth:
                gt_solution = self._try_extract_grid(ground_truth)
                if gt_solution:
                    attributes, attr_values = extract_solution_info(gt_solution)
                    if attributes and attr_values:
                        house_count = len(gt_solution.get("rows", []))
                        z3_checker.set_house_count(house_count)
                        z3_checker.set_attributes(attributes)
                        z3_checker.set_attribute_values(attr_values)
                        z3_checker.add_base_constraints()

            user_prompt_1 = SolutionPrompt_User.format(puzzle=puzzle_text)
            response_1 = self._generate(SolutionPrompt_System, user_prompt_1)
            parsed_1 = self._extract_json(response_1)

            current_solution: Dict[str, Any] = {}
            current_reasoning: str = response_1

            if isinstance(parsed_1, dict):
                if isinstance(parsed_1.get("reasoning"), str):
                    current_reasoning = parsed_1["reasoning"]
                sol = parsed_1.get("solution")
                if isinstance(sol, dict) and len(sol) > 0:
                    current_solution = sol

            trace_steps.append({
                "stage": "SolutionPrompt",
                "solution_parsed": parsed_1 is not None,
                "response_length": len(response_1)
            })

            if not current_solution:
                sample_result = {
                    "sample_idx": sample_idx,
                    "status": "no_initial_solution",
                    "final_solution": current_solution,
                    "attempts_used": 0,
                    "trace": trace_steps,
                    "round_scores": [],
                    "best_score": 0.0,
                    "is_correct": False,
                    "has_solution": False,
                    "solution_status": "no_initial_solution",
                    "attempt_statuses": []
                }
                all_samples_results.append(sample_result)
                continue

            status = "max_attempts_reached"
            final_solution: Dict[str, Any] = current_solution
            has_solution = False
            solution_status = "havent found any solution yet"
            is_correct = False
            
            round_scores: List[float] = []
            attempt_statuses: List[Dict[str, Any]] = []

            for i in range(self.max_attempts):
                hard_verified: Optional[bool] = None
                hard_score_info: Optional[Dict[str, Any]] = None
                current_attempt_status = {
                    "attempt": i+1,
                    "verification_passed": False,
                    "is_correct": False,
                    "verification_ratio": 0.0
                }

                z3_feedback = ""
                z3_valid = False
                if hasattr(z3_checker, 'house_count') and z3_checker.house_count > 0:
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Checking solution with Z3 constraint checker...")
                    z3_analysis = z3_checker.analyze_solution(current_solution)
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

                # Step 1: Run verification on the solution (before ground truth check)
                ver_summary = None
                ver_stats = {
                    "is_verified": None,
                    "violated_clues": [],
                    "total_clues": 0,
                    "correct_clues": 0,
                    "verification_ratio": 0.0,
                    "raw": None
                }
                
                try:
                    ver_user_prompt = VerificationPrompt_User.format(
                        problem_id=problem_id,
                        clues_text=puzzle_text,  
                        solution_text=json.dumps(current_solution, ensure_ascii=False)
                    )
                    ver_response = self._generate(VerificationPrompt_System, ver_user_prompt)
                    ver_json = self._extract_json(ver_response)

                    if ver_json:
                        ver_result = self._interpret_verification_result(ver_json)
                        ver_summary = {
                            "all_clues_satisfied": ver_result["is_verified"],
                            "violated_clues": ver_result["violated_clues"],
                            "raw": ver_json
                        }
                        # Update verification stats
                        ver_stats = ver_result
                        ver_stats["raw"] = ver_json
                        
                        print(f"    ✓ Verification: {ver_result['correct_clues']}/{ver_result['total_clues']} clues satisfied (ratio: {ver_result['verification_ratio']:.4f})")
                        current_attempt_status["verification_passed"] = ver_result["verification_ratio"] == 1.0
                        current_attempt_status["verification_ratio"] = ver_result["verification_ratio"]
                        
                        trace_steps.append({
                            "stage": f"VerificationPrompt_Attempt_{i+1}",
                            "verification_passed": ver_result["is_verified"],
                            "verification_ratio": ver_result["verification_ratio"],
                            "violated_clues_count": len(ver_result["violated_clues"]),
                            "total_clues": ver_result["total_clues"],
                            "response_length": len(ver_response)
                        })
                    else:
                        ver_summary = {
                            "all_clues_satisfied": None,
                            "violated_clues": [],
                            "raw": None,
                            "error": "failed to parse JSON from verification model response"
                        }
                        print(f"    ✗ Verification: Failed to parse JSON response")
                        
                        trace_steps.append({
                            "stage": f"VerificationPrompt_Attempt_{i+1}",
                            "verification_passed": False,
                            "error": "failed_to_parse_json",
                            "response_length": len(ver_response)
                        })
                except Exception as e:
                    ver_summary = {
                        "all_clues_satisfied": None,
                        "violated_clues": [],
                        "raw": None,
                        "error": f"verification call raised exception: {e}"
                    }
                    print(f"    ✗ Verification: Exception occurred during verification")
                    
                    trace_steps.append({
                        "stage": f"VerificationPrompt_Attempt_{i+1}",
                        "verification_passed": False,
                        "error": f"exception: {type(e).__name__}",
                        "response_length": 0
                    })
                
                # Step 2: Only check against ground truth if verification passed
                if ground_truth is not None and current_attempt_status["verification_passed"]:
                    has_solution = True
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Verification passed, checking against ground_truth...")
                    hard_score_info = self._score_with_ground_truth(current_solution, ground_truth)
                    hard_verified = hard_score_info["score"] == 1.0
                    trace_steps.append({
                        "stage": f"GroundTruthCheck_Attempt_{i+1}",
                        "score_info": hard_score_info,
                        "verification_stats": ver_stats  # Add verification stats to trace
                    })
                    print(f"    Accuracy: {hard_score_info['correct_cells']}/{hard_score_info['total_cells']} = {hard_score_info['score']:.4f}")
                    round_scores.append(hard_score_info["score"])
                    
                    current_attempt_status["is_correct"] = hard_verified
                    is_correct = hard_verified

                    if hard_verified:
                        status = "success"
                        solution_status = "correct"
                        final_solution = current_solution
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Verified by ground_truth. All attributes correct.")
                        break
                    else:
                        solution_status = "false_positive"
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Ground Truth] Solution does not match ground_truth (score: {hard_score_info['score']:.4f}).")
                        break  # Exit loop if verification passed but answer is wrong
                elif ground_truth is None and z3_valid:
                    has_solution = True
                    solution_status = "correct"
                    is_correct = True
                    status = "success"
                    final_solution = current_solution
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Z3 constraints satisfied (no ground_truth).")
                    break
                else:
                    # Verification not passed, continue to next attempt
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Verification not passed, continuing to next attempt...")
                    round_scores.append(0.0)
                    trace_steps.append({
                        "stage": f"VerificationFailed_Attempt_{i+1}",
                        "verification_stats": ver_stats
                    })
                
                attempt_statuses.append(current_attempt_status)

                if i < self.max_attempts - 1:
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Solution incorrect. Refining...")

                    # Create feedback payload based on control flags
                    feedback_payload = {}
                    
                    # Only include z3_analysis if the flag is True
                    if self.refinement_include_z3 and 'z3_analysis' in locals():
                        feedback_payload["z3_analysis"] = z3_analysis
                    
                    # Only include accuracy if the flag is True
                    if self.refinement_include_accuracy:
                        feedback_payload["accuracy"] = hard_score_info
                    
                    # Only include verification if the flag is True
                    if self.refinement_include_verification:
                        feedback_payload["verification"] = ver_summary

                    ref_user_prompt = RefinementPrompt_User.format(
                        puzzle=puzzle_text,
                        previous_reasoning=current_reasoning,
                        previous_solution=json.dumps(current_solution, ensure_ascii=False),
                        failed_clues=json.dumps(feedback_payload, ensure_ascii=False, indent=2)
                    )

                    ref_response = self._generate(RefinementPrompt_System, ref_user_prompt)
                    ref_json = self._extract_json(ref_response)

                    trace_steps.append({
                        "stage": f"RefinementPrompt_Attempt_{i+1}",
                        "refinement_applied": ref_json is not None,
                        "new_solution_found": ref_json is not None and ref_json.get("new_solution") is not None,
                        "response_length": len(ref_response)
                    })

                    if not ref_json:
                        print(f"    [Sample {sample_idx+1}/{n_samples}] [Warning] Failed to parse Refinement JSON at all. Loop continues with old solution.")
                    else:
                        new_reas = ref_json.get("new_reasoning")
                        if isinstance(new_reas, str) and new_reas.strip():
                            current_reasoning = new_reas

                        raw_new_sol = ref_json.get("new_solution")
                        candidate_grid = None

                        if isinstance(raw_new_sol, dict):
                            if "header" in raw_new_sol and "rows" in raw_new_sol:
                                candidate_grid = raw_new_sol
                            elif "solution" in raw_new_sol and isinstance(raw_new_sol["solution"], dict):
                                inner = raw_new_sol["solution"]
                                if "header" in inner and "rows" in inner:
                                    candidate_grid = inner
                        else:
                            if "solution" in ref_json and isinstance(ref_json["solution"], dict):
                                inner = ref_json["solution"]
                                if "header" in inner and "rows" in inner:
                                    candidate_grid = inner

                        if candidate_grid is not None:
                            current_solution = candidate_grid
                            final_solution = candidate_grid
                            print(f"    [Sample {sample_idx+1}/{n_samples}] [Info] Updated solution from refinement.")
                        else:
                            print(f"    [Sample {sample_idx+1}/{n_samples}] [Warning] Refinement JSON parsed but no valid 'new_solution' grid found. Keeping previous solution.")
                else:
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Reached max attempts without success.")
                    break

            attempts_used = i + 1 if 'i' in locals() else 0
            best_score = max(round_scores) if round_scores else 0.0
            
            # Collect verification statistics for the sample
            verification_stats = {
                "total_verification_steps": len([step for step in trace_steps if "verification_stats" in step]),
                "avg_verification_ratio": 0.0,
                "false_positives": 0,       
                "false_negatives": 0,       
                "verification_results": []
            }
            
            # Calculate verification metrics
            verification_results = []
            for step in trace_steps:
                if "verification_stats" in step and step["verification_stats"].get("raw") is not None:
                    verification_results.append(step["verification_stats"])
            
            if verification_results:
                # Calculate average verification ratio
                total_ratio = sum(res["verification_ratio"] for res in verification_results)
                verification_stats["avg_verification_ratio"] = total_ratio / len(verification_results)
                verification_stats["verification_results"] = verification_results
                
                # Count false positives and false negatives
                for res in verification_results:
                    if res["verification_ratio"] == 1.0 and not is_correct:
                        verification_stats["false_positives"] += 1
                    elif res["verification_ratio"] < 1.0 and is_correct:
                        verification_stats["false_negatives"] += 1
            
            sample_result = {
                "sample_idx": sample_idx,
                "status": status,
                "final_solution": final_solution,
                "attempts_used": attempts_used,
                "trace": trace_steps,
                "round_scores": round_scores,
                "best_score": best_score,
                "is_correct": is_correct,
                "has_solution": has_solution,
                "solution_status": solution_status,
                "attempt_statuses": attempt_statuses,
                "verification_stats": verification_stats
            }
            all_samples_results.append(sample_result)

        correct_samples = sum(1 for sample in all_samples_results if sample["is_correct"])
        false_positive_samples = sum(1 for sample in all_samples_results if sample["solution_status"] == "false_positive")
        answer_avoided_samples = sum(1 for sample in all_samples_results if sample["solution_status"] == "havent found any solution yet")
        total_samples = len(all_samples_results)

        pass_at_k = {}
        for k in range(1, n_samples + 1):
            has_correct = any(sample["is_correct"] for sample in all_samples_results[:k])
            pass_at_k[k] = 1 if has_correct else 0

        overall_status = "success" if correct_samples >= 1 else "max_attempts_reached"
        
        # Calculate overall verification statistics
        overall_verification_stats = {
            "total_verification_steps": 0,
            "avg_verification_ratio": 0.0,
            "total_false_positives": false_positive_samples,
            "total_false_negatives": 0,
            "answer_avoided": answer_avoided_samples,
            "samples_with_verification": 0
        }
        
        total_ratio = 0.0
        total_steps = 0
        total_false_positives = 0
        total_false_negatives = 0
        samples_with_verification = 0
        
        for sample in all_samples_results:
            sample_verification = sample.get("verification_stats", {})
            if sample_verification.get("verification_results"):
                samples_with_verification += 1
                total_ratio += sample_verification["avg_verification_ratio"]
                total_steps += sample_verification["total_verification_steps"]
                total_false_positives += sample_verification["false_positives"]
                total_false_negatives += sample_verification.get("false_negatives", 0)
        
        if samples_with_verification > 0:
            overall_verification_stats["avg_verification_ratio"] = total_ratio / samples_with_verification
        
        overall_verification_stats["total_verification_steps"] = total_steps
        overall_verification_stats["total_false_positives"] = total_false_positives
        overall_verification_stats["total_false_negatives"] = total_false_negatives
        overall_verification_stats["samples_with_verification"] = samples_with_verification

        return {
            "id": problem_id,
            "puzzle_snippet": puzzle_text,
            "status": overall_status,
            "attempts_used": max(sample["attempts_used"] for sample in all_samples_results),
            "all_samples_results": all_samples_results,
            "correct_samples": correct_samples,
            "false_positive_samples": false_positive_samples,
            "answer_avoided_samples": answer_avoided_samples,
            "total_samples": total_samples,
            "pass_at_k": pass_at_k,
            "round_scores": [sample["best_score"] for sample in all_samples_results],
            "best_score": max(sample["best_score"] for sample in all_samples_results),
            "verification_stats": overall_verification_stats
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
    parser.add_argument("--model_path", type=str, default='/export/home/asifali/HF_cache/Qwen2.5-7B-Instruct', help="LLM")
    parser.add_argument("--data_",  type=str, default="small", help="(.json, .jsonl or .parquet)")
    parser.add_argument("--output_dir", type=str, default="/export/home/asifali/Reasoning360/Prompts_Results/ZebraPuzzle_1000_main_results", help="output directory")

    parser.add_argument("--n_samples", type=int, default=4, help="Number of samples per prompt (default 4 for zebra_puzzle)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--tokenizer_mode", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=-1, help="Only the first K data is used for quick debugging, -1 indicates running the full amount")
    parser.add_argument("--max_attempts", type=int, default=15, help="Maximum number of refinement attempts per sample")
    
    # Refinement feedback control arguments
    parser.add_argument("--refinement_include_z3", type=bool, default=True, help="Whether to include Z3 check results in refinement feedback (default: False)")
    parser.add_argument("--refinement_include_accuracy", type=bool, default=False, help="Whether to include accuracy information in refinement feedback (default: False)")
    parser.add_argument("--refinement_include_verification", type=bool, default=False, help="Whether to include verification results in refinement feedback (default: True)")

    args = parser.parse_args()

    if args.data_ == "small":
        args.data_path = "/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_small_320.json"
    elif args.data_ == "medium":
        args.data_path = "/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_medium_280.json"
    elif args.data_ == "large":
        args.data_path = "/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_large_200.json"
    elif args.data_ == "XL":
        args.data_path = "/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_xl_200.json"

    print("Program Arguments = {}".format(args))
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_local_dataset(args.data_path)
    if not dataset:
        return
    if args.limit > 0:
        dataset = dataset[:args.limit]

    try:
        system = ZebraVerificationSystem(
            model_path=args.model_path,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer_mode=args.tokenizer_mode,
            refinement_include_z3=args.refinement_include_z3,
            refinement_include_accuracy=args.refinement_include_accuracy,
            refinement_include_verification=args.refinement_include_verification
        )
    except Exception:
        sys.exit(1)

    print(f"\n" + "=" * 50)
    print(f"[System] Evaluating with max_attempts={args.max_attempts}")
    print("=" * 50)

    all_results: List[Dict[str, Any]] = []
    print(f"[System] Starting Verification Loop...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    limit_str = f"_{args.limit}" if args.limit > 0 else "_full"
    output_file = os.path.join(args.output_dir, f"results_all_{args.data_}_n_{timestamp}_jobid_{job_id}_limit_{limit_str}.jsonl")

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing puzzles"):
            pid = item.get("id", f"sample_{idx}")
            puzzle_text = get_puzzle_text(item)
            if not puzzle_text:
                continue

            ground_truth = item.get("solution")

            tqdm.write(f"Processing {pid} ... ")
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
    max_n = args.max_attempts
    for n in range(0, max_n + 1):
        n_stats[n] = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "false_positive_tasks": 0,
            "answer_avoided_tasks": 0,
            "total_score": 0.0
        }

    pass_at_k_stats = {}
    max_k = args.n_samples
    for k in range(1, max_k + 1):
        pass_at_k_stats[k] = {
            "total_tasks": 0,
            "pass_tasks": 0
        }

    verification_summary = {
        "total_verification_steps": 0,
        "total_false_positives": 0,   
        "total_false_negatives": 0,   
        "samples_with_verification": 0,
        "total_results": len(all_results)
    }
    
    average_k_scores = {k: 0.0 for k in range(1, args.max_attempts + 1)}
    average_k_counts = {k: 0 for k in range(1, args.max_attempts + 1)}

    for result in all_results:
        sample_results = result.get("all_samples_results", [])

        sample_scores = []
        for s in sample_results:
            scores = s.get("round_scores", [])
            if not isinstance(scores, list):
                scores = []
            sample_scores.append(scores)

        for n in range(0, max_n + 1):
            n_stats[n]["total_tasks"] += 1

            success_n = False
            false_positive_n = False
            answer_avoided_n = False

            # Step 1: Check if any sample succeeded within n attempts (highest priority)
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

            if not success_n:
                # Step 2: Check if any sample had a false positive within n attempts (medium priority)
                for sample in sample_results:
                    if sample.get("solution_status") == "false_positive":
                        sample_round_scores = sample.get("round_scores", [])
                        if (n == 0 and len(sample_round_scores) > 0) or \
                           (n > 0 and n <= len(sample_round_scores)):
                            false_positive_n = True
                            break

            # Step 3: If not successful or false positive, it's answer avoided by default
            # Answer avoided means no verified solution found within N attempts
            answer_avoided_n = not success_n and not false_positive_n

            # Ensure mutual exclusivity: exactly one category per puzzle
            if success_n:
                n_stats[n]["successful_tasks"] += 1
            elif false_positive_n:
                n_stats[n]["false_positive_tasks"] += 1
            else:  # answer_avoided_n is True by definition
                n_stats[n]["answer_avoided_tasks"] += 1

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

        for k in range(1, max_k + 1):
            pass_at_k_stats[k]["total_tasks"] += 1
            if result["pass_at_k"].get(k, 0) == 1:
                pass_at_k_stats[k]["pass_tasks"] += 1
        
        # Update verification summary statistics
        ver_stats = result.get("verification_stats", {})
        verification_summary["total_verification_steps"] += ver_stats.get("total_verification_steps", 0)
        verification_summary["total_false_positives"] += ver_stats.get("total_false_positives", 0)
        verification_summary["total_false_negatives"] += ver_stats.get("total_false_negatives", 0)
        if ver_stats.get("samples_with_verification", 0) > 0:
            verification_summary["samples_with_verification"] += 1
        
        for sample in sample_results:
            round_scores = sample.get("round_scores", [])
            for k in range(1, args.max_attempts + 1):
                upto_k_scores = round_scores[:min(k, len(round_scores))]
                if upto_k_scores:
                    best_score = max(upto_k_scores)
                    average_k_scores[k] += best_score
                    average_k_counts[k] += 1
    
    # Calculate average@k for each k
    for k in range(1, args.max_attempts + 1):
        if average_k_counts[k] > 0:
            verification_summary[f"average@{k}"] = average_k_scores[k] / average_k_counts[k]
        else:
            verification_summary[f"average@{k}"] = 0.0

    # Calculate mutually exclusive categories for each N
    for n in range(0, max_n + 1):
        stats = n_stats[n]
        total_tasks = stats["total_tasks"]
        
        # Ensure categories are mutually exclusive and sum to total_tasks
        # Priority: success > false_positive > answer_avoided
        success_count = stats["successful_tasks"]
        remaining_after_success = total_tasks - success_count
        
        # Only count false positives from the remaining tasks
        false_positive_count = min(stats["false_positive_tasks"], remaining_after_success)
        remaining_after_fp = remaining_after_success - false_positive_count
        
        # Answer avoided ge ts the rest
        answer_avoided_count = remaining_after_fp
        
        # Update the stats with mutually exclusive counts
        stats["successful_tasks"] = success_count
        stats["false_positive_tasks"] = false_positive_count
        stats["answer_avoided_tasks"] = answer_avoided_count

    print(f"{'N':<5} {'Accuracy':<15} {'False Positive':<15} {'Answer Avoided':<15} {'Total Tasks':<15}")
    print("-" * 60)

    detailed_stats = {}

    for n in range(0, max_n + 1):
        stats = n_stats[n]
        total_tasks = stats["total_tasks"]
        accuracy = (stats["successful_tasks"] / total_tasks * 100) if total_tasks > 0 else 0.0
        false_positive = (stats["false_positive_tasks"] / total_tasks * 100) if total_tasks > 0 else 0.0
        answer_avoided = (stats["answer_avoided_tasks"] / total_tasks * 100) if total_tasks > 0 else 0.0

        print(f"{n:<5} {accuracy:<15.2f} {false_positive:<15.2f} {answer_avoided:<15.2f} {total_tasks:<15}")

        detailed_stats[n] = {
            "n": n,
            "accuracy": accuracy,
            "false_positive": false_positive,
            "answer_avoided": answer_avoided,
            "total_tasks": total_tasks
        }

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
    
    print("\n" + "=" * 60)
    print("AVERAGE@K SUMMARY")
    print("=" * 60)
    print(f"{'K':<5} {'average@k':<15} {'Total Samples':<15}")
    print("-" * 60)
    
    for k in range(1, args.max_attempts + 1):
        avg_k = verification_summary.get(f"average@{k}", 0.0)
        # Calculate total samples for this k
        total_samples = average_k_counts.get(k, 0)
        print(f"{k:<5} {avg_k:<15.4f} {total_samples:<15}")
        detailed_stats[f"average@{k}"] = {
            "k": k,
            "average@k": avg_k,
            "total_samples": total_samples
        }
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total Results: {verification_summary['total_results']}")
    print(f"Total Verification Steps: {verification_summary['total_verification_steps']}")
    print(f"Samples with Verification: {verification_summary['samples_with_verification']}")
    print(f"Total False Positives: {verification_summary['total_false_positives']}")
    print(f"Total False Negatives: {verification_summary['total_false_negatives']}")

    limit_str = f"_{args.limit}" if args.limit > 0 else "_full"
    stats_output = os.path.join(args.output_dir, f"stats_all_{args.data_}_n{timestamp}_jobid_{job_id}_limit_{limit_str}.json")
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(detailed_stats, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed statistics saved to: {stats_output}")


if __name__ == "__main__":
    main()