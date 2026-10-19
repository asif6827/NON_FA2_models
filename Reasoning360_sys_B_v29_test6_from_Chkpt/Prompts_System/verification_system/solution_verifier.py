from typing import Dict, List, Optional, Any, Tuple
from .base import ZebraVerificationSystemBase
from z3_impl.constraint_checker import Z3ConstraintChecker, extract_solution_info
import json
from utils.json import extract_json
from utils.grid import try_extract_grid, normalize_grid, score_with_ground_truth, check_with_ground_truth
import prompts

class SolutionVerifier(ZebraVerificationSystemBase):
    """
    Solution-based verification system.
    
    This system generates a solution first, then verifies it using Z3 constraints and LLM verification.
    """
    def __init__(self, model_path: str, max_attempts: int, temperature: float, top_p: float, tokenizer_mode: str = "auto",
                 refinement_include_z3: bool = False, refinement_include_accuracy: bool = False, refinement_include_verification: bool = True):
        """
        Initialize the solution verifier.
        
        Args:
            model_path: Path to the local model
            max_attempts: Maximum number of refinement attempts per sample
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tokenizer_mode: Tokenizer mode
            refinement_include_z3: Whether to include Z3 check results in refinement feedback
            refinement_include_accuracy: Whether to include accuracy information in refinement feedback
            refinement_include_verification: Whether to include verification results in refinement feedback
        """
        super().__init__(model_path, max_attempts, temperature, top_p, tokenizer_mode)
        
        # Refinement feedback control parameters
        self.refinement_include_z3 = refinement_include_z3
        self.refinement_include_accuracy = refinement_include_accuracy
        self.refinement_include_verification = refinement_include_verification
        
        print(f"[System] Refinement Feedback Config: include_z3={refinement_include_z3}, include_accuracy={refinement_include_accuracy}, include_verification={refinement_include_verification}")
    
    def solve_puzzle(self, problem_id: str, puzzle_text: str, ground_truth: Optional[Dict[str, Any]] = None, n_samples: int = 1) -> Dict[str, Any]:
        """
        Solve a puzzle using solution-based verification.
        
        Args:
            problem_id: Problem ID
            puzzle_text: Puzzle text
            ground_truth: Ground truth solution (optional)
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with solution results
        """
        import time
        start_time = time.time()
        all_samples_results: List[Dict[str, Any]] = []

        for sample_idx in range(n_samples):
            print(f"  > [Sample {sample_idx+1}/{n_samples}] Generating solution using SolutionPrompt...")
            trace_steps: List[Dict[str, Any]] = []

            # Initialize Z3 constraint checker
            z3_checker = Z3ConstraintChecker()

            # Extract attribute information from ground_truth and build Z3 variables
            if ground_truth:
                gt_solution = try_extract_grid(ground_truth)
                if gt_solution:
                    attributes, attr_values = extract_solution_info(gt_solution)
                    if attributes and attr_values:
                        house_count = len(gt_solution.get("rows", []))
                        z3_checker.set_house_count(house_count)
                        z3_checker.set_attributes(attributes)
                        z3_checker.set_attribute_values(attr_values)
                        z3_checker.add_base_constraints()

            user_prompt_1 = prompts.SOLUTION_PROMPT_USER_SOLUTION_BASED.format(puzzle=puzzle_text)
            response_1 = self._generate(prompts.SOLUTION_PROMPT_SYSTEM_SOLUTION_BASED, user_prompt_1)
            parsed_1 = extract_json(response_1)

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
                "user_prompt": user_prompt_1,
                "system_prompt": prompts.SOLUTION_PROMPT_SYSTEM_SOLUTION_BASED,
                "response": response_1,
                "parsed_response": parsed_1,
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

            # When max_attempts is 0, we still need to run one attempt (initial attempt without refinement)
            for i in range(max(1, self.max_attempts)):
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

                # Special case for max_attempts=0: skip verification, directly check against ground truth
                if self.max_attempts == 0:
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] max_attempts=0: Skipping verification, directly checking against ground truth...")
                    
                    # Directly check against ground truth without verification
                    if ground_truth is not None:
                        has_solution = True
                        hard_score_info = score_with_ground_truth(current_solution, ground_truth)
                        hard_verified = hard_score_info["score"] == 1.0
                        
                        trace_steps.append({
                            "stage": f"DirectGroundTruthCheck_Attempt_{i+1}",
                            "score_info": hard_score_info
                        })
                        
                        print(f"    Accuracy: {hard_score_info['correct_cells']}/{hard_score_info['total_cells']} = {hard_score_info['score']:.4f}")
                        round_scores.append(hard_score_info["score"])
                        
                        current_attempt_status["is_correct"] = hard_verified
                        current_attempt_status["verification_passed"] = True  # Mark as verified for output compatibility
                        current_attempt_status["verification_ratio"] = 1.0
                        attempt_statuses.append(current_attempt_status)
                        
                        if hard_verified:
                            status = "success"
                            solution_status = "correct"
                            final_solution = current_solution
                            is_correct = True
                            print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Verified by ground_truth. All attributes correct.")
                            break
                        else:
                            solution_status = "incorrect"
                            is_correct = False
                            print(f"  > [Sample {sample_idx+1}/{n_samples}] [Ground Truth] Solution does not match ground_truth (score: {hard_score_info['score']:.4f}).")
                            break  # Exit after one attempt for max_attempts=0
                else:
                    # Normal mode with verification
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
                        ver_user_prompt = prompts.VERIFICATION_PROMPT_USER_SOLUTION_BASED.format(
                            problem_id=problem_id,
                            clues_text=puzzle_text,  
                            solution_text=json.dumps(current_solution, ensure_ascii=False)
                        )
                        ver_response = self._generate(prompts.VERIFICATION_PROMPT_SYSTEM_SOLUTION_BASED, ver_user_prompt)
                        ver_json = extract_json(ver_response)

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
                            
                            # Use checkmark only if all clues are satisfied, otherwise use cross
                            symbol = "✓" if ver_result["verification_ratio"] == 1.0 else "✗"
                            print(f"    {symbol} Verification: {ver_result['correct_clues']}/{ver_result['total_clues']} clues satisfied (ratio: {ver_result['verification_ratio']:.4f})")
                            current_attempt_status["verification_passed"] = ver_result["verification_ratio"] == 1.0
                            current_attempt_status["verification_ratio"] = ver_result["verification_ratio"]
                            
                            trace_steps.append({
                                "stage": f"VerificationPrompt_Attempt_{i+1}",
                                "user_prompt": ver_user_prompt,
                                "system_prompt": prompts.VERIFICATION_PROMPT_SYSTEM_SOLUTION_BASED,
                                "response": ver_response,
                                "parsed_response": ver_json,
                                "verification_passed": ver_result["is_verified"],
                                "verification_ratio": ver_result["verification_ratio"],
                                "violated_clues_count": len(ver_result["violated_clues"]),
                                "total_clues": ver_result["total_clues"],
                                "response_length": len(ver_response),
                                "verification_stats": ver_stats
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
                                "user_prompt": ver_user_prompt,
                                "system_prompt": prompts.VERIFICATION_PROMPT_SYSTEM_SOLUTION_BASED,
                                "response": ver_response,
                                "error": "failed_to_parse_json",
                                "verification_passed": False,
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
                            "user_prompt": ver_user_prompt if 'ver_user_prompt' in locals() else "",
                            "system_prompt": prompts.VERIFICATION_PROMPT_SYSTEM_SOLUTION_BASED,
                            "error": f"exception: {type(e).__name__}",
                            "verification_passed": False,
                            "response_length": 0
                        })
                    
                    # Step 2: Only check against ground truth if verification passed
                    if ground_truth is not None and current_attempt_status["verification_passed"]:
                        has_solution = True
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {i+1}] Verification passed, checking against ground_truth...")
                        hard_score_info = score_with_ground_truth(current_solution, ground_truth)
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
                            attempt_statuses.append(current_attempt_status)
                            break
                        else:
                            solution_status = "false_positive"
                            print(f"  > [Sample {sample_idx+1}/{n_samples}] [Ground Truth] Solution does not match ground_truth (score: {hard_score_info['score']:.4f}).")
                            attempt_statuses.append(current_attempt_status)
                            break  # Exit loop if verification passed but answer is wrong
                    elif ground_truth is None and z3_valid:
                        has_solution = True
                        solution_status = "correct"
                        is_correct = True
                        status = "success"
                        final_solution = current_solution
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Z3 constraints satisfied (no ground_truth).")
                        attempt_statuses.append(current_attempt_status)
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

                # Special case for max_attempts=0: skip refinement, directly compare with ground truth
                if self.max_attempts > 0:
                    # Only refine if we haven't reached max attempts
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

                    ref_user_prompt = prompts.REFINEMENT_PROMPT_USER_SOLUTION_BASED.format(
                        puzzle=puzzle_text,
                        previous_reasoning=current_reasoning,
                        previous_solution=json.dumps(current_solution, ensure_ascii=False),
                        failed_clues=json.dumps(feedback_payload, ensure_ascii=False, indent=2)
                    )

                    ref_response = self._generate(prompts.REFINEMENT_PROMPT_SYSTEM_SOLUTION_BASED, ref_user_prompt)
                    ref_json = extract_json(ref_response)

                    trace_steps.append({
                        "stage": f"RefinementPrompt_Attempt_{i+1}",
                        "user_prompt": ref_user_prompt,
                        "system_prompt": prompts.REFINEMENT_PROMPT_SYSTEM_SOLUTION_BASED,
                        "response": ref_response,
                        "parsed_response": ref_json,
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

            attempts_used = 0 if self.max_attempts == 0 else (i + 1 if 'i' in locals() else 0)
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
                
                # Count false positives (each sample can have at most 1)
                verification_stats["false_positives"] = 1 if solution_status == "false_positive" else 0
                # Count false negatives for each verification step where verification failed but solution was correct
                false_negatives_count = 0
                for res in verification_results:
                    if res["verification_ratio"] < 1.0 and is_correct:
                        false_negatives_count += 1
                verification_stats["false_negatives"] = false_negatives_count
            
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
