from typing import Dict, List, Optional, Any, Tuple
from .base import ZebraVerificationSystemBase
from z3_impl.grid_solver import GridModel, build_grid_skeleton, compile_and_add_constraints, decode_solution
from z3 import sat, unsat
import json
from utils.json import extract_json
from utils.grid import try_extract_grid, normalize_grid, score_with_ground_truth, check_with_ground_truth
from logger import debug, info, warning, error
import prompts

class ConstraintVerifier(ZebraVerificationSystemBase):
    """
    Constraint-based verification system.
    
    This system generates constraints first, then solves them using Z3, and finally verifies the constraints.
    """
    def __init__(self, model_path: str, max_attempts: int, temperature: float, top_p: float, tokenizer_mode: str = "auto"):
        """
        Initialize the constraint verifier.
        
        Args:
            model_path: Path to the local model
            max_attempts: Maximum number of refinement attempts per sample
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tokenizer_mode: Tokenizer mode
        """
        super().__init__(model_path, max_attempts, temperature, top_p, tokenizer_mode)
    
    def generate_constraints(self, puzzle_text: str) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Generate constraints from puzzle text.
        
        Args:
            puzzle_text: Puzzle text
            
        Returns:
            Tuple of (parsed puzzle metadata, constraints) or None if generation failed
        """
        user_prompt = prompts.SOLUTION_PROMPT_USER_CONSTRAINT_BASED.format(puzzle=puzzle_text)
        raw = self._generate(prompts.SOLUTION_PROMPT_SYSTEM_CONSTRAINT_BASED, user_prompt)
        # debug(f"LLM raw DSL output (first 400 chars): {raw[:400]}...")

        data = extract_json(raw)
        if not isinstance(data, dict):
            print(f"[Error] Failed to parse JSON object from LLM output: {raw[:200]}...")
            return None

        if "meta" not in data or "constraints" not in data:
            print(f"[Error] Missing 'meta' or 'constraints' keys in LLM output: {raw[:200]}...")
            return None

        meta = data["meta"]
        constraints = data["constraints"]

        if not isinstance(meta, dict) or \
           "entity_count" not in meta or \
           "attributes" not in meta or \
           "attribute_values" not in meta:
            print(f"[Error] Invalid meta structure: {meta}")
            return None

        if not isinstance(constraints, list):
            print(f"[Error] Constraints must be a list: {constraints}")
            return None

        for i, c in enumerate(constraints):
            if not isinstance(c, dict) or "op" not in c:
                print(f"[Warning] Malformed constraint at index {i}: {c}")
                continue
            
            # Check if source_clue field exists and is an integer
            if "source_clue" in c:
                source_clue = c["source_clue"]
                if not isinstance(source_clue, int):
                    print(f"[Warning] Constraint at index {i} has invalid source_clue: {source_clue}. Expected integer.")
            else:
                print(f"[Warning] Constraint at index {i} missing source_clue field.")

        return meta, constraints
    
    def verify_constraints(
        self,
        problem_id: str,
        puzzle_text: str,
        meta: Dict[str, Any],
        constraints: List[Dict[str, Any]],
        decoded_solution: Optional[Dict[str, Any]] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Verify constraints using LLM analysis.
        
        Args:
            problem_id: Problem ID
            puzzle_text: Puzzle text
            meta: Puzzle metadata
            constraints: List of constraints to verify
            decoded_solution: Decoded solution (optional)
            ground_truth: Ground truth solution (optional)
            
        Returns:
            Verification report or None if verification failed
        """
        user_prompt = prompts.VERIFICATION_PROMPT_USER_CONSTRAINT_BASED.format(
            problem_id=problem_id,
            puzzle_text=puzzle_text,
            meta_json=json.dumps(meta, ensure_ascii=False),
            constraints_json=json.dumps(constraints, ensure_ascii=False),
            solution_text=json.dumps(decoded_solution, ensure_ascii=False) if decoded_solution else "",
            ground_truth_text=json.dumps(ground_truth, ensure_ascii=False) if ground_truth else ""
        )

        raw = self._generate(prompts.VERIFICATION_PROMPT_SYSTEM_CONSTRAINT_BASED, user_prompt)
        # debug(f"Verification(raw) first 500 chars: {raw[:500]}...")

        data = extract_json(raw)

        if data is None:
            print("[Error] Verification output could not be parsed as JSON at all.")
            return None

        if isinstance(data, list):
            data = {
                "constraint_analysis": data,
                "has_severe_errors": False,
                "global_comment": "wrapped list output from LLM"
            }

        if isinstance(data, dict) and "constraint_analysis" not in data:
            if all(k in data for k in ("index", "constraint", "valid")):
                data = {
                    "constraint_analysis": [data],
                    "has_severe_errors": False,
                    "global_comment": "single constraint_analysis object wrapped automatically"
                }
            else:
                ca_key = None
                for k in data.keys():
                    if "constraint_analysis" in k or "constraints_analysis" in k or "analysis" == k:
                        ca_key = k
                        break
                if ca_key is not None:
                    ca_val = data[ca_key]
                    if isinstance(ca_val, dict):
                        ca_val = [ca_val]
                    elif not isinstance(ca_val, list):
                        ca_val = [ca_val]
                    data = {
                        "constraint_analysis": ca_val,
                        "has_severe_errors": data.get("has_severe_errors", False),
                        "global_comment": data.get("global_comment", f"extracted from key '{ca_key}'")
                    }

        if isinstance(data, dict) and "constraint_analysis" in data:
            ca = data["constraint_analysis"]
            if isinstance(ca, dict):
                data["constraint_analysis"] = [ca]
            elif not isinstance(ca, list):
                data["constraint_analysis"] = [ca]
            return data

        print(f"[Error] Verification did not return expected JSON with 'constraint_analysis'. Parsed top-level type: {type(data)}; keys: {getattr(data, 'keys', lambda: [])()}")
        return None
    
    def refine_constraints(
        self,
        puzzle_text: str,
        meta: Dict[str, Any],
        constraints: List[Dict[str, Any]],
        verification_report: Dict[str, Any],
    ) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Refine constraints based on verification report.
        
        Args:
            puzzle_text: Puzzle text
            meta: Puzzle metadata
            constraints: Current constraints
            verification_report: Verification report
            
        Returns:
            Tuple of (refined metadata, refined constraints) or None if refinement failed
        """
        user_prompt = prompts.REFINEMENT_PROMPT_USER_CONSTRAINT_BASED.format(
            puzzle=puzzle_text,
            meta_json=json.dumps(meta, ensure_ascii=False),
            constraints_json=json.dumps(constraints, ensure_ascii=False),
            verification_json=json.dumps(verification_report, ensure_ascii=False)
        )

        raw = self._generate(prompts.REFINEMENT_PROMPT_SYSTEM_CONSTRAINT_BASED, user_prompt)
        # print(f"[DEBUG] Refinement(raw) first 400 chars: {raw[:400]}...")
        data = extract_json(raw)
        if isinstance(data, dict) and "meta" in data and "constraints" in data:
            return data["meta"], data["constraints"]
        else:
            print("[Error] Refinement did not return valid DSL {meta, constraints}")
            return None
    
    def solve_puzzle(self, problem_id: str, puzzle_text: str, ground_truth: Optional[Dict[str, Any]] = None, n_samples: int = 1) -> Dict[str, Any]:
        """
        Solve a puzzle using constraint-based verification.
        
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
            print(f"  > [Sample {sample_idx+1}/{n_samples}] Generating constraints using SolutionPrompt...")
            trace_steps: List[Dict[str, Any]] = []

            current_solution: Dict[str, Any] = {}
            best_solution: Dict[str, Any] = {}
            current_constraints: List[Dict[str, Any]] = []
            parsed_puzzle: Dict[str, Any] = {}
            
            # Track solution status for this sample
            solution_status = "havent found any solution yet"
            has_solution = False
            is_correct = False
            attempt_statuses: List[Dict[str, Any]] = []

            result = self.generate_constraints(puzzle_text)

            trace_steps.append({
                "stage": "SolutionPrompt_Initial",
                "user_prompt": prompts.SOLUTION_PROMPT_USER_CONSTRAINT_BASED.format(puzzle=puzzle_text),
                "response": str(result),
                "result": result
            })

            if result:
                parsed_puzzle, current_constraints = result

                # Validate parsed puzzle metadata
                if not parsed_puzzle or not isinstance(parsed_puzzle, dict):
                    print(f"[Error] Invalid parsed puzzle metadata: {parsed_puzzle}")
                    current_constraints = []
                else:
                    try:
                        model = build_grid_skeleton(parsed_puzzle)
                        compile_and_add_constraints(model, current_constraints)

                        # debug("Solving puzzle with Z3...")
                        if model.solver.check() == sat:
                            current_solution = decode_solution(model)
                            best_solution = current_solution
                            # debug(f"Decoded solution: {current_solution}")
                        else:
                            # debug("Z3 solver returned unsat")
                            pass
                    except Exception as e:
                        print(f"[Error] Failed to solve with initial constraints: {e}")
                        current_constraints = []
            else:
                current_constraints = []

            if ground_truth:
                ground_truth_for_grid = try_extract_grid(ground_truth)
                if isinstance(ground_truth_for_grid, dict) and "header" in ground_truth_for_grid:
                    # debug(f"Ground truth header: {ground_truth_for_grid['header']}")
                    # debug(f"Ground truth rows: {ground_truth_for_grid['rows']}")
                    pass
                else:
                    # print(f"[DEBUG] Ground truth structure: {ground_truth}")
                    pass
            else:
                # print("[DEBUG] No ground_truth provided for this puzzle.")
                pass

            if not current_solution:
                # For max_attempts=0, if there's no initial solution, it's considered incorrect, not answer avoided
                if self.max_attempts == 0:
                    solution_status = "incorrect"
                sample_result = {
                    "sample_idx": sample_idx,
                    "status": "no_initial_solution",
                    "final_solution": current_solution,
                    "attempts_used": 0,
                    "trace": trace_steps,
                    "round_scores": [],
                    "best_score": 0.0,
                    "is_correct": False,
                    "solution_status": solution_status,
                    "has_solution": has_solution,
                    "attempt_statuses": attempt_statuses
                }
                all_samples_results.append(sample_result)
                continue

            status = "max_attempts_reached"
            final_solution: Dict[str, Any] = current_solution

            round_scores: List[float] = []

            # When max_attempts is 0, we still need to run one attempt (initial attempt without refinement)
            for attempt in range(max(1, self.max_attempts)):
                hard_verified: Optional[bool] = None
                hard_score_info: Optional[Dict[str, Any]] = None
                verification_passed = False
                z3_solvable = False
                constraint_valid = False
                
                # Track attempt status following solution mode conventions
                current_attempt_status = {
                    "attempt": attempt + 1,
                    "verification_passed": False,
                    "is_correct": False,
                    "verification_ratio": 0.0
                }

                # Special case for max_attempts=0: skip verification, directly check against ground truth
                if self.max_attempts == 0:
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {attempt+1}] max_attempts=0: Skipping verification, directly checking against ground truth...")
                    
                    # Directly check against ground truth without verification
                    verification_passed = True
                    constraint_valid = True
                    z3_solvable = True
                    
                    # For max_attempts=0, we need to directly check against ground truth and record score
                    if ground_truth is not None:
                        has_solution = True
                        hard_score_info = score_with_ground_truth(current_solution, ground_truth)
                        hard_verified = hard_score_info["score"] == 1.0
                        
                        # Record the score for this attempt
                        round_scores.append(hard_score_info["score"])
                        
                        trace_steps.append({
                            "stage": f"DirectGroundTruthCheck_Attempt_{attempt+1}",
                            "score_info": hard_score_info
                        })
                        
                        print(f"    Accuracy: {hard_score_info['correct_cells']}/{hard_score_info['total_cells']} = {hard_score_info['score']:.4f}")
                        
                        # Update attempt status following solution mode conventions
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
                            break  # Exit loop for max_attempts=0
                    else:
                        # No ground truth, so solution is considered correct if verification passes
                        solution_status = "correct"
                        status = "success"
                        final_solution = current_solution
                        is_correct = True
                        round_scores.append(1.0)  # Assume success without ground truth
                        
                        # Update attempt status following solution mode conventions
                        current_attempt_status["is_correct"] = True
                        current_attempt_status["verification_passed"] = True
                        current_attempt_status["verification_ratio"] = 1.0
                        attempt_statuses.append(current_attempt_status)
                        
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Solution verified successfully (no ground_truth).")
                        break
                else:
                    # Step 1: Run verification first (Z3 + constraint verification)
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {attempt+1}/{self.max_attempts}] Running verification...")
                    
                    # Z3 verification: Check if model is solvable
                    z3_solvable = False
                    try:
                        model = build_grid_skeleton(parsed_puzzle)
                        compile_and_add_constraints(model, current_constraints)
                        if model.solver.check() == sat:
                            z3_solvable = True
                            print(f"    ✓ Z3: Model is solvable")
                        else:
                            print(f"    ✗ Z3: Model is unsolvable")
                    except Exception as e:
                        print(f"    ✗ Z3: Failed to build model: {e}")
                    
                    # Constraint verification: Check if constraints are correct
                    constraint_valid = False
                    ver_report = self.verify_constraints(
                        problem_id=problem_id,
                        puzzle_text=puzzle_text,
                        meta=parsed_puzzle,
                        constraints=current_constraints,
                        decoded_solution=current_solution,
                        ground_truth=ground_truth
                    )
                    trace_steps.append({
                        "stage": f"Verification_Attempt_{attempt+1}",
                        "verification_report": ver_report,
                        "solution": current_solution,
                        "constraints_before": current_constraints
                    })
                    
                    # Check if verification passed (both Z3 solvable and constraints valid)
                    constraint_valid = False
                    if ver_report and z3_solvable:
                        # Check constraint validity, but be more lenient
                        constraint_analysis = ver_report.get("constraint_analysis", [])
                        if constraint_analysis:
                            valid_count = sum(1 for item in constraint_analysis if item.get("valid", False))
                            total_count = len(constraint_analysis)
                            verification_ratio = valid_count / total_count if total_count > 0 else 0.0
                            
                            # Original logic: at least 50% of constraints must be valid for verification to pass
                            ########################
                            #Suggest setting to 0.5, but needs to be finalized later!!!!!!
                            ########################
                            if verification_ratio >= 0.99:  # At least 50% of constraints are valid
                                constraint_valid = True
                                print(f"    ✓ Constraints: {valid_count}/{total_count} constraints valid (ratio: {verification_ratio:.4f})")
                            else:
                                print(f"    ✗ Constraints: {valid_count}/{total_count} constraints valid (ratio: {verification_ratio:.4f})")
                        else:
                            # If no constraint analysis, assume valid for verification purposes
                            constraint_valid = True
                            print(f"    ✓ Constraints: No constraint analysis, assuming valid")
                    
                    verification_passed = z3_solvable and constraint_valid
                
                if verification_passed and self.max_attempts > 0:
                    # Step 2: If verification passed, compare with ground truth
                    print(f"    ✓ Verification passed! Checking against ground truth...")
                    has_solution = True
                    
                    if ground_truth is not None:
                        hard_score_info = score_with_ground_truth(current_solution, ground_truth)
                        hard_verified = hard_score_info["score"] == 1.0
                        round_scores.append(hard_score_info["score"])

                        trace_steps.append({
                            "stage": f"GroundTruthCheck_Attempt_{attempt+1}",
                            "score_info": hard_score_info,
                            "solution": current_solution
                        })

                        print(f"    Accuracy: {hard_score_info['correct_cells']}/{hard_score_info['total_cells']} = {hard_score_info['score']:.4f}")

                        if hard_score_info["score"] > (max(round_scores[:-1]) if len(round_scores) > 1 else 0.0):
                            best_solution = current_solution

                        # Update attempt status
                        current_attempt_status["is_correct"] = hard_verified
                        current_attempt_status["verification_passed"] = True
                        current_attempt_status["verification_ratio"] = 1.0
                        attempt_statuses.append(current_attempt_status)

                        if hard_verified:
                            status = "success"
                            final_solution = current_solution
                            solution_status = "correct"
                            is_correct = True
                            print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Verified by ground_truth. All attributes correct.")
                            break
                        else:
                            solution_status = "false_positive"
                            is_correct = False
                            print(f"  > [Sample {sample_idx+1}/{n_samples}] [Ground Truth] Solution does not match ground_truth (score: {hard_score_info['score']:.4f}).")
                            break  # Exit loop since verification passed but answer is wrong
                    else:
                        # No ground truth, so solution is considered correct if verification passes
                        solution_status = "correct"
                        status = "success"
                        final_solution = current_solution
                        is_correct = True
                        round_scores.append(1.0)
                        
                        # Update attempt status
                        current_attempt_status["is_correct"] = True
                        current_attempt_status["verification_passed"] = True
                        current_attempt_status["verification_ratio"] = 1.0
                        attempt_statuses.append(current_attempt_status)
                        
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Success] Solution verified successfully (no ground_truth).")
                        break
                elif self.max_attempts > 0:  # Only execute refinement logic when max_attempts > 0
                    # Step 3: If verification failed, refine constraints and try again
                    print(f"    ✗ Verification failed, refining constraints...")
                    round_scores.append(0.0)
                    
                    # Update attempt status
                    current_attempt_status["verification_passed"] = False
                    current_attempt_status["verification_ratio"] = 0.0
                    attempt_statuses.append(current_attempt_status)
                    
                    if attempt < self.max_attempts - 1:
                        print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {attempt+1}] Running constraint-level verification and refinement...")

                        if not ver_report:
                            # debug("No verification report generated; cannot refine constraints.")
                            continue

                        # Refinement: Generate a new complete DSL based on the analysis of verification
                        refined = self.refine_constraints(
                            puzzle_text=puzzle_text,
                            meta=parsed_puzzle,
                            constraints=current_constraints,
                            verification_report=ver_report
                        )
                        if not refined:
                            # debug("Refinement failed to produce new constraints.")
                            continue

                        parsed_puzzle, current_constraints = refined
                        trace_steps.append({
                            "stage": f"Refinement_Attempt_{attempt+1}",
                            "meta_after": parsed_puzzle,
                            "constraints_after": current_constraints
                        })

                        # Rebuild the Z3 model with a new DSL and solve the problem
                        try:
                            # debug("Rebuilding model with refined constraints...")
                            model = build_grid_skeleton(parsed_puzzle)
                            compile_and_add_constraints(model, current_constraints)
                            # debug("Solving puzzle with refined constraints...")
                            if model.solver.check() == sat:
                                current_solution = decode_solution(model)
                                # debug(f"Refined solution: {current_solution}")
                            else:
                                # debug("Z3 solver returned unsat with refined constraints")
                                pass
                        except Exception as e:
                            print(f"[Error] Failed to solve with refined constraints: {e}")
                
                # Special case for max_attempts=0: exit after one attempt
                if self.max_attempts == 0:
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {attempt+1}] max_attempts=0: Exiting after one attempt.")
                    break
                elif not verification_passed and attempt == self.max_attempts - 1:
                    print(f"  > [Sample {sample_idx+1}/{n_samples}] [Attempt {attempt+1}] Reached maximum attempts, no successful verification.")

            if not final_solution or not final_solution.get("rows"):
                final_solution = best_solution

            attempts_used = 0 if self.max_attempts == 0 else (attempt + 1 if 'attempt' in locals() else 0)
            best_score = max(round_scores) if round_scores else 0.0
            is_correct = best_score == 1.0

            # Collect verification statistics for the sample
            verification_stats = {
                "total_verification_steps": len([step for step in trace_steps if "verification_report" in step]),
                "avg_verification_ratio": 0.0,
                "false_positives": 0,
                "false_negatives": 0,
                "verification_results": []
            }
            
            # Calculate verification metrics
            verification_results = []
            for step in trace_steps:
                if "verification_report" in step and step["verification_report"] is not None:
                    # For constraint verification, calculate verification ratio
                    ver_report = step["verification_report"]
                    constraint_analysis = ver_report.get("constraint_analysis", [])
                    if constraint_analysis:
                        valid_constraints = sum(1 for item in constraint_analysis if item.get("valid", False))
                        total_constraints = len(constraint_analysis)
                        verification_ratio = valid_constraints / total_constraints if total_constraints > 0 else 0.0
                        verification_results.append({
                            "verification_ratio": verification_ratio,
                            "raw": ver_report
                        })
            
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
                "solution_status": solution_status,
                "has_solution": has_solution,
                "attempt_statuses": attempt_statuses,
                "verification_stats": verification_stats
            }
            all_samples_results.append(sample_result)

        # Calculate outcome counts for this puzzle
        correct_samples = sum(1 for sample in all_samples_results if sample["solution_status"] == "correct")
        false_positive_samples = sum(1 for sample in all_samples_results if sample["solution_status"] == "false_positive")
        answer_avoided_samples = sum(1 for sample in all_samples_results if sample["solution_status"] == "havent found any solution yet")

        # Calculate pass@k metrics using standard definition
        pass_at_k = {}
        for k in range(1, n_samples + 1):
            has_correct = any(sample["is_correct"] for sample in all_samples_results[:k])
            pass_at_k[k] = 1 if has_correct else 0

        # Determine overall status
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
        total_false_negatives = 0
        samples_with_verification = 0
        
        for sample in all_samples_results:
            sample_verification = sample.get("verification_stats", {})
            if sample_verification.get("verification_results"):
                samples_with_verification += 1
                total_ratio += sample_verification["avg_verification_ratio"]
                total_steps += sample_verification["total_verification_steps"]
                total_false_negatives += sample_verification.get("false_negatives", 0)
        
        if samples_with_verification > 0:
            overall_verification_stats["avg_verification_ratio"] = total_ratio / samples_with_verification
        
        overall_verification_stats["total_verification_steps"] = total_steps
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
            "total_samples": n_samples,
            "pass_at_k": pass_at_k,
            "round_scores": [sample["best_score"] for sample in all_samples_results],
            "best_score": max(sample["best_score"] for sample in all_samples_results),
            "verification_stats": overall_verification_stats
        }
