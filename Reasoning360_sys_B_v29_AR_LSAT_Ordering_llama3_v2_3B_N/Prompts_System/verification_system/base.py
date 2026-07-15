import sys
from typing import Dict, List, Optional, Any, Tuple
from vllm import LLM, SamplingParams
from utils.json import extract_json
from utils.grid import try_extract_grid, normalize_grid, score_with_ground_truth, check_with_ground_truth

class ZebraVerificationSystemBase:
    """
    Base class for puzzle verification systems.
    
    This class provides shared functionality for both solution-based and constraint-based verification systems.
    """
    def __init__(self, model_path: str, max_attempts: int, temperature: float, top_p: float, tokenizer_mode: str = "auto"):
        """
        Initialize the verification system.
        
        Args:
            model_path: Path to the local model
            max_attempts: Maximum number of refinement attempts per sample
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            tokenizer_mode: Tokenizer mode
        """
        self.max_attempts = max_attempts
        print(f"[System] Loading local model from: {model_path}")
        print(f"[System] Config: Max Attempts={max_attempts}, Temp={temperature}, Top_P={top_p}")

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=28672,
            stop=None
        )

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

    def _generate(self, system: str, user: str) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            system: System prompt
            user: User prompt
            
        Returns:
            Generated response
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text

    def solve_puzzle(self, problem_id: str, puzzle_text: str, ground_truth: Optional[Dict[str, Any]] = None, n_samples: int = 1) -> Dict[str, Any]:
        """
        Solve a puzzle using the verification system.
        
        This method should be implemented by subclasses.
        
        Args:
            problem_id: Problem ID
            puzzle_text: Puzzle text
            ground_truth: Ground truth solution (optional)
            n_samples: Number of samples to generate
            
        Returns:
            Dictionary with solution results
        """
        raise NotImplementedError("solve_puzzle method must be implemented by subclass")

    def _interpret_verification_result(self, ver_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpret verification result and return detailed statistics.
        
        Args:
            ver_json: Verification JSON result
            
        Returns:
            Dictionary with verification status, violated clues, total clues, correct clues, and ratio
        """
        result = {
            "is_verified": False,
            "violated_clues": [],
            "total_clues": 0,
            "correct_clues": 0,
            "verification_ratio": 0.0,
            "raw": None
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
