# our_puzzles_dataset.py
import re
import ast
import json
import signal
import contextlib
from typing import Dict, List, Any, Optional, Tuple

# 导入Z3验证器（保持你原来的接口）
from .z3_verifier import verify_solution_with_z3

# 导入 RayClueVerifier 单例获取函数
from .ray_clue_verifier import get_global_ray_clue_verifier


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """
    Process-level timeout (signal-based). Works in main thread on Unix.
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _safe_parse_list_str(s: str):
    """
    Parse a python-list-like string safely.
    """
    try:
        return ast.literal_eval(s)
    except Exception:
        # fallback: extract quoted tokens
        items = re.findall(r"""['"]([^'"]+)['"]""", s)
        return items if items else None


def _fix_solution_table(solution: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix solution['header'] / solution['rows'] if they come as strings.
    """
    if not isinstance(solution, dict):
        return solution

    # header
    if isinstance(solution.get("header"), str):
        parsed = _safe_parse_list_str(solution["header"])
        if parsed is not None:
            solution["header"] = parsed

    # rows
    if isinstance(solution.get("rows"), str):
        parsed = _safe_parse_list_str(solution["rows"])
        if parsed is not None:
            solution["rows"] = parsed

    return solution


def extract_reasoning_and_solution(solution_str: str) -> Tuple[Optional[str], Optional[Any]]:
    """
    Extract reasoning text and predicted solution (table dict or list) from model output.

    Supports:
    - <answer>{...}</answer> JSON
    - raw JSON in text (best-effort fallback)
    - keys: "reasoning" (str) or "parsed_reasoning" (list[str]) or "parsed_reasoning" (str)
    - solution table in: obj["solution"] or top-level {"header","rows"}
    """
    if not solution_str:
        return None, None

    # 1) Prefer <answer>...</answer>
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if matches:
        payload = matches[-1].group(1).strip()
        try:
            obj = json.loads(payload)
            # reasoning
            reasoning = None
            if isinstance(obj.get("reasoning"), str):
                reasoning = obj.get("reasoning")
            elif "parsed_reasoning" in obj:
                pr = obj.get("parsed_reasoning")
                if isinstance(pr, list):
                    reasoning = "\n".join([str(x) for x in pr])
                elif isinstance(pr, str):
                    reasoning = pr

            # solution
            sol = None
            if isinstance(obj.get("solution"), dict):
                sol = obj.get("solution")
                sol = _fix_solution_table(sol)
            elif isinstance(obj.get("solution"), list):
                sol = obj.get("solution")
            elif isinstance(obj, dict) and "header" in obj and "rows" in obj:
                sol = _fix_solution_table({"header": obj.get("header"), "rows": obj.get("rows")})

            return reasoning, sol
        except Exception:
            # If not valid JSON, try list
            if payload.startswith("[") and payload.endswith("]"):
                try:
                    return None, ast.literal_eval(payload)
                except Exception:
                    return None, None

    # 2) Fallback: try to parse JSON objects from text (best effort)
    # Strategy: find all JSON objects, try first 3 complete objects, select the best one
    
    best_reasoning = None
    best_solution = None
    best_score = -1
    
    # Find all potential JSON object starts
    json_starts = [m.start() for m in re.finditer(r'\{', solution_str)]
    
    # Try to parse each potential JSON object
    for start_idx in json_starts[:3]:  # Try first 3 potential starts
        # Find matching end brace to get complete JSON object
        brace_count = 1
        end_idx = start_idx + 1
        
        while end_idx < len(solution_str) and brace_count > 0:
            if solution_str[end_idx] == '{':
                brace_count += 1
            elif solution_str[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
        
        # If we found a complete JSON object
        if brace_count == 0:
            json_str = solution_str[start_idx:end_idx]
            try:
                parsed = json.loads(json_str)
                reasoning = parsed.get("reasoning", None)
                solution = parsed.get("solution", None)
                
                # 修复solution中的header和rows，如果它们是字符串形式，尝试转换为实际数据结构
                if isinstance(solution, dict):
                    solution = _fix_solution_table(solution)
                elif isinstance(parsed, dict) and "header" in parsed and "rows" in parsed:
                    solution = _fix_solution_table({"header": parsed.get("header"), "rows": parsed.get("rows")})
                
                # Calculate a score for this JSON object
                score = 0
                if reasoning is not None:
                    score += 1
                if solution is not None and solution != {}:
                    # Check if solution is not empty
                    if isinstance(solution, dict):
                        if solution.get("header") or solution.get("rows"):
                            score += 2
                    else:
                        score += 2
                
                # Update best if this is better
                if score > best_score:
                    best_score = score
                    best_reasoning = reasoning
                    best_solution = solution
                    
            except json.JSONDecodeError:
                continue
    
    # If we found a good JSON object, return it
    if best_score > -1:
        return best_reasoning, best_solution
    
    # 额外尝试：如果整个字符串是JSON，直接解析
    try:
        parsed = json.loads(solution_str)
        reasoning = parsed.get("reasoning", None)
        solution = parsed.get("solution", None)
        
        if isinstance(solution, dict):
            solution = _fix_solution_table(solution)
        elif isinstance(parsed, dict) and "header" in parsed and "rows" in parsed:
            solution = _fix_solution_table({"header": parsed.get("header"), "rows": parsed.get("rows")})
        
        return reasoning, solution
    except json.JSONDecodeError:
        pass
    
    return None, None


def compute_edit_distance(list1, list2) -> int:
    """
    Standard edit distance (Levenshtein) for two lists.
    """
    dp = [[0 for _ in range(len(list2) + 1)] for _ in range(len(list1) + 1)]
    for i in range(len(list1) + 1):
        dp[i][0] = i
    for j in range(len(list2) + 1):
        dp[0][j] = j
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(list1)][len(list2)]


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    method: str = "strict",
    timeout: float = 12.0,
    acc_weight: float = 1.0,
    clue_weight: float = 1.0,
    z3_weight: float = 1.0,
    z3_threshold: float = 0.7,
    clue_timeout: float = 30.0,
    meta: Optional[Dict[str, Any]] = None,
    # Ray verifier config (so you can control model load ONCE)
    clue_model_config: Optional[Dict[str, Any]] = None,
    ray_namespace: str = "clue_verifier",
    ray_actor_name: str = "clue_verification_actor",
    ray_address: Optional[str] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Triple-scoring: ACC + Z3 + ClueVerifier (Ray actor with single loaded model).

    Key changes vs your old version:
    - No ThreadPoolExecutor.
    - Uses ray.get(timeout) + ray.cancel(force=True) to truly stop timed-out verifications.
    - Uses a global detached actor => model loads once globally.
    - Keeps a global time_limit to prevent Z3 or parsing from hanging forever.
    """
    import logging
    logger = logging.getLogger(__name__)

    acc_score = 0.0
    z3_score = 0.0
    clue_score = 0.0
    clue_validation_result = None

    try:
        with time_limit(timeout):
            reasoning, predicted = extract_reasoning_and_solution(solution_str)

            # -----------------------
            # 1) ACC
            # -----------------------
            if predicted is not None:
                target = ground_truth.tolist() if not isinstance(ground_truth, (list, dict)) else ground_truth

                if isinstance(predicted, list) and isinstance(target, list):
                    if predicted == target:
                        acc_score = 1.0
                    elif method != "strict":
                        ed = compute_edit_distance(predicted, target)
                        denom = max(len(predicted), len(target)) or 1
                        acc_score = max(1.0 - ed / denom, 0.0)

                elif isinstance(predicted, dict) and isinstance(target, dict):
                    # convert possible numpy arrays to lists (nested)
                    def convert_numpy(obj):
                        try:
                            import numpy as np
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                        except Exception:
                            pass
                        if isinstance(obj, dict):
                            return {k: convert_numpy(v) for k, v in obj.items()}
                        if isinstance(obj, list):
                            return [convert_numpy(x) for x in obj]
                        return obj

                    predicted_c = convert_numpy(predicted)
                    target_c = convert_numpy(target)

                    if predicted_c == target_c:
                        acc_score = 1.0
                    else:
                        header = predicted.get("header", [])
                        rows = predicted.get("rows", [])
                        gt_header = target.get("header", [])
                        gt_rows = target.get("rows", [])

                        header = convert_numpy(header)
                        rows = convert_numpy(rows)
                        gt_header = convert_numpy(gt_header)
                        gt_rows = convert_numpy(gt_rows)

                        if header == gt_header and isinstance(rows, list) and isinstance(gt_rows, list) and len(rows) == len(gt_rows):
                            correct = 0
                            total = 0
                            for row, gt_row in zip(rows, gt_rows):
                                if isinstance(row, list) and isinstance(gt_row, list) and len(row) == len(gt_row):
                                    total += len(row)
                                    correct += sum(1 for a, b in zip(row, gt_row) if a == b)
                            if total > 0:
                                acc_score = correct / total

            # -----------------------
            # 2) Z3
            # -----------------------
            if isinstance(predicted, dict) and "header" in predicted and "rows" in predicted:
                z3_result = verify_solution_with_z3(predicted, ground_truth, meta)
                z3_score = float(z3_result.get("z3_score", 0.0))

            # -----------------------
            # 3) ClueVerifier (Ray), gated by Z3 threshold
            # -----------------------
            if z3_score >= z3_threshold and reasoning:
                clues = []
                if isinstance(extra_info, dict) and extra_info.get("clues"):
                    clues = extra_info.get("clues", [])
                elif isinstance(meta, dict) and meta.get("clues"):
                    clues = meta.get("clues", [])

                # numpy -> list
                try:
                    import numpy as np
                    if isinstance(clues, np.ndarray):
                        clues = clues.tolist()
                except Exception:
                    pass

                if clues:
                    verifier = get_global_ray_clue_verifier(
                        model_config=clue_model_config or {},
                        ray_address=ray_address,
                        runtime_env=runtime_env,
                        namespace=ray_namespace,
                        actor_name=ray_actor_name,
                    )
                    clue_validation_result = verifier.verify_clues_with_timeout(
                        reasoning=reasoning,
                        clues=clues,
                        use_cache=True,
                        timeout_s=clue_timeout,
                    )
                    clue_score = float(clue_validation_result.get("clue_score", 0.0))

            # -----------------------
            # Weighted total
            # -----------------------
            total_weight = acc_weight + clue_weight + z3_weight
            weighted_total = (
                (acc_score * acc_weight + clue_score * clue_weight + z3_score * z3_weight) / total_weight
                if total_weight > 0 else 0.0
            )

    except TimeoutException:
        # Hard timeout
        weighted_total = 0.0
        acc_score = 0.0
        clue_score = 0.0
        z3_score = 0.0
        clue_validation_result = None
    except Exception as e:
        import traceback
        print(f"Error in compute_score: {e}")
        traceback.print_exc()
        weighted_total = 0.0
        acc_score = 0.0
        clue_score = 0.0
        z3_score = 0.0
        clue_validation_result = None

    result = {
        "score": weighted_total,
        "acc": acc_score,
        "clue_score": clue_score,
        "z3_score": z3_score,
        "weights": {
            "acc_weight": acc_weight,
            "clue_weight": clue_weight,
            "z3_weight": z3_weight,
        },
    }
    if clue_validation_result is not None:
        result["clue_validation_result"] = clue_validation_result
    return result


if __name__ == "__main__":
    # Minimal smoke test (replace with your own ground_truth/meta)
    example_solution_str = """<answer>{
      "reasoning": "We used clue 1 and clue 2.",
      "solution": {
        "header": ["House","Name"],
        "rows": [["1","Alice"],["2","Bob"]]
      }
    }</answer>"""

    example_ground_truth = {
        "header": ["House", "Name"],
        "rows": [["1", "Alice"], ["2", "Bob"]],
    }
    example_meta = {
        "clues": ["Alice is in house 1.", "Bob is in house 2."]
    }

    scores = compute_score(
        example_solution_str,
        example_ground_truth,
        meta=example_meta,
        z3_threshold=0.0,   # force run Z3
        clue_timeout=5.0,
        clue_model_config={
            #######################################################
            # Set the local model path to enable model validation.
            # "model_path": "/path/to/local/model",
            ######################################################
            "num_gpus": 0,  # 有 GPU 模型就设 1
            "cache_size": 2000,
        },
    )
    print(scores)