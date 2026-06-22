import re
import os
import random
import ast
import operator
import json
import signal
import contextlib
from Prompts_System.utils.json import extract_json
from Prompts_System.utils.grid import try_extract_grid, normalize_grid, score_with_ground_truth, check_with_ground_truth

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def extract_solution(solution_str):
    
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if matches:
        final_answer = matches[-1].group(1).strip()
        try:
            solution = ast.literal_eval(final_answer)
            return solution
        except (SyntaxError, ValueError):
            try:
                solution = json.loads(final_answer)
                return solution
            except json.JSONDecodeError:
                return None
        except Exception as e:
            print(f"Error extracting solution: {e}")
            return None
    else:
        return None


def compute_accuracy(answer, ground_truth):
    """
    compare grid level accuracy of the final answer w the ground truth
    """
    if not isinstance(answer, dict):
        return 0
    
    # num_objects
    num_rows = len(ground_truth["rows"])
    #num_attributes
    num_cols = len(ground_truth["header"])

    #total_correct_cells
    correct_cells = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if answer["rows"][i][j] == ground_truth["rows"][i][j]:
                correct_cells += 1
    #accuracy
    accuracy = correct_cells / (num_rows * num_cols)
    return accuracy

def compute_score(solution_str, ground_truth, extra_info: any = None, method='strict', timeout: float = 10.0):
    if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
        print(f"DEBUG-MODE: IN REWARD COMPUTE SOLUTION STR = {solution_str}\n\n")
        print(f"DEBUG-MODE: IN REWARD COMPUTE GROUND-TRUTH STR = {ground_truth}\n\n")


    
    try:
        with time_limit(timeout):
            try:
                parsed_1 = extract_json(solution_str)
                if isinstance(parsed_1, dict):
                    if isinstance(parsed_1.get("reasoning"), str):
                        current_reasoning = parsed_1["reasoning"]
                    sol = parsed_1.get("solution")
                    if isinstance(sol, dict) and len(sol) > 0:
                        current_solution = sol
                score_info = score_with_ground_truth(current_solution, ground_truth)
                score = score_info["score"]
                #hard_verified = hard_score_info["score"] == 1.0

            except Exception as e:
                score = 0.0

    except TimeoutException:
        print("Computation timed out in zebra_puzzle")
        score = 0.0
    except Exception as e:
        print(f"Error in compute_score in zebra_puzzle: {e}")
        score = 0.0

    return {"score": score, "acc": score}


