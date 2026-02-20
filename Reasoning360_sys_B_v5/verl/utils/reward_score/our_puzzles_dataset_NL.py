# -*- coding: utf-8 -*-
"""
our_puzzles_dataset.py

Reward scoring for Zebra/Logic puzzles with triple scoring:
- ACC (grid match / cell-level)
- Z3 validity score
- Clue self-check score (LLM-as-verifier via Ray)
"""

import re
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys
# from z3_verifier_v13 import compute_dsl_components
from verl.utils.reward_score.z3_verifier_v15 import compute_z3_reward

os.environ.setdefault("CLUE_TIMEOUT_S", "3.0")
os.environ.setdefault("Z3_TIMEOUT_S", "1.5")
os.environ.setdefault("Z3_CLUE_GATE", "0.7")
os.environ.setdefault("CLUE_MAX_NEW_TOKENS", "256")
os.environ.setdefault("CLUE_MAX_INFLIGHT", "1")

job_id = os.getenv("SLURM_JOB_ID")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)


def parse_answer_tag(solution_str: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    answer_pattern = r'<answer>([\s\S]*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def log_case(case_type: str, solution_str: str, ground_truth: Any, logger):
    """Log special cases like non-boxed answers."""
    logger.info(f"{case_type} case:")
    logger.info(f"Solution_str: {solution_str}")
    logger.info(f"Ground_truth: {ground_truth}")


def _try_parse_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Try parsing JSON object from text.
    - fast path: json.loads(text)
    - fallback: brute scan for the first valid {...} object
    """
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    starts = [m.start() for m in re.finditer(r'\{', text)]
    for st in starts:
        for ed in range(len(text), st, -1):
            if text[ed - 1] != '}':
                continue
            chunk = text[st:ed]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def find_last_answer_block(text: str) -> Optional[str]:
    """
    Returns the *last* <answer>...</answer> block found in `text`,
    or None if no block exists.

    - Case-insensitive tags: <answer> or <ANSWER>
    - Allows attributes in the opening tag: <answer id="x">
    - Dot matches newlines so multi-line blocks work
    """
    pattern = re.compile(
        r"<answer\b[^>]*>.*?</answer\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return None

    return matches[-1].group(0)


def extract_reasoning_and_solution(solution_str: str):
    """
    Extract both reasoning and solution.
    Returns (reasoning, solution, status)
    """
    answer_content = find_last_answer_block(solution_str)
    if answer_content:
        parsed = _try_parse_first_json_obj(answer_content)
        if parsed is not None:
            return parsed.get("reasoning", None), parsed.get("solution", None), "success_answer_tag"
        return None, None, "answer_tag_json_error"

    parsed = _try_parse_first_json_obj(solution_str)
    if parsed is not None:
        return parsed.get("reasoning", None), parsed.get("solution", None), "success_direct_json"

    return None, None, "parsing_failed"


# -------------------- normalization helpers --------------------

def convert_numpy_arrays(obj: Any) -> Any:
    """Convert numpy arrays nested in dict/list to python lists."""
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_arrays(x) for x in obj]
    return obj


def _normalize_atom(x: Any) -> str:
    return str(x).strip().lower()


def normalize_grid(data: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize grid for comparison:
    - lower/strip everything
    - ignore 'house'/'position' column if present
    - sort rows for permutation invariance
    """
    if not isinstance(data, dict):
        return None
    if "header" not in data or "rows" not in data:
        return None

    try:
        header = [_normalize_atom(h) for h in data["header"]]

        ignore_cols = {"house", "position"}
        keep_indices = [i for i, h in enumerate(header) if h not in ignore_cols]
        if not keep_indices:
            keep_indices = list(range(len(header)))

        rows_norm: List[List[str]] = []
        for row in data["rows"]:
            row_norm_full = [_normalize_atom(x) for x in row]
            row_norm = [row_norm_full[i] for i in keep_indices]
            rows_norm.append(row_norm)

        header_kept = [header[i] for i in keep_indices]
        return {"header": header_kept, "rows": sorted(rows_norm)}
    except Exception:
        return None


def normalize_strings(obj):
    """
    Recursively normalize:
      - strings -> lower-case + spaces -> underscores
      - list/tuple -> normalize each element (supports list of lists)
      - dict -> normalize values (keys unchanged; change if you want)
    """
    if isinstance(obj, str):
        return obj.strip().lower().replace(" ", "_")
    if isinstance(obj, (list, tuple)):
        return [normalize_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize_strings(v) for k, v in obj.items()}
    return obj


def _compute_acc_from_normalized(norm_pred: Dict[str, Any], norm_gt: Dict[str, Any]) -> float:
    """Exact match => 1.0, otherwise cell-level acc if shapes align."""
    if not norm_pred or not norm_gt:
        return (0.0, 0.0)
    if norm_pred == norm_gt:
        return (1.0, 1.0)

    ph, pr = norm_pred.get("header", []), norm_pred.get("rows", [])
    gh, gr = norm_gt.get("header", []), norm_gt.get("rows", [])
    ph, pr = normalize_strings(ph), normalize_strings(pr)
    gh, gr = normalize_strings(gh), normalize_strings(gr)
    if not ph or not gh or not pr or not gr:
        return (0.0, 0.0)
    if ph != gh or len(pr) != len(gr):
        return (0.0, 0.0)

    puzzle_accuracy = 0.0
    correct = 0
    total = 0
    for rp, rg in zip(pr, gr):
        if len(rp) != len(rg):
            return 0.0
        total += len(rp)
        correct += sum(1 for a, b in zip(rp, rg) if a == b)
    cell_accuracy = correct / total if total > 0 else 0.0
    if cell_accuracy < 1.0:
        puzzle_accuracy = 0.0
    else:
        puzzle_accuracy = 1.0
    return (cell_accuracy, puzzle_accuracy)



def compute_score(
        solution_str,
        ground_truth,
        extra_info: Any = None,
        score_method: str = "gt+z3",
        timeout: float = 3.0,
        acc_weight: float = 0.8,
        clue_weight: float = 1.0,
        z3_weight: float = 0.2,
        meta: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Triple scoring with:
    - ACC
    - Z3
    - Clue-check (gated by Z3)

    NOTE:
    - clue 超时/失败不会中断整个评分
    - clue 超时后会 cancel 对应 Ray 任务，防止“永远不停止/队列堆积”
    """

    import logging
    import time

    start_t = time.monotonic()

    if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
        print(f"DEBUG-MODE: IN COMPUTE-SCORE, SCORING_METHOD = {score_method}")

    def time_left() -> float:
        return max(0.0, float(timeout) - (time.monotonic() - start_t))

    # puzzle_text = extract_puzzle_text(solution_str).strip()

    epoch = int(os.getenv("CURRENT_EPOCH", "90"))
    total_epochs = int(os.getenv("TOTAL_EPOCH", "100"))

    cell_acc_score = 0.0
    puzzle_acc_score = 0.0
    z3_reward = 0.0
    clue_score = 0.0
    final_prompt = ""
    reward = 0.0
    z3_breakdown = {}

    score_method = str(os.environ.get("TRAIN_SCORE_METHOD", "gt+z3"))
    # print(f"Score method in Model Training: {score_method}")
    try:
        reasoning, predicted_arrangement, parse_status = extract_reasoning_and_solution(solution_str=solution_str)
        # print(f"Parsed  Clues: {parsed_clues}")
        # print(f"Parsed  Reasoning: {parsed_reasoning}")
        # print(f"Predicted Arrangement: {predicted_arrangement}")
        # print(f"Attribute Values: {attribute_values}")
        # print(f"N-Houses : {n_houses}")

        if parse_status != "success_answer_tag":
            if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
                log_case("non_boxed_answer", solution_str, ground_truth, logger)

        # normalize score_method
        sm = (score_method or "all").lower().strip()
        if sm == "all":
            methods = {"gt", "z3", "clue"}
        else:
            for sep in [",", " "]:
                sm = sm.replace(sep, "+")
            methods = {m for m in sm.split("+") if m}

        compute_acc = "gt" in methods
        compute_z3 = "z3" in methods
        compute_clue = "clue" in methods

        # meta selection
        meta_used = meta
        if meta_used is None and isinstance(extra_info, dict):
            meta_used = extra_info.get("meta") or extra_info

        # ---------------- ACC ----------------
        if compute_acc and predicted_arrangement is not None and isinstance(predicted_arrangement, dict) > 0:
            try:
                pred_conv = convert_numpy_arrays(predicted_arrangement)
                gt_conv = convert_numpy_arrays(ground_truth)

                norm_pred = normalize_grid(pred_conv)
                norm_gt = normalize_grid(gt_conv)

                if norm_pred and norm_gt:
                    cell_acc_score, puzzle_acc_score = _compute_acc_from_normalized(norm_pred, norm_gt)
                    # puzzle_acc_score, cell_acc_score = puzzle_and_cell_accuracy(norm_pred, norm_gt)
                else:
                    cell_acc_score = 1.0 if pred_conv == gt_conv else 0.0
                    puzzle_acc_score = 1.0 if pred_conv == gt_conv else 0.0
            except Exception as e:
                logger.error(f"Error calculating ACC score: {e}")
                cell_acc_score = 0.0
                puzzle_acc_score = 0.0

        # ---------------- Z3 (先算 Z3，作为 clue gate) ----------------
        if compute_z3 and reasoning and predicted_arrangement is not None and isinstance(predicted_arrangement, dict) > 0:
            pass
        # ---------------- CLUE (Z3 gate + 独立超时 + cancel) ----------------

        if compute_clue and reasoning:
            pass


        # ---------------- weighted total in sequence gt+z3+clue----------------
        if score_method == "gt":
            weighted_total = cell_acc_score
            reward = weighted_total

    except Exception as e:
        logger.info(f"Z3-Result: {z3_reward}")
        logger.exception("Crash in compute_score")  # includes line number + stack
        logger.error(f"Error in compute_score in our_puzzles_dataset: {e}")
        reward = 0.0

        # print(f" Solution String : {solution_str}")
        # print(f" Ground Truth : {ground_truth}")
        # print(f" Extra Info : {extra_info}")
    # print(f"reward = {reward}, Breakdown = {reward_breakdown}")
    final_result = {"epoch": epoch, "total-epoch": total_epochs, "score": reward, "reward_logged": reward, "acc": cell_acc_score,
                    "PUZZLE_ACCURACY": puzzle_acc_score, "CELL_ACCURACY": cell_acc_score, "verification_prompt": final_prompt}

    if score_method == "gt":
        final_result = {}
        reward = cell_acc_score
        final_result["acc"] = reward
        final_result["PUZZLE_ACCURACY"] = puzzle_acc_score
        final_result["CELL_ACCURACY"] = cell_acc_score
        final_result["score"] = reward
        final_result["verification_prompt"] = ""


    return final_result


def pretty(x):
    print(json.dumps(x, indent=2, ensure_ascii=False))


def main():
    llm_response = """    <answer>{
        "attribute_values": {
        "Name": ["Arnold", "Eric", "Peter"],
        "Drink": ["milk", "water", "tea"],
        "Hobby": ["photography", "cooking", "gardening"]
        },
        "n_houses":3,
        "parsed_clues" : [
            "C1 = set(2,Name,Peter).",
            "C2 = immediately_left_of(Name=Arnold,Drink=water).",
            "C3 = immediately_left_of(Drink=water,Drink=milk)."
        ],
        "parsed_reasoning" : [
            "S1 [C1] set(2,Name,Peter).",
            "S2 [C3] not(3,Drink,water).",
            "S3 [C3] not(1,Drink,milk).",
            "S4 [C2] not(3,Name,Arnold).",
            "S5 [C2+C3] set(1,Name,Arnold)."
        ],
        "solution": {
            "header": ["House", "name", "Drink", "Hobby"],
            "rows": [
                ["1", "Eric", "milk", "photography"],
                ["2", "Peter", "water", "cooking"],
                ["3", "Arnold", "tea", "gardening"]
            ]
        }
    }
    </answer>
    """

    clues = [
        "The Dane is somewhere to the left of the person who has black hair.",
        "The person who is a doctor is Eric.",
        "The person who is a pizza lover is in the second house.",
        "Arnold is directly left of the person who has a cat."
    ]

    ground_truth = {
        "header": ["House", "Name", "Drink", "Hobby"],
        "rows": [
            ["1", "Eric", "milk", "photography"],
            ["2", "Peter", "water", "cooking"],
            ["3", "Arnold", "tea", "gardening"]
        ]
    }

    # Extra info carries clues/meta for clue prompt generation
    extra_info = {"clues": clues, "meta": {"clues": clues}}

    # ----------------------------
    # 3) Combined (gt+z3+clue)
    # - clue part builds a prompt in `verification_`
    # - clue_score may remain 0.0 unless you add the actual clue verifier call later
    # ----------------------------
    os.environ["TRAIN_SCORE_METHOD"] = "gt+z3"
    print("\n=== TRAIN_SCORE_METHOD=gt+z3 ===")
    out_ = compute_score(llm_response, ground_truth, extra_info=extra_info, timeout=3.0)

    # out_bad  = compute_score(llm_bad,  ground_truth, extra_info=extra_info, timeout=3.0)
    print("Output:")
    pretty(out_)


if __name__ == "__main__":
    main()