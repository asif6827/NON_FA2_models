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

from verl.utils.reward_score.z3_verifier import verify_solution_with_z3

os.environ.setdefault("CLUE_TIMEOUT_S", "3.0")
os.environ.setdefault("Z3_TIMEOUT_S", "1.5")
os.environ.setdefault("Z3_CLUE_GATE", "0.7")
os.environ.setdefault("CLUE_MAX_NEW_TOKENS", "256")
os.environ.setdefault("CLUE_MAX_INFLIGHT", "1")

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


def extract_reasoning_and_solution(solution_str: str) -> Tuple[Optional[str], Optional[Any], str]:
    """
    Extract both reasoning and solution.
    Returns (reasoning, solution, status)
    """
    answer_content = parse_answer_tag(solution_str)
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


def _compute_acc_from_normalized(norm_pred: Dict[str, Any], norm_gt: Dict[str, Any]) -> float:
    """Exact match => 1.0, otherwise cell-level acc if shapes align."""
    if not norm_pred or not norm_gt:
        return 0.0
    if norm_pred == norm_gt:
        return 1.0

    ph, pr = norm_pred.get("header", []), norm_pred.get("rows", [])
    gh, gr = norm_gt.get("header", []), norm_gt.get("rows", [])
    if not ph or not gh or not pr or not gr:
        return 0.0
    if ph != gh or len(pr) != len(gr):
        return 0.0

    correct = 0
    total = 0
    for rp, rg in zip(pr, gr):
        if len(rp) != len(rg):
            return 0.0
        total += len(rp)
        correct += sum(1 for a, b in zip(rp, rg) if a == b)
    return (correct / total) if total > 0 else 0.0


# -------------------- ray verifier singleton --------------------

_RAY_VERIFIER = None

def _get_ray_verifier(model_config: Dict[str, Any]):
    """Per-process singleton: avoid recreating verifier/actor for each sample."""
    global _RAY_VERIFIER
    if _RAY_VERIFIER is None:
        from verl.utils.reward_score.ray_clue_verifier import RayClueVerifier
        _RAY_VERIFIER = RayClueVerifier(model_config=model_config)
    return _RAY_VERIFIER


# -------------------- z3 timeout patch --------------------

_Z3_SOLVER_PATCHED = False
_Z3_LAST_TIMEOUT_MS = None

def _ensure_z3_timeout(timeout_s: float):
    """
    Monkey-patch verl.utils.reward_score.z3_verifier.Solver to set timeout for new solvers.
    """
    global _Z3_SOLVER_PATCHED, _Z3_LAST_TIMEOUT_MS
    try:
        import verl.utils.reward_score.z3_verifier as zv
    except Exception:
        return

    ms = int(max(0.0, float(timeout_s)) * 1000)
    if _Z3_SOLVER_PATCHED and _Z3_LAST_TIMEOUT_MS == ms:
        return

    orig_solver = getattr(zv, "Solver", None)
    if orig_solver is None:
        return

    def _solver_with_timeout(*args, **kwargs):
        s = orig_solver(*args, **kwargs)
        try:
            s.set("timeout", ms)
        except Exception:
            pass
        return s

    zv.Solver = _solver_with_timeout
    _Z3_SOLVER_PATCHED = True
    _Z3_LAST_TIMEOUT_MS = ms



def compute_score(
    solution_str,
    ground_truth,
    extra_info: Any = None,
    score_method: str = "gt",
    timeout: float = 3.0,
    acc_weight: float = 1.0,
    clue_weight: float = 1.0,
    z3_weight: float = 1.0,
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
        print(f"DEBUG-MODE: IN TESTING COMPUTE-SCORE, SCORING_METHOD = {score_method}")

    def time_left() -> float:
        return max(0.0, float(timeout) - (time.monotonic() - start_t))

    acc_score = 0.0
    z3_score = 0.0
    clue_score = 0.0
    score_method = os.environ.get("TEST_SCORE_METHOD")
    #print(f"Score method in Model Testing: {score_method}")


    #if not self.async_rollout_mode:
    #    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
    #else:
    #    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)


    try:
        reasoning, predicted_arrangement, parse_status = extract_reasoning_and_solution(solution_str=solution_str)
        if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
            if parse_status != "success_answer_tag":
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
        if compute_acc and predicted_arrangement is not None and isinstance(predicted_arrangement, dict) and time_left() > 0:
            try:
                pred_conv = convert_numpy_arrays(predicted_arrangement)
                gt_conv = convert_numpy_arrays(ground_truth)

                norm_pred = normalize_grid(pred_conv)
                norm_gt = normalize_grid(gt_conv)

                if norm_pred and norm_gt:
                    acc_score = _compute_acc_from_normalized(norm_pred, norm_gt)
                else:
                    acc_score = 1.0 if pred_conv == gt_conv else 0.0
            except Exception as e:
                logger.error(f"Error calculating ACC score: {e}")
                acc_score = 0.0

        # ---------------- Z3 (先算 Z3，作为 clue gate) ----------------
        if compute_z3 and predicted_arrangement is not None and isinstance(predicted_arrangement, dict) and time_left() > 0:
            try:
                pred_conv = convert_numpy_arrays(predicted_arrangement)

                z3_timeout_s = float(os.environ.get("Z3_TIMEOUT_S", "1.5"))
                z3_timeout_s = min(z3_timeout_s, time_left())

                _ensure_z3_timeout(z3_timeout_s)

                # 注意：ACC 已经单独算过了，这里不需要 ground_truth 对比，减少开销
                z3_result = verify_solution_with_z3(pred_conv, ground_truth=None, meta=meta_used)
                z3_score = float(z3_result.get("z3_score", 0.0))
            except Exception as e:
                logger.error(f"Error calculating Z3 score: {e}")
                z3_score = 0.0

        # ---------------- CLUE (Z3 gate + 独立超时 + cancel) ----------------

        if compute_clue and reasoning and time_left() > 0:
            try:
                z3_gate = float(os.environ.get("Z3_CLUE_GATE", "0.7"))
                if z3_score < z3_gate:
                    clue_score = 0.0
                else:
                    clues: List[str] = []

                    if isinstance(extra_info, dict) and "clues" in extra_info:
                        clues = extra_info.get("clues")
                    if (not clues) and isinstance(meta_used, dict):
                        clues = meta_used.get("clues")

                    if clues is None:
                        clues = []

                    # numpy array => list，避免你之前的 ValueError: truth value is ambiguous
                    try:
                        import numpy as np
                        if isinstance(clues, np.ndarray):
                            clues = clues.tolist()
                    except Exception:
                        pass

                    if not isinstance(clues, list):
                        clues = [str(clues)]

                    if not clues:
                        clue_score = 0.0
                    else:
                        clue_timeout_s = float(os.environ.get("CLUE_TIMEOUT_S", "3.0"))
                        clue_timeout_s = min(clue_timeout_s, time_left())
                        if clue_timeout_s <= 0.0:
                            clue_score = 0.0
                        else:
                            # verifier 配置（避免硬编码）
                            model_config = {
                                "model_path": os.environ.get("CLUE_MODEL_PATH", ""),
                                "device": os.environ.get("CLUE_DEVICE", "cpu"),
                                "max_new_tokens": int(os.environ.get("CLUE_MAX_NEW_TOKENS", "256")),
                                "max_inflight": int(os.environ.get("CLUE_MAX_INFLIGHT", "1")),
                                "actor_name": os.environ.get("CLUE_ACTOR_NAME", "clue_verification_actor"),
                                "num_gpus": float(os.environ.get("CLUE_NUM_GPUS", "0")),
                                "num_cpus": float(os.environ.get("CLUE_NUM_CPUS", "1")),
                                "max_concurrency": int(os.environ.get("CLUE_MAX_CONCURRENCY", "1")),
                                "cache_size": int(os.environ.get("CLUE_CACHE_SIZE", "1000")),
                            }

                            try:
                                import ray
                                from ray.exceptions import GetTimeoutError
                            except Exception:
                                clue_score = 0.0
                            else:
                                verifier = _get_ray_verifier(model_config=model_config)
                                obj_ref = verifier.verify_clues_async(reasoning, clues, use_cache=True)
                                try:
                                    result = ray.get(obj_ref, timeout=clue_timeout_s)
                                    clue_score = float(result.get("clue_score", 0.0))
                                except GetTimeoutError:
                                    # 关键：取消后台任务，防止验证“永远不停止”
                                    try:
                                        ray.cancel(obj_ref, force=True)
                                    except Exception:
                                        pass
                                    clue_score = 0.0
                                except Exception:
                                    try:
                                        ray.cancel(obj_ref, force=True)
                                    except Exception:
                                        pass
                                    clue_score = 0.0

            except Exception as e:
                logger.error(f"Error calculating clue score: {e}")
                clue_score = 0.0

        # ---------------- weighted total in sequence gt+z3+clue----------------
        if score_method == "gt":
            weighted_total = acc_score
        elif score_method == "z3":
            weighted_total = z3_score
        elif score_method == "clue":
            weighted_total = clue_score
        elif score_method == 'gt+z3':
            total_weight = float(acc_score + z3_score)
            weighted_total = ((acc_score * acc_weight + z3_score * z3_weight) / total_weight) if total_weight > 0 else 0.0
        elif score_method == 'z3+clue':
            total_weight = float(z3_score + clue_score)
            weighted_total = ((z3_score * z3_weight + clue_score * clue_weight ) / total_weight) if total_weight > 0 else 0.0
        else:
            total_weight = float(acc_weight + z3_weight + clue_weight)
            weighted_total = ((acc_score * acc_weight + z3_score * z3_weight + clue_score * clue_weight) / total_weight) if total_weight > 0 else 0.0

        #return float(weighted_total)

    except Exception as e:
        logger.error(f"Error in compute_score in our_puzzles_dataset: {e}")
        weighted_total = 0.0
        acc_score = 0.0

    return {"score": weighted_total, "acc": acc_score, "clues": clues, "reasoning": reasoning}