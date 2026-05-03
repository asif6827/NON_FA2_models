# -*- coding: utf-8 -*-
"""Crash-safe reward scoring for AR-LSAT ASSIGNMENT outputs.

This version is VERL-safe: compute_score always returns the same numeric keys.
It also includes diagnostics for low BASE_sat_full_GT.
"""
from __future__ import annotations

import json, logging, os, re, sys
from typing import Any, Dict, Optional

try:
    from z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
except Exception:
    try:
        from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
    except Exception:
        def solve_and_validate_payload(payload, *, timeout_s=2.0, conflict_tolerant_clues=False):
            return {"parse_status": "Z3_IMPORT_FAIL", "base_sat_full_GT": False}

try:
    from check_interleved_format import check_interleaved_reasoning
except Exception:
    try:
        from verl.utils.reward_score.check_interleved_format import check_interleaved_reasoning
    except Exception:
        def check_interleaved_reasoning(reasoning, *, n_houses=0):
            return False

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

RESULT_KEYS = [
    "acc", "score", "reward_logged", "ACCURACY", "parsing_reward", "schema_reward", "format_reward",
    "z3_reward", "consistency_score", "Normalizer", "BASE_sat_full_GT", "missed_data",
    "BASE_n_steps_total", "BASE_n_steps_parsed_ok", "BASE_n_steps_valid", "BASE_n_steps_novel_inc_clues", "BASE_n_non_valid_contradiction",
    "novel_step_score", "contradiction_ratio", "selected_option_present", "ground_truth_present", "parse_status_ok", "schema_status_ok", "z3_status_ok", "format_status_ok",
    "z3_base_sat", "z3_solver_selected_ok", "z3_gt_match", "z3_rule_parse_error_count", "z3_fact_parse_error_count", "z3_option_parse_error_count", "z3_selected_option_parse_ok",
    "reward_exception", "parse_error_flag", "epoch", "total_epochs",
]


def _safe_epoch(name: str, default: int) -> float:
    try: return float(int(os.getenv(name, str(default))))
    except Exception: return float(default)


def _default_result(reward: float = -0.5, missed_data: float = 1.0) -> Dict[str, float]:
    out = {k: 0.0 for k in RESULT_KEYS}
    out["acc"] = out["score"] = out["reward_logged"] = float(reward)
    out["missed_data"] = float(missed_data)
    out["Normalizer"] = 1.0
    out["epoch"] = _safe_epoch("CURRENT_EPOCH", 0)
    out["total_epochs"] = _safe_epoch("TOTAL_EPOCH", 1)
    return out


def _numeric_only(d: Dict[str, Any]) -> Dict[str, float]:
    out = _default_result()
    for k in RESULT_KEYS:
        if k not in d: continue
        v = d[k]
        try:
            if isinstance(v, bool): out[k] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)): out[k] = float(v)
            else: out[k] = float(v)
        except Exception:
            out[k] = 0.0
    return out


def _clamp_reward(x: Any, lo=-1.0, hi=1.0) -> float:
    try: v = float(x)
    except Exception: return lo
    return max(lo, min(hi, v))


def _norm_option_label(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x).strip().upper()
    s = s.replace('SELECTED_OPTION', '').replace('OPTION', '').replace('ANSWER', '')
    s = re.sub(r"[^A-Z]", "", s)
    if len(s) == 1 and 'A' <= s <= 'Z': return s
    m = re.search(r"[A-Z]", s)
    return m.group(0) if m else None


def find_last_answer_block(text: Any) -> Optional[str]:
    text = "" if text is None else text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else str(text)
    matches = list(re.finditer(r"<answer\b[^>]*>(.*?)</answer\s*>", text, flags=re.IGNORECASE | re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def _try_parse_first_json_obj(text: Any) -> Optional[Dict[str, Any]]:
    if text is None: return None
    raw = text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else str(text)
    raw = raw.strip()
    if not raw: return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw); return obj if isinstance(obj, dict) else None
    except Exception: pass
    for st in [m.start() for m in re.finditer(r"\{", raw)]:
        for ed in range(len(raw), st, -1):
            if raw[ed-1] != "}": continue
            try:
                obj = json.loads(raw[st:ed])
                if isinstance(obj, dict): return obj
            except Exception: continue
    return None


def parse_ar_lsat_answer(solution_str: Any):
    block = find_last_answer_block(solution_str)
    if block is not None:
        parsed = _try_parse_first_json_obj(block)
        return (parsed, "success_answer_tag") if parsed is not None else (None, "answer_tag_json_error")
    parsed = _try_parse_first_json_obj(solution_str)
    return (parsed, "success_direct_json") if parsed is not None else (None, "parsing_failed")


def _selected_from_ground_truth(gt: Any) -> Optional[str]:
    if isinstance(gt, str): return _norm_option_label(gt)
    if isinstance(gt, dict):
        for k in ("answer", "selected_option", "ground_truth_option"):
            if gt.get(k) is not None: return _norm_option_label(gt[k])
    return None


def _selected_from_prediction(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict): return None
    sol = payload.get("solution") or {}
    if isinstance(sol, dict) and sol.get("selected_option") is not None:
        return _norm_option_label(sol["selected_option"])
    return None


def _infer_n_values(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict): return None
    wm = payload.get("world_model") or {}; domains = wm.get("domains", {}) if isinstance(wm, dict) else {}
    raw_values = domains.get("values") or domains.get("assignments") or domains.get("projects") or domains.get("colors") or domains.get("rooms") or domains.get("days") or domains.get("slots") or domains.get("tasks") or domains.get("teams") or []
    if isinstance(raw_values, list) and raw_values: return len(raw_values)
    flat = []
    if isinstance(domains, dict):
        for v in domains.values():
            if isinstance(v, list): flat.extend(v)
    return len(set(str(x) for x in flat)) if flat else None


def _infer_n_entities(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict): return None
    wm = payload.get("world_model") or {}; entities = wm.get("entities", []) if isinstance(wm, dict) else []
    return len(entities) if isinstance(entities, list) and entities else None


def _schema_ok(payload: Optional[Dict[str, Any]]) -> bool:
    """Minimal schema for the simplified Assignment prompt.

    Expected output now only needs:
      {
        "reasoning": ["natural-language sentence.", ...],
        "solution": {"selected_option": "A"}
      }

    We no longer require problem_type, world_model, rules, facts, question_semantics,
    options, or formal interleaved solver steps.
    """
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("reasoning"), list):
        return False
    sol = payload.get("solution") or {}
    if not isinstance(sol, dict):
        return False
    return _norm_option_label(sol.get("selected_option")) is not None


def _reasoning_format_ok(reasoning: Any) -> bool:
    """Check the simplified natural-language reasoning format.

    New prompt requirement: reasoning is a list of natural-language strings only.
    No S<k> formal steps are required or expected.
    """
    if not isinstance(reasoning, list) or len(reasoning) == 0:
        return False
    step_re = re.compile(r"^\s*S\d+\s*:", re.IGNORECASE)
    for item in reasoning:
        if not isinstance(item, str) or not item.strip():
            return False
        if step_re.match(item):
            return False
        if not item.strip().endswith("."):
            return False
    return True

def compute_score(solution_str, ground_truth, extra_info: Any = None, score_method: str = "gt", timeout: float = 3.0, acc_weight: float = 0.8, clue_weight: float = 1.0, z3_weight: float = 0.2, meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Simplified Assignment reward.

    New expected output format:
      {
        "reasoning": ["natural-language sentence.", ...],
        "solution": {"selected_option": "E"}
      }

    Reward is intentionally simple:
      reward = 0.6 * accuracy

    We still return the same numeric key set for VERL compatibility. Z3/process
    metrics are kept as numeric zeros because the new prompt no longer asks for
    formal rules/options/reasoning steps.
    """
    try:
        out: Dict[str, Any] = _default_result(reward=0.0, missed_data=0.0)

        payload, parse_status = parse_ar_lsat_answer(solution_str)

        parsing_reward = 1.0 if parse_status == "success_answer_tag" else 0.5 if parse_status == "success_direct_json" else 0.0
        out["parsing_reward"] = parsing_reward
        out["parse_status_ok"] = 1.0 if parse_status == "success_answer_tag" else 0.0
        out["parse_error_flag"] = 0.0 if parse_status in {"success_answer_tag", "success_direct_json"} else 1.0

        selected = _selected_from_prediction(payload)
        gt = _selected_from_ground_truth(ground_truth)
        out["selected_option_present"] = 1.0 if selected else 0.0
        out["ground_truth_present"] = 1.0 if gt else 0.0

        accuracy = 1.0 if selected and gt and selected == gt else 0.0
        out["ACCURACY"] = float(accuracy)

        schema_reward = 1.0 if _schema_ok(payload) else 0.0
        out["schema_reward"] = schema_reward
        out["schema_status_ok"] = schema_reward

        reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
        format_reward = 1.0 if _reasoning_format_ok(reasoning) else 0.0
        out["format_reward"] = format_reward
        out["format_status_ok"] = format_reward

        # Z3/process fields intentionally remain zero for the simplified prompt.
        out["z3_reward"] = 0.0
        out["BASE_sat_full_GT"] = 0.0
        out["consistency_score"] = 0.0
        out["BASE_n_steps_total"] = 0.0
        out["BASE_n_steps_parsed_ok"] = 0.0
        out["BASE_n_steps_valid"] = 0.0
        out["BASE_n_steps_novel_inc_clues"] = 0.0
        out["BASE_n_non_valid_contradiction"] = 0.0
        out["novel_step_score"] = 0.0
        out["contradiction_ratio"] = 0.0
        out["Normalizer"] = 1.0
        out["missed_data"] = 0.0 if isinstance(payload, dict) else 1.0

        reward = 0.6 * float(accuracy)
        out["acc"] = out["score"] = out["reward_logged"] = _clamp_reward(reward)
        return _numeric_only(out)
    except Exception:
        logger.exception("assignment compute_score failed; returning complete penalty reward dict")
        out = _default_result(reward=-0.5, missed_data=1.0)
        out["reward_exception"] = 1.0
        return out

def _wrap(payload: Dict[str, Any]) -> str:
    return "<answer>" + json.dumps(payload, ensure_ascii=False, indent=2) + "</answer>"


def _make_answer(selected: str = "E", bad_format: bool = False) -> str:
    reasoning = [
        "Exactly one employee is assigned to P2.",
        "B and C must share the same project.",
        "This forces both B and C to be assigned consistently.",
        "Option E is always true under all valid assignments.",
    ]
    if bad_format:
        reasoning = ["S1: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))."]
    payload = {
        "reasoning": reasoning,
        "solution": {"selected_option": selected},
    }
    return _wrap(payload)


def _make_missing_solution_answer() -> str:
    payload = {
        "reasoning": ["Exactly one employee is assigned to P2."],
    }
    return _wrap(payload)

if __name__ == "__main__":
    tests = [
        ("correct_answer", _make_answer("E"), "E"),
        ("wrong_answer", _make_answer("A"), "E"),
        ("option_prefix_correct", _make_answer("Option_E"), "E"),
        ("bad_reasoning_format_correct_answer", _make_answer("E", bad_format=True), "E"),
        ("missing_solution", _make_missing_solution_answer(), "E"),
        ("malformed_json", "<answer>{bad json</answer>", "E"),
        ("none_output", None, "E"),
    ]
    for name, pred, gt in tests:
        print(f"\n=== {name} ===")
        result = compute_score(pred, gt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        assert set(result.keys()) == set(RESULT_KEYS)
        assert all(isinstance(v, float) for v in result.values())
