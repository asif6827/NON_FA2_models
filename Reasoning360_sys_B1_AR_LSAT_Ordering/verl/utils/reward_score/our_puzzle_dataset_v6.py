# -*- coding: utf-8 -*-
"""Safe reward scoring for AR-LSAT ORDERING outputs.

This version returns a complete numeric metric dictionary for every sample and
adds Z3 diagnostic counters to debug low BASE_sat_full_GT.
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
        def check_interleaved_reasoning(reasoning, *, n_houses):
            return False

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

RESULT_KEYS = [
    "acc", "score", "reward_logged", "ACCURACY", "parsing_reward", "schema_reward", "format_reward",
    "z3_reward", "consistency_score", "Normalizer", "BASE_sat_full_GT", "missed_data",
    "BASE_n_steps_total", "BASE_n_steps_parsed_ok", "BASE_n_steps_valid", "BASE_n_steps_novel_inc_clues", "BASE_n_non_valid_contradiction",
    "novel_step_score", "contradiction_ratio", "selected_option_present", "ground_truth_present", "parse_status_ok", "schema_status_ok", "z3_status_ok", "format_status_ok",
    "z3_base_sat", "z3_solver_selected_ok", "z3_gt_match", "z3_rule_parse_error_count", "z3_fact_parse_error_count", "z3_option_parse_error_count", "z3_selected_option_parse_ok",
    "reward_exception", "epoch", "total_epochs",
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
    if isinstance(gt, str): return gt.strip().upper()
    if isinstance(gt, dict):
        for k in ("answer", "selected_option", "ground_truth_option"):
            if gt.get(k) is not None: return str(gt[k]).strip().upper()
    return None


def _selected_from_prediction(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict): return None
    sol = payload.get("solution") or {}
    if isinstance(sol, dict) and sol.get("selected_option") is not None:
        return str(sol["selected_option"]).strip().upper()
    return None


def _infer_n_positions(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict): return None
    wm = payload.get("world_model") or {}; domains = wm.get("domains", {}) if isinstance(wm, dict) else {}
    vals = []
    for p in (domains.get("positions") or domains.get("position") or []):
        try: vals.append(int(str(p)))
        except Exception: pass
    if vals: return len(set(vals))
    entities = wm.get("entities", []) if isinstance(wm, dict) else []
    return len(entities) if isinstance(entities, list) and entities else None


def _infer_n_entities(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict): return None
    wm = payload.get("world_model") or {}; entities = wm.get("entities", []) if isinstance(wm, dict) else []
    return len(entities) if isinstance(entities, list) and entities else None


def _schema_ok(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict): return False
    required = ["problem_type", "world_model", "rules", "facts", "question_semantics", "options", "reasoning", "solution"]
    if any(k not in payload for k in required): return False
    if payload.get("problem_type") != "ordering": return False
    return isinstance(payload.get("world_model"), dict) and isinstance(payload.get("rules"), list) and isinstance(payload.get("facts"), list) and isinstance(payload.get("question_semantics"), dict) and isinstance(payload.get("options"), dict) and isinstance(payload.get("reasoning"), list) and isinstance(payload.get("solution"), dict) and bool(payload["solution"].get("selected_option"))


def compute_score(solution_str, ground_truth, extra_info: Any = None, score_method: str = "gt", timeout: float = 3.0, acc_weight: float = 0.8, clue_weight: float = 1.0, z3_weight: float = 0.2, meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    try:
        out: Dict[str, Any] = _default_result(reward=0.0, missed_data=0.0)
        payload, parse_status = parse_ar_lsat_answer(solution_str)
        parsing_reward = 1.0 if parse_status == "success_answer_tag" else 0.5 if parse_status == "success_direct_json" else 0.0
        out["parsing_reward"] = parsing_reward; out["parse_status_ok"] = 1.0 if parse_status == "success_answer_tag" else 0.0
        selected = _selected_from_prediction(payload); gt = _selected_from_ground_truth(ground_truth)
        out["selected_option_present"] = 1.0 if selected else 0.0; out["ground_truth_present"] = 1.0 if gt else 0.0
        accuracy = 1.0 if selected and gt and selected == gt else 0.0
        out["ACCURACY"] = accuracy
        schema_reward = 1.0 if _schema_ok(payload) else 0.0
        out["schema_reward"] = out["schema_status_ok"] = schema_reward
        reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
        n_positions = _infer_n_positions(payload); n_entities = _infer_n_entities(payload)
        try:
            format_ok = check_interleaved_reasoning(reasoning, n_houses=int(n_positions or 0))
        except Exception:
            format_ok = False
        out["format_reward"] = out["format_status_ok"] = 1.0 if format_ok else 0.0
        z3_out: Dict[str, Any] = {}
        if isinstance(payload, dict) and schema_reward > 0.0:
            z3_payload = dict(payload); z3_payload["ground_truth"] = ground_truth
            if isinstance(extra_info, dict) and extra_info.get("question_type"):
                z3_payload["question_type"] = extra_info["question_type"]
            try: z3_out = solve_and_validate_payload(z3_payload, timeout_s=timeout, conflict_tolerant_clues=False)
            except Exception as e: z3_out = {"parse_status": "Z3_EXCEPTION", "error": f"{type(e).__name__}: {e}"}
        z3_status = str(z3_out.get("parse_status", ""))
        out["z3_status_ok"] = 1.0 if z3_status.endswith("SUCCESS") else 0.0
        out["z3_base_sat"] = 1.0 if bool(z3_out.get("base_sat", False)) else 0.0
        out["z3_solver_selected_ok"] = 1.0 if bool(z3_out.get("solver_selected_ok", False)) else 0.0
        out["z3_gt_match"] = 1.0 if bool(z3_out.get("gt_match", False)) else 0.0
        out["z3_rule_parse_error_count"] = float(z3_out.get("n_rule_parse_errors", 0) or 0)
        out["z3_fact_parse_error_count"] = float(z3_out.get("n_fact_parse_errors", 0) or 0)
        out["z3_option_parse_error_count"] = float(z3_out.get("n_option_parse_errors", 0) or 0)
        out["z3_selected_option_parse_ok"] = 1.0 if bool(z3_out.get("selected_option_parse_ok", False)) else 0.0
        sat_ok = 1.0 if bool(z3_out.get("base_sat_full_GT", False)) else 0.0
        out["z3_reward"] = out["BASE_sat_full_GT"] = sat_ok
        out["consistency_score"] = float(z3_out.get("consistency_score", 0.0) or 0.0)
        out["BASE_n_steps_total"] = float(z3_out.get("n_steps_total", 0) or 0)
        out["BASE_n_steps_parsed_ok"] = float(z3_out.get("n_steps_parsed_ok", 0) or 0)
        out["BASE_n_steps_valid"] = float(z3_out.get("n_steps_valid", 0) or 0)
        out["BASE_n_steps_novel_inc_clues"] = float(z3_out.get("n_steps_novel_inc_clues", 0) or 0)
        out["BASE_n_non_valid_contradiction"] = float(z3_out.get("n_non_valid_contradiction", 0) or 0)
        reward, normalizer = 0.0, 1.0
        if isinstance(payload, dict) and n_positions is not None and n_entities is not None:
            normalizer = max(2.0 * max(int(n_positions) * int(n_entities), 1), 1.0)
            n_novel = out["BASE_n_steps_novel_inc_clues"]; n_contra = out["BASE_n_non_valid_contradiction"]
            novel_step_score = min(n_novel / normalizer, 1.0); contradiction_ratio = min(n_contra / normalizer, 1.0)

            reward = 0.15 * parsing_reward + 0.10 * out["format_reward"] + 0.60 * accuracy
            out["novel_step_score"] = novel_step_score; out["contradiction_ratio"] = contradiction_ratio
        else:
            reward = 0.0; out["missed_data"] = 1.0
        out["Normalizer"] = normalizer
        out["acc"] = out["score"] = out["reward_logged"] = _clamp_reward(reward)
        return _numeric_only(out)
    except Exception:
        logger.exception("compute_score failed; returning complete penalty reward dict")
        out = _default_result(reward=-0.5, missed_data=1.0); out["reward_exception"] = 1.0; return out


def _wrap(payload: Dict[str, Any]) -> str:
    return "<answer>" + json.dumps(payload, ensure_ascii=False, indent=2) + "</answer>"


def _make_answer(selected: str = "A") -> str:
    payload = {"problem_type": "ordering", "world_model": {"entities": ["A", "B", "C", "D"], "domains": {"positions": ["1", "2", "3", "4"]}}, "rules": ["Distinct(A, B, C, D)", "A < B", "C == A + 1", "D != 1"], "facts": ["B == 4"], "question_semantics": {"question_type": "could_be_true"}, "options": {"A": "A == 2", "B": "C == 4", "C": "D == 2"}, "reasoning": ["The question fixes B in fourth position.", "S1: B == 4.", "C cannot be fourth because B is already fourth.", "S2: C != 4.", "Option A can be extended to a valid ordering.", "S3: Sat(Option_A)."], "solution": {"selected_option": selected}}
    return _wrap(payload)


if __name__ == "__main__":
    for name, pred, gt in [("correct_could_be_true", _make_answer("A"), "A"), ("wrong_selected_option", _make_answer("B"), "A")]:
        print("\n===", name, "===")
        print(json.dumps(compute_score(pred, gt), indent=2))
