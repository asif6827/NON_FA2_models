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
    if not isinstance(payload, dict): return False
    required = ["problem_type", "world_model", "rules", "facts", "question_semantics", "options", "reasoning", "solution"]
    if any(k not in payload for k in required): return False
    if str(payload.get("problem_type") or "").strip().lower() != "assignment": return False
    return isinstance(payload.get("world_model"), dict) and isinstance(payload.get("rules"), list) and isinstance(payload.get("facts"), list) and isinstance(payload.get("question_semantics"), dict) and isinstance(payload.get("options"), dict) and isinstance(payload.get("reasoning"), list) and isinstance(payload.get("solution"), dict) and bool(payload["solution"].get("selected_option"))


def compute_score(solution_str, ground_truth, extra_info: Any = None, score_method: str = "gt", timeout: float = 3.0, acc_weight: float = 0.8, clue_weight: float = 1.0, z3_weight: float = 0.2, meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    try:
        out: Dict[str, Any] = _default_result(reward=0.0, missed_data=0.0)
        payload, parse_status = parse_ar_lsat_answer(solution_str)
        parsing_reward = 1.0 if parse_status == "success_answer_tag" else 0.5 if parse_status == "success_direct_json" else 0.0
        out["parsing_reward"] = parsing_reward; out["parse_status_ok"] = 1.0 if parse_status == "success_answer_tag" else 0.0
        out["parse_error_flag"] = 0.0 if parse_status in {"success_answer_tag", "success_direct_json"} else 1.0
        selected = _selected_from_prediction(payload); gt = _selected_from_ground_truth(ground_truth)
        out["selected_option_present"] = 1.0 if selected else 0.0; out["ground_truth_present"] = 1.0 if gt else 0.0
        accuracy = 1.0 if selected and gt and selected == gt else 0.0
        out["ACCURACY"] = accuracy
        schema_reward = 1.0 if _schema_ok(payload) else 0.0
        out["schema_reward"] = out["schema_status_ok"] = schema_reward
        reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
        n_values = _infer_n_values(payload); n_entities = _infer_n_entities(payload)
        try:
            format_ok = check_interleaved_reasoning(reasoning, n_houses=int(n_values or 0))
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
        if isinstance(payload, dict) and n_values is not None and n_entities is not None:
            normalizer = max(2.0 * max(int(n_values) * int(n_entities), 1), 1.0)
            n_novel = out["BASE_n_steps_novel_inc_clues"]; n_contra = out["BASE_n_non_valid_contradiction"]
            novel_step_score = min(n_novel / normalizer, 1.0); contradiction_ratio = min(n_contra / normalizer, 1.0)
            if sat_ok == 0.0:
                reward = 0.15 * parsing_reward + 0.10 * out["format_reward"] + 0.60 * accuracy - 0.20 * contradiction_ratio
            else:
                base_quality = 0.60 * accuracy + 0.20 * parsing_reward + 0.20 * out["format_reward"]
                process_bonus = 0.00 * novel_step_score + 0.30 * out["consistency_score"] - 0.15 * contradiction_ratio
                reward = base_quality + accuracy * process_bonus
            out["novel_step_score"] = novel_step_score; out["contradiction_ratio"] = contradiction_ratio
        else:
            reward = -0.5; out["missed_data"] = 1.0
        out["Normalizer"] = normalizer
        out["acc"] = out["score"] = out["reward_logged"] = _clamp_reward(reward)
        return _numeric_only(out)
    except Exception:
        logger.exception("assignment compute_score failed; returning complete penalty reward dict")
        out = _default_result(reward=-0.5, missed_data=1.0); out["reward_exception"] = 1.0; return out


def _wrap(payload: Dict[str, Any]) -> str:
    return "<answer>" + json.dumps(payload, ensure_ascii=False, indent=2) + "</answer>"


def _make_answer(selected: str = "A", bad_format: bool = False) -> str:
    reasoning = ["A is not assigned to P1 by the first rule.", "S1: Not(Assign(A, P1)).", "Exactly one employee is assigned to P2.", "S2: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2)).", "Option A can be extended to a full valid assignment.", "S3: Sat(Option_A)."]
    if bad_format: reasoning = ["S1: Not(Assign(A, P1)).", "This starts with a formal step, so format should fail."]
    payload = {"problem_type": "assignment", "world_model": {"entities": ["A", "B", "C"], "domains": {"values": ["P1", "P2", "P3"]}, "structural_assumptions": ["each entity is assigned exactly one value"]}, "rules": ["Not(Assign(A, P1))", "Assign(B, P1) == Assign(C, P1)", "Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))"], "facts": [], "question_semantics": {"question_type": "could_be_true"}, "options": {"Option_A": "Assign(A, P2)", "B": "Assign(A, P1)", "C": "Assign(B, P2)"}, "reasoning": reasoning, "solution": {"selected_option": selected}}
    return _wrap(payload)


def _make_must_be_true_answer() -> str:
    payload = {"problem_type": "assignment", "world_model": {"entities": ["A", "B"], "domains": {"values": ["P1", "P2"]}, "structural_assumptions": ["each entity is assigned exactly one value"]}, "rules": ["Assign(A, P1)"], "facts": [], "question_semantics": {"question_type": "must_be_true"}, "options": {"A": "Assign(A, P1)", "B": "Assign(A, P2)"}, "reasoning": ["The passage directly fixes A to P1.", "S1: Assign(A, P1).", "Option A is forced by all valid assignments.", "S2: Unsat(Not(Option_A))."], "solution": {"selected_option": "Option_A"}}
    return _wrap(payload)


if __name__ == "__main__":
    tests = [("correct_could_be_true", _make_answer("Option_A"), "A"), ("wrong_selected_option", _make_answer("B"), "A"), ("bad_format_correct_answer", _make_answer("A", bad_format=True), "A"), ("must_be_true", _make_must_be_true_answer(), "A"), ("malformed_json", "<answer>{bad json</answer>", "A"), ("none_output", None, "A")]
    for name, pred, gt in tests:
        print(f"\n=== {name} ===")
        result = compute_score(pred, gt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        assert set(result.keys()) == set(RESULT_KEYS)
        assert all(isinstance(v, float) for v in result.values())
