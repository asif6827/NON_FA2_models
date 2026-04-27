# -*- coding: utf-8 -*-
"""Reward scoring for AR-LSAT assignment outputs.

Expected model output is one JSON object inside <answer>...</answer> with fields:
problem_type, world_model, rules, facts, question_semantics, options, reasoning, solution.

This is adapted from the ZebraPuzzle process-reward formulation, but replaces
Puzzle/Cell Accuracy with answer-option accuracy: selected option vs ground truth.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional

try:
    from z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
except Exception:
    from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload

try:
    from check_interleved_format import check_interleaved_reasoning
except Exception:
    from verl.utils.reward_score.check_interleved_format import check_interleaved_reasoning

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)
job_id = os.getenv("SLURM_JOB_ID", "local")


def find_last_answer_block(text: str) -> Optional[str]:
    pattern = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", flags=re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(text or ""))
    return matches[-1].group(1).strip() if matches else None


def _try_parse_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    for st in [m.start() for m in re.finditer(r"\{", raw)]:
        for ed in range(len(raw), st, -1):
            if raw[ed - 1] != "}":
                continue
            try:
                obj = json.loads(raw[st:ed])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def parse_ar_lsat_answer(solution_str: str):
    answer_content = find_last_answer_block(solution_str)
    if answer_content is not None:
        parsed = _try_parse_first_json_obj(answer_content)
        return (parsed, "success_answer_tag") if parsed is not None else (None, "answer_tag_json_error")
    parsed = _try_parse_first_json_obj(solution_str)
    return (parsed, "success_direct_json") if parsed is not None else (None, "parsing_failed")


def _selected_from_ground_truth(ground_truth: Any) -> Optional[str]:
    if isinstance(ground_truth, str):
        return ground_truth.strip().upper()
    if isinstance(ground_truth, dict):
        for key in ("answer", "selected_option", "ground_truth_option"):
            if ground_truth.get(key):
                return str(ground_truth[key]).strip().upper()
    return None


def _selected_from_prediction(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    sol = payload.get("solution") or {}
    return str(sol["selected_option"]).strip().upper() if isinstance(sol, dict) and sol.get("selected_option") else None


def _infer_n_values(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    wm = payload.get("world_model") or {}
    domains = wm.get("domains", {}) if isinstance(wm, dict) else {}
    raw_values = domains.get("values") or domains.get("assignments") or domains.get("projects") or domains.get("colors") or domains.get("rooms") or domains.get("days") or []
    if isinstance(raw_values, list) and raw_values:
        return len(raw_values)
    flat = []
    if isinstance(domains, dict):
        for v in domains.values():
            if isinstance(v, list):
                flat.extend(v)
    return len(set(str(x) for x in flat)) if flat else None


def _infer_n_entities(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    wm = payload.get("world_model") or {}
    entities = wm.get("entities", []) if isinstance(wm, dict) else []
    return len(entities) if isinstance(entities, list) and entities else None


def _schema_ok(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    required = ["problem_type", "world_model", "rules", "facts", "question_semantics", "options", "reasoning", "solution"]
    if any(k not in payload for k in required):
        return False
    if payload.get("problem_type") != "assignment":
        return False
    return (
        isinstance(payload.get("world_model"), dict)
        and isinstance(payload.get("rules"), list)
        and isinstance(payload.get("facts"), list)
        and isinstance(payload.get("question_semantics"), dict)
        and isinstance(payload.get("options"), dict)
        and isinstance(payload.get("reasoning"), list)
        and isinstance(payload.get("solution"), dict)
        and bool(payload["solution"].get("selected_option"))
    )


def _numeric_only(final_result: Dict[str, Any]) -> Dict[str, float]:
    numeric_result: Dict[str, float] = {}
    for k, v in final_result.items():
        try:
            if isinstance(v, bool):
                numeric_result[k] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                numeric_result[k] = float(v)
            else:
                numeric_result[k] = float(v)
        except Exception:
            numeric_result[k] = 0.0
    return numeric_result


def compute_score(solution_str, ground_truth, extra_info: Any = None, score_method: str = "gt", timeout: float = 3.0, acc_weight: float = 0.8, clue_weight: float = 1.0, z3_weight: float = 0.2, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    epoch = int(os.getenv("CURRENT_EPOCH", "0"))
    total_epochs = int(os.getenv("TOTAL_EPOCH", "1"))
    final_result: Dict[str, Any] = {
        "acc": 0.0, "score": 0.0, "reward_logged": 0.0, "ACCURACY": 0.0,
        "parsing_reward": 0.0, "schema_reward": 0.0, "format_reward": 0.0,
        "z3_reward": 0.0, "consistency_score": 0.0, "Normalizer": 1.0,
        "BASE_sat_full_GT": 0.0, "missed_data": 0.0,
        "BASE_n_steps_total": 0.0, "BASE_n_steps_parsed_ok": 0.0,
        "BASE_n_steps_valid": 0.0, "BASE_n_steps_novel_inc_clues": 0.0,
        "BASE_n_non_valid_contradiction": 0.0, "novel_step_score": 0.0,
        "contradiction_ratio": 0.0, "selected_option_present": 0.0,
        "ground_truth_present": 0.0, "parse_status_ok": 0.0, "schema_status_ok": 0.0,
        "z3_status_ok": 0.0, "format_status_ok": 0.0,
        "epoch": float(epoch), "total_epochs": float(total_epochs),
    }
    payload, parse_status = parse_ar_lsat_answer(solution_str)
    parsing_reward = 1.0 if parse_status == "success_answer_tag" else 0.5 if parse_status == "success_direct_json" else 0.0
    final_result["parsing_reward"] = parsing_reward
    final_result["parse_status_ok"] = 1.0 if parse_status == "success_answer_tag" else 0.0

    selected = _selected_from_prediction(payload)
    gt_selected = _selected_from_ground_truth(ground_truth)
    final_result["selected_option_present"] = 1.0 if selected else 0.0
    final_result["ground_truth_present"] = 1.0 if gt_selected else 0.0
    accuracy = 1.0 if (selected is not None and gt_selected is not None and selected == gt_selected) else 0.0
    final_result["ACCURACY"] = float(accuracy)

    schema_reward = 1.0 if _schema_ok(payload) else 0.0
    final_result["schema_reward"] = schema_reward
    final_result["schema_status_ok"] = schema_reward

    reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
    n_values = _infer_n_values(payload)
    n_entities = _infer_n_entities(payload)
    try:
        format_ok = check_interleaved_reasoning(reasoning, n_houses=int(n_values or 1))
    except Exception:
        format_ok = False
    format_reward = 1.0 if format_ok else 0.0
    final_result["format_reward"] = format_reward
    final_result["format_status_ok"] = format_reward

    z3_out: Dict[str, Any] = {}
    if isinstance(payload, dict) and schema_reward > 0.0:
        z3_payload = dict(payload)
        z3_payload["ground_truth"] = ground_truth
        if isinstance(extra_info, dict) and extra_info.get("question_type"):
            z3_payload["question_type"] = extra_info["question_type"]
        try:
            z3_out = solve_and_validate_payload(z3_payload, timeout_s=timeout, conflict_tolerant_clues=False)
        except Exception as e:
            z3_out = {"parse_status": "Z3_EXCEPTION", "error": f"{type(e).__name__}: {e}"}

    final_result["z3_status_ok"] = 1.0 if str(z3_out.get("parse_status", "")).endswith("SUCCESS") else 0.0
    sat_ok = 1.0 if bool(z3_out.get("base_sat_full_GT", False)) else 0.0
    final_result["z3_reward"] = sat_ok
    final_result["BASE_sat_full_GT"] = sat_ok
    final_result["consistency_score"] = float(z3_out.get("consistency_score", 0.0) or 0.0)
    final_result["BASE_n_steps_total"] = float(z3_out.get("n_steps_total", 0) or 0)
    final_result["BASE_n_steps_parsed_ok"] = float(z3_out.get("n_steps_parsed_ok", 0) or 0)
    final_result["BASE_n_steps_valid"] = float(z3_out.get("n_steps_valid", 0) or 0)
    final_result["BASE_n_steps_novel_inc_clues"] = float(z3_out.get("n_steps_novel_inc_clues", 0) or 0)
    final_result["BASE_n_non_valid_contradiction"] = float(z3_out.get("n_non_valid_contradiction", 0) or 0)

    try:
        reward = 0.0
        normalizer = 1.0
        n_novel_steps = float(final_result.get("BASE_n_steps_novel_inc_clues", 0.0))
        has_required_inputs = isinstance(payload, dict) and n_values is not None and n_entities is not None
        if has_required_inputs:
            n_houses_i = max(int(n_values), 0)
            n_attrs_i = max(int(n_entities), 0)
            normalizer = max(2.0 * max(n_houses_i * n_attrs_i, 1), 1.0)
            n_contradictions = float(final_result.get("BASE_n_non_valid_contradiction", 0.0))
            novel_step_score = float(min(n_novel_steps / normalizer, 1.0))
            contradiction_ratio = float(min(n_contradictions / normalizer, 1.0))
            sat_ok = float(final_result.get("BASE_sat_full_GT", 0.0))
            consistency_score = float(final_result.get("consistency_score", 0.0))

            reward = 0.15 * parsing_reward + 0.10 * format_reward + 0.60 * float(accuracy)
        else:
            reward = 0.0
            final_result["missed_data"] = 1.0
        final_result["Normalizer"] = float(normalizer)
        final_result["acc"] = float(reward)
        final_result["score"] = float(reward)
        final_result["reward_logged"] = float(reward)
    except Exception:
        logger.exception("Crash in final reward scoring")
        final_result["acc"] = 0.0
        final_result["score"] = 0.0
        final_result["reward_logged"] = 0.0
    return _numeric_only(final_result)


def _wrap(payload: Dict[str, Any]) -> str:
    return "<answer>" + json.dumps(payload, ensure_ascii=False, indent=2) + "</answer>"


def _make_answer(selected: str = "A", bad_format: bool = False) -> str:
    reasoning = [
        "A is not assigned to P1 by the first rule.", "S1: Not(Assign(A, P1)).",
        "Exactly one employee is assigned to P2.", "S2: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2)).",
        "Option A can be extended to a full valid assignment.", "S3: Sat(Option_A).",
    ]
    if bad_format:
        reasoning = ["S1: Not(Assign(A, P1)).", "This starts with a formal step, so format should fail."]
    payload = {
        "problem_type": "assignment",
        "world_model": {"entities": ["A", "B", "C"], "domains": {"values": ["P1", "P2", "P3"]}, "structural_assumptions": ["each entity is assigned exactly one value"]},
        "rules": ["Not(Assign(A, P1))", "Assign(B, P1) == Assign(C, P1)", "Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))"],
        "facts": [],
        "question_semantics": {"question_type": "could_be_true", "option_interpretation_rule": "SAT(option)"},
        "options": {"A": "Assign(A, P2)", "B": "Assign(A, P1)", "C": "Assign(B, P2)"},
        "reasoning": reasoning,
        "solution": {"selected_option": selected},
    }
    return _wrap(payload)


def _make_must_be_true_answer() -> str:
    payload = {
        "problem_type": "assignment",
        "world_model": {"entities": ["A", "B"], "domains": {"values": ["P1", "P2"]}, "structural_assumptions": ["each entity is assigned exactly one value"]},
        "rules": ["Assign(A, P1)"],
        "facts": [],
        "question_semantics": {"question_type": "must_be_true", "option_interpretation_rule": "UNSAT(Not(option))"},
        "options": {"A": "Assign(A, P1)", "B": "Assign(A, P2)"},
        "reasoning": ["The passage directly fixes A to P1.", "S1: Assign(A, P1).", "Option A is forced by all valid assignments.", "S2: Unsat(Not(Option_A))."],
        "solution": {"selected_option": "A"},
    }
    return _wrap(payload)


if __name__ == "__main__":
    tests = [("correct_could_be_true", _make_answer("A"), "A"), ("wrong_selected_option", _make_answer("B"), "A"), ("bad_format_correct_answer", _make_answer("A", bad_format=True), "A"), ("must_be_true", _make_must_be_true_answer(), "A")]
    for name, pred, gt in tests:
        print(f"\n=== {name} ===")
        print(json.dumps(compute_score(pred, gt), indent=2, ensure_ascii=False))
