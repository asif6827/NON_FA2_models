# -*- coding: utf-8 -*-
"""Reward scoring for AR-LSAT ordering outputs.

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

# Prefer sibling/local files first.
try:
    from z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
except Exception:
    from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload

try:
    from check_interleved_format import check_interleaved_reasoning
except Exception:
    from verl.utils.reward_score.check_interleved_format import check_interleaved_reasoning

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)
job_id = os.getenv("SLURM_JOB_ID", "local")


def find_last_answer_block(text: str) -> Optional[str]:
    pattern = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", flags=re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(1).strip()


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

    starts = [m.start() for m in re.finditer(r"\{", raw)]
    for st in starts:
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
        if parsed is not None:
            return parsed, "success_answer_tag"
        return None, "answer_tag_json_error"

    parsed = _try_parse_first_json_obj(solution_str)
    if parsed is not None:
        return parsed, "success_direct_json"
    return None, "parsing_failed"


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
    if isinstance(sol, dict) and sol.get("selected_option"):
        return str(sol["selected_option"]).strip().upper()
    return None


def _infer_n_positions(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    wm = payload.get("world_model") or {}
    domains = wm.get("domains", {}) if isinstance(wm, dict) else {}
    positions = domains.get("positions") or domains.get("position") or []
    vals = []
    for p in positions:
        try:
            vals.append(int(str(p)))
        except Exception:
            pass
    if vals:
        return max(vals)
    entities = wm.get("entities", []) if isinstance(wm, dict) else []
    if entities:
        return len(entities)
    return None


def _infer_n_entities(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    wm = payload.get("world_model") or {}
    entities = wm.get("entities", []) if isinstance(wm, dict) else []
    if isinstance(entities, list) and entities:
        return len(entities)
    return None


def _schema_ok(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    required = [
        "problem_type",
        "world_model",
        "rules",
        "facts",
        "question_semantics",
        "options",
        "reasoning",
        "solution",
    ]
    if any(k not in payload for k in required):
        return False
    if payload.get("problem_type") != "ordering":
        return False
    if not isinstance(payload.get("world_model"), dict):
        return False
    if not isinstance(payload.get("rules"), list):
        return False
    if not isinstance(payload.get("facts"), list):
        return False
    if not isinstance(payload.get("question_semantics"), dict):
        return False
    if not isinstance(payload.get("options"), dict):
        return False
    if not isinstance(payload.get("reasoning"), list):
        return False
    if not isinstance(payload.get("solution"), dict):
        return False
    return bool(payload["solution"].get("selected_option"))


def compute_score(
    solution_str,
    ground_truth,
    extra_info: Any = None,
    score_method: str = "gt",
    timeout: float = 3.0,
    acc_weight: float = 0.8,
    clue_weight: float = 1.0,
    z3_weight: float = 0.2,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute reward for AR-LSAT ordering answer-option tasks.

    Zebra mapping:
      puzzle_acc_score -> ACCURACY = 1 iff selected_option == ground_truth option else 0.
      n_houses -> number of ordering positions.
      n_attrs -> number of ordered entities.
      consistency_score -> valid reasoning option-status step supports final selected_option.
    """
    epoch = int(os.getenv("CURRENT_EPOCH", "0"))
    total_epochs = int(os.getenv("TOTAL_EPOCH", "1"))

    final_result: Dict[str, Any] = {
        "acc": 0.0,
        "score": 0.0,
        "reward_logged": 0.0,
        "ACCURACY": 0.0,
        "parsing_reward": 0.0,
        "schema_reward": 0.0,
        "format_reward": 0.0,
        "z3_reward": 0.0,
        "consistency_score": 0.0,
        "Normalizer": 1.0,
        "BASE_sat_full_GT": 0.0,
        "missed_data": 0.0,
        "BASE_n_steps_total": 0.0,
        "BASE_n_steps_parsed_ok": 0.0,
        "BASE_n_steps_valid": 0.0,
        "BASE_n_steps_novel_inc_clues": 0.0,
        "BASE_n_non_valid_contradiction": 0.0,
        "novel_step_score": 0.0,
        "contradiction_ratio": 0.0,
        "selected_option": None,
        "ground_truth_option": _selected_from_ground_truth(ground_truth),
        "parse_status": "INIT",
        "z3_parse_status": "NOT_RUN",
        "z3_error": "",
        "format_error": "",
        "reward_error": "",
        "epoch": epoch,
        "total_epochs": total_epochs,
    }

    payload, parse_status = parse_ar_lsat_answer(solution_str)
    final_result["parse_status"] = parse_status

    if parse_status == "success_answer_tag":
        parsing_reward = 1.0
    elif parse_status == "success_direct_json":
        parsing_reward = 0.5
    else:
        parsing_reward = 0.0
    final_result["parsing_reward"] = parsing_reward

    selected = _selected_from_prediction(payload)
    gt_selected = final_result["ground_truth_option"]
    final_result["selected_option"] = selected

    # AR-LSAT answer-option accuracy replaces Zebra Puzzle/Cell accuracy.
    accuracy = 1.0 if (selected is not None and gt_selected is not None and selected == gt_selected) else 0.0
    final_result["ACCURACY"] = float(accuracy)

    schema_reward = 1.0 if _schema_ok(payload) else 0.0
    final_result["schema_reward"] = schema_reward

    reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
    n_positions = _infer_n_positions(payload)
    n_entities = _infer_n_entities(payload)

    try:
        format_ok = check_interleaved_reasoning(reasoning, n_houses=int(n_positions or 0))
    except Exception as e:
        final_result["format_error"] = f"{type(e).__name__}: {e}"
        format_ok = False
    format_reward = 1.0 if format_ok else 0.0
    final_result["format_reward"] = format_reward

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

    final_result["z3_parse_status"] = z3_out.get("parse_status", "NOT_RUN")
    final_result["z3_error"] = z3_out.get("error", "")

    sat_ok = 1.0 if bool(z3_out.get("base_sat_full_GT", False)) else 0.0
    consistency_score = float(z3_out.get("consistency_score", 0.0) or 0.0)
    final_result["z3_reward"] = sat_ok
    final_result["consistency_score"] = consistency_score
    #final_result["solution_support_steps"] = z3_out.get("solution_support_steps", [])
    final_result["BASE_sat_full_GT"] = sat_ok
    final_result["BASE_n_steps_total"] = float(z3_out.get("n_steps_total", 0) or 0)
    final_result["BASE_n_steps_parsed_ok"] = float(z3_out.get("n_steps_parsed_ok", 0) or 0)
    final_result["BASE_n_steps_valid"] = float(z3_out.get("n_steps_valid", 0) or 0)
    final_result["BASE_n_steps_novel_inc_clues"] = float(z3_out.get("n_steps_novel_inc_clues", 0) or 0)
    final_result["BASE_n_non_valid_contradiction"] = float(z3_out.get("n_non_valid_contradiction", 0) or 0)

    # -----------------------
    # Reward components: Zebra-style formula adapted to AR-LSAT ordering.
    # -----------------------
    try:
        reward = 0.0
        normalizer = 1.0
        n_novel_steps = float(final_result.get("BASE_n_steps_novel_inc_clues", 0.0))
        has_required_inputs = (
            isinstance(payload, dict)
            and n_positions is not None
            and n_entities is not None
            and n_novel_steps > 0
        )

        if has_required_inputs:
            n_houses_i = max(int(n_positions), 0)
            n_attrs_i = max(int(n_entities), 0)
            normalizer = max(2.0 * max(n_houses_i * n_attrs_i, 1), 1.0)

            n_contradictions = float(final_result.get("BASE_n_non_valid_contradiction", 0.0))
            novel_step_score = float(min(n_novel_steps / normalizer, 1.0))
            contradiction_ratio = float(min(n_contradictions / normalizer, 1.0))
            sat_ok = float(final_result.get("BASE_sat_full_GT", 0.0))
            consistency_score = float(final_result.get("consistency_score", 0.0))

            if sat_ok == 0.0:
                reward = (
                    0.15 * parsing_reward
                    + 0.10 * format_reward
                    + 0.60 * float(accuracy)
                    - 0.20 * contradiction_ratio
                )
            else:
                base_quality = (
                    0.60 * float(accuracy)
                    + 0.20 * parsing_reward
                    + 0.20 * format_reward
                )
                process_bonus = (
                    0.40 * novel_step_score
                    + 0.30 * consistency_score
                    - 0.15 * contradiction_ratio
                )
                reward = base_quality + float(accuracy) * process_bonus

            final_result["novel_step_score"] = float(novel_step_score)
            final_result["contradiction_ratio"] = float(contradiction_ratio)
        else:
            reward = -0.5
            final_result["missed_data"] = 1.0
            final_result["novel_step_score"] = 0.0
            final_result["contradiction_ratio"] = 0.0

        final_result["Normalizer"] = float(normalizer)
        final_result["acc"] = float(reward)
        final_result["score"] = float(reward)
        final_result["reward_logged"] = float(reward)
    except Exception as e:
        logger.exception("Crash in final reward scoring")
        final_result["reward_error"] = f"{type(e).__name__}: {e}"
        final_result["acc"] = 0.0
        final_result["score"] = 0.0
        final_result["reward_logged"] = 0.0

    return final_result


def _make_answer(selected: str = "A", bad_format: bool = False) -> str:
    reasoning = [
        "The question condition fixes B in the fourth position.",
        "S1: B == 4.",
        "Since C is immediately after A and B is fourth, C cannot be fourth.",
        "S2: C != 4.",
        "Since C must immediately follow A, A cannot be third.",
        "S3: Not(A == 3).",
        "Option A can be extended to a full valid ordering.",
        "S4: Sat(Option_A).",
    ]
    if bad_format:
        reasoning = ["S1: B == 4.", "This starts with a formal step, so format should fail."]
    payload = {
        "problem_type": "ordering",
        "world_model": {
            "entities": ["A", "B", "C", "D"],
            "domains": {"positions": ["1", "2", "3", "4"]},
            "structural_assumptions": ["each speaker occupies exactly one position"],
        },
        "rules": ["Distinct(A, B, C, D)", "A < B", "C == A + 1", "D != 1"],
        "facts": ["B == 4"],
        "question_semantics": {"question_type": "could_be_true", "option_interpretation_rule": "SAT(option)"},
        "options": {"A": "A == 2", "B": "C == 4", "C": "D == 2", "D": "A == 3", "E": "C == 1"},
        "reasoning": reasoning,
        "solution": {"selected_option": selected},
    }
    return "<answer>" + json.dumps(payload, ensure_ascii=False, indent=2) + "</answer>"


def _make_must_be_true_answer() -> str:
    payload = {
        "problem_type": "ordering",
        "world_model": {
            "entities": ["A", "B", "C"],
            "domains": {"positions": ["1", "2", "3"]},
            "structural_assumptions": ["each entity occupies exactly one position"],
        },
        "rules": ["Distinct(A, B, C)", "A < B", "B < C"],
        "facts": [],
        "question_semantics": {"question_type": "must_be_true", "option_interpretation_rule": "UNSAT(Not(option))"},
        "options": {"A": "A < C", "B": "C < A", "C": "B == 1"},
        "reasoning": [
            "Since A is before B and B is before C, A must be before C.",
            "S1: A < C.",
            "Option A is forced by all valid orderings.",
            "S2: Unsat(Option_B).",
        ],
        "solution": {"selected_option": "A"},
    }
    return "<answer>" + json.dumps(payload, ensure_ascii=False, indent=2) + "</answer>"


if __name__ == "__main__":
    tests = [
        ("correct_could_be_true", _make_answer("A"), "A"),
        ("wrong_selected_option", _make_answer("B"), "A"),
        ("bad_format_correct_answer", _make_answer("A", bad_format=True), "A"),
        ("must_be_true", _make_must_be_true_answer(), "A"),
    ]
    for name, pred, gt in tests:
        print(f"\n=== {name} ===")
        print(json.dumps(compute_score(pred, gt), indent=2, ensure_ascii=False))
