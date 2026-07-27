# -*- coding: utf-8 -*-
"""Crash-safe reward scoring for AR-LSAT ORDERING outputs.

The scorer keeps a fixed numeric schema for VERL aggregation. Z3 independently
solves all answer options; model answer accuracy is scored separately.
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
    try:
        from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
    except Exception:
        def solve_and_validate_payload(payload, *, timeout_s=2.0, conflict_tolerant_clues=False):
            return {"parse_status": "Z3_IMPORT_FAIL", "base_sat_full_GT": False}

try:
    from check_interleved_format import (
        check_interleaved_reasoning,
        check_interleaved_reasoning_detailed,
    )
except Exception:
    try:
        from verl.utils.reward_score.check_interleved_format import (
            check_interleaved_reasoning,
            check_interleaved_reasoning_detailed,
        )
    except Exception:
        def check_interleaved_reasoning_detailed(reasoning, *, n_houses=0, require_terminal_period=True):
            errors = []
            if not isinstance(reasoning, list) or not reasoning:
                errors.append({"code": "INVALID_REASONING"})
            elif len(reasoning) % 2 != 0:
                errors.append({"code": "UNPAIRED_ENTRY"})
            return {
                "ok": not errors,
                "errors": errors,
                "n_entries": len(reasoning) if isinstance(reasoning, list) else 0,
                "n_pairs": len(reasoning) // 2 if isinstance(reasoning, list) else 0,
            }

        def check_interleaved_reasoning(reasoning, *, n_houses=0):
            return bool(check_interleaved_reasoning_detailed(
                reasoning,
                n_houses=n_houses,
                require_terminal_period=True,
            )["ok"])

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

RESULT_KEYS = [
    "acc", "score", "reward_logged", "ACCURACY", "parsing_reward", "schema_reward", "format_reward",
    "z3_reward", "consistency_score", "Normalizer", "BASE_sat_full_GT", "missed_data",
    "BASE_n_steps_total", "BASE_n_steps_parsed_ok", "BASE_n_steps_valid", "BASE_n_steps_novel_inc_clues",
    "BASE_n_non_valid_contradiction", "novel_step_score", "contradiction_ratio",
    "selected_option_present", "ground_truth_present", "parse_status_ok", "schema_status_ok",
    "z3_status_ok", "format_status_ok", "z3_base_sat", "z3_formalization_complete",
    "z3_solver_answer_present", "z3_solver_has_unique_answer", "z3_solver_matches_gt",
    "z3_model_matches_solver", "z3_model_matches_gt", "z3_answer_correct",
    # Compatibility aliases retained for existing dashboards.
    "z3_solver_selected_ok", "z3_gt_match", "z3_rule_parse_error_count",
    "z3_fact_parse_error_count", "z3_option_parse_error_count", "z3_selected_option_parse_ok",
    "reward_exception", "parse_error_flag", "epoch", "total_epochs",
]


def _safe_epoch(name: str, default: int) -> float:
    try:
        return float(int(os.getenv(name, str(default))))
    except Exception:
        return float(default)


def _default_result(reward: float = -0.5, missed_data: float = 1.0) -> Dict[str, float]:
    output = {key: 0.0 for key in RESULT_KEYS}
    output["acc"] = output["score"] = output["reward_logged"] = float(reward)
    output["missed_data"] = float(missed_data)
    output["Normalizer"] = 1.0
    output["epoch"] = _safe_epoch("CURRENT_EPOCH", 0)
    output["total_epochs"] = _safe_epoch("TOTAL_EPOCH", 1)
    return output


def _numeric_only(data: Dict[str, Any]) -> Dict[str, float]:
    output = _default_result()
    for key in RESULT_KEYS:
        if key not in data:
            continue
        value = data[key]
        try:
            if isinstance(value, bool):
                output[key] = 1.0 if value else 0.0
            else:
                output[key] = float(value)
        except Exception:
            output[key] = 0.0
    return output


def _clamp_reward(value: Any, low: float = -1.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except Exception:
        return low
    return max(low, min(high, number))


def find_last_answer_block(text: Any) -> Optional[str]:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    elif text is None:
        text = ""
    else:
        text = str(text)
    matches = list(re.finditer(
        r"<answer\b[^>]*>(.*?)</answer\s*>", text,
        flags=re.IGNORECASE | re.DOTALL,
    ))
    return matches[-1].group(1).strip() if matches else None


def _try_parse_first_json_obj(text: Any) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    raw = text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else str(text)
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            parsed, _ = decoder.raw_decode(raw[match.start():])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def parse_ar_lsat_answer(solution_str: Any):
    block = find_last_answer_block(solution_str)
    if block is not None:
        parsed = _try_parse_first_json_obj(block)
        return (parsed, "success_answer_tag") if parsed is not None else (None, "answer_tag_json_error")
    parsed = _try_parse_first_json_obj(solution_str)
    return (parsed, "success_direct_json") if parsed is not None else (None, "parsing_failed")


def _normalize_option_label(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, int) and 0 <= x <= 4:
        return chr(ord("A") + x)
    text = str(x).strip().upper()
    compact = text.replace("-", "_").replace(" ", "_").replace(":", "_")
    match = re.fullmatch(r"(?:SELECTED_)?(?:OPTION|ANSWER|CHOICE)?_?([A-E])", compact)
    if match:
        return match.group(1)
    match = re.search(r"(?:OPTION|ANSWER|CHOICE|SELECTED_OPTION)\s*[:_\-\s]*([A-E])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-E])\b\s*[\.)\]]?\s*$", text)
    if match:
        return match.group(1)
    letters = re.findall(r"\b([A-E])\b", text)
    return letters[-1] if letters else None


def _selected_from_ground_truth(ground_truth: Any) -> Optional[str]:
    if isinstance(ground_truth, (str, int)):
        return _normalize_option_label(ground_truth)
    if isinstance(ground_truth, dict):
        for key in ("answer", "selected_option", "ground_truth_option", "label"):
            if ground_truth.get(key) is not None:
                return _selected_from_ground_truth(ground_truth[key])
    return None


def _selected_from_prediction(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    solution = payload.get("solution") or {}
    if isinstance(solution, dict):
        return _normalize_option_label(solution.get("selected_option"))
    return None


def _infer_n_positions(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    world_model = payload.get("world_model") or {}
    domains = world_model.get("domains") or {} if isinstance(world_model, dict) else {}
    positions = domains.get("positions") or domains.get("position") or []
    if isinstance(positions, list) and positions:
        normalized = set()
        for position in positions:
            try:
                normalized.add(int(str(position).strip()))
            except Exception:
                return None
        return len(normalized)
    entities = world_model.get("entities") or [] if isinstance(world_model, dict) else []
    return len(entities) if isinstance(entities, list) and entities else None


def _infer_n_entities(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    world_model = payload.get("world_model") or {}
    entities = world_model.get("entities") or [] if isinstance(world_model, dict) else []
    return len(entities) if isinstance(entities, list) and entities else None


def _schema_ok(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False
    required = {
        "problem_type", "world_model", "rules", "facts",
        "question_semantics", "options", "reasoning", "solution",
    }
    if not required.issubset(payload):
        return False
    if str(payload.get("problem_type") or "").strip().lower() != "ordering":
        return False
    world_model = payload.get("world_model")
    if not isinstance(world_model, dict):
        return False
    entities = world_model.get("entities")
    if not isinstance(entities, list) or not entities:
        return False
    if not isinstance(payload.get("rules"), list):
        return False
    if not isinstance(payload.get("facts"), list):
        return False
    if not isinstance(payload.get("question_semantics"), dict):
        return False
    if not payload["question_semantics"].get("question_type"):
        return False
    if not isinstance(payload.get("options"), dict) or not payload["options"]:
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
) -> Dict[str, float]:
    del score_method, acc_weight, clue_weight, z3_weight, meta

    try:
        output: Dict[str, Any] = _default_result(reward=0.0, missed_data=0.0)
        payload, parse_status = parse_ar_lsat_answer(solution_str)

        parsing_reward = (
            1.0 if parse_status == "success_answer_tag"
            else 0.5 if parse_status == "success_direct_json"
            else 0.0
        )
        output["parsing_reward"] = parsing_reward
        output["parse_status_ok"] = 1.0 if parse_status == "success_answer_tag" else 0.0
        output["parse_error_flag"] = 0.0 if parse_status in {"success_answer_tag", "success_direct_json"} else 1.0

        selected = _selected_from_prediction(payload)
        ground_truth_option = _selected_from_ground_truth(ground_truth)
        output["selected_option_present"] = 1.0 if selected else 0.0
        output["ground_truth_present"] = 1.0 if ground_truth_option else 0.0
        accuracy = float(bool(selected and ground_truth_option and selected == ground_truth_option))
        output["ACCURACY"] = accuracy

        schema_reward = 1.0 if _schema_ok(payload) else 0.0
        output["schema_reward"] = schema_reward
        output["schema_status_ok"] = schema_reward

        reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
        n_positions = _infer_n_positions(payload)
        n_entities = _infer_n_entities(payload)

        try:
            format_details = check_interleaved_reasoning_detailed(
                reasoning,
                n_houses=int(n_positions or 0),
                require_terminal_period=True,
            )
            format_ok = bool(format_details.get("ok", False))
            if not format_ok:
                logger.debug(
                    "Ordering interleaved-format validation failed: %s",
                    format_details.get("errors", []),
                )
        except Exception:
            logger.exception("Ordering interleaved-format checker crashed")
            format_details = {"ok": False, "errors": [{"code": "CHECKER_EXCEPTION"}]}
            format_ok = False
        output["format_reward"] = output["format_status_ok"] = 1.0 if format_ok else 0.0

        z3_out: Dict[str, Any] = {}
        if isinstance(payload, dict) and schema_reward > 0.0:
            z3_payload = dict(payload)
            z3_payload["ground_truth"] = ground_truth
            if isinstance(extra_info, dict) and extra_info.get("question_type"):
                z3_payload["question_type"] = extra_info["question_type"]
            try:
                z3_out = solve_and_validate_payload(
                    z3_payload,
                    timeout_s=timeout,
                    conflict_tolerant_clues=False,
                )
            except Exception as exc:
                logger.exception("Ordering Z3 validator crashed")
                z3_out = {
                    "parse_status": "Z3_EXCEPTION",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        z3_status = str(z3_out.get("parse_status", ""))
        output["z3_status_ok"] = 1.0 if z3_status == "AR_LSAT_ORDERING_SUCCESS" else 0.0
        output["z3_base_sat"] = float(bool(z3_out.get("base_sat", False)))
        output["z3_formalization_complete"] = float(bool(z3_out.get("formalization_complete", False)))
        output["z3_solver_answer_present"] = float(bool(z3_out.get("solver_answer")))
        output["z3_solver_has_unique_answer"] = float(bool(z3_out.get("solver_has_unique_answer", False)))
        output["z3_solver_matches_gt"] = float(bool(z3_out.get("solver_matches_gt", False)))
        output["z3_model_matches_solver"] = float(bool(z3_out.get("model_matches_solver", False)))
        output["z3_model_matches_gt"] = float(bool(z3_out.get("model_matches_gt", False)))
        output["z3_answer_correct"] = float(bool(z3_out.get("answer_correct", False)))

        # Compatibility aliases.
        output["z3_solver_selected_ok"] = output["z3_model_matches_solver"]
        output["z3_gt_match"] = output["z3_solver_matches_gt"]

        output["z3_rule_parse_error_count"] = float(z3_out.get("n_rule_parse_errors", 0) or 0)
        output["z3_fact_parse_error_count"] = float(z3_out.get("n_fact_parse_errors", 0) or 0)
        output["z3_option_parse_error_count"] = float(z3_out.get("n_option_parse_errors", 0) or 0)
        output["z3_selected_option_parse_ok"] = float(bool(z3_out.get("selected_option_parse_ok", False)))

        sat_ok = float(bool(z3_out.get("base_sat_full_GT", False)))
        output["z3_reward"] = output["BASE_sat_full_GT"] = sat_ok
        output["consistency_score"] = float(z3_out.get("consistency_score", 0.0) or 0.0)
        output["BASE_n_steps_total"] = float(z3_out.get("n_steps_total", 0) or 0)
        output["BASE_n_steps_parsed_ok"] = float(z3_out.get("n_steps_parsed_ok", 0) or 0)
        output["BASE_n_steps_valid"] = float(z3_out.get("n_steps_valid", 0) or 0)
        output["BASE_n_steps_novel_inc_clues"] = float(z3_out.get("n_steps_novel_inc_clues", 0) or 0)
        output["BASE_n_non_valid_contradiction"] = float(z3_out.get("n_non_valid_contradiction", 0) or 0)

        has_required_inputs = bool(
            isinstance(payload, dict)
            and n_positions is not None
            and n_entities is not None
            and n_positions > 0
            and n_entities > 0
        )

        normalizer = max(float(n_entities or 1), 1.0)
        if has_required_inputs:
            novel_count = output["BASE_n_steps_novel_inc_clues"]
            contradiction_count = output["BASE_n_non_valid_contradiction"]
            novel_step_score = min(novel_count / normalizer, 1.0)
            contradiction_ratio = min(contradiction_count / normalizer, 1.0)

            if sat_ok == 0.0:
                reward = (
                    0.15 * parsing_reward
                    + 0.10 * output["format_reward"]
                    + 0.60 * accuracy
                    - 0.20 * contradiction_ratio
                )
            else:
                base_quality = (
                    0.60 * accuracy
                    + 0.20 * parsing_reward
                    + 0.20 * output["format_reward"]
                )
                process_bonus = (
                    0.40 * novel_step_score
                    + 0.30 * output["consistency_score"]
                    - 0.15 * contradiction_ratio
                )
                reward = base_quality + accuracy * process_bonus

            output["novel_step_score"] = novel_step_score
            output["contradiction_ratio"] = contradiction_ratio
            output["missed_data"] = 0.0
        else:
            reward = -0.5
            output["missed_data"] = 1.0

        output["Normalizer"] = normalizer
        final_reward = _clamp_reward(reward)
        output["acc"] = output["score"] = output["reward_logged"] = final_reward
        return _numeric_only(output)

    except Exception:
        logger.exception("Ordering compute_score failed; returning penalty")
        output = _default_result(reward=-0.5, missed_data=1.0)
        output["reward_exception"] = 1.0
        return output


def _wrap(payload: Dict[str, Any]) -> str:
    return "<answer>" + json.dumps(payload, ensure_ascii=False, indent=2) + "</answer>"


def _make_answer(selected: str = "A") -> str:
    # Unique ordering: A=1, C=2, D=3, B=4.
    payload = {
        "problem_type": "ordering",
        "world_model": {
            "entities": ["A", "B", "C", "D"],
            "domains": {"positions": [1, 2, 3, 4]},
            "structural_assumptions": ["each entity occupies exactly one distinct position"],
        },
        "rules": [
            "Before(A, B)",
            "ImmediatelyBefore(A, C)",
            "Not(AtPosition(D, 1))",
        ],
        "facts": ["AtPosition(B, 4)"],
        "question_semantics": {"question_type": "could_be_true"},
        "options": {
            "A": "AtPosition(A, 1)",
            "B": "AtPosition(C, 4)",
            "C": "AtPosition(D, 2)",
        },
        "reasoning": [
            "B is fixed in fourth position by the question condition.",
            "S1: AtPosition(B, 4).",
            "A immediately precedes C.",
            "S2: ImmediatelyBefore(A, C).",
            "The complete ordering therefore places A first.",
            "S3: AtPosition(A, 1).",
            "Option A is satisfiable under the ordering theory.",
            "S4: Sat(Option_A).",
        ],
        "solution": {"selected_option": selected},
    }
    return _wrap(payload)


if __name__ == "__main__":
    print("\n=== INTERLEAVED FORMAT DEMOS ===")
    format_demos = {
        "valid_predicates": [
            "B is fixed in fourth position.",
            "S1: AtPosition(B, 4).",
            "A immediately precedes C.",
            "S2: ImmediatelyBefore(A, C).",
            "Option A is satisfiable.",
            "S3: Sat(Option_A).",
        ],
        "valid_arithmetic": [
            "C occurs immediately after A.",
            "S1: Position(C) == Position(A) + 1.",
            "A and B are two positions apart.",
            "S2: Distance(A, B, 2).",
        ],
        "invalid_starts_formal": [
            "S1: AtPosition(B, 4).",
            "B is fixed in fourth position.",
        ],
        "invalid_unpaired": [
            "B is fixed in fourth position.",
            "S1: AtPosition(B, 4).",
            "This explanation has no formal partner.",
        ],
        "invalid_step_number": [
            "B is fixed in fourth position.",
            "S2: AtPosition(B, 4).",
        ],
        "invalid_out_of_range": [
            "B is placed outside the domain.",
            "S1: AtPosition(B, 5).",
        ],
    }

    for name, reasoning in format_demos.items():
        details = check_interleaved_reasoning_detailed(
            reasoning,
            n_houses=4,
            require_terminal_period=True,
        )
        print(
            f"{name:28s} ok={details['ok']!s:5s} "
            f"errors={[item['code'] for item in details['errors']]}"
        )

    print("\n=== COMPLETE REWARD DEMOS ===")
    tests = [
        ("correct_could_be_true", _make_answer("A"), "A"),
        ("wrong_selected_option", _make_answer("B"), "A"),
    ]
    for name, prediction, ground_truth in tests:
        print(f"\n=== {name} ===")
        result = compute_score(prediction, ground_truth)
        print(json.dumps(result, indent=2))
