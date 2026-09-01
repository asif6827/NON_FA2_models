# -*- coding: utf-8 -*-
"""
our_puzzle_dataset_v7_nlspa.py

Reward scorer for the new Zebra NL_i / S_i / PA_i output format.

Expected reasoning:
{
    "NL1": "...",
    "S1": "Arnold == 2.",
    "PA1": {
        "header": [...],
        "rows": [...]
    },
    ...
}

Key changes from v6
-------------------
1. reasoning is a JSON object rather than list[str].
2. S_i steps are adapted to the existing Z3 validator.
3. PA_i checkpoints receive their own PRM:
   - structure/domain validity
   - correctness of resolved cells
   - support by clue + S-prefix explicit equality state
   - monotonicity
   - progressive filling
4. S-step PRM is normalized by the number of emitted S steps, not
   2 * houses * attributes.
5. All S_i deductions, not only novel deductions, are checked against the
   final predicted solution.
6. Fixes two v6 reward issues:
   - normal <answer> JSON receives parsing_reward=1.
   - zero novel steps no longer automatically forces reward=-0.5.

The stable v6 parsing/accuracy helpers and v9 Z3 core remain reused.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

try:
    from verl.utils.reward_score import our_puzzle_dataset_v6 as _legacy_reward
except Exception:
    import our_puzzle_dataset_v6 as _legacy_reward

try:
    from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v10_nlspa import (
        solve_and_validate_payload,
        normalize_header,
    )
except Exception:
    from z3_reasoning_validator_v13_gt_solve_v10_nlspa import (
        solve_and_validate_payload,
        normalize_header,
    )

try:
    from verl.utils.reward_score.z3_reasoning_vs_solution_verifier_v3_nlspa import (
        verify_solution_two_step,
    )
except Exception:
    from z3_reasoning_vs_solution_verifier_v3_nlspa import verify_solution_two_step

try:
    from verl.utils.reward_score.nlspa_reward_utils import (
        validate_reasoning_schema,
        score_partial_answers,
        compute_s_prm_metrics,
        combine_process_prm,
    )
except Exception:
    from nlspa_reward_utils import (
        validate_reasoning_schema,
        score_partial_answers,
        compute_s_prm_metrics,
        combine_process_prm,
    )


logger = logging.getLogger(__name__)


def _clamp_reward(x: float) -> float:
    """
    Preserve the approximate v6 scale.

    v6 base quality is <=1 and its process bonus can add roughly <=0.7.
    """
    try:
        x = float(x)
    except Exception:
        return -0.5
    return max(-0.5, min(1.7, x))


def _parse_payload(solution_str: str) -> tuple[Optional[Dict[str, Any]], str]:
    """
    Parse the model output while distinguishing the required <answer> form.
    """
    answer = _legacy_reward.find_last_answer_block(solution_str)
    if answer:
        payload = _legacy_reward._try_parse_first_json_obj(answer)
        if isinstance(payload, dict):
            return payload, "success_answer_tag"
        return None, "answer_tag_json_error"

    payload = _legacy_reward._try_parse_first_json_obj(solution_str)
    if isinstance(payload, dict):
        return payload, "success_direct_json"

    return None, "parsing_failed"


def _top_level_schema_ok(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return list(payload.keys()) == [
        "n_houses",
        "attribute_values",
        "syntactic_clues",
        "reasoning",
        "solution",
    ]


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


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
):
    """
    Return VERL-safe numeric metrics.

    The public signature is intentionally kept compatible with v6.
    """
    epoch = int(os.getenv("CURRENT_EPOCH", "90"))
    total_epochs = int(os.getenv("TOTAL_EPOCH", "100"))

    out: Dict[str, float] = {
        "acc": 0.0,
        "score": 0.0,
        "reward_logged": 0.0,
        "PUZZLE_ACCURACY": 0.0,
        "CELL_ACCURACY": 0.0,
        "parsing_reward": 0.0,
        "format_reward": 0.0,
        "BASE_sat_full_GT": 0.0,
        "BASE_n_steps_total": 0.0,
        "BASE_n_steps_parsed_ok": 0.0,
        "BASE_n_steps_valid": 0.0,
        "BASE_n_steps_novel_inc_clues": 0.0,
        "BASE_n_non_valid_contradiction": 0.0,
        "Normalizer": 1.0,
        "novel_step_score": 0.0,
        "contradiction_ratio": 0.0,
        "consistency_score": 0.0,
        "NLSPA_format_ok": 0.0,
        "NLSPA_n_nl": 0.0,
        "NLSPA_n_s": 0.0,
        "NLSPA_n_pa": 0.0,
        "s_parse_ratio": 0.0,
        "s_validity_ratio": 0.0,
        "s_novelty_ratio": 0.0,
        "s_contradiction_ratio": 0.0,
        "s_prm_score": 0.0,
        "pa_present": 0.0,
        "PA_n_total": 0.0,
        "PA_n_structurally_valid": 0.0,
        "PA_resolved_cells": 0.0,
        "PA_correct_resolved_cells": 0.0,
        "PA_incorrect_resolved_cells": 0.0,
        "PA_supported_resolved_cells": 0.0,
        "PA_unsupported_resolved_cells": 0.0,
        "pa_structure_score": 0.0,
        "pa_cell_precision": 0.0,
        "pa_final_coverage": 0.0,
        "pa_effective_progress": 0.0,
        "pa_monotonicity_score": 0.0,
        "pa_transition_progress_score": 0.0,
        "pa_prefix_support_score": 0.0,
        "pa_prm_score": 0.0,
        "process_prm_score": 0.0,
        "missed_data": 0.0,
        "reward_exception": 0.0,
        "epoch": float(epoch),
        "total_epochs": float(total_epochs),
    }

    try:
        gt = _legacy_reward.normalize_ground_truth(ground_truth)
        gt = normalize_header(gt)

        payload, parse_status = _parse_payload(str(solution_str))
        parsing_reward = 1.0 if parse_status in {
            "success_answer_tag",
            "success_direct_json",
        } else 0.0
        out["parsing_reward"] = parsing_reward

        if not isinstance(payload, dict):
            out["missed_data"] = 1.0
            out["acc"] = out["score"] = out["reward_logged"] = -0.5
            return out

        n_houses = payload.get("n_houses")
        attribute_values = payload.get("attribute_values")
        syntactic_clues = payload.get("syntactic_clues")
        reasoning = payload.get("reasoning")
        predicted = payload.get("solution")

        # ---------------- ACC ----------------
        cell_acc = puzzle_acc = 0.0
        if isinstance(predicted, dict):
            pred_conv = _legacy_reward.convert_numpy_arrays(predicted)
            gt_conv = _legacy_reward.convert_numpy_arrays(gt)
            norm_pred = _legacy_reward.normalize_table(pred_conv)
            norm_gt = _legacy_reward.normalize_table(gt_conv)
            if norm_pred and norm_gt:
                norm_pred = normalize_header(norm_pred)
                cell_acc, puzzle_acc = _legacy_reward._compute_acc_from_normalized(
                    norm_pred,
                    norm_gt,
                )

        out["CELL_ACCURACY"] = float(cell_acc)
        out["PUZZLE_ACCURACY"] = float(puzzle_acc)

        # ---------------- NL/S/PA FORMAT ----------------
        schema = validate_reasoning_schema(
            reasoning,
            n_houses=int(n_houses or 0),
            expected_header=gt.get("header", []),
            attribute_values=attribute_values or {},
        )

        top_schema_ok = _top_level_schema_ok(payload)
        required_answer_wrapper = parse_status == "success_answer_tag"
        format_ok = bool(
            schema["ok"]
            and top_schema_ok
            and required_answer_wrapper
        )

        out["NLSPA_format_ok"] = 1.0 if schema["ok"] else 0.0
        out["NLSPA_n_nl"] = float(schema["n_nl"])
        out["NLSPA_n_s"] = float(schema["n_s"])
        out["NLSPA_n_pa"] = float(schema["n_pa"])
        out["format_reward"] = 1.0 if format_ok else 0.0

        # ---------------- Z3 S-STEP VALIDATION ----------------
        z3_out: Dict[str, Any] = {}
        required_inputs = (
            isinstance(n_houses, int)
            and n_houses > 0
            and isinstance(attribute_values, dict)
            and bool(attribute_values)
            and isinstance(syntactic_clues, list)
            and isinstance(reasoning, dict)
            and schema["n_s"] > 0
        )

        if required_inputs:
            z3_payload = {
                "n_houses": n_houses,
                "attribute_values": attribute_values,
                "syntactic_clues": syntactic_clues,
                # v10 adapter accepts the raw NL/S/PA object.
                "reasoning": reasoning,
                "ground_truth": gt,
            }
            try:
                z3_out = solve_and_validate_payload(
                    z3_payload,
                    timeout_s=max(float(timeout), 0.1),
                    conflict_tolerant_clues=False,
                )
            except Exception as e:
                logger.exception("NLSPA Z3 validation failed: %s", e)
                z3_out = {}
        else:
            out["missed_data"] = 1.0

        sat_ok = 1.0 if bool(z3_out.get("base_sat_full_GT", False)) else 0.0
        out["BASE_sat_full_GT"] = sat_ok
        out["BASE_n_steps_total"] = _safe_float(
            z3_out.get("n_steps_total", 0)
        )
        out["BASE_n_steps_parsed_ok"] = _safe_float(
            z3_out.get("n_steps_parsed_ok", 0)
        )
        out["BASE_n_steps_valid"] = _safe_float(
            z3_out.get("n_steps_valid", 0)
        )
        out["BASE_n_steps_novel_inc_clues"] = _safe_float(
            z3_out.get("n_steps_novel_inc_clues", 0)
        )
        out["BASE_n_non_valid_contradiction"] = _safe_float(
            z3_out.get("n_non_valid_contradiction", 0)
        )

        # ---------------- ALL-S vs FINAL SOLUTION ----------------
        reasoning_solution_consistency = 0.0
        if (
            isinstance(syntactic_clues, list)
            and isinstance(predicted, dict)
            and schema["n_s"] > 0
        ):
            try:
                verification = verify_solution_two_step(
                    syntactic_clues,
                    reasoning,  # v3 adapter extracts ALL S_i.
                    predicted,
                )
                reasoning_solution_consistency = _safe_float(
                    verification.get("reward", 0.0)
                )
            except Exception:
                reasoning_solution_consistency = 0.0

        out["consistency_score"] = reasoning_solution_consistency

        # ---------------- S PRM ----------------
        s_metrics = compute_s_prm_metrics(
            z3_out,
            n_s_steps=int(schema["n_s"]),
            reasoning_solution_consistency=reasoning_solution_consistency,
        )
        out.update(s_metrics)

        # Backward-compatible dashboard names now use S-count normalization.
        out["Normalizer"] = float(max(int(schema["n_s"]), 1))
        out["novel_step_score"] = s_metrics["s_novelty_ratio"]
        out["contradiction_ratio"] = s_metrics["s_contradiction_ratio"]

        # ---------------- PA PRM ----------------
        pa_metrics = score_partial_answers(
            reasoning,
            n_houses=int(n_houses or 0),
            expected_header=gt.get("header", []),
            attribute_values=attribute_values or {},
            ground_truth=gt,
            syntactic_clues=syntactic_clues or [],
        )

        # Keep only numeric PA metrics in the VERL return dictionary.
        for key, value in pa_metrics.items():
            if key == "pa_details":
                continue
            if isinstance(value, (int, float, bool)):
                out[key] = float(value)

        process_prm = combine_process_prm(
            s_prm_score=s_metrics["s_prm_score"],
            pa_prm_score=pa_metrics["pa_prm_score"],
            pa_present=pa_metrics["pa_present"],
        )
        out["process_prm_score"] = process_prm

        # ---------------- FINAL REWARD ----------------
        #
        # Preserve v6's approximate scale:
        #   base quality <= 1.0
        #   process bonus <= 0.7
        #
        # PA is optional. If absent, combined process PRM == S PRM.
        if not required_inputs:
            reward = -0.5
        elif sat_ok == 0.0:
            reward = (
                0.15 * parsing_reward
                + 0.10 * out["format_reward"]
                + 0.60 * float(puzzle_acc)
                - 0.20 * s_metrics["s_contradiction_ratio"]
            )
        else:
            base_quality = (
                0.60 * float(puzzle_acc)
                + 0.20 * parsing_reward
                + 0.20 * out["format_reward"]
            )

            process_bonus = 0.70 * process_prm

            # Preserve the existing final-answer gate: process reward adds to
            # the scalar objective only when the final puzzle is correct.
            reward = (
                base_quality
                + float(puzzle_acc) * process_bonus
            )

        reward = _clamp_reward(reward)
        out["acc"] = reward
        out["score"] = reward
        out["reward_logged"] = reward
        return out

    except Exception:
        logger.exception("Crash in NLSPA reward computation")
        out["reward_exception"] = 1.0
        out["acc"] = out["score"] = out["reward_logged"] = -0.5
        return out


if __name__ == "__main__":
    # Lightweight schema example only. Full execution requires the existing
    # Zebra reward environment / pid dictionary used by v6.
    example = {
        "n_houses": 3,
        "attribute_values": {
            "Name": ["Peter", "Eric", "Arnold"],
            "Color": ["red", "white", "yellow"],
            "Children": ["Fred", "Meredith", "Bella"],
        },
        "syntactic_clues": [
            "C1: Arnold == red.",
            "C2: Fred < Eric.",
            "C3: red == 2.",
            "C4: Bella == 1.",
            "C5: white == Meredith.",
        ],
        "reasoning": {
            "NL1": "Clues 1 and 3 together place Arnold in house 2.",
            "S1": "Arnold == 2.",
            "NL2": "Fred is left of Eric, so Eric cannot be in house 1.",
            "S2": "Eric != 1.",
            "PA1": {
                "header": ["House", "Name", "Color", "Children"],
                "rows": [
                    ["1", "?", "?", "Bella"],
                    ["2", "Arnold", "red", "?"],
                    ["3", "?", "?", "?"],
                ],
            },
        },
        "solution": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "Peter", "yellow", "Bella"],
                ["2", "Arnold", "red", "Fred"],
                ["3", "Eric", "white", "Meredith"],
            ],
        },
    }
    print(json.dumps(example, indent=2))
