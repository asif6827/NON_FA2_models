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
4. Interleaving PRM gives dense partial credit for NL_i -> S_i ordering,
   index continuity, and PA placement.
5. S-step PRM is normalized by the number of emitted S steps, not
   2 * houses * attributes.
6. All S_i deductions, not only novel deductions, are checked against the
   final predicted solution.
7. Fixes two v6 reward issues:
   - normal <answer> JSON receives parsing_reward=1.
   - zero novel steps no longer automatically forces reward=-0.5.

Reward formulas added in this version
-------------------------------------
R_interleave =
    0.40 * pair_score
  + 0.25 * order_score
  + 0.20 * index_score
  + 0.15 * PA_placement_score

With PA:
    R_process = 0.20 * R_interleave + 0.55 * R_S + 0.25 * R_PA

Without PA:
    R_process = 0.25 * R_interleave + 0.75 * R_S

The stable v6 parsing/accuracy helpers and v9 Z3 core remain reused.
"""

from __future__ import annotations

import json
import copy
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
        score_partial_answers,
        compute_s_prm_metrics,
    )
except Exception:
    from nlspa_reward_utils import (
        score_partial_answers,
        compute_s_prm_metrics,
    )

try:
    from verl.utils.reward_score.check_interleved_format_v7_nlspa import (
        check_interleaved_reasoning_detailed,
    )
except Exception:
    from check_interleved_format_v7_nlspa import (
        check_interleaved_reasoning_detailed,
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



def _combine_process_prm_with_interleave(
    *,
    interleave_reward: float,
    s_prm_score: float,
    pa_prm_score: float,
    pa_present: float,
) -> float:
    """
    Combined process PRM.

    With PA:
        R_process = 0.20 R_interleave + 0.55 R_S + 0.25 R_PA

    Without PA (PA is optional):
        R_process = 0.25 R_interleave + 0.75 R_S
    """
    ri = max(0.0, min(1.0, _safe_float(interleave_reward)))
    rs = max(0.0, min(1.0, _safe_float(s_prm_score)))
    rpa = max(0.0, min(1.0, _safe_float(pa_prm_score)))

    if _safe_float(pa_present) > 0.0:
        score = 0.20 * ri + 0.55 * rs + 0.25 * rpa
    else:
        score = 0.25 * ri + 0.75 * rs
    return max(0.0, min(1.0, float(score)))


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
        "Z3_s_analysis_ok": 0.0,
        "Z3_base_validation_exception": 0.0,
        "Z3_s_analysis_exception": 0.0,
        "Normalizer": 1.0,
        "novel_step_score": 0.0,
        "contradiction_ratio": 0.0,
        "consistency_score": 0.0,
        "NLSPA_format_ok": 0.0,
        "NLSPA_n_nl": 0.0,
        "NLSPA_n_s": 0.0,
        "NLSPA_n_pa": 0.0,
        "interleave_pair_score": 0.0,
        "interleave_order_score": 0.0,
        "interleave_index_score": 0.0,
        "interleave_pa_placement_score": 0.0,
        "interleave_reward": 0.0,
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
                try:
                    acc_result = _legacy_reward._compute_acc_from_normalized(
                        norm_pred,
                        norm_gt,
                    )
                    if (
                        isinstance(acc_result, (tuple, list))
                        and len(acc_result) == 2
                    ):
                        cell_acc, puzzle_acc = acc_result
                    else:
                        # Legacy helper has a malformed-row branch that can
                        # return a scalar 0.0 instead of (0.0, 0.0). Treat
                        # malformed grids as zero accuracy rather than
                        # crashing the entire reward computation.
                        cell_acc = 0.0
                        puzzle_acc = 0.0
                except Exception as acc_error:
                    logger.warning(
                        "ACC computation failed; assigning zero accuracy: %s",
                        acc_error,
                    )
                    cell_acc = 0.0
                    puzzle_acc = 0.0

        out["CELL_ACCURACY"] = float(cell_acc)
        out["PUZZLE_ACCURACY"] = float(puzzle_acc)

        # ---------------- NL/S/PA FORMAT ----------------
        schema = check_interleaved_reasoning_detailed(
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

        # Dense process-format reward. Unlike format_reward, this is NOT
        # all-or-nothing: mostly-correct interleaving receives partial credit.
        for key in (
            "interleave_pair_score",
            "interleave_order_score",
            "interleave_index_score",
            "interleave_pa_placement_score",
            "interleave_reward",
        ):
            out[key] = _safe_float(schema.get(key, 0.0))

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

        # Explicit diagnostics so a disconnected S/Z3 pipeline cannot silently
        # look like a valid zero-score trace.
        out["Z3_s_analysis_ok"] = (
            1.0 if z3_out.get("s_analysis_status") == "SUCCESS" else 0.0
        )
        out["Z3_base_validation_exception"] = (
            1.0 if bool(z3_out.get("base_validation_exception")) else 0.0
        )
        out["Z3_s_analysis_exception"] = (
            1.0 if bool(z3_out.get("s_analysis_exception")) else 0.0
        )
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

        process_prm = _combine_process_prm_with_interleave(
            interleave_reward=out["interleave_reward"],
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
        # PA is optional. R_interleave always contributes to process quality.
        if not required_inputs:
            reward = -0.5
        elif sat_ok == 0.0:
            reward = (
                0.15 * parsing_reward
                + 0.10 * out["format_reward"]
                + 0.60 * float(puzzle_acc)
                + 0.15 * out["interleave_reward"]
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
    # ============================================================
    # NL/S/PA REWARD REGRESSION SUITE
    # ============================================================
    #
    # Run in the normal VERL reward environment:
    #
    #   python our_puzzle_dataset_v8_nlspa_corner_cases.py
    #
    # Each case changes only one aspect of the prediction and checks the
    # reward/PRM signal that should respond to that change.

    ATTRIBUTE_VALUES = {
        "Name": ["Peter", "Eric", "Arnold"],
        "Color": ["red", "white", "yellow"],
        "Children": ["Fred", "Meredith", "Bella"],
    }

    SYNTACTIC_CLUES = [
        "C1: Arnold == red.",
        "C2: Fred < Eric.",
        "C3: red == 2.",
        "C4: Bella == 1.",
        "C5: white == Meredith.",
    ]

    GROUND_TRUTH = {
        "header": ["House", "Name", "Color", "Children"],
        "rows": [
            ["1", "Peter", "yellow", "Bella"],
            ["2", "Arnold", "red", "Fred"],
            ["3", "Eric", "white", "Meredith"],
        ],
    }

    BASE_REASONING = {
        "NL1": "Clues 1 and 3 together show that Arnold, who has the red favorite color, must occupy house 2.",
        "S1": "Arnold == 2.",

        "NL2": "Because Fred is somewhere to the left of Eric, Eric cannot occupy house 1.",
        "S2": "Eric != 1.",

        "NL3": "Arnold already occupies house 2, so Eric cannot occupy house 2 and therefore must occupy house 3.",
        "S3": "Eric == 3.",

        "PA1": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "?", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "Eric", "?", "?"],
            ],
        },

        "NL4": "With Arnold in house 2 and Eric in house 3, the remaining person Peter must occupy house 1.",
        "S4": "Peter == 1.",

        "NL5": "Since Eric is in house 3 and Fred must be somewhere to his left, Fred can only be in house 1 or house 2.",
        "S5": "Or(Fred == 1, Fred == 2).",

        "NL6": "Bella is fixed in house 1 by Clue 4, so child uniqueness forces Fred into house 2.",
        "S6": "Fred == 2.",

        "NL7": "With Bella in house 1 and Fred in house 2, the remaining child Meredith must occupy house 3.",
        "S7": "Meredith == 3.",

        "PA2": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "Peter", "?", "Bella"],
                ["2", "Arnold", "red", "Fred"],
                ["3", "Eric", "?", "Meredith"],
            ],
        },

        "NL8": "Clue 5 places white in the same house as Meredith, so white must occupy house 3.",
        "S8": "white == 3.",

        "NL9": "Red is already in house 2 and white is in house 3, so color uniqueness forces yellow into house 1.",
        "S9": "yellow == 1.",
    }

    def make_prediction():
        return {
            "n_houses": 3,
            "attribute_values": copy.deepcopy(ATTRIBUTE_VALUES),
            "syntactic_clues": copy.deepcopy(SYNTACTIC_CLUES),
            "reasoning": copy.deepcopy(BASE_REASONING),
            "solution": copy.deepcopy(GROUND_TRUTH),
        }

    def wrap_answer(payload):
        return (
            "<answer>\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
            + "\n</answer>"
        )

    def run_case(name, payload, checks, *, use_answer_wrapper=True):
        text = (
            wrap_answer(payload)
            if use_answer_wrapper
            else json.dumps(payload, indent=2, ensure_ascii=False)
        )

        result = compute_score(
            text,
            copy.deepcopy(GROUND_TRUTH),
        )

        failures = []
        for description, predicate in checks:
            try:
                passed = bool(predicate(result))
            except Exception as exc:
                passed = False
                description += (
                    f" [predicate raised {type(exc).__name__}: {exc}]"
                )
            if not passed:
                failures.append(description)

        print("\n" + "=" * 80)
        print(("PASS" if not failures else "FAIL") + f": {name}")
        print("-" * 80)
        summary_keys = [
            "score",
            "PUZZLE_ACCURACY",
            "CELL_ACCURACY",
            "parsing_reward",
            "format_reward",
            "NLSPA_format_ok",
            "NLSPA_n_s",
            "NLSPA_n_pa",
            "interleave_pair_score",
            "interleave_order_score",
            "interleave_index_score",
            "interleave_pa_placement_score",
            "interleave_reward",
            "BASE_sat_full_GT",
            "Z3_s_analysis_ok",
            "Z3_base_validation_exception",
            "Z3_s_analysis_exception",
            "s_parse_ratio",
            "s_validity_ratio",
            "s_novelty_ratio",
            "s_contradiction_ratio",
            "consistency_score",
            "pa_present",
            "pa_structure_score",
            "pa_cell_precision",
            "pa_prefix_support_score",
            "pa_monotonicity_score",
            "pa_transition_progress_score",
            "pa_prm_score",
            "process_prm_score",
        ]
        print(
            json.dumps(
                {k: result.get(k) for k in summary_keys},
                indent=2,
                ensure_ascii=False,
            )
        )

        if failures:
            print("Failed expectations:")
            for item in failures:
                print("  -", item)
            raise AssertionError(
                f"{name}: {len(failures)} regression expectation(s) failed."
            )

        return result

    cases = []

    # 1) Fully valid trace.
    cases.append((
        "1. valid_full_trace",
        make_prediction(),
        [
            ("parsing reward = 1", lambda r: r["parsing_reward"] == 1.0),
            ("format valid", lambda r: r["NLSPA_format_ok"] == 1.0),
            ("9 S steps", lambda r: r["NLSPA_n_s"] == 9.0),
            ("2 PAs", lambda r: r["NLSPA_n_pa"] == 2.0),
            ("puzzle correct", lambda r: r["PUZZLE_ACCURACY"] == 1.0),
            ("PA precision = 1", lambda r: r["pa_cell_precision"] == 1.0),
            ("PA prefix support = 1", lambda r: r["pa_prefix_support_score"] == 1.0),
            ("PA monotonicity = 1", lambda r: r["pa_monotonicity_score"] == 1.0),
            ("interleave reward = 1", lambda r: r["interleave_reward"] == 1.0),
            ("pair score = 1", lambda r: r["interleave_pair_score"] == 1.0),
            ("order score = 1", lambda r: r["interleave_order_score"] == 1.0),
            ("index score = 1", lambda r: r["interleave_index_score"] == 1.0),
            ("PA placement score = 1", lambda r: r["interleave_pa_placement_score"] == 1.0),
            (
                "process PRM uses 20% interleave + 55% S + 25% PA",
                lambda r: abs(
                    r["process_prm_score"]
                    - (
                        0.20 * r["interleave_reward"]
                        + 0.55 * r["s_prm_score"]
                        + 0.25 * r["pa_prm_score"]
                    )
                ) < 1e-9,
            ),
            ("S analysis actually ran", lambda r: r["Z3_s_analysis_ok"] == 1.0),
            ("all S steps parse", lambda r: r["s_parse_ratio"] == 1.0),
            ("all valid-example S steps are valid", lambda r: r["s_validity_ratio"] == 1.0),
            ("reasoning agrees with final solution", lambda r: r["consistency_score"] == 1.0),
        ],
        True,
    ))

    # 2) PAs omitted entirely. They are currently optional.
    p = make_prediction()
    p["reasoning"] = {
        k: v for k, v in p["reasoning"].items()
        if not k.startswith("PA")
    }
    cases.append((
        "2. valid_trace_without_PA",
        p,
        [
            ("format still valid", lambda r: r["NLSPA_format_ok"] == 1.0),
            ("no PA", lambda r: r["NLSPA_n_pa"] == 0.0),
            ("pa_present = 0", lambda r: r["pa_present"] == 0.0),
            ("no-PA interleave stays perfect", lambda r: r["interleave_reward"] == 1.0),
            ("no-PA placement is neutral", lambda r: r["interleave_pa_placement_score"] == 1.0),
            (
                "no-PA process PRM uses 25% interleave + 75% S",
                lambda r: abs(
                    r["process_prm_score"]
                    - (
                        0.25 * r["interleave_reward"]
                        + 0.75 * r["s_prm_score"]
                    )
                ) < 1e-9,
            ),
            ("puzzle correct", lambda r: r["PUZZLE_ACCURACY"] == 1.0),
            ("S analysis still runs without PA", lambda r: r["Z3_s_analysis_ok"] == 1.0),
            ("S parse ratio remains 1", lambda r: r["s_parse_ratio"] == 1.0),
        ],
        True,
    ))

    # 3) PA illegally inserted between NL3 and S3.
    p = make_prediction()
    original = p["reasoning"]
    reordered = {}
    for key, value in original.items():
        if key == "S3":
            reordered["PA1"] = copy.deepcopy(original["PA1"])
        if key == "PA1":
            continue
        reordered[key] = copy.deepcopy(value)
    p["reasoning"] = reordered
    cases.append((
        "3. invalid_PA_between_NL_and_S",
        p,
        [
            ("schema rejected", lambda r: r["NLSPA_format_ok"] == 0.0),
            ("format reward = 0", lambda r: r["format_reward"] == 0.0),
            ("interleave reward gives partial credit", lambda r: 0.0 < r["interleave_reward"] < 1.0),
            ("bad PA placement is detected", lambda r: r["interleave_pa_placement_score"] < 1.0),
        ],
        True,
    ))

    # 4) Wrong PA header.
    p = make_prediction()
    p["reasoning"]["PA1"]["header"] = [
        "House", "Name", "WrongColor", "Children"
    ]
    cases.append((
        "4. invalid_PA_header",
        p,
        [
            ("schema rejected", lambda r: r["NLSPA_format_ok"] == 0.0),
            ("PA structure penalized", lambda r: r["pa_structure_score"] < 1.0),
            ("interleaving itself remains perfect", lambda r: r["interleave_reward"] == 1.0),
        ],
        True,
    ))

    # 5) Correct GT cell filled too early.
    # Peter=1 is correct but has not been established at PA1 (which is after S3).
    p = make_prediction()
    p["reasoning"]["PA1"]["rows"][0][1] = "Peter"
    cases.append((
        "5. premature_but_correct_PA_cell",
        p,
        [
            ("schema still valid", lambda r: r["NLSPA_format_ok"] == 1.0),
            ("PA cells remain GT-correct", lambda r: r["pa_cell_precision"] == 1.0),
            ("prefix support drops", lambda r: r["pa_prefix_support_score"] < 1.0),
            ("interleaving remains perfect", lambda r: r["interleave_reward"] == 1.0),
        ],
        True,
    ))

    # 6) Wrong resolved PA cell, but structurally valid and monotonic.
    p = make_prediction()
    p["reasoning"]["PA1"]["rows"][0][2] = "white"
    p["reasoning"]["PA2"]["rows"][0][2] = "white"
    cases.append((
        "6. wrong_PA_cell",
        p,
        [
            ("structure still valid", lambda r: r["pa_structure_score"] == 1.0),
            ("PA precision drops", lambda r: r["pa_cell_precision"] < 1.0),
            ("monotonicity stays valid", lambda r: r["pa_monotonicity_score"] == 1.0),
            ("prefix support drops", lambda r: r["pa_prefix_support_score"] < 1.0),
        ],
        True,
    ))

    # 7) Non-monotonic PA: Arnold resolved in PA1 then reverted to '?' in PA2.
    p = make_prediction()
    p["reasoning"]["PA2"]["rows"][1][1] = "?"
    cases.append((
        "7. non_monotonic_PA",
        p,
        [
            ("individual PA structures valid", lambda r: r["pa_structure_score"] == 1.0),
            ("monotonicity drops", lambda r: r["pa_monotonicity_score"] < 1.0),
            ("interleaving remains perfect", lambda r: r["interleave_reward"] == 1.0),
        ],
        True,
    ))

    # 8) Contradictory S step.
    # This is the regression that exposed the old silent-zero S pipeline.
    p = make_prediction()
    p["reasoning"]["S4"] = "Peter == 2."
    cases.append((
        "8. contradictory_S_step",
        p,
        [
            ("contradiction detected", lambda r: r["s_contradiction_ratio"] > 0.0),
            ("interleaving remains perfect", lambda r: r["interleave_reward"] == 1.0),
            ("final consistency drops", lambda r: r["consistency_score"] < 1.0),
        ],
        True,
    ))

    # 9) Structurally valid S string, but not parseable by the Z3 DSL.
    p = make_prediction()
    p["reasoning"]["S4"] = "Peter occupies house 1."
    cases.append((
        "9. unparseable_S_step",
        p,
        [
            ("NL/S object structure remains valid", lambda r: r["NLSPA_format_ok"] == 1.0),
            ("S parse ratio drops", lambda r: r["s_parse_ratio"] < 1.0),
            ("interleaving remains perfect", lambda r: r["interleave_reward"] == 1.0),
        ],
        True,
    ))

    # 10) Wrong final solution with otherwise correct reasoning and PAs.
    p = make_prediction()
    p["solution"]["rows"][0][1] = "Eric"
    p["solution"]["rows"][2][1] = "Peter"
    cases.append((
        "10. wrong_final_solution",
        p,
        [
            ("puzzle accuracy = 0", lambda r: r["PUZZLE_ACCURACY"] == 0.0),
            ("interleaving remains perfect", lambda r: r["interleave_reward"] == 1.0),
            ("reasoning/final consistency drops", lambda r: r["consistency_score"] < 1.0),
        ],
        True,
    ))

    # 11) Direct JSON without the required <answer> wrapper.
    p = make_prediction()
    cases.append((
        "11. direct_JSON_without_answer_wrapper",
        p,
        [
            ("JSON still parses", lambda r: r["parsing_reward"] == 1.0),
            ("response-contract format reward = 0", lambda r: r["format_reward"] == 0.0),
        ],
        False,
    ))

    # 12) Final NL step without matching S step.
    p = make_prediction()
    p["reasoning"]["NL10"] = (
        "This natural-language deduction has no matching syntactic step."
    )
    cases.append((
        "12. unpaired_final_NL",
        p,
        [
            ("schema rejected", lambda r: r["NLSPA_format_ok"] == 0.0),
            ("format reward = 0", lambda r: r["format_reward"] == 0.0),
            ("unpaired NL lowers pair reward", lambda r: r["interleave_pair_score"] < 1.0),
            ("unpaired NL lowers interleave reward", lambda r: r["interleave_reward"] < 1.0),
        ],
        True,
    ))

    # 13) Consecutive NL steps: dense interleaving reward should drop even
    # though we can still inspect the rest of the trajectory.
    p = make_prediction()
    r = p["reasoning"]
    bad = {}
    for key, value in r.items():
        if key == "S2":
            continue
        bad[key] = copy.deepcopy(value)
    p["reasoning"] = bad
    cases.append((
        "13. consecutive_NL_missing_S2",
        p,
        [
            ("strict format rejected", lambda r: r["NLSPA_format_ok"] == 0.0),
            ("pair score drops", lambda r: r["interleave_pair_score"] < 1.0),
            ("order score drops", lambda r: r["interleave_order_score"] < 1.0),
            ("dense interleave remains > 0", lambda r: r["interleave_reward"] > 0.0),
            ("dense interleave remains < 1", lambda r: r["interleave_reward"] < 1.0),
        ],
        True,
    ))

    # 14) Skip NL/S index 2 while preserving local NL->S pairs. This isolates
    # the index-continuity component.
    p = make_prediction()
    p["reasoning"] = {
        "NL1": "Arnold must be in house 2.",
        "S1": "Arnold == 2.",
        "NL3": "Peter must be in house 1.",
        "S3": "Peter == 1.",
    }
    cases.append((
        "14. skipped_NL_S_indices",
        p,
        [
            ("strict format rejected", lambda r: r["NLSPA_format_ok"] == 0.0),
            ("index score drops", lambda r: r["interleave_index_score"] < 1.0),
            ("interleave reward drops", lambda r: r["interleave_reward"] < 1.0),
        ],
        True,
    ))

    # 15) PA at the very beginning: isolates PA-placement reward.
    p = make_prediction()
    r = p["reasoning"]
    first_pa = copy.deepcopy(r["PA1"])
    p["reasoning"] = {"PA1": first_pa, **{k: copy.deepcopy(v) for k, v in r.items() if k != "PA1"}}
    cases.append((
        "15. PA_before_first_pair",
        p,
        [
            ("strict format rejected", lambda r: r["NLSPA_format_ok"] == 0.0),
            ("PA placement score drops", lambda r: r["interleave_pa_placement_score"] < 1.0),
            ("interleave reward drops", lambda r: r["interleave_reward"] < 1.0),
        ],
        True,
    ))

    passed = 0
    for name, payload, checks, use_wrapper in cases:
        run_case(
            name,
            payload,
            checks,
            use_answer_wrapper=use_wrapper,
        )
        passed += 1

    print("\n" + "=" * 80)
    print(f"ALL {passed} NL/S/PA + R_interleave REGRESSION CASES PASSED.")
