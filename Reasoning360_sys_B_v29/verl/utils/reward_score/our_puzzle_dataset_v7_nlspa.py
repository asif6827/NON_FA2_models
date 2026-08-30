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
   - novelty remains diagnostic and does not gate the binary reward.

Binary scalar reward
--------------------
The detailed S/PA/interleaving metrics are retained for diagnostics only.

    R = 1.0  iff ALL required checks pass
    R = 0.0  otherwise

Novelty is logged but is NOT a binary reward gate.

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


def _binary_reward(x: Any) -> float:
    """Return exactly 1.0 for truthy success, otherwise 0.0."""
    return 1.0 if bool(x) else 0.0


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
    Diagnostic process PRM only; NOT used by the scalar binary reward.

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
        "interleave_ok": 0.0,
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
        # Binary all-correct reward gates.
        "all_correct": 0.0,
        "gate_required_inputs": 0.0,
        "gate_output_format": 0.0,
        "gate_interleave": 0.0,
        "gate_s_all_parsed": 0.0,
        "gate_s_all_valid": 0.0,
        "gate_s_no_contradiction": 0.0,
        "gate_s_solution_consistent": 0.0,
        "gate_pa_present": 0.0,
        "gate_pa_nonempty": 0.0,
        "gate_pa_structure": 0.0,
        "gate_pa_cells_correct": 0.0,
        "gate_pa_prefix_supported": 0.0,
        "gate_pa_monotonic": 0.0,
        "gate_pa_progress": 0.0,
        "gate_final_solution": 0.0,
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
            out["acc"] = out["score"] = out["reward_logged"] = 0.0
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
        out["interleave_ok"] = 1.0 if schema.get("interleave_ok", False) else 0.0

        # Interleaving is binary for the scalar objective. Fractional
        # component scores are kept only as diagnostics.
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

        # ---------------- BINARY ALL-CORRECT REWARD ----------------
        #
        # R = 1.0 iff ALL gates pass; otherwise R = 0.0.
        #
        # Novelty is intentionally NOT a gate. It remains an analysis metric.

        n_s = int(schema.get("n_s", 0) or 0)
        n_pa = int(schema.get("n_pa", 0) or 0)

        gate_required_inputs = bool(required_inputs)

        # Requires valid <answer> wrapper, exact top-level schema, strict
        # NL/S/PA value/grid schema, and the PA-mandatory trajectory.
        gate_output_format = bool(format_ok)
        gate_interleave = bool(schema.get("interleave_ok", False))

        gate_s_all_parsed = bool(
            n_s > 0
            and int(out["BASE_n_steps_parsed_ok"]) == n_s
        )
        gate_s_all_valid = bool(
            n_s > 0
            and int(out["BASE_n_steps_valid"]) == n_s
        )
        gate_s_no_contradiction = bool(
            int(out["BASE_n_non_valid_contradiction"]) == 0
        )
        gate_s_solution_consistent = bool(
            reasoning_solution_consistency == 1.0
        )

        gate_pa_present = bool(
            n_pa >= 1
            and pa_metrics.get("pa_present", 0.0) == 1.0
        )
        gate_pa_nonempty = bool(
            pa_metrics.get("PA_resolved_cells", 0.0) > 0.0
        )
        gate_pa_structure = bool(
            n_pa >= 1
            and int(pa_metrics.get("PA_n_total", 0.0)) == n_pa
            and int(pa_metrics.get("PA_n_structurally_valid", 0.0)) == n_pa
            and pa_metrics.get("pa_structure_score", 0.0) == 1.0
        )
        gate_pa_cells_correct = bool(
            pa_metrics.get("PA_incorrect_resolved_cells", 0.0) == 0.0
            and pa_metrics.get("pa_cell_precision", 0.0) == 1.0
        )
        gate_pa_prefix_supported = bool(
            pa_metrics.get("PA_unsupported_resolved_cells", 0.0) == 0.0
            and pa_metrics.get("pa_prefix_support_score", 0.0) == 1.0
        )
        gate_pa_monotonic = bool(
            pa_metrics.get("pa_monotonicity_score", 0.0) == 1.0
        )
        gate_pa_progress = bool(
            pa_metrics.get("pa_transition_progress_score", 0.0) == 1.0
        )

        # Exact final solution. CELL_ACCURACY is included as an additional
        # defensive check even though PUZZLE_ACCURACY==1 should imply it.
        gate_final_solution = bool(
            float(puzzle_acc) == 1.0
            and float(cell_acc) == 1.0
        )

        gates = {
            "gate_required_inputs": gate_required_inputs,
            "gate_output_format": gate_output_format,
            "gate_interleave": gate_interleave,
            "gate_s_all_parsed": gate_s_all_parsed,
            "gate_s_all_valid": gate_s_all_valid,
            "gate_s_no_contradiction": gate_s_no_contradiction,
            "gate_s_solution_consistent": gate_s_solution_consistent,
            "gate_pa_present": gate_pa_present,
            "gate_pa_nonempty": gate_pa_nonempty,
            "gate_pa_structure": gate_pa_structure,
            "gate_pa_cells_correct": gate_pa_cells_correct,
            "gate_pa_prefix_supported": gate_pa_prefix_supported,
            "gate_pa_monotonic": gate_pa_monotonic,
            "gate_pa_progress": gate_pa_progress,
            "gate_final_solution": gate_final_solution,
        }

        for gate_name, gate_value in gates.items():
            out[gate_name] = 1.0 if gate_value else 0.0

        all_correct = all(gates.values())
        reward = 1.0 if all_correct else 0.0
        out["all_correct"] = reward

        out["acc"] = reward
        out["score"] = reward
        out["reward_logged"] = reward
        return out

    except Exception:
        logger.exception("Crash in NLSPA reward computation")
        out["reward_exception"] = 1.0
        out["acc"] = out["score"] = out["reward_logged"] = 0.0
        return out


if __name__ == "__main__":
    # ============================================================
    # BINARY ALL-CORRECT REWARD REGRESSION SUITE
    # ============================================================
    #
    # Required scalar formulation:
    #
    #   all correct -> R = 1.0
    #   otherwise   -> R = 0.0
    #
    # Novelty remains diagnostic and does not gate R.

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
        "NL1": "Clues 1 and 3 together show that Arnold must occupy house 2.",
        "S1": "Arnold == 2.",
        "NL2": "Because Fred is somewhere to the left of Eric, Eric cannot occupy house 1.",
        "S2": "Eric != 1.",
        "NL3": "Arnold is already in house 2, so Eric must occupy house 3.",
        "S3": "Eric == 3.",
        "PA1": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "?", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "Eric", "?", "?"],
            ],
        },
        "NL4": "The remaining person Peter must occupy house 1.",
        "S4": "Peter == 1.",
        "NL5": "Fred must be in house 1 or house 2.",
        "S5": "Or(Fred == 1, Fred == 2).",
        "NL6": "Bella is in house 1, so Fred must occupy house 2.",
        "S6": "Fred == 2.",
        "NL7": "The remaining child Meredith must occupy house 3.",
        "S7": "Meredith == 3.",
        "PA2": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "Peter", "?", "Bella"],
                ["2", "Arnold", "red", "Fred"],
                ["3", "Eric", "?", "Meredith"],
            ],
        },
        "NL8": "White is with Meredith, so white occupies house 3.",
        "S8": "white == 3.",
        "NL9": "The remaining color yellow occupies house 1.",
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

    GATE_KEYS = [
        "gate_required_inputs",
        "gate_output_format",
        "gate_interleave",
        "gate_s_all_parsed",
        "gate_s_all_valid",
        "gate_s_no_contradiction",
        "gate_s_solution_consistent",
        "gate_pa_present",
        "gate_pa_nonempty",
        "gate_pa_structure",
        "gate_pa_cells_correct",
        "gate_pa_prefix_supported",
        "gate_pa_monotonic",
        "gate_pa_progress",
        "gate_final_solution",
    ]

    def run_case(name, payload, expected_reward, *, use_wrapper=True):
        text = (
            wrap_answer(payload)
            if use_wrapper
            else json.dumps(payload, indent=2, ensure_ascii=False)
        )
        result = compute_score(
            text,
            copy.deepcopy(GROUND_TRUTH),
        )

        failed_gates = [
            key for key in GATE_KEYS
            if result.get(key, 0.0) != 1.0
        ]

        print("\n" + "=" * 88)
        print(name)
        print("-" * 88)
        print(json.dumps({
            "reward": result["score"],
            "all_correct": result["all_correct"],
            "PUZZLE_ACCURACY": result["PUZZLE_ACCURACY"],
            "CELL_ACCURACY": result["CELL_ACCURACY"],
            "NLSPA_format_ok": result["NLSPA_format_ok"],
            "interleave_ok": result["interleave_ok"],
            "NLSPA_n_s": result["NLSPA_n_s"],
            "NLSPA_n_pa": result["NLSPA_n_pa"],
            "s_parse_ratio": result["s_parse_ratio"],
            "s_validity_ratio": result["s_validity_ratio"],
            "s_novelty_ratio": result["s_novelty_ratio"],
            "s_contradiction_ratio": result["s_contradiction_ratio"],
            "consistency_score": result["consistency_score"],
            "pa_structure_score": result["pa_structure_score"],
            "pa_cell_precision": result["pa_cell_precision"],
            "pa_prefix_support_score": result["pa_prefix_support_score"],
            "pa_monotonicity_score": result["pa_monotonicity_score"],
            "pa_transition_progress_score": result["pa_transition_progress_score"],
            "failed_gates": failed_gates,
        }, indent=2, ensure_ascii=False))

        assert result["score"] == expected_reward, (
            name,
            expected_reward,
            result["score"],
            failed_gates,
        )
        assert result["all_correct"] == expected_reward
        return result

    # 1) Everything correct -> 1.
    run_case(
        "1. ALL CORRECT",
        make_prediction(),
        1.0,
    )

    # 2) PA is mandatory.
    p = make_prediction()
    p["reasoning"] = {
        k: v for k, v in p["reasoning"].items()
        if not k.startswith("PA")
    }
    r = run_case("2. NO PA -> 0", p, 0.0)
    assert r["gate_interleave"] == 0.0
    assert r["gate_pa_present"] == 0.0

    # 3) PA cannot appear between NL_i and S_i.
    p = make_prediction()
    old = p["reasoning"]
    bad = {}
    for key, value in old.items():
        if key == "S3":
            bad["PA1"] = copy.deepcopy(old["PA1"])
        if key == "PA1":
            continue
        bad[key] = copy.deepcopy(value)
    p["reasoning"] = bad
    r = run_case("3. PA BETWEEN NL3/S3 -> 0", p, 0.0)
    assert r["gate_interleave"] == 0.0

    # 4) Consecutive PAs are forbidden: a new NL/S pair is required.
    p = make_prediction()
    old = p["reasoning"]
    bad = {}
    for key, value in old.items():
        bad[key] = copy.deepcopy(value)
        if key == "PA1":
            bad["PA2"] = copy.deepcopy(old["PA2"])
        if key == "PA2":
            continue
    p["reasoning"] = bad
    r = run_case("4. CONSECUTIVE PA -> 0", p, 0.0)
    assert r["gate_interleave"] == 0.0

    # 5) Last reasoning item cannot be PA.
    p = make_prediction()
    p["reasoning"] = {
        "NL1": BASE_REASONING["NL1"],
        "S1": BASE_REASONING["S1"],
        "PA1": copy.deepcopy(BASE_REASONING["PA1"]),
    }
    r = run_case("5. PA AT END -> 0", p, 0.0)
    assert r["gate_interleave"] == 0.0

    # 6) Wrong PA cell -> 0.
    p = make_prediction()
    p["reasoning"]["PA1"]["rows"][0][2] = "white"
    p["reasoning"]["PA2"]["rows"][0][2] = "white"
    r = run_case("6. WRONG PA CELL -> 0", p, 0.0)
    assert r["gate_pa_cells_correct"] == 0.0

    # 7) GT-correct but premature PA cell -> 0 because prefix unsupported.
    p = make_prediction()
    p["reasoning"]["PA1"]["rows"][0][1] = "Peter"
    r = run_case("7. PREMATURE PA CELL -> 0", p, 0.0)
    assert r["gate_pa_prefix_supported"] == 0.0

    # 8) Non-monotonic PA -> 0.
    p = make_prediction()
    p["reasoning"]["PA2"]["rows"][1][1] = "?"
    r = run_case("8. NON-MONOTONIC PA -> 0", p, 0.0)
    assert r["gate_pa_monotonic"] == 0.0

    # 9) PA2 makes no new progress -> 0.
    p = make_prediction()
    p["reasoning"]["PA2"] = copy.deepcopy(p["reasoning"]["PA1"])
    r = run_case("9. PA WITHOUT NEW GRID PROGRESS -> 0", p, 0.0)
    assert r["gate_pa_progress"] == 0.0

    # 10) Contradictory S -> 0.
    p = make_prediction()
    p["reasoning"]["S4"] = "Peter == 2."
    r = run_case("10. CONTRADICTORY S -> 0", p, 0.0)
    assert (
        r["gate_s_all_valid"] == 0.0
        or r["gate_s_no_contradiction"] == 0.0
        or r["gate_s_solution_consistent"] == 0.0
    )

    # 11) Unparseable S -> 0.
    p = make_prediction()
    p["reasoning"]["S4"] = "Peter occupies house 1."
    r = run_case("11. UNPARSEABLE S -> 0", p, 0.0)
    assert r["gate_s_all_parsed"] == 0.0

    # 12) Wrong final solution -> 0.
    p = make_prediction()
    p["solution"]["rows"][0][1] = "Eric"
    p["solution"]["rows"][2][1] = "Peter"
    r = run_case("12. WRONG FINAL SOLUTION -> 0", p, 0.0)
    assert r["gate_final_solution"] == 0.0

    # 13) Missing required <answer> wrapper -> 0.
    p = make_prediction()
    r = run_case(
        "13. DIRECT JSON WITHOUT <answer> -> 0",
        p,
        0.0,
        use_wrapper=False,
    )
    assert r["gate_output_format"] == 0.0

    # 14) Missing S2 / broken NL-S pairing -> 0.
    p = make_prediction()
    p["reasoning"].pop("S2")
    r = run_case("14. BROKEN NL/S PAIR -> 0", p, 0.0)
    assert r["gate_interleave"] == 0.0

    # 15) Non-novel but valid repeated deduction is NOT itself a reward gate.
    # Add one final valid NL/S pair repeating Arnold==2.
    p = make_prediction()
    p["reasoning"]["NL10"] = "Arnold remains in house 2."
    p["reasoning"]["S10"] = "Arnold == 2."
    r = run_case("15. VALID BUT REDUNDANT S STEP: novelty is diagnostic", p, 1.0)
    assert r["s_novelty_ratio"] <= 1.0
    assert r["gate_s_all_valid"] == 1.0

    print("\n" + "=" * 88)
    print("ALL BINARY REWARD REGRESSION CASES PASSED.")
