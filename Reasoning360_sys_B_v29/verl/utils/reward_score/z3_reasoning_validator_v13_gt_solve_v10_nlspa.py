# -*- coding: utf-8 -*-
"""
z3_reasoning_validator_v13_gt_solve_v11_nlspa.py

NL/S/PA compatibility layer over the existing
z3_reasoning_validator_v13_gt_solve_v9.py.

IMPORTANT CHANGE FROM v10
-------------------------
The legacy v9 solve_and_validate_payload() computes reasoning-step metrics only
inside:

    if report["base_sat_full_GT"]:

That means a base-solver/GT mismatch makes ALL S-step metrics silently become
zero, even when S1..Sk are perfectly parseable and can be independently
validated.

v11 fixes this by:
1. adapting the new reasoning object to legacy S-lines;
2. running the legacy base solver/GT check;
3. ALWAYS running count_distinct_reasoning_steps_v13_relaxed() independently
   on the emitted S_i trace;
4. merging those S metrics into the returned report.

Thus:
- base_sat_full_GT remains a base-clue/GT diagnostic;
- S parsing, validity, novelty, and contradiction metrics are independent.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from verl.utils.reward_score.nlspa_reward_utils import (
        reasoning_to_legacy_s_lines,
        validate_reasoning_schema,
    )
except Exception:
    from nlspa_reward_utils import (
        reasoning_to_legacy_s_lines,
        validate_reasoning_schema,
    )

try:
    from verl.utils.reward_score import (
        z3_reasoning_validator_v13_gt_solve_v9 as _legacy
    )
except Exception:
    import z3_reasoning_validator_v13_gt_solve_v9 as _legacy


normalize_header = _legacy.normalize_header
normalize_months_in_rows = _legacy.normalize_months_in_rows


def _adapt_reasoning(reasoning: Any):
    if isinstance(reasoning, dict):
        return reasoning_to_legacy_s_lines(reasoning)
    return reasoning


def count_distinct_reasoning_steps_v13_relaxed(
    reasoning_lines,
    *args,
    **kwargs,
):
    return _legacy.count_distinct_reasoning_steps_v13_relaxed(
        _adapt_reasoning(reasoning_lines),
        *args,
        **kwargs,
    )


def validate_reasoning_steps_syntactic_only(
    *,
    reasoning,
    base_solver_with_clues,
    var_map,
    timeout_s,
):
    return _legacy.validate_reasoning_steps_syntactic_only(
        reasoning=_adapt_reasoning(reasoning),
        base_solver_with_clues=base_solver_with_clues,
        var_map=var_map,
        timeout_s=timeout_s,
    )


def _empty_s_metrics() -> Dict[str, Any]:
    return {
        "n_steps_total": 0,
        "n_steps_parsed_ok": 0,
        "n_steps_valid": 0,
        "n_steps_novel_inc_clues": 0,
        "n_steps_novel_exc_clues": 0,
        "n_non_valid_contradiction": 0,
        "list_all_steps": [],
        "list_steps_valid": [],
        "list_steps_non_valid": [],
        "list_novel_steps_inc_clues": [],
        "list_novel_steps_exc_clues": [],
        "list_skipped_steps_inc_clues": [],
        "list_skipped_steps_exc_clues": [],
        "list_clue_parse_errors": [],
        "list_step_parse_errors": [],
    }


def solve_and_validate_payload(
    payload: Dict[str, Any],
    *,
    timeout_s: float = 2.0,
    conflict_tolerant_clues: bool = False,
) -> Dict[str, Any]:
    p = dict(payload or {})
    raw_reasoning = p.get("reasoning")

    gt = p.get("ground_truth")
    expected_header = (
        gt.get("header", [])
        if isinstance(gt, dict)
        else []
    )

    schema = None
    if isinstance(raw_reasoning, dict):
        schema = validate_reasoning_schema(
            raw_reasoning,
            n_houses=int(p.get("n_houses") or 0),
            expected_header=expected_header,
            attribute_values=p.get("attribute_values") or {},
        )
        s_lines = list(schema["s_lines"])
    else:
        s_lines = list(raw_reasoning or [])

    p["reasoning"] = s_lines

    # ------------------------------------------------------------
    # A) Legacy base-clue / GT check.
    # ------------------------------------------------------------
    try:
        out = _legacy.solve_and_validate_payload(
            p,
            timeout_s=timeout_s,
            conflict_tolerant_clues=conflict_tolerant_clues,
        )
        if not isinstance(out, dict):
            out = {}
        out["base_validation_exception"] = ""
    except Exception as exc:
        out = {
            "base_sat_full_GT": False,
            "parse_status": "BASE_VALIDATION_EXCEPTION",
            "base_validation_exception":
                f"{type(exc).__name__}: {exc}",
        }

    # ------------------------------------------------------------
    # B) ALWAYS compute S-step analysis independently.
    #
    # Do NOT gate this on base_sat_full_GT.
    # ------------------------------------------------------------
    s_metrics = _empty_s_metrics()
    try:
        s_metrics = _legacy.count_distinct_reasoning_steps_v13_relaxed(
            reasoning_lines=s_lines,
            n_houses=int(p["n_houses"]),
            attribute_values=p["attribute_values"],
            syntactic_clues=p.get("syntactic_clues") or [],
            timeout_s=timeout_s,
            distinct_from_syntactic_clues_semantic_xor=True,
            entailed_only=True,
            require_token_novelty=False,
            omit_tautologies=True,
        )
        if not isinstance(s_metrics, dict):
            s_metrics = _empty_s_metrics()
        out["s_analysis_status"] = "SUCCESS"
        out["s_analysis_exception"] = ""
    except Exception as exc:
        s_metrics = _empty_s_metrics()
        out["s_analysis_status"] = "FAIL"
        out["s_analysis_exception"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # Independent S analysis takes precedence over v9's conditionally
    # populated step fields.
    out.update(s_metrics)

    # Preserve the base parse status separately; expose a combined status too.
    base_status = str(out.get("parse_status", ""))
    out["base_parse_status"] = base_status
    if out.get("s_analysis_status") == "SUCCESS":
        out["parse_status"] = "NLSPA_S_ANALYSIS_SUCCESS"
    else:
        out["parse_status"] = "NLSPA_S_ANALYSIS_FAIL"

    if schema is not None:
        out.update({
            "nlspa_schema_ok": bool(schema["ok"]),
            "nlspa_n_nl": int(schema["n_nl"]),
            "nlspa_n_s": int(schema["n_s"]),
            "nlspa_n_pa": int(schema["n_pa"]),
            "nlspa_format_error_count": int(len(schema["errors"])),
        })

    return out


def __getattr__(name: str):
    return getattr(_legacy, name)


if __name__ == "__main__":
    payload = {
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
            "NL2": "Fred is to the left of Eric, so Eric cannot be in house 1.",
            "S2": "Eric != 1.",
            "NL3": "Arnold already occupies house 2, so Eric must occupy house 3.",
            "S3": "Eric == 3.",
            "PA1": {
                "header": ["House", "Name", "Color", "Children"],
                "rows": [
                    ["1", "?", "?", "Bella"],
                    ["2", "Arnold", "red", "?"],
                    ["3", "Eric", "?", "?"],
                ],
            },
        },
        "ground_truth": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "Peter", "yellow", "Bella"],
                ["2", "Arnold", "red", "Fred"],
                ["3", "Eric", "white", "Meredith"],
            ],
        },
    }

    result = solve_and_validate_payload(payload, timeout_s=5.0)
    assert result["n_steps_total"] == 3, result
    assert result["n_steps_parsed_ok"] == 3, result
    assert result["s_analysis_status"] == "SUCCESS", result

    print("v11 NL/S/PA S-analysis regression test: PASS")
    print(result)
