# -*- coding: utf-8 -*-
"""
z3_reasoning_vs_solution_verifier_v4_nlspa.py

NL/S/PA-aware wrapper around the ACTUAL
z3_reasoning_vs_solution_verifier_v2.py.

The old v3 wrapper incorrectly passed arguments/functions that v2 does not
provide. In particular:
- verify_solution_two_step() in v2 has NO `people=` argument.
- v2 does not expose verify_knights_solution_two_step().
- v2 does not expose verify_zebra_solution_two_step() as a separate function.

Those mismatches were being swallowed by the reward code and caused
consistency_score = 0.0.

v4 adapts only the reasoning representation and otherwise preserves the exact
v2 API.
"""

from __future__ import annotations

from typing import Any, Dict

try:
    from verl.utils.reward_score.nlspa_reward_utils import (
        reasoning_to_legacy_s_lines,
    )
except Exception:
    from nlspa_reward_utils import reasoning_to_legacy_s_lines

try:
    from verl.utils.reward_score import (
        z3_reasoning_vs_solution_verifier_v2 as _legacy
    )
except Exception:
    import z3_reasoning_vs_solution_verifier_v2 as _legacy


def _adapt(reasoning: Any):
    if isinstance(reasoning, dict):
        return reasoning_to_legacy_s_lines(reasoning)
    return reasoning


def extract_syntactic_steps_with_evidence(reasoning: Any):
    return _legacy.extract_syntactic_steps_with_evidence(
        _adapt(reasoning)
    )


def verify_solution_two_step(
    syntactic_clues: Any,
    reasoning: Any,
    solution: Dict[str, Any],
    *,
    enforce_alldiff: bool = True,
):
    return _legacy.verify_solution_two_step(
        syntactic_clues,
        _adapt(reasoning),
        solution,
        enforce_alldiff=enforce_alldiff,
    )


def validate_reasoning_and_solution(
    syntactic_clues: Any,
    reasoning: Any,
    solution: Dict[str, Any],
    *,
    enforce_alldiff: bool = True,
):
    return _legacy.validate_reasoning_and_solution(
        syntactic_clues,
        _adapt(reasoning),
        solution,
        enforce_alldiff=enforce_alldiff,
    )


def __getattr__(name: str):
    return getattr(_legacy, name)


if __name__ == "__main__":
    syntactic_clues = [
        "C1: Arnold == red.",
        "C2: Fred < Eric.",
        "C3: red == 2.",
        "C4: Bella == 1.",
        "C5: white == Meredith.",
    ]

    reasoning = {
        "NL1": "Clues 1 and 3 together place Arnold in house 2.",
        "S1": "Arnold == 2.",
        "NL2": "Fred is left of Eric, so Eric cannot be in house 1.",
        "S2": "Eric != 1.",
        "NL3": "Arnold occupies house 2, so Eric must occupy house 3.",
        "S3": "Eric == 3.",
        "PA1": {
            "header": ["House", "Name", "Color", "Children"],
            "rows": [
                ["1", "?", "?", "Bella"],
                ["2", "Arnold", "red", "?"],
                ["3", "Eric", "?", "?"],
            ],
        },
    }

    solution = {
        "header": ["House", "Name", "Color", "Children"],
        "rows": [
            ["1", "Peter", "yellow", "Bella"],
            ["2", "Arnold", "red", "Fred"],
            ["3", "Eric", "white", "Meredith"],
        ],
    }

    steps, errors = extract_syntactic_steps_with_evidence(reasoning)
    assert not errors, errors
    assert len(steps) == 3

    result = verify_solution_two_step(
        syntactic_clues,
        reasoning,
        solution,
    )
    assert result["r1"] == 1.0, result
    assert result["r2"] == 1.0, result
    assert result["reward"] == 1.0, result

    print("v4 NL/S/PA final-solution verifier regression test: PASS")
    print(result)
