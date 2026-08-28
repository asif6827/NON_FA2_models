# -*- coding: utf-8 -*-
"""
check_interleved_format_nlspa.py

Format checker for the new Zebra NL_i / S_i / PA_i reasoning object.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:
    from verl.utils.reward_score.nlspa_reward_utils import validate_reasoning_schema
except Exception:
    from nlspa_reward_utils import validate_reasoning_schema


def check_interleaved_reasoning_detailed(
    reasoning: Any,
    *,
    n_houses: int = 0,
    expected_header: List[str] | None = None,
    attribute_values: Dict[str, List[Any]] | None = None,
    require_terminal_period: bool = True,
) -> Dict[str, Any]:
    return validate_reasoning_schema(
        reasoning,
        n_houses=int(n_houses or 0),
        expected_header=list(expected_header or []),
        attribute_values=dict(attribute_values or {}),
        require_terminal_period=require_terminal_period,
    )


def check_interleaved_reasoning(
    reasoning: Any,
    *,
    n_houses: int = 0,
    expected_header: List[str] | None = None,
    attribute_values: Dict[str, List[Any]] | None = None,
    require_terminal_period: bool = True,
) -> bool:
    return bool(
        check_interleaved_reasoning_detailed(
            reasoning,
            n_houses=n_houses,
            expected_header=expected_header,
            attribute_values=attribute_values,
            require_terminal_period=require_terminal_period,
        )["ok"]
    )


if __name__ == "__main__":
    header = ["House", "Name", "Color", "Children"]
    attribute_values = {
        "Name": ["Peter", "Eric", "Arnold"],
        "Color": ["red", "white", "yellow"],
        "Children": ["Fred", "Meredith", "Bella"]
    }

    valid_reasoning = {
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
                ["3", "Eric", "?", "?"]
            ]
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
                ["3", "Eric", "?", "Meredith"]
            ]
        },
        "NL8": "Clue 5 places white in the same house as Meredith, so white must occupy house 3.",
        "S8": "white == 3.",
        "NL9": "Red is already in house 2 and white is in house 3, so color uniqueness forces yellow into house 1.",
        "S9": "yellow == 1."
    }

    invalid_reasoning = {
        "NL1": "Arnold must occupy house 2.",
        "PA1": {
            "header": header,
            "rows": [
                ["1", "?", "?", "?"],
                ["2", "Arnold", "?", "?"],
                ["3", "?", "?", "?"]
            ]
        },
        "S1": "Arnold == 2."
    }

    valid = check_interleaved_reasoning_detailed(
        valid_reasoning,
        n_houses=3,
        expected_header=header,
        attribute_values=attribute_values,
    )
    invalid = check_interleaved_reasoning_detailed(
        invalid_reasoning,
        n_houses=3,
        expected_header=header,
        attribute_values=attribute_values,
    )

    assert valid["ok"], valid["errors"]
    assert not invalid["ok"]
    print("Valid NL/S/PA format: PASS")
    print("Invalid PA-between-NL-and-S format rejected: PASS")
    print("Invalid errors:", invalid["errors"])
