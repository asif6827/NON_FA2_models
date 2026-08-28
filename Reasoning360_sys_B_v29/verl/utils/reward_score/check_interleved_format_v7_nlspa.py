# -*- coding: utf-8 -*-
"""
check_interleved_format_nlspa.py

Format checker for the new Zebra NL_i / S_i / PA_i reasoning object.
"""

from __future__ import annotations

from typing import Any, Dict, List
import re

try:
    from verl.utils.reward_score.nlspa_reward_utils import validate_reasoning_schema
except Exception:
    from nlspa_reward_utils import validate_reasoning_schema



_NL_RE = re.compile(r"^NL(\d+)$")
_S_RE = re.compile(r"^S(\d+)$")
_PA_RE = re.compile(r"^PA(\d+)$")


def _key_info(key: Any):
    """Return (kind, index) for NLi/Si/PAi, else (UNKNOWN, None)."""
    s = str(key)
    for kind, pattern in (("NL", _NL_RE), ("S", _S_RE), ("PA", _PA_RE)):
        m = pattern.fullmatch(s)
        if m:
            return kind, int(m.group(1))
    return "UNKNOWN", None


def _mean(values: List[float], default: float = 1.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def compute_interleaving_reward(reasoning: Any) -> Dict[str, Any]:
    """
    Dense partial-credit reward for the NL_i -> S_i -> [PA_k] trajectory.

    R_interleave =
        0.40 * pair_score
      + 0.25 * order_score
      + 0.20 * index_score
      + 0.15 * pa_placement_score

    Components
    ----------
    pair_score:
        Fraction of NL/S opportunities that form an immediate matching
        NL_i -> S_i pair.

    order_score:
        Fraction of adjacent key transitions that are legal:
          NL_i -> S_i
          S_i  -> NL_{i+1}
          S_i  -> PA_k
          PA_k -> NL_{i+1}
        The index relation is included where it is intrinsic to the
        transition. Unknown keys/transitions receive no credit.

    index_score:
        Consecutive numbering quality for NL, S, and PA key streams.
        Each present stream is compared with 1,2,3,... independently.
        If no PA exists, PA numbering is neutral rather than penalized.

    pa_placement_score:
        Fraction of PAs that occur immediately after an S_i. If no PA is
        emitted (PA is optional), this component is 1.0.

    This reward intentionally does NOT evaluate S-step logic or PA-grid
    correctness; those belong to S-PRM and PA-PRM respectively.
    """
    if not isinstance(reasoning, dict) or not reasoning:
        return {
            "interleave_pair_score": 0.0,
            "interleave_order_score": 0.0,
            "interleave_index_score": 0.0,
            "interleave_pa_placement_score": 0.0,
            "interleave_reward": 0.0,
            "interleave_n_correct_pairs": 0.0,
            "interleave_n_pair_opportunities": 0.0,
            "interleave_n_legal_transitions": 0.0,
            "interleave_n_transitions": 0.0,
            "interleave_n_legal_pa": 0.0,
            "interleave_n_pa": 0.0,
        }

    keys = [str(k) for k in reasoning.keys()]
    infos = [_key_info(k) for k in keys]

    n_nl = sum(1 for kind, _ in infos if kind == "NL")
    n_s = sum(1 for kind, _ in infos if kind == "S")
    n_pa = sum(1 for kind, _ in infos if kind == "PA")

    # 1) Exact immediate NL_i -> S_i pairs.
    correct_pairs = 0
    for i in range(len(infos) - 1):
        kind1, idx1 = infos[i]
        kind2, idx2 = infos[i + 1]
        if kind1 == "NL" and kind2 == "S" and idx1 == idx2:
            correct_pairs += 1

    pair_den = max(n_nl, n_s, 1)
    pair_score = correct_pairs / pair_den

    # 2) Legal adjacent transitions.
    legal_transitions = 0
    n_transitions = max(len(infos) - 1, 0)
    for i in range(n_transitions):
        kind1, idx1 = infos[i]
        kind2, idx2 = infos[i + 1]
        legal = False

        if kind1 == "NL" and kind2 == "S":
            legal = idx1 == idx2
        elif kind1 == "S" and kind2 == "NL":
            legal = idx2 == idx1 + 1
        elif kind1 == "S" and kind2 == "PA":
            # PA has its own numbering; its exact PA index is handled by
            # index_score. Structurally it is legal after a completed S.
            legal = True
        elif kind1 == "PA" and kind2 == "NL":
            # Resume with the NL step after the S step preceding this PA.
            prev_s_idx = None
            for j in range(i - 1, -1, -1):
                if infos[j][0] == "S":
                    prev_s_idx = infos[j][1]
                    break
            legal = prev_s_idx is not None and idx2 == prev_s_idx + 1

        if legal:
            legal_transitions += 1

    order_score = (
        legal_transitions / n_transitions
        if n_transitions > 0
        else (1.0 if len(infos) == 1 else 0.0)
    )

    # 3) Consecutive index streams NL1.., S1.., PA1...
    stream_scores = []
    for kind in ("NL", "S", "PA"):
        indices = [idx for k, idx in infos if k == kind]
        if not indices:
            if kind == "PA":
                # PA is optional.
                continue
            stream_scores.append(0.0)
            continue
        correct = sum(
            1 for expected, observed in enumerate(indices, start=1)
            if observed == expected
        )
        stream_scores.append(correct / len(indices))
    index_score = _mean(stream_scores, default=0.0)

    # 4) PA placement: immediately after S_i. No PA => neutral 1.0.
    legal_pa = 0
    for i, (kind, _) in enumerate(infos):
        if kind != "PA":
            continue
        if i > 0 and infos[i - 1][0] == "S":
            legal_pa += 1
    pa_placement_score = (legal_pa / n_pa) if n_pa else 1.0

    interleave_reward = (
        0.40 * pair_score
        + 0.25 * order_score
        + 0.20 * index_score
        + 0.15 * pa_placement_score
    )

    # Numerical safety.
    interleave_reward = max(0.0, min(1.0, float(interleave_reward)))

    return {
        "interleave_pair_score": float(pair_score),
        "interleave_order_score": float(order_score),
        "interleave_index_score": float(index_score),
        "interleave_pa_placement_score": float(pa_placement_score),
        "interleave_reward": interleave_reward,
        "interleave_n_correct_pairs": float(correct_pairs),
        "interleave_n_pair_opportunities": float(pair_den),
        "interleave_n_legal_transitions": float(legal_transitions),
        "interleave_n_transitions": float(n_transitions),
        "interleave_n_legal_pa": float(legal_pa),
        "interleave_n_pa": float(n_pa),
    }


def check_interleaved_reasoning_detailed(
    reasoning: Any,
    *,
    n_houses: int = 0,
    expected_header: List[str] | None = None,
    attribute_values: Dict[str, List[Any]] | None = None,
    require_terminal_period: bool = True,
) -> Dict[str, Any]:
    """Strict schema result + dense R_interleave metrics."""
    strict = validate_reasoning_schema(
        reasoning,
        n_houses=int(n_houses or 0),
        expected_header=list(expected_header or []),
        attribute_values=dict(attribute_values or {}),
        require_terminal_period=require_terminal_period,
    )
    strict.update(compute_interleaving_reward(reasoning))
    return strict


def check_interleaved_reasoning(
    reasoning: Any,
    *,
    n_houses: int = 0,
    expected_header: List[str] | None = None,
    attribute_values: Dict[str, List[Any]] | None = None,
    require_terminal_period: bool = True,
) -> bool:
    """Backward-compatible binary strict-format check."""
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
        "Children": ["Fred", "Meredith", "Bella"],
    }

    PA = {
        "header": header,
        "rows": [
            ["1", "?", "?", "Bella"],
            ["2", "Arnold", "red", "?"],
            ["3", "Eric", "?", "?"],
        ],
    }

    tests = {
        "valid": {
            "NL1": "Arnold must be in house 2.",
            "S1": "Arnold == 2.",
            "NL2": "Eric cannot be in house 1.",
            "S2": "Eric != 1.",
            "PA1": PA,
            "NL3": "Peter must be in house 1.",
            "S3": "Peter == 1.",
        },
        "pa_between_nl_and_s": {
            "NL1": "Arnold must be in house 2.",
            "PA1": PA,
            "S1": "Arnold == 2.",
        },
        "consecutive_nl": {
            "NL1": "Arnold must be in house 2.",
            "NL2": "Eric cannot be in house 1.",
            "S2": "Eric != 1.",
        },
        "skipped_indices": {
            "NL1": "Arnold must be in house 2.",
            "S1": "Arnold == 2.",
            "NL3": "Peter must be in house 1.",
            "S3": "Peter == 1.",
        },
        "no_pa": {
            "NL1": "Arnold must be in house 2.",
            "S1": "Arnold == 2.",
            "NL2": "Eric cannot be in house 1.",
            "S2": "Eric != 1.",
        },
    }

    results = {}
    for name, reasoning in tests.items():
        result = check_interleaved_reasoning_detailed(
            reasoning,
            n_houses=3,
            expected_header=header,
            attribute_values=attribute_values,
        )
        results[name] = result
        print("\n" + "=" * 72)
        print(name)
        for key in (
            "ok",
            "interleave_pair_score",
            "interleave_order_score",
            "interleave_index_score",
            "interleave_pa_placement_score",
            "interleave_reward",
        ):
            print(f"{key}: {result[key]}")

    assert results["valid"]["ok"] is True
    assert results["valid"]["interleave_reward"] == 1.0
    assert results["no_pa"]["interleave_reward"] == 1.0

    assert results["pa_between_nl_and_s"]["ok"] is False
    assert results["pa_between_nl_and_s"]["interleave_reward"] < 1.0
    assert results["pa_between_nl_and_s"]["interleave_pa_placement_score"] < 1.0

    assert results["consecutive_nl"]["interleave_pair_score"] < 1.0
    assert results["consecutive_nl"]["interleave_order_score"] < 1.0

    assert results["skipped_indices"]["interleave_index_score"] < 1.0
    assert results["skipped_indices"]["interleave_reward"] < 1.0

    print("\nAll R_interleave regression checks: PASS")
