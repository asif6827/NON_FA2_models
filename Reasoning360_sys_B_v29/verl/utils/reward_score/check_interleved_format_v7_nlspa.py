# -*- coding: utf-8 -*-
"""
check_interleved_format_v8_binary_nlspa.py

Strict checker for the Zebra NL_i / S_i / PA_i reasoning trajectory.

Target trajectory
-----------------
    (NL,S)^+ -> PA1 -> (NL,S)^+ -> PA2 -> ... -> (NL,S)^+ -> solution

There is NO fixed number of NL/S pairs between PAs.

Binary interleaving condition:
    interleave_reward = 1.0  iff the entire trajectory is valid
    interleave_reward = 0.0  otherwise

Required trajectory rules
-------------------------
1. reasoning is a non-empty JSON object.
2. NL/S numbering starts at 1 and is consecutive.
3. Every NL_i is immediately followed by matching S_i.
4. At least one PA is mandatory.
5. PA numbering starts at 1 and is consecutive.
6. Every PA occurs only after at least one complete NEW NL/S pair since the
   previous PA (or since the beginning for PA1).
7. Consecutive PAs are forbidden.
8. A PA cannot appear between NL_i and S_i.
9. After the last PA, at least one complete NL/S pair must be emitted before
   the final solution; therefore reasoning must end on an S_i, not a PA.

The underlying validate_reasoning_schema() still checks NL/S strings and
PA-grid structure. This file adds the stricter PA-mandatory trajectory rule.

The fractional component scores are retained ONLY as diagnostics. They are
NOT used as additive reward terms.
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
    s = str(key)
    for kind, pattern in (("NL", _NL_RE), ("S", _S_RE), ("PA", _PA_RE)):
        m = pattern.fullmatch(s)
        if m:
            return kind, int(m.group(1))
    return "UNKNOWN", None


def _diagnostic_components(reasoning: Any) -> Dict[str, float]:
    """Fractional diagnostics only; they do not determine scalar reward."""
    if not isinstance(reasoning, dict) or not reasoning:
        return {
            "interleave_pair_score": 0.0,
            "interleave_order_score": 0.0,
            "interleave_index_score": 0.0,
            "interleave_pa_placement_score": 0.0,
            "interleave_n_correct_pairs": 0.0,
            "interleave_n_pair_opportunities": 0.0,
            "interleave_n_legal_pa": 0.0,
            "interleave_n_pa": 0.0,
        }

    infos = [_key_info(k) for k in reasoning.keys()]
    n_nl = sum(kind == "NL" for kind, _ in infos)
    n_s = sum(kind == "S" for kind, _ in infos)
    n_pa = sum(kind == "PA" for kind, _ in infos)

    correct_pairs = sum(
        1
        for i in range(len(infos) - 1)
        if (
            infos[i][0] == "NL"
            and infos[i + 1][0] == "S"
            and infos[i][1] == infos[i + 1][1]
        )
    )
    pair_den = max(n_nl, n_s, 1)
    pair_score = correct_pairs / pair_den

    legal_transitions = 0
    n_transitions = max(len(infos) - 1, 0)
    for i in range(n_transitions):
        k1, i1 = infos[i]
        k2, i2 = infos[i + 1]
        legal = False
        if k1 == "NL" and k2 == "S":
            legal = i1 == i2
        elif k1 == "S" and k2 == "NL":
            legal = i2 == i1 + 1
        elif k1 == "S" and k2 == "PA":
            legal = True
        elif k1 == "PA" and k2 == "NL":
            legal = True
        if legal:
            legal_transitions += 1
    order_score = (
        legal_transitions / n_transitions
        if n_transitions
        else 0.0
    )

    stream_scores: List[float] = []
    for kind in ("NL", "S", "PA"):
        indices = [idx for k, idx in infos if k == kind]
        if not indices:
            if kind == "PA":
                stream_scores.append(0.0)  # PA is mandatory here.
            else:
                stream_scores.append(0.0)
            continue
        correct = sum(
            observed == expected
            for expected, observed in enumerate(indices, start=1)
        )
        stream_scores.append(correct / len(indices))
    index_score = sum(stream_scores) / len(stream_scores)

    legal_pa = sum(
        1
        for i, (kind, _) in enumerate(infos)
        if kind == "PA" and i > 0 and infos[i - 1][0] == "S"
    )
    pa_placement = (legal_pa / n_pa) if n_pa else 0.0

    return {
        "interleave_pair_score": float(pair_score),
        "interleave_order_score": float(order_score),
        "interleave_index_score": float(index_score),
        "interleave_pa_placement_score": float(pa_placement),
        "interleave_n_correct_pairs": float(correct_pairs),
        "interleave_n_pair_opportunities": float(pair_den),
        "interleave_n_legal_pa": float(legal_pa),
        "interleave_n_pa": float(n_pa),
    }


def validate_interleaving_trajectory(reasoning: Any) -> Dict[str, Any]:
    """
    Deterministic state-machine validation of the full NL/S/PA trajectory.
    """
    errors: List[Dict[str, Any]] = []

    if not isinstance(reasoning, dict) or not reasoning:
        return {
            "interleave_ok": False,
            "interleave_reward": 0.0,
            "interleave_errors": [{
                "code": "REASONING_NOT_OBJECT",
                "message": "reasoning must be a non-empty JSON object.",
            }],
            "interleave_n_pairs": 0,
            "interleave_n_pa": 0,
        }

    expected_pair = 1
    expected_pa = 1
    state = "EXPECT_NL"
    pairs_since_last_pa = 0
    n_pairs = 0
    n_pa = 0

    keys = list(reasoning.keys())

    for pos, raw_key in enumerate(keys):
        key = str(raw_key)
        kind, idx = _key_info(key)

        if state == "EXPECT_NL":
            if kind != "NL" or idx != expected_pair:
                errors.append({
                    "position": pos,
                    "key": key,
                    "code": "EXPECTED_NL",
                    "message": f"Expected NL{expected_pair}, found {key}.",
                })
                break
            state = "EXPECT_S"
            continue

        if state == "EXPECT_S":
            if kind != "S" or idx != expected_pair:
                errors.append({
                    "position": pos,
                    "key": key,
                    "code": "EXPECTED_MATCHING_S",
                    "message": (
                        f"NL{expected_pair} must be immediately followed by "
                        f"S{expected_pair}; found {key}."
                    ),
                })
                break
            n_pairs += 1
            pairs_since_last_pa += 1
            expected_pair += 1
            state = "AFTER_PAIR"
            continue

        # AFTER_PAIR: either continue reasoning with the next NL or emit PA.
        if state == "AFTER_PAIR":
            if kind == "NL" and idx == expected_pair:
                state = "EXPECT_S"
                continue

            if kind == "PA":
                if idx != expected_pa:
                    errors.append({
                        "position": pos,
                        "key": key,
                        "code": "PA_NON_CONSECUTIVE",
                        "message": f"Expected PA{expected_pa}, found {key}.",
                    })
                    break
                if pairs_since_last_pa < 1:
                    errors.append({
                        "position": pos,
                        "key": key,
                        "code": "PA_WITHOUT_NEW_REASONING",
                        "message": (
                            f"{key} must follow at least one complete new "
                            "NL/S pair since the previous PA."
                        ),
                    })
                    break

                n_pa += 1
                expected_pa += 1
                pairs_since_last_pa = 0
                state = "EXPECT_NL"
                continue

            errors.append({
                "position": pos,
                "key": key,
                "code": "ILLEGAL_AFTER_PAIR",
                "message": (
                    f"After S{expected_pair - 1}, expected either "
                    f"NL{expected_pair} or PA{expected_pa}; found {key}."
                ),
            })
            break

    # End-state requirements.
    if not errors:
        if state == "EXPECT_S":
            errors.append({
                "position": len(keys),
                "key": None,
                "code": "UNPAIRED_FINAL_NL",
                "message": f"NL{expected_pair} has no matching S{expected_pair}.",
            })
        elif state == "EXPECT_NL":
            # This happens when the final emitted item was a PA.
            errors.append({
                "position": len(keys),
                "key": None,
                "code": "PA_AT_END",
                "message": (
                    "A PA cannot be the final reasoning item; at least one new "
                    "NL/S pair must follow the last PA before the solution."
                ),
            })

    if n_pa < 1:
        errors.append({
            "position": len(keys),
            "key": None,
            "code": "PA_REQUIRED",
            "message": "At least one PA checkpoint is mandatory.",
        })

    ok = not errors
    result = {
        "interleave_ok": bool(ok),
        "interleave_reward": 1.0 if ok else 0.0,
        "interleave_errors": errors,
        "interleave_error_count": float(len(errors)),
        "interleave_n_pairs": float(n_pairs),
        "interleave_n_pa": float(n_pa),
    }
    result.update(_diagnostic_components(reasoning))
    return result


def check_interleaved_reasoning_detailed(
    reasoning: Any,
    *,
    n_houses: int = 0,
    expected_header: List[str] | None = None,
    attribute_values: Dict[str, List[Any]] | None = None,
    require_terminal_period: bool = True,
) -> Dict[str, Any]:
    """
    Full strict result:
      - validate_reasoning_schema(): value/grid/schema checks
      - validate_interleaving_trajectory(): PA-mandatory state machine
    """
    schema = validate_reasoning_schema(
        reasoning,
        n_houses=int(n_houses or 0),
        expected_header=list(expected_header or []),
        attribute_values=dict(attribute_values or {}),
        require_terminal_period=require_terminal_period,
    )
    trajectory = validate_interleaving_trajectory(reasoning)

    schema_errors = list(schema.get("errors", []))
    interleave_errors = list(trajectory.get("interleave_errors", []))

    schema["schema_ok_without_interleave"] = bool(schema.get("ok", False))
    schema["interleave_ok"] = bool(trajectory["interleave_ok"])
    schema["interleave_reward"] = float(trajectory["interleave_reward"])
    schema["interleave_errors"] = interleave_errors
    schema["interleave_error_count"] = float(len(interleave_errors))

    for key, value in trajectory.items():
        if key not in {
            "interleave_ok",
            "interleave_reward",
            "interleave_errors",
            "interleave_error_count",
        }:
            schema[key] = value

    # "ok" now means the entire required NL/S/PA format is correct.
    schema["ok"] = bool(
        schema["schema_ok_without_interleave"]
        and trajectory["interleave_ok"]
    )
    schema["errors"] = schema_errors + interleave_errors
    return schema


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
        # (NL,S)^+ -> PA -> (NL,S)^+
        "valid": {
            "NL1": "Arnold must be in house 2.",
            "S1": "Arnold == 2.",
            "NL2": "Eric cannot be in house 1.",
            "S2": "Eric != 1.",
            "PA1": PA,
            "NL3": "Peter must be in house 1.",
            "S3": "Peter == 1.",
        },
        "no_pa": {
            "NL1": "Arnold must be in house 2.",
            "S1": "Arnold == 2.",
            "NL2": "Eric cannot be in house 1.",
            "S2": "Eric != 1.",
        },
        "pa_between_nl_and_s": {
            "NL1": "Arnold must be in house 2.",
            "PA1": PA,
            "S1": "Arnold == 2.",
        },
        "consecutive_pa": {
            "NL1": "Arnold must be in house 2.",
            "S1": "Arnold == 2.",
            "PA1": PA,
            "PA2": PA,
            "NL2": "Peter must be in house 1.",
            "S2": "Peter == 1.",
        },
        "pa_at_end": {
            "NL1": "Arnold must be in house 2.",
            "S1": "Arnold == 2.",
            "PA1": PA,
        },
        "skipped_pair_index": {
            "NL1": "Arnold must be in house 2.",
            "S1": "Arnold == 2.",
            "PA1": PA,
            "NL3": "Peter must be in house 1.",
            "S3": "Peter == 1.",
        },
    }

    expected = {
        "valid": True,
        "no_pa": False,
        "pa_between_nl_and_s": False,
        "consecutive_pa": False,
        "pa_at_end": False,
        "skipped_pair_index": False,
    }

    for name, reasoning in tests.items():
        result = check_interleaved_reasoning_detailed(
            reasoning,
            n_houses=3,
            expected_header=header,
            attribute_values=attribute_values,
        )
        print("\n" + "=" * 72)
        print(name)
        print("ok:", result["ok"])
        print("interleave_ok:", result["interleave_ok"])
        print("interleave_reward:", result["interleave_reward"])
        print("errors:", result["interleave_errors"])
        assert result["interleave_ok"] is expected[name]
        assert result["interleave_reward"] == (1.0 if expected[name] else 0.0)

    print("\nAll binary interleaving regression checks: PASS")
