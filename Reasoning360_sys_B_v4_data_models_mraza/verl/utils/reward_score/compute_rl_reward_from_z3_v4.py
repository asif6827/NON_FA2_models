# -*- coding: utf-8 -*-
"""
compute_rl_reward_from_z3_v3.py

Reward shaping for Zebra/Logic puzzles using:
- puzzle_accuracy, cell_accuracy
- Z3 validator output (z3_out), emphasizing STRICT entailment

Key improvements vs v2:
- Uses strict entailed rate (strict_status == "ENTAILED") instead of chain entailed.
- Denominator defaults to total reasoning steps (n_steps_total).
- Skip penalty: penalizes parse failures + underconstrained steps (anything not parsed_ok).
- Mid/Late shaping: uses pass_rate^gamma with gamma=2 by default.
- Tapered blend after epoch_start (default 25) instead of a hard switch.
- Safe for deployment: handles z3_out=None, missing fields, bad types.

Expected z3_out (from your validator):
- base_sat_full (bool|None), base_sat_raw (bool|None)
- steps: list[dict] where each step dict may include:
    parsed_ok: bool
    strict_status: "ENTAILED" | "NOT_ENTAILED" | "CONTRADICTION" | "UNKNOWN" | "PARSE_ERROR" | "BASE_UNSAT"
    chain_status: same labels as above
    parse_error: str|None

Optionally precomputed:
- n_steps_total, n_steps_parsed_ok
- n_steps_entailed_strict, n_steps_contradiction_chain
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math


# ------------------------- small helpers -------------------------

def _clamp01(x: Optional[float]) -> float:
    try:
        if x is None:
            return 0.0
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default

def _sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def _blend_weight(
    epoch: Optional[int],
    total_epoch: Optional[int],
    *,
    epoch_start: int = 25,
    ramp_width: Optional[float] = None,
) -> float:
    """
    Returns alpha in [0,1] used to blend from base-only -> step-based shaping.
    alpha ~ 0 before epoch_start, then ramps smoothly to ~1.

    If epoch is None, returns 1 (assume we want full shaping).
    """
    if epoch is None:
        return 1.0
    e = int(epoch)

    # pick a reasonable default ramp width if not provided
    if ramp_width is None:
        if total_epoch is None or total_epoch <= 0:
            ramp_width = 8.0
        else:
            ramp_width = max(6.0, 0.12 * float(total_epoch))

    return float(_sigmoid((float(e) - float(epoch_start)) / float(ramp_width)))


# ------------------------- main reward -------------------------

def compute_rl_reward_from_z3_v3(
    z3_out: Optional[Dict[str, Any]],
    puzzle_accuracy: Optional[float],
    cell_accuracy: Optional[float],
    *,
    # curriculum inputs
    epoch: Optional[int] = None,
    total_epoch: Optional[int] = None,
    epoch_start: int = 25,
    ramp_width: Optional[float] = None,

    # base shaping (your established formula)
    base_cap: float = 0.95,
    base_a: float = 0.2,
    base_b: float = 0.75,

    # step shaping
    step_cap: float = 0.99,         # allow "almost perfect" non-solved rewards
    gamma: float = 2.0,             # pass_rate exponent (mid/late)
    beta: float = 1.25,             # coverage exponent
    skip_lambda: float = 2.0,       # exp(-skip_lambda * skip_rate)
    quality_scale: float = 1.15,    # slight lift so high-quality reasoning approaches base

    # safety / honesty caps
    cap_if_unsat: float = 0.25,     # if base_sat_full is False
    cap_if_contradiction: float = 0.35,  # if chain contradictions appear

    # solved gating
    solved_requires_consistency: bool = True,
    solved_skip_rate_max: float = 0.20,  # if too many skipped steps, don't pay full 1.0

    # perfect-puzzle shaping (when puzzle_accuracy == 1.0)
    perfect_mode: str = "penalty",  # "penalty" => 1 - small_penalties, "blend" => 0.9 + 0.1*reasoning_score
    perfect_floor: float = 0.90,
    perfect_mix: float = 0.10,
    perfect_skip_weight: float = 0.06,
    perfect_contra_weight: float = 0.12,
    perfect_penalty_cap: float = 0.20,
) -> Tuple[float, Dict[str, Any]]:
    """
    Policy summary:
      - base = min(base_cap, base_a + base_b * cell_acc)
      - if puzzle_acc == 1:
          - reward = 1.0 iff consistent enough (optional gating), else fallback to base
      - else:
          - if no strict-entailed steps: reward = base
          - else: step_reward ~ base * step_quality(pass_rate^gamma * coverage^beta * exp(-lambda*skip_rate))
          - reward = (1-alpha)*base + alpha*step_reward  (tapered after epoch_start)
      - honesty caps: UNSAT or contradictions cap reward
    """

    # --- normalize accuracies
    p = _clamp01(puzzle_accuracy)
    c = _clamp01(cell_accuracy)
    base = min(float(base_cap), float(base_a) + float(base_b) * c)

    # --- sanitize z3_out
    if not isinstance(z3_out, dict):
        z3_out = {}
    steps = z3_out.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    # --- step counts (prefer precomputed, else compute from steps list)
    n_total = _safe_int(z3_out.get("n_steps_total"), len(steps))

    n_parsed_ok = z3_out.get("n_steps_parsed_ok")
    n_entailed_strict = z3_out.get("n_steps_entailed_strict")
    n_contra_chain = z3_out.get("n_steps_contradiction_chain")

    if n_parsed_ok is None or n_entailed_strict is None or n_contra_chain is None:
        parsed_ok = 0
        entailed_strict = 0
        contra_chain = 0
        parse_fail = 0
        underconstrained = 0

        for s in steps:
            if not isinstance(s, dict):
                continue

            ok = (s.get("parsed_ok") is True)
            if ok:
                parsed_ok += 1
            else:
                parse_fail += 1
                pe = s.get("parse_error")
                if isinstance(pe, str) and ("underconstrained" in pe.lower()):
                    underconstrained += 1

            if s.get("strict_status") == "ENTAILED":
                entailed_strict += 1
            if s.get("chain_status") == "CONTRADICTION":
                contra_chain += 1

        if n_parsed_ok is None:
            n_parsed_ok = parsed_ok
        if n_entailed_strict is None:
            n_entailed_strict = entailed_strict
        if n_contra_chain is None:
            n_contra_chain = contra_chain

        # store diagnostics (optional)
        z3_out.setdefault("_diag_parse_fail", parse_fail)
        z3_out.setdefault("_diag_underconstrained", underconstrained)

    n_parsed_ok = _safe_int(n_parsed_ok, 0)
    n_entailed_strict = _safe_int(n_entailed_strict, 0)
    n_contra_chain = _safe_int(n_contra_chain, 0)

    denom_total = max(1, n_total)
    coverage = float(n_parsed_ok) / float(denom_total)
    coverage = 0.0 if coverage < 0.0 else (1.0 if coverage > 1.0 else coverage)

    skip_rate = 1.0 - coverage
    skip_rate = 0.0 if skip_rate < 0.0 else (1.0 if skip_rate > 1.0 else skip_rate)

    pass_rate_strict = float(n_entailed_strict) / float(denom_total)
    pass_rate_strict = 0.0 if pass_rate_strict < 0.0 else (1.0 if pass_rate_strict > 1.0 else pass_rate_strict)

    # --- base sat signals
    base_sat_full = z3_out.get("base_sat_full", z3_out.get("base_sat"))
    base_sat_raw = z3_out.get("base_sat_raw", None)

    # --- Curriculum weight alpha
    alpha = _blend_weight(epoch, total_epoch, epoch_start=epoch_start, ramp_width=ramp_width)

    # --- Solved case: shape reward (avoid flat 1.0 that encourages overfitting)
    if p >= 1.0:
        # reasoning_score in [0,1] from STRICT validation quality
        reasoning_score = (pass_rate_strict ** float(gamma)) * (coverage ** float(beta)) * math.exp(-float(skip_lambda) * skip_rate)
        if reasoning_score < 0.0:
            reasoning_score = 0.0
        elif reasoning_score > 1.0:
            reasoning_score = 1.0

        if str(perfect_mode).lower() == "blend":
            # Option B: 0.9 + 0.1*reasoning_score
            reward = float(perfect_floor) + float(perfect_mix) * float(reasoning_score)
            reason = "puzzle_accuracy=1 => 0.9+0.1*reasoning_score"
        else:
            # Option A: 1.0 - small_penalties(skip, contradictions)
            small_pen = float(perfect_skip_weight) * float(skip_rate) + float(perfect_contra_weight) * float(contra_rate_chain)
            # If you require global consistency for solved puzzles, treat UNSAT as an extra contradiction-like hit.
            if solved_requires_consistency and base_sat_full is False:
                small_pen += float(perfect_contra_weight)
            if small_pen < 0.0:
                small_pen = 0.0
            if small_pen > float(perfect_penalty_cap):
                small_pen = float(perfect_penalty_cap)
            reward = 1.0 - small_pen
            reason = "puzzle_accuracy=1 => 1-small_penalties"

        # clamp [0,1]
        if reward < 0.0:
            reward = 0.0
        elif reward > 1.0:
            reward = 1.0
        metrics = {
            "reward_reason": reason,
            "reward": float(reward),
            "epoch": epoch,
            "total_epoch": total_epoch,
            "alpha": alpha,
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "base": base,
            "base_sat_raw": base_sat_raw,
            "base_sat_full": base_sat_full,
            "n_steps_total": n_total,
            "n_steps_parsed_ok": n_parsed_ok,
            "n_steps_entailed_strict": n_entailed_strict,
            "n_steps_contradiction_chain": n_contra_chain,
            "pass_rate_strict": pass_rate_strict,
            "coverage": coverage,
            "skip_rate": skip_rate,
        }
        return float(reward), metrics

    # --- Not solved: build step_quality
    if n_entailed_strict <= 0:
        # no strict-entailed steps => fallback base
        reward = float(base)
        reason = "no_strict_steps_passed => fallback_base"
        step_quality = 0.0
        step_reward = float(base)
    else:
        # step quality combines pass rate + coverage, penalizes skipped steps
        step_quality = (pass_rate_strict ** float(gamma)) * (coverage ** float(beta)) * math.exp(-float(skip_lambda) * skip_rate)

        # tie step shaping to cell score via base (already includes cell accuracy)
        step_reward = min(float(step_cap), float(base) * float(quality_scale) * float(step_quality))

        # tapered blend after epoch_start
        reward = (1.0 - alpha) * float(base) + alpha * float(step_reward)
        reason = "steps_passed => tapered_blend(base, step_reward)"

    # --- Honesty caps
    if base_sat_full is False:
        reward = min(float(reward), float(cap_if_unsat))
        reason += " + cap(base_sat_full=False)"

    if n_contra_chain > 0:
        reward = min(float(reward), float(cap_if_contradiction))
        reason += " + cap(chain_contradictions>0)"

    # --- Final clamp
    if reward < 0.0:
        reward = 0.0
    if reward > 1.0:
        reward = 1.0

    metrics = {
        "reward_reason": reason,
        "reward": float(reward),
        "epoch": epoch,
        "total_epoch": total_epoch,
        "epoch_start": epoch_start,
        "ramp_width": ramp_width,
        "alpha": alpha,
        "puzzle_accuracy": p,
        "cell_accuracy": c,
        "base": base,
        "step_cap": float(step_cap),
        "gamma": float(gamma),
        "beta": float(beta),
        "skip_lambda": float(skip_lambda),
        "quality_scale": float(quality_scale),
        "step_quality": float(step_quality),
        "step_reward": float(step_reward),
        "base_sat_raw": base_sat_raw,
        "base_sat_full": base_sat_full,
        "n_steps_total": n_total,
        "n_steps_parsed_ok": n_parsed_ok,
        "n_steps_entailed_strict": n_entailed_strict,
        "n_steps_contradiction_chain": n_contra_chain,
        "pass_rate_strict": pass_rate_strict,
        "coverage": coverage,
        "skip_rate": skip_rate,
        "diag_parse_fail": _safe_int(z3_out.get("_diag_parse_fail"), 0),
        "diag_underconstrained": _safe_int(z3_out.get("_diag_underconstrained"), 0),
    }
    return float(reward), metrics
