# -*- coding: utf-8 -*-
"""
compute_rl_reward_from_z3_option_a.py

Option A (safest): bounded bonus-on-top of your existing base reward.

Base (your current):
  base = min(base_cap, base_a + base_b * cell_accuracy)

Final reward:
  if puzzle_accuracy == 1 -> 1.0
  else -> min(base_cap, base + lam * base * (pass_rate ** gamma))

Where:
  pass_rate = n_steps_entailed_strict / n_steps_total   (STRICT entailment, denominator = total steps)

This file is intentionally small and deployment-safe:
- works even if z3_out is None / missing fields / bad types
- computes counts from z3_out["steps"] if precomputed fields are missing
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def compute_rl_reward_from_z3_option_a(
    z3_out: Optional[Dict[str, Any]],
    puzzle_accuracy: Optional[float],
    cell_accuracy: Optional[float],
    *,
    # your base shaping
    base_cap: float = 0.95,
    base_a: float = 0.2,
    base_b: float = 0.75,

    # Option-A reasoning bonus (bounded)
    lam: float = 0.10,      # max ~10% relative boost over base (still capped by base_cap)
    gamma: float = 2.0,     # makes partial pass_rate contribute less (robust to noise)
) -> Tuple[float, Dict[str, Any]]:
    """
    Returns: (reward, metrics)

    Metrics include:
      - base
      - pass_rate_strict (strict entailed / total steps)
      - bonus
      - reward_reason
    """
    p = _clamp01(puzzle_accuracy)
    c = _clamp01(cell_accuracy)

    # Perfect puzzle => max reward
    if p >= 1.0:
        return 1.0, {
            "reward_reason": "puzzle_accuracy=1",
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "reward": 1.0,
        }

    # Base (your proven stable formula)
    base = min(float(base_cap), float(base_a) + float(base_b) * c)

    # Sanitize z3_out
    if not isinstance(z3_out, dict):
        z3_out = {}

    steps = z3_out.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    # Prefer precomputed; otherwise derive from steps
    n_total = _safe_int(z3_out.get("n_steps_total"), len(steps))
    n_entailed_strict = z3_out.get("n_steps_entailed_strict")

    if n_entailed_strict is None:
        entailed_strict = 0
        for s in steps:
            if isinstance(s, dict) and s.get("strict_status") == "ENTAILED":
                entailed_strict += 1
        n_entailed_strict = entailed_strict

    n_total = max(1, _safe_int(n_total, 1))
    n_entailed_strict = _safe_int(n_entailed_strict, 0)

    pass_rate = float(n_entailed_strict) / float(n_total)
    if pass_rate < 0.0:
        pass_rate = 0.0
    if pass_rate > 1.0:
        pass_rate = 1.0

    # Option A: bounded bonus-on-top
    lam = _safe_float(lam, 0.0)
    gamma = _safe_float(gamma, 1.0)
    if lam < 0.0:
        lam = 0.0
    if gamma <= 0.0:
        gamma = 1.0

    bonus = lam * base * (pass_rate ** gamma)
    reward = min(float(base_cap), base + bonus)

    metrics = {
        "reward_reason": "option_a: base + lam*base*pass_rate^gamma (strict/total)",
        "puzzle_accuracy": p,
        "cell_accuracy": c,
        "base": float(base),
        "base_cap": float(base_cap),
        "lam": float(lam),
        "gamma": float(gamma),
        "n_steps_total": int(n_total),
        "n_steps_entailed_strict": int(n_entailed_strict),
        "pass_rate_strict": float(pass_rate),
        "bonus": float(bonus),
        "reward": float(reward),
    }
    return float(reward), metrics


if __name__ == "__main__":
    # Tiny sanity demo
    demo_z3 = {
        "n_steps_total": 5,
        "steps": [
            {"parsed_ok": True, "strict_status": "ENTAILED"},
            {"parsed_ok": True, "strict_status": "ENTAILED"},
            {"parsed_ok": True, "strict_status": "NOT_ENTAILED"},
            {"parsed_ok": False, "strict_status": "PARSE_ERROR"},
            {"parsed_ok": True, "strict_status": "ENTAILED"},
        ],
    }
    r, m = compute_rl_reward_from_z3_option_a(
        demo_z3,
        puzzle_accuracy=0.0,
        cell_accuracy=0.50,
        lam=0.10,
        gamma=2.0,
    )
    print("reward =", r)
    print("metrics =", m)
