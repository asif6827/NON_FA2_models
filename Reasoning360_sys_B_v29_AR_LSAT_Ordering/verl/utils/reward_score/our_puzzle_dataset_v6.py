# -*- coding: utf-8 -*-
"""
AR-LSAT ordering reward scoring.

This file intentionally keeps the public function name `compute_score` used by VERL,
but changes the expected model output from ZebraPuzzle table format to AR-LSAT
ordering format:

<answer>{
  "problem_type": "ordering",
  "world_model": {...},
  "rules": [...],
  "facts": [...],
  "question_semantics": {...},
  "options": {"A": "...", ...},
  "reasoning": [...],
  "solution": {"selected_option": "A"}
}</answer>
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

try:
    from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
except Exception:  # local testing fallback
    from z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload

try:
    from verl.utils.reward_score.check_interleved_format import check_interleaved_reasoning
except Exception:  # local testing fallback
    from check_interleved_format import check_interleaved_reasoning

job_id = os.getenv("SLURM_JOB_ID") or "local"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _try_parse_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    starts = [m.start() for m in re.finditer(r"\{", text)]
    for st in starts:
        for ed in range(len(text), st, -1):
            if text[ed - 1] != "}":
                continue
            chunk = text[st:ed]
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def find_last_answer_block(text: str) -> Optional[str]:
    pattern = re.compile(r"<answer\b[^>]*>.*?</answer\s*>", flags=re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(text or ""))
    if not matches:
        return None
    return matches[-1].group(0)


def _extract_ar_lsat_payload(solution_str: str) -> Tuple[Optional[Dict[str, Any]], str]:
    answer_block = find_last_answer_block(solution_str or "")
    if answer_block:
        # Strip tags so json.loads has a cleaner path, while fallback still works.
        inner = re.sub(r"^\s*<answer\b[^>]*>", "", answer_block, flags=re.IGNORECASE | re.DOTALL)
        inner = re.sub(r"</answer\s*>\s*$", "", inner, flags=re.IGNORECASE | re.DOTALL)
        obj = _try_parse_first_json_obj(inner)
        if isinstance(obj, dict):
            return obj, "success_answer_tag"
        return None, "answer_tag_json_error"

    obj = _try_parse_first_json_obj(solution_str or "")
    if isinstance(obj, dict):
        return obj, "success_direct_json"
    return None, "parsing_failed"


def _extract_gt_option(ground_truth: Any, extra_info: Any = None, meta: Any = None) -> Optional[str]:
    candidates = [ground_truth]
    if isinstance(extra_info, dict):
        candidates.extend([extra_info.get("answer"), extra_info.get("selected_option"), extra_info.get("ground_truth")])
    if isinstance(meta, dict):
        candidates.extend([meta.get("answer"), meta.get("selected_option"), meta.get("ground_truth")])

    for c in candidates:
        if isinstance(c, dict):
            c = c.get("answer") or c.get("selected_option") or c.get("ground_truth")
        if c is None:
            continue
        s = str(c).strip().upper()
        if re.fullmatch(r"[A-Z]", s):
            return s
    return None


def _selected_option(payload: Dict[str, Any]) -> Optional[str]:
    sol = payload.get("solution") or {}
    if not isinstance(sol, dict):
        return None
    selected = sol.get("selected_option")
    if selected is None:
        return None
    s = str(selected).strip().upper()
    return s if re.fullmatch(r"[A-Z]", s) else None


def _schema_score(payload: Dict[str, Any]) -> float:
    required = [
        "problem_type",
        "world_model",
        "rules",
        "facts",
        "question_semantics",
        "options",
        "reasoning",
        "solution",
    ]
    return sum(1 for k in required if k in payload) / len(required)


def _get_n_positions(payload: Dict[str, Any]) -> int:
    try:
        wm = payload.get("world_model") or {}
        domains = wm.get("domains") or {}
        positions = domains.get("positions") or []
        if positions:
            return max(int(str(p)) for p in positions)
        entities = wm.get("entities") or []
        return max(len(entities), 1)
    except Exception:
        return 1


def _safe_rate(num: Any, den: Any) -> float:
    try:
        den = max(float(den), 1.0)
        return max(0.0, min(float(num) / den, 1.0))
    except Exception:
        return 0.0


def compute_score(
    solution_str,
    ground_truth,
    extra_info: Any = None,
    score_method: str = "ar_lsat_ordering",
    timeout: float = 3.0,
    acc_weight: float = 0.60,
    clue_weight: float = 0.0,
    z3_weight: float = 0.10,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute reward for AR-LSAT ordering outputs.

    Main reward components:
      - selected option accuracy against ground_truth answer
      - valid <answer> JSON parsing
      - required schema coverage
      - interleaved reasoning format via check_interleaved_reasoning
      - Z3 semantic validation of rules/facts/options/reasoning
    """
    epoch = int(os.getenv("CURRENT_EPOCH", "0"))
    total_epochs = int(os.getenv("TOTAL_EPOCH", "1"))

    final_result: Dict[str, Any] = {
        "acc": 0.0,
        "score": 0.0,
        "reward_logged": 0.0,
        "AR_OPTION_ACCURACY": 0.0,
        "parsing_reward": 0.0,
        "schema_reward": 0.0,
        "format_reward": 0.0,
        "z3_reward": 0.0,
        "BASE_sat_full_GT": 0.0,
        "BASE_n_steps_total": 0.0,
        "BASE_n_steps_parsed_ok": 0.0,
        "BASE_n_steps_valid": 0.0,
        "BASE_n_steps_novel_inc_clues": 0.0,
        "BASE_n_non_valid_contradiction": 0.0,
        "selected_option": None,
        "ground_truth_option": None,
        "parse_status": "INIT",
        "z3_parse_status": "INIT",
        "epoch": epoch,
        "total_epochs": total_epochs,
    }

    payload: Dict[str, Any] = {}
    z3_out: Dict[str, Any] = {}

    try:
        parsed_payload, parse_status = _extract_ar_lsat_payload(solution_str)
        final_result["parse_status"] = parse_status
        if not isinstance(parsed_payload, dict):
            return final_result

        payload = parsed_payload
        parsing_reward = 1.0 if parse_status == "success_answer_tag" else 0.7
        schema_reward = _schema_score(payload)

        selected = _selected_option(payload)
        gt_option = _extract_gt_option(ground_truth, extra_info=extra_info, meta=meta)
        option_acc = 1.0 if selected and gt_option and selected == gt_option else 0.0

        reasoning = payload.get("reasoning")
        n_positions = _get_n_positions(payload)
        try:
            format_ok = check_interleaved_reasoning(reasoning, n_houses=int(n_positions))
        except Exception:
            format_ok = False
        format_reward = 1.0 if format_ok else 0.0

        payload_for_z3 = dict(payload)
        payload_for_z3["ground_truth"] = gt_option
        try:
            z3_out = solve_and_validate_payload(payload_for_z3, timeout_s=float(timeout), conflict_tolerant_clues=False)
        except Exception as e:
            z3_out = {"parse_status": "Z3_EXCEPTION", "error": f"{type(e).__name__}: {e}"}

        n_total = int(z3_out.get("n_steps_total", 0) or 0)
        n_parsed = int(z3_out.get("n_steps_parsed_ok", 0) or 0)
        n_valid = int(z3_out.get("n_steps_valid", 0) or 0)
        n_novel = int(z3_out.get("n_steps_novel_inc_clues", 0) or 0)
        n_contra = int(z3_out.get("n_non_valid_contradiction", 0) or 0)

        base_gt = float(z3_out.get("base_sat_full_GT", 0.0) or 0.0)
        selected_semantics_ok = float(z3_out.get("selected_option_semantics_ok", 0.0) or 0.0)
        parse_step_rate = _safe_rate(n_parsed, n_total)
        valid_step_rate = _safe_rate(n_valid, n_total)
        novel_step_score = min(float(n_novel) / 3.0, 1.0)
        contradiction_penalty = _safe_rate(n_contra, n_total)

        z3_reward_value = (
            0.40 * base_gt
            + 0.20 * selected_semantics_ok
            + 0.15 * parse_step_rate
            + 0.15 * valid_step_rate
            + 0.10 * novel_step_score
        )
        z3_reward_value = max(0.0, z3_reward_value - 0.25 * contradiction_penalty)

        # Conservative final reward. Correct answer is the anchor; process rewards are secondary.
        reward = (
            0.60 * option_acc
            + 0.15 * parsing_reward
            + 0.10 * schema_reward
            + 0.05 * format_reward
            + 0.10 * z3_reward_value
        )
        reward = max(0.0, min(float(reward), 1.0))

        final_result.update({
            "acc": reward,
            "score": reward,
            "reward_logged": reward,
            "AR_OPTION_ACCURACY": float(option_acc),
            "parsing_reward": float(parsing_reward),
            "schema_reward": float(schema_reward),
            "format_reward": float(format_reward),
            "z3_reward": float(z3_reward_value),
            "BASE_sat_full_GT": float(base_gt),
            "BASE_n_steps_total": float(n_total),
            "BASE_n_steps_parsed_ok": float(n_parsed),
            "BASE_n_steps_valid": float(n_valid),
            "BASE_n_steps_novel_inc_clues": float(n_novel),
            "BASE_n_non_valid_contradiction": float(n_contra),
            "selected_option": selected,
            "ground_truth_option": gt_option,
            "z3_parse_status": z3_out.get("parse_status"),
        })

        if os.environ.get("VALID_STATUS", "0") in ("1", "2"):
            feedback_root = os.path.join(os.environ.get("PUZZLE_FEEDBACK_PATH", "./"), f"jobid_{job_id}")
            feedback_path = os.path.join(feedback_root, f"ar_lsat_ordering_epoch_{epoch}_feedback.jsonl")
            _append_jsonl(feedback_path, {
                "payload": payload,
                "ground_truth": ground_truth,
                "z3_out": z3_out,
                "final_result": final_result,
                "solution_str": solution_str,
            })

        return final_result

    except Exception as e:
        logger.exception("Crash in AR-LSAT ordering compute_score")
        final_result["parse_status"] = "COMPUTE_SCORE_EXCEPTION"
        final_result["reward_error"] = f"{type(e).__name__}: {e}"
        return final_result


def pretty(x):
    print(json.dumps(x, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    def run_case(name: str, sol: str, gt: Any):
        print(f"\n=== {name} ===")
        out = compute_score(sol, gt)
        print(json.dumps(out, indent=2, ensure_ascii=False))

    # Test 1: correct could_be_true ordering answer.
    sol_correct = """<answer>{
      "problem_type": "ordering",
      "world_model": {
        "entities": ["A", "B", "C", "D"],
        "domains": {"positions": ["1", "2", "3", "4"]},
        "structural_assumptions": [
          "each speaker occupies exactly one position",
          "each position is occupied by exactly one speaker"
        ]
      },
      "rules": [
        "Distinct(A, B, C, D)",
        "A < B",
        "C == A + 1",
        "D != 1"
      ],
      "facts": ["B == 4"],
      "question_semantics": {
        "question_type": "could_be_true",
        "option_interpretation_rule": "choose the option whose formalization is satisfiable with rules and facts"
      },
      "options": {
        "A": "A == 2",
        "B": "C == 4",
        "C": "D == 2",
        "D": "A == 3",
        "E": "C == 1"
      },
      "reasoning": [
        "The question condition fixes B in the fourth position.",
        "S1: B == 4.",
        "Since C is immediately after A, C is exactly one position after A.",
        "S2: C == A + 1.",
        "Option A can be extended to a full valid ordering.",
        "S3: Sat(Option_A).",
        "Option B conflicts because C fourth would require A third while B is already fourth.",
        "S4: Unsat(Option_B)."
      ],
      "solution": {"selected_option": "A"}
    }</answer>"""

    # Test 2: wrong selected option.
    sol_wrong = sol_correct.replace('"selected_option": "A"', '"selected_option": "B"')

    # Test 3: must_be_true semantics.
    sol_must_true = """<answer>{
      "problem_type": "ordering",
      "world_model": {
        "entities": ["A", "B", "C"],
        "domains": {"positions": ["1", "2", "3"]},
        "structural_assumptions": [
          "each entity occupies exactly one position",
          "each position is occupied by exactly one entity"
        ]
      },
      "rules": ["Distinct(A, B, C)", "A < B", "B < C"],
      "facts": [],
      "question_semantics": {
        "question_type": "must_be_true",
        "option_interpretation_rule": "choose the option whose negation is unsatisfiable with rules and facts"
      },
      "options": {
        "A": "A == 1",
        "B": "B == 1",
        "C": "C == 1",
        "D": "A > C",
        "E": "C < B"
      },
      "reasoning": [
        "Since A is before B and B is before C, A must be before C.",
        "S1: A < C.",
        "The only possible ordering is A first, B second, and C third.",
        "S2: And(A == 1, B == 2, C == 3).",
        "Option A must be true because A is forced into position 1.",
        "S3: Sat(Option_A)."
      ],
      "solution": {"selected_option": "A"}
    }</answer>"""

    run_case("correct_could_be_true", sol_correct, {"answer": "A"})
    run_case("wrong_selected_option", sol_wrong, {"answer": "A"})
    run_case("must_be_true", sol_must_true, {"answer": "A"})
