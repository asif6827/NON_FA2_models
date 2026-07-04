# -*- coding: utf-8 -*-
"""Crash-safe reward scoring for AR-LSAT ASSIGNMENT outputs.

VERL-safe: compute_score always returns the same numeric keys.
Updated for AR-LSAT assignment runs where the model often emits:
  <answer>{ ... balanced JSON ... }
without a closing </answer> tag.
"""
from __future__ import annotations

import json, logging, os, re, sys
from typing import Any, Dict, Optional

try:
    from z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
except Exception:
    try:
        from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
    except Exception:
        def solve_and_validate_payload(payload, *, timeout_s=2.0, conflict_tolerant_clues=False):
            return {"parse_status": "Z3_IMPORT_FAIL", "base_sat_full_GT": False}

# Kept for backward compatibility, but this script now uses an AR-LSAT assignment
# specific reasoning/format checker instead of Zebra-style check_interleaved_reasoning.
try:
    from check_interleved_format import check_interleaved_reasoning
except Exception:
    try:
        from verl.utils.reward_score.check_interleved_format import check_interleaved_reasoning
    except Exception:
        def check_interleaved_reasoning(reasoning, *, n_houses=0):
            return False

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s", handlers=[logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

RESULT_KEYS = [
    "acc", "score", "reward_logged", "ACCURACY", "parsing_reward",
    # Answer-block / raw output diagnostics
    "starts_with_answer_open", "contains_answer_open", "contains_answer_close",
    "success_answer_open_json", "missing_answer_close", "contains_markdown_fence",
    "contains_latex_boxed", "contains_latex_begin", "no_latex_wrapper",
    # Schema diagnostics
    "schema_reward", "schema_partial_reward", "schema_required_keys_present",
    "schema_problem_type_ok", "schema_world_model_ok", "schema_entities_ok",
    "schema_domains_ok", "schema_solution_ok", "format_reward",
    # Formal field and raw reasoning diagnostics
    "formal_fields_reward", "raw_n_reasoning_items", "raw_n_s_steps_total",
    "raw_n_s_steps_parseable", "raw_n_non_option_s_steps", "raw_n_option_s_steps",
    "raw_s_step_parse_rate", "assignment_format_reward", "selected_option_test_ok",
    # Z3 / process diagnostics
    "z3_reward", "consistency_score", "Normalizer", "BASE_sat_full_GT", "missed_data",
    "BASE_n_steps_total", "BASE_n_steps_parsed_ok", "BASE_n_steps_valid", "BASE_n_steps_novel_inc_clues", "BASE_n_non_valid_contradiction",
    "novel_step_score", "contradiction_ratio", "selected_option_present", "ground_truth_present", "parse_status_ok", "schema_status_ok", "z3_status_ok", "format_status_ok",
    "z3_base_sat", "z3_solver_selected_ok", "z3_gt_match", "z3_rule_parse_error_count", "z3_fact_parse_error_count", "z3_option_parse_error_count", "z3_selected_option_parse_ok",
    "reward_exception", "parse_error_flag", "epoch", "total_epochs",
]


def _safe_epoch(name: str, default: int) -> float:
    try:
        return float(int(os.getenv(name, str(default))))
    except Exception:
        return float(default)


def _default_result(reward: float = -0.5, missed_data: float = 1.0) -> Dict[str, float]:
    out = {k: 0.0 for k in RESULT_KEYS}
    out["acc"] = out["reward_logged"] = float(reward)
    out["score"] = out["reward_logged"] = float(reward)
    out["missed_data"] = float(missed_data)
    out["Normalizer"] = 1.0
    out["epoch"] = _safe_epoch("CURRENT_EPOCH", 0)
    out["total_epochs"] = _safe_epoch("TOTAL_EPOCH", 1)
    return out


def _numeric_only(d: Dict[str, Any]) -> Dict[str, float]:
    out = _default_result()
    for k in RESULT_KEYS:
        if k not in d:
            continue
        v = d[k]
        try:
            if isinstance(v, bool):
                out[k] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                out[k] = float(v)
            else:
                out[k] = float(v)
        except Exception:
            out[k] = 0.0
    return out


def _clamp_reward(x: Any, lo=-1.0, hi=1.0) -> float:
    try:
        v = float(x)
    except Exception:
        return lo
    return max(lo, min(hi, v))


def _textify(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _norm_option_label(x: Any) -> Optional[str]:
    """Normalize selected/ground-truth/option labels to A-E."""
    if x is None:
        return None
    s = str(x).strip().upper()
    if not s:
        return None

    compact = s.replace("-", "_").replace(" ", "_").replace(":", "_")
    m = re.fullmatch(r"(?:SELECTED_)?(?:OPTION|ANSWER)?_?([A-E])", compact)
    if m:
        return m.group(1)

    m = re.search(r"(?:OPTION|ANSWER|CHOICE|SELECTED_OPTION)\s*[:_\-\s]*([A-E])\b", s)
    if m:
        return m.group(1)

    m = re.search(r"\b([A-E])\b\s*[\.)\]]?\s*$", s)
    if m:
        return m.group(1)

    letters = re.findall(r"\b([A-E])\b", s)
    if letters:
        return letters[-1]

    return None


def _raw_output_shape_stats(solution_str: Any) -> Dict[str, float]:
    raw = _textify(solution_str)
    stripped = raw.strip()
    lower = stripped.lower()
    contains_open = bool(re.search(r"<answer\b[^>]*>", stripped, flags=re.IGNORECASE))
    contains_close = "</answer" in lower
    return {
        "starts_with_answer_open": 1.0 if re.match(r"^\s*<answer\b[^>]*>", stripped, flags=re.IGNORECASE) else 0.0,
        "contains_answer_open": 1.0 if contains_open else 0.0,
        "contains_answer_close": 1.0 if contains_close else 0.0,
        "missing_answer_close": 1.0 if contains_open and not contains_close else 0.0,
        "contains_markdown_fence": 1.0 if "```" in stripped else 0.0,
        "contains_latex_boxed": 1.0 if "\\boxed" in stripped else 0.0,
        "contains_latex_begin": 1.0 if "\\begin" in stripped else 0.0,
        "no_latex_wrapper": 0.0 if ("\\boxed" in stripped or "\\begin" in stripped) else 1.0,
    }


def find_last_answer_block(text: Any) -> Optional[str]:
    raw = _textify(text)
    matches = list(re.finditer(r"<answer\b[^>]*>(.*?)</answer\s*>", raw, flags=re.IGNORECASE | re.DOTALL))
    return matches[-1].group(1).strip() if matches else None


def _extract_balanced_json_from(text: str, start_pos: int = 0) -> Optional[str]:
    """Extract the first balanced JSON object beginning at or after start_pos."""
    if not text:
        return None

    start = text.find("{", start_pos)
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def find_answer_open_json(text: Any) -> Optional[str]:
    """Find balanced JSON after opening <answer>, even if </answer> is missing."""
    raw = _textify(text)
    m = re.search(r"<answer\b[^>]*>", raw, flags=re.IGNORECASE)
    if not m:
        return None
    return _extract_balanced_json_from(raw, m.end())


def _strip_code_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _try_parse_first_json_obj(text: Any) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    raw = _strip_code_fence(_textify(text))
    if not raw:
        return None

    # Exact/full JSON parse first.
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Controlled fallback: parse first balanced JSON object in the provided text.
    # This is safe when the caller has already narrowed the text to an answer block
    # or to text after an opening <answer> tag.
    candidate = _extract_balanced_json_from(raw, 0)
    if candidate is not None:
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def parse_ar_lsat_answer(solution_str: Any):
    # 1. Best case: complete <answer>...</answer> block.
    block = find_last_answer_block(solution_str)
    if block is not None:
        parsed = _try_parse_first_json_obj(block)
        return (parsed, "success_answer_tag") if parsed is not None else (None, "answer_tag_json_error")

    # 2. Common recoverable case: <answer>{...balanced JSON...} without </answer>.
    open_json = find_answer_open_json(solution_str)
    if open_json is not None:
        parsed = _try_parse_first_json_obj(open_json)
        return (parsed, "success_answer_open_json") if parsed is not None else (None, "answer_open_json_error")

    # 3. Weak fallback only if the whole output starts directly with JSON.
    raw = _textify(solution_str).strip()
    if raw.startswith("{"):
        parsed = _try_parse_first_json_obj(raw)
        return (parsed, "success_direct_json") if parsed is not None else (None, "direct_json_error")

    return None, "parsing_failed"


def _normalize_assignment_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Inject pipeline-known metadata that the LLM should not be punished for."""
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    # Z3/validator can still receive problem_type, but the model no longer needs
    # to generate it correctly.
    normalized["problem_type"] = "assignment"
    return normalized


def _selected_from_ground_truth(gt: Any) -> Optional[str]:
    if isinstance(gt, str):
        return _norm_option_label(gt)
    if isinstance(gt, dict):
        for k in ("answer", "selected_option", "ground_truth_option"):
            if gt.get(k) is not None:
                return _norm_option_label(gt[k])
    return None


def _selected_from_prediction(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    sol = payload.get("solution") or {}
    if isinstance(sol, dict) and sol.get("selected_option") is not None:
        return _norm_option_label(sol["selected_option"])
    return None


def _flatten_scalars(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (int, float, bool)):
        return [str(x)]
    if isinstance(x, list):
        out: list[str] = []
        for item in x:
            out.extend(_flatten_scalars(item))
        return out
    if isinstance(x, dict):
        out: list[str] = []
        for item in x.values():
            out.extend(_flatten_scalars(item))
        return out
    return [str(x)]


def _unique_nonempty(xs: list[str]) -> list[str]:
    seen, out = set(), []
    for x in xs:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _assign_args_from_payload(payload: Optional[Dict[str, Any]], arg_index: int) -> list[str]:
    if not isinstance(payload, dict):
        return []

    chunks: list[str] = []
    for key in ("rules", "facts", "reasoning"):
        value = payload.get(key)
        if isinstance(value, list):
            chunks.extend(str(v) for v in value)

    options = payload.get("options")
    if isinstance(options, dict):
        chunks.extend(str(v) for v in options.values())
    elif isinstance(options, list):
        chunks.extend(str(v) for v in options)

    text = "\n".join(chunks)
    found: list[str] = []
    for m in re.finditer(r"Assign\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)", text):
        token = m.group(1 if arg_index == 0 else 2).strip()
        if token:
            found.append(token)
    return _unique_nonempty(found)


def _infer_n_values(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None

    wm = payload.get("world_model") or {}
    domains = wm.get("domains", {}) if isinstance(wm, dict) else {}
    raw_values: list[str] = []

    if isinstance(domains, dict):
        preferred_keys = (
            "values", "assignments", "projects", "colors", "rooms",
            "days", "slots", "tasks", "teams", "domains"
        )
        for key in preferred_keys:
            values = _flatten_scalars(domains.get(key))
            if values:
                raw_values = values
                break
        if not raw_values:
            raw_values = _flatten_scalars(domains)
    elif isinstance(domains, (list, str)):
        raw_values = _flatten_scalars(domains)

    values = _unique_nonempty(raw_values)
    if values:
        return len(values)

    values = _assign_args_from_payload(payload, arg_index=1)
    return len(values) if values else None


def _infer_n_entities(payload: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None

    wm = payload.get("world_model") or {}
    entities = wm.get("entities", []) if isinstance(wm, dict) else []
    entity_values = _unique_nonempty(_flatten_scalars(entities))
    if entity_values:
        return len(entity_values)

    entity_values = _assign_args_from_payload(payload, arg_index=0)
    return len(entity_values) if entity_values else None


def _schema_ok(payload: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(payload, dict):
        return False

    # problem_type is injected internally and is no longer required from the model.
    required = [
        "world_model", "rules", "facts",
        "question_semantics", "options", "reasoning", "solution",
    ]
    if any(k not in payload for k in required):
        return False

    world_model = payload.get("world_model")
    if not isinstance(world_model, dict):
        return False

    domains = world_model.get("domains")
    domains_ok = isinstance(domains, dict) and any(
        isinstance(v, list) and len(v) > 0 for v in domains.values()
    )

    solution = payload.get("solution")
    return (
        domains_ok
        and isinstance(world_model.get("entities", []), list)
        and isinstance(payload.get("rules"), list)
        and isinstance(payload.get("facts"), list)
        and isinstance(payload.get("question_semantics"), dict)
        and isinstance(payload.get("options"), dict)
        and isinstance(payload.get("reasoning"), list)
        and isinstance(solution, dict)
        and bool(solution.get("selected_option"))
    )


def _schema_partial_score(payload: Optional[Dict[str, Any]]) -> tuple[float, Dict[str, float]]:
    details = {
        "schema_required_keys_present": 0.0,
        "schema_problem_type_ok": 0.0,
        "schema_world_model_ok": 0.0,
        "schema_entities_ok": 0.0,
        "schema_domains_ok": 0.0,
        "schema_solution_ok": 0.0,
    }

    if not isinstance(payload, dict):
        return 0.0, details

    # problem_type is normalized internally, so this diagnostic means the final
    # payload sent to Z3 has the correct type; it is not a model-output gate.
    if str(payload.get("problem_type") or "").strip().lower() == "assignment":
        details["schema_problem_type_ok"] = 1.0

    required = [
        "world_model", "rules", "facts",
        "question_semantics", "options", "reasoning", "solution",
    ]
    required_ratio = sum(1 for k in required if k in payload) / len(required)
    details["schema_required_keys_present"] = required_ratio

    world_model = payload.get("world_model")
    if isinstance(world_model, dict):
        details["schema_world_model_ok"] = 1.0

        entities = world_model.get("entities")
        if isinstance(entities, list) and len(_unique_nonempty(_flatten_scalars(entities))) > 0:
            details["schema_entities_ok"] = 1.0

        domains = world_model.get("domains")
        if isinstance(domains, dict) and any(isinstance(v, list) and len(v) > 0 for v in domains.values()):
            details["schema_domains_ok"] = 1.0

    solution = payload.get("solution")
    if isinstance(solution, dict) and _norm_option_label(solution.get("selected_option")):
        details["schema_solution_ok"] = 1.0

    type_fields_ok = 0.0
    type_fields_ok += 1.0 if isinstance(payload.get("rules"), list) else 0.0
    type_fields_ok += 1.0 if isinstance(payload.get("facts"), list) else 0.0
    type_fields_ok += 1.0 if isinstance(payload.get("question_semantics"), dict) else 0.0
    type_fields_ok += 1.0 if isinstance(payload.get("options"), dict) else 0.0
    type_fields_ok += 1.0 if isinstance(payload.get("reasoning"), list) else 0.0
    type_fields_ratio = type_fields_ok / 5.0

    score = (
        0.25 * details["schema_required_keys_present"]
        + 0.10 * details["schema_problem_type_ok"]
        + 0.10 * details["schema_world_model_ok"]
        + 0.10 * details["schema_entities_ok"]
        + 0.15 * details["schema_domains_ok"]
        + 0.20 * type_fields_ratio
        + 0.10 * details["schema_solution_ok"]
    )
    return min(max(score, 0.0), 1.0), details


FORMAL_FIELD_RE = re.compile(
    r"(Assign\s*\(|Not\s*\(|And\s*\(|Or\s*\(|Implies\s*\(|Xor\s*\(|Exactly\s*\(|AtLeast\s*\(|AtMost\s*\(|Sat\s*\(|Unsat\s*\()"
)


def _formal_fields_score(payload: Optional[Dict[str, Any]]) -> float:
    if not isinstance(payload, dict):
        return 0.0

    items: list[str] = []
    for key in ("rules", "facts"):
        value = payload.get(key)
        if isinstance(value, list):
            items.extend(str(x) for x in value)

    options = payload.get("options")
    if isinstance(options, dict):
        items.extend(str(x) for x in options.values())

    if not items:
        return 0.0

    ok = sum(1 for x in items if FORMAL_FIELD_RE.search(x))
    return ok / max(len(items), 1)


FORMAL_STEP_RE = re.compile(
    r"""^S\d+:\s*(
        (Assign|Not|And|Or|Implies|Xor|Exactly|AtLeast|AtMost|Sat|Unsat)\s*\(.*\)
        |
        Assign\s*\(.*?\)\s*(==|!=)\s*Assign\s*\(.*?\)
    )\.$""",
    re.VERBOSE,
)

OPTION_STEP_RE = re.compile(r"^S\d+:\s*(Sat|Unsat)\s*\(.*Option_[A-E].*\)\.$")


def _assignment_reasoning_stats(reasoning: Any) -> Dict[str, float]:
    out = {
        "raw_n_reasoning_items": 0.0,
        "raw_n_s_steps_total": 0.0,
        "raw_n_s_steps_parseable": 0.0,
        "raw_n_non_option_s_steps": 0.0,
        "raw_n_option_s_steps": 0.0,
        "raw_s_step_parse_rate": 0.0,
        "assignment_format_reward": 0.0,
    }

    if not isinstance(reasoning, list):
        return out

    out["raw_n_reasoning_items"] = float(len(reasoning))
    s_steps: list[str] = []
    alternating_ok = 0
    expected_s = 1

    for i, item in enumerate(reasoning):
        if not isinstance(item, str):
            continue
        text = item.strip()

        if i % 2 == 0:
            if text.endswith(".") and not re.match(r"^S\d+:", text):
                alternating_ok += 1
        else:
            expected_prefix = f"S{expected_s}:"
            if text.startswith(expected_prefix) and text.endswith("."):
                alternating_ok += 1
            expected_s += 1

            if re.match(r"^S\d+:", text):
                s_steps.append(text)

    parseable = [s for s in s_steps if FORMAL_STEP_RE.match(s)]
    option_steps = [s for s in parseable if OPTION_STEP_RE.match(s)]
    non_option_steps = [s for s in parseable if not OPTION_STEP_RE.match(s)]

    out["raw_n_s_steps_total"] = float(len(s_steps))
    out["raw_n_s_steps_parseable"] = float(len(parseable))
    out["raw_n_option_s_steps"] = float(len(option_steps))
    out["raw_n_non_option_s_steps"] = float(len(non_option_steps))

    if s_steps:
        out["raw_s_step_parse_rate"] = len(parseable) / max(len(s_steps), 1)

    alternation_score = alternating_ok / max(len(reasoning), 1)
    min_step_score = min(len(s_steps) / 5.0, 1.0)
    non_option_score = min(len(non_option_steps) / 3.0, 1.0)
    option_score = 1.0 if len(option_steps) >= 1 else 0.0
    parse_score = out["raw_s_step_parse_rate"]

    out["assignment_format_reward"] = (
        0.30 * alternation_score
        + 0.25 * min_step_score
        + 0.20 * non_option_score
        + 0.15 * option_score
        + 0.10 * parse_score
    )
    return out


def _question_type_from_payload(payload: Optional[Dict[str, Any]], extra_info: Any = None) -> str:
    if isinstance(extra_info, dict) and extra_info.get("question_type"):
        return str(extra_info["question_type"]).strip().lower()
    if isinstance(payload, dict):
        qs = payload.get("question_semantics")
        if isinstance(qs, dict) and qs.get("question_type"):
            return str(qs["question_type"]).strip().lower()
    return ""


def _selected_option_test_ok(payload: Optional[Dict[str, Any]], extra_info: Any = None) -> float:
    selected = _selected_from_prediction(payload)
    if not selected or not isinstance(payload, dict):
        return 0.0

    qtype = _question_type_from_payload(payload, extra_info)
    reasoning = payload.get("reasoning")
    if not isinstance(reasoning, list):
        return 0.0

    option = f"Option_{selected}"
    text = "\n".join(str(x) for x in reasoning)
    expected = {
        "could_be_true": f"Sat({option})",
        "must_be_true": f"Unsat(Not({option}))",
        "cannot_be_true": f"Unsat({option})",
        "could_be_false": f"Sat(Not({option}))",
        "acceptability": f"Sat({option})",
    }.get(qtype)

    if not expected:
        return 0.0
    return 1.0 if expected in text else 0.0


def compute_score(solution_str, ground_truth, extra_info: Any = None, score_method: str = "gt", timeout: float = 3.0, acc_weight: float = 0.8, clue_weight: float = 1.0, z3_weight: float = 0.2, meta: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    try:
        out: Dict[str, Any] = _default_result(reward=0.0, missed_data=0.0)
        out.update(_raw_output_shape_stats(solution_str))

        payload, parse_status = parse_ar_lsat_answer(solution_str)
        payload = _normalize_assignment_payload(payload)

        parsing_reward = (
            1.0 if parse_status == "success_answer_tag"
            else 0.75 if parse_status == "success_answer_open_json"
            else 0.15 if parse_status == "success_direct_json"
            else 0.0
        )
        out["parsing_reward"] = parsing_reward
        out["parse_status_ok"] = 1.0 if parse_status in {"success_answer_tag", "success_answer_open_json"} else 0.0
        out["parse_error_flag"] = 0.0 if parse_status in {"success_answer_tag", "success_answer_open_json", "success_direct_json"} else 1.0
        out["success_answer_open_json"] = 1.0 if parse_status == "success_answer_open_json" else 0.0

        selected = _selected_from_prediction(payload)
        gt = _selected_from_ground_truth(ground_truth)
        out["selected_option_present"] = 1.0 if selected else 0.0
        out["ground_truth_present"] = 1.0 if gt else 0.0
        accuracy = 1.0 if selected and gt and selected == gt else 0.0
        out["ACCURACY"] = accuracy

        # If nothing parseable was produced, keep reward near zero and do not
        # reward superficial text shape.
        if payload is None:
            raw = _textify(solution_str)
            reward = -0.05
            if out["contains_answer_open"] > 0.0:
                reward += 0.02
            if "\\boxed" in raw or "\\begin" in raw:
                reward -= 0.05
            reward = max(-0.10, min(reward, 0.02))
            out["acc"] = _clamp_reward(reward)
            out["score"] = out["reward_logged"] = _clamp_reward(reward)
            out["missed_data"] = 1.0
            return _numeric_only(out)

        schema_reward = 1.0 if _schema_ok(payload) else 0.0
        schema_partial_reward, schema_details = _schema_partial_score(payload)
        out["schema_reward"] = out["schema_status_ok"] = schema_reward
        out["schema_partial_reward"] = schema_partial_reward
        out.update(schema_details)

        reasoning = payload.get("reasoning") if isinstance(payload, dict) else None
        reasoning_stats = _assignment_reasoning_stats(reasoning)
        out.update(reasoning_stats)
        out["format_reward"] = out["format_status_ok"] = out["assignment_format_reward"]
        out["formal_fields_reward"] = _formal_fields_score(payload)
        out["selected_option_test_ok"] = _selected_option_test_ok(payload, extra_info)

        n_values = _infer_n_values(payload)
        n_entities = _infer_n_entities(payload)

        z3_out: Dict[str, Any] = {}
        if isinstance(payload, dict) and schema_reward > 0.0:
            z3_payload = dict(payload)
            z3_payload["problem_type"] = "assignment"
            z3_payload["ground_truth"] = ground_truth
            if isinstance(extra_info, dict) and extra_info.get("question_type"):
                z3_payload["question_type"] = extra_info["question_type"]
            try:
                z3_out = solve_and_validate_payload(z3_payload, timeout_s=timeout, conflict_tolerant_clues=False)
            except Exception as e:
                z3_out = {"parse_status": "Z3_EXCEPTION", "error": f"{type(e).__name__}: {e}"}

        z3_status = str(z3_out.get("parse_status", ""))
        out["z3_status_ok"] = 1.0 if z3_status.endswith("SUCCESS") else 0.0
        out["z3_base_sat"] = 1.0 if bool(z3_out.get("base_sat", False)) else 0.0
        out["z3_solver_selected_ok"] = 1.0 if bool(z3_out.get("solver_selected_ok", False)) else 0.0
        out["z3_gt_match"] = 1.0 if bool(z3_out.get("gt_match", False)) else 0.0
        out["z3_rule_parse_error_count"] = float(z3_out.get("n_rule_parse_errors", 0) or 0)
        out["z3_fact_parse_error_count"] = float(z3_out.get("n_fact_parse_errors", 0) or 0)
        out["z3_option_parse_error_count"] = float(z3_out.get("n_option_parse_errors", 0) or 0)
        out["z3_selected_option_parse_ok"] = 1.0 if bool(z3_out.get("selected_option_parse_ok", False)) else 0.0
        sat_ok = 1.0 if bool(z3_out.get("base_sat_full_GT", False)) else 0.0
        out["z3_reward"] = out["BASE_sat_full_GT"] = sat_ok
        out["consistency_score"] = float(z3_out.get("consistency_score", 0.0) or 0.0)
        out["BASE_n_steps_total"] = float(z3_out.get("n_steps_total", 0) or 0)
        out["BASE_n_steps_parsed_ok"] = float(z3_out.get("n_steps_parsed_ok", 0) or 0)
        out["BASE_n_steps_valid"] = float(z3_out.get("n_steps_valid", 0) or 0)
        out["BASE_n_steps_novel_inc_clues"] = float(z3_out.get("n_steps_novel_inc_clues", 0) or 0)
        out["BASE_n_non_valid_contradiction"] = float(z3_out.get("n_non_valid_contradiction", 0) or 0)

        reward, normalizer = 0.0, 1.0
        if n_values is not None and n_entities is not None:
            normalizer = max(2.0 * max(int(n_values) * int(n_entities), 1), 1.0)
        else:
            out["missed_data"] = 1.0

        n_novel = out["BASE_n_steps_novel_inc_clues"]
        n_contra = out["BASE_n_non_valid_contradiction"]
        novel_step_score = min(n_novel / normalizer, 1.0)
        contradiction_ratio = min(n_contra / normalizer, 1.0)

        base_quality = (0.60 * accuracy)

        reward = base_quality

        out["novel_step_score"] = novel_step_score
        out["contradiction_ratio"] = contradiction_ratio
        out["Normalizer"] = normalizer
        out["acc"] = _clamp_reward(reward)
        out["score"] = out["reward_logged"] = _clamp_reward(reward)
        return _numeric_only(out)
    except Exception:
        logger.exception("assignment compute_score failed; returning complete penalty reward dict")
        out = _default_result(reward=-0.5, missed_data=1.0)
        out["reward_exception"] = 1.0
        return out


def _wrap(payload: Dict[str, Any], close: bool = True) -> str:
    s = "<answer>" + json.dumps(payload, ensure_ascii=False, indent=2)
    return s + "</answer>" if close else s


def _make_answer(selected: str = "A", bad_format: bool = False, close: bool = True) -> str:
    reasoning = [
        "A is not assigned to P1 by the first rule.",
        "S1: Not(Assign(A, P1)).",
        "Exactly one employee is assigned to P2.",
        "S2: Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2)).",
        "Option A can be extended to a full valid assignment.",
        "S3: Sat(Option_A).",
    ]
    if bad_format:
        reasoning = ["S1: Not(Assign(A, P1)).", "This starts with a formal step, so format should fail."]
    payload = {
        "world_model": {"entities": ["A", "B", "C"], "domains": {"values": ["P1", "P2", "P3"]}, "structural_assumptions": ["each entity is assigned exactly one value"]},
        "rules": ["Not(Assign(A, P1))", "Assign(B, P1) == Assign(C, P1)", "Exactly(1, Assign(A, P2), Assign(B, P2), Assign(C, P2))"],
        "facts": [],
        "question_semantics": {"question_type": "could_be_true"},
        "options": {"Option_A": "Assign(A, P2)", "B": "Assign(A, P1)", "C": "Assign(B, P2)"},
        "reasoning": reasoning,
        "solution": {"selected_option": selected},
    }
    return _wrap(payload, close=close)


def _make_must_be_true_answer() -> str:
    payload = {
        "world_model": {"entities": ["A", "B"], "domains": {"values": ["P1", "P2"]}, "structural_assumptions": ["each entity is assigned exactly one value"]},
        "rules": ["Assign(A, P1)"],
        "facts": [],
        "question_semantics": {"question_type": "must_be_true"},
        "options": {"A": "Assign(A, P1)", "B": "Assign(A, P2)"},
        "reasoning": ["The passage directly fixes A to P1.", "S1: Assign(A, P1).", "Option A is forced by all valid assignments.", "S2: Unsat(Not(Option_A))."],
        "solution": {"selected_option": "Option_A"},
    }
    return _wrap(payload)


if __name__ == "__main__":
    malformed_domains_payload = {
        "world_model": {
            "entities": ["A", "B", "C"],
            "domains": ["P1", "P2", "P3"],
            "structural_assumptions": ["each entity is assigned exactly one value"],
        },
        "rules": ["Not(Assign(A, P1))"],
        "facts": [],
        "question_semantics": {"question_type": "could_be_true"},
        "options": {"A": "Assign(A, P2)", "B": "Assign(A, P1)"},
        "reasoning": [
            "A is not assigned to P1 by the first rule.",
            "S1: Not(Assign(A, P1)).",
            "Option A can be extended to a full valid assignment.",
            "S2: Sat(Option_A).",
        ],
        "solution": {"selected_option": "A"},
    }
    tests = [
        ("correct_could_be_true", _make_answer("Option_A"), "A"),
        ("open_answer_json_no_close", _make_answer("A", close=False), "A"),
        ("wrong_selected_option", _make_answer("B"), "A"),
        ("bad_format_correct_answer", _make_answer("A", bad_format=True), "A"),
        ("must_be_true", _make_must_be_true_answer(), "A"),
        ("malformed_domains_list_no_crash", _wrap(malformed_domains_payload), "A"),
        ("malformed_json", "<answer>{bad json</answer>", "A"),
        ("none_output", None, "A"),
    ]
    for name, pred, gt in tests:
        print(f"\n=== {name} ===")
        result = compute_score(pred, gt)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        assert set(result.keys()) == set(RESULT_KEYS)
        assert all(isinstance(v, float) for v in result.values())
