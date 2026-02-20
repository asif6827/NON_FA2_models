# -*- coding: utf-8 -*-
"""
our_puzzles_dataset.py

Reward scoring for Zebra/Logic puzzles with triple scoring:
- ACC (grid match / cell-level)
- Z3 validity score
- Clue self-check score (LLM-as-verifier via Ray)
"""

import re
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys
#from z3_verifier_v13 import compute_dsl_components
from verl.utils.reward_score.z3_verifier_v15 import compute_z3_reward



os.environ.setdefault("CLUE_TIMEOUT_S", "3.0")
os.environ.setdefault("Z3_TIMEOUT_S", "1.5")
os.environ.setdefault("Z3_CLUE_GATE", "0.7")
os.environ.setdefault("CLUE_MAX_NEW_TOKENS", "256")
os.environ.setdefault("CLUE_MAX_INFLIGHT", "1")

job_id = os.getenv("SLURM_JOB_ID")




logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)


def parse_answer_tag(solution_str: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    answer_pattern = r'<answer>([\s\S]*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def log_case(case_type: str, solution_str: str, ground_truth: Any, logger):
    """Log special cases like non-boxed answers."""
    logger.info(f"{case_type} case:")
    logger.info(f"Solution_str: {solution_str}")
    logger.info(f"Ground_truth: {ground_truth}")


def _try_parse_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """
    Try parsing JSON object from text.
    - fast path: json.loads(text)
    - fallback: brute scan for the first valid {...} object
    """
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    starts = [m.start() for m in re.finditer(r'\{', text)]
    for st in starts:
        for ed in range(len(text), st, -1):
            if text[ed - 1] != '}':
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
    """
    Returns the *last* <answer>...</answer> block found in `text`,
    or None if no block exists.

    - Case-insensitive tags: <answer> or <ANSWER>
    - Allows attributes in the opening tag: <answer id="x">
    - Dot matches newlines so multi-line blocks work
    """
    pattern = re.compile(
        r"<answer\b[^>]*>.*?</answer\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        return None

    return matches[-1].group(0)

def extract_reasoning_and_solution(solution_str: str):
    """
    Extract both reasoning and solution.
    Returns (reasoning, solution, status)
    """
    answer_content = find_last_answer_block(solution_str)
    if answer_content:
        parsed = _try_parse_first_json_obj(answer_content)
        if parsed is not None:
            return parsed.get("parsed_clues", None), parsed.get("parsed_reasoning", None),  parsed.get("solution", None), parsed.get("attribute_values", None), parsed.get("n_houses", None), "success_answer_tag"
        return None, None, None, None, None, "answer_tag_json_error"

    parsed = _try_parse_first_json_obj(solution_str)
    if parsed is not None:
        return parsed.get("parsed_clues", None), parsed.get("parsed_reasoning", None),  parsed.get("solution", None), parsed.get("attribute_values", None), parsed.get("n_houses", None), "success_direct_json"

    return None, None, None, None, None, "parsing_failed"


# -------------------- normalization helpers --------------------

def convert_numpy_arrays(obj: Any) -> Any:
    """Convert numpy arrays nested in dict/list to python lists."""
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_arrays(x) for x in obj]
    return obj


def _normalize_atom(x: Any) -> str:
    return str(x).strip().lower()


def normalize_grid(data: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize grid for comparison:
    - lower/strip everything
    - ignore 'house'/'position' column if present
    - sort rows for permutation invariance
    """
    if not isinstance(data, dict):
        return None
    if "header" not in data or "rows" not in data:
        return None

    try:
        header = [_normalize_atom(h) for h in data["header"]]

        ignore_cols = {"house", "position"}
        keep_indices = [i for i, h in enumerate(header) if h not in ignore_cols]
        if not keep_indices:
            keep_indices = list(range(len(header)))

        rows_norm: List[List[str]] = []
        for row in data["rows"]:
            row_norm_full = [_normalize_atom(x) for x in row]
            row_norm = [row_norm_full[i] for i in keep_indices]
            rows_norm.append(row_norm)

        header_kept = [header[i] for i in keep_indices]
        return {"header": header_kept, "rows": sorted(rows_norm)}
    except Exception:
        return None

def normalize_strings(obj):
    """
    Recursively normalize:
      - strings -> lower-case + spaces -> underscores
      - list/tuple -> normalize each element (supports list of lists)
      - dict -> normalize values (keys unchanged; change if you want)
    """
    if isinstance(obj, str):
        return obj.strip().lower().replace(" ", "_")
    if isinstance(obj, (list, tuple)):
        return [normalize_strings(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize_strings(v) for k, v in obj.items()}
    return obj

def _compute_acc_from_normalized(norm_pred: Dict[str, Any], norm_gt: Dict[str, Any]) -> float:
    """Exact match => 1.0, otherwise cell-level acc if shapes align."""
    if not norm_pred or not norm_gt:
        return (0.0, 0.0)
    if norm_pred == norm_gt:
        return (1.0, 1.0)

    ph, pr = norm_pred.get("header", []), norm_pred.get("rows", [])
    gh, gr = norm_gt.get("header", []), norm_gt.get("rows", [])
    ph, pr = normalize_strings(ph), normalize_strings(pr)
    gh, gr = normalize_strings(gh), normalize_strings(gr)
    if not ph or not gh or not pr or not gr:
        return (0.0, 0.0)
    if ph != gh or len(pr) != len(gr):
        return (0.0,0.0)
    
    puzzle_accuracy = 0.0
    correct = 0
    total = 0
    for rp, rg in zip(pr, gr):
        if len(rp) != len(rg):
            return 0.0
        total += len(rp)
        correct += sum(1 for a, b in zip(rp, rg) if a == b)
    cell_accuracy = correct / total if total > 0 else 0.0
    if cell_accuracy < 1.0:
        puzzle_accuracy = 0.0
    else:
        puzzle_accuracy = 1.0
    return (cell_accuracy, puzzle_accuracy)


# -------------------- ray verifier singleton --------------------

_RAY_VERIFIER = None

def _get_ray_verifier(model_config: Dict[str, Any]):
    """Per-process singleton: avoid recreating verifier/actor for each sample."""
    global _RAY_VERIFIER
    if _RAY_VERIFIER is None:
        from verl.utils.reward_score.ray_clue_verifier import RayClueVerifier
        _RAY_VERIFIER = RayClueVerifier(model_config=model_config)
    return _RAY_VERIFIER


# -------------------- z3 timeout patch --------------------

_Z3_SOLVER_PATCHED = False
_Z3_LAST_TIMEOUT_MS = None

def _ensure_z3_timeout(timeout_s: float):
    """
    Monkey-patch verl.utils.reward_score.z3_verifier.Solver to set timeout for new solvers.
    """
    global _Z3_SOLVER_PATCHED, _Z3_LAST_TIMEOUT_MS
    try:
        import verl.utils.reward_score.z3_verifier as zv
    except Exception:
        return

    ms = int(max(0.0, float(timeout_s)) * 1000)
    if _Z3_SOLVER_PATCHED and _Z3_LAST_TIMEOUT_MS == ms:
        return

    orig_solver = getattr(zv, "Solver", None)
    if orig_solver is None:
        return

    def _solver_with_timeout(*args, **kwargs):
        s = orig_solver(*args, **kwargs)
        try:
            s.set("timeout", ms)
        except Exception:
            pass
        return s

    zv.Solver = _solver_with_timeout
    _Z3_SOLVER_PATCHED = True
    _Z3_LAST_TIMEOUT_MS = ms

from typing import Dict, Any, List, Tuple, Optional
import re

def puzzle_and_cell_accuracy(
    solution: Dict[str, Any],
    ground_truth: Dict[str, Any],
    *,
    custom_aliases: Optional[Dict[str, str]] = None,
) -> Tuple[float, float]:
    """
    Returns (puzzle_accuracy, cell_accuracy)

    Improvements vs basic version:
      - Strong value canonicalization to reduce false mismatches:
        * hyphens/model names: "Ford F-150" ~ "ford f150" ~ "fordf-150"
        * plural vs singular: "horses" ~ "horse"
        * phrasing: "keeps horses" ~ "horse" (best-effort)
        * devices: "iPhone 13" ~ "iphone13"
      - Aligns columns by header name (case-insensitive)
      - Aligns rows by "House" if present in both, else by row index
      - Compares only intersection of columns

    You can pass `custom_aliases` for dataset-specific equivalences.
    """

    # --- Normalization helpers ------------------------------------------------
    DEFAULT_ALIASES = {
        # nationality/common short forms
        "british": "brit",
        "danish": "dane",
        "german": "germany",  # example; adjust/remove if not needed
        # common device normalization examples
        "iphone": "iphone",
        "i phone": "iphone",
    }
    # user overrides / adds aliases
    alias_map = {**DEFAULT_ALIASES, **(custom_aliases or {})}

    STOPWORDS = {
        "the", "a", "an", "person", "who", "whose", "is", "in", "of", "to", "and",
        "somewhere", "directly", "immediately", "left", "right", "next", "between",
        "house", "loves", "likes", "keeps", "owns", "uses", "has", "drinks", "eats",
        "enjoys", "prefers", "wears", "with", "named", "name",
    }

    def _norm_header(x: Any) -> str:
        return str(x).strip().lower()

    def _singularize_token(tok: str) -> str:
        # very light singularization: horses->horse, cars->car; avoids "glass"->"glas"
        if len(tok) >= 4 and tok.endswith("s") and not tok.endswith("ss"):
            return tok[:-1]
        return tok

    def _apply_alias(s: str) -> str:
        # apply aliases to full string first (handles british->brit etc.)
        # but also allow token-level later
        s2 = s
        for k, v in alias_map.items():
            # whole-word replacement
            s2 = re.sub(rf"\b{re.escape(k)}\b", v, s2)
        return s2

    def _normalize_value(x: Any) -> str:
        """
        Canonicalize a cell value to a stable comparable key.
        Output is a compact alphanumeric string (no spaces/punct), e.g.:
          "Ford F-150" -> "fordf150"
          "iPhone 13"  -> "iphone13"
          "keeps horses" -> "horse" (best-effort)
        """
        s = str(x).strip().lower()
        s = s.strip("`'\"")
        s = _apply_alias(s)

        # Replace separators with spaces (hyphens, slashes, underscores, commas, etc.)
        s = re.sub(r"[\-_/.,:;()\[\]{}]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Tokenize on spaces
        toks = s.split(" ")
        # Remove stopwords + singularize remaining
        kept: List[str] = []
        for t in toks:
            if not t:
                continue
            if t in STOPWORDS:
                continue
            # token-level alias too (e.g., british -> brit)
            t = alias_map.get(t, t)
            t = _singularize_token(t)
            kept.append(t)

        if not kept:
            # fallback: keep alphanumerics from original string
            return re.sub(r"[^a-z0-9]+", "", s)

        # Special handling: if string looks like "iphone 13" etc., joining works already.
        # Return compact canonical key
        return "".join(re.sub(r"[^a-z0-9]+", "", t) for t in kept)

    # --- Load and normalize table structures ---------------------------------
    sol_header = [_norm_header(h) for h in (solution.get("header") or [])]
    gt_header  = [_norm_header(h) for h in (ground_truth.get("header") or [])]

    sol_rows_raw = (solution.get("rows") or [])
    gt_rows_raw  = (ground_truth.get("rows") or [])

    sol_rows = [[_normalize_value(v) for v in r] for r in sol_rows_raw if isinstance(r, list)]
    gt_rows  = [[_normalize_value(v) for v in r] for r in gt_rows_raw if isinstance(r, list)]

    if not sol_header or not gt_header or not sol_rows or not gt_rows:
        return 0.0, 0.0

    # Column intersection mapping
    gt_col = {name: j for j, name in enumerate(gt_header)}
    sol_col = {name: i for i, name in enumerate(sol_header)}
    common_cols = [c for c in gt_header if c in sol_col]
    if not common_cols:
        return 0.0, 0.0

    # Row alignment: by house if present in both; else by row index
    use_house = ("house" in sol_col) and ("house" in gt_col)

    def build_row_map(rows: List[List[str]], col_map: Dict[str, int]) -> Dict[str, List[str]]:
        m: Dict[str, List[str]] = {}
        if use_house:
            hi = col_map["house"]
            for idx, r in enumerate(rows):
                key = r[hi] if hi < len(r) else str(idx + 1)
                m[key] = r
        else:
            for idx, r in enumerate(rows):
                m[str(idx)] = r
        return m

    sol_map = build_row_map(sol_rows, sol_col)
    gt_map  = build_row_map(gt_rows, gt_col)

    keys = sorted(set(sol_map.keys()) & set(gt_map.keys()),
                  key=lambda x: (int(x) if x.isdigit() else 10**9, x))
    if not keys:
        return 0.0, 0.0

    total = 0
    correct = 0

    for k in keys:
        sr = sol_map[k]
        gr = gt_map[k]
        for c in common_cols:
            si = sol_col[c]
            gi = gt_col[c]
            total += 1
            if si >= len(sr) or gi >= len(gr):
                continue
            if sr[si] == gr[gi]:
                correct += 1

    cell_acc = correct / total if total > 0 else 0.0

    # Puzzle accuracy: strict match on compared region + same aligned key set sizes
    puzzle_acc = 1.0 if (correct == total and len(sol_map) == len(gt_map) == len(keys)) else 0.0
    return float(puzzle_acc), float(cell_acc)

def extract_puzzle_text(
    full_string: str,
    start_marker: str = "PUZZLE TO SOLVE",
    end_markers: Optional[List[str]] = None,) -> str:
    """
    Extract the puzzle statement + clues block from a larger string.

    Defaults:
    - Starts right after a line containing `start_marker`
    - Skips separator lines (-----) and blank lines
    - Stops before the first line that matches any end marker (e.g. "Solve the puzzle above")

    Raises:
        ValueError: if start_marker is not found.
    """
    if end_markers is None:
        end_markers = [
            r"^\s*Solve the puzzle above\b",                  # common instruction line
            r"^\s*returning\s+ONLY\s+the\s*<answer>",   # other common variant
        ]

    # Pre-compile patterns
    start_re = re.compile(rf"^\s*-*\s*{re.escape(start_marker)}\s*-*\s*$", re.IGNORECASE)
    end_res = [re.compile(pat, re.IGNORECASE) for pat in end_markers]
    sep_re = re.compile(r"^\s*-{3,}\s*$")  # lines of -----

    lines = full_string.splitlines(keepends=True)

    # 1) Find the start marker line
    start_idx = None
    for i, line in enumerate(lines):
        if start_re.match(line):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"start_marker not found: {start_marker!r}")

    # 2) Move to the first "content" line after the marker (skip separators + blanks)
    j = start_idx + 1
    while j < len(lines) and (lines[j].strip() == "" or sep_re.match(lines[j])):
        j += 1

    # 3) Find end (first matching end marker line)
    k = len(lines)
    for i in range(j, len(lines)):
        if any(er.match(lines[i]) for er in end_res):
            k = i
            break

    # 4) Return extracted block
    block = "".join(lines[j:k]).strip("\n")
    return block

def clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def logic_quality(z3_sat: float, clue_sat: float, parse_cov: float, cov_floor: float = 0.25) -> float:
    z3_sat = clamp01(z3_sat)
    clue_sat = clamp01(clue_sat)
    cov = clamp01(parse_cov)

    reliable = 1.0 if cov >= cov_floor else 0.0

    # When reliable: mix sat + clue_sat; when not reliable: don’t punish (quality=1)
    if reliable < 1.0:
        return 1.0

    q = 0.70 * z3_sat + 0.30 * clue_sat
    return clamp01(q)

def schedule(epoch: int, total_epochs: int):
    """
    Returns:
      w_puz: weight for puzzle_acc in GT anchor (0->1)
      alpha: penalty exponent (increases over time)
      cov_floor: min parse_cov to trust Z3
      parse_bonus_w: small bonus early to encourage parsable outputs
    """
    t = clamp01(epoch / max(1, total_epochs - 1))   # 0..1

    w_puz = t**1.5                 # slow start, stronger later
    alpha = 0.5 + 3.0 * (t**2.0)   # Z3 penalty weak early, strong late
    cov_floor = 0.20 + 0.15 * t    # stricter later
    parse_bonus_w = 0.05 * (1.0 - t)  # only early

    return w_puz, alpha, cov_floor, parse_bonus_w

def curriculum_with_z3(
    cell_acc: float,
    puzzle_acc: float,
    z3_sat: float,
    clue_sat: float,
    parse_cov: float,
    epoch: int,
    total_epochs: int,
    cell_beta: float = 2.0,
) -> float:
    c = clamp01(cell_acc)
    p = clamp01(puzzle_acc)

    if p >= 1.0:
        return 1.0
    if p <= 0.0 and c <= 0.0:
        return 0.0

    w_puz, alpha, cov_floor, parse_bonus_w = schedule(epoch, total_epochs)

    # GT anchor (early: shaped cell, late: puzzle)
    c_shaped = c ** max(1e-6, float(cell_beta))
    gt_anchor = clamp01((1.0 - w_puz) * c_shaped + w_puz * p)

    # Z3 logic quality (reliable only when parse_cov high)
    q_logic = logic_quality(z3_sat=z3_sat, clue_sat=clue_sat, parse_cov=parse_cov, cov_floor=cov_floor)

    # penalty ramps up with alpha over time
    reward = gt_anchor * (q_logic ** max(1e-6, float(alpha)))

    # small early parse bonus (prevents “ignore Z3 parser” collapse early)
    cov = clamp01(parse_cov)
    reward = clamp01(reward + parse_bonus_w * cov)

    return float(reward)



def compute_score(
    solution_str,
    ground_truth,
    extra_info: Any = None,
    score_method: str = "gt+z3",
    timeout: float = 3.0,
    acc_weight: float = 0.8,
    clue_weight: float = 1.0,
    z3_weight: float = 0.2,
    meta: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Triple scoring with:
    - ACC
    - Z3
    - Clue-check (gated by Z3)

    NOTE:
    - clue 超时/失败不会中断整个评分
    - clue 超时后会 cancel 对应 Ray 任务，防止“永远不停止/队列堆积”
    """

    import logging
    import time

    start_t = time.monotonic()

    if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
        print(f"DEBUG-MODE: IN COMPUTE-SCORE, SCORING_METHOD = {score_method}")

    def time_left() -> float:
        return max(0.0, float(timeout) - (time.monotonic() - start_t))

    #puzzle_text = extract_puzzle_text(solution_str).strip()

    epoch = int(os.getenv("CURRENT_EPOCH", "90"))
    total_epochs = int(os.getenv("TOTAL_EPOCH", "100"))

    cell_acc_score = 0.0
    puzzle_acc_score = 0.0
    z3_reward = 0.0
    clue_score = 0.0
    final_prompt = ""
    reward = 0.0
    z3_breakdown = {}


    score_method = str(os.environ.get("TRAIN_SCORE_METHOD", "gt+z3"))
    #print(f"Score method in Model Training: {score_method}")
    try:
        parsed_clues, parsed_reasoning, predicted_arrangement, attribute_values, n_houses, parse_status = extract_reasoning_and_solution(solution_str=solution_str)
        #print(f"Parsed  Clues: {parsed_clues}")
        #print(f"Parsed  Reasoning: {parsed_reasoning}")
        #print(f"Predicted Arrangement: {predicted_arrangement}")
        #print(f"Attribute Values: {attribute_values}")
        #print(f"N-Houses : {n_houses}")


        if parse_status != "success_answer_tag":
            if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
                log_case("non_boxed_answer", solution_str, ground_truth, logger)

        # normalize score_method
        sm = (score_method or "all").lower().strip()
        if sm == "all":
            methods = {"gt", "z3", "clue"}
        else:
            for sep in [",", " "]:
                sm = sm.replace(sep, "+")
            methods = {m for m in sm.split("+") if m}

        compute_acc = "gt" in methods
        compute_z3 = "z3" in methods
        compute_clue = "clue" in methods

        # meta selection
        meta_used = meta
        if meta_used is None and isinstance(extra_info, dict):
            meta_used = extra_info.get("meta") or extra_info

        # ---------------- ACC ----------------
        if compute_acc and predicted_arrangement is not None and isinstance(predicted_arrangement, dict) > 0:
            try:
                pred_conv = convert_numpy_arrays(predicted_arrangement)
                gt_conv = convert_numpy_arrays(ground_truth)

                norm_pred = normalize_grid(pred_conv)
                norm_gt = normalize_grid(gt_conv)

                if norm_pred and norm_gt:
                    cell_acc_score, puzzle_acc_score = _compute_acc_from_normalized(norm_pred, norm_gt)
                    #puzzle_acc_score, cell_acc_score = puzzle_and_cell_accuracy(norm_pred, norm_gt)
                else:
                    cell_acc_score = 1.0 if pred_conv == gt_conv else 0.0
                    puzzle_acc_score = 1.0 if pred_conv == gt_conv else 0.0
            except Exception as e:
                logger.error(f"Error calculating ACC score: {e}")
                cell_acc_score = 0.0
                puzzle_acc_score = 0.0

        # ---------------- Z3 (先算 Z3，作为 clue gate) ----------------
        if compute_z3 and parsed_reasoning and predicted_arrangement is not None and isinstance(predicted_arrangement, dict) > 0:
            try:
                pred_conv = convert_numpy_arrays(predicted_arrangement)

                z3_timeout_s = float(os.environ.get("Z3_TIMEOUT_S", "1.5"))
                z3_timeout_s = min(z3_timeout_s, time_left())

                _ensure_z3_timeout(z3_timeout_s)
                if isinstance(extra_info, dict) and "clues" in extra_info:
                    clues = extra_info.get("clues")

                # 注意：ACC 已经单独算过了，这里不需要 ground_truth 对比，减少开销

                #z3_result = compute_z3_components(reasoning, predicted_arrangement, clues, puzzle_text, gt_conv, debug=True)
                #z3_score = z3_result # float(z3_result.get("total_reward", 0.0))
                #z3_result = compute_dsl_components(parsed_clues_lines=parsed_clues,
                #                                  parsed_reasoning_lines=parsed_reasoning,
                #                                  predicted_solution=pred_conv,
                #                                  raw_clues_text=clues,
                #                                  puzzle_text=None,
                #                                  ground_truth=gt_conv,
                #                                  debug=False)  # float(z3_result.get("total_reward", 0.0))
                #print(f"z3_result = {z3_result}")
                z3_reward, z3_breakdown = compute_z3_reward(
                    parsed_clues=parsed_clues,
                    parsed_reasoning=parsed_reasoning,
                    solution=pred_conv,
                    attribute_values=attribute_values,
                    n_houses=n_houses,
                    puzzle_accuracy=puzzle_acc_score,
                    cell_accuracy=cell_acc_score,
                )


            except Exception as e:
                z3_reward = 0
                z3_breakdown = {}
                logger.exception("Crash in compute_score")  # includes line number + stack
                logger.error(f"Error calculating Z3 score: {e}")



        # ---------------- CLUE (Z3 gate + 独立超时 + cancel) ----------------

        if compute_clue and parsed_reasoning > 0:
            try:
                z3_gate = float(os.environ.get("Z3_CLUE_GATE", "0.7"))
                if z3_reward < z3_gate:
                    clue_score = 0.0
                    clues = []
                    parsed_reasoning = ""
                else:
                    clues: List[str] = []

                    if isinstance(extra_info, dict) and "clues" in extra_info:
                        clues = extra_info.get("clues")
                    if (not clues) and isinstance(meta_used, dict):
                        clues = meta_used.get("clues")

                    if clues is None:
                        clues = []


                    # numpy array => list，避免你之前的 ValueError: truth value is ambiguous
                    try:
                        import numpy as np
                        if isinstance(clues, np.ndarray):
                            clues = clues.tolist()
                    except Exception:
                        pass
                system_prompt = """You are an expert logic puzzle solver. I need you to verify if a given solution satisfies all the clues in a logic puzzle."""
                clues_text = "\n".join([f"{i + 1}. {clue}" for i, clue in enumerate(clues)])

                verification_prompt = f"""Problem ID: unknown

                CLUES: {clues_text}

                PROPOSED SOLUTION: {parsed_reasoning}

                Please check if the proposed solution satisfies ALL the clues. For each clue, first reason about whether it is satisfied or violated by the solution, and then state your final answer.

                Respond with a JSON object in the following format:
                {{
                  "clue_analysis": [
                    {{ "clue_number": 1, "reasoning": "work out if clue is satisfied", "satisfied": true }},
                    {{ "clue_number": 2, "reasoning": "work out if clue is satisfied", "satisfied": false }}
                  ],
                  "violated_clues": [1, 3],
                  "all_clues_satisfied": false
                }}
                """
                final_prompt = f"""<s>[INST] <<SYS>> {system_prompt} <</SYS>> {verification_prompt} [/INST]"""
            except Exception as e:
                logger.info(f"Z3-Result: {z3_reward}")
                logger.exception("Crash in compute_score")  # includes line number + stack
                logger.error(f"Error calculating Reasoning and Clues: {e}")

                final_prompt = ""
        else:
            final_prompt = ""


        # ---------------- weighted total in sequence gt+z3+clue----------------
        if score_method == "gt":
            weighted_total = cell_acc_score
        elif score_method == "z3":
            weighted_total = z3_reward
        else:
            #total_weight = float(acc_weight + z3_weight)
            #weighted_total = ((acc_score * acc_weight + z3_score * z3_weight) / total_weight) if total_weight > 0 else 0.0

            ACC_W = float(os.environ.get("ACC_W", "0.7"))
            Z3_W = float(os.environ.get("Z3_W", "0.3"))
            

            # z3_score should be in [0,1]
            # gate: when z3 fails, keep only a fraction of GT credit (prevents "GT-only hacks")
            #gate = 0.25 + 0.75 * z3_score  # z3=0 -> 0.25, z3=1 -> 1.0

            #weighted_total = ACC_W * (acc_score * gate) + Z3_W * z3_score
            #return float(weighted_total)
            #weighted_total = boundary_penalized_reward(acc_score, z3_result["z3_effective"], z3_result["reason_score"], acc_w=ACC_W, z3_w=Z3_W)
            #reward = boundary_penalized_reward(gt_score=acc_score, z3_sat=z3_result["z3_sat"], clue_sat=z3_result["clue_sat"],
            #                                   parse_cov=z3_result["parse_cov"], alpha=2.5, gamma=2.0,)
            #reward = curriculum_reward_cell_to_puzzle_only(
            #    cell_acc=acc_score,
            #    puzzle_acc=puzzle_acc_score,
            #    z3_sat=z3_result["z3_sat"],
            #    clue_sat=z3_result["clue_sat"],
            #    parse_cov=z3_result["parse_cov"],
            #    epoch=epoch,
            #    total_epochs=total_epochs,
            #)
            #reward = dapo_reward(puzzle_accuracy=puzzle_acc_score, cell_accuracy=cell_acc_score, z3_analysis=z3_result, epoch=epoch, total_epochs=total_epochs)

            #reward = curriculum_with_z3(cell_acc=cell_acc_score, puzzle_acc=puzzle_acc_score, z3_sat=z3_result["z3_sat"],
            #                            clue_sat=z3_result["clue_sat"],parse_cov=z3_result["parse_cov"],
            #                            epoch=epoch,total_epochs=total_epochs)
            #reward, reward_breakdown = curriculum_reward(z3_result, epoch=epoch, total_epochs=total_epochs)
            reward = z3_reward
            reward_breakdown = z3_breakdown
            #print(reward_breakdown)
            
    except Exception as e:
        logger.info(f"Z3-Result: {z3_reward}")
        logger.exception("Crash in compute_score")  # includes line number + stack
        logger.error(f"Error in compute_score in our_puzzles_dataset: {e}")
        reward = 0.0


        #print(f" Solution String : {solution_str}")
        #print(f" Ground Truth : {ground_truth}")
        #print(f" Extra Info : {extra_info}")
    #print(f"reward = {reward}, Breakdown = {reward_breakdown}")
    final_result = {"epoch": epoch, "total-epoch": total_epochs, "score": reward, "reward_logged": reward, "acc": cell_acc_score,
                    "PUZZLE_ACCURACY": puzzle_acc_score, "CELL_ACCURACY": cell_acc_score, "verification_prompt": final_prompt}

    if score_method == "gt":
        final_result = {}
        reward = 1.0 if puzzle_acc_score == 1.0 else min(0.95, 0.2 + 0.75 * cell_acc_score)
        final_result["acc"] = reward
        final_result["PUZZLE_ACCURACY"] = puzzle_acc_score
        final_result["CELL_ACCURACY"] = cell_acc_score
        final_result["score"] = reward
        final_result["verification_prompt"] = ""

    '''
    for key,val in z3_result.items():
        if key == "domains":
            continue
        elif key == "errors":
            continue
        elif isinstance(val, dict):
            final_result["z3_" + str(key)] = ", ".join([f"{k}={v}" for k, v in val.items()])
        else:
            final_result["z3_"+str(key)] = val

    for key,val in reward_breakdown.items():
        if isinstance(val, dict):
            final_result["rb_" + str(key)] = ", ".join([f"{k}={v}" for k, v in val.items()])
        else:
            final_result["rb_"+str(key)] = val
    '''
    #print(f"final_result: {final_result}")
    #reward_log_path = os.path.join((os.environ.get("REWARD_LOG_PATH", "./")), f"_jobid_{job_id}")

    #os.makedirs(reward_log_path, exist_ok=True)
    #filename = os.path.join(reward_log_path, "reward_log.jsonl")

    #print('Reward Log filename = {}'.format(filename))
    #with open(filename, "a") as f:
    #    f.write("\n".join(final_result) + "\n")

    #with open(filename, "a", encoding="utf-8") as f:
    #    json.dump(final_result, f, ensure_ascii=False, indent=2)
    #    f.write("\n\n")

    return final_result



def pretty(x):
    print(json.dumps(x, indent=2, ensure_ascii=False))


def main():

    llm_response = """    <answer>{
        "attribute_values": {
        "Name": ["Arnold", "Eric", "Peter"],
        "Drink": ["milk", "water", "tea"],
        "Hobby": ["photography", "cooking", "gardening"]
        },
        "n_houses":3,
        "parsed_clues" : [
            "C1 = set(2,Name,Peter).",
            "C2 = immediately_left_of(Name=Arnold,Drink=water).",
            "C3 = immediately_left_of(Drink=water,Drink=milk)."
        ],
        "parsed_reasoning" : [
            "S1 [C1] set(2,Name,Peter).",
            "S2 [C3] not(3,Drink,water).",
            "S3 [C3] not(1,Drink,milk).",
            "S4 [C2] not(3,Name,Arnold).",
            "S5 [C2+C3] set(1,Name,Arnold)."
        ],
        "solution": {
            "header": ["House", "name", "Drink", "Hobby"],
            "rows": [
                ["1", "Eric", "milk", "photography"],
                ["2", "Peter", "water", "cooking"],
                ["3", "Arnold", "tea", "gardening"]
            ]
        }
    }
    </answer>
    """

    clues = [
        "The Dane is somewhere to the left of the person who has black hair.",
        "The person who is a doctor is Eric.",
        "The person who is a pizza lover is in the second house.",
        "Arnold is directly left of the person who has a cat."
    ]

    ground_truth = {
        "header": ["House", "Name", "Drink", "Hobby"],
        "rows": [
            ["1", "Eric", "milk", "photography"],
            ["2", "Peter", "water", "cooking"],
            ["3", "Arnold", "tea", "gardening"]
        ]
    }


    # Extra info carries clues/meta for clue prompt generation
    extra_info = {"clues": clues, "meta": {"clues": clues}}



    # ----------------------------
    # 3) Combined (gt+z3+clue)
    # - clue part builds a prompt in `verification_`
    # - clue_score may remain 0.0 unless you add the actual clue verifier call later
    # ----------------------------
    os.environ["TRAIN_SCORE_METHOD"] = "gt+z3"
    print("\n=== TRAIN_SCORE_METHOD=gt+z3 ===")
    out_ = compute_score(llm_response, ground_truth, extra_info=extra_info, timeout=3.0)

    #out_bad  = compute_score(llm_bad,  ground_truth, extra_info=extra_info, timeout=3.0)
    print("Output:")
    pretty(out_)




if __name__ == "__main__":
    main()