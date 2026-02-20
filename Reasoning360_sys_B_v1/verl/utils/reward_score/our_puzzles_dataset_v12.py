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
import numpy as np
# from z3_verifier_v13 import compute_dsl_components
from typing import Dict, Any, List, Tuple, Optional
import re
from verl.utils.reward_score.z3_reasoning_validator_demo_v12_gt_solve import validate_reasoning_steps

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
        return (0.0, 0.0)

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


def _clamp01(x: Optional[float]) -> float:
    if x is None:
        return 0.0
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        # avoid int(True)=1 surprises unless you want it
        if isinstance(x, bool):
            return default
        return int(x)
    except Exception:
        return default


def _safe_bool(x: Any) -> Optional[bool]:
    if x is True:
        return True
    if x is False:
        return False
    return None


def compute_rl_reward_from_z3_v1(
        z3_out: Optional[Dict[str, Any]],
        puzzle_accuracy: Optional[float],
        cell_accuracy: Optional[float],
        *,
        # base shaping (your formula)
        base_cap: float = 0.95,
        base_a: float = 0.2,
        base_b: float = 0.75,

        # z3-based shaping cap
        step_cap: float = 0.95,

        # curriculum switch
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,  # kept for logging; not required
        switch_epoch: int = 25,
) -> Tuple[float, Dict[str, Any]]:
    """
    Policy:
      - If puzzle_accuracy == 1 -> reward = 1
      - Else if epoch >= switch_epoch -> reward = base(cell) ONLY
      - Else (early epochs):
          * If no steps passed -> reward = base(cell)
          * If some steps passed -> reward = min(step_cap, base(cell) * pass_rate_total)
            where pass_rate_total = n_good / max(1, n_steps_total)
    """

    p = _clamp01(puzzle_accuracy)
    c = _clamp01(cell_accuracy)

    # Always give full reward for perfect puzzle
    if p >= 1.0:
        return 1.0, {
            "reward_reason": "puzzle_accuracy=1",
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "pass_rate_total": 1.0,
        }

    base = min(base_cap, base_a + base_b * c)

    # Switch after epoch >= switch_epoch
    ep = _safe_int(epoch, -1)
    if ep >= switch_epoch:
        return float(base), {
            "reward_reason": f"epoch>={switch_epoch} => base_only",
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "base": base,
            "epoch": ep,
            "total_epochs": total_epochs,
            "pass_rate_total": 0.0,
        }

    # --- Early phase: use Z3 pass-rate against TOTAL steps ---
    if not isinstance(z3_out, dict):
        z3_out = {}

    steps = z3_out.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    n_total = _safe_int(z3_out.get("n_steps_total"), len(steps))
    n_parsed_ok = z3_out.get("n_steps_parsed_ok")
    n_good = z3_out.get("n_steps_entailed_chain")
    n_contra = z3_out.get("n_steps_contradiction_chain")

    # If missing, compute from steps list
    if n_good is None:
        good = 0
        for s in steps:
            if isinstance(s, dict) and s.get("chain_status") == "ENTAILED":
                good += 1
        n_good = good

    n_good = _safe_int(n_good, 0)

    # If no steps passed -> fallback to base
    if n_good <= 0:
        return float(base), {
            "reward_reason": "no_steps_passed => fallback_base",
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "base": base,
            "epoch": ep,
            "total_epochs": total_epochs,
            "n_steps_total": n_total,
            "n_steps_entailed_chain": n_good,
            "pass_rate_total": 0.0,
        }

    denom = max(1, n_parsed_ok)  # <-- CHANGE: denominator is TOTAL, not parsed_ok
    pass_rate = float(n_good) / float(denom)
    pass_rate = 0.0 if pass_rate < 0.0 else (1.0 if pass_rate > 1.0 else pass_rate)

    reward = min(step_cap, pass_rate)

    return float(reward), {
        "reward_reason": "steps_passed => base*pass_rate_total",
        "puzzle_accuracy": p,
        "cell_accuracy": c,
        "base": base,
        "step_cap": step_cap,
        "epoch": ep,
        "total_epochs": total_epochs,
        "pass_rate_total": pass_rate,
        "denom_total": denom,
        "n_steps_total": n_total,
        "n_steps_entailed_chain": n_good,
    }

def compute_rl_reward_from_z3(
    z3_out: Optional[Dict[str, Any]],
    puzzle_accuracy: Optional[float],
    cell_accuracy: Optional[float],
    *,
    # base shaping (your formula)
    base_cap: float = 0.95,
    base_a: float = 0.2,
    base_b: float = 0.75,

    # z3-based shaping cap
    step_cap: float = 0.95,

    # curriculum switch
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,  # logging only
    switch_epoch: int = 25,

    # scaling strength (optional)
    # parsed_scale_gamma>1 penalizes low parsed coverage more strongly
    parsed_scale_gamma: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Policy (updated):
      - If puzzle_accuracy == 1 -> reward = 1
      - Else if epoch >= switch_epoch -> reward = base(cell) ONLY
      - Else (early epochs):
          * If gt-validated strict pass rate is missing/0 -> reward = base(cell)
          * Else:
              pass_gt = pass_rate_strict_gt_validated  (0..1)
              parsed_scale = (n_steps_parsed_ok / max(1,n_steps_total)) ^ parsed_scale_gamma
              reward = min(step_cap, base(cell) * pass_gt * parsed_scale)
    """

    p = _clamp01(puzzle_accuracy)
    c = _clamp01(cell_accuracy)

    # 1) Perfect puzzle => full reward
    if p >= 1.0:
        return 1.0, {
            "reward_reason": "puzzle_accuracy=1",
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "pass_rate_strict_gt_validated": 1.0,
            "parsed_scale": 1.0,
        }

    # 2) Base shaping
    base = min(float(base_cap), float(base_a) + float(base_b) * c)

    '''
    # 3) Switch after epoch >= switch_epoch => base only
    ep = _safe_int(epoch, -1)
    if ep >= switch_epoch:
        return float(base), {
            "reward_reason": f"epoch>={switch_epoch} => base_only",
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "base": base,
            "epoch": ep,
            "total_epochs": total_epochs,
            "pass_rate_strict_gt_validated": 0.0,
            "parsed_scale": 0.0,
        }
    '''
    # 4) Early phase: use gt-validated strict pass rate scaled by parsed coverage
    if not isinstance(z3_out, dict):
        z3_out = {}

    pass_gt = z3_out.get("pass_rate_strict_gt_validated", None)
    try:
        pass_gt = float(pass_gt) if pass_gt is not None else 0.0
    except Exception:
        pass_gt = 0.0
    pass_gt = 0.0 if pass_gt < 0.0 else (1.0 if pass_gt > 1.0 else pass_gt)

    n_total = _safe_int(z3_out.get("n_steps_total"), 0)
    n_parsed_ok = _safe_int(z3_out.get("n_steps_parsed_ok"), 0)

    denom_total = max(1, n_total if n_total > 0 else 1)
    parsed_scale = float(n_parsed_ok) / float(denom_total)  # 0..1
    if parsed_scale < 0.0: parsed_scale = 0.0
    if parsed_scale > 1.0: parsed_scale = 1.0

    # optional stronger penalty if few steps are parsed
    if parsed_scale_gamma is not None and float(parsed_scale_gamma) != 1.0:
        parsed_scale = parsed_scale ** float(parsed_scale_gamma)

    # If no gt-validated strict progress -> fallback to base
    if pass_gt <= 0.0:
        return float(base), {
            "reward_reason": "pass_rate_strict_gt_validated<=0 => fallback_base",
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "base": base,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "pass_rate_strict_gt_validated": pass_gt,
            "n_steps_total": n_total,
            "n_steps_parsed_ok": n_parsed_ok,
            "parsed_scale": parsed_scale,
        }

    # 5) Main reward: base * (gt-validated strict pass rate) * (parsed coverage scale)
    raw_reward = float(base) * float(pass_gt) * float(parsed_scale)
    reward = min(float(step_cap), raw_reward)

    return float(reward), {
        "reward_reason": "early => base * pass_rate_strict_gt_validated * parsed_scale",
        "puzzle_accuracy": p,
        "cell_accuracy": c,
        "base": base,
        "step_cap": float(step_cap),
        "epoch": epoch,
        "total_epochs": total_epochs,
        "pass_rate_strict_gt_validated": pass_gt,
        "n_steps_total": n_total,
        "n_steps_parsed_ok": n_parsed_ok,
        "parsed_scale": parsed_scale,
        "raw_reward": raw_reward,
    }

def compute_rl_reward_from_z3_old(
        z3_out: Optional[Dict[str, Any]],
        puzzle_accuracy: Optional[float],
        cell_accuracy: Optional[float],
        *,
        base_cap: float = 0.95,
        base_a: float = 0.2,
        base_b: float = 0.75,
        step_cap: float = 0.95,
        denom_mode: str = "parsed_ok",  # "parsed_ok" | "total"
) -> Tuple[float, Dict[str, Any]]:
    p = _clamp01(puzzle_accuracy)
    c = _clamp01(cell_accuracy)

    if p >= 1.0:
        return 1.0, {"reward_reason": "puzzle_accuracy=1", "puzzle_accuracy": p, "cell_accuracy": c}

    base = min(base_cap, base_a + base_b * c)

    if not isinstance(z3_out, dict):
        z3_out = {}
    steps = z3_out.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    n_total = _safe_int(z3_out.get("n_steps_total"), len(steps))
    n_parsed_ok = z3_out.get("n_steps_parsed_ok")
    n_good = z3_out.get("n_steps_entailed_chain")
    n_contra = z3_out.get("n_steps_contradiction_chain")

    if n_parsed_ok is None or n_good is None or n_contra is None:
        parsed_ok = good = contra = 0
        for s in steps:
            if not isinstance(s, dict):
                continue
            if s.get("parsed_ok") is True:
                parsed_ok += 1
            if s.get("chain_status") == "ENTAILED":
                good += 1
            if s.get("chain_status") == "CONTRADICTION":
                contra += 1
        if n_parsed_ok is None: n_parsed_ok = parsed_ok
        if n_good is None: n_good = good
        if n_contra is None: n_contra = contra

    n_parsed_ok = _safe_int(n_parsed_ok, 0)
    n_good = _safe_int(n_good, 0)
    n_contra = _safe_int(n_contra, 0)

    # If no steps passed -> fallback to base
    if n_good <= 0:
        return float(base), {
            "reward_reason": "no_steps_passed => fallback_base",
            "puzzle_accuracy": p,
            "cell_accuracy": c,
            "base": base,
            "n_steps_total": n_total,
            "n_steps_parsed_ok": n_parsed_ok,
            "n_steps_entailed_chain": n_good,
            "n_steps_contradiction_chain": n_contra,
        }

    # Some steps passed -> proportional to BOTH pass_rate and cell score (base)
    denom = max(1, n_parsed_ok if denom_mode == "parsed_ok" else n_total)
    pass_rate = float(n_good) / float(denom)
    pass_rate = 0.0 if pass_rate < 0.0 else (1.0 if pass_rate > 1.0 else pass_rate)

    reward = min(step_cap, base * pass_rate)

    return float(reward), {
        "reward_reason": f"steps_passed => base*pass_rate (denom_mode={denom_mode})",
        "puzzle_accuracy": p,
        "cell_accuracy": c,
        "base": base,
        "step_cap": step_cap,
        "pass_rate": pass_rate,
        "denom": denom,
        "n_steps_total": n_total,
        "n_steps_parsed_ok": n_parsed_ok,
        "n_steps_entailed_chain": n_good,
        "n_steps_contradiction_chain": n_contra,
    }


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
    gt_header = [_norm_header(h) for h in (ground_truth.get("header") or [])]

    sol_rows_raw = (solution.get("rows") or [])
    gt_rows_raw = (ground_truth.get("rows") or [])

    sol_rows = [[_normalize_value(v) for v in r] for r in sol_rows_raw if isinstance(r, list)]
    gt_rows = [[_normalize_value(v) for v in r] for r in gt_rows_raw if isinstance(r, list)]

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
    gt_map = build_row_map(gt_rows, gt_col)

    keys = sorted(set(sol_map.keys()) & set(gt_map.keys()),
                  key=lambda x: (int(x) if x.isdigit() else 10 ** 9, x))
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
        end_markers: Optional[List[str]] = None, ) -> str:
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
            r"^\s*Solve the puzzle above\b",  # common instruction line
            r"^\s*returning\s+ONLY\s+the\s*<answer>",  # other common variant
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
    t = clamp01(epoch / max(1, total_epochs - 1))  # 0..1

    w_puz = t ** 1.5  # slow start, stronger later
    alpha = 0.5 + 3.0 * (t ** 2.0)  # Z3 penalty weak early, strong late
    cov_floor = 0.20 + 0.15 * t  # stricter later
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
            return parsed.get("parsed_clues", None), parsed.get("parsed_reasoning", None), parsed.get("solution", None), parsed.get("attribute_values", None), parsed.get("n_houses",
                                                                                                                                                                          None), "success_answer_tag"
        return None, None, None, None, None, "answer_tag_json_error"

    parsed = _try_parse_first_json_obj(solution_str)
    if parsed is not None:
        return parsed.get("parsed_clues", None), parsed.get("parsed_reasoning", None), parsed.get("solution", None), parsed.get("attribute_values", None), parsed.get("n_houses",
                                                                                                                                                                      None), "success_direct_json"

    return None, None, None, None, None, "parsing_failed"

def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x



def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json_str = json.dumps(record, default=to_jsonable, indent=2)
        f.write(json.dumps(json_str, ensure_ascii=False) + "\n")

def compute_score(
        solution_str,
        ground_truth,
        extra_info: Any = None,
        score_method: str = "gt",
        timeout: float = 3.0,
        acc_weight: float = 0.8,
        clue_weight: float = 1.0,
        z3_weight: float = 0.2,
        meta: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Scoring with:
    - ACC
    - Z3
    """

    import logging
    import time

    #print("\n" * 10)
    #print("In scoring script type of ground_truth:", type(ground_truth))
    #print("In Scoring script ground_truth:", (ground_truth))
    #print("\n"*10)

    start_t = time.monotonic()

    if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
        print(f"DEBUG-MODE: IN COMPUTE-SCORE, SCORING_METHOD = {score_method}")

    def time_left() -> float:
        return max(0.0, float(timeout) - (time.monotonic() - start_t))

    # puzzle_text = extract_puzzle_text(solution_str).strip()



    epoch = int(os.getenv("CURRENT_EPOCH", "90"))
    total_epochs = int(os.getenv("TOTAL_EPOCH", "100"))
    switch_epoch = int(os.environ.get("SWITCH_EPOCH", "25"))
    feedback_path = os.path.join(os.environ.get("PUZZLE_FEEDBACK_PATH", "./"), f"jobid_{job_id}")
    feedback_path = os.path.join(feedback_path, f"jobid_{job_id}_epoch_{str(epoch+1)}_feedback.jsonl")
    cell_acc_score = 0.0
    puzzle_acc_score = 0.0

    final_result = {}
    final_result["n_steps_total"] = 0.0
    final_result["n_steps_parsed_ok"] = 0.0
    final_result["base_sat_full"] = 0.0
    final_result["base_sat_raw"] = 0.0
    final_result["missed_data"] = 1.0
    final_result["n_steps_entailed_chain"] = 0.0
    final_result["n_steps_contradiction_chain"] = 0.0
    final_result["n_steps_entailed_strict"] = 0
    final_result["pass_rate_strict_gt_validated"] = 0.0
    final_result["pass_rate_chain_gt_validated"] = 0.0
    final_result["gt_valid"] = 0.0
    final_result["gt_factor"] = 0.0
    final_result["acc"] = 0.0
    final_result["PUZZLE_ACCURACY"] = 0.0
    final_result["CELL_ACCURACY"] = 0.0
    final_result["score"] = 0.0
    final_result["epoch"] = epoch
    final_result["total_epochs"] = total_epochs
    final_result["reward_logged"] = 0.0

    try:
        parsed_clues, parsed_reasoning, predicted_arrangement, attribute_values, n_houses, parse_status = extract_reasoning_and_solution(solution_str=solution_str)
        if parse_status != "success_answer_tag":
            if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
                log_case("non_boxed_answer", solution_str, ground_truth, logger)

        # meta selection
        meta_used = meta
        if meta_used is None and isinstance(extra_info, dict):
            meta_used = extra_info.get("meta") or extra_info
    except Exception as e:
        logger.error(f"Error in solution parsing: {e}")
        n_houses = None
        attribute_values = None
        parsed_clues = None
        parsed_reasoning = None

    # ---------------- ACC ----------------
    if predicted_arrangement:
        try:
            pred_conv = convert_numpy_arrays(predicted_arrangement)
            gt_conv = convert_numpy_arrays(ground_truth)

            norm_pred = normalize_grid(pred_conv)
            norm_gt = normalize_grid(gt_conv)

            if norm_pred and norm_gt:
                cell_acc_score, puzzle_acc_score = _compute_acc_from_normalized(norm_pred, norm_gt)
                # puzzle_acc_score, cell_acc_score = puzzle_and_cell_accuracy(norm_pred, norm_gt)
            else:
                cell_acc_score = 1.0 if pred_conv == gt_conv else 0.0
                puzzle_acc_score = 1.0 if pred_conv == gt_conv else 0.0
        except Exception as e:
            logger.error(f"Error calculating ACC score: {e}")
            cell_acc_score = 0.0
            puzzle_acc_score = 0.0

    # ---------------- Z3 ----------------
    z3_out = {
            "base_sat": None,  # kept for backward compatibility
            "base_sat_raw": None,
            "base_sat_full": None,
            "pass_rate_strict": 0.0,
            "pass_rate_chain": 0.0,
            "n_steps_entailed_strict": 0,
            "pass_rate_strict_gt_validated": 0.0,
            "pass_rate_chain_gt_validated": 0.0,
            "n_steps_total": 0,
            "n_steps_parsed_ok": 0,
            "n_steps_entailed_chain": 0,
            "n_steps_contradiction_chain": 0,
            "gt_valid": None,
            "gt_factor": 0.0,
        }
    if n_houses and attribute_values and parsed_clues and parsed_reasoning:
        try:
            z3_out = validate_reasoning_steps(
                n_houses=n_houses,
                attribute_values=attribute_values,
                parsed_clues=parsed_clues,
                parsed_reasoning=parsed_reasoning,
                ground_truth_solution=normalize_ground_truth(ground_truth),
                timeout_s=5.0,
                conflict_tolerant_clues=False,
            )

        except Exception as e:
            z3_out = {
                "base_sat": None,  # kept for backward compatibility
                "base_sat_raw": None,
                "base_sat_full": None,
                "pass_rate_strict": 0.0,
                "pass_rate_chain": 0.0,
                "n_steps_entailed_strict": 0,
                "pass_rate_strict_gt_validated": 0.0,
                "pass_rate_chain_gt_validated": 0.0,
                "n_steps_total": 0,
                "n_steps_parsed_ok": 0,
                "n_steps_entailed_chain": 0,
                "n_steps_contradiction_chain": 0,
                "gt_valid": None,
                "gt_factor":0.0,
            }
            logger.exception("Crash in Z3 Scoring")
            logger.error(f"Error calculating Z3 score: {e}")

        final_result["n_steps_total"] = z3_out["n_steps_total"]
        final_result["n_steps_parsed_ok"] = z3_out["n_steps_parsed_ok"]
        final_result["n_steps_entailed_chain"] = z3_out["n_steps_entailed_chain"]
        final_result["n_steps_contradiction_chain"] = z3_out["n_steps_contradiction_chain"]
        final_result["base_sat_full"] = 1.0 if z3_out["base_sat_full"] else 0.0
        final_result["base_sat_raw"] = 1.0 if z3_out["base_sat_raw"] else 0.0
        final_result["missed_data"] = 0.0
        final_result["n_steps_entailed_strict"] = z3_out["n_steps_entailed_strict"]
        final_result["pass_rate_strict_gt_validated"] = z3_out["pass_rate_strict_gt_validated"]
        final_result["pass_rate_chain_gt_validated"] = z3_out["pass_rate_chain_gt_validated"]
        final_result["gt_valid"] = 1.0 if z3_out["gt_valid"] else 0.0
        final_result["gt_factor"] = z3_out["gt_factor"]

    else:
        final_result["n_steps_total"] = 0.0
        final_result["n_steps_parsed_ok"] = 0.0
        final_result["base_sat_full"] = 0.0
        final_result["base_sat_raw"] = 0.0
        final_result["missed_data"] = 1.0
        final_result["n_steps_entailed_chain"] = 0.0
        final_result["n_steps_contradiction_chain"] = 0.0
        final_result["n_steps_entailed_strict"] = 0
        final_result["pass_rate_strict_gt_validated"] = 0.0
        final_result["pass_rate_chain_gt_validated"] = 0.0
        final_result["gt_valid"] = 0.0
        final_result["gt_factor"] = 0.0


    try:
        #reward, metrics = compute_rl_reward_from_z3(z3_out=z3_out, puzzle_accuracy=puzzle_acc_score, cell_accuracy=cell_acc_score)
        reward, reward_metrics = compute_rl_reward_from_z3(
            z3_out=z3_out,
            puzzle_accuracy=puzzle_acc_score,
            cell_accuracy=cell_acc_score,
            epoch=epoch,
            total_epochs=total_epochs,
            switch_epoch=switch_epoch,
        )
        reward = 1.0 if puzzle_acc_score == 1.0 else min(0.80, 0.2 + 0.75 * cell_acc_score)
        final_result["acc"] = reward
        final_result["PUZZLE_ACCURACY"] = puzzle_acc_score
        final_result["CELL_ACCURACY"] = cell_acc_score
        final_result["score"] = reward
        final_result["epoch"] = epoch
        final_result["total_epochs"] = total_epochs
        final_result["reward_logged"] = reward

    except Exception as e:
        final_result["acc"] = 0.0
        final_result["PUZZLE_ACCURACY"] = 0.0
        final_result["CELL_ACCURACY"] = 0.0
        final_result["score"] = 0.0
        final_result["epoch"] = epoch
        final_result["total_epochs"] = total_epochs
        final_result["reward_logged"] = 0.0

        logger.exception("Crash in Final Reward Scoring")  # includes line number + stack
        logger.error(f"Error calculating Z3 score: {e}")


    if os.environ.get("VALID_STATUS", "0") == "1":
        try:
            example_={}

            example_["solution_str"] = solution_str
            example_["ground_truth"] = ground_truth
            example_["z3_out"] = z3_out
            example_["reward_metrics"] = reward_metrics
            example_["final_result"] = final_result
            if example_:
                os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
                _append_jsonl(feedback_path, example_)

        except Exception as e:
            logger.exception("Crash in Writing Feedback")

    #os.environ["VALID_STATUS"] = "0"
    return final_result


def pretty(x):
    print(json.dumps(x, indent=2, ensure_ascii=False))



def normalize_ground_truth(ground_truth: dict) -> dict:
    """
    Converts numpy arrays inside ground_truth to plain Python lists.
    Expected input:
      {"header": np.ndarray, "rows": np.ndarray (possibly nested arrays)}
    """
    header = ground_truth.get("header", [])
    rows = ground_truth.get("rows", [])

    header_list = header.tolist() if isinstance(header, np.ndarray) else list(header)

    # rows can be an ndarray of ndarrays (dtype=object)
    if isinstance(rows, np.ndarray):
        rows_list = [r.tolist() if isinstance(r, np.ndarray) else list(r) for r in rows]
    else:
        rows_list = [r.tolist() if isinstance(r, np.ndarray) else list(r) for r in rows]

    return {"header": header_list, "rows": rows_list}





def make_solution_str() -> str:
    """
    This is the LLM output format your pipeline expects:
    a single <answer>...</answer> block with JSON inside.
    """
    payload = {
        "n_houses": 3,
        "attribute_values": {
            "Name": ["Arnold", "Peter", "Eric"],
            "Drink": ["tea", "water", "milk"]
        },
        "parsed_clues": [
            "C1 = set(2,Name,Peter).",
            "C2 = immediately_left_of(Name=Arnold,Drink=water).",
            "C3 = immediately_left_of(Drink=water,Drink=milk)."
        ],
        "parsed_reasoning": [
            "S1 [C1] set(2,Name,Peter).",
            "S2 [C3] not_set(3,Drink,water).",
            "S3 [C3] not_set(1,Drink,milk).",
            "S4 [C2] not_set(3,Name,Arnold).",
            "S5 [C2+C3] set(1,Name,Arnold)."
        ],
        "solution": {
            "header": ["House", "Name", "Drink"],
            "rows": [
                ["1", "Arnold", "tea"],
                ["2", "Peter", "water"],
                ["3", "Eric", "milk"]
            ]
        }
    }

    # IMPORTANT: solution_str must be wrapped in <answer> ... </answer>
    return "<answer>" + json.dumps(payload, ensure_ascii=False) + "</answer>"


def make_ground_truth() -> dict:
    """
    Your GT format: the same 'solution' table shape.
    """
    return {
        "header": ["House", "Name", "Drink"],
        "rows": [
            ["1", "Arnold", "tea"],
            ["2", "Peter", "water"],
            ["3", "Eric", "milk"]
        ],
    }


def main():
    import pprint
    solution_str = make_solution_str()
    ground_truth = make_ground_truth()
    ground_truth = json.dumps(ground_truth, ensure_ascii=False)
    #ground_truth = json.loads(ground_truth)

    #pred_conv = convert_numpy_arrays(predicted_arrangement)
    #gt_conv = convert_numpy_arrays(ground_truth)

    #norm_pred = normalize_grid(pred_conv)
    #norm_gt = normalize_grid(gt_conv)

    #ground_truth = convert_numpy_arrays(ground_truth)

    # 2) CALL compute_score
    # -----
    # IMPORTANT: adjust params to match YOUR compute_score signature.
    #
    # Common patterns I’ve seen in your code:
    #   compute_score(solution_str=..., ground_truth=..., timeout_s=..., conflict_tolerant_clues=...)
    #
    # If your signature is different, just rename/remove args accordingly.
    final_result = compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth)

    # 3) PRINT KEY FIELDS
    print("\n=== compute_score output (selected) ===")
    keys = [
        "CELL_ACCURACY",
        "PUZZLE_ACCURACY",
        "z3_base_sat_raw",
        "z3_base_sat_full",
        "pass_rate_strict_gt_validated",
        "pass_rate_chain_gt_validated",
        "n_steps_total",
        "n_steps_parsed_ok",
        "n_steps_entailed_strict",
        "n_steps_entailed_chain",
    ]
    for k in keys:
        if k in final_result:
            print(f"{k}: {final_result[k]}")

    print("\n=== Full final_result (pretty) ===")
    pprint.pprint(final_result)


if __name__ == "__main__":
    main()