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


from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import normalize_header, normalize_months_in_rows

from verl.utils.reward_score.check_interleved_format import check_interleaved_reasoning
from verl.utils.reward_score.z3_reasoning_vs_solution_verifier_v2 import verify_solution_two_step


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


#pid_to_puzzle_dic_file = '/export/home/asifali/HF_cache/ZebraLogic/pid_to_puzzle_dic.json'
pid_to_puzzle_dic_file = os.environ.get("PUZZLE_DIC_PATH", "/home/asif/data3/HF_cache/ZebraLogic/pid_to_puzzle_dic.json")

with open(pid_to_puzzle_dic_file, "r", encoding="utf-8") as f:
    pid_to_puzzle_dic = json.load(f)   # this is a dict (if JSON root is an object)



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
            return parsed.get("syntactic_clues", None), parsed.get("reasoning", None), parsed.get("solution", None), parsed.get("attribute_values", None), parsed.get("n_houses", None), "success_answer_tag"
        return None, None, None, None, None, "answer_tag_json_error"

    parsed = _try_parse_first_json_obj(solution_str)
    if parsed is not None:
        return parsed.get("syntactic_clues", None), parsed.get("reasoning", None), parsed.get("solution", None), parsed.get("attribute_values", None), parsed.get("n_houses", None), "success_direct_json"

    return None, None, None, None, None, "parsing_failed"

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



def append_jsonl_pretty(filepath, record):
    """
    Appends ONE JSON object to a .jsonl file.
    The object is pretty-formatted but written as a single line
    (JSONL compliant).
    """
    # Pretty format
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    pretty = json.dumps(record, indent=2, ensure_ascii=False)

    # Collapse into a single line while keeping spacing readable
    #one_line = "".join(pretty.splitlines())

    with open(filepath, "a", encoding="utf-8") as f:
        f.write(pretty + "\n")



def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

from typing import Any, Dict, List, Optional, Tuple

def _to_py_list(x: Any):
    # Convert numpy arrays (and similar) to python lists safely
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x

def normalize_table(t: Any) -> Optional[Dict[str, Any]]:
    """
    Ensures:
      - dict with keys: header (list[str]), rows (list[list[str]])
      - rows never contain np.ndarray elements
      - avoids boolean checks on arrays
    """
    if t is None or not isinstance(t, dict):
        return None

    header = _to_py_list(t.get("header", []))
    rows = _to_py_list(t.get("rows", []))

    # Important: DO NOT do `if rows:` before this normalization.
    if header is None:
        header = []
    if rows is None:
        rows = []

    # Make sure header is a list
    if not isinstance(header, list):
        header = _to_py_list(header)
        if not isinstance(header, list):
            header = []

    # Make sure rows is a list of lists
    if not isinstance(rows, list):
        rows = _to_py_list(rows)
        if not isinstance(rows, list):
            rows = []

    norm_rows: List[List[str]] = []
    for r in rows:
        r = _to_py_list(r)
        if isinstance(r, tuple):
            r = list(r)
        if not isinstance(r, list):
            # last resort: wrap scalar as single-cell row
            r = [r]
        # stringify cells to make comparisons stable
        norm_rows.append([str(c) for c in r])

    return {
        "header": [str(h) for h in header],
        "rows": norm_rows,
    }

def _extract_ar_lsat_payload(solution_str: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract AR-LSAT JSON payload from <answer>...</answer> or direct JSON.
    """
    answer_block = find_last_answer_block(solution_str or "")
    if answer_block:
        obj = _try_parse_first_json_obj(answer_block)
        if isinstance(obj, dict):
            return obj, "success_answer_tag"
        return None, "answer_tag_json_error"

    obj = _try_parse_first_json_obj(solution_str or "")
    if isinstance(obj, dict):
        return obj, "success_direct_json"

    return None, "parsing_failed"


def _ar_selected_option_from_gt(ground_truth: Any, extra_info: Any = None, meta: Any = None) -> Optional[str]:
    """
    Resolve the correct AR-LSAT option label from ground_truth / extra_info / meta.
    Accepts:
      - "A"
      - {"answer": "A"}
      - {"selected_option": "A"}
      - extra_info/meta with answer.
    """
    candidates = [ground_truth]
    if isinstance(extra_info, dict):
        candidates.extend([extra_info.get("answer"), extra_info.get("selected_option")])
    if isinstance(meta, dict):
        candidates.extend([meta.get("answer"), meta.get("selected_option")])

    for c in candidates:
        if isinstance(c, dict):
            c = c.get("answer") or c.get("selected_option")
        if c is None:
            continue
        s = str(c).strip().upper()
        if re.fullmatch(r"[A-Z]", s):
            return s
    return None


def _ar_check_interleaved_reasoning(reasoning: Any) -> bool:
    """
    AR-LSAT ordering format check:
      [NL, S1, NL, S2, ...]
    Natural lines must not start with S<number>:.
    Formal lines must start with S<number>: and be sequential.
    """
    if not isinstance(reasoning, list) or len(reasoning) < 2:
        return False
    expected_s = 1
    for i, line in enumerate(reasoning):
        if not isinstance(line, str) or not line.strip().endswith("."):
            return False
        is_formal_position = (i % 2 == 1)
        has_s_prefix = re.match(r"^\s*S(\d+)\s*:", line.strip(), re.IGNORECASE)
        if is_formal_position:
            if not has_s_prefix:
                return False
            sid = int(has_s_prefix.group(1))
            if sid != expected_s:
                return False
            expected_s += 1
        else:
            if has_s_prefix:
                return False
    return True


def _ar_rule_field_score(payload: Dict[str, Any]) -> float:
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
    ok = 0
    for k in required:
        if k in payload:
            ok += 1
    return ok / len(required)


def compute_score(
        solution_str,
        ground_truth,
        extra_info: Any = None,
        score_method: str = "ar_lsat_ordering",
        timeout: float = 3.0,
        acc_weight: float = 0.6,
        clue_weight: float = 0.0,
        z3_weight: float = 0.2,
        meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    AR-LSAT ORDERING reward.

    Expected model output:
      <answer>{
        "problem_type": "ordering",
        "world_model": {...},
        "rules": [...],
        "facts": [...],
        "question_semantics": {"question_type": "...", ...},
        "options": {"A": "...", ...},
        "reasoning": [...],
        "solution": {"selected_option": "A"}
      }</answer>

    Reward components:
      - answer accuracy: selected_option == ground_truth answer
      - parsing/schema reward
      - interleaved format reward
      - Z3 option semantics and reasoning validation
    """

    import time

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
        schema_reward = _ar_rule_field_score(payload)

        selected = ((payload.get("solution") or {}).get("selected_option") or "")
        selected = str(selected).strip().upper() if selected is not None else ""
        gt_option = _ar_selected_option_from_gt(ground_truth, extra_info=extra_info, meta=meta)

        option_acc = 1.0 if selected and gt_option and selected == gt_option else 0.0

        reasoning = payload.get("reasoning")
        format_reward = 1.0 if _ar_check_interleaved_reasoning(reasoning) else 0.0

        payload_for_z3 = dict(payload)
        payload_for_z3["ground_truth"] = gt_option

        try:
            z3_out = solve_and_validate_payload(
                payload_for_z3,
                timeout_s=float(timeout),
                conflict_tolerant_clues=False,
            )
        except Exception as e:
            z3_out = {
                "parse_status": "Z3_EXCEPTION",
                "error": f"{type(e).__name__}: {e}",
            }

        base_gt = float(z3_out.get("base_sat_full_GT", 0.0) or 0.0)
        selected_semantics_ok = float(z3_out.get("selected_option_semantics_ok", 0.0) or 0.0)

        n_total = int(z3_out.get("n_steps_total", 0) or 0)
        n_parsed = int(z3_out.get("n_steps_parsed_ok", 0) or 0)
        n_valid = int(z3_out.get("n_steps_valid", 0) or 0)
        n_novel = int(z3_out.get("n_steps_novel_inc_clues", 0) or 0)
        n_contra = int(z3_out.get("n_non_valid_contradiction", 0) or 0)

        parse_step_rate = n_parsed / max(1, n_total)
        valid_step_rate = n_valid / max(1, n_total)
        novel_step_score = min(n_novel / 3.0, 1.0)
        contradiction_penalty = min(n_contra / max(1, n_total), 1.0)

        z3_reward_value = (
            0.40 * base_gt
            + 0.20 * selected_semantics_ok
            + 0.15 * parse_step_rate
            + 0.15 * valid_step_rate
            + 0.10 * novel_step_score
        )
        z3_reward_value = max(0.0, z3_reward_value - 0.25 * contradiction_penalty)

        # Conservative final reward: answer correctness remains the anchor.
        final_reward = (
            0.60 * option_acc
            + 0.15 * parsing_reward
            + 0.10 * schema_reward
            + 0.05 * format_reward
            + 0.10 * z3_reward_value
        )

        final_result.update({
            "acc": float(final_reward),
            "score": float(final_reward),
            "reward_logged": float(final_reward),
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
            os.makedirs(feedback_root, exist_ok=True)
            feedback_path = os.path.join(feedback_root, f"ar_lsat_ordering_epoch_{epoch}_feedback.jsonl")
            _append_jsonl(feedback_path, {
                "payload": payload,
                "ground_truth": ground_truth,
                "z3_out": z3_out,
                "final_result": final_result,
                "solution_str": solution_str,
                "timestamp": time.time(),
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
    def run_case(name, sol, gt):
        print(f"\n=== {name} ===")
        out = compute_score(sol, gt)
        print(json.dumps(out, indent=2, ensure_ascii=False))

    # Test 1: Correct could_be_true ordering answer.
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
      "facts": [
        "B == 4"
      ],
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

    # Test 2: Wrong selected option should receive low answer accuracy.
    sol_wrong = sol_correct.replace('"selected_option": "A"', '"selected_option": "B"')
    sol_wrong = sol_wrong.replace('"S3: Sat(Option_A)."', '"S3: Unsat(Option_A)."')

    # Test 3: Must-be-true semantics.
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
      "rules": [
        "Distinct(A, B, C)",
        "A < B",
        "B < C"
      ],
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
