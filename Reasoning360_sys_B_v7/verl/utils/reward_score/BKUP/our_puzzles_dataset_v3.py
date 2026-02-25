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


from verl.utils.reward_score.z3_verifier_v9 import curriculum_reward_cell_to_puzzle, curriculum_reward_cell_to_puzzle_only
from verl.utils.reward_score.z3_verifier_v9 import compute_z3_components

os.environ.setdefault("CLUE_TIMEOUT_S", "3.0")
os.environ.setdefault("Z3_TIMEOUT_S", "1.5")
os.environ.setdefault("Z3_CLUE_GATE", "0.7")
os.environ.setdefault("CLUE_MAX_NEW_TOKENS", "256")
os.environ.setdefault("CLUE_MAX_INFLIGHT", "1")





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


def extract_reasoning_and_solution(solution_str: str) -> Tuple[Optional[str], Optional[Any], str]:
    """
    Extract both reasoning and solution.
    Returns (reasoning, solution, status)
    """
    answer_content = parse_answer_tag(solution_str)
    if answer_content:
        parsed = _try_parse_first_json_obj(answer_content)
        if parsed is not None:
            return parsed.get("reasoning", None), parsed.get("solution", None), "success_answer_tag"
        return None, None, "answer_tag_json_error"

    parsed = _try_parse_first_json_obj(solution_str)
    if parsed is not None:
        return parsed.get("reasoning", None), parsed.get("solution", None), "success_direct_json"

    return None, None, "parsing_failed"


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


def _compute_acc_from_normalized(norm_pred: Dict[str, Any], norm_gt: Dict[str, Any]) -> float:
    """Exact match => 1.0, otherwise cell-level acc if shapes align."""
    if not norm_pred or not norm_gt:
        return (0.0, 0.0)
    if norm_pred == norm_gt:
        return (1.0, 1.0)

    ph, pr = norm_pred.get("header", []), norm_pred.get("rows", [])
    gh, gr = norm_gt.get("header", []), norm_gt.get("rows", [])
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

    epoch = int(os.getenv("CURRENT_EPOCH", "1"))
    total_epochs = int(os.getenv("TOTAL_EPOCH", "100"))

    acc_score = 0.0
    puzzle_acc_score = 0.0
    z3_score = 0.0
    clue_score = 0.0
    z3_result = {"structure_score": 0.0, "parse_cov": 0.0, "clue_sat": 0.0, "reason_score": 0.0,
                 "z3_effective": 0.0, "z3_sat": 0.0, "parsed_constraints": 0, "parsed_clues": 0}

    score_method = str(os.environ.get("TRAIN_SCORE_METHOD", "gt+z3"))
    #print(f"Score method in Model Training: {score_method}")

    try:
        reasoning, predicted_arrangement, parse_status = extract_reasoning_and_solution(solution_str=solution_str)

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
                    #acc_score, puzzle_acc_score = _compute_acc_from_normalized(norm_pred, norm_gt)
                    acc_score, puzzle_acc_score = puzzle_and_cell_accuracy(norm_pred, norm_gt)
                else:
                    acc_score = 1.0 if pred_conv == gt_conv else 0.0
                    puzzle_acc_score = 1.0 if pred_conv == gt_conv else 0.0
            except Exception as e:
                logger.error(f"Error calculating ACC score: {e}")
                acc_score = 0.0
                puzzle_acc_score = 0.0

        # ---------------- Z3 (先算 Z3，作为 clue gate) ----------------
        if compute_z3 and reasoning and predicted_arrangement is not None and isinstance(predicted_arrangement, dict) > 0:
            try:
                pred_conv = convert_numpy_arrays(predicted_arrangement)

                z3_timeout_s = float(os.environ.get("Z3_TIMEOUT_S", "1.5"))
                z3_timeout_s = min(z3_timeout_s, time_left())

                _ensure_z3_timeout(z3_timeout_s)
                if isinstance(extra_info, dict) and "clues" in extra_info:
                    clues = extra_info.get("clues")

                # 注意：ACC 已经单独算过了，这里不需要 ground_truth 对比，减少开销

                z3_result = compute_z3_components(reasoning, pred_conv, clues, debug=False)
                z3_score = z3_result # float(z3_result.get("total_reward", 0.0))
            except Exception as e:
                logger.exception("Crash in compute_score")  # includes line number + stack
                logger.error(f"Error calculating Z3 score: {e}")
                z3_result = {"structure_score": 0.0, "parse_cov": 0.0, "clue_sat": 0.0, "reason_score": 0.0,
                 "z3_effective": 0.0, "z3_sat": 0.0, "parsed_constraints": 0, "parsed_clues": 0}
                #print()
                #print()
                #print(f" Reasoning : {reasoning}")
                #print(f" Pred_conv : {pred_conv}")
                #print(f" Clues : {clues}")
                #print()
                #print()

        # ---------------- CLUE (Z3 gate + 独立超时 + cancel) ----------------

        if compute_clue and reasoning > 0:
            try:
                z3_gate = float(os.environ.get("Z3_CLUE_GATE", "0.7"))
                if z3_score < z3_gate:
                    clue_score = 0.0
                    clues = []
                    reasoning = ""
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

                PROPOSED SOLUTION: {reasoning}

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
                logger.info(f"Z3-Result: {z3_result}")
                logger.exception("Crash in compute_score")  # includes line number + stack
                logger.error(f"Error calculating Reasoning and Clues: {e}")

                final_prompt = ""
        else:
            final_prompt = ""


        # ---------------- weighted total in sequence gt+z3+clue----------------
        if score_method == "gt":
            weighted_total = acc_score
        elif score_method == "z3":
            weighted_total = z3_score
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
            reward = curriculum_reward_cell_to_puzzle_only(
                cell_acc=acc_score,
                puzzle_acc=puzzle_acc_score,
                epoch=epoch,
                total_epochs=total_epochs)
            #weighted_total = 0.0


    except Exception as e:
        logger.info(f"Z3-Result: {z3_result}")
        logger.exception("Crash in compute_score")  # includes line number + stack
        logger.error(f"Error in compute_score in our_puzzles_dataset: {e}")

        weighted_total = 0.0
        acc_score = 0.0
        z3_score = 0.0
        clue_score = 0.0
        reward = 0.0


        #print(f" Solution String : {solution_str}")
        #print(f" Ground Truth : {ground_truth}")
        #print(f" Extra Info : {extra_info}")
    final_result = {"epoch": epoch, "total-epoch": total_epochs, "score": reward, "cell_acc": acc_score, "puzzle_acc_score":puzzle_acc_score,
                    "cell_acc_score":acc_score, "z3_result":z3_result, "verification_prompt": final_prompt}
    #print(f"final_result: {final_result}")
    return final_result



def pretty(x):
    print(json.dumps(x, indent=2, ensure_ascii=False))


def main():
    # ----------------------------
    # Minimal 3-house toy example
    # ----------------------------
    clues = [
        "Arnold is the person whose favorite color is red.",
        "The person whose child is named Fred is somewhere to the left of Eric.",
        "The person whose favorite color is red is in the second house.",
        "The person whose child is named Bella is in the first house.",
        "The person who loves white is the person whose child is named Meredith."
    ]

    ground_truth = {
            "header": ["Person", "Name", "Favorite Color", "Child's Name"],
            "rows": [["Arnold", "Arnold", "red", "Fred"],
            ["Eric", "Eric", "white", "Bella"],
            ["Peter", "Peter", "yellow", "Meredith"]]
            }
    # ✅ Perfect sample
    llm_example_1 = """
    <answer>{"reasoning": ["Clue 2 is directly telling us the name of the person whose child is Fred. Thus, we know who Fred is. This also gives us the first person in the list.", 
            "Clue 5 tells us the person who loves white is Meredith, so we know the fourth person in the list is Meredith. Hence, the third person must be the one with white.", 
            "Clue 3 tells us that the person whose favorite color is red is in the second house. Hence, the second person in the list must be the one with red.", 
            "Clue 4 tells us that the person whose child is named Bella is in the first house. Hence, the first person in the list must be the one with white.", 
            "The only person left in the list is the one with yellow, and the only place left for them to be is in the last house."],
            "solution": {
            "header": ["Person", "Name", "Favorite Color", "Child's Name"], 
            "rows": [["Arnold", "Arnold", "red", "Fred"], 
            ["Eric", "Eric", "white", "Bella"], 
            ["Peter", "Peter", "yellow", "Meredith"]]
            }
    }</answer>
    """

    # Extra info carries clues/meta for clue prompt generation
    extra_info = {"clues": clues, "meta": {"clues": clues}}



    # ----------------------------
    # 3) Combined (gt+z3+clue)
    # - clue part builds a prompt in `verification_`
    # - clue_score may remain 0.0 unless you add the actual clue verifier call later
    # ----------------------------
    os.environ["TRAIN_SCORE_METHOD"] = "gt+z3"
    print("\n=== TRAIN_SCORE_METHOD=gt+z3 ===")
    out_ = compute_score(llm_example_1, ground_truth, extra_info=extra_info, timeout=3.0)

    #out_bad  = compute_score(llm_bad,  ground_truth, extra_info=extra_info, timeout=3.0)
    print("Output:")
    pretty(out_)



    '''
        # ----------------------------
    # 1) GT only (no Z3 invoked)
    # ----------------------------
    os.environ["TRAIN_SCORE_METHOD"] = "gt"
    print("\n=== TRAIN_SCORE_METHOD=gt ===")
    out_good = compute_score(llm_good, ground_truth, extra_info=extra_info, timeout=3.0)
    out_bad  = compute_score(llm_bad,  ground_truth, extra_info=extra_info, timeout=3.0)
    print("GOOD:")
    pretty(out_good)
    print("BAD:")
    pretty(out_bad)

    # ----------------------------
    # 2) Z3 only (invokes verify_solution_with_z3)
    # ----------------------------
    os.environ["TRAIN_SCORE_METHOD"] = "z3"
    print("\n=== TRAIN_SCORE_METHOD=z3 ===")
    out_good = compute_score(llm_good, ground_truth, extra_info=extra_info, timeout=3.0)
    out_bad  = compute_score(llm_bad,  ground_truth, extra_info=extra_info, timeout=3.0)
    print("GOOD:")
    pretty(out_good)
    print("BAD:")
    pretty(out_bad)
    
    # ----------------------------
    # 3) Combined (gt+z3+clue)
    # - clue part builds a prompt in `verification_`
    # - clue_score may remain 0.0 unless you add the actual clue verifier call later
    # ----------------------------
    os.environ["TRAIN_SCORE_METHOD"] = "gt+z3"
    print("\n=== TRAIN_SCORE_METHOD=gt+z3 ===")
    out_good = compute_score(llm_good, ground_truth, extra_info=extra_info, timeout=3.0)
    #out_bad  = compute_score(llm_bad,  ground_truth, extra_info=extra_info, timeout=3.0)
    print("GOOD:")
    pretty(out_good)

    print("BAD:")
    pretty(out_bad)

    # Print just the generated clue-verification prompt
    print("\n--- verification_ prompt preview ---")
    print(out_good.get("verification_", "")[:600], "...\n")
    
    '''


if __name__ == "__main__":
    main()
