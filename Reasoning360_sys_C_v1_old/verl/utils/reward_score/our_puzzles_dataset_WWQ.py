# our_puzzles_dataset_WWQ.py
import os
import re
import ast
import sys
import json
import time
import numpy as np
import signal
import logging
import contextlib
from typing import Dict, List, Any, Optional, Tuple


from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import solve_and_validate_payload
from verl.utils.reward_score.z3_reasoning_validator_v13_gt_solve_v9 import normalize_header, normalize_months_in_rows
from verl.utils.reward_score.check_interleved_format import check_interleaved_reasoning
from verl.utils.reward_score.z3_reasoning_vs_solution_verifier_v2 import verify_solution_two_step
from prompt_step_2 import SOLUTION_PROMPT_VERIFIER_V2, SOLUTION_PROMPT_1_SHOT_VERIFIER_USER_V2

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


try:
    from .ray_clue_verifier_WWQ import RayClueVerifier, get_global_ray_clue_verifier  # noqa: F401
except Exception:
    RayClueVerifier = None
    get_global_ray_clue_verifier = None


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    """
    Process-level timeout (signal-based). Works in main thread on Unix.
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)



# -------------------------
# Canonicalization helpers
# -------------------------
def _canon_value(val: Any) -> str:
    """
    Canonicalize a cell value:
      - strip
      - whitespace -> underscore
      - collapse underscores
      - lowercase
    """
    if val is None:
        return ""
    s = str(val).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.lower()


def _canon_attr(attr: Any) -> str:
    """
    Attribute names: keep case (but strip).
    """
    if attr is None:
        return ""
    return str(attr).strip()


def _safe_parse_list_str(s: str):
    """
    Parse a python-list-like string safely.
    """
    try:
        return ast.literal_eval(s)
    except Exception:
        items = re.findall(r"""['"]([^'"]+)['"]""", s)
        return items if items else None





def _fix_solution_table(solution: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix solution['header'] / solution['rows'] if they come as strings.
    """
    if not isinstance(solution, dict):
        return solution

    if isinstance(solution.get("header"), str):
        parsed = _safe_parse_list_str(solution["header"])
        if parsed is not None:
            solution["header"] = parsed

    if isinstance(solution.get("rows"), str):
        parsed = _safe_parse_list_str(solution["rows"])
        if parsed is not None:
            solution["rows"] = parsed

    return solution

def _find_bad_key(d):
    for k, v in d.items():
        try:
            json.dumps(v, ensure_ascii=False)
        except Exception as e:
            print("BAD KEY:", k, "TYPE:", type(v), "ERR:", e)
            return k
    return None

def _canon_table_solution(sol: Any) -> Any:
    """
    Canonicalize solution table values (except House index).
    """
    if not isinstance(sol, dict):
        return sol
    sol = _fix_solution_table(sol)
    header = sol.get("header", [])
    rows = sol.get("rows", [])
    if not isinstance(header, list) or not isinstance(rows, list):
        return sol

    # find house col
    house_idx = None
    for i, h in enumerate(header):
        if str(h).strip().lower() == "house":
            house_idx = i
            break

    new_rows = []
    for r in rows:
        if not isinstance(r, list):
            new_rows.append(r)
            continue
        rr = []
        for j, cell in enumerate(r):
            if house_idx is not None and j == house_idx:
                rr.append(str(cell).strip())
            else:
                rr.append(_canon_value(cell))
        new_rows.append(rr)

    sol["header"] = [str(x).strip() for x in header]
    sol["rows"] = new_rows
    return sol


# -------------------------
# Full payload extraction
# -------------------------
def extract_full_payload(solution_str: str) -> Dict[str, Any]:
    """
    Extract the full 5-key payload from model output:
    - n_houses, attribute_values, parsed_clues, parsed_reasoning, solution
    """
    out = {
        "n_houses": None,
        "attribute_values": None,
        "parsed_clues": None,
        "parsed_reasoning": None,
        "reasoning_text": None,
        "solution": None,
    }
    if not solution_str:
        return out

    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE))
    payload_text = matches[-1].group(1).strip() if matches else None

    # 1) Prefer <answer> JSON
    if payload_text:
        try:
            obj = json.loads(payload_text)

            out["n_houses"] = obj.get("n_houses", None)

            av = obj.get("attribute_values", None)
            if av is None and "attribute_valves" in obj:  # 常见 typo 兼容
                av = obj.get("attribute_valves")
            out["attribute_values"] = av

            pc = obj.get("parsed_clues", None)
            out["parsed_clues"] = pc

            pr = obj.get("parsed_reasoning", None)
            if isinstance(pr, list):
                pr_list = [str(x).strip() for x in pr if str(x).strip()]
                out["parsed_reasoning"] = pr_list if pr_list else None
                out["reasoning_text"] = "\n".join(pr_list) if pr_list else None
            elif isinstance(pr, str):
                lines = [x.strip() for x in pr.splitlines() if x.strip()]
                out["parsed_reasoning"] = lines if lines else None
                out["reasoning_text"] = pr.strip()

            sol = obj.get("solution", None)
            if isinstance(sol, dict):
                out["solution"] = _fix_solution_table(sol)
            elif isinstance(sol, list):
                out["solution"] = sol
            elif isinstance(obj, dict) and "header" in obj and "rows" in obj:
                out["solution"] = _fix_solution_table({"header": obj.get("header"), "rows": obj.get("rows")})

            return out
        except Exception:
            pass

    # 2) Fallback: best-effort JSON object in text
    json_starts = [m.start() for m in re.finditer(r"\{", solution_str)]
    for start_idx in json_starts[:3]:
        brace_count = 1
        end_idx = start_idx + 1
        while end_idx < len(solution_str) and brace_count > 0:
            if solution_str[end_idx] == "{":
                brace_count += 1
            elif solution_str[end_idx] == "}":
                brace_count -= 1
            end_idx += 1
        if brace_count == 0:
            js = solution_str[start_idx:end_idx]
            try:
                obj = json.loads(js)
                out["n_houses"] = obj.get("n_houses", out["n_houses"])

                av = obj.get("attribute_values", None)
                if av is None and "attribute_valves" in obj:
                    av = obj.get("attribute_valves")
                out["attribute_values"] = av or out["attribute_values"]

                out["parsed_clues"] = obj.get("parsed_clues", out["parsed_clues"])

                pr = obj.get("parsed_reasoning", None)
                if pr:
                    if isinstance(pr, list):
                        pr_list = [str(x).strip() for x in pr if str(x).strip()]
                        out["parsed_reasoning"] = pr_list if pr_list else out["parsed_reasoning"]
                        out["reasoning_text"] = "\n".join(pr_list) if pr_list else out["reasoning_text"]
                    elif isinstance(pr, str):
                        lines = [x.strip() for x in pr.splitlines() if x.strip()]
                        out["parsed_reasoning"] = lines if lines else out["parsed_reasoning"]
                        out["reasoning_text"] = pr.strip()

                sol = obj.get("solution", None)
                if isinstance(sol, dict):
                    out["solution"] = _fix_solution_table(sol)
                elif isinstance(obj, dict) and "header" in obj and "rows" in obj:
                    out["solution"] = _fix_solution_table({"header": obj.get("header"), "rows": obj.get("rows")})
                return out
            except Exception:
                continue

    return out


def extract_reasoning_and_solution(solution_str: str) -> Tuple[Optional[str], Optional[Any]]:
    """
    Backward-compatible wrapper.
    """
    payload = extract_full_payload(solution_str)
    return payload.get("reasoning_text"), payload.get("solution")


# -------------------------
# Edit distance (kept)
# -------------------------
def compute_edit_distance(list1, list2) -> int:
    """
    Standard edit distance (Levenshtein) for two lists.
    """
    dp = [[0 for _ in range(len(list2) + 1)] for _ in range(len(list1) + 1)]
    for i in range(len(list1) + 1):
        dp[i][0] = i
    for j in range(len(list2) + 1):
        dp[0][j] = j
    for i in range(1, len(list1) + 1):
        for j in range(1, len(list2) + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(list1)][len(list2)]


# ============================================================
# Z3 DSL -> Solver builder + Reasoning step verifier (核心)
# ============================================================
class _Z3DSLCompiler:
    """
    Compile parsed_clues DSL + reasoning steps DSL to Z3 constraints.
    """

    def __init__(self, n_houses: int, attribute_values: Dict[str, List[str]]):
        self.z3 = __import__("z3")
        self.n = int(n_houses)

        self.attrs: List[str] = []
        self.vals: Dict[str, List[str]] = {}
        self.val2idx: Dict[str, Dict[str, int]] = {}
        self.vars: Dict[str, List[Any]] = {}

        # Canon values
        for attr, values in (attribute_values or {}).items():
            a = _canon_attr(attr)
            if not isinstance(values, list):
                continue
            canon_vals = [_canon_value(v) for v in values]
            # unique
            uniq = []
            for v in canon_vals:
                if v not in uniq:
                    uniq.append(v)
            canon_vals = uniq
            self.attrs.append(a)
            self.vals[a] = canon_vals
            self.val2idx[a] = {v: i for i, v in enumerate(canon_vals)}

        # Create house vars
        for attr in self.attrs:
            self.vars[attr] = [self.z3.Int(f"{attr}_h{h}") for h in range(1, self.n + 1)]

        # Base constraints: domain + Distinct per attribute
        self.base_constraints = []
        for attr in self.attrs:
            m = max(len(self.vals[attr]), self.n) if self.vals.get(attr) else self.n
            for h in range(self.n):
                self.base_constraints.append(self.z3.And(self.vars[attr][h] >= 0, self.vars[attr][h] < m))
            self.base_constraints.append(self.z3.Distinct(*self.vars[attr]))

    def _idx(self, attr: str, val: str) -> int:
        a = _canon_attr(attr)
        v = _canon_value(val)
        if a not in self.val2idx or v not in self.val2idx[a]:
            raise KeyError(f"Value '{val}' not in attribute_values for attribute '{attr}'. Canon='{v}'")
        return self.val2idx[a][v]

    def _house_var(self, attr: str, house: int):
        a = _canon_attr(attr)
        if a not in self.vars:
            raise KeyError(f"Unknown attribute: {attr}")
        if not (1 <= house <= self.n):
            raise ValueError(f"House out of range: {house}")
        return self.vars[a][house - 1]

    def pos(self, attr: str, val: str):
        """
        Return Z3 expression: the house index (1..n) where attr=val.
        Relies on Distinct to make it well-defined.
        """
        a = _canon_attr(attr)
        idx = self._idx(a, val)
        terms = []
        for h in range(1, self.n + 1):
            terms.append(self.z3.If(self._house_var(a, h) == idx, h, 0))
        return self.z3.Sum(*terms)

    def compile_predicate(self, pred: str):
        pred = pred.strip()
        pred = pred[:-1].strip() if pred.endswith(".") else pred

        m = re.match(r"^(\w+)\((.*)\)$", pred)
        if not m:
            raise ValueError(f"Bad predicate format: {pred}")
        name = m.group(1)
        inside = m.group(2).strip()

        if name in ("set", "not_set"):
            parts = [p.strip() for p in inside.split(",", 2)]
            if len(parts) != 3:
                raise ValueError(f"Bad {name} args: {pred}")
            H = int(parts[0])
            Attr = _canon_attr(parts[1])
            Val = parts[2]
            v = self._idx(Attr, Val)
            hv = self._house_var(Attr, H)
            return (hv == v) if name == "set" else (hv != v)

        if name in ("immediately_left_of", "left_of", "right_of", "adjacent", "same_house"):
            parts = [p.strip() for p in inside.split(",", 1)]
            if len(parts) != 2:
                raise ValueError(f"Bad {name} args: {pred}")
            A, B = parts
            a_attr, a_val = [x.strip() for x in A.split("=", 1)]
            b_attr, b_val = [x.strip() for x in B.split("=", 1)]
            pa = self.pos(a_attr, a_val)
            pb = self.pos(b_attr, b_val)

            if name == "immediately_left_of":
                return pa + 1 == pb
            if name == "left_of":
                return pa < pb
            if name == "right_of":
                return pa > pb
            if name == "adjacent":
                return self.z3.Abs(pa - pb) == 1
            if name == "same_house":
                return pa == pb

        if name == "between":
            parts = [p.strip() for p in inside.split(",", 2)]
            if len(parts) != 3:
                raise ValueError(f"Bad between args: {pred}")
            A, B, K = parts
            a_attr, a_val = [x.strip() for x in A.split("=", 1)]
            b_attr, b_val = [x.strip() for x in B.split("=", 1)]
            k = int(K)
            pa = self.pos(a_attr, a_val)
            pb = self.pos(b_attr, b_val)
            return self.z3.Abs(pa - pb) == (k + 1)

        raise ValueError(f"Unsupported predicate: {name}")

    def build_base_solver(self, parsed_clues: List[str], timeout_ms: int = 3000):
        s = self.z3.Solver()
        s.set("timeout", int(timeout_ms))

        for c in self.base_constraints:
            s.add(c)

        if parsed_clues:
            for line in parsed_clues:
                pred = self._extract_clue_predicate(line)
                if not pred:
                    continue
                # 只要能 parse，就加入；parse 失败就跳过（鲁棒）
                try:
                    s.add(self.compile_predicate(pred))
                except Exception:
                    continue
        return s

    @staticmethod
    def _extract_clue_predicate(line: str) -> Optional[str]:
        if not line:
            return None
        t = str(line).strip()

        m = re.match(r"^C\d+\s*=\s*(.+)$", t)
        if m:
            pred = m.group(1).strip()
            pred = pred[:-1].strip() if pred.endswith(".") else pred
            return pred

        m2 = re.search(r"(set|not_set|immediately_left_of|left_of|right_of|adjacent|same_house|between)\(.*\)", t)
        if m2:
            pred = m2.group(0).strip()
            pred = pred[:-1].strip() if pred.endswith(".") else pred
            return pred
        return None

    @staticmethod
    def parse_reasoning_step(step_line: str) -> Optional[Dict[str, Any]]:
        if not step_line:
            return None
        t = str(step_line).strip()
        if not t.endswith("."):
            t = t + "."

        m = re.match(r"^(S\d+)\s*\[([^\]]*)\]\s*(set|not|not_set)\((.*)\)\.\s*$", t)
        if not m:
            m2 = re.match(r"^(S\d+)\s*(set|not|not_set)\((.*)\)\.\s*$", t)
            if not m2:
                return None
            sid = m2.group(1)
            evidence = ""
            op = m2.group(2)
            args = m2.group(3)
        else:
            sid = m.group(1)
            evidence = m.group(2).strip()
            op = m.group(3)
            args = m.group(4)

        parts = [p.strip() for p in args.split(",", 2)]
        if len(parts) != 3:
            return None
        try:
            H = int(parts[0])
        except Exception:
            return None
        Attr = _canon_attr(parts[1])
        Val = parts[2].strip()

        return {
            "sid": sid,
            "evidence": evidence,
            "op": op,
            "H": H,
            "Attr": Attr,
            "Val": Val,
            "dsl": f"{op}({H},{Attr},{Val})",
            "raw": step_line,
        }

    def compile_step_atom(self, step: Dict[str, Any]):
        op = step["op"]
        H = int(step["H"])
        Attr = step["Attr"]
        Val = step["Val"]
        hv = self._house_var(Attr, H)
        idx = self._idx(Attr, Val)
        if op == "set":
            return hv == idx
        if op in ("not", "not_set"):
            return hv != idx
        raise ValueError(f"Unsupported step op: {op}")


def _verify_reasoning_steps_with_z3(
    n_houses: int,
    attribute_values: Dict[str, List[str]],
    parsed_clues: List[str],
    parsed_reasoning: List[str],
    step_timeout_ms: int = 1200,
    base_timeout_ms: int = 3000,
) -> Dict[str, Any]:
    """
    严格按你的定义：
      - BAD: base + step -> UNSAT
      - GOOD: base + Not(step) -> UNSAT   (被 base 蕴含)
      - UNKNOWN: else

    并且 base 会累积加入“已判定 GOOD 的 step”，用于后续 step 的验证（允许依赖早先正确推理）。
    """
    result = {
        "good_steps": [],
        "bad_steps": [],
        "unknown_steps": [],
        "all_steps": [],
        "enabled": False,
        "error": None,
    }

    if not (n_houses and attribute_values and parsed_clues and parsed_reasoning):
        result["enabled"] = False
        result["error"] = "MISSING_FIELDS_FOR_STEP_VERIFICATION"
        return result

    try:
        compiler = _Z3DSLCompiler(int(n_houses), attribute_values)
        base = compiler.build_base_solver(parsed_clues, timeout_ms=base_timeout_ms)

        base.set("timeout", int(base_timeout_ms))
        if base.check() == compiler.z3.unsat:
            result["enabled"] = True
            result["error"] = "BASE_CONSTRAINTS_UNSAT"
            # 仍继续，但标记 error
    except Exception as e:
        result["enabled"] = False
        result["error"] = f"BUILD_SOLVER_ERROR: {repr(e)}"
        return result

    result["enabled"] = True

    for line in parsed_reasoning:
        step = compiler.parse_reasoning_step(line)
        if not step:
            item = {"sid": None, "raw": line, "dsl": None, "label": "UNKNOWN", "note": "UNPARSABLE_STEP"}
            result["unknown_steps"].append(item)
            result["all_steps"].append(item)
            continue

        try:
            atom = compiler.compile_step_atom(step)
        except Exception as e:
            item = {**step, "label": "UNKNOWN", "note": f"COMPILE_ERROR: {repr(e)}"}
            result["unknown_steps"].append(item)
            result["all_steps"].append(item)
            continue

        # BAD test: base + atom UNSAT ?
        is_bad = False
        base.push()
        base.set("timeout", int(step_timeout_ms))
        base.add(atom)
        try:
            chk = base.check()
            if chk == compiler.z3.unsat:
                is_bad = True
        except Exception:
            is_bad = False
        base.pop()

        if is_bad:
            item = {**step, "label": "BAD", "note": "CONTRADICTION_WITH_BASE"}
            result["bad_steps"].append(item)
            result["all_steps"].append(item)
            continue

        # GOOD test: base + Not(atom) UNSAT ?
        is_good = False
        base.push()
        base.set("timeout", int(step_timeout_ms))
        base.add(compiler.z3.Not(atom))
        try:
            chk = base.check()
            if chk == compiler.z3.unsat:
                is_good = True
        except Exception:
            is_good = False
        base.pop()

        if is_good:
            item = {**step, "label": "GOOD", "note": "ENTAILED_BY_BASE"}
            result["good_steps"].append(item)
            result["all_steps"].append(item)
            # 累积加入 base
            base.add(atom)
        else:
            item = {**step, "label": "UNKNOWN", "note": "NOT_ENTAILED"}
            result["unknown_steps"].append(item)
            result["all_steps"].append(item)

    return result
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

def _normalize_atom(x: Any) -> str:
    return str(x).strip().lower()

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


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")




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




# ============================================================
# Main scoring function (Supports Step-1 and Step-2; interface remains unchanged)
# ============================================================
def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    method: str = "strict",
    timeout: float = 12.0,
    acc_weight: float = 1.0,
    clue_weight: float = 1.0,      # Reserve but do not use
    z3_weight: float = 1.0,
    z3_threshold: float = 0.7,     # Reserve but do not use
    clue_timeout: float = 30.0,    # Reserve but do not use
    meta: Optional[Dict[str, Any]] = None,
    clue_model_config: Optional[Dict[str, Any]] = None,  # Reserve but do not use
    ray_namespace: str = "clue_verifier",                # Reserve but do not use
    ray_actor_name: str = "clue_verification_actor",     # Reserve but do not use
    ray_address: Optional[str] = None,                   # Reserve but do not use
    runtime_env: Optional[Dict[str, Any]] = None,        # Reserve but do not use
) -> Dict[str, Any]:

    '''
    Supports scoring functions for Step-1 and Step-2:
    - Parse the five-key output
    - Z3 verification of the final solution table (verify_solution_with_z3)
    - Z3 line-by-line verification of parsed_reasoning (GOOD/BAD/UNKNOWN)
    - Write feedback JSONL for subsequent Step-2 training
    Applies to:
    - Step-1: Initial solution generation and scoring
    - Step-2: Feedback-based solution generation and scoring
    '''

    meta = meta or {}

    cell_acc_score = 0.0
    puzzle_acc_score = 0.0
    z3_score = 0.0
    clue_score = 0.0  # Never used, always 0
    start_time = time.time()

    step_verif = None
    step_score = 0.0

    # feedback config
    feedback_path = os.path.join(os.environ.get("PUZZLE_FEEDBACK_PATH", "./"), f"jobid_{job_id}")
    feedback_path = os.path.join(feedback_path, f"jobid_{job_id}_feedback.jsonl")
    enable_step_feedback = bool(meta.get("enable_step_feedback", True))
    step_timeout_ms = int(meta.get("step_timeout_ms", 1200))
    base_timeout_ms = int(meta.get("base_timeout_ms", 3000))
    step_weight = float(meta.get("step_weight", 1.0))

    # puzzle id
    puzzle_id = None
    if isinstance(extra_info, dict):
        puzzle_id = extra_info.get("id") or extra_info.get("puzzle_id")
    if puzzle_id is None and isinstance(meta, dict):
        puzzle_id = meta.get("id") or meta.get("puzzle_id")

    try:
        epoch = int(os.getenv("CURRENT_EPOCH", "90"))
        total_epochs = int(os.getenv("TOTAL_EPOCH", "100"))
        switch_epoch = int(os.environ.get("SWITCH_EPOCH", "25"))
        feedback_path = os.path.join(os.environ.get("PUZZLE_FEEDBACK_PATH", "./"), f"jobid_{job_id}")
        # feedback_path = os.path.join(feedback_path, f"jobid_{job_id}_epoch_{str(epoch)}_feedback.jsonl")

        puzzle_id = ""

        puzzle_id = None
        if isinstance(extra_info, dict):
            puzzle_id = extra_info.get("id") or extra_info.get("puzzle_id")
        if puzzle_id is None and isinstance(meta, dict):
            puzzle_id = meta.get("id") or meta.get("puzzle_id")

        ground_truth = normalize_ground_truth(ground_truth)
        ground_truth = normalize_header(ground_truth)
        # ground_truth = normalize_months_in_rows(ground_truth)
        norm_pred = None
        cell_acc_score = 0.0
        puzzle_acc_score = 0.0
        n_houses = 1
        verification = []
        parse_error = ""
        acc_error = ""
        z3_error = ""
        reward_error = ""

        final_result = {}
        z3_out = {}
        payload = {}
        final_result["BASE_sat_full_GT"] = 0.0
        final_result["missed_data"] = 0.0
        final_result["BASE_n_steps_total"] = 0.0
        final_result["BASE_n_steps_parsed_ok"] = 0.0
        final_result["BASE_n_steps_valid"] = 0.0
        final_result["BASE_n_steps_novel_inc_clues"] = 0.0
        final_result["BASE_n_non_valid_contradiction"] = 0.0
        final_result["Normalizer"] = 0.0

        final_result["acc"] = 0.0
        final_result["PUZZLE_ACCURACY"] = 0.0
        final_result["CELL_ACCURACY"] = 0.0
        final_result["score"] = 0.0
        final_result["epoch"] = epoch
        final_result["total_epochs"] = total_epochs
        final_result["reward_logged"] = 0.0

    except Exception as e:
        # logger.exception(f"Failed to get puzzle id from extra_info: {e}")
        logger.error(f"Failed to get puzzle id from extra_info: {e}")
        final_result = {}
        payload = {}
        z3_out = {}
        final_result["BASE_sat_full_GT"] = 0.0
        final_result["missed_data"] = 0.0

        final_result["BASE_n_steps_total"] = 0.0
        final_result["BASE_n_steps_parsed_ok"] = 0.0
        final_result["BASE_n_steps_valid"] = 0.0
        final_result["BASE_n_steps_novel_inc_clues"] = 0.0
        final_result["BASE_n_non_valid_contradiction"] = 0.0
        final_result["BASE_n_novel_inc_clues_contradiction"] = 0.0

        final_result["acc"] = 0.0
        final_result["PUZZLE_ACCURACY"] = 0.0
        final_result["CELL_ACCURACY"] = 0.0
        final_result["score"] = 0.0
        final_result["epoch"] = epoch
        final_result["total_epochs"] = total_epochs
        final_result["reward_logged"] = 0.0

    parsing_reward = 0.0
    try:
        syntactic_clues, parsed_reasoning, predicted_arrangement, attribute_values, n_houses, parse_status = extract_reasoning_and_solution(solution_str=solution_str)
        if parse_status != "success_answer_tag":
            if os.environ.get("DEBUG_CODE", "0").lower() in ("1", "true", "yes"):
                log_case("non_boxed_answer", solution_str, ground_truth, logger)

        if parse_status == "success_direct_json" or parse_status == "success_answer_json":
            parsing_reward = 1.0
        # meta selection
        meta_used = meta

        if meta_used is None and isinstance(extra_info, dict):
            meta_used = extra_info.get("meta") or extra_info
    except Exception as parse_error:
        logger.error(f"Error in solution parsing: {parse_error}")
        n_houses = 1
        parsing_reward = 0.0
        num_blocks = 0
        attribute_values = None
        syntactic_clues = None
        parsed_reasoning = None
        # final_result["missed_data"] = 1.0

    # ---------------- ACC ----------------
    if predicted_arrangement:
        try:
            pred_conv = convert_numpy_arrays(predicted_arrangement)
            gt_conv = convert_numpy_arrays(ground_truth)

            norm_pred = normalize_table(pred_conv)  # normalize_grid(pred_conv)
            norm_gt = normalize_table(gt_conv)  # normalize_grid(gt_conv)

            if norm_pred and norm_gt:
                norm_pred = normalize_header(norm_pred)
                # norm_pred = normalize_months_in_rows(norm_pred)
                cell_acc_score, puzzle_acc_score = _compute_acc_from_normalized(norm_pred, norm_gt)
                # puzzle_acc_score, cell_acc_score = puzzle_and_cell_accuracy(norm_pred, norm_gt)
            else:
                cell_acc_score = 1.0 if pred_conv == gt_conv else 0.0
                puzzle_acc_score = 1.0 if pred_conv == gt_conv else 0.0
        except Exception as acc_error:
            # print('Failed case prediction:', pred_conv)
            # print()
            # print('Failed case ground-truth:', gt_conv)
            # print()
            # logger.exception("Crash in ACC Scoring")
            # logger.error(f"Error calculating ACC score: {acc_error}")
            cell_acc_score = 0.0
            puzzle_acc_score = 0.0
    # -----------------------
    # Z3 scoring (base metrics)
    # -----------------------

    MISSING_BASE_DEFAULTS = {
        "BASE_sat_full_GT": 0.0,
        "n_steps_total": 0,
        "n_steps_parsed_ok": 0,
        "n_steps_valid": 0,
        "n_steps_novel_inc_clues": 0,
        "n_non_valid_contradiction": 0,
        # Optional lists for downstream debugging
        "list_steps_non_valid": [],
        "list_novel_steps_inc_clues": [],
    }

    FINAL_BASE_KEYS_MAP = {
        "n_steps_total": "BASE_n_steps_total",
        "n_steps_parsed_ok": "BASE_n_steps_parsed_ok",
        "n_steps_valid": "BASE_n_steps_valid",
        "n_steps_novel_inc_clues": "BASE_n_steps_novel_inc_clues",
        "n_non_valid_contradiction": "BASE_n_non_valid_contradiction",
    }

    def _normalize_binary(value) -> float:
        """Normalize any truthy/falsy value to 1.0 or 0.0."""
        return 1.0 if bool(value) else 0.0

    def _apply_base_results(final_result: dict, z3_out: dict, *, missed_data: float) -> None:
        """Write Z3 base results into final_result with consistent keys."""
        final_result["missed_data"] = float(missed_data)

        for src_key, dst_key in FINAL_BASE_KEYS_MAP.items():
            final_result[dst_key] = z3_out.get(
                src_key,
                MISSING_BASE_DEFAULTS[src_key], )

    def _is_sat_check_failure(z3_out: dict) -> bool:
        """Return True if the Z3 solver failed its SAT check."""
        return z3_out.get("parse_status") == "SAT_CHECK_FAIL"

    required_inputs_present = all([
        n_houses,
        attribute_values,
        syntactic_clues,
        parsed_reasoning,
    ])

    if required_inputs_present:
        payload = {
            "n_houses": n_houses,
            "attribute_values": attribute_values,
            "syntactic_clues": syntactic_clues,
            "reasoning": parsed_reasoning,
            "ground_truth": ground_truth,
        }

        try:
            z3_out = solve_and_validate_payload(payload, timeout_s=5.0, conflict_tolerant_clues=False, )
            logger.debug("Z3 parse_status=%s", z3_out.get("parse_status"))
        except Exception:
            z3_out = dict(MISSING_BASE_DEFAULTS)
            # logger.error("Crash while calculating Z3 score")

        if _is_sat_check_failure(z3_out):
            final_result["BASE_sat_full_GT"] = 0.0
            _apply_base_results(final_result, MISSING_BASE_DEFAULTS, missed_data=1.0)
        else:
            final_result["BASE_sat_full_GT"] = _normalize_binary(z3_out.get("base_sat_full_GT", 0.0))
            _apply_base_results(final_result, z3_out, missed_data=0.0)

    else:
        final_result["BASE_sat_full_GT"] = 0.0
        _apply_base_results(final_result, MISSING_BASE_DEFAULTS, missed_data=1.0)

    # -----------------------
    # Format Reward
    # -----------------------
    format_ok = False
    # if num_blocks==1 and parsed_reasoning:
    if parsed_reasoning:
        try:
            format_ok = check_interleaved_reasoning(parsed_reasoning, n_houses=int(n_houses))
            # print(parsed_reasoning)
            # print(format_ok)
        except Exception:
            # logger.error("Error computing format penalty..!")
            format_ok = False
    else:
        format_ok = False
    # print('Format reward = {}'.format(format_ok))

    # ---------------------------
    # Reasoning + Clues vs Solution Validator
    # ---------------------------
    consistency_score = 0
    reasoning_vs_sol_validate = {}
    if syntactic_clues and predicted_arrangement and z3_out:
        try:
            list_novel_steps_inc_clues = z3_out.get("list_novel_steps_inc_clues", [])
            reasoning_vs_sol_validate = verify_solution_two_step(syntactic_clues, list_novel_steps_inc_clues, predicted_arrangement)
            consistency_score = reasoning_vs_sol_validate['reward']
        except Exception:
            consistency_score = 0
    # print("Consistency score:", consistency_score)

    # -----------------------
    # Reward components (safe defaults)
    # -----------------------
    try:

        reward = 0.0
        normalizer = 1.0  # will be overwritten if inputs are valid

        has_required_inputs = (attribute_values is not None) and (n_houses is not None)

        if has_required_inputs:
            n_houses_i = max(int(n_houses), 0)
            n_attrs_i = max(len(attribute_values), 0)

            # Keep strictly positive to avoid division by zero.
            normalizer = max(5.0 * max(n_houses_i * n_attrs_i, 1), 1.0)

            n_contradictions = float(final_result.get("BASE_n_non_valid_contradiction", 0.0))
            n_novel_steps = float(final_result.get("BASE_n_steps_novel_inc_clues", 0.0))
            sat_ok = float(final_result.get("BASE_sat_full_GT", 0.0))  # expected 1.0 or 0.0

            if format_ok:
                format_reward = 1.0
            else:
                format_reward = 0.0
            # print("Format reward = {}".format(format_reward))

            if sat_ok == 0.0:
                reward = 0.2 * parsing_reward + 0.6 * float(puzzle_acc_score)
            else:
                # reward = (0.6 * float(puzzle_acc_score) + 0.4 * (n_novel_steps / normalizer) - 0.2 * (n_contradictions / normalizer) - 0.2 * format_penalty)
                # reward = 0.6 * float(puzzle_acc_score) + 0.4 * (n_novel_steps / normalizer) - 0.4 * (n_contradictions / normalizer) + 0.5 * format_reward + 0.5 * consistency_score
                # reward = 0.6 * float(puzzle_acc_score) + 0.1 * (n_novel_steps / normalizer) - 0.01 * (n_contradictions / normalizer) # + 0.5 * format_reward #- 0.4 * (n_contradictions / normalizer)  # + 0.5 * format_reward # + 0.5 * consistency_score
                reward = 0.2 * parsing_reward + 0.6 * float(puzzle_acc_score) + 0.1 * (n_novel_steps / normalizer) + 0.1 * consistency_score
                # reward = (1.0 * float(puzzle_acc_score) - 0.4 * (n_contradictions / normalizer))


        else:
            reward = 0.0

        # reward = 0.6 * float(puzzle_acc_score)
        # normalizer = 0.0

        # -----------------------
        # Log / persist to final_result
        # -----------------------
        final_result["Normalizer"] = float(normalizer)
        final_result["acc"] = float(reward)
        final_result["PUZZLE_ACCURACY"] = float(puzzle_acc_score)
        final_result["CELL_ACCURACY"] = float(cell_acc_score)
        final_result["score"] = float(reward)
        final_result["epoch"] = epoch
        final_result["total_epochs"] = total_epochs
        final_result["reward_logged"] = float(reward)


    except Exception:
        reward = 0.0
        # Hard fail-safe: never crash reward computation pipeline.
        logger.exception("Crash in Final Reward Scoring")

        final_result["Normalizer"] = 0.0
        final_result["acc"] = reward
        final_result["PUZZLE_ACCURACY"] = 0.0
        final_result["CELL_ACCURACY"] = 0.0
        final_result["score"] = reward
        final_result["epoch"] = epoch
        final_result["total_epochs"] = total_epochs
        final_result["reward_logged"] = reward

    if os.environ.get("VALID_STATUS", "0") == "1":
        feedback_path = os.path.join(feedback_path, f"jobid_{job_id}_epoch_{str(epoch)}_valid_feedback.jsonl")
        try:
            example_ = {}
            example_["pid"] = puzzle_id
            example_["puzzle_text"] = pid_to_puzzle_dic[puzzle_id]
            example_["z3_out"] = z3_out
            example_["payload"] = payload
            example_["ground_truth"] = ground_truth
            example_["original_prediction"] = predicted_arrangement
            if norm_pred != None:
                example_["processed_prediction"] = norm_pred
            else:
                example_["processed_prediction"] = "WRONG OUTPUT FORMAT"
            example_["z3_out"] = z3_out
            example_["reasoning_vs_sol_validate"] = reasoning_vs_sol_validate
            example_["reward"] = reward
            example_["Format_Check"] = format_ok
            example_["final_result"] = final_result
            if example_:
                os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
                _append_jsonl(feedback_path, example_)

        except Exception as e:
            logger.exception("Crash in Writing Feedback")

    elif os.environ.get("VALID_STATUS", "0") == "2":
        feedback_path = os.path.join(feedback_path, f"jobid_{job_id}_epoch_{str(epoch)}_train_feedback.jsonl")
        try:
            example_ = {}
            example_["pid"] = puzzle_id
            example_["puzzle_text"] = pid_to_puzzle_dic[puzzle_id]
            example_["z3_out"] = z3_out
            example_["reasoning_vs_sol_validate"] = reasoning_vs_sol_validate
            example_["payload"] = payload
            example_["ground_truth"] = ground_truth
            example_["original_prediction"] = predicted_arrangement
            if norm_pred != None:
                example_["processed_prediction"] = norm_pred
            else:
                example_["processed_prediction"] = "WRONG OUTPUT FORMAT"
            example_["z3_out"] = z3_out
            example_["reward"] = reward
            example_["Format_Check"] = format_ok
            example_["final_result"] = final_result
            if example_:
                os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
                _append_jsonl(feedback_path, example_)

        except Exception as e:
            logger.exception("Crash in Writing Feedback")

    # os.environ["VALID_STATUS"] = "0"
    # sorted_result = dict(sorted(final_result.items(), key=lambda x: x[0]))

    #feedback_path = os.path.join(feedback_path, f"_jobid_{job_id}")



    # -----------------------
    # 1) ACC (Optional, but you can keep it for debugging or training assistance)
    # -----------------------

    try:
        with time_limit(timeout):
            try:
                payload = extract_full_payload(solution_str)
                reasoning_text = payload.get("reasoning_text")
                parsed_reasoning = payload.get("parsed_reasoning")
                parsed_clues = payload.get("parsed_clues")
                attribute_values = payload.get("attribute_values")
                n_houses = payload.get("n_houses")
                predicted = payload.get("solution")

                # canonicalize prediction & gt
                predicted_c = _canon_table_solution(predicted) if isinstance(predicted, dict) else predicted

                if not isinstance(ground_truth, (list, dict)):
                    try:
                        ground_truth = ground_truth.tolist()
                    except Exception:
                        pass
                target_c = _canon_table_solution(ground_truth) if isinstance(ground_truth, dict) else ground_truth


                #print(f"[ACC CALCULATION] Starting...")
                pred_conv = convert_numpy_arrays(predicted_c)
                gt_conv = convert_numpy_arrays(target_c)

                norm_pred = normalize_grid(pred_conv)
                norm_gt = normalize_grid(gt_conv)

                if norm_pred and norm_gt:
                    cell_acc_score, puzzle_acc_score = _compute_acc_from_normalized(norm_pred, norm_gt)
                    # puzzle_acc_score, cell_acc_score = puzzle_and_cell_accuracy(norm_pred, norm_gt)
                else:
                    cell_acc_score = 1.0 if pred_conv == gt_conv else 0.0
                    puzzle_acc_score = 1.0 if pred_conv == gt_conv else 0.0

                #print(f"[ACC CALCULATION] ACC Score: {acc_score:.4f}")
            except Exception as e:
                cell_acc_score = 0.0
                puzzle_acc_score = 0.0
                logger.exception("Crash in Accuracy Computation")
                logger.error(f"Error in Accuracy Computation: {e}")

    except TimeoutException:
        cell_acc_score = 0.0
        puzzle_acc_score = 0.0
        print(f"[TIMEOUT] Accuracy Score calculation timed out after {timeout:.2f} seconds")



    # -----------------------
    # 2) Final-table Z3 verification
    # -----------------------
    try:
        with time_limit(timeout):
            try:
                #print(f"[Z3 FINAL TABLE] Starting...")
                if isinstance(predicted_c, dict) and "header" in predicted_c and "rows" in predicted_c:
                    z3_result = verify_solution_with_z3(predicted_c, target_c, meta)
                    z3_score = float(z3_result.get("z3_score", 0.0))
                else:
                    z3_score = 0.0
                #print(f"[Z3 FINAL TABLE] Z3 Score: {z3_score:.4f}")
            except Exception as e:
                z3_score = 0.0
                logger.exception("Crash in Z3 Score Computation")
                logger.error(f"Error in Z3 Score Computation: {e}")
    except TimeoutException:
        z3_score = 0.0
        print(f"[TIMEOUT] Z3 Score calculation timed out after {timeout:.2f} seconds")


    # -----------------------
    # 3) Step-level Z3 verification (core)
    # -----------------------
    try:
        with time_limit(timeout):
            try:
                #print(f"[STEP VERIFICATION] Starting...")
                if isinstance(parsed_reasoning, list) and isinstance(parsed_clues, list) and isinstance(attribute_values, dict) and n_houses:
                    av_c = {}
                    for a, vs in attribute_values.items():
                        if isinstance(vs, list):
                            av_c[_canon_attr(a)] = [_canon_value(v) for v in vs]

                    step_verif = _verify_reasoning_steps_with_z3(
                        n_houses=int(n_houses),
                        attribute_values=av_c,
                        parsed_clues=parsed_clues,
                        parsed_reasoning=parsed_reasoning,
                        step_timeout_ms=step_timeout_ms,
                        base_timeout_ms=base_timeout_ms,
                    )
                    g = len(step_verif.get("good_steps", []))
                    b = len(step_verif.get("bad_steps", []))
                    u = len(step_verif.get("unknown_steps", []))
                    denom = max(g + b + u, 1)
                    step_score = g / denom
                    #print(f"[STEP VERIFICATION] GOOD={g}, BAD={b}, UNKNOWN={u}, StepScore={step_score:.4f}")
                else:
                    step_verif = {
                        "enabled": False,
                        "error": "STEP_VERIFICATION_SKIPPED_MISSING_FIELDS",
                        "good_steps": [],
                        "bad_steps": [],
                        "unknown_steps": [],
                        "all_steps": [],
                    }
                    #print("[STEP VERIFICATION] Skipped: missing fields from model output.")
            except Exception as e:
                step_verif = {}
                logger.exception("Crash in Z3 Step Verification")
                logger.error(f"Error in Z3 Step Verification: {e}")
    except TimeoutException:
        step_verif = {}
        print(f"[TIMEOUT] Z3 Step Verification timed out after {timeout:.2f} seconds")



    # -----------------------
    # 4) Clue verifier removed ()
    # -----------------------
    clue_score = 0.0
    #print(f"[CLUE VERIFICATION] Disabled by design. Clue Score: {clue_score:.4f}")



    # -----------------------
    # 5) Weighted total (using only ACC + Z3 (final) + StepScore)
    # If you don't want ACC to affect training: set meta['acc_weight']=0.0
    # -----------------------
    #print(f"[WEIGHTED TOTAL] Calculating...")

    reward = 1.0 if puzzle_acc_score == 1.0 else min(0.95, 0.2 + 0.75 * cell_acc_score)
    total_weight = acc_weight + z3_weight + step_weight

    weighted_total = (
        (cell_acc_score * acc_weight + z3_score * z3_weight + step_score * step_weight) / total_weight
        if total_weight > 0 else 0.0
    )
    #print(f"[WEIGHTED TOTAL] Weighted Score: {weighted_total:.4f}")

    # -----------------------
    # 6) Not used; only compatible with Write feedback JSONL (for Step-2).
    # -----------------------
    # Ensure that feedback_path is set correctly
    if not feedback_path:
        feedback_path = os.path.join(os.getcwd(), f"feedback_{job_id}.jsonl")
        print(f"WARNING: feedback_path is not set, using the default path: {feedback_path}")

    # Ensure enable_step_feedback is True
    if not enable_step_feedback:
        enable_step_feedback = True
        print("WARNING: enable_step_feedback为False，设置为True")
    
    #print(f"DEBUG: enable_step_feedback={enable_step_feedback}, feedback_path={feedback_path}, step_verif类型={type(step_verif)}")
    
    if enable_step_feedback and feedback_path and isinstance(step_verif, dict):
        # Extract the original puzzle text
        puzzle_text = None
        if isinstance(extra_info, dict):
            puzzle_text = extra_info.get("puzzle_text") or extra_info.get("text")
        
        # Collect the complete step1 JSON data
        step1_json = {
            "n_houses": n_houses,
            "attribute_values": attribute_values,
            "parsed_clues": parsed_clues,
            "parsed_reasoning": parsed_reasoning,
            "solution": predicted_c
        }
        
        # Prepare verification feedback JSON
        verifier_feedback = {
            "passed_steps": [step["raw"] for step in step_verif.get("good_steps", [])],
            "failed_steps": [{"step": step["raw"], "error": step["note"]} for step in step_verif.get("bad_steps", [])],
            "notes": step_verif.get("error", None)
        }
        
        record = {
            "timestamp": time.time(),
            "puzzle_id": puzzle_id,
            "epoch": meta.get("epoch", 1) if isinstance(meta, dict) else 1,
            "n_houses": n_houses,
            "attribute_values": attribute_values,
            "parsed_clues": parsed_clues,
            "parsed_reasoning": parsed_reasoning,
            "solution_predicted": predicted_c,
            "scores": {
                "acc_score": cell_acc_score,
                "z3_score": z3_score,
                "step_score": step_score,
                "weighted_total": weighted_total,
            },
            "step_verification": {
                "enabled": step_verif.get("enabled", False),
                "error": step_verif.get("error"),
                "good_steps": step_verif.get("good_steps", []),
                "bad_steps": step_verif.get("bad_steps", []),
                "unknown_steps": step_verif.get("unknown_steps", []),
            },
            # Required information for Step-2 prompt generation
            "puzzle_text": puzzle_text,
            "step1_json": step1_json,
            "verifier_feedback": verifier_feedback,
            # Optional: Save the raw output for debugging.
            "raw_model_output": solution_str[:20000],
        }


        record_extra = {
            "timestamp": time.time(),
            "puzzle_id": puzzle_id,
            "epoch": meta.get("epoch", 1) if isinstance(meta, dict) else 1,
            "n_houses": n_houses,
            "attribute_values": attribute_values,
            "parsed_clues": parsed_clues,
            "parsed_reasoning": parsed_reasoning,
            "solution_predicted": predicted_c,
            "scores": {
                "acc_score": cell_acc_score,
                "z3_score": z3_score,
                "step_score": step_score,
                "weighted_total": weighted_total,
            },
            "step_verification": {
                "enabled": step_verif.get("enabled", False),
                "error": step_verif.get("error"),
                "good_steps": step_verif.get("good_steps", []),
                "bad_steps": step_verif.get("bad_steps", []),
                "unknown_steps": step_verif.get("unknown_steps", []),
            },
            # Required information for Step-2 prompt generation
            "puzzle_text": pid_to_puzzle_dic[puzzle_id],
            "step1_json": step1_json,
            "verifier_feedback": verifier_feedback,
            # Optional: Save the raw output for debugging.
            "raw_model_output": solution_str[:20000],
        }

        if os.environ.get("STEP1_STATUS", "0") == "1":
            try:

                if puzzle_acc_score != 1.0:
                    puzzle_text = pid_to_puzzle_dic[puzzle_id]
                    grid = pid_to_puzzle_dic[puzzle_id + '_sol']

                    if os.environ.get("VERIFICATION_PASSED", "0") == "1":
                        #print('Writing example with verification passed')
                        if verifier_feedback["passed_steps"] != []:
                            example_ = {
                                "prompt": SOLUTION_PROMPT_VERIFIER_V2 + SOLUTION_PROMPT_1_SHOT_VERIFIER_USER_V2.format(
                                    puzzle=puzzle_text, reasoning_steps=verifier_feedback["passed_steps"]),
                                "id": puzzle_id,
                                "solution": grid,
                                "puzzle": puzzle_text,
                            }
                    else:
                        #print('Writing example without verification passed')
                        example_ = {
                            "prompt": SOLUTION_PROMPT_VERIFIER_V2 + SOLUTION_PROMPT_1_SHOT_VERIFIER_USER_V2.format(
                                puzzle=puzzle_text, reasoning_steps=verifier_feedback["passed_steps"]),
                            "id": puzzle_id,
                            "solution": grid,
                            "puzzle": puzzle_text,
                        }
                    # Ensure the directory exists
                    #print(f"Example returned {example_}")
                    if example_:
                        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
                        _append_jsonl(feedback_path, example_)

                    #bad = _find_bad_key(example_)
                    #print(f"Found bad key: {bad}")
                    #print(f"DEBUG: In scoring script, write feedback to: {feedback_path}")

                    #print(f"[FEEDBACK]: In scoring script, wrote step feedback to: {feedback_path}")
                    #print(f"example = {example_}")
            except Exception as e:
                #print(f"Failing example = {example_}")
                logger.exception("Crash in Writing Feedback")
                print(f"[FEEDBACK] Failed to write feedback file: {e}")
                print(f"[FEEDBACK] Error details: {str(e)}")
                print(f"[FEEDBACK] Feedback path: {feedback_path}")
                print(f"[FEEDBACK] Record: {json.dumps(example_, ensure_ascii=False, indent=2)}")


    ret = {
        "score": reward,
        "acc": reward,  # Compatible with your old framework
        "acc_score": cell_acc_score,
        "z3_score": z3_score,
        "clue_score": 0.0,      # Constant as 0: Strictly adhered to procedures and not used.
        "total_score": weighted_total,
        "puzzle_acc_score": 1.0 if cell_acc_score == 1.0 else 0.0,
        "cell_acc": cell_acc_score,
        "calculation_time": time.time() - start_time,

        # step fields
        "step_score": step_score,
        "good_steps_cnt": len(step_verif.get("good_steps", [])) if isinstance(step_verif, dict) else 0,
        "bad_steps_cnt": len(step_verif.get("bad_steps", [])) if isinstance(step_verif, dict) else 0,
        "unknown_steps_cnt": len(step_verif.get("unknown_steps", [])) if isinstance(step_verif, dict) else 0,
        "step_verification_enabled": 1 if (isinstance(step_verif, dict) and step_verif.get("enabled", False)) else 0,
        "step_verification_error": 1 if (isinstance(step_verif, dict) and step_verif.get("error")) else 0,
        
        # Return full verification data for in-memory Step-2 training
        "step_verification_data": {
            "good_steps": step_verif.get("good_steps", []) if isinstance(step_verif, dict) else [],
            "bad_steps": step_verif.get("bad_steps", []) if isinstance(step_verif, dict) else [],
            "unknown_steps": step_verif.get("unknown_steps", []) if isinstance(step_verif, dict) else [],
            "puzzle_text": puzzle_text if 'puzzle_text' in locals() else None,
            "step1_json": step1_json if 'step1_json' in locals() else None,
            "verifier_feedback": verifier_feedback if 'verifier_feedback' in locals() else None,
        }
    }
    return ret



if __name__ == "__main__":
    # Minimal smoke test
    example_solution_str = """<answer>{
      "n_houses": 3,
      "attribute_values": {"Name":["Peter","Eric","Arnold"],"Drink":["tea","water","milk"]},
      "parsed_clues": [
        "C1 = set(2,Name,Peter).",
        "C2 = immediately_left_of(Name=Arnold,Drink=water).",
        "C3 = immediately_left_of(Drink=water,Drink=milk)."
      ],
      "parsed_reasoning": [
        "S1 [C1] set(2,Name,Peter).",
        "S2 [C3] not(3,Drink,water).",
        "S3 [C3] not(1,Drink,milk).",
        "S4 [C2] not(3,Name,Arnold).",
        "S5 [C2+C3] set(1,Name,Arnold)."
      ],
      "solution": {
        "header": ["House","Name","Drink"],
        "rows": [["1","Arnold","tea"],["2","Peter","water"],["3","Eric","milk"]]
      }
    }</answer>"""

    example_ground_truth = {
        "header": ["House", "Name", "Drink"],
        "rows": [["1", "Arnold", "tea"], ["2", "Peter", "water"], ["3", "Eric", "milk"]],
    }

    scores = compute_score(
        example_solution_str,
        example_ground_truth,
        extra_info={
            "puzzle_id": "lgp-test-5x6-16",
            "enable_step_feedback": True,
            "step_timeout_ms": 800,
            "base_timeout_ms": 2000,
            "z3_threshold": 0.0,  # Not used, only compatible
            "step_weight": 1.0,
            # If you don't want ACC to affect training, set acc_weight to 0.0 (you can also pass it in the training framework).
        },
        meta={
            "feedback_path": "./reasoning_feedback.jsonl",
            "enable_step_feedback": True,
            "step_timeout_ms": 800,
            "base_timeout_ms": 2000,
            "z3_threshold": 0.0,  # Not used, only compatible
            "step_weight": 1.0,
            # If you don't want ACC to affect training, set acc_weight to 0.0 (you can also pass it in the training framework).
        },
        z3_threshold=0.0,    # Not used, only compatible
        clue_timeout=5.0,    # Not used, only compatible
    )
    #print(json.dumps(scores, indent=2, ensure_ascii=False))

