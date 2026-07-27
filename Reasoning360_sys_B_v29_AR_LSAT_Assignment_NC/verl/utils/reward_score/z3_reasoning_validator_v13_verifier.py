import re
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# Regex patterns for Self_Verification (STRICT "aligned" format)
# ============================================================
# Accepted example:
#   "S4 (cat + 1 == coffee) is satisfied because cat is in house 2 and coffee is in house 3, and 2+1=3. — verified by the final solution."
#
# Notes:
# - constraint must be inside parentheses after S<k>
# - line must end with: "— verified by the final solution."
# - dash is an em-dash "—" in the spec, but we accept ASCII "--" too.

RE_STEP_LINE = re.compile(
    r"""^\s*
    S(?P<sid>\d+)\s*                          # S<k>
    \(\s*(?P<constraint>.+?)\s*\)            # (constraint) - non-greedy, allows inner parens
    \s+is\s+satisfied\s+because\s+
    (?P<because>.+?)                             # justification
    \s*(?:—|--)\s*verified\s+by\s+the\s+final\s+solution\.\s*$
    """,
    re.VERBOSE,
)


# Optional final summary line (single line, no S<k>)
RE_SUMMARY_LINE = re.compile(
    r"""^\s*
    (All\s+steps\s+S(?P<start>\d+)–S(?P<end>\d+)\s+are\s+consistent\s+with\s+the\s+final\s+solution\s+table,?\s+and\s+the\s+solution\s+satisfies\s+all\s+constraints\s+derived\s+from\s+the\s+clues\s+and\s+reasoning\.
    \s*(?:—|--)\s*verified\s+by\s+the\s+final\s+solution\.)
    \s*$""",
    re.VERBOSE,
)

# Disallow trivial tautology templates inside the "because" part:
# "X == 2 implies X == 2" or "X != 1 implies X != 1"
RE_TRIVIAL_TAUTOLOGY = re.compile(
    r"""(?ix)
    \b(?P<lemma>[A-Za-z_][A-Za-z0-9_]*\s*(?:==|!=)\s*\d+)\b
    \s*implies\s*
    \b(?P<lemma2>[A-Za-z_][A-Za-z0-9_]*\s*(?:==|!=)\s*\d+)\b
    """
)

# ============================================================
# Parsing syntactic steps from reasoning (11 keys)
# ============================================================
# Expected syntactic step line format:
#   S17: Or(milk == 1, milk == 2). [S12]
# NL lines are allowed and ignored. Self_Verification is NL-only by contract.

RE_SYN_STEP_LINE = re.compile(
    r"""^\s*
    S(?P<sid>\d+)\s*:\s*
    (?P<constraint>.+?)
    \s*\.\s*
    \[\s*(?P<evidence>[^\]]*)\s*\]
    \s*$""",
    re.VERBOSE,
)

def _canon_constraint(s: str) -> str:
    """Remove whitespace for stable constraint matching."""
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s

def extract_syntactic_steps_from_reasoning(reasoning: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """
    Extract {"S1": "<constraint>", ...} from a reasoning dict with the 11 category keys.
    NL entries are ignored. Self_Verification is skipped.

    Returns:
      (steps_dict, errors)
    """
    errors: List[str] = []
    steps: Dict[str, str] = {}

    if not isinstance(reasoning, dict):
        return {}, ["reasoning is not a dict"]

    for cat, entries in reasoning.items():
        if cat == "Self_Verification":
            continue
        if not isinstance(entries, list):
            errors.append(f"Category {cat!r} is not a list.")
            continue

        for i, line in enumerate(entries):
            if not isinstance(line, str):
                continue
            line_s = line.strip()
            if not line_s.startswith("S"):
                continue

            m = RE_SYN_STEP_LINE.match(line_s)
            if not m:
                errors.append(f"Unparseable syntactic step in {cat}[{i}]: {line_s!r}")
                continue

            sid = int(m.group("sid"))
            constraint = m.group("constraint").strip()
            key = f"S{sid}"

            if key in steps:
                if _canon_constraint(steps[key]) != _canon_constraint(constraint):
                    errors.append(
                        f"Duplicate step {key} with conflicting constraints: {steps[key]!r} vs {constraint!r}"
                    )
                continue

            steps[key] = constraint

    return steps, errors

# ============================================================
# Solution table helpers
# ============================================================
def _extract_solution_positions(solution: Dict[str, Any]) -> Dict[str, int]:
    """
    Build token -> house_index map from the solution table.
    Expects solution["rows"] like: [["1", "Bob", "dog", "milk"], ...]
    and header includes "House".
    """
    header = solution.get("header")
    rows = solution.get("rows")
    if not isinstance(header, list) or not isinstance(rows, list):
        return {}

    try:
        house_col = header.index("House")
    except ValueError:
        return {}

    pos: Dict[str, int] = {}
    for r in rows:
        if not isinstance(r, list) or len(r) != len(header):
            continue
        try:
            h = int(str(r[house_col]).strip())
        except Exception:
            continue
        for j, val in enumerate(r):
            if j == house_col:
                continue
            tok = str(val).strip()
            if tok:
                pos[tok] = h
    return pos

RE_HOUSE_MENTION = re.compile(r"(?i)\bhouse\s+(?P<h>\d+)\b")

# ============================================================
# Two-input validator
# ============================================================
def validate_reasoning_and_solution(
    reasoning: Dict[str, Any],
    solution: Dict[str, Any],
    *,
    allow_summary_line: bool = True,
    forbid_trivial_tautologies: bool = True,
    require_house_mentions_for_atomic: bool = True,
    require_all_steps_covered: bool = False,
) -> Dict[str, Any]:
    """
    Validate ONLY using two inputs:
      1) reasoning dict (must include "Self_Verification" key as last key by prompt contract)
      2) solution table dict

    What it checks:
    - Parses syntactic steps from reasoning (S<k>: ... . [evidence]) across all categories except Self_Verification.
    - Parses Self_Verification lines in the STRICT aligned format:
        S<k> (<constraint>) is satisfied because ... — verified by the final solution.
    - Ensures each Self_Verification constraint matches the parsed S<k> constraint (canonicalized).
    - Does LIGHT checks against solution for simple constraint forms (==, !=, +k==, <, >, token==token).
    - Lets NL entries in categories be ignored automatically.

    Returns:
      {
        "ok": bool,
        "errors": [...],
        "reasoning_parse_errors": [...],
        "parsed_steps": {"S1":"...", ...},
        "self_ver_parsed": [{"sid":1,"constraint":"...","because":"..."}...],
        "seen_steps": [...],
        "summary_present": bool,
      }
    """
    errors: List[str] = []
    self_ver_parsed: List[Dict[str, Any]] = []
    seen_steps: List[int] = []
    summary_present = False

    parsed_steps, parse_errors = extract_syntactic_steps_from_reasoning(reasoning)
    reasoning_parse_errors = [f"Reasoning parse: {pe}" for pe in parse_errors]
    errors.extend(reasoning_parse_errors)

    step_constraints_canon = {k: _canon_constraint(v) for k, v in parsed_steps.items()}

    sol_pos = _extract_solution_positions(solution)

    # Pull self-verification list
    self_ver = reasoning.get("Self_Verification", [])
    if not isinstance(self_ver, list):
        errors.append("Self_Verification is missing or not a list.")
        self_ver = []

    for idx, line in enumerate(self_ver):
        if not isinstance(line, str):
            errors.append(f"Self_Verification[{idx}]: not a string.")
            continue

        if allow_summary_line and RE_SUMMARY_LINE.match(line):
            summary_present = True
            continue

        m = RE_STEP_LINE.match(line)
        if not m:
            errors.append(
                f"Self_Verification[{idx}]: does not match required format "
                f'`S<k> (<constraint>) is satisfied because ... — verified by the final solution.`'
            )
            continue

        sid = int(m.group("sid"))
        constraint = m.group("constraint").strip()
        because = m.group("because").strip()

        # Forbid trivial tautologies in "because"
        if forbid_trivial_tautologies:
            t = RE_TRIVIAL_TAUTOLOGY.search(because)
            if t and _canon_constraint(t.group("lemma")) == _canon_constraint(t.group("lemma2")):
                errors.append(f"Self_Verification[{idx}]: trivial tautology in justification: `{t.group(0).strip()}`")

        # Ensure constraint matches parsed reasoning step constraint
        key = f"S{sid}"
        if key not in step_constraints_canon:
            errors.append(f"Self_Verification[{idx}]: references {key} but it is missing from parsed reasoning steps.")
        else:
            if _canon_constraint(constraint) != step_constraints_canon[key]:
                errors.append(
                    f"Self_Verification[{idx}]: constraint mismatch for {key}. "
                    f"Self_Ver has `{constraint}` but reasoning step has `{parsed_steps[key]}`."
                )

        # Require house mentions for atomic constraints (discourages vague justifications)
        if require_house_mentions_for_atomic:
            is_atomic = (
                "And(" not in constraint
                and "Or(" not in constraint
                and "Not(" not in constraint
                and "Implies(" not in constraint
            )
            if is_atomic and not RE_HOUSE_MENTION.search(because):
                errors.append(
                    f"Self_Verification[{idx}]: atomic constraint verification should mention house indices "
                    f'(e.g., "house 2"), but none found.'
                )

        # LIGHT linkage check against solution for common atomic constraint forms
        if sol_pos:
            c = _canon_constraint(constraint)

            def _get_house(tok: str) -> Optional[int]:
                return sol_pos.get(tok)

            # token == k
            mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)==(\d+)", c)
            if mm:
                tok, k = mm.group(1), int(mm.group(2))
                ht = _get_house(tok)
                if ht is None or ht != k:
                    errors.append(f"Self_Verification[{idx}]: solution does not support `{tok} == {k}` (solution has {tok} at {ht}).")

            # token != k
            mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)!=(\d+)", c)
            if mm:
                tok, k = mm.group(1), int(mm.group(2))
                ht = _get_house(tok)
                if ht is None:
                    errors.append(f"Self_Verification[{idx}]: token `{tok}` not found in solution table.")
                elif ht == k:
                    errors.append(f"Self_Verification[{idx}]: solution contradicts `{tok} != {k}` (it is at house {k}).")

            # A + d == B  (also supports +2, +3, etc.)
            mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\+(\d+)==([A-Za-z_][A-Za-z0-9_]*)", c)
            if mm:
                a, d, b = mm.group(1), int(mm.group(2)), mm.group(3)
                ha, hb = _get_house(a), _get_house(b)
                if ha is None or hb is None:
                    errors.append(f"Self_Verification[{idx}]: adjacency tokens not found in solution: `{a}`, `{b}`.")
                elif ha + d != hb:
                    errors.append(f"Self_Verification[{idx}]: solution contradicts `{a}+{d}=={b}` (houses are {ha} and {hb}).")

            # A < B or A > B
            mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(<|>)([A-Za-z_][A-Za-z0-9_]*)", c)
            if mm:
                a, op, b = mm.group(1), mm.group(2), mm.group(3)
                ha, hb = _get_house(a), _get_house(b)
                if ha is None or hb is None:
                    errors.append(f"Self_Verification[{idx}]: ordering tokens not found in solution: `{a}`, `{b}`.")
                else:
                    ok = (ha < hb) if op == "<" else (ha > hb)
                    if not ok:
                        errors.append(f"Self_Verification[{idx}]: solution contradicts `{a}{op}{b}` (houses are {ha} and {hb}).")

            # A == B (same house), excluding token==k
            mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)==([A-Za-z_][A-Za-z0-9_]*)", c)
            if mm and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*==\d+", c):
                a, b = mm.group(1), mm.group(2)
                ha, hb = _get_house(a), _get_house(b)
                if ha is None or hb is None:
                    errors.append(f"Self_Verification[{idx}]: equality tokens not found in solution: `{a}`, `{b}`.")
                elif ha != hb:
                    errors.append(f"Self_Verification[{idx}]: solution contradicts `{a}=={b}` (houses are {ha} and {hb}).")

        self_ver_parsed.append({"sid": sid, "constraint": constraint, "because": because, "raw": line})
        seen_steps.append(sid)

    # Optionally require coverage of all parsed steps
    if require_all_steps_covered and step_constraints_canon:
        expected = sorted(int(k[1:]) for k in step_constraints_canon.keys() if k[1:].isdigit())
        got = sorted(set(seen_steps))
        if expected != got:
            errors.append(f"Self_Verification does not cover all steps. expected={expected} got={got}")

    # Optional ordering check
    if seen_steps and sorted(seen_steps) != seen_steps:
        errors.append("Self_Verification step ids are not in increasing order.")

    result = {
        "ok": len(errors) == 0,
        "errors": errors,
        "reasoning_parse_errors": reasoning_parse_errors,
        "parsed_steps": parsed_steps,
        "self_ver_parsed": self_ver_parsed,
        "seen_steps": seen_steps,
        "summary_present": summary_present,
    }

    reward = reward_from_validation(result)

    return {
        "reward": reward,
        "ok": len(errors) == 0,
        "errors": errors,
        "reasoning_parse_errors": reasoning_parse_errors,
        "parsed_steps": parsed_steps,
        "self_ver_parsed": self_ver_parsed,
        "seen_steps": seen_steps,
        "summary_present": summary_present,
    }

# ============================================================
# Reward helper (simple, deterministic)
# ============================================================
def reward_from_validation(result: Dict[str, Any]) -> float:
    """
    Suggested reward shaping:
      +1.0 : fully valid
      -2.0 : any solution contradiction or constraint mismatch
      -1.0 : parsing failures / missing self-ver / missing steps
      -0.5 : other format errors
    """
    if result.get("ok"):
        return 1.0

    errs = " | ".join(result.get("errors", []))

    hard = (
        "solution contradicts" in errs
        or "constraint mismatch" in errs
        or "does not support" in errs
    )
    if hard:
        return -2.0

    parseish = (
        "Unparseable syntactic step" in errs
        or "missing from parsed reasoning steps" in errs
        or "Self_Verification is missing" in errs
        or "does not match required format" in errs
    )
    if parseish:
        return -1.0

    return -0.5


# ============================================================
# Running examples (1 positive, 2 negative)
# ============================================================
if __name__ == "__main__":
    # -----------------------
    # Example A (POSITIVE)
    # -----------------------
    reasoning_A = {
        "Abs_Placement": ["S1: milk == 1. [C1]"],
        "Direct_Equality": ["S2: Bob == milk. [C2]"],
        "Directed_Adjacency": [],
        "Structural_Positioning": [],
        "Domain_Restriction": [],
        "Exclusion": [],
        "Propagation": [],
        "Forced_Resolution": [],
        "Disjunction": ["S3: Or(milk == 1, milk == 2). [S1]"],
        "Case_Split": [],
        "Self_Verification": [
            "S1 (milk == 1) is satisfied because milk appears in house 1 in the final table. — verified by the final solution.",
            "S2 (Bob == milk) is satisfied because Bob and milk are both in house 1 in the final table. — verified by the final solution.",
            "S3 (Or(milk == 1, milk == 2)) is satisfied because milk is in house 1 in the final table, satisfying the disjunction. — verified by the final solution.",
        ],
    }
    solution_A = {
        "header": ["House", "Name", "Drink"],
        "rows": [
            ["1", "Bob", "milk"],
            ["2", "Alice", "tea"],
        ],
    }
    resA = validate_reasoning_and_solution(reasoning_A, solution_A)
    print(resA)

    # -----------------------
    # Example B (NEGATIVE: constraint mismatch)
    # Self_Ver claims S1 is milk==2 but reasoning has milk==1
    # -----------------------
    reasoning_B = {
        "Abs_Placement": ["S1: milk == 1. [C1]"],
        "Direct_Equality": [],
        "Directed_Adjacency": [],
        "Structural_Positioning": [],
        "Domain_Restriction": [],
        "Exclusion": [],
        "Propagation": [],
        "Forced_Resolution": [],
        "Disjunction": [],
        "Case_Split": [],
        "Self_Verification": [
            "S1 (milk == 2) is satisfied because milk appears in house 2 in the final table. — verified by the final solution.",
        ],
    }
    solution_B = {
        "header": ["House", "Drink"],
        "rows": [["1", "milk"], ["2", "tea"]],
    }
    resB = validate_reasoning_and_solution(reasoning_B, solution_B)
    print(resB)

    # -----------------------
    # Example C (NEGATIVE: solution contradiction)
    # Reasoning+Self_Ver claim cat+1==coffee but solution violates it.
    # -----------------------
    reasoning_C = {
        "Abs_Placement": [],
        "Direct_Equality": [],
        "Directed_Adjacency": ["S1: cat + 1 == coffee. [C3]"],
        "Structural_Positioning": [],
        "Domain_Restriction": [],
        "Exclusion": [],
        "Propagation": [],
        "Forced_Resolution": [],
        "Disjunction": [],
        "Case_Split": [],
        "Self_Verification": [
            "S1 (cat + 1 == coffee) is satisfied because cat is in house 2 and coffee is in house 3, and 2+1=3. — verified by the final solution.",
        ],
    }
    solution_C = {
        "header": ["House", "Pet", "Drink"],
        "rows": [
            ["1", "dog", "coffee"],
            ["2", "cat", "tea"],  # coffee is NOT in house 3 -> violates cat+1==coffee
            ["3", "fish", "milk"],
        ],
    }
    resC = validate_reasoning_and_solution(reasoning_C, solution_C)
    print(resC)
