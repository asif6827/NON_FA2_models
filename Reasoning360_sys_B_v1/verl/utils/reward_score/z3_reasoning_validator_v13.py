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
    \s+(?:is\s+satisfied|holds)\s+because\s+
    (?P<because>.+?)                             # justification
    \s*(?:—|--)\s*verified\s+by\s+the\s+final\s+solution\.\s*$
    """,
    re.VERBOSE,
)



# NEW: Clue verification line format (C<i>)
RE_CLUE_LINE = re.compile(
    r"""^\s*
    C(?P<cid>\d+)\s*                          # C<i>
    \(\s*(?P<constraint>.+?)\s*\)            # (constraint) - non-greedy, allows inner parens
    \s+(?:is\s+satisfied|holds)\s+because\s+
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

def _iter_constraints_from_syntactic_clues(syntactic_clues: Any) -> Tuple[List[Tuple[str, str]], List[str]]:
    """
    Parse syntactic clues into a list of (cid, constraint) pairs.

    Accepts:
      - list[str] lines like:
          "C1: milk == 1."
          "C2: cat + 1 == coffee."
          "milk == 1"   (no id)
      - dict[str,str] mapping id->constraint

    Returns:
      (pairs, errors)
    """
    errors: List[str] = []
    pairs: List[Tuple[str, str]] = []

    if syntactic_clues is None:
        return pairs, errors

    if isinstance(syntactic_clues, dict):
        for k, v in syntactic_clues.items():
            if not isinstance(k, str) or not isinstance(v, str):
                errors.append("syntactic_clues dict must map str -> str.")
                continue
            cid = k.strip() or "C?"
            constraint = v.strip().rstrip(".")
            pairs.append((cid, constraint))
        return pairs, errors

    if not isinstance(syntactic_clues, list):
        return pairs, ["syntactic_clues must be a list[str] or dict[str,str]."]

    for i, line in enumerate(syntactic_clues):
        if not isinstance(line, str):
            errors.append(f"syntactic_clues[{i}] is not a string.")
            continue
        s = line.strip()
        if not s:
            continue

        # Try "Ck: <constraint>."
        mm = re.match(r"^\s*(C\d+)\s*:\s*(.+?)\s*\.?\s*$", s)
        if mm:
            cid = mm.group(1)
            constraint = mm.group(2).strip().rstrip(".")
            pairs.append((cid, constraint))
            continue

        # Fallback: treat as raw constraint with unknown id
        pairs.append((f"C{i+1}", s.rstrip(".")))

    return pairs, errors


def validate_syntactic_clues_against_solution(
    syntactic_clues: Any,
    solution: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Lightly validate syntactic clue constraints against the provided solution table.
    This uses the same "token -> house" extraction as the self-verification checks and
    supports the same common atomic constraint forms:
      - token == k
      - token != k
      - A + d == B
      - A < B / A > B
      - A == B  (same house)
    """
    errors: List[str] = []
    pairs, parse_errs = _iter_constraints_from_syntactic_clues(syntactic_clues)
    errors.extend([f"Clue parse: {e}" for e in parse_errs])

    sol_pos = _extract_solution_positions(solution)
    if not sol_pos:
        errors.append("Solution table could not be parsed (missing header/rows/House).")

    def _get_house(tok: str) -> Optional[int]:
        return sol_pos.get(tok)

    checked = []
    for cid, constraint in pairs:
        c = _canon_constraint(constraint)
        checked.append({"cid": cid, "constraint": constraint})

        if not sol_pos:
            continue

        # token == k
        mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)==(\d+)", c)
        if mm:
            tok, k = mm.group(1), int(mm.group(2))
            ht = _get_house(tok)
            if ht is None or ht != k:
                errors.append(f"{cid}: solution does not support `{tok} == {k}` (solution has {tok} at {ht}).")
            continue

        # token != k
        mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)!=(\d+)", c)
        if mm:
            tok, k = mm.group(1), int(mm.group(2))
            ht = _get_house(tok)
            if ht is None:
                errors.append(f"{cid}: token `{tok}` not found in solution table.")
            elif ht == k:
                errors.append(f"{cid}: solution contradicts `{tok} != {k}` (it is at house {k}).")
            continue

        # A + d == B
        mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\+(\d+)==([A-Za-z_][A-Za-z0-9_]*)", c)
        if mm:
            a, d, b = mm.group(1), int(mm.group(2)), mm.group(3)
            ha, hb = _get_house(a), _get_house(b)
            if ha is None or hb is None:
                errors.append(f"{cid}: adjacency tokens not found in solution: `{a}`, `{b}`.")
            elif ha + d != hb:
                errors.append(f"{cid}: solution contradicts `{a}+{d}=={b}` (houses are {ha} and {hb}).")
            continue

        # A < B or A > B
        mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(<|>)([A-Za-z_][A-Za-z0-9_]*)", c)
        if mm:
            a, op, b = mm.group(1), mm.group(2), mm.group(3)
            ha, hb = _get_house(a), _get_house(b)
            if ha is None or hb is None:
                errors.append(f"{cid}: ordering tokens not found in solution: `{a}`, `{b}`.")
            else:
                ok = (ha < hb) if op == "<" else (ha > hb)
                if not ok:
                    errors.append(f"{cid}: solution contradicts `{a}{op}{b}` (houses are {ha} and {hb}).")
            continue

        # A == B (same house), excluding token==k
        mm = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)==([A-Za-z_][A-Za-z0-9_]*)", c)
        if mm and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*==\d+", c):
            a, b = mm.group(1), mm.group(2)
            ha, hb = _get_house(a), _get_house(b)
            if ha is None or hb is None:
                errors.append(f"{cid}: equality tokens not found in solution: `{a}`, `{b}`.")
            elif ha != hb:
                errors.append(f"{cid}: solution contradicts `{a}=={b}` (houses are {ha} and {hb}).")
            continue

        # If we can't parse the constraint form, don't fail hard; just report.
        errors.append(f"{cid}: unsupported clue constraint form (skipped): `{constraint}`")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "checked": checked,
    }


def validate_solution_against_ground_truth(
    solution: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate that the solution table matches the ground truth table.

    We compare using a token->house map (ignoring header ordering beyond requiring 'House').
    This makes the check robust to column ordering differences.
    """
    errors: List[str] = []

    sol_pos = _extract_solution_positions(solution)
    gt_pos = _extract_solution_positions(ground_truth)

    if not sol_pos:
        errors.append("Solution table could not be parsed (missing header/rows/House).")
    if not gt_pos:
        errors.append("Ground-truth table could not be parsed (missing header/rows/House).")

    if sol_pos and gt_pos:
        # exact match on all tokens present in either map
        all_toks = sorted(set(sol_pos.keys()) | set(gt_pos.keys()))
        for tok in all_toks:
            hs = sol_pos.get(tok)
            hg = gt_pos.get(tok)
            if hs != hg:
                errors.append(f"Ground-truth mismatch: `{tok}` is at house {hs} in solution but house {hg} in ground_truth.")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
    }


# ============================================================
# Four-input validator (clues + reasoning + solution + ground_truth)
# ============================================================
def validate_reasoning_and_solution(
    syntactic_clues: Any,
    reasoning_steps: Dict[str, Any],
    solution: Dict[str, Any],
    ground_truth: Dict[str, Any],
    *,
    w_reasoning: float = 0.5,
    w_clues: float = 0.25,
    w_ground_truth: float = 0.25,
    allow_summary_line: bool = True,
    forbid_trivial_tautologies: bool = True,
    require_house_mentions_for_atomic: bool = True,
    require_all_steps_covered: bool = False,
) -> Dict[str, Any]:
    """
    Updated validator to accept FOUR inputs:
      (i)  syntactic_clues
      (ii) reasoning_steps
      (iii) solution
      (iv) ground_truth

    It produces three validation reports:
      A) reasoning_steps vs solution (Self_Verification alignment + light solution checks)
      B) syntactic_clues vs solution (light checks for common constraint forms)
      C) solution vs ground_truth (token->house agreement)

    Final reward is a weighted combination of the three sub-rewards.
    """

    # A) reuse the existing two-input logic by treating `reasoning_steps` as the old `reasoning`
    reasoning_result = _validate_reasoning_steps_against_solution(
        reasoning_steps,
        solution,
        syntactic_clues=syntactic_clues,
        allow_summary_line=allow_summary_line,
        forbid_trivial_tautologies=forbid_trivial_tautologies,
        require_house_mentions_for_atomic=require_house_mentions_for_atomic,
        require_all_steps_covered=require_all_steps_covered,
    )

    # B) clue validation
    clues_result = validate_syntactic_clues_against_solution(syntactic_clues, solution)

    # C) ground-truth validation
    gt_result = validate_solution_against_ground_truth(solution, ground_truth)

    # Combine reward
    w_sum = max(1e-9, float(w_reasoning + w_clues + w_ground_truth))
    w_reasoning_n = float(w_reasoning) / w_sum
    w_clues_n = float(w_clues) / w_sum
    w_gt_n = float(w_ground_truth) / w_sum

    r_reasoning = reward_from_validation(reasoning_result)
    r_clues = 1.0 if clues_result.get("ok") else -1.0
    r_gt = 1.0 if gt_result.get("ok") else -2.0

    reward = (w_reasoning_n * r_reasoning) + (w_clues_n * r_clues) + (w_gt_n * r_gt)

    ok = bool(reasoning_result.get("ok") and clues_result.get("ok") and gt_result.get("ok"))

    return {
        "reward": float(reward),
        "ok": ok,
        "reasoning_validation": reasoning_result,
        "clues_validation": clues_result,
        "ground_truth_validation": gt_result,
        "weights": {
            "w_reasoning": w_reasoning_n,
            "w_clues": w_clues_n,
            "w_ground_truth": w_gt_n,
        },
    }


def _validate_reasoning_steps_against_solution(
    reasoning: Dict[str, Any],
    solution: Dict[str, Any],
    *,
    syntactic_clues: Any = None,
    allow_summary_line: bool = True,
    forbid_trivial_tautologies: bool = True,
    require_house_mentions_for_atomic: bool = True,
    require_all_steps_covered: bool = False,
) -> Dict[str, Any]:
    """
    (Internal) the original two-input validator logic, kept intact but renamed.
    Returns the same shape as before (minus top-level reward).
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

    # Canonical clue constraints for matching C<i> self-verification lines
    clue_pairs, clue_parse_errs = _iter_constraints_from_syntactic_clues(syntactic_clues)
    for e in clue_parse_errs:
        errors.append(f"Clue parse: {e}")

    clue_constraints_canon: Dict[str, str] = {}
    clue_constraints_raw: Dict[str, str] = {}
    for cid, cstr in clue_pairs:
        clue_constraints_canon[cid] = _canon_constraint(cstr)
        clue_constraints_raw[cid] = cstr

    # Pull self-verification list

    for idx, line in enumerate(self_ver):
        if not isinstance(line, str):
            errors.append(f"Self_Verification[{idx}]: not a string.")
            continue

        if allow_summary_line and RE_SUMMARY_LINE.match(line):
            summary_present = True
            continue

        # Try S-step Self_Verification line
        ms = RE_STEP_LINE.match(line)
        if ms:
            sid = int(ms.group("sid"))
            constraint = ms.group("constraint").strip()
            because = ms.group("because").strip()
            kind = "S"
        else:
            # Try C-clue Self_Verification line
            mc = RE_CLUE_LINE.match(line)
            if mc:
                cid_num = int(mc.group("cid"))
                cid = f"C{cid_num}"
                constraint = mc.group("constraint").strip()
                because = mc.group("because").strip()
                kind = "C"
            else:
                errors.append(
                    f"Self_Verification[{idx}]: does not match required format "
                    f"`S<k> (<constraint>) holds/is satisfied because ... — verified by the final solution.` "
                    f"or `C<i> (<constraint>) holds/is satisfied because ... — verified by the final solution.`"
                )
                continue

        # Forbid trivial tautologies in "because"
        if forbid_trivial_tautologies:
            t = RE_TRIVIAL_TAUTOLOGY.search(because)
            if t and _canon_constraint(t.group("lemma")) == _canon_constraint(t.group("lemma2")):
                errors.append(f"Self_Verification[{idx}]: trivial tautology in justification: `{t.group(0).strip()}`")

        # Ensure constraint matches parsed source constraint (reasoning step S<k> or clue C<i>)
        if kind == "S":
            key = f"S{sid}"
            if key not in step_constraints_canon:
                errors.append(f"Self_Verification[{idx}]: references {key} but it is missing from parsed reasoning steps.")
            else:
                if _canon_constraint(constraint) != step_constraints_canon[key]:
                    errors.append(
                        f"Self_Verification[{idx}]: constraint mismatch for {key}. "
                        f"Self_Ver has `{constraint}` but reasoning step has `{parsed_steps[key]}`."
                    )
        else:
            # kind == "C"
            if cid not in clue_constraints_canon:
                errors.append(f"Self_Verification[{idx}]: references {cid} but it is missing from syntactic_clues.")
            else:
                if _canon_constraint(constraint) != clue_constraints_canon[cid]:
                    errors.append(
                        f"Self_Verification[{idx}]: constraint mismatch for {cid}. "
                        f"Self_Ver has `{constraint}` but syntactic_clues has `{clue_constraints_raw[cid]}`."
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
            if kind == "C":
                tmp = validate_syntactic_clues_against_solution([f"{cid}: {constraint}."], solution)
                if not tmp.get("ok"):
                    for e in tmp.get("errors", []):
                        errors.append(f"Self_Verification[{idx}]: {e}")
                # Still continue to allow generic parsing below
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

            # A + d == B
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

        if kind == "S":
            self_ver_parsed.append({"kind": "S", "sid": sid, "constraint": constraint, "because": because, "raw": line})
            seen_steps.append(sid)
        else:
            self_ver_parsed.append({"kind": "C", "cid": cid, "constraint": constraint, "because": because, "raw": line})

    if require_all_steps_covered and step_constraints_canon:
        expected = sorted(int(k[1:]) for k in step_constraints_canon.keys() if k[1:].isdigit())
        got = sorted(set(seen_steps))
        if expected != got:
            errors.append(f"Self_Verification does not cover all steps. expected={expected} got={got}")

    if seen_steps and sorted(seen_steps) != seen_steps:
        errors.append("Self_Verification step ids are not in increasing order.")

    return {
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
    syntactic_clues = [
        "C1: Alice == cat.",
        "C2: cat + 1 == coffee."
    ]

    reasoning = [
        "S1: Alice == cat.",
        "S2: cat + 1 == coffee.",
        "S3: And(cat == 1, coffee == 2)."
    ]

    solution = {
        "header": ["House", "Name", "Pet", "Drink"],
        "rows": [
            ["1", "Alice", "cat", "tea"],
            ["2", "Bob", "dog", "coffee"]
        ]
    }

    res = validate_reasoning_and_solution(syntactic_clues, reasoning, solution)
    for k,v in res.items():
        print(f"{k}: {v}")