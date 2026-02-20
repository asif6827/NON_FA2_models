# z3_reward_v2.py
# pip install z3-solver

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import re
import traceback
import logging
import sys

from z3 import Solver, Int, Distinct, And, Abs, Not, sat, unsat

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger(__name__)

# ----------------------------
# 0) Helpers
# ----------------------------

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def clamp(x: float, lo: float, hi: float) -> float:
    try:
        x = float(x)
    except Exception:
        return lo
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def smoothstep(t: float) -> float:
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def print_exc():
    traceback.print_exc()


# ----------------------------
# 1) Value normalization (for matching across hyphens/plurals/iphone etc.)
# ----------------------------

STOPWORDS = {
    "the", "a", "an", "person", "who", "whose", "is", "in", "of", "to", "and",
    "somewhere", "directly", "immediately", "left", "right", "next", "between",
    "house", "loves", "likes", "keeps", "owns", "uses", "has", "drinks", "eats",
    "enjoys", "prefers", "wears", "with", "named", "name", "lover", "keeper",
    "arrangement", "type", "favorite", "favourite",
}

ALIASES = {
    "british": "brit",
    "danish": "dane",
    "i phone": "iphone",
    "i-phone": "iphone",
    "iphones": "iphone",
}

ORDINAL = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}

def _apply_aliases(s: str) -> str:
    s2 = s
    for k, v in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        s2 = re.sub(rf"\b{re.escape(k)}\b", v, s2)
    return s2

def _singularize(tok: str) -> str:
    # light singularization to fix horses->horse, cars->car
    if len(tok) >= 4 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok

def canon_value(x: Any) -> str:
    """
    Produces a compact comparable key:
      "Ford F-150" -> "fordf150"
      "Tesla Model 3" -> "teslamodel3"
      "iPhone 13" -> "iphone13"
      "keeps horses" -> "horse" (best effort by stopword removal)
    """
    s = str(x).strip().lower()
    s = s.strip("`'\"")
    s = _apply_aliases(s)

    # punctuation -> spaces
    s = re.sub(r"[\-_/.,:;()\[\]{}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    toks = []
    for t in s.split(" "):
        if not t:
            continue
        if t in STOPWORDS:
            continue
        t = _apply_aliases(t)
        t = _singularize(t)
        toks.append(t)

    if not toks:
        return re.sub(r"[^a-z0-9]+", "", s)

    # compact join (iphone + 13 => iphone13)
    return "".join(re.sub(r"[^a-z0-9]+", "", t) for t in toks)


# ----------------------------
# 2) Table utilities
# ----------------------------

def normalize_table(sol: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures:
      - header: list[str]
      - rows: list[list[str]]
    Adds House column if missing (by row order).
    Tries to move House to col0.
    """
    header = sol.get("header")
    if header is None or len(header) == 0:
        header = []
    rows = sol.get("rows")
    if rows is None or len(rows) == 0:
        rows = []
    header = [str(h).strip() for h in header]

    norm_rows: List[List[str]] = []
    for r in rows:
        if isinstance(r, list):
            norm_rows.append([str(x).strip() for x in r])
        else:
            norm_rows.append([str(r).strip()])

    if not header and norm_rows:
        header = [f"col{i}" for i in range(len(norm_rows[0]))]

    # find house column
    hl = [h.strip().lower() for h in header]
    house_idx = None
    for i, h in enumerate(hl):
        if h in ("house", "home", "position", "index"):
            house_idx = i
            break

    def col_all_int(ci: int) -> bool:
        try:
            for r in norm_rows:
                int(r[ci])
            return True
        except Exception:
            return False

    if house_idx is None:
        if norm_rows and len(norm_rows[0]) >= 1 and col_all_int(0):
            house_idx = 0
            header[0] = "House"
        else:
            header = ["House"] + header
            norm_rows = [[str(i + 1)] + r for i, r in enumerate(norm_rows)]
            house_idx = 0

    # move house to front
    if house_idx != 0:
        header = [header[house_idx]] + [h for i, h in enumerate(header) if i != house_idx]
        new_rows = []
        for r in norm_rows:
            if house_idx < len(r):
                new_rows.append([r[house_idx]] + [v for i, v in enumerate(r) if i != house_idx])
            else:
                new_rows.append([str(len(new_rows) + 1)] + r)
        norm_rows = new_rows

    # coerce house numbers if possible; else reassign
    ok = True
    for r in norm_rows:
        try:
            int(r[0])
        except Exception:
            ok = False
            break
    if not ok:
        for i, r in enumerate(norm_rows):
            r[0] = str(i + 1)

    return {"header": header, "rows": norm_rows}

def structure_score(sol: Dict[str, Any]) -> float:
    try:
        header = sol["header"]
        rows = sol["rows"]
        if not header or not rows:
            return 0.0
        m = len(header)
        good = sum(1 for r in rows if isinstance(r, list) and len(r) == m)
        return clamp01(good / max(1, len(rows)))
    except Exception:
        return 0.0


# ----------------------------
# 3) Build Z3 model from GT domains (not predicted values)
# ----------------------------

def sanitize_sym(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "v"

def build_domains_from_gt(gt: Dict[str, Any]) -> Tuple[int, List[str], Dict[str, List[str]]]:
    """
    Returns:
      n_houses, attrs (excluding House), domains[attr] = list of values (original strings)
    """
    gt = normalize_table(gt)
    header = gt["header"]
    rows = gt["rows"]
    n = len(rows)

    attrs = header[1:]
    domains: Dict[str, List[str]] = {a: [] for a in attrs}

    seen: Dict[str, set] = {a: set() for a in attrs}
    for r in rows:
        for a, v in zip(attrs, r[1:]):
            vv = str(v).strip()
            if vv not in seen[a]:
                seen[a].add(vv)
                domains[a].append(vv)
    return n, attrs, domains

def build_vocab_index(domains: Dict[str, List[str]]) -> Tuple[List[Tuple[int, str, str, str]], Dict[str, Tuple[str, str]], float]:
    """
    Builds an index for matching values in text:
      patterns: list of (start_priority, regex, attr, original_value, canon_key) as regex compiled later
      canon_map: canon_key -> (attr, original_value), only if unambiguous
      ambiguity penalty used to compute domain_cov reliably
    """
    # canon_key -> list of (attr,value)
    buckets: Dict[str, List[Tuple[str, str]]] = {}
    for attr, vals in domains.items():
        for v in vals:
            ck = canon_value(v)
            buckets.setdefault(ck, []).append((attr, v))

    canon_map: Dict[str, Tuple[str, str]] = {}
    ambiguous = 0
    total = 0
    for ck, lst in buckets.items():
        total += 1
        if len(lst) == 1:
            canon_map[ck] = lst[0]
        else:
            ambiguous += 1

    # a rough "domain uniqueness" score (higher is better)
    uniq_score = 1.0 - (ambiguous / max(1, total))

    # build regex patterns to find values in raw clue text robustly
    patterns: List[Tuple[int, re.Pattern, str, str, str]] = []
    for attr, vals in domains.items():
        for v in vals:
            raw = str(v).strip().lower()
            raw = _apply_aliases(raw)
            # tokens split on non-alnum
            tokens = re.findall(r"[a-z0-9]+", raw)
            if not tokens:
                continue

            # allow optional spaces/hyphens between tokens
            # e.g. ford f 150 matches ford f-150 / ford f150 / ford f 150
            sep = r"[\s\-]*"
            pat = r"\b" + sep.join(map(re.escape, tokens)) + r"\b"
            compiled = re.compile(pat, re.IGNORECASE)

            # longer tokens should be matched first (priority)
            priority = -len(tokens)
            ck = canon_value(v)
            patterns.append((priority, compiled, attr, v, ck))

            # also add a singular-last-token variant for simple plurals (horses->horse)
            last = tokens[-1]
            if last.endswith("s") and len(last) >= 4:
                tokens2 = tokens[:-1] + [last[:-1]]
                pat2 = r"\b" + sep.join(map(re.escape, tokens2)) + r"\b"
                patterns.append((priority - 1, re.compile(pat2, re.IGNORECASE), attr, v, ck))

    # sort: longer/more-specific first
    patterns.sort(key=lambda x: x[0])
    return patterns, canon_map, uniq_score

def build_z3_from_domains(n: int, domains: Dict[str, List[str]]) -> Tuple[Solver, Dict[str, Dict[str, Any]]]:
    """
    Z[attr][canon_key_of_value] -> Int var for house index.
    Uses canon_value(value) keys so clue matching can map to vars.
    """
    S = Solver()
    Z: Dict[str, Dict[str, Any]] = {}
    for attr, vals in domains.items():
        Z[attr] = {}
        for v in vals:
            ck = canon_value(v)
            Z[attr][ck] = Int(f"{sanitize_sym(attr)}__{sanitize_sym(ck)}")
        if len(Z[attr]) > 1:
            S.add(Distinct(*Z[attr].values()))
        for vv in Z[attr].values():
            S.add(vv >= 1, vv <= n)
    return S, Z


# ----------------------------
# 4) Clue parsing using GT vocab
# ----------------------------

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = _apply_aliases(s)
    s = re.sub(r"[`\"']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def extract_house_num(text: str) -> Optional[int]:
    t = normalize_text(text)

    # "in house 2"
    m = re.search(r"\bhouse\s+(\d+)\b", t)
    if m:
        return int(m.group(1))

    # "in the second house"
    m2 = re.search(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+house\b", t)
    if m2:
        return ORDINAL.get(m2.group(1))

    # "in the 2nd house"
    m3 = re.search(r"\b(\d+)(st|nd|rd|th)\s+house\b", t)
    if m3:
        return int(m3.group(1))

    return None

def find_values_in_sentence(sentence: str,
                            patterns: List[Tuple[int, re.Pattern, str, str, str]]) -> List[Tuple[int, int, str, str, str]]:
    """
    returns list of (start, end, attr, original_value, canon_key) in appearance order,
    using precompiled patterns robust to hyphen/space changes.
    """
    s = sentence
    hits: List[Tuple[int, int, str, str, str]] = []
    occupied = [False] * (len(s) + 1)

    for _, pat, attr, v, ck in patterns:
        for m in pat.finditer(s):
            st, ed = m.start(), m.end()
            if any(occupied[i] for i in range(st, ed)):
                continue
            for i in range(st, ed):
                occupied[i] = True
            hits.append((st, ed, attr, v, ck))

    hits.sort(key=lambda x: x[0])
    return hits

def H(ck: str, Z: Dict[str, Dict[str, Any]], canon_map: Dict[str, Tuple[str, str]]) -> Optional[Any]:
    """
    ck is canon_key; canon_map[ck] -> (attr, original_value)
    Return the Z3 var for this value.
    """
    if ck not in canon_map:
        return None
    attr, _ = canon_map[ck]
    if attr not in Z:
        return None
    if ck not in Z[attr]:
        return None
    return Z[attr][ck]

def clue_to_constraints(clue: str,
                        Z: Dict[str, Dict[str, Any]],
                        canon_map: Dict[str, Tuple[str, str]],
                        patterns: List[Tuple[int, re.Pattern, str, str, str]],
                        n: int,
                        debug: bool = False) -> List[Any]:
    """
    Converts a clue sentence to 0+ Z3 constraints.
    Conservative: returns [] if cannot confidently parse.
    """
    raw = clue or ""
    s = normalize_text(raw)

    vals = find_values_in_sentence(s, patterns)
    cks = [ck for _, _, _, _, ck in vals]

    out: List[Any] = []

    # IN HOUSE
    k = extract_house_num(s)
    if k is not None and cks:
        hx = H(cks[0], Z, canon_map)
        if hx is not None:
            out.append(hx == k)
            return out

    # NOT IN HOUSE
    if "not" in s and "house" in s and k is not None and cks:
        hx = H(cks[0], Z, canon_map)
        if hx is not None:
            out.append(hx != k)
            return out

    # immediately left/right
    if ("immediately left of" in s or "directly left of" in s) and len(cks) >= 2:
        ha, hb = H(cks[0], Z, canon_map), H(cks[1], Z, canon_map)
        if ha is not None and hb is not None:
            out.append(ha + 1 == hb)
            return out

    if ("immediately right of" in s or "directly right of" in s) and len(cks) >= 2:
        ha, hb = H(cks[0], Z, canon_map), H(cks[1], Z, canon_map)
        if ha is not None and hb is not None:
            out.append(ha == hb + 1)
            return out

    # somewhere left/right
    if ("somewhere to the left of" in s or "left of" in s) and len(cks) >= 2:
        ha, hb = H(cks[0], Z, canon_map), H(cks[1], Z, canon_map)
        if ha is not None and hb is not None:
            out.append(ha < hb)
            return out

    if ("somewhere to the right of" in s or "right of" in s) and len(cks) >= 2:
        ha, hb = H(cks[0], Z, canon_map), H(cks[1], Z, canon_map)
        if ha is not None and hb is not None:
            out.append(ha > hb)
            return out

    # next to
    if ("next to" in s or "are next to each other" in s) and len(cks) >= 2:
        ha, hb = H(cks[0], Z, canon_map), H(cks[1], Z, canon_map)
        if ha is not None and hb is not None:
            out.append(Abs(ha - hb) == 1)
            return out

    # one house between
    if "one house between" in s and len(cks) >= 2:
        ha, hb = H(cks[0], Z, canon_map), H(cks[1], Z, canon_map)
        if ha is not None and hb is not None:
            out.append(Abs(ha - hb) == 2)
            return out

    # equality ("X is Y", "X and Y are the same person" style)
    # We trigger equality if we saw 2 values and the clue contains " is " or " are "
    if len(cks) >= 2 and (re.search(r"\bis\b", s) or re.search(r"\bare\b", s)):
        ha, hb = H(cks[0], Z, canon_map), H(cks[1], Z, canon_map)
        if ha is not None and hb is not None:
            out.append(ha == hb)
            return out

    if debug:
        print("[CLUE NOT PARSED]", clue)
    return []

def extract_clue_constraints(clues: List[str],
                             Z: Dict[str, Dict[str, Any]],
                             canon_map: Dict[str, Tuple[str, str]],
                             patterns: List[Tuple[int, re.Pattern, str, str, str]],
                             n: int,
                             debug: bool = False) -> Tuple[List[Any], int]:
    cons: List[Any] = []
    parsed = 0
    for c in clues:
        cs = clue_to_constraints(c, Z, canon_map, patterns, n, debug=debug)
        if cs:
            parsed += 1
            cons.extend(cs)
    return cons, parsed


# ----------------------------
# 5) Add assignment constraints from a table (predicted or GT)
# ----------------------------

def add_assignment_constraints(S: Solver,
                               Z: Dict[str, Dict[str, Any]],
                               canon_map: Dict[str, Tuple[str, str]],
                               domains: Dict[str, List[str]],
                               table: Dict[str, Any]) -> float:
    """
    Adds constraints matching table assignments to solver.
    Returns domain_cov in [0,1] = fraction of comparable cells that map to known domain values.
    """
    tab = normalize_table(table)
    header = [h.strip() for h in tab["header"]]
    rows = tab["rows"]
    n_cells = 0
    n_in_dom = 0

    # build header -> index
    h2i = {h.strip().lower(): i for i, h in enumerate(header)}

    for r in rows:
        # house
        try:
            house = int(r[h2i.get("house", 0)])
        except Exception:
            continue

        # compare on attributes that exist in GT domains
        for attr, dom_vals in domains.items():
            ai = h2i.get(attr.strip().lower())
            if ai is None or ai >= len(r):
                continue
            n_cells += 1
            v = r[ai]
            ck = canon_value(v)
            # check if ck corresponds to a known GT value for some attr
            if ck in canon_map:
                a2, _ = canon_map[ck]
                if a2 == attr:
                    n_in_dom += 1
                    # add equality: value in this attr at this house
                    if attr in Z and ck in Z[attr]:
                        S.add(Z[attr][ck] == house)

    return float(n_in_dom / max(1, n_cells))


# ----------------------------
# 6) Reasoning validation (conservative)
# ----------------------------

def validate_reasoning(reasoning: List[str],
                       kb_solver: Solver,
                       sol_solver: Solver,
                       Z: Dict[str, Dict[str, Any]],
                       canon_map: Dict[str, Tuple[str, str]],
                       patterns: List[Tuple[int, re.Pattern, str, str, str]],
                       n: int) -> Dict[str, int]:
    """
    For each reasoning sentence:
      - parse into constraints (same parser)
      - classify each parsed constraint:
        * valid if entailed by KB (base+clues)
        * contradicted if conflicts with predicted solution assignment
        * unknown otherwise
    """
    valid = 0
    contradicted = 0
    unknown = 0
    parsed = 0

    for sent in reasoning or []:
        cs = clue_to_constraints(sent, Z, canon_map, patterns, n, debug=False)
        if not cs:
            continue
        parsed += 1
        for c in cs:
            # entailed by KB?
            kb_solver.push()
            kb_solver.add(Not(c))
            entailed = (kb_solver.check() == unsat)
            kb_solver.pop()
            if entailed:
                valid += 1
                continue

            # contradict predicted solution?
            sol_solver.push()
            sol_solver.add(c)
            is_contra = (sol_solver.check() == unsat)
            sol_solver.pop()
            if is_contra:
                contradicted += 1
            else:
                unknown += 1

    return {"parsed": parsed, "valid": valid, "contradicted": contradicted, "unknown": unknown}


# ----------------------------
# 7) Main: compute_z3(reasoning, pred_solution, clues, ground_truth)
# ----------------------------

def compute_z3_analysis(reasoning: List[str],
               pred_solution: Dict[str, Any],
               clues: List[str],
               ground_truth: Dict[str, Any],
               debug: bool = False) -> Dict[str, Any]:
    """
    Returns z3_analysis dict with:
      - parse_cov, structure_score, domain_cov_pred
      - base_sat, clues_sat, sol_sat, all_sat
      - clue_entail_pred, clue_entail_gt
      - reliable (uses parse_cov + clue_entail_gt + structure)
      - z3_quality (neutral when unreliable; hard 0 on reliable UNSAT; else clue_entail_pred)
      - reasoning stats: reason_parsed/valid/contradicted/unknown
    Safe: never raises.
    """
    try:
        gt = normalize_table(ground_truth)
        n, attrs, domains = build_domains_from_gt(gt)
        patterns, canon_map, uniq_score = build_vocab_index(domains)

        # Z3 base model from GT domains
        base, Z = build_z3_from_domains(n, domains)

        # clue constraints from clues (using GT vocab)
        clue_constraints, parsed_clues = extract_clue_constraints(clues, Z, canon_map, patterns, n, debug=debug)
        parse_cov = float(parsed_clues) / float(max(1, len(clues)))

        # SAT checks
        S_base = Solver()
        S_base.add(base.assertions())
        base_sat = 1.0 if S_base.check() == sat else 0.0

        S_clues = Solver()
        S_clues.add(base.assertions())
        for c in clue_constraints:
            S_clues.add(c)
        clues_sat = 1.0 if S_clues.check() == sat else 0.0

        # Predicted solution assignment
        pred_norm = normalize_table(pred_solution)
        s_score = structure_score(pred_norm)

        S_sol = Solver()
        S_sol.add(base.assertions())
        domain_cov_pred = add_assignment_constraints(S_sol, Z, canon_map, domains, pred_norm)
        sol_sat = 1.0 if S_sol.check() == sat else 0.0

        S_all = Solver()
        S_all.add(base.assertions())
        for c in clue_constraints:
            S_all.add(c)
        _ = add_assignment_constraints(S_all, Z, canon_map, domains, pred_norm)
        all_sat = 1.0 if S_all.check() == sat else 0.0

        # entailment of each parsed clue under predicted assignment
        clue_entail_pred = 0.0
        if clue_constraints and sol_sat > 0.5:
            sat_count = 0
            for c in clue_constraints:
                S_sol.push()
                S_sol.add(Not(c))
                ent = (S_sol.check() == unsat)
                S_sol.pop()
                if ent:
                    sat_count += 1
            clue_entail_pred = float(sat_count) / float(len(clue_constraints))

        # entailment of parsed clues under GT assignment (parser quality test)
        S_gt = Solver()
        S_gt.add(base.assertions())
        _ = add_assignment_constraints(S_gt, Z, canon_map, domains, gt)
        gt_sat = 1.0 if S_gt.check() == sat else 0.0

        clue_entail_gt = 0.0
        if clue_constraints and gt_sat > 0.5:
            sat_count = 0
            for c in clue_constraints:
                S_gt.push()
                S_gt.add(Not(c))
                ent = (S_gt.check() == unsat)
                S_gt.pop()
                if ent:
                    sat_count += 1
            clue_entail_gt = float(sat_count) / float(len(clue_constraints))

        # Knowledge base for reasoning: base + clue constraints
        S_kb = Solver()
        S_kb.add(base.assertions())
        for c in clue_constraints:
            S_kb.add(c)

        reason_stats = validate_reasoning(reasoning, S_kb, S_sol, Z, canon_map, patterns, n)

        # Reliability gating:
        # - require good structure
        # - require enough clue parsing
        # - require clue parser to agree with GT on those parsed clues
        cov_floor = 0.35  # you can tune
        gt_entail_floor = 0.80
        struct_floor = 0.80

        reliable = 1.0 if (parse_cov >= cov_floor and s_score >= struct_floor and clue_entail_gt >= gt_entail_floor and clues_sat >= 0.5) else 0.0

        # z3_quality: neutral when unreliable, else punish UNSAT hard, else use entailment
        if reliable < 0.5:
            z3_quality = 1.0
        else:
            if all_sat < 0.5:
                z3_quality = 0.0
            else:
                z3_quality = clamp01(clue_entail_pred)

        return {
            "parse_cov": float(parse_cov),
            "structure_score": float(s_score),
            "domain_cov_pred": float(clamp01(domain_cov_pred)),
            "domain_uniq": float(clamp01(uniq_score)),
            "base_sat": float(base_sat),
            "clues_sat": float(clues_sat),
            "sol_sat": float(sol_sat),
            "all_sat": float(all_sat),
            "clue_entail_pred": float(clue_entail_pred),
            "clue_entail_gt": float(clue_entail_gt),
            "reliable": float(reliable),
            "z3_quality": float(clamp01(z3_quality)),
            "reason_stats": reason_stats,
            "parsed_clues": int(parsed_clues),
            "parsed_constraints": int(len(clue_constraints)),
        }

    except Exception:
        if debug:
            print('IN Z3 Analyses, Ground-Truth = ', (ground_truth))
            logger.exception("Crash in compute_score")  # includes line number + stack
            print_exc()
        return {
            "parse_cov": 0.0,
            "structure_score": 0.0,
            "domain_cov_pred": 0.0,
            "domain_uniq": 0.0,
            "base_sat": 0.0,
            "clues_sat": 0.0,
            "sol_sat": 0.0,
            "all_sat": 0.0,
            "clue_entail_pred": 0.0,
            "clue_entail_gt": 0.0,
            "reliable": 0.0,
            "z3_quality": 0.0,
            "reason_stats": {"parsed": 0, "valid": 0, "contradicted": 0, "unknown": 0},
            "parsed_clues": 0,
            "parsed_constraints": 0,
        }


# ----------------------------
# 8) Reward scheduler for DAPO
# ----------------------------

def dapo_reward(puzzle_accuracy: float,
                cell_accuracy: float,
                z3_analysis: Dict[str, Any],
                epoch: int,
                total_epochs: int = 70,
                cell_beta: float = 2.0) -> float:
    """
    Curriculum:
      - early epochs: anchor ~ (cell_acc^beta)
      - late epochs:  anchor ~ puzzle_acc
    Logic penalty:
      - only strong when z3_analysis["reliable"] == 1
      - when reliable and all_sat==0 => hard penalty
      - else penalty uses z3_quality (0..1)
    """
    p = clamp01(puzzle_accuracy)
    c = clamp01(cell_accuracy)

    # hard boundary behavior (as you requested earlier)
    if p >= 1.0:
        return 1.0
    if p <= 0.0 and c <= 0.0:
        return 0.0

    # schedule weight from cell->puzzle
    if total_epochs <= 1:
        t = 1.0
    else:
        t = clamp01(epoch / max(1, total_epochs - 1))

    # w_puz ramps up
    w_puz = lerp(0.05, 0.95, smoothstep(t))

    c_shaped = c ** max(1e-9, float(cell_beta))
    gt_anchor = clamp01((1.0 - w_puz) * c_shaped + w_puz * p)

    reliable = float(z3_analysis.get("reliable", 0.0))
    all_sat = float(z3_analysis.get("all_sat", 0.0))
    z3q = clamp01(float(z3_analysis.get("z3_quality", 1.0)))

    # penalty strength increases with training
    alpha = lerp(1.5, 5.0, smoothstep(t))

    if reliable < 0.5:
        penalty = 1.0  # neutral when Z3 is unreliable (avoid punishing parse failures)
    else:
        if all_sat < 0.5:
            penalty = 0.05  # hard fail
        else:
            penalty = (z3q ** alpha)

    # combine
    r = gt_anchor * penalty

    # small encouragement for being parseable + well-formed, but only small
    parse_cov = clamp01(float(z3_analysis.get("parse_cov", 0.0)))
    struct = clamp01(float(z3_analysis.get("structure_score", 0.0)))
    bonus = 0.03 * (parse_cov * struct)

    return float(clamp01(r + bonus))


# ----------------------------
# 9) Quick demo
# ----------------------------

if __name__ == "__main__":
    # Example inspired by your earlier failure case (2 houses)
    clues = [
        "The person who loves yellow is the British person.",
        "The person who has black hair is the person's child is named Fred.",
        "The person who loves yellow is the person's child is named Bella.",
        "The British person is in the second house.",
        "Eric is the person who loves yellow.",
        "The person whose birthday is in April is Eric.",
    ]

    ground_truth = {
        "header": ["House", "Name", "Birthday", "Color", "Nationality", "HairColor", "Child"],
        "rows": [
            ["1", "Arnold", "sept", "red", "dane", "brown", "Fred"],
            ["2", "Eric", "april", "yellow", "brit", "black", "Bella"],
        ]
    }

    # predicted (intentionally inconsistent)
    pred = {
        "header": ["House", "Name", "Birthday", "Color", "Nationality", "HairColor", "Child"],
        "rows": [
            ["1", "Eric", "sept", "yellow", "dane", "black", "Fred"],
            ["2", "Arnold", "april", "red", "brit", "brown", "Bella"],
        ]
    }

    reasoning = [
        "Eric is the person who loves yellow.",
        "The British person is in the second house.",
    ]

    z3 = compute_z3_analysis(reasoning, pred, clues, ground_truth, debug=False)
    print("z3_analysis =", z3)

    # Suppose you computed these elsewhere:
    puzzle_acc = 0.0
    cell_acc = 0.6

    for ep in [0, 20, 50, 69]:
        r = dapo_reward(puzzle_acc, cell_acc, z3, epoch=ep, total_epochs=70)
        print(f"epoch={ep:02d} reward={r:.4f}")