# z3_zebra_verifier_refined.py
from __future__ import annotations

import re
import json
from typing import Any, Dict, List, Optional, Tuple

from z3 import Solver, Int, Distinct, And, Or, Not, Abs, sat, unsat


# -----------------------------
# Utilities
# -----------------------------

ORDINAL_TO_INT = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

def clamp01(x: float) -> float:
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

def norm_header(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def canon_text(s: str) -> str:
    """Aggressive canonicalization for matching values in clue text."""
    s = (s or "").lower()
    # unify hyphens/underscores
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    # strip punctuation (keep digits/letters/spaces)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

ROLE_WORDS = [
    "lover", "lovers", "enthusiast", "enthusiasts", "drinker", "drinkers",
    "smoker", "smokers", "keeper", "keepers", "owner", "owners",
    "eater", "eaters", "fan", "fans", "person", "people"
]

def strip_role_words(s: str) -> str:
    """Remove trailing role words to better match domain values (e.g., 'pizza lover' -> 'pizza')."""
    toks = canon_text(s).split()
    while toks and toks[-1] in ROLE_WORDS:
        toks.pop()
    return " ".join(toks)

def split_into_sentences(clue: str) -> List[str]:
    """
    Split a clue line into atomic sentences.
    Handles cases like "...house.Eric is ..." (no space after period) and
    multiple sentences in one line.
    """
    if not clue:
        return []
    s = clue.strip()

    # remove leading numbering like "1." or "12."
    s = re.sub(r"^\s*\d+\.\s*", "", s)

    # split on . ? ! where next char is uppercase OR end; allow no-space after punctuation
    parts = re.split(r"(?<=[.!?])\s*(?=[A-Z])|(?<=[.!?])(?=[A-Z])", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # also split if there are double spaces + capital start
        # (sometimes "....  The person ..." stays together)
        subparts = re.split(r"\s{2,}(?=[A-Z])", p)
        for sp in subparts:
            sp = sp.strip()
            if sp:
                out.append(sp)
    return out


# -----------------------------
# Parsing <answer> output (optional demo helper)
# -----------------------------

def parse_answer_block(text: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Extract {"reasoning": [...], "solution": {...}} from inside <answer>...</answer>.
    RL-safe: raises ValueError on failure.
    """
    m = re.search(r"<answer>\s*(\{.*?\})\s*</answer>", text, flags=re.DOTALL)
    if not m:
        raise ValueError("Missing <answer> block")
    raw = m.group(1)
    obj = json.loads(raw)
    reasoning = obj.get("reasoning", [])
    solution = obj.get("solution", {})
    if not isinstance(reasoning, list):
        raise ValueError("reasoning must be a list")
    if not isinstance(solution, dict):
        raise ValueError("solution must be a dict")
    return reasoning, solution


# -----------------------------
# Domains: prefer GT or puzzle text
# -----------------------------

def domains_from_table(table: Dict[str, Any]) -> Tuple[int, Dict[str, List[str]]]:
    """
    Build domains from a GT (or any correct) solution table.
    Returns (house_count, domains).
    domains[attr] = list of values (strings).
    """
    header = table.get("header", [])
    rows = table.get("rows", [])
    if not header or not rows:
        return 0, {}

    # house_count from max numeric in House col if present
    hidx = None
    for i, h in enumerate(header):
        if norm_header(h) == "house":
            hidx = i
            break

    n = 0
    if hidx is not None:
        for r in rows:
            if hidx < len(r):
                try:
                    n = max(n, int(str(r[hidx]).strip()))
                except Exception:
                    pass
    if n <= 0:
        n = len(rows)

    domains: Dict[str, List[str]] = {}
    for ci, h in enumerate(header):
        if norm_header(h) == "house":
            continue
        vals = []
        for r in rows:
            if ci < len(r):
                v = str(r[ci]).strip()
                if v:
                    vals.append(v)
        # keep unique in order
        seen = set()
        uniq = []
        for v in vals:
            cv = canon_text(v)
            if cv not in seen:
                uniq.append(v)
                seen.add(cv)
        domains[h] = uniq
    return n, domains

def domains_from_puzzle_text(puzzle_text: str) -> Tuple[int, Dict[str, List[str]]]:
    """
    Parse house_count + domains from ZebraPuzzle text.
    Looks for:
      "There are N houses..."
      "- Each person has a unique X: `a`, `b`, ..."
    """
    if not puzzle_text:
        return 0, {}

    n = 0
    m = re.search(r"There are\s+(\d+)\s+houses", puzzle_text, flags=re.IGNORECASE)
    if m:
        try:
            n = int(m.group(1))
        except Exception:
            n = 0

    domains: Dict[str, List[str]] = {}
    for line in puzzle_text.splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        # "- People own unique car models: `tesla model 3`, `ford f150`"
        m2 = re.search(r"^\-\s*(.+?)\s*:\s*(.+)$", line)
        if not m2:
            continue
        attr_raw = m2.group(1).strip()
        rhs = m2.group(2)
        vals = re.findall(r"`([^`]+)`", rhs)
        if not vals:
            continue
        domains[attr_raw] = vals

    return n, domains


def collapse_duplicate_domains(domains: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    If two attributes have the exact same value set (common in generated data: Person + Name),
    collapse them to reduce ambiguity.
    """
    items = list(domains.items())
    used = set()
    collapsed: Dict[str, List[str]] = {}

    def valset(vs: List[str]) -> Tuple[str, ...]:
        return tuple(sorted({canon_text(x) for x in vs}))

    for i, (ai, vi) in enumerate(items):
        if ai in used:
            continue
        si = valset(vi)
        keep = ai
        used.add(ai)
        # merge others with same set into keep
        for j in range(i + 1, len(items)):
            aj, vj = items[j]
            if aj in used:
                continue
            if valset(vj) == si:
                used.add(aj)
        collapsed[keep] = vi
    return collapsed


# -----------------------------
# Z3 Model
# -----------------------------

def build_z3_from_domains(n: int, domains: Dict[str, List[str]]) -> Tuple[Solver, Dict[str, Dict[str, Any]]]:
    """
    Build base Z3 model:
      For each attribute attr and each value v, create Int var House(attr=v) in [1..n]
      Enforce Distinct for each attribute.
    Z[attr][canon_value] -> IntRef
    """
    S = Solver()
    Z: Dict[str, Dict[str, Any]] = {}

    for attr, values in domains.items():
        Z[attr] = {}
        vars_for_attr = []
        for v in values:
            ck = canon_text(v)
            if ck in Z[attr]:
                continue
            var = Int(f"H__{norm_header(attr)}__{ck}")
            Z[attr][ck] = var
            vars_for_attr.append(var)
            S.add(And(var >= 1, var <= n))
        if vars_for_attr:
            S.add(Distinct(*vars_for_attr))
    return S, Z


def build_vocab_index(domains: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Map canon_value -> list of (attr, canon_value_for_that_attr).
    We keep all mappings (ambiguity is handled later with confidence).
    """
    vocab: Dict[str, List[Tuple[str, str]]] = {}
    for attr, values in domains.items():
        for v in values:
            ck = canon_text(v)
            vocab.setdefault(ck, []).append((attr, ck))
            # also store stripped role version if it helps (e.g., "tesla model 3" already ok)
            ck2 = strip_role_words(v)
            if ck2 and ck2 != ck:
                vocab.setdefault(ck2, []).append((attr, ck))
    return vocab

def find_value_keys_in_text(text: str, vocab: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    """
    Find which vocab keys appear in text (canonicalized).
    Uses substring match (multiword ok). Prefers longer matches first.
    """
    t = canon_text(text)
    hits = []
    # check longer keys first to avoid "model 3" matching before "tesla model 3"
    keys = sorted(vocab.keys(), key=lambda k: (-len(k), k))
    for k in keys:
        if not k:
            continue
        # require token-boundary-ish match
        if re.search(r"(?:^|\s)" + re.escape(k) + r"(?:$|\s)", t):
            hits.append(k)
    # de-dup preserve order
    seen = set()
    out = []
    for h in hits:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out

def parse_house_index(text: str) -> Optional[int]:
    t = text.lower()
    # "in the second house" / "in house 2"
    m = re.search(r"\bhouse\s+(\d+)\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    m = re.search(r"\b(in|at)\s+the\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+house\b", t)
    if m:
        return ORDINAL_TO_INT.get(m.group(2))
    return None

def parse_between_distance(text: str) -> Optional[int]:
    """
    "There is one house between X and Y." -> distance = 2
    "There are two houses between ..." -> distance = 3
    """
    t = text.lower()
    m = re.search(r"\bthere\s+is\s+one\s+house\s+between\b", t)
    if m:
        return 2
    m = re.search(r"\bthere\s+are\s+two\s+houses\s+between\b", t)
    if m:
        return 3
    m = re.search(r"\bthere\s+are\s+three\s+houses\s+between\b", t)
    if m:
        return 4
    return None


def clue_to_constraints(
    sentence: str,
    Z: Dict[str, Dict[str, Any]],
    vocab: Dict[str, List[Tuple[str, str]]],
    n: int,
) -> Tuple[List[Any], float]:
    """
    Returns (constraints, confidence) for ONE atomic sentence.
    confidence in (0,1], reduced when ambiguous (value maps to many attrs).
    """
    s = sentence.strip()
    if not s:
        return [], 0.0

    # remove trailing period
    if s.endswith("."):
        s = s[:-1].strip()

    # gather values in sentence
    keys = find_value_keys_in_text(s, vocab)

    def candidates(key: str) -> List[Any]:
        cands = []
        for (attr, ck) in vocab.get(key, []):
            var = Z.get(attr, {}).get(ck)
            if var is not None:
                cands.append(var)
        return cands

    def conf_for(key: str) -> float:
        c = len(candidates(key))
        return 1.0 / float(c) if c > 0 else 0.0

    # helper to make Or over all candidate pairs
    def or_pairs(a: str, b: str, mk):
        ca, cb = candidates(a), candidates(b)
        if not ca or not cb:
            return [], 0.0
        disj = []
        for va in ca:
            for vb in cb:
                disj.append(mk(va, vb))
        # ambiguity reduces confidence multiplicatively
        conf = (1.0 / (len(ca) * len(cb)))
        return [Or(*disj)] if len(disj) > 1 else disj, conf

    # Split patterns by priority.
    t = canon_text(s)

    # (A) BETWEEN distance
    dist = parse_between_distance(s)
    if dist is not None and len(keys) >= 2:
        a, b = keys[0], keys[1]
        ca, cb = candidates(a), candidates(b)
        if ca and cb:
            disj = [Abs(va - vb) == dist for va in ca for vb in cb]
            conf = 1.0 / (len(ca) * len(cb))
            return ([Or(*disj)] if len(disj) > 1 else disj, conf)

    # (B) Direct adjacency left/right
    if ("directly left of" in t) or ("immediately left of" in t):
        if len(keys) >= 2:
            a, b = keys[0], keys[1]
            cons, conf = or_pairs(a, b, lambda va, vb: va + 1 == vb)
            return cons, conf

    if ("directly right of" in t) or ("immediately right of" in t):
        if len(keys) >= 2:
            a, b = keys[0], keys[1]
            cons, conf = or_pairs(a, b, lambda va, vb: va == vb + 1)
            return cons, conf

    # (C) Next to / not next to
    if ("not next to" in t) or ("not adjacent to" in t):
        if len(keys) >= 2:
            a, b = keys[0], keys[1]
            cons, conf = or_pairs(a, b, lambda va, vb: Abs(va - vb) != 1)
            return cons, conf

    if ("next to" in t) or ("adjacent to" in t):
        if len(keys) >= 2:
            a, b = keys[0], keys[1]
            cons, conf = or_pairs(a, b, lambda va, vb: Abs(va - vb) == 1)
            return cons, conf

    # (D) Somewhere left/right
    if ("somewhere to the left of" in t) or ("to the left of" in t):
        if len(keys) >= 2:
            a, b = keys[0], keys[1]
            cons, conf = or_pairs(a, b, lambda va, vb: va < vb)
            return cons, conf

    if ("somewhere to the right of" in t) or ("to the right of" in t):
        if len(keys) >= 2:
            a, b = keys[0], keys[1]
            cons, conf = or_pairs(a, b, lambda va, vb: va > vb)
            return cons, conf

    # (E) NOT IN HOUSE (must be checked before IN HOUSE)
    hk = parse_house_index(s)
    if hk is not None and ("not in" in t):
        # pick first value mentioned
        if len(keys) >= 1:
            a = keys[0]
            ca = candidates(a)
            if ca:
                conf = 1.0 / len(ca)
                disj = [va != hk for va in ca]
                return ([Or(*disj)] if len(disj) > 1 else disj, conf)

    # (F) IN HOUSE
    if hk is not None and ((" in the " in t) or (" in house " in t) or (" is in " in t)):
        if len(keys) >= 1:
            a = keys[0]
            ca = candidates(a)
            if ca:
                conf = 1.0 / len(ca)
                disj = [va == hk for va in ca]
                return ([Or(*disj)] if len(disj) > 1 else disj, conf)

    # (G) Equality "X is Y"
    if (" is " in t) or (" are " in t):
        if len(keys) >= 2:
            a, b = keys[0], keys[1]
            cons, conf = or_pairs(a, b, lambda va, vb: va == vb)
            return cons, conf

    return [], 0.0


def extract_clue_constraints(
    clues: List[str],
    Z: Dict[str, Dict[str, Any]],
    domains: Dict[str, List[str]],
    n: int,
    debug: bool = False,
) -> Tuple[List[Any], float, int]:
    """
    Returns (constraints, parse_cov, parsed_sentence_count).
    parse_cov is confidence-weighted coverage over original clue lines.
    """
    vocab = build_vocab_index(domains)

    all_constraints: List[Any] = []
    total_conf = 0.0
    parsed_sentences = 0

    # We compute coverage vs original clue lines; each clue line can become multiple sentences.
    for clue in clues:
        sentences = split_into_sentences(clue)
        clue_conf = 0.0
        clue_parsed_any = False

        for sent in sentences:
            cons, conf = clue_to_constraints(sent, Z, vocab, n)
            if cons:
                clue_parsed_any = True
                parsed_sentences += 1
                all_constraints.extend(cons)
                # confidence accumulates (cap at 1 per clue-line)
                clue_conf = min(1.0, clue_conf + conf)

        total_conf += clue_conf
        if debug and not clue_parsed_any:
            print("[CLUE NOT PARSED]", clue)

    parse_cov = total_conf / max(1, len(clues))
    return all_constraints, float(clamp01(parse_cov)), int(parsed_sentences)


# -----------------------------
# Predicted solution constraints
# -----------------------------

def header_to_domain_attr(pred_header: List[str], domains: Dict[str, List[str]]) -> Dict[int, Optional[str]]:
    """
    Map each predicted column index -> best matching domain attribute name.
    Uses normalized string containment as a simple heuristic.
    """
    dom_keys = list(domains.keys())
    dom_norm = [(k, norm_header(k)) for k in dom_keys]

    mapping: Dict[int, Optional[str]] = {}
    for i, h in enumerate(pred_header):
        hn = norm_header(h)
        if hn == "house":
            mapping[i] = None
            continue

        best = None
        best_score = -1
        for k, kn in dom_norm:
            # exact match strongest
            if hn == kn:
                best = k
                best_score = 10
                break
            # containment / overlap
            if hn and kn and (hn in kn or kn in hn):
                score = min(len(hn), len(kn))
                if score > best_score:
                    best_score = score
                    best = k
        mapping[i] = best
    return mapping

def add_pred_solution_constraints(
    S: Solver,
    Z: Dict[str, Dict[str, Any]],
    domains: Dict[str, List[str]],
    pred_solution: Dict[str, Any],
) -> float:
    """
    Add constraints from predicted solution.
    Returns assignment_coverage in [0,1] (how many cells could be grounded into Z).
    """
    header = pred_solution.get("header", [])
    rows = pred_solution.get("rows", [])
    if not header or not rows:
        return 0.0

    # identify house column if present
    house_col = None
    for i, h in enumerate(header):
        if norm_header(h) == "house":
            house_col = i
            break

    col_to_attr = header_to_domain_attr(header, domains)

    grounded = 0
    total_cells = 0

    # Build quick reverse dict for value lookup per attr
    # Z[attr] is keyed by canon(value) already
    for ri, row in enumerate(rows):
        # determine house index
        if house_col is not None and house_col < len(row):
            try:
                hidx = int(str(row[house_col]).strip())
            except Exception:
                hidx = ri + 1
        else:
            hidx = ri + 1

        for ci, cell in enumerate(row):
            if ci == house_col:
                continue
            total_cells += 1

            attr = col_to_attr.get(ci)
            if not attr:
                continue

            v = str(cell).strip()
            if not v:
                continue

            ck = canon_text(v)
            var = Z.get(attr, {}).get(ck)

            # fallback: try strip_role_words and canonical
            if var is None:
                ck2 = strip_role_words(v)
                if ck2:
                    var = Z.get(attr, {}).get(ck2)

            if var is None:
                continue

            grounded += 1
            S.add(var == hidx)

    if total_cells <= 0:
        return 0.0
    return float(clamp01(grounded / total_cells))


# -----------------------------
# Main scoring components
# -----------------------------

def compute_z3_components(
    reasoning: List[str],
    predicted_solution: Dict[str, Any],
    clues: List[str],
    puzzle_text: Optional[str] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Revised, robust core:
      inputs: reasoning(list[str]), predicted_solution(dict), clues(list[str]),
              optional puzzle_text, optional ground_truth(table)

      outputs:
        - parse_cov: how much of clues were parsed (confidence weighted)
        - clue_sat: fraction of parsed constraints satisfied by predicted assignment
        - z3_sat: 1.0 if all parsed constraints + predicted assignment is SAT else 0.0
        - assignment_cov: how much predicted table could be grounded into the Z3 domain
        - parsed_constraints / parsed_sentences
    """
    out = {
        "parse_cov": 0.0,
        "clue_sat": 0.0,
        "z3_sat": 0.0,
        "assignment_cov": 0.0,
        "parsed_constraints": 0,
        "parsed_sentences": 0,
    }

    try:
        # 1) domains prefer GT, else puzzle text, else derive from predicted table
        n = 0
        domains: Dict[str, List[str]] = {}

        if ground_truth:
            n, domains = domains_from_table(ground_truth)

        if (not domains) and puzzle_text:
            n2, domains2 = domains_from_puzzle_text(puzzle_text)
            if domains2:
                domains = domains2
                if n <= 0:
                    n = n2

        if not domains:
            # last resort: domains from predicted solution (weaker)
            n, domains = domains_from_table(predicted_solution)

        domains = collapse_duplicate_domains(domains)
        if n <= 0:
            n = len(predicted_solution.get("rows", []))

        if n <= 0 or not domains:
            return out

        # 2) build base model
        base, Z = build_z3_from_domains(n, domains)

        # 3) parse clues -> constraints
        clue_constraints, parse_cov, parsed_sentences = extract_clue_constraints(
            clues, Z, domains, n, debug=debug
        )
        out["parse_cov"] = float(parse_cov)
        out["parsed_constraints"] = int(len(clue_constraints))
        out["parsed_sentences"] = int(parsed_sentences)

        # 4) solver with predicted assignment
        S_sol = Solver()
        S_sol.add(base.assertions())
        assign_cov = add_pred_solution_constraints(S_sol, Z, domains, predicted_solution)
        out["assignment_cov"] = float(assign_cov)

        if not clue_constraints:
            # nothing parsed => can't trust Z3
            out["clue_sat"] = 0.0
            out["z3_sat"] = 0.0
            return out

        # 5) clue_sat: count constraints satisfied by assignment
        sat_count = 0
        for c in clue_constraints:
            S_sol.push()
            S_sol.add(Not(c))
            entailed = (S_sol.check() == unsat)  # UNSAT => assignment implies c
            S_sol.pop()
            if entailed:
                sat_count += 1
        out["clue_sat"] = float(sat_count / max(1, len(clue_constraints)))

        # 6) z3_sat: all constraints together satisfiable under assignment?
        S_all = Solver()
        S_all.add(base.assertions())
        add_pred_solution_constraints(S_all, Z, domains, predicted_solution)
        for c in clue_constraints:
            S_all.add(c)
        out["z3_sat"] = 1.0 if (S_all.check() == sat) else 0.0

        return out

    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return out


# -----------------------------
# Quick demos
# -----------------------------

def demo_from_unparsed_file(path: str, n_take: int = 10):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [x.strip() for x in f.read().splitlines() if x.strip()]

    # fabricate minimal domains from typical zebra values found in the file
    # (for demo only). In real use: pass puzzle_text or ground_truth.
    demo_domains = {
        "Name": ["Arnold", "Eric", "Peter", "Alice", "Bob", "Carol"],
        "CarModel": ["tesla model 3", "ford f150", "toyota camry", "honda civic"],
        "FavoriteSport": ["soccer", "basketball", "tennis", "swimming"],
        "HouseStyle": ["colonial", "victorian", "ranch", "modern", "mediterranean", "craftsman"],
        "Cigar": ["pall mall", "prince", "dunhill", "blue master"],
        "Pet": ["dog", "cat", "fish", "bird", "hamster", "rabbit"],
        "Food": ["pizza", "grilled cheese", "spaghetti", "stew"],
        "Mother": ["holly", "aniya", "kailyn", "janelle"],
        "Height": ["short", "very short", "average", "tall"],
        "BookGenre": ["mystery", "science fiction", "romance"],
        "MusicGenre": ["rock", "pop", "classical"],
        "Child": ["bella", "fred", "meredith"],
    }
    n = 6
    base, Z = build_z3_from_domains(n, collapse_duplicate_domains(demo_domains))
    cons, cov, parsed_sent = extract_clue_constraints(lines[:n_take], Z, demo_domains, n, debug=True)
    print(f"[DEMO] parsed_constraints={len(cons)} parsed_sentences={parsed_sent} parse_cov={cov:.3f}")

def demo_case_minimal():
    # Your 2-house failing style example (use GT-derived domains)
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
            ["1", "Arnold", "sept", "red", "dane", "black", "fred"],
            ["2", "Eric", "april", "yellow", "brit", "brown", "bella"],
        ]
    }

    # intentionally wrong prediction (swap some)
    predicted = {
        "header": ["House", "Name", "Birthday", "Color", "Nationality", "HairColor", "Child"],
        "rows": [
            ["1", "Eric", "sept", "yellow", "dane", "black", "fred"],
            ["2", "Arnold", "april", "red", "brit", "brown", "bella"],
        ]
    }

    z = compute_z3_components(
        reasoning=[],
        predicted_solution=predicted,
        clues=clues,
        puzzle_text=None,
        ground_truth=ground_truth,
        debug=False
    )
    print("[DEMO minimal] z3:", z)

if __name__ == "__main__":
    demo_case_minimal()
    # If you have the file in the same directory:
    # demo_from_unparsed_file("not-parsed.txt", n_take=20)
