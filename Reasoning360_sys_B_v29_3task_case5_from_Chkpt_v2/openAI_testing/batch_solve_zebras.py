#!/usr/bin/env python3
"""
Batch-generate ZebraPuzzle solutions in your STRICT <answer>{JSON}</answer> format.

Input file format (your uploaded file):
[
  {"id": "some-id", "puzzle": "PUZZLE TEXT ..."},
  ...
]

Output file format (JSON):
[
  {"id": "...", "answer": { ... five keys ... }, "raw": "<answer>...</answer>"},
  ...
]

This script is API-agnostic: it works with any OpenAI-compatible /v1/chat/completions endpoint
(e.g., OpenAI, vLLM, LM Studio, Ollama + OpenAI wrapper, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


SYSTEM_PROMPT = """You are an expert logic puzzle solver. You are provided with a logic puzzle.

Your task is to:
    - Extract the domain (N houses + all attribute values).
    - Parse each clue into a canonical, machine-checkable form.
    - Perform step-by-step deductions using only canonical atoms.
    - Derive a correct final solution.
    - Return the result STRICTLY as a single valid JSON object wrapped inside <answer>...</answer>.

CRITICAL FORMAT REQUIREMENTS:
    - Output MUST contain ONLY ONE <answer>...</answer> block and NOTHING ELSE.
    - Do NOT include extra text, markdown, explanations, or code fences.
    - Inside <answer>...</answer>, the content MUST be a single valid JSON object.
    - The JSON object MUST have exactly FIVE top-level keys:
        "n_houses", "attribute_values", "parsed_clues", "parsed_reasoning", "solution".
    - Do NOT add any other keys.

NORMALIZATION RULES:
    - Use underscores instead of spaces in VALUES (e.g., grilled_cheese, root_beer, bmw_3_series).
    - Attribute names MUST match the puzzle text exactly (case-sensitive), e.g., Name, Drink, Pet, HairColor, Lunch, Nationality, PhoneModel, etc.
    - House numbers are integers 1..N.
    - Convert ordinals to integers: first=1, second=2, third=3, fourth=4, fifth=5, sixth=6.
    - Do NOT invent values. Every <Val> must be one of the allowed values listed in the puzzle text (after normalization).
    - If the clue mentions a bare person name (e.g., "Bob"), treat it as Name=Bob.
    - If the clue mentions a bare demonym (e.g., "The German"), map it to Nationality=german (or the matching attribute in the puzzle text).
    - If the clue uses a descriptor like "cat lover", "dog owner", "coffee drinker", map it to the matching attribute/value from the puzzle text (e.g., Pet=cat, Drink=coffee), choosing the closest listed value.

D) DOMAIN OUTPUT (MANDATORY)
    - "n_houses" MUST be an integer N equal to the number of houses in the puzzle.
    - "attribute_values" MUST be a JSON object mapping each attribute name to the FULL list of allowed values from the puzzle text.
    - Each attribute list MUST contain exactly N unique values (after normalization).
    - Include every attribute listed in the puzzle text, and only those attributes.
    - Do NOT infer extra attributes that are not explicitly listed in the puzzle text.

A) parsed_clues (MANDATORY, PARSABLE)
    - "parsed_clues" MUST be a list of strings.
    - Each string must be exactly 1 sentence and end with a period.
    - There MUST be exactly one entry per clue, in the same order as the clues.
    - Each parsed clue MUST follow this exact DSL format:

    C<i> = <predicate>.

Allowed <predicate> forms (use exactly these):
    - set(<H>,<Attr>,<Val>)
    - not_set(<H>,<Attr>,<Val>)
    - immediately_left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - left_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - right_of(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - adjacent(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - same_house(<AttrA>=<ValA>,<AttrB>=<ValB>)
    - between(<AttrA>=<ValA>,<AttrB>=<ValB>,<K>)

Semantics:
    - immediately_left_of(A,B): A is exactly 1 house left of B.
    - left_of(A,B): A is somewhere left of B (strictly smaller house index).
    - right_of(A,B): A is somewhere right of B (strictly larger house index).
    - adjacent(A,B): houses differ by exactly 1.
    - between(A,B,K): there are exactly K houses strictly between A and B.
      (So K=1 => positions differ by 2, and K=2 => positions differ by 3.)

B) parsed_reasoning (MANDATORY, PARSABLE)
    - "parsed_reasoning" MUST be a list of strings.
    - Each string must be exactly 1 sentence and end with a period.
    - There is NO LIMIT on the number of entries.
    - Each entry MUST follow this exact DSL format:

    S<k> [C<i>(+C<j>...)] <op>(<H>,<Attr>,<Val>).

Where:
    - <k> is a step number starting at 1 and increasing by 1 each step.
    - Evidence inside [...] must reference clue ids, e.g. [C1] or [C1+C3].
    - <op> is either set or not.
    - <H> is a house number integer (1..N).
    - <Attr> and <Val> must come from the puzzle text (normalized with underscores for values).

LOGICAL VALIDITY REQUIREMENT:
    - Every step in "parsed_reasoning" MUST be logically entailed by the parsed clues plus any earlier reasoning steps.
    - If you cannot deduce a set(...) fact with certainty, output a not(...) fact that is guaranteed true.

C) solution (MANDATORY TABLE)
    - "solution" MUST be in tabular form with:
      - "header": a list of column names
      - "rows": a list of rows, each row being a list of strings matching the header order.
    - The header MUST include "House" and then all attribute columns from the puzzle text.
    - The rows MUST list houses in increasing order from 1..N.
    - All solution VALUES must be normalized with underscores (same as above).
"""


USER_TEMPLATE = """
--------------------------------
PUZZLE TO SOLVE
--------------------------------

{puzzle}

Solve the puzzle above and provide parsed_reasoning parsed_clues and solution by returning ONLY the <answer>...</answer> block, with no additional text.
""".strip()


ANSWER_RE = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)


def extract_answer_block(text: str) -> Tuple[Optional[str], Optional[dict], Optional[str]]:
    """
    Returns (json_str, json_obj, err)
    """
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None, None, "No <answer>{...}</answer> block found."
    json_str = matches[-1].strip()
    try:
        obj = json.loads(json_str)
    except Exception as e:
        return json_str, None, f"JSON parse error: {e}"
    return json_str, obj, None


def validate_answer_obj(obj: dict) -> Optional[str]:
    required_keys = ["n_houses", "attribute_values", "parsed_clues", "parsed_reasoning", "solution"]
    if list(obj.keys()) != required_keys:
        # Be strict: exact key set AND order (order matters when you re-dump),
        # but many models reorder keys; we still accept if set matches.
        if set(obj.keys()) != set(required_keys):
            return f"Top-level keys mismatch. Got: {list(obj.keys())}"
    # minimal structural checks
    if not isinstance(obj.get("n_houses"), int):
        return "n_houses must be an int."
    if not isinstance(obj.get("attribute_values"), dict):
        return "attribute_values must be an object."
    if not isinstance(obj.get("parsed_clues"), list):
        return "parsed_clues must be a list."
    if not isinstance(obj.get("parsed_reasoning"), list):
        return "parsed_reasoning must be a list."
    sol = obj.get("solution")
    if not isinstance(sol, dict) or "header" not in sol or "rows" not in sol:
        return "solution must be an object with header and rows."
    return None


def chat_completion(
    base_url: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 2500,
    timeout: int = 120,
) -> str:
    """
    OpenAI-compatible chat completions endpoint:
    POST {base_url}/v1/chat/completions
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    return out["choices"][0]["message"]["content"]


def solve_one(
    base_url: str,
    api_key: Optional[str],
    model: str,
    puzzle_text: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    backoff_s: float,
) -> Dict[str, Any]:
    user_prompt = USER_TEMPLATE.format(puzzle=puzzle_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    last_err = None
    last_raw = None
    last_json_str = None
    last_obj = None

    for attempt in range(1, retries + 1):
        try:
            raw = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            last_raw = raw

            json_str, obj, err = extract_answer_block(raw)
            last_json_str, last_obj, last_err = json_str, obj, err

            if err is None and obj is not None:
                v_err = validate_answer_obj(obj)
                if v_err is None:
                    return {"ok": True, "answer": obj, "raw": f"<answer>{json_str}</answer>"}
                last_err = v_err
        except Exception as e:
            last_err = str(e)

        time.sleep(backoff_s * attempt)

    return {
        "ok": False,
        "error": last_err or "Unknown error",
        "raw": last_raw,
        "json_str": last_json_str,
        "answer": last_obj,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSON file.")
    ap.add_argument("--output", required=True, help="Path to output JSON file.")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000"), help="OpenAI-compatible base URL.")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="API key (or set OPENAI_API_KEY).")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"), help="Model name.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff-s", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="If >0, only solve first N puzzles (useful for testing).")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise SystemExit("Input JSON must be a list of objects: [{id,puzzle}, ...].")

    if args.limit and args.limit > 0:
        items = items[: args.limit]

    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        pid = item.get("id")
        puzzle = item.get("puzzle")
        if not isinstance(pid, str) or not isinstance(puzzle, str):
            results.append({"id": pid, "ok": False, "error": "Bad input item (needs string id and puzzle)."})
            continue

        print(f"[{idx}/{len(items)}] Solving {pid} ...", flush=True)
        out = solve_one(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            puzzle_text=puzzle,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            backoff_s=args.backoff_s,
        )
        results.append({"id": pid, **out})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r.get("ok"))
    print(f"Done. OK={ok}/{len(results)}. Output: {args.output}")


if __name__ == "__main__":
    main()
