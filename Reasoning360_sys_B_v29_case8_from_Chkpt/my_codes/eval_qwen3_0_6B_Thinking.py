#!/usr/bin/env python
"""
Evaluate Qwen3-0.6B-Thinking on HuggingFaceH4/MATH-500.

Requirements:
    pip install "transformers>=4.51.0" datasets accelerate tqdm torch
"""

import re
import math
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


MODEL_NAME = "Qwen/Qwen3-0.6B-Thinking"
DATASET_NAME = "HuggingFaceH4/MATH-500"

MAX_NEW_TOKENS = 768
BATCH_SIZE = 1


def build_messages(problem: str) -> List[dict]:
    """
    Build messages for the thinking model.

    We encourage long chain-of-thought, but still make sure the final line
    is parsable as `Answer: <...>`.
    """
    system_prompt = (
        "You are a deliberate, chain-of-thought mathematical reasoner. "
        "For each problem, you must think carefully step by step, write down your "
        "reasoning, and finally provide ONLY the final answer on the last line in "
        "the format: Answer: <answer>.\n\n"
        "Do not skip steps. Do not give multiple candidate answers."
    )
    user_prompt = f"Problem:\n{problem}\n\nThink step by step and solve it."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages


def normalize_answer(s: str) -> str:
    """Normalize the final answer for strict comparison."""
    s = s.strip()

    # Prefer the last explicit "Answer: ..." if present
    m = re.findall(r"Answer\s*[:\-]\s*(.*)", s, flags=re.IGNORECASE)
    if m:
        s = m[-1].strip()

    # Remove \boxed{...}
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)

    # Remove surrounding $...$
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()

    # Remove spaces
    s = "".join(s.split())

    # Remove enclosing parentheses
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    return s


def maybe_numeric_equal(pred: str, gold: str) -> bool:
    """Try numeric equality for simple numeric answers."""
    def to_num(x: str):
        try:
            if any(c.isalpha() for c in x.replace("e", "")):
                return None
            return float(x)
        except Exception:
            return None

    p = to_num(pred)
    g = to_num(gold)
    if p is None or g is None:
        return False
    return math.isclose(p, g, rel_tol=1e-6, abs_tol=1e-6)


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device == "cuda" else torch.float32)
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    ds = load_dataset(DATASET_NAME, split="test")

    num_total = 0
    num_correct = 0

    for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Evaluating (Thinking)"):
        batch = ds[i : i + BATCH_SIZE]

        problems: List[str] = batch["problem"]
        gold_answers: List[str] = batch["answer"]

        prompts = []
        for p in problems:
            messages = build_messages(p)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,        # start with greedy; you can later switch to sampling
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j in range(outputs.size(0)):
            input_len = enc["input_ids"][j].shape[0]
            generated_ids = outputs[j, input_len:]
            text_out = tokenizer.decode(generated_ids, skip_special_tokens=True)

            gold_raw = gold_answers[j]
            pred_norm = normalize_answer(text_out)
            gold_norm = normalize_answer(gold_raw)

            correct = pred_norm == gold_norm or maybe_numeric_equal(pred_norm, gold_norm)

            num_total += 1
            num_correct += int(correct)

            print(f"\nExample #{num_total}")
            print("-" * 40)
            print("Problem:", problems[j])
            print("Gold answer:", gold_raw)
            print("Model output:", text_out)
            print("Pred (norm):", pred_norm)
            print("Gold (norm):", gold_norm)
            print("Correct:", correct)

    acc = num_correct / num_total if num_total > 0 else 0.0
    print("\n==============================")
    print(f"Thinking model: {MODEL_NAME}")
    print(f"Total examples: {num_total}")
    print(f"Correct:        {num_correct}")
    print(f"Accuracy:       {acc * 100:.2f}%")
    print("==============================")

    return acc


if __name__ == "__main__":
    evaluate()
