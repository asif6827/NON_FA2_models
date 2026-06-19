#!/usr/bin/env python
#"""
#Evaluate Qwen3-0.6B-Base on HuggingFaceH4/MATH-500.
#Requirements:
#    pip install "transformers>=4.51.0" datasets accelerate tqdm torch
#"""

import re
import math
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME = "HuggingFaceH4/MATH-500"

MAX_NEW_TOKENS = 256
BATCH_SIZE = 1      # 0.6B is small; you can increase if VRAM allows.


def build_prompt(problem: str) -> str:
    """
    Very simple prompt for the *base* model.
    You can improve this to get better performance.
    """
    return (
        "You are a math problem solver. "
        "Solve the following problem and give ONLY the final answer.\n\n"
        f"Problem:\n{problem}\n\nAnswer:"
    )


def normalize_answer(s: str) -> str:
    """
    Normalize LaTeX-style answers for string comparison.
    This is deliberately simple; you can refine as needed.
    """
    s = s.strip()

    # Strip common LaTeX boxing
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)

    # Strip surrounding $...$
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # Remove spaces
    s = "".join(s.split())

    # Remove enclosing parentheses if present
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    return s


def maybe_numeric_equal(pred: str, gold: str) -> bool:
    """
    Try to see if answers are numerically equal (for simple numeric cases).
    Very conservative; falls back to string equality otherwise.
    """
    def to_num(x: str):
        # crude: only handle simple integers / floats
        try:
            # reject obviously non-numeric forms
            if any(c.isalpha() for c in x.replace("e", "")):
                return None
            return float(x)
        except Exception:
            return None

    p = to_num(pred)
    g = to_num(gold)
    if p is None or g is None:
        return False
    # exact or very close
    return math.isclose(p, g, rel_tol=1e-6, abs_tol=1e-6)


def extract_model_answer(full_output: str) -> str:
    """
    Get the model's 'final answer' from the decoded continuation.
    For now, just take the last line.
    You can make this more sophisticated if desired.
    """
    text = full_output.strip()
    # Sometimes models respond like "The answer is 42."
    # Try to grab everything after 'Answer:' if present.
    m = re.search(r"Answer\s*[:\-]\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Take last line as best guess
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text
    return lines[-1]


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use bfloat16 or float16 on GPU if available, else float32 on CPU
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,)
    model.eval()

    ds = load_dataset(DATASET_NAME, split="test")

    num_total = 0
    num_correct = 0

    for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Evaluating"):
        batch = ds[i : i + BATCH_SIZE]

        problems: List[str] = batch["problem"]
        gold_answers: List[str] = batch["answer"]

        prompts = [build_prompt(p) for p in problems]

        enc = tokenizer(prompts, return_tensors="pt", padding=True,truncation=True, ).to(device)

        with torch.no_grad():
            outputs = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id,)

        # For each example, slice off the prompt tokens and decode only the continuation
        for j in range(outputs.size(0)):
            input_len = enc["input_ids"][j].shape[0]
            generated_ids = outputs[j, input_len:]
            text_out = tokenizer.decode(generated_ids, skip_special_tokens=True)

            pred_raw = extract_model_answer(text_out)
            gold_raw = gold_answers[j]

            pred_norm = normalize_answer(pred_raw)
            gold_norm = normalize_answer(gold_raw)

            correct = False
            if pred_norm == gold_norm:
                correct = True
            elif maybe_numeric_equal(pred_norm, gold_norm):
                correct = True

            num_total += 1
            num_correct += int(correct)

            print(f"\nExample #{num_total}")
            print("-" * 40)
            print("Problem:", problems[j])
            print("Gold answer:", gold_raw)
            print("Model raw output:", text_out)
            print("Pred (norm):", pred_norm)
            print("Gold (norm):", gold_norm)
            print("Correct:", correct)

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    print("\n==============================")
    print(f"Total examples: {num_total}")
    print(f"Correct:        {num_correct}")
    print(f"Accuracy:       {accuracy * 100:.2f}%")
    print("==============================")

    return accuracy


if __name__ == "__main__":
    evaluate()

