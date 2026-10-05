#!/usr/bin/env python
"""
Evaluate Qwen3-0.6B-Instruct on HuggingFaceH4/MATH-500.

Requirements:
    pip install "transformers>=4.51.0" datasets accelerate tqdm torch
"""
import os
import re
import math
from typing import List
import pandas as pd

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

hp_dell = True
panther = False


if hp_dell:
    print("Using HP-DELL Configration")
    os.environ["HF_HOME"] = "/home/asif/data3/HF_cache"
    os.environ["HF_DATASETS_CACHE"] = "/home/asif/data3/HF_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/home/asif/data3/HF_cache"
    model_path = "Qwen/Qwen3-0.6B"
    DATASET_NAME = "/home/asif/data3/HF_Cache/guru_data_20/offline_eval/math__math_500_20.parquet"

elif panther:
    print("Using Panther server Configration")
    os.environ["HF_HOME"] = "/export/home/asifali/HF_cache"
    os.environ["HF_DATASETS_CACHE"] = "/export/home/asifali/HF_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/export/home/asifali/HF_cache"
    model_path = "/export/home/asifali/HF_cache/Qwen3-0.6B"
    DATASET_NAME = "/export/home/asifali/HF_cache/guru_data_20/offline_eval/math__math_500_20.parquet"
    #DATASET_NAME = "/export/home/asifali/HF_cache/guru_data/offline_eval/math__math_500.parquet"




MAX_NEW_TOKENS = 1024
BATCH_SIZE = 1


def build_messages(problem: str) -> List[dict]:
    """
    Build chat messages for the instruct model.
    This uses the chat template of Qwen3.
    """
    system_prompt = (
        "You are a highly skilled math competition assistant. "
        "Solve the following problem step by step, then output ONLY the final answer "
        "on the last line in the format: Answer: <answer>."
    )
    user_prompt = f"Problem:\n{problem}\n\nSolve it carefully."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages



def normalize_gold_tag(expr: str) -> str:
    """Normalize simple LaTeX math like \\left(3,\\frac{\\pi}{2}\\right) to plain text (3,π/2)."""
    s = expr.strip()

    # Remove \left and \right
    s = s.replace(r'\left', '').replace(r'\right', '')

    # Replace \frac{a}{b} with a/b
    def frac_repl(m):
        num, den = m.group(1), m.group(2)
        return f"{num}/{den}"

    s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', frac_repl, s)

    # Replace \pi with π (or 'pi' if you prefer)
    s = s.replace(r'\pi', 'π')

    # Remove remaining braces
    s = s.replace('{', '').replace('}', '')

    # Collapse whitespace
    s = re.sub(r'\s+', '', s)

    return s

def normalize_pred_tag(expr: str) -> str:
    """Extract and normalize the inner content of <answer>...</answer>."""
    m = re.search(r'<answer>(.*?)</answer>', expr, flags=re.DOTALL)
    if m:
        inner = m.group(1)
    else:
        inner = expr
    inner = inner.strip()
    inner = re.sub(r'\s+', '', inner)
    return inner

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
    if hp_dell:
        device = torch.device("cpu")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device == "cuda" else torch.float32)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    #ds = load_dataset(DATASET_NAME, split="test")
    ds = pd.read_parquet(DATASET_NAME)
    
    num_total = 0
    num_correct = 0

    for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Evaluating (Instruct)"):
        batch = ds[i : i + BATCH_SIZE]
        
        problems: List[str] = batch["prompt"]
        gold_answers: List[str] = batch["reward_model"]

        # Build chat prompts
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
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j in range(outputs.size(0)):
            input_len = enc["input_ids"][j].shape[0]
            generated_ids = outputs[j, input_len:]
            text_out = tokenizer.decode(generated_ids, skip_special_tokens=True)

            gold_raw = gold_answers[i+j]
            pred_norm = normalize_pred_tag(text_out)
            gold_norm = normalize_gold_tag(gold_raw['ground_truth'])

            correct = pred_norm == gold_norm or maybe_numeric_equal(pred_norm, gold_norm)

            num_total += 1
            num_correct += int(correct)

            print(f"\nExample #{num_total}")
            print("-" * 40)
            print("Problem:", problems[i+j])
            print("Gold answer:", gold_raw)
            print("Model output:", text_out)
            print("Pred (norm):", pred_norm)
            print("Gold (norm):", gold_norm)
            print("Correct:", correct)

    acc = num_correct / num_total if num_total > 0 else 0.0
    print("\n==============================")
    print(f"Instruct model: {model_path}")
    print(f"Total examples: {num_total}")
    print(f"Correct:        {num_correct}")
    print(f"Accuracy:       {acc * 100:.2f}%")
    print("==============================")

    return acc


if __name__ == "__main__":
    evaluate()
