#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract reasoning content from puzzle-solving results to build SFT dataset.
Separates correct and incorrect solutions for training.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any
from datasets import Dataset, DatasetDict
from datetime import datetime

job_id = os.getenv("SLURM_JOB_ID")
print("SLURM Job ID:", job_id)
debug = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract SFT data from puzzle-solving results")
    parser.add_argument("--input_files", nargs="+", required=True, help="Path to input JSONL files containing puzzle results")
    parser.add_argument("--output_dir", default="/export/home/asifali/Reasoning360/Prompts_SFT_data/", help="Directory to save the extracted SFT datasets")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/test split ratio (default: 0.9)")
    parser.add_argument("--correct_limit", type=int, default=-1, help="Train/test split ratio (default: 0.9)")
    parser.add_argument("--incorrect_limit", type=int, default=-1, help="Train/test split ratio (default: 0.9)")

    return parser.parse_args()


def extract_reasoning_from_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trace:
        return {}

    clean_reasoning = []
    final_solution = {}
    has_valid_solution = False

    for i, step in enumerate(trace):
        if "result" in step:
            result = step["result"]
            if isinstance(result, list) and len(result) == 2:
                parsed_puzzle, constraints = result
                clean_reasoning.append("Generated meta information and constraints for puzzle solution.")
                final_solution = {
                    "meta": parsed_puzzle,
                    "constraints": constraints
                }
                has_valid_solution = True

        elif "response" in step:
            response = step["response"]
            stage = step.get("stage", "")

            extracted_content = ""

            if isinstance(response, str):
                if stage == "SolutionPrompt" or stage == "SolutionPrompt_Initial":
                    try:
                        response_data = json.loads(response)
                        if isinstance(response_data, dict):
                            if "reasoning" in response_data:
                                extracted_content = response_data["reasoning"]
                            if "solution" in response_data:
                                final_solution = response_data["solution"]
                                has_valid_solution = True
                    except (json.JSONDecodeError, TypeError):
                        if "Reasoning:" in response:
                            extracted_content = response.split("Reasoning:")[-1].strip()
                        elif "### Analysis" in response:
                            extracted_content = response.split("### Analysis")[-1].strip()

                elif stage == "VerificationPrompt" or stage == "VerificationPrompt_Attempt_1":
                    try:
                        try:
                            response_end_index = response.rfind('}')
                            response = response[:response_end_index + 1]
                        except (json.JSONDecodeError, TypeError):
                            pass
                        response_data = json.loads(response)
                        if isinstance(response_data, dict) and "clue_analysis" in response_data:
                            clue_analysis = response_data["clue_analysis"]
                            if isinstance(clue_analysis, list):
                                analysis_text = []
                                for analysis in clue_analysis:
                                    if isinstance(analysis, dict) and "reasoning" in analysis:
                                        analysis_text.append(f"Clue {analysis.get('clue_number', '')}: {analysis['reasoning']}")
                                if analysis_text:
                                    extracted_content = "\n".join(analysis_text)
                    except (json.JSONDecodeError, TypeError):
                        pass

                elif stage == "RefinementPrompt" or stage == "RefinementPrompt_Attempt_1":
                    if "### Analysis and New Reasoning Solution:" in response:
                        extracted_content = response.split("### Analysis and New Reasoning Solution:")[-1].strip()
                    elif "### Refined Reasoning:" in response:
                        extracted_content = response.split("### Refined Reasoning:")[-1].strip()

                if extracted_content and len(extracted_content) > 10:
                    clean_reasoning.append(extracted_content)

            elif isinstance(response, dict):
                if "reasoning" in response:
                    clean_reasoning.append(response["reasoning"])
                if "solution" in response:
                    final_solution = response["solution"]
                    has_valid_solution = True

    complete_reasoning = "\n\n".join(clean_reasoning)

    if not has_valid_solution:
        return {}

    return {
        "reasoning": complete_reasoning,
        "solution": final_solution
    }


def process_puzzle_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples = []
    puzzle_snippet = entry["puzzle_snippet"]

    for sample_result in entry["all_samples_results"]:
        response_data = extract_reasoning_from_trace(sample_result["trace"])

        if response_data and ("reasoning" in response_data or "solution" in response_data):
            is_correct = False

            if "is_correct" in sample_result:
                is_correct = sample_result["is_correct"]

            elif "solution_status" in sample_result:
                status = sample_result["solution_status"]
                is_correct = status in ["correct", "success", "Verified by ground_truth. All attributes correct."]

            elif "round_scores" in sample_result:
                round_scores = sample_result["round_scores"]
                if isinstance(round_scores, list) and round_scores:
                    best_score = max(round_scores)
                    is_correct = best_score == 1.0


            elif "best_score" in sample_result:
                is_correct = sample_result["best_score"] == 1.0

            elif "score_info" in sample_result and isinstance(sample_result["score_info"], dict):
                score = sample_result["score_info"].get("score", 0.0)
                is_correct = score == 1.0

            elif "status" in sample_result:
                sample_status = sample_result["status"]
                is_correct = sample_status in ["success", "correct"]

            reasoning = response_data.get("reasoning", "Generated solution for puzzle")
            solution = response_data.get("solution", {})

            assistant_content = f"{reasoning}\n\nSolution: {json.dumps(solution, ensure_ascii=False)}"

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": puzzle_snippet + "\n\nPlease provide your reasoning and solution for this puzzle."
                    },
                    {
                        "role": "assistant",
                        "content": 'Reasoning:\n\n' + assistant_content
                    }
                ],
                "is_correct": is_correct,
                "puzzle_id": entry["id"]
            }
            examples.append(example)

    return examples


def main() -> None:
    """Main function to extract SFT data."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = 'result_data_' + args.input_files[0].split('_job')[-1].split('_limit')[0]
    args.output_dir = os.path.join(args.output_dir, f"{fname}_time_{timestamp}_jobid_{job_id}")
    os.makedirs(args.output_dir, exist_ok=True)

    all_examples = []

    for input_file in args.input_files:
        print(f"Processing file: {input_file}")
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        examples = process_puzzle_entry(entry)
                        all_examples.extend(examples)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {input_file}: {e}")
                        continue

    print(f"Extracted {len(all_examples)} total examples")

    correct_examples = [ex for ex in all_examples if ex["is_correct"]]
    incorrect_examples = [ex for ex in all_examples if not ex["is_correct"]]

    correct_examples = correct_examples[:args.correct_limit]
    incorrect_examples = incorrect_examples[:args.incorrect_limit]
    all_examples = correct_examples + incorrect_examples
    
    random.shuffle(correct_examples)
    random.shuffle(incorrect_examples)
    random.shuffle(all_examples)
    
    print(f"Correct examples: {len(correct_examples)}")
    print(f"Incorrect examples: {len(incorrect_examples)}")
    print(f"Total examples: {len(all_examples)}")
    

    def create_dataset_split(examples: List[Dict[str, Any]]) -> DatasetDict:
        """Create train/test split for a list of examples."""
        dataset = Dataset.from_list(examples)

        if len(examples) <= 1:

            return DatasetDict({
                "train": dataset,
                "test": dataset.select([])
            })
        elif len(examples) == 2:
            return DatasetDict({
                "train": dataset.select([0]),
                "test": dataset.select([1])
            })

        split = dataset.train_test_split(test_size=1.0 - args.split_ratio, seed=42)
        return split

    print("\nSaving combined dataset...")
    combined_split = create_dataset_split(all_examples)
    combined_split.save_to_disk(os.path.join(args.output_dir, "combined"))

    if correct_examples:
        print("Saving correct examples dataset...")
        correct_split = create_dataset_split(correct_examples)
        correct_split.save_to_disk(os.path.join(args.output_dir, "correct_only"))

    if incorrect_examples:
        print("Saving incorrect examples dataset...")
        incorrect_split = create_dataset_split(incorrect_examples)
        incorrect_split.save_to_disk(os.path.join(args.output_dir, "incorrect_only"))

    print("\nSaving JSONL files...")
    jsonl_dir = os.path.join(args.output_dir, "jsonl")
    os.makedirs(jsonl_dir, exist_ok=True)

    print("Saving combined JSONL files...")
    with open(os.path.join(jsonl_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for example in combined_split["train"]:
            example_to_save = example.copy()
            example_to_save.pop("puzzle_id", None)
            f.write(json.dumps(example_to_save, ensure_ascii=False) + "\n")

    with open(os.path.join(jsonl_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for example in combined_split["test"]:
            example_to_save = example.copy()
            example_to_save.pop("puzzle_id", None)
            f.write(json.dumps(example_to_save, ensure_ascii=False) + "\n")

    if correct_examples:
        print("Saving correct-only JSONL files...")
        correct_jsonl_dir = os.path.join(jsonl_dir, "correct_only")
        os.makedirs(correct_jsonl_dir, exist_ok=True)

        with open(os.path.join(correct_jsonl_dir, "train.jsonl"), "w", encoding="utf-8") as f:
            for example in correct_split["train"]:
                example_to_save = example.copy()
                example_to_save.pop("puzzle_id", None)
                f.write(json.dumps(example_to_save, ensure_ascii=False) + "\n")

        with open(os.path.join(correct_jsonl_dir, "test.jsonl"), "w", encoding="utf-8") as f:
            for example in correct_split["test"]:
                example_to_save = example.copy()
                example_to_save.pop("puzzle_id", None)
                f.write(json.dumps(example_to_save, ensure_ascii=False) + "\n")

    if incorrect_examples:
        print("Saving incorrect-only JSONL files...")
        incorrect_jsonl_dir = os.path.join(jsonl_dir, "incorrect_only")
        os.makedirs(incorrect_jsonl_dir, exist_ok=True)

        with open(os.path.join(incorrect_jsonl_dir, "train.jsonl"), "w", encoding="utf-8") as f:
            for example in incorrect_split["train"]:
                example_to_save = example.copy()
                example_to_save.pop("puzzle_id", None)
                f.write(json.dumps(example_to_save, ensure_ascii=False) + "\n")

        with open(os.path.join(incorrect_jsonl_dir, "test.jsonl"), "w", encoding="utf-8") as f:
            for example in incorrect_split["test"]:
                example_to_save = example.copy()
                example_to_save.pop("puzzle_id", None)
                f.write(json.dumps(example_to_save, ensure_ascii=False) + "\n")

    print(f"\nAll datasets saved to: {args.output_dir}")
    print("Extraction completed successfully!")


if __name__ == "__main__":
    main()