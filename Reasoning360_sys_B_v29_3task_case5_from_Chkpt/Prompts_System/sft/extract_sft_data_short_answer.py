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
    parser = argparse.ArgumentParser(description="Extract SFT data from puzzle results")
    parser.add_argument("--input_files", nargs="+", required=True, help="Path to input JSONL files containing puzzle results")
    parser.add_argument("--output_dir", default="/export/home/asifali/Reasoning360/Prompts_SFT_data/", help="Directory to save the extracted SFT datasets")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/test split ratio (default: 0.9)")
    parser.add_argument("--correct_limit", type=int, default=-1, help="Train/test split ratio (default: 0.9)")
    parser.add_argument("--incorrect_limit", type=int, default=-1, help="Train/test split ratio (default: 0.9)")

    return parser.parse_args()


def extract_reasoning_from_trace(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trace:
        return {}

    # trace
    stages = []
    final_solution = {}
    has_valid_solution = False

    for i, step in enumerate(trace):
        stage_data = {
            "stage": step.get("stage", ""),
            "type": "unknown",
            "content": "",
            "is_correct": None,
            "verification_ratio": 0.0,
            "reasoning": "",
            "solution": {}
        }

        if "result" in step:
            result = step["result"]
            if isinstance(result, list) and len(result) == 2:
                parsed_puzzle, constraints = result
                stage_data["type"] = "initialization"
                stage_data["content"] = "Generated meta information and constraints for puzzle solution."
                stage_data["solution"] = {
                    "meta": parsed_puzzle,
                    "constraints": constraints
                }
                stages.append(stage_data)

        elif "response" in step:
            response = step["response"]
            stage = step.get("stage", "")

            if isinstance(response, str):
                if stage == "SolutionPrompt" or stage == "SolutionPrompt_Initial":
                    stage_data["type"] = "answer"
                    try:
                        response_data = json.loads(response)
                        if isinstance(response_data, dict):
                            if "reasoning" in response_data:
                                stage_data["reasoning"] = response_data["reasoning"]
                                stage_data["content"] = response_data["reasoning"]
                            if "solution" in response_data:
                                stage_data["solution"] = response_data["solution"]
                                final_solution = response_data["solution"]
                                has_valid_solution = True
                    except (json.JSONDecodeError, TypeError):
                        if "Reasoning:" in response:
                            stage_data["reasoning"] = response.split("Reasoning:")[-1].strip()
                            stage_data["content"] = stage_data["reasoning"]
                        elif "### Analysis" in response:
                            stage_data["reasoning"] = response.split("### Analysis")[-1].strip()
                            stage_data["content"] = stage_data["reasoning"]
                    stages.append(stage_data)

                elif stage == "VerificationPrompt" or stage == "VerificationPrompt_Attempt_1":
                    stage_data["type"] = "verification"
                    try:
                        response_data = json.loads(response)
                        if isinstance(response_data, dict):
                            if "clue_analysis" in response_data:
                                clue_analysis = response_data["clue_analysis"]
                                if isinstance(clue_analysis, list):
                                    analysis_text = []
                                    valid_count = 0
                                    total_count = len(clue_analysis)
                                    for analysis in clue_analysis:
                                        if isinstance(analysis, dict):
                                            if "reasoning" in analysis:
                                                analysis_text.append(f"Clue {analysis.get('clue_number', '')}: {analysis['reasoning']}")
                                            if analysis.get("valid", False):
                                                valid_count += 1
                                    stage_data["content"] = "\n".join(analysis_text)
                                    stage_data["verification_ratio"] = valid_count / total_count if total_count > 0 else 0.0
                                    stage_data["is_correct"] = stage_data["verification_ratio"] == 1.0
                            elif "is_verified" in response_data:
                                stage_data["is_correct"] = response_data["is_verified"]
                                stage_data["verification_ratio"] = 1.0 if response_data["is_verified"] else 0.0
                                stage_data["content"] = "Verification result: " + ("Correct" if response_data["is_verified"] else "Incorrect")
                    except (json.JSONDecodeError, TypeError):
                        pass
                    stages.append(stage_data)

                elif stage == "RefinementPrompt" or stage == "RefinementPrompt_Attempt_1":
                    stage_data["type"] = "refinement"
                    refinement_content = ""
                    if "### Analysis and New Reasoning Solution:" in response:
                        refinement_content = response.split("### Analysis and New Reasoning Solution:")[-1].strip()
                    elif "### Refined Reasoning:" in response:
                        refinement_content = response.split("### Refined Reasoning:")[-1].strip()

                    # Try to extract new solutions from refined content.remove_chkpt
                    try:
                        refined_data = json.loads(refinement_content)
                        if isinstance(refined_data, dict):
                            if "new_reasoning" in refined_data:
                                stage_data["reasoning"] = refined_data["new_reasoning"]
                                stage_data["content"] = refined_data["new_reasoning"]
                            if "new_solution" in refined_data:
                                stage_data["solution"] = refined_data["new_solution"]
                                final_solution = refined_data["new_solution"]
                                has_valid_solution = True
                            elif "solution" in refined_data:
                                stage_data["solution"] = refined_data["solution"]
                                final_solution = refined_data["solution"]
                                has_valid_solution = True
                    except (json.JSONDecodeError, TypeError):
                        stage_data["content"] = refinement_content
                        stage_data["reasoning"] = refinement_content

                    stages.append(stage_data)

            elif isinstance(response, dict):
                if stage == "SolutionPrompt" or stage == "SolutionPrompt_Initial":
                    stage_data["type"] = "answer"
                    if "reasoning" in response:
                        stage_data["reasoning"] = response["reasoning"]
                        stage_data["content"] = response["reasoning"]
                    if "solution" in response:
                        stage_data["solution"] = response["solution"]
                        final_solution = response["solution"]
                        has_valid_solution = True
                    stages.append(stage_data)
                elif stage == "VerificationPrompt" or stage == "VerificationPrompt_Attempt_1":
                    stage_data["type"] = "verification"
                    if "is_verified" in response:
                        stage_data["is_correct"] = response["is_verified"]
                        stage_data["verification_ratio"] = 1.0 if response["is_verified"] else 0.0
                        stage_data["content"] = "Verification result: " + ("Correct" if response["is_verified"] else "Incorrect")
                    stages.append(stage_data)

    result = {
        "stages": stages,
        "final_solution": final_solution,
        "has_valid_solution": has_valid_solution
    }

    return result


def process_puzzle_entry(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples = []
    puzzle_snippet = entry["puzzle_snippet"]

    for sample_result in entry["all_samples_results"]:
        trace_data = extract_reasoning_from_trace(sample_result["trace"])

        if not trace_data or not trace_data.get("stages"):
            continue

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

        stages = trace_data["stages"]
        final_solution = trace_data["final_solution"]

        valid_stages = [stage for stage in stages if stage["type"] != "unknown"]

        if not valid_stages:
            continue

        first_answer_idx = -1
        first_verification_idx = -1

        for i, stage in enumerate(valid_stages):
            if stage["type"] == "answer" and first_answer_idx == -1:
                first_answer_idx = i
            elif stage["type"] == "verification" and first_verification_idx == -1:
                first_verification_idx = i

        if first_answer_idx != -1:

            user_input = puzzle_snippet

            assistant_output = ""
            answer_stage = valid_stages[first_answer_idx]
            if answer_stage["solution"]:
                assistant_output += f"Solution: {json.dumps(answer_stage['solution'], ensure_ascii=False)}" + "\n\n"
            if answer_stage["reasoning"]:
                assistant_output += 'Reasoning: ' +  answer_stage['reasoning'] + "\n\n"

            if first_verification_idx != -1:
                verification_stage = valid_stages[first_verification_idx]

                if verification_stage["content"]:
                    assistant_output += f"Verification: {verification_stage['content']}" + "\n\n"

            if final_solution and is_correct:
                #assistant_output += f"Final Answer: {json.dumps(final_solution, ensure_ascii=False)}" + "\n\n"
                assistant_output += "Verification: This answer is correct."

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    },
                    {
                        "role": "assistant",
                        "content": assistant_output.strip()
                    }
                ],
                "is_correct": is_correct,
                "puzzle_id": entry["id"],
                "format_type": "question_to_answer_verification"
            }
            examples.append(example)

            user_input = puzzle_snippet

            assistant_output = ""
            if answer_stage["solution"]:
                user_input += f"Solution: {json.dumps(answer_stage['solution'], ensure_ascii=False)}" + "\n\n"
            answer_stage = valid_stages[first_answer_idx]

            #if answer_stage["reasoning"]:
            #    assistant_output += answer_stage["reasoning"] + "\n\n"

            if first_verification_idx != -1:
                verification_stage = valid_stages[first_verification_idx]

                if verification_stage["content"]:
                    assistant_output += f"Verification: {verification_stage['content']}" + "\n\n"

            if final_solution and is_correct:
                #assistant_output += f"Final Answer: {json.dumps(final_solution, ensure_ascii=False)}" + "\n\n"
                assistant_output += "Verification: This answer is correct."

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    },
                    {
                        "role": "assistant",
                        "content": assistant_output.strip()
                    }
                ],
                "is_correct": is_correct,
                "puzzle_id": entry["id"],
                "format_type": "question_to_answer_verification"
            }
            examples.append(example)


        if first_answer_idx != -1 and first_verification_idx != -1:
            verification_stage = valid_stages[first_verification_idx]

            if verification_stage["content"] or verification_stage["verification_ratio"] > 0:
                user_input = puzzle_snippet + "\n\n"
                answer_stage = valid_stages[first_answer_idx]
                if answer_stage["reasoning"]:
                    user_input += answer_stage["reasoning"] + "\n\n"
                if answer_stage["solution"]:
                    user_input += f"Solution: {json.dumps(answer_stage['solution'], ensure_ascii=False)}"

                if verification_stage["content"]:
                    assistant_output = f"Verification: {verification_stage['content']}"
                else:
                    if verification_stage["is_correct"] or verification_stage["verification_ratio"] == 1.0:
                        assistant_output = "Verification: This answer is correct."
                    elif verification_stage["verification_ratio"] > 0.5:
                        assistant_output = "Verification: This answer is partially correct."
                    else:
                        assistant_output = "Verification: This answer is incorrect."

                example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input.strip()
                        },
                        {
                            "role": "assistant",
                            "content": assistant_output.strip()
                        }
                    ],
                    "is_correct": is_correct,
                    "puzzle_id": entry["id"],
                    "format_type": "question_answer_to_verification"
                }
                examples.append(example)

        answer_verification_pairs = []
        current_answer = None

        for stage in valid_stages:
            if stage["type"] == "answer":
                if current_answer:
                    answer_verification_pairs.append((current_answer, None))
                current_answer = stage
            elif stage["type"] == "verification" and current_answer:
                answer_verification_pairs.append((current_answer, stage))
                current_answer = None

        if current_answer:
            answer_verification_pairs.append((current_answer, None))

        if len(answer_verification_pairs) > 1:

            user_input = puzzle_snippet + "\n\n"

            for i in range(len(answer_verification_pairs) - 1):
                answer, verification = answer_verification_pairs[i]
                if answer["reasoning"]:
                    user_input += answer["reasoning"] + "\n\n"
                if answer["solution"]:
                    user_input += f"Solution: {json.dumps(answer['solution'], ensure_ascii=False)}" + "\n\n"
                if verification and verification["content"]:
                    user_input += f"Verification: {verification['content']}" + "\n\n"

            assistant_output = ""

            last_answer, last_verification = answer_verification_pairs[-1]
            if last_answer["solution"]:
                assistant_output += f"Solution: {json.dumps(last_answer['solution'], ensure_ascii=False)}" + "\n\n"
            if last_answer["reasoning"]:
                assistant_output += last_answer["reasoning"] + "\n\n"
            if last_verification and last_verification["content"]:
                assistant_output += f"Verification: {last_verification['content']}" + "\n\n"

            if final_solution and is_correct:
                #assistant_output += f"Final Answer: {json.dumps(final_solution, ensure_ascii=False)}" + "\n\n"
                assistant_output += "Verification: This answer is correct."

            example = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input.strip()
                    },
                    {
                        "role": "assistant",
                        "content": assistant_output.strip()
                    }
                ],
                "is_correct": is_correct,
                "puzzle_id": entry["id"],
                "format_type": "error_chain"
            }
            examples.append(example)

        for i in range(len(answer_verification_pairs) - 1):
            answer, verification = answer_verification_pairs[i]

            is_answer_correct = verification["is_correct"] if verification else False

            if not is_answer_correct:
                user_input = puzzle_snippet + "\n\n"
                if answer["reasoning"]:
                    user_input += answer["reasoning"] + "\n\n"
                if answer["solution"]:
                    user_input += f"Solution: {json.dumps(answer['solution'], ensure_ascii=False)}"

                assistant_output = ""

                if verification and verification["content"]:
                    assistant_output += f"Verification: {verification['content']}" + "\n\n"
                else:
                    assistant_output += "Verification: This answer is wrong." + "\n\n"

                refinement_content = ""
                for j in range(first_answer_idx + 1, len(valid_stages)):
                    if valid_stages[j]["type"] == "refinement":
                        refinement_content = valid_stages[j]["content"]
                        break

                if refinement_content:
                    assistant_output += f"Refinement Reasoning: {refinement_content}" + "\n\n"

                if final_solution and is_correct:
                    assistant_output += f"Correct Answer: {json.dumps(final_solution, ensure_ascii=False)}" + "\n\n"
                    assistant_output += "Verification: This answer is correct."

                example = {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input.strip()
                        },
                        {
                            "role": "assistant",
                            "content": assistant_output.strip()
                        }
                    ],
                    "is_correct": is_correct,
                    "puzzle_id": entry["id"],
                    "format_type": "wrong_answer_refinement"
                }
                examples.append(example)

    return examples


def main() -> None:
    """Main function to extract SFT data."""
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = 'result_data_' + args.input_files[0].split('_job')[-1].split('_limit')[0]
    args.output_dir = os.path.join(args.output_dir, f"{fname}_time_{timestamp}_shortANS_jobid_{job_id}")
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
