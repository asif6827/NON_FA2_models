import json
import argparse
import os
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any

from utils.dataset import load_local_dataset, get_puzzle_text
from verification_system.solution_verifier import SolutionVerifier
from verification_system.constraint_verifier import ConstraintVerifier
from metrics import generate_summary_report, print_summary_report, aggregate_results
from logger import get_logger, info, debug, warning, error

#job_id = os.getenv("SLURM_JOB_ID")
#print("SLURM Job ID:", job_id)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Puzzle Solving System with Z3 Verification")

    # Model and data configuration
    parser.add_argument("--model_path", type=str, default='/export/home/asifali/HF_cache/Qwen2.5-7B-Instruct', help="LLM")
    parser.add_argument("--data_", type=str, default="small", help="(.json, .jsonl or .parquet)")
    parser.add_argument("--output_dir", type=str, default="/export/home/asifali/Reasoning360/Prompts_Results/ZebraPuzzle_1000_main_results", help="output directory")

    # Verification mode selection
    parser.add_argument("--mode", type=str, choices=["solution", "constraint"], default="solution", help=" solution or constraint")

    # Model generation parameters
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (default: 0.0)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter (default: 0.9)")
    parser.add_argument("--tokenizer_mode", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=-1, help="Only the first K data is used for quick debugging, -1 indicates running the full amount")
    parser.add_argument("--max_attempts", type=int, default=15, help="Maximum number of refinement attempts per sample")

    # Refinement feedback control arguments (only for solution mode)
    parser.add_argument("--refinement_include_z3", type=bool, default=True, help="Whether to include Z3 check results in refinement feedback (default: False)")
    parser.add_argument("--refinement_include_accuracy", type=bool, default=False, help="Whether to include accuracy information in refinement feedback (default: False)")
    parser.add_argument("--refinement_include_verification", type=bool, default=False, help="Whether to include verification results in refinement feedback (default: True)")

    # Logging configuration
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level")
    # args = parser.parse_args()

    return parser.parse_args()


def main() -> None:
    fname = "/home/asif/data3/Codes_QCRI/Reasoning360/Prompts_Results/ZebraPuzzle_1000_main_results/results_solution_small_n_20251211_235354_jobid_236769_limit__5.jsonl"
    all_results = [json.loads(line) for line in open(fname)]

    n_samples = 1
    max_attempts = 10
    # Generate and print summary report
    if all_results:
        report = generate_summary_report(all_results, max_attempts, n_samples)
        print_summary_report(report)



        # stats_output = os.path.join(args.output_dir, f"stats_all_{args.data_}_n{timestamp}_jobid_{job_id}_limit_{limit_str}.json")

        # Save aggregated results
        aggregated = aggregate_results(all_results)
        print(aggregated)

    info(f"[System] Puzzle Solving System completed successfully")


if __name__ == "__main__":
    main()