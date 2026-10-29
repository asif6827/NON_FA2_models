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

job_id = os.getenv("SLURM_JOB_ID")
print("SLURM Job ID:", job_id)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Puzzle Solving System with Z3 Verification")
    
    # Model and data configuration
    parser.add_argument("--model_path", type=str, default='/export/home/asifali/HF_cache/Qwen2.5-7B-Instruct', help="LLM")
    parser.add_argument("--data_",  type=str, default="small", help="(.json, .jsonl or .parquet)")
    parser.add_argument("--output_dir", type=str, default="/export/home/asifali/Reasoning360/Prompts_Results/ZebraPuzzle_1000_main_results", help="output directory")
    
    # Verification mode selection
    parser.add_argument("--mode", type=str, choices=["solution", "constraint"], default="solution",  help=" solution or constraint")
    
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
    #args = parser.parse_args()


    return parser.parse_args()

def main() -> None:
    """
    Main function to run the puzzle solving system.
    """
    # Parse arguments
    args = parse_args()

    if args.data_ == "small":
        args.data_path = "/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_small_320.json"
    elif args.data_ == "medium":
        args.data_path = "/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_medium_280.json"
    elif args.data_ == "large":
        args.data_path = "/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_large_200.json"
    elif args.data_ == "XL":
        args.data_path = "/export/home/asifali/HF_cache/ZebraLogic/Zebra_Puzzle_xl_200.json"


    print("Program Arguments = {}".format(args))

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    import logging
    log_file = os.path.join(args.output_dir, f"puzzle_solver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger(name="puzzle_solver", log_file=log_file, level=getattr(logging, args.log_level))
    
    info(f"[System] Starting Puzzle Solving System in {args.mode} mode")
    info(f"[System] Model: {args.model_path}")
    info(f"[System] Data: {args.data_path}")
    info(f"[System] Output: {args.output_dir}")
    info(f"[System] Config: n_samples={args.n_samples}, max_attempts={args.max_attempts}")
    
    # Load dataset
    dataset = load_local_dataset(args.data_path)
    if not dataset:
        error("[System] Failed to load dataset")
        return
    
    # Limit dataset if specified
    if args.limit > 0:
        dataset = dataset[:args.limit]
        info(f"[System] Limited to {args.limit} samples")
    
    # Initialize verification system based on mode
    if args.mode == "solution":
        info(f"[System] Initializing SolutionVerifier")
        system = SolutionVerifier(
            model_path=args.model_path,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer_mode=args.tokenizer_mode,
            refinement_include_z3=args.refinement_include_z3,
            refinement_include_accuracy=args.refinement_include_accuracy,
            refinement_include_verification=args.refinement_include_verification
        )
    else:  # constraint mode
        info(f"[System] Initializing ConstraintVerifier")
        system = ConstraintVerifier(
            model_path=args.model_path,
            max_attempts=args.max_attempts,
            temperature=args.temperature,
            top_p=args.top_p,
            tokenizer_mode=args.tokenizer_mode
        )
    
    info(f"\n" + "=" * 50)
    info(f"[System] Evaluating with max_attempts={args.max_attempts}")
    info("=" * 50)
    
    all_results: List[Dict[str, Any]] = []
    info(f"[System] Starting Verification Loop...")
    
    # Generate output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    limit_str = f"{args.limit}" if args.limit > 0 else "full"


    output_file = os.path.join(args.output_dir, f"results_{args.mode}_pass_{args.n_samples}_{args.data_}_n_{timestamp}_jobid_{job_id}_limit_{limit_str}.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing puzzle={}"):
            pid = item.get("id", f"sample_{idx}")
            puzzle_text = get_puzzle_text(item)
            if not puzzle_text:
                warning(f"[Data] Skipping sample {pid} due to missing puzzle text")
                continue

            ground_truth = item.get("solution")

            tqdm.write(f"Processing idx:{idx}, pid:{pid} ... ")
            info(f"[Processing idx:{idx}, pid: {pid}")
            
            try:
                result = system.solve_puzzle(pid, puzzle_text, ground_truth=ground_truth, n_samples=args.n_samples)
                #tqdm.write(f"Puzzle Idx: {idx}, Status: {result['status']}, Attempts: {result['attempts_used']}, Correct Samples: {result['correct_samples']}/{result['total_samples']}")
                tqdm.write(f"Puzzle Idx: {idx}, Status: {result['status']}, Attempts: {result['attempts_used']}, Correct Samples: {result['correct_samples']}/{result['total_samples']}")
                all_results.append(result)

                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            except Exception as e:
                error(f"[Error] Failed to process {pid}: {e}")
                continue
    
    info(f"[System] All results saved to {output_file}")
    
    # Generate and print summary report
    if all_results:
        report = generate_summary_report(all_results, args.max_attempts, args.n_samples)
        print_summary_report(report)
        
        # Save summary report
        limit_str = f"{args.limit}" if args.limit > 0 else "full"
        summary_file = os.path.join(args.output_dir, f"stats_all_{args.mode}_pass_{args.n_samples}_{args.data_}_{timestamp}_jobid_{job_id}_limit_{limit_str}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        info(f"[System] Summary report saved to {summary_file}")



        #stats_output = os.path.join(args.output_dir, f"stats_all_{args.data_}_n{timestamp}_jobid_{job_id}_limit_{limit_str}.json")

        # Save aggregated results
        aggregated = aggregate_results(all_results)
        aggregated_file = os.path.join(args.output_dir, f"aggregated_{args.mode}_pass_{args.n_samples}_{args.data_}_{timestamp}_jobid_{job_id}_limit_{limit_str}.json")
        with open(aggregated_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)
        info(f"[System] Aggregated results saved to {aggregated_file}")
    
    info(f"[System] Puzzle Solving System completed successfully")

if __name__ == "__main__":
    main()
