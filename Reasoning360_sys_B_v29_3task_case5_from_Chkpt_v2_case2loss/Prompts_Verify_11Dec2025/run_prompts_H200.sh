#!/bin/bash -l

#SBATCH -J Prompt-Verify #job name
#SBATCH -p gpu-H200 # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 30  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

#!/usr/bin/env bash
set -euo pipefail
module load cuda12.4/toolkit
source activate Reasoning360
PY=python
export PYTHONUNBUFFERED=1  # Force real-time Python output; use tee for live logs
export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES

#### MY Parameters ####
export USE_Thinking=0


nvidia-smi
export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"
export WANDB_API_KEY="64305b88cc27033d4132d6ce147ecce132e6955d"

export PYTHONPATH="/export/home/asifali/Reasoning360:${PYTHONPATH:-}"

#python ./Prompts_Verify_11Dec2025/prompt_SFT_data_v2.py
#python ./Prompts_Verify_11Dec2025/prompt_SFT_data_v4.py
#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1.py
#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_3B_small.py
#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_7B_medium_sample.py
#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_3B_small_sample.py

#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_3B_debug.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_3B_small.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_3B_medium.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_7B_small.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_7B_medium.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_3B_large.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_7B_large.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_3B_XL.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_7B_XL.py



#python ./Prompts_Verify_11Dec2025/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 2 --data_ 'medium' --limit -1

#python ./Prompts_Verify_11Dec2025/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 4 --data_ 'medium' --limit -1

#python ./Prompts_Verify_11Dec2025/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 7 --data_ 'medium' --limit -1

#python ./Prompts_Verify_11Dec2025/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 10 data_ 'medium' --limit -1


#python ./Prompts_System/sft/extract_sft_data.py --input-files /export/home/asifali/Reasoning360/Prompts_Results/ZebraPuzzle_1000_main_results/

python ./Prompting_System/sft/extract_sft_data.py --input-files /export/home/asifali/Reasoning360/Prompts_Results/ZebraPuzzle_1000_main_results/results_constraint_small_n_20251211_003645_jobid_236548_limit__50.jsonl --output-dir /export/home/asifali/Reasoning360/Prompts_SFT_data

#python ./Prompts_System/prompt_iterate.py --mode constraint --temperature 0.7 --n_samples 5 --data_ 'small' --limit 50