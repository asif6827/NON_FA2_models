#!/bin/bash -l

#SBATCH -J Prompt-Verify-7B-small-sample #job name
#SBATCH -p gpu-A100 # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 30  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH -A A100
#SBATCH -q a100_qos
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


#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v2.py
#echo "puzzle_eval_SFT_v1_3B_medium.py"

#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_3B_medium_sample.py

#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_7B_medium.py



#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_3B_medium_sample.py

#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_7B_medium_sample.py

#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_3B_large_sample.py

#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_7B_large_sample.py

#python ./Prompts_Verify_11Dec2025/puzzle_eval_SFT_v1_3B_XL_sample.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_3B_small_1.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_3B_small_2.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_3B_small_3.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_3B_small_1.py

#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_7B_small_1.py
#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_7B_small_2.py
#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_7B_small_3.py
#python ./Prompts_Verify_11Dec2025/Z3/puzzle_eval_add_pass@k_z3_rl_7B_small_10.py


#python ./Prompts_Verify_11Dec2025/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 1 --data_ 'medium' --limit -1

#python ./Prompts_Verify_11Dec2025/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 3 --data_ 'medium' --limit -1

#python ./Prompts_Verify_11Dec2025/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 5 --data_ 'medium' --limit -1

#python ./Prompts_Verify_11Dec2025/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 8 --data_ 'medium' --limit -1

python ./Prompts_Verify/Prompts/puzzle_eval_add_pass@k_z3_constraint.py --n_samples 10 --data_ 'medium' --limit -1


