#!/bin/bash -l

#SBATCH -J Data_process #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:4 #number of gpus needed, default is 1
#SBATCH -c 30  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x.out
#SBATCH --error=./all_logs/%j-%x.err
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

#!/usr/bin/env bash
set -euo pipefail
module load cuda12.4/toolkit
source activate Reasoning360
PY=python
export PYTHONUNBUFFERED=1  # Force real-time Python output; use tee for live logs
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES

#### MY Parameters ####
export USE_Thinking=0

nvidia-smi
export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"
export WANDB_API_KEY="64305b88cc27033d4132d6ce147ecce132e6955d"


export PYTHONPATH="/export/home/asifali/Reasoning360:${PYTHONPATH:-}"


#N_SAMPLES=$1
#MODE=$2
#python ./Prompts_System/prompt_iterate.py --n_samples ${N_SAMPLES} --data_ 'small' --max_attempts 15 --mode ${MODE} --limit -1
#python ./Prompts_System/prompt_iterate.py --n_samples ${N_SAMPLES} --data_ 'small' --max_attempts 15 --mode ${MODE} --limit -1


#python ./Prompts_System/sft/sft_train_simple.py --data_ 'combined' --num_train_epochs 1 --eval_steps 100 --save_steps 100



#N_SAMPLES=$1
#DATA=$2
#ATTEMPTS=$3
#MODE=$4
#PATH=$5
#python ./Prompts_System/prompt_iterate.py --n_samples ${N_SAMPLES} --data_ ${DATA} --max_attempts ${ATTEMPTS} --mode ${MODE} --limit -1 --model_path ${PATH}


python ./Prompts_System/sft/extract_sft_data_short_answer.py --input_files '/export/home/asifali/Reasoning360/Prompts_Results/ZebraPuzzle_1000_main_results/results_solution_small_n_20251212_100814_jobid_236791_limit__full.jsonl'


#python ./Prompts_System/sft/extract_sft_data.py --input_files '/export/home/asifali/Reasoning360/Prompts_Results/ZebraPuzzle_1000_main_results/results_solution_small_n_20251212_100814_jobid_236791_limit__full.jsonl'