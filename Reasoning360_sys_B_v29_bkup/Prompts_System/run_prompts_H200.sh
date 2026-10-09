#!/bin/bash -l

#SBATCH -J Train_Eval #job name
#SBATCH -p gpu-H200 # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 30  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j.out
#SBATCH --error=./all_logs/%j.err
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



N_SAMPLES=$1
DATA=$2
ATTEMPTS=$3
MODE=$4
LIMIT=$5
REMOVE_CHKPT=$6
CHKPT_PATH=$7
python ./Prompts_System/prompt_iterate_chkpt.py --n_samples ${N_SAMPLES} --data_ ${DATA} --max_attempts ${ATTEMPTS} --mode ${MODE} --limit ${LIMIT} --remove_chkpt ${REMOVE_CHKPT} --model_path ${CHKPT_PATH}


#DATA_CLASS=$1
#DATA_PATH=$2
#EPOCH=$3
#python ./Prompts_System/sft/sft_train_simple.py --select_ ${DATA_CLASS} --data_path ${DATA_PATH} --num_train_epochs ${EPOCH}












#N_SAMPLES=$1
#MODE=$2
#python ./Prompts_System/prompt_iterate.py --n_samples ${N_SAMPLES} --data_ 'small' --max_attempts 15 --mode ${MODE} --limit -1

#DATA_CLASS=$1
#DATA_PATH=$2
#EPOCH=$3
#EVAL_STEP=$4
#SAVE_STEP=$5
#python ./Prompts_System/sft/sft_train_simple.py --select_ ${DATA_CLASS} --data_path ${DATA_PATH} --num_train_epochs ${EPOCH} --eval_steps ${EVAL_STEP} --save_steps ${SAVE_STEP}




#N_SAMPLES=$1
#DATA=$2
#ATTEMPTS=$3
#MODE=$4
#LIMIT=$5
#REMOVE_CHKPT=$6
#CHKPT_PATH=$7
#python ./Prompts_System/prompt_iterate_chkpt.py --n_samples ${N_SAMPLES} --data_ ${DATA} --max_attempts ${ATTEMPTS} --mode ${MODE} --limit ${LIMIT} --remove_chkpt ${REMOVE_CHKPT} --model_path ${CHKPT_PATH}


#python ./Prompts_System/prompt_iterate.py --n_samples ${N_SAMPLES} --data_ ${DATA} --max_attempts ${ATTEMPTS} --mode ${MODE} --limit -1 --model_path '/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_correct_only_20251212_205544_jobid_236829/checkpoint-1470'


#N_SAMPLES=$1
#DATA=$2
#ATTEMPTS=$3
#MODE=$4
#python ./Prompts_System/prompt_iterate.py --n_samples ${N_SAMPLES} --data_ ${DATA} --max_attempts ${ATTEMPTS} --mode ${MODE} --limit -1 --model_path '/export/home/asifali/Reasoning360/Prompts_Checkpoint/data_correct_only_20251212_205544_jobid_236829/checkpoint-1470'
