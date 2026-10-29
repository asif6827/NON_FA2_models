#!/bin/bash -l

#SBATCH -J Generate_math_ALL_Qwen3_0-6B_Instruct #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:4 #number of gpus needed, default is 1
#SBATCH -c 30  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%x-%j-slurm.out
#SBATCH --error=./all_logs/%x-%j-slurm.err
#SBATCH --mail-user=asif6827@gmail.com


#!/usr/bin/env bash
set -euo pipefail
module load cuda12.4/toolkit
nvidia-smi
source activate Reason360_v1
PY=python
export CUDA_VISIBLE_DEVICES=0,1
unset ROCR_VISIBLE_DEVICES
export PYTHONPATH="/export/home/asifali/Reasoning360:${PYTHONPATH:-}"

nvidia-smi
#########################################################
# Add Data Paths...!


python ./my_codes/eval_qwen3_0_6B_Instruct.py
