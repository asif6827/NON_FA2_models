#!/bin/bash -l

#SBATCH -J Data-Processing-v3task_v3 #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 8  #number of CPUs needed, default is 1 
#SBATCH --mem 32GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com


set -e

module load cuda12.4/toolkit

nvidia-smi
export TRANSFORMERS_CACHE="/export/alt-insight/HF_cache"
export HF_HOME="/export/alt-insight/HF_cache"
export HF_DATASETS_CACHE="/export/alt-insight/HF_cache"

source activate zebrapuzzles


python "/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v4_data_models/data_preprocess/logic/v3task_v3/our_pre_process_zebrapuzzle_to_guru_parsed_v3task_v3_MLXL.py"

nvidia-smi
