#!/bin/bash -l

#SBATCH -J Download-Qwen-2-5_7B #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 8  #number of CPUs needed, default is 1 
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x-slurm.out
#SBATCH --error=./all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

nvidia-smi
export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"

source activate Reasoning360
python ./scripts_data_etc/download_hub_Qwen2.py
nvidia-smi