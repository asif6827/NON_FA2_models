#!/bin/bash -l

#SBATCH -J Download-SmolLM3-3B-model #job name
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
export HF_TOKEN="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"


source activate zebrapuzzles

python ./scripts_data_etc/download_hub_SmolLM3_3B.py
#python ./scripts_data_etc/download_hub_Qwen3.py

nvidia-smi