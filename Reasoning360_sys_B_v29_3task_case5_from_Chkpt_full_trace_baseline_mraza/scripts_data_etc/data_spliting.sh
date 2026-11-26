#!/bin/bash -l

#SBATCH -J data_spliting #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 8  #number of CPUs needed, default is 1 
#SBATCH --mem 32GB #amount of memory needed, default
#SBATCH --output=/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_3task_case5_from_Chkpt_full_trace_baseline_mraza/all_logs/%j-%x-slurm.out
#SBATCH --error=/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_3task_case5_from_Chkpt_full_trace_baseline_mraza/all_logs/%j-%x-slurm.err
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

nvidia-smi
export TRANSFORMERS_CACHE="/export/alt-insight/HF_cache"
export HF_HOME="/export/alt-insight/HF_cache"
export HF_DATASETS_CACHE="/export/alt-insight/HF_cache"

source activate Reasoning360
#python data_spliting_panther.py

python ./scripts_data_etc/data_spliting_zebra_guru.py
nvidia-smi