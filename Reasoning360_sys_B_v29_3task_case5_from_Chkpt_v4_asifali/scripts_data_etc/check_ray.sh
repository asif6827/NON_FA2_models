#!/bin/bash -l

#SBATCH -J hello_A100 #job name
#SBATCH -p gpu-A100 # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 8  #number of CPUs needed, default is 1 
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_case5_from_Chkpt_asifali/all_logs/%x-%j-slurm.out
#SBATCH --error=/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_case5_from_Chkpt_asifali/all_logs/%x-%j-slurm.err
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

nvidia-smi
export TRANSFORMERS_CACHE="/export/alt-insight/HF_cache"
export HF_HOME="/export/alt-insight/HF_cache"
export HF_DATASETS_CACHE="/export/alt-insight/HF_cache"

source activate Reason360
python ./scripts_data_etc/check_ray.py
nvidia-smi
