#!/bin/bash -l

#SBATCH -J test-verl #job name
#SBATCH -p gpu-H200 # queue used
#SBATCH --gres gpu:4 #number of gpus needed, default is 1
#SBATCH -c 128  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_3task_case5_from_Chkpt_full_trace_baseline_mraza/all_logs/%x-%j-slurm.out
#SBATCH --error=/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_3task_case5_from_Chkpt_full_trace_baseline_mraza/all_logs/%x-%j-slurm.err
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

nvidia-smi
source activate Reason360

export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES

export TRANSFORMERS_CACHE="/export/alt-insight/HF_cache"
export HF_HOME="/export/alt-insight/HF_cache"
export HF_DATASETS_CACHE="/export/alt-insight/HF_cache"

#export PYTHONPATH="/export/alt-insight/NON_FA2_models/Reasoning360_sys_B_v29_3task_case5_from_Chkpt_full_trace_baseline_mraza:${PYTHONPATH:-}"

export ACC_W=0.90
export Z3_W=0.10



python ./test_verl.py