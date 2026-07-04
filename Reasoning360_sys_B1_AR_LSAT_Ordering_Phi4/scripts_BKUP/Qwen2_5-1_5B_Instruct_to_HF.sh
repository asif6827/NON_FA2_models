#!/bin/bash -l

#SBATCH -J merge_to_HF #job name
#SBATCH -p gpu-all # queue used
#SBATCH --gres gpu:4 #number of gpus needed, default is 1
#SBATCH -c 30  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%x-%j-slurm.out
#SBATCH --error=./all_logs/%x-%j-slurm.err
#SBATCH --mail-user=asif6827@gmail.com

set -euo pipefail
module load cuda12.4/toolkit
nvidia-smi
source activate Reason360_v1
PY=python
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES
export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"

# ========== Environment ==========
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/export/home/asifali/Reasoning360:${PYTHONPATH:-}"
# export DISABLE_RAY=1  # Remove/comment out this line to enable Ray
export PYTHONWARNINGS="ignore::FutureWarning:transformers.utils.hub"

# Explicitly disable FA2 (if switching to FA2 later, change to 0/1 and adjust model.attn_implementation)
export FLASH_ATTENTION_FORCE_DISABLED=1
export HF_USE_FLASH_ATTENTION_2=0

# =================== Devices and paths (modify as needed) ===================
NUM_GPUS=4
n_nodes=${n_nodes:-1}
n_gpus_per_node=${n_gpus_per_node:-4}   # GPUs per node (set to 1 for single GPU)
gpu_ids=${gpu_ids:-"0,1,2,3"}               # Comma-separated GPU IDs; use "0" for single GPU
export CUDA_VISIBLE_DEVICES=${gpu_ids}
nvidia-smi


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /export/home/asifali/Reasoning360/checkpoints/Reasoning360_6K_logic_math_mix_Qwen2_5_1_5B/single-node-20251125-130536-Qwen2.5-1.5B-Instruct/global_step_70/actor \
    --target_dir /export/home/asifali/Reasoning360/checkpoints/HF_Merged_6k_maths_logic_mix_70