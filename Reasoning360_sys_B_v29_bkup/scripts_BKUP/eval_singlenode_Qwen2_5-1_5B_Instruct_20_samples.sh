#!/bin/bash -l

#SBATCH -J eval-guru-math-20 #job name
#SBATCH -p gpu-A100 # queue used
#SBATCH --gres gpu:1 #number of gpus needed, default is 1
#SBATCH -c 64  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%x-%j-slurm.out
#SBATCH --error=./all_logs/%x-%j-slurm.err
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --mail-user=asif6827@gmail.com


#!/usr/bin/env bash
set -euo pipefail

########################################################


module load cuda12.4/toolkit

nvidia-smi

echo "=== lscpu ==="
lscpu

echo "=== Memory (/proc/meminfo) ==="
grep -E 'MemTotal|MemFree|MemAvailable|SwapTotal|SwapFree' /proc/meminfo

echo "=== free -h ==="
free -h

source activate Reason360

export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES

export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"


#######################################################
n_nodes=1
n_gpus_per_node=1
gpu_ids=0
export CUDA_VISIBLE_DEVICES=${gpu_ids}

SHARED_DATA_PATH=/export/home/asifali/HF_cache/guru_data_20
data_folder=${SHARED_DATA_PATH}/offline_eval
save_folder=../evaluation_results/test_offline_guru_20_Qwen2_5_1_5B/
#model_path=/export/home/asifali/HF_cache/Qwen3-1.7B
#model_path=Qwen/Qwen3-1.7B
model_path=/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct
model_name=$(basename "$model_path")

########################################################

# CONDA_BIN_PATH=/root/miniconda3/bin/
# PY=${CONDA_BIN_PATH:-}/python
PY=python

export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/export/home/asifali/Reasoning360_v2:${PYTHONPATH:-}"
export DISABLE_RAY=1  # Locally disable Ray to avoid unnecessary cluster initialization.
unset LD_LIBRARY_PATH

# =================== (Offline) Leaderboard Eval Config ===================
leaderboard_list=(
    "math"
    #"aime"
    #"math"
    # "mbpp"
    # "humaneval"
    # "livecodebench"
    # "arcagi1"
    # "zebra_puzzle_dataset"
    # "gpqa_diamond"
    # "supergpqa"
    # "finqa"
    # "hitab"
    # "multihier"
    # "codeio"
    # "cruxeval-i"
    # "cruxeval-o"
    # "livebench_reasoning"
    # "livebench_language"
    # "livebench_data_analysis"
    # "ifeval"
)

# 域映射
declare -A domain_mappings
domain_mappings["aime"]="math"
domain_mappings["math"]="math"
domain_mappings["humaneval"]="codegen"
domain_mappings["livecodebench"]="codegen"
domain_mappings["mbpp"]="codegen"
domain_mappings["arcagi1"]="logic"
domain_mappings["zebra_puzzle_dataset"]="logic"
domain_mappings["finqa"]="table"
domain_mappings["hitab"]="table"
domain_mappings["multihier"]="table"
domain_mappings["codeio"]="simulation"
domain_mappings["cruxeval-i"]="simulation"
domain_mappings["cruxeval-o"]="simulation"
domain_mappings["gpqa_diamond"]="stem"
domain_mappings["supergpqa"]="stem"
domain_mappings["livebench_reasoning"]="ood"
domain_mappings["livebench_language"]="ood"
domain_mappings["livebench_data_analysis"]="ood"
domain_mappings["ifeval"]="ood"

# =================== 设备与路径 ===================


# 目录准备
mkdir -p "$save_folder"
logs_dir="${save_folder}logs/"
mkdir -p "$logs_dir"

# =================== Generate Parameters (Conservative Configuration for Single Card) ===================
# If using vLLM, change top_k to -1
batch_size=16
temperature=1.0
top_p=0.7
top_k=0           # 0: HF rollout, -1: vLLM rollout
tensor_model_parallel_size=1
gpu_memory_utilization=0.9

for leaderboard in "${leaderboard_list[@]}"; do
    domain=${domain_mappings[$leaderboard]}

    # 采样倍数
    if [ "$leaderboard" == "aime" ] || [ "$leaderboard" == "aime2025" ]; then
        n_samples=4
    elif [ "$leaderboard" == "arcagi1" ] || [ "$leaderboard" == "livecodebench" ] || [ "$leaderboard" == "humaneval" ] || [ "$leaderboard" == "zebra_puzzle_dataset" ] || [ "$leaderboard" == "multihier" ] || [ "$leaderboard" == "codeio" ] || [ "$leaderboard" == "gpqa_diamond" ]; then
        n_samples=4
    else
        n_samples=1
    fi


    if [ "$leaderboard" == "arcagi1" ] || [ "$leaderboard" == "multihier" ]; then
        prompt_length=4096
        response_length=4096
    else
        prompt_length=2048
        response_length=4096
    fi

    gen_log_file="${logs_dir}${model_name}_${leaderboard}_gen.log"
    eval_log_file="${logs_dir}${model_name}_${leaderboard}_eval.log"


    if [ "$leaderboard" == "aime" ] || [ "$leaderboard" == "aime2025" ]; then
        file_pattern="${domain}__${leaderboard}_repeated_8x_[0-9a-zA-Z]*.parquet"
    else
        file_pattern="${domain}__${leaderboard}_[0-9a-zA-Z]*.parquet"
    fi

    data_file=$(find "$data_folder" -name "$file_pattern" -type f | head -n 1)
    echo "data_file: $data_file"
    if [ -z "$data_file" ]; then
        echo "No file found matching pattern: $file_pattern. Skipping." | tee -a "$gen_log_file"
        continue
    fi

    file_name=$(basename "$data_file")
    save_path="${save_folder}/${model_name}/${file_name}"
    mkdir -p "$(dirname "$save_path")"

    echo "Processing $leaderboard: $data_file -> $save_path" | tee -a "$gen_log_file"

    # ===== Generate =====
    echo "Starting generation for $leaderboard at $(date)" | tee -a "$gen_log_file"
    {
        model_name="$(basename "${model_path}")"
        output_dir="${save_folder}/${model_name}"
        mkdir -p "${output_dir}"
        output_path="${output_dir}/$(basename "${data_file}")"

        python -m verl.trainer.main_generation \
          trainer.nnodes=1 \
          trainer.n_gpus_per_node=1 \
          model.path="${model_path}" \
          +model.trust_remote_code=True \
          +model.attn_implementation=eager \
          data.path="${data_file}" \
          data.prompt_key=prompt \
          data.n_samples="${n_samples}" \
          data.batch_size="${batch_size}" \
          data.output_path="${output_path}" \
          rollout.do_sample=True \
          rollout.temperature="${temperature}" \
          rollout.top_k="${top_k}" \
          rollout.top_p="${top_p}" \
          rollout.prompt_length="${prompt_length}" \
          rollout.response_length="${response_length}"\
          rollout.gpu_memory_utilization=$gpu_memory_utilization
    } 2>&1 | tee -a "$gen_log_file"
    echo "Completed generation for $leaderboard at $(date)" | tee -a "$gen_log_file"

    # ===== Evaluate =====
    echo "Starting evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"
    unset LD_LIBRARY_PATH
    {
        ${PY} -m verl.trainer.main_eval \
            data.path="$save_path" \
            data.prompt_key=prompt \
            data.response_key=responses \
            data.data_source_key=data_source \
            data.reward_model_key=reward_model
    } 2>&1 | tee -a "$eval_log_file"
    echo "Completed evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"

    echo "Completed processing $leaderboard. Generation log: $gen_log_file, Evaluation log: $eval_log_file"
done