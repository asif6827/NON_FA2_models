#!/bin/bash -l

#SBATCH -J ray-eval-only-all-qwen0_6B_0_1_1_pos_thinking #job name
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


#### MY Parameters ####
export USE_Thinking=0

#export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"

export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/export/home/asifali/Reasoning360:${PYTHONPATH:-}"
# export DISABLE_RAY=1  # Remove/comment out this line to enable Ray
export PYTHONUNBUFFERED=1  # Force real-time Python output; use tee for live logs
export HYDRA_FULL_ERROR=1  # Print full stack traces on errors for easier debugging

export FLASH_ATTENTION_FORCE_DISABLED=1
export HF_USE_FLASH_ATTENTION_2=0

# =================== Devices and paths (modify as needed) ===================
n_nodes=${n_nodes:-1}
n_gpus_per_node=${n_gpus_per_node:-2}   # GPUs per node (set to 1 for single GPU)
gpu_ids=${gpu_ids:-"0,1"}               # Comma-separated GPU IDs; use "0" for single GPU
export CUDA_VISIBLE_DEVICES=${gpu_ids}

nvidia-smi
#########################################################
# Add Data Paths...!

SHARED_DATA_PATH=/export/home/asifali/HF_cache/guru_data
data_folder=${SHARED_DATA_PATH}/offline_eval
save_folder=/export/home/asifali/Reasoning360/evaluation_results/ray_eval_guru_all_Qwen3-0_6B_pos_thinking
#model_path=Qwen/Qwen3-0.6B
model_path=/export/home/asifali/HF_cache/Qwen3-0.6B
#model_path=/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct
model_name=$(basename "$model_path")

# Prepare directories
mkdir -p "${save_folder}"
logs_dir="${save_folder%/}/logs"
mkdir -p "${logs_dir}"

# =================== Auto-start local Ray (optional) ===================
if command -v ray >/dev/null 2>&1; then
  if ! ray status >/dev/null 2>&1; then
    echo "Ray 未运行，尝试启动本地 head 节点..."
    # Start local Ray; ignore errors if it's already running
    ray start --head --num-cpus "$(nproc)" || true
  fi
else
  echo "未找到 ray 命令，请先安装 Ray（pip install ray）。" >&2
fi

# =================== (Offline) Leaderboard Eval Config ===================
leaderboard_list=(
  "math"
  #"aime"
  # 如需更多任务，按需取消注释：
  # "mbpp" "humaneval" "livecodebench" "arcagi1" "zebra_puzzle_dataset"
  # "gpqa_diamond" "supergpqa" "finqa" "hitab" "multihier" "codeio"
  # "cruxeval-i" "cruxeval-o" "livebench_reasoning" "livebench_language"
  # "livebench_data_analysis" "ifeval"
)

# Domain mappings
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

# =================== Generation parameters (Ray + vLLM friendly) ===================
batch_size=${batch_size:-16}
temperature=${temperature:-0.0}
top_p=${top_p:-1.0}
use_vllm=${use_vllm:-1}  # 1: use vLLM (Ray-friendly, parallel); 0: use HF (single-machine)

  if [[ "${use_vllm}" -eq 1 ]]; then
  top_k=-1                     # -1 means vLLM rollout
  tensor_model_parallel_size=${tensor_model_parallel_size:-${n_gpus_per_node}}
  gpu_memory_utilization=${gpu_memory_utilization:-0.9}
  rollout_name=vllm
else
  top_k=0                      # 0 means HF rollout
  tensor_model_parallel_size=${tensor_model_parallel_size:-1}
  gpu_memory_utilization=${gpu_memory_utilization:-0.9}
  rollout_name=hf
fi

for leaderboard in "${leaderboard_list[@]}"; do
  domain=${domain_mappings[$leaderboard]}

  # Sampling multiplier
  if [[ "${leaderboard}" == "aime" || "${leaderboard}" == "aime2025" ]]; then
    n_samples=4
  elif [[ "${leaderboard}" == "arcagi1" || "${leaderboard}" == "livecodebench" || "${leaderboard}" == "humaneval" || "${leaderboard}" == "zebra_puzzle_dataset" || "${leaderboard}" == "multihier" || "${leaderboard}" == "codeio" || "${leaderboard}" == "gpqa_diamond" ]]; then
    n_samples=4
  else
    n_samples=1
  fi

  # Sequence lengths
  if [[ "${leaderboard}" == "arcagi1" || "${leaderboard}" == "multihier" ]]; then
    prompt_length=4096
    response_length=4096
  else
    prompt_length=2048
    response_length=4096
  fi

gen_log_file="${logs_dir}/${model_name}_${leaderboard}_gen.log"
eval_log_file="${logs_dir}/${model_name}_${leaderboard}_eval.log"

  # Data file pattern
  if [[ "${leaderboard}" == "aime" || "${leaderboard}" == "aime2025" ]]; then
    file_pattern="${domain}__${leaderboard}_repeated_8x_[0-9a-zA-Z]*.parquet"
  else
    file_pattern="${domain}__${leaderboard}_[0-9a-zA-Z]*.parquet"
  fi

  # Iterate all shards for this task (not only the first)
  mapfile -t matched_files < <(find "${data_folder}" -type f -name "${file_pattern}" | sort)
  if [[ ${#matched_files[@]} -eq 0 ]]; then
    echo "No file found matching pattern: ${file_pattern}. Skipping." | tee -a "${gen_log_file}"
    continue
  fi

  for data_file in "${matched_files[@]}"; do
    file_name=$(basename "${data_file}")
    save_path="${save_folder}/${model_name}/${file_name}"
    mkdir -p "$(dirname "${save_path}")"

    echo "Processing ${leaderboard}: ${data_file} -> ${save_path}" | tee -a "${gen_log_file}"

    # ===== Generate =====
    #echo "Starting generation for ${leaderboard} at $(date)" | tee -a "${gen_log_file}"
    #{
    #  ${PY} -m verl.trainer.main_generation \
    #    trainer.nnodes="${n_nodes}" \
    #    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    #    model.path="${model_path}" \
    #    +model.trust_remote_code=True \
    #    +model.attn_implementation=eager \
    #    +model.use_flash_attention_2=false \
    #    data.path="${data_file}" \
    #    data.prompt_key=prompt \
    #    data.n_samples="${n_samples}" \
    #    data.batch_size="${batch_size}" \
    #    data.output_path="${save_path}" \
    #    rollout.name="${rollout_name}" \
    #    rollout.do_sample=True \
    #    rollout.temperature="${temperature}" \
    #    rollout.top_k="${top_k}" \
    #    rollout.top_p="${top_p}" \
    #    rollout.prompt_length="${prompt_length}" \
    #    rollout.response_length="${response_length}" \
    #    rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    #    rollout.tensor_model_parallel_size="${tensor_model_parallel_size}"
    #} 2>&1 | tee -a "${gen_log_file}"
    #echo "Completed generation for ${leaderboard} at $(date)" | tee -a "${gen_log_file}"

    # ===== Evaluate =====
    echo "Starting evaluation for ${leaderboard} at $(date)" | tee -a "${eval_log_file}"
    unset LD_LIBRARY_PATH
    {
      ${PY} -m verl.trainer.main_eval \
        data.path="${save_path}" \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model
    } 2>&1 | tee -a "${eval_log_file}"
    echo "Completed evaluation for ${leaderboard} at $(date)" | tee -a "${eval_log_file}"
  done

  echo "Completed processing ${leaderboard}. Generation log: ${gen_log_file}, Evaluation log: ${eval_log_file}"
done

echo "All tasks finished. Logs in: ${logs_dir}"
