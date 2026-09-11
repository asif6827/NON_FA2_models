#!/bin/bash -l

#SBATCH -J ray-gen-only-all-qwen0.6B-gpu_all_checkpoint_3K #job name
#SBATCH -p gpu-A100 # queue used
#SBATCH --gres gpu:4 #number of gpus needed, default is 1
#SBATCH -c 40  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%x-%j-slurm.out
#SBATCH --error=./all_logs/%x-%j-slurm.err
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --mail-user=asif6827@gmail.com

set -euo pipefail
module load cuda12.4/toolkit
nvidia-smi
source activate Reason360_v1
PY=python
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES
#export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"

#### MY Parameters ####
export USE_Thinking=0

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

#########################################################
# Add Data Paths...!
# "BaseHF model path" (If not using checkpoint merging, the model will be directly used from this path)
MODEL_PATH=/export/home/asifali/HF_cache/Qwen3-0.6B

# Checkpoint root directory + step (provide these two variables => automatically merge and prioritize the merged model)
CKPTS_DIR=/export/home/asifali/Reasoning360/checkpoints/Reasoning360_3K_0_6B/single-node-20251118-163721-Qwen3-0.6B
GLOBAL_STEP=${GLOBAL_STEP:-220}

# Data and output
SHARED_DATA_PATH=/export/home/asifali/HF_cache/guru_data
data_folder=${SHARED_DATA_PATH}/offline_eval
save_folder=/export/home/asifali/Reasoning360/evaluation_results/ray_eval_guru_all_Qwen3_0_6B_checkpoint_3K


# rollout：0=HF  1=vLLM
use_vllm=${use_vllm:-${USE_VLLM:-0}}

# Whether to merge checkpoints
prefer_checkpoint=${prefer_checkpoint:-${PREFER_CHECKPOINT:-1}}

# Generate parameters
batch_size=${batch_size:-32}
temperature=${temperature:-0.0}
top_p=${top_p:-1.0}
gpu_memory_utilization=${gpu_memory_utilization:-0.7}

# ========== Logging helpers ==========
info(){ printf "\033[1;32m[INFO]\033[0m %s\n" "$*" >&2; }
warn(){ printf "\033[1;33m[WARN]\033[0m %s\n" "$*" >&2; }
err(){  printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; }

# ========== Prepare dirs ==========
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

# ========== Domain & Leaderboards ==========
leaderboard_list=(
  "math"
  #"aime"
  #"humaneval" "mbpp" "livecodebench"
  # "arcagi1" "zebra_puzzle_dataset" "gpqa_diamond" "supergpqa"
  # "finqa" "hitab" "multihier" "codeio" "cruxeval-i" "cruxeval-o"
  # "livebench_reasoning" "livebench_language" "livebench_data_analysis" "ifeval"
)

declare -A DOMAIN
DOMAIN["math"]="math"
DOMAIN["aime"]="math"
DOMAIN["humaneval"]="codegen"
DOMAIN["livecodebench"]="codegen"
DOMAIN["mbpp"]="codegen"
DOMAIN["arcagi1"]="logic"
DOMAIN["zebra_puzzle_dataset"]="logic"
DOMAIN["finqa"]="table"
DOMAIN["hitab"]="table"
DOMAIN["multihier"]="table"
DOMAIN["codeio"]="simulation"
DOMAIN["cruxeval-i"]="simulation"
DOMAIN["cruxeval-o"]="simulation"
DOMAIN["gpqa_diamond"]="stem"
DOMAIN["supergpqa"]="stem"
DOMAIN["livebench_reasoning"]="ood"
DOMAIN["livebench_language"]="ood"
DOMAIN["livebench_data_analysis"]="ood"
DOMAIN["ifeval"]="ood"

# ========== Small helpers ==========
normalize_run_dir() {
  local base="$1"
  [[ -z "$base" ]] && { echo ""; return 0; }
  if [[ "$base" =~ /global_step_[0-9]+/actor/huggingface/?$ ]]; then
    base=$(dirname "$(dirname "$(dirname "$base")")")
  elif [[ "$base" =~ /global_step_[0-9]+/actor/?$ ]]; then
    base=$(dirname "$(dirname "$base")")
  elif [[ "$base" =~ /global_step_[0-9]+/?$ ]]; then
    base=$(dirname "$base")
  fi
  echo "$base"
}

ensure_merged_hf() {
  # 输入：global_step_dir  输出：actor_hf_dir
  local gs_dir="$1"
  local actor_hf_dir="${gs_dir}/actor/huggingface"
  mkdir -p "$actor_hf_dir"

  local has_weight=0
  [[ -f "${actor_hf_dir}/pytorch_model.bin" ]] && has_weight=1
  compgen -G "${actor_hf_dir}/*.safetensors" >/dev/null && has_weight=1

  if [[ "$has_weight" -eq 0 ]]; then
    info "No merged HF weights detected, starting automatic merge...."
    echo "[INFO] No merged HF weights detected, starting automatic merge...." | tee -a "${logs_dir}/model_merge.log"
    { ${PY} -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${gs_dir}/actor" \
        --target_dir "${actor_hf_dir}"
    } 2>&1 | tee -a "${logs_dir}/model_merge.log"

    has_weight=0
    [[ -f "${actor_hf_dir}/pytorch_model.bin" ]] && has_weight=1
    compgen -G "${actor_hf_dir}/*.safetensors" >/dev/null && has_weight=1
    [[ "$has_weight" -eq 0 ]] && { err "Merge failed: No weights generated. See details.${logs_dir}/model_merge.log"; exit 1; }
    info "Completion merged：${actor_hf_dir}"
    echo "[INFO] Completion merged：${actor_hf_dir}" | tee -a "${logs_dir}/model_merge.log"
  fi
  echo "$actor_hf_dir"
}

# ========== Select model path: Checkpoint Priority or Local Model ==========
MODEL_TO_USE=""
model_name=""

if [[ "${prefer_checkpoint}" -eq 1 ]]; then
  # Only attempt to merge when CKPTS_DIR and GLOBAL_STEP are provided
  if [[ -n "${CKPTS_DIR:-}" && -n "${GLOBAL_STEP:-}" ]]; then
    run_dir="$(normalize_run_dir "${CKPTS_DIR}")"
    [[ -z "$run_dir" || ! -d "$run_dir" ]] && { err "运行根目录无效：${CKPTS_DIR}（规范化后：${run_dir}）"; exit 1; }
    gs_dir="${run_dir}/global_step_${GLOBAL_STEP}"
    [[ ! -d "$gs_dir" ]] && { err "找不到指定步数目录：${gs_dir}"; exit 1; }
    actor_hf_dir="$(ensure_merged_hf "$gs_dir")"
    MODEL_TO_USE="$actor_hf_dir"
  fi
fi

if [[ -z "${MODEL_TO_USE}" ]]; then
  # 支持两种变量名：model_path（小写优先）或 MODEL_PATH
  if [[ -n "${model_path:-}" ]]; then
    [[ ! -d "${model_path}" ]] && { err "Local model directory does not exist:${model_path}"; exit 1; }
    MODEL_TO_USE="${model_path}"
  else
    [[ ! -d "${MODEL_PATH}" ]] && { err "Local model directory does not exist:${MODEL_PATH}"; exit 1; }
    MODEL_TO_USE="${MODEL_PATH}"
  fi
fi

model_name="$(basename "${MODEL_TO_USE}")"
info "Use model catalog:${MODEL_TO_USE}"
info "Evaluation output directory:${save_folder}/${model_name}"
echo "Use model catalog:${MODEL_TO_USE}"
echo "Evaluation output directory:${save_folder}/${model_name}"
mkdir -p "${save_folder}/${model_name}"

# ========== vLLM / HF Backend ==========
if [[ "${use_vllm}" -eq 1 ]]; then
  top_k=-1
  tensor_model_parallel_size=${tensor_model_parallel_size:-${n_gpus_per_node}}
  rollout_name="vllm"
  # Automatically pull up the local Ray (if available)
  if command -v ray >/dev/null 2>&1; then
    if ! ray status >/dev/null 2>&1; then
      warn "Ray is not running, trying to start local head node......"
      ray start --head --num-cpus "$(nproc)" || true
    fi
  else
    warn "Command not found: ray (pip install ray), the vLLM mode may not be able to accelerate in parallel."
  fi
else
  top_k=0
  tensor_model_parallel_size=${tensor_model_parallel_size:-1}
  rollout_name="hf"
fi

# ========== Generation + Evaluation ==========
for leaderboard in "${leaderboard_list[@]}"; do
  domain=${DOMAIN[$leaderboard]:-}
  [[ -z "$domain" ]] && { err "Unknown leaderboard：${leaderboard}"; exit 1; }

  if [[ "$leaderboard" == "aime" || "$leaderboard" == "aime2025" ]]; then
    file_pattern="${domain}__${leaderboard}_repeated_8x_[0-9A-Za-z]*.parquet"
  else
    file_pattern="${domain}__${leaderboard}_[0-9A-Za-z]*.parquet"
  fi

  gen_log_file="${logs_dir}/${model_name}_${leaderboard}_gen.log"
  eval_log_file="${logs_dir}/${model_name}_${leaderboard}_eval.log"

  mapfile -t matched_files < <(find "${data_folder}" -type f -name "${file_pattern}" | sort)
  if [[ ${#matched_files[@]} -eq 0 ]]; then
    warn "No data file found matching pattern: ${file_pattern}. Skipping ${leaderboard}." | tee -a "${gen_log_file}"
    continue
  fi

  n_samples=1

  # 长序列任务
  if [[ "$leaderboard" == "arcagi1" || "$leaderboard" == "multihier" ]]; then
    prompt_length=4096
    response_length=4096
  else
    prompt_length=2048
    response_length=4096
  fi

  for data_file in "${matched_files[@]}"; do
    file_name=$(basename "${data_file}")
    save_path="${save_folder}/${model_name}/${file_name}"
    mkdir -p "$(dirname "${save_path}")"

    info "Generate ${leaderboard}: ${data_file} -> ${save_path}" | tee -a "${gen_log_file}"
    echo "Model to use is : ${MODEL_TO_USE}"
    {
      ${PY} -m verl.trainer.main_generation \
        trainer.nnodes="${n_nodes}" \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        model.path="${MODEL_TO_USE}" \
        +model.trust_remote_code=True \
        +model.attn_implementation=eager \
        +model.use_flash_attention_2=false \
        data.path="${data_file}" \
        data.prompt_key=prompt \
        data.n_samples="${n_samples}" \
        data.batch_size="${batch_size}" \
        data.output_path="${save_path}" \
        rollout.name="${rollout_name}" \
        rollout.do_sample=False \
        rollout.temperature="${temperature}" \
        rollout.top_k="${top_k}" \
        rollout.top_p="${top_p}" \
        rollout.prompt_length="${prompt_length}" \
        rollout.response_length="${response_length}" \
        rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
        rollout.tensor_model_parallel_size="${tensor_model_parallel_size}"
    } 2>&1 | tee -a "${gen_log_file}"
    info "生成完成 ${leaderboard}" | tee -a "${gen_log_file}"

    #info "评测 ${leaderboard}: ${save_path}" | tee -a "${eval_log_file}"
    #unset LD_LIBRARY_PATH
    #{
    #  ${PY} -m verl.trainer.main_eval \
    #    data.path="${save_path}" \
    #    data.prompt_key=prompt \
    #    data.response_key=responses \
    #    data.data_source_key=data_source \
    #    data.reward_model_key=reward_model
    #} 2>&1 | tee -a "${eval_log_file}"
    #info "Evaluation completed ${leaderboard}" | tee -a "${eval_log_file}"
  done
done

info "All offline evaluations completed: see log ${logs_dir}"


