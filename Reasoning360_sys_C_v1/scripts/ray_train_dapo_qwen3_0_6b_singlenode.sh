#!/usr/bin/env bash
set -euo pipefail

# =================== Environment Setup ===================
# 激活conda环境
echo "Activating conda environment: puzzle-asif"
source activate puzzle-asif

# 检查conda环境是否激活
if [[ "$CONDA_DEFAULT_ENV" != "puzzle-asif" ]]; then
    echo "ERROR: Failed to activate puzzle-asif conda environment"
    exit 1
fi

# 设置CUDA可见设备（单GPU 5090）
export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES

echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# =================== Environment Variables ===================
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/wwq416/snap/wwq/puzzle-asif/Reasoning360:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

export FLASH_ATTENTION_FORCE_DISABLED=1
export HF_USE_FLASH_ATTENTION_2=0

# 设置HF缓存路径
export HF_HOME="/home/wwq416/snap/wwq/.cache/huggingface"
export HF_DATASETS_CACHE="/home/wwq416/snap/wwq/.cache/huggingface/datasets"

# =================== Directory Setup ===================
# 项目根目录
PROJECT_ROOT="/home/wwq416/snap/wwq/puzzle-asif/Reasoning360"

# 数据路径
DATA_ROOT="/home/wwq416/snap/wwq/puzzle-asif/Output/data-process/small_train_small_test"
TRAIN_DATA_DIR="${DATA_ROOT}/train"
TEST_DATA_DIR="${DATA_ROOT}/test"

# 模型路径
MODEL_PATH="/home/wwq416/snap/wwq/model/Qwen/Qwen3-0.6B"
MODEL_NAME=$(basename "$MODEL_PATH")

# 保存路径
SAVE_DIR="${PROJECT_ROOT}/output/dapo_train_qwen3_0_6b"
mkdir -p "$SAVE_DIR"

# 日志路径
LOGS_DIR="${SAVE_DIR}/logs"
mkdir -p "$LOGS_DIR"

# =================== Ray Configuration ===================
# 停止之前的Ray实例
echo "Stopping any existing Ray instances..."
ray stop -f 2>/dev/null || true
sleep 2

# 启动本地Ray集群（单节点，1 GPU）
echo "Starting local Ray cluster with 1 GPU..."
ray start --head --num-gpus 1 --num-cpus 8 --temp-dir "/tmp/ray-dapo" --include-dashboard false
sleep 5

# =================== Training Configuration ===================
# 设备配置
NUM_NODES=1
NUM_GPUS_PER_NODE=1
GPU_IDS="0"

# 数据文件配置
# 查找训练和测试文件
train_files=$(find "$TRAIN_DATA_DIR" -name "*.parquet" | sort | head -1 | python -c "import sys, json; print(json.dumps([line.rstrip() for line in sys.stdin]))")
test_files=$(find "$TEST_DATA_DIR" -name "*.parquet" | sort | head -1 | python -c "import sys, json; print(json.dumps([line.rstrip() for line in sys.stdin]))")

if [[ -z "$train_files" || "$train_files" == "[]" ]]; then
    echo "ERROR: No training files found in $TRAIN_DATA_DIR"
    exit 1
fi

if [[ -z "$test_files" || "$test_files" == "[]" ]]; then
    echo "ERROR: No test files found in $TEST_DATA_DIR"
    exit 1
fi

echo "Training files: $train_files"
echo "Test files: $test_files"

# 只使用少量数据进行训练和验证
echo "Limiting to 2 training steps and 1 validation example"
DATA_LIMIT_CONFIG="+data.limit_train=2 +data.limit_val=1"

# 训练参数
WANDB_PROJECT="dapo-training-qwen3-0.6b"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
WANDB_EXPERIMENT_NAME="single-node-${TIMESTAMP}-${MODEL_NAME}"

# 模型和算法配置
ADV_ESTIMATOR="grpo"
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=-1  # -1 for vLLM rollout

# 序列长度配置 - 调整为更大值，确保完整包含提示词
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=2048

# 批处理大小配置 - 调整为1以解决CUDA内存不足问题
TRAIN_PROMPT_BSZ=1
gen_prompt_bsz=1
N_RESP_PER_PROMPT=1
train_prompt_mini_bsz=1

# =================== DAPO Training Command ===================
echo "Starting DAPO training for ${MODEL_NAME} at $(date)"
echo "Log files will be saved to: $LOGS_DIR"

# 运行DAPO训练
python -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name=dapo_trainer \
    \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=False \
    \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation=right \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.val_batch_size=1 \
    data.dataloader_num_workers=2 \
    ${DATA_LIMIT_CONFIG} \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    +actor_rollout_ref.model.attn_implementation=eager \
    +actor_rollout_ref.model.use_flash_attention_2=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    \
    critic.model.path="${MODEL_PATH}" \
    critic.model.tokenizer_path="${MODEL_PATH}" \
    critic.model.trust_remote_code=True \
    +critic.model.attn_implementation=eager \
    +critic.model.use_flash_attention_2=false \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.use_remove_padding=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.forward_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.top_k=${TOP_K} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.mode=sync \
    +actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    reward_model.micro_batch_size_per_gpu=1 \
    \
    trainer.logger=[console] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=${NUM_NODES} \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_only=False \
    trainer.log_val_generations=1 \
    ray_init.num_cpus=4 \
    trainer.total_training_steps=2 \
    +meta.enable_step_feedback=True \
    +meta.feedback_path="${LOGS_DIR}/reasoning_feedback_${TIMESTAMP}.jsonl" \
    2>&1 | tee "${LOGS_DIR}/train_${TIMESTAMP}.log"

# 停止Ray集群
echo "Stopping Ray cluster..."
ray stop -f

echo "DAPO training completed at $(date)"
echo "All logs saved to: ${LOGS_DIR}"
echo "Training results saved to: ${SAVE_DIR}"