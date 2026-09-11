#!/usr/bin/env bash
set -euo pipefail

# =================== Environment Setup ===================
echo "Activating conda environment: puzzle-asif"
source activate puzzle-asif

if [[ "$CONDA_DEFAULT_ENV" != "puzzle-asif" ]]; then
    echo "ERROR: Failed to activate puzzle-asif conda environment"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES

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
export HF_HOME="/home/wwq416/snap/wwq/.cache/huggingface"
export HF_DATASETS_CACHE="/home/wwq416/snap/wwq/.cache/huggingface/datasets"

# =================== Directory Setup ===================
PROJECT_ROOT="/home/wwq416/snap/wwq/puzzle-asif/Reasoning360"
DATA_ROOT="/home/wwq416/snap/wwq/puzzle-asif/Output/data-process/small_train_small_test"
TRAIN_DATA_DIR="${DATA_ROOT}/train"
TEST_DATA_DIR="${DATA_ROOT}/test"
MODEL_PATH="/home/wwq416/snap/wwq/model/Qwen/Qwen3-0.6B"
MODEL_NAME=$(basename "$MODEL_PATH")
SAVE_DIR="${PROJECT_ROOT}/output/test_dapo_memory_flow"
mkdir -p "$SAVE_DIR"
LOGS_DIR="${SAVE_DIR}/logs"
mkdir -p "$LOGS_DIR"

# =================== Ray Configuration ===================
echo "Stopping any existing Ray instances..."
ray stop -f 2>/dev/null || true
sleep 2

echo "Starting local Ray cluster with 1 GPU..."
ray start --head --num-gpus 1 --num-cpus 8 --temp-dir "/tmp/ray-test" --include-dashboard false
sleep 5

# =================== Training Configuration ===================
NUM_NODES=1
NUM_GPUS_PER_NODE=1

# Change directory to project root so Hydra can find configs
cd "$PROJECT_ROOT"
echo "Changed directory to: $(pwd)"

# Find one training file
train_files=$(find "$TRAIN_DATA_DIR" -name "*.parquet" | sort | head -1 | python -c "import sys, json; print(json.dumps([line.rstrip() for line in sys.stdin]))")
test_files=$(find "$TEST_DATA_DIR" -name "*.parquet" | sort | head -1 | python -c "import sys, json; print(json.dumps([line.rstrip() for line in sys.stdin]))")

# Test Params
TRAIN_PROMPT_BSZ=1  # Evaluate/Train 1 data item per batch
N_RESP_PER_PROMPT=3 # Sample 3 outputs in Step 1
DATA_LIMIT_CONFIG="+data.limit_train=1 +data.limit_val=1" # Total 1 item for training

echo "Starting DAPO Memory Flow Test..."
python -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name=dapo_trainer \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=False \
    \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation=right \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    data.gen_batch_size=1 \
    data.val_batch_size=1 \
    data.dataloader_num_workers=0 \
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
    critic.ppo_mini_batch_size=1 \
    critic.forward_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.mode=sync \
    +actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=False \
    reward_model.micro_batch_size_per_gpu=1 \
    \
    trainer.logger=[console] \
    trainer.project_name="dapo-test" \
    trainer.experiment_name="memory-flow-test" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${NUM_GPUS_PER_NODE} \
    trainer.nnodes=${NUM_NODES} \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    ray_init.num_cpus=4 \
    \
    +trainer.enable_step2=True \
    +trainer.step2_iterations=1 \
    +trainer.write_step1_outputs=True \
    +meta.enable_step_feedback=True \
    2>&1 | tee "${LOGS_DIR}/test.log"

echo "Test completed."
ray stop -f
