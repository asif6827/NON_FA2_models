#!/bin/bash -l

#SBATCH -J Sys-A #job name
#SBATCH -p gpu-A100 # queue used
#SBATCH --gres gpu:4 #number of gpus needed, default is 1
#SBATCH -c 128  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%j-%x.out
#SBATCH --error=./all_logs/%j-%x.err
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --mail-user=asif6827@gmail.com


module load cuda12.4/toolkit

nvidia-smi
source activate Reasoning360

export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES

#### MY Parameters
export USE_Thinking=0
#export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"


# ==================== All INPUTS =================================
TRAIN_TEMP=$1
TEST_TEMP=$2
SCORING_METHOD=$3
EPOCH=$4
TEST_FREQUENCY=$5
ACC_W=$6
Z3_W=$7
SWITCH_EPOCH=$8
SYSTEM_NAME="${9}"
EVAL_PATH="${10}"
DATA_PATH="${11}"


echo "Submitted job: TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, SCORING-METHOD=$SCORING_METHOD, EPOCH=$EPOCH, TEST-FREQUENCY=$TEST_FREQUENCY,
ACC_W=$ACC_W, Z3_W=$Z3_W, EPOCH_SWITCH=$SWITCH_EPOCH SYSTEM_NAME=$SYSTEM_NAME EVAL_PATH=$EVAL_PATH DATA_PATH=$DATA_PATH"

# ===============================================================
#SYSTEM_NAME="Reasoning360_sys_B_v2"
export PYTHONPATH="/export/home/asifali/${SYSTEM_NAME}:${PYTHONPATH:-}"
echo "Python Path = ${PYTHONPATH}"


# =================== User-Configurable Settings ===================
# --- Execution Environment ---
NUM_GPUS=4 # Set the number of GPUs to use on this node
gpu_memory_utilization=0.7
# --- Resuming & Logging ---
RESUME_CKPT_DIR_NAME=""  # Fill in the W&B experiment name to resume from, otherwise leave empty to start from scratch
WANDB_PROJECT="Sys_B_Asif_Qwen2_5_1_5B_STST_1_1shot_A100_v1" # Your wandb project name

# --- External Services ---
export STEM_LLM_JUDGE_URL="<STEM_LLM_JUDGE_URL>"  # Optional: Fill in the llm-as-judge hosted URL for 'STEM' domain evaluation
export WANDB_API_KEY="64305b88cc27033d4132d6ce147ecce132e6955d"

# =================== Environment Setup ===================
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_TORCHVISION=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONUNBUFFERED=1
# export CUDA_LAUNCH_BLOCKING=1 # Uncomment for easier debugging of CUDA errors
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
unset LD_LIBRARY_PATH
export DEBUG_CODE=0
export USE_NL=0 # Using NL Prompts
export TEST_SCORE_METHOD='gt'
export TRAIN_SCORE_METHOD=${SCORING_METHOD}
export Z3_CLUE_GATE=0.7
export ACC_W=${ACC_W}
export Z3_W=${Z3_W}
export SWITCH_EPOCH=${SWITCH_EPOCH}




# =================== Model ===================
#BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
#BASE_MODEL=/export/home/asifali/HF_cache/Qwen2.5-7B-Instruct
BASE_MODEL=/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct
#BASE_MODEL=/export/home/asifali/HF_cache/Qwen3-1.7B
MODEL_NAME=$(basename "$BASE_MODEL" | tr -s ' ' '_' | tr -d -c '[:alnum:]_')
MODEL_NAME=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
#BASE_MODEL=/export/home/asifali/HF_cache/Qwen3-1.7B

export REWARD_LOG_PATH=/export/home/asifali/${SYSTEM_NAME}/evaluation_results/${EVAL_PATH}/${MODEL_NAME}
SHARED_DATA_PATH=/export/home/asifali/HF_cache/${DATA_PATH}
VALID_GENERATION_PATH=/export/home/asifali/${SYSTEM_NAME}/evaluation_results/${EVAL_PATH}/${MODEL_NAME}
export PUZZLE_FEEDBACK_PATH=/export/home/asifali/${SYSTEM_NAME}/evaluation_results/${EVAL_PATH}/${MODEL_NAME}


echo "Validation Folder NAME: $VALID_GENERATION_PATH"

#SHARED_DATA_PATH=./data
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train
TEST_DATA_DIR=${SHARED_DATA_PATH}/test

### Math (train)
#math_train_path=${TRAIN_DATA_DIR}/math__combined_54.4k_3.0k.parquet
### Math (test)
#math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
#aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
#amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet

### Code (train)
#leetcode_train_path=${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k_216.parquet
#livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__livecodebench_440_73.parquet
#primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__primeintellect_7.5k_1.2k.parquet
#taco_train_path=${TRAIN_DATA_DIR}/codegen__taco_8.8k_1.5k.parquet
### Code (test)
#humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
#mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_200.parquet
#livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet

### Logic (train)
#arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_111_52.parquet
#arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_190_90.parquet
#barc_train_path=${TRAIN_DATA_DIR}/logic__barc_1.6k_761.parquet
#graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_1.2k_571.parquet
#ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_1.9k_904.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic_our_zebra_puzzle_new_reward_140.parquet
### Logic (test)
#ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_100.parquet
zebralogic_test_path=${TEST_DATA_DIR}/logic_our_zebra_puzzle_new_reward_test_140.parquet
#arcagi_test_path=${TEST_DATA_DIR}/logic__arcagi1_200.parquet

### Simulation (train)
#codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_3.7k_3k.parquet
### Simulation (test)
#codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_200.parquet

### Table (train)
#hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_4.3k_2.2k.parquet
#multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_1.5k_775.parquet
### Table (test)
#multihier_test_path=${TEST_DATA_DIR}/table__multihier_200.parquet
#hitab_test_path=${TEST_DATA_DIR}/table__hitab_200.parquet


### Stem (train)
#webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k_3.0k.parquet
### Stem (test)
#supergpqa_test_path=${TEST_DATA_DIR}/stem__supergpqa_200.parquet


train_files="['${zebra_train_path}']"  # Use math as example, add to more tasks as needed
test_files="['${zebralogic_test_path}']"  # Use math as example, add to more tasks as needed


# =================== Logging ===================
# Generate a unique experiment name if not resuming
if [[ -n "$RESUME_CKPT_DIR_NAME" ]]; then
    WANDB_EXPERIMENT_NAME="$RESUME_CKPT_DIR_NAME"
else
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    WANDB_EXPERIMENT_NAME="single-node-${TIMESTAMP}-${BASE_MODEL##*/}"
fi

# =================== Ray Start (Single Node) ===================
# Stop any previous Ray instances
${CONDA_BIN_PATH}ray stop -f

# Start a new Ray cluster on the local machine
# The number of CPUs is often best left for Ray to determine automatically.
echo "Starting Ray on the local node with ${NUM_GPUS} GPUs..."
#${CONDA_BIN_PATH}ray start --head --num-gpus ${NUM_GPUS} --include-dashboard=True --dashboard-port 8265
${CONDA_BIN_PATH}ray start --head --num-gpus ${NUM_GPUS} --include-dashboard=True --dashboard-port 8265
sleep 5


# =================== RL Config ===================
# Note, we borrowed the config format from DAPO while here disabled all DAPO features to run the naive RL baseline.

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 4))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=128  # on-policy model update batchsize: train_prompt_bsz * rollout.n
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=8
train_prompt_mini_bsz=16  # model grad update batchsize

# Algorithm
top_p=0.9
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Training config
# NOTE: sp_size and gen_tp are parallelism settings.
# sp_size: Sequence Parallelism size.
# gen_tp: Tensor Parallelism size for vLLM generation.
# For a 32B model on 8 GPUs, TP=2 is a reasonable starting point. Adjust if you have memory issues.
sp_size=1
gen_tp=2
gen_max_num_seqs=1024
infer_micro_batch_size=null
train_micro_batch_size=null
use_dynamic_bsz=True
actor_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))  # increase this to speed up model forward & backward but note memory overflow
infer_ppo_max_token_len=$(( (max_prompt_length + max_response_length) * 2))  # increase this to speed up model forward, but note memory overflow
offload=True

# =================== Start RL training ===================
# Ensure your python environment (e.g., conda) is activated before running this script.
echo "Starting training..."
python -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name="dapo_fsdp_config.yaml" \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_num_seqs=${gen_max_num_seqs} \
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMP} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p}\
    actor_rollout_ref.rollout.val_kwargs.temperature=${TEST_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.mode="sync" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=async_multi_process \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=${TEST_FREQUENCY} \
    trainer.total_epochs=${EPOCH} \
    trainer.log_val_generations=127 \
    trainer.resume_mode=auto \
    trainer.validation_data_dir=${VALID_GENERATION_PATH}
