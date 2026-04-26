#!/bin/bash -l

#SBATCH -J test-job-RAY-Dashboard-test
#SBATCH -p gpu-H200 # queue used
#SBATCH --gres gpu:4 #number of gpus needed, default is 1
#SBATCH -c 128  #number of CPUs needed, default is 1
#SBATCH --mem 256GB #amount of memory needed, default
#SBATCH --output=./all_logs/%x-%j-slurm.out
#SBATCH --error=./all_logs/%x-%j-slurm.err
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --mail-user=asif6827@gmail.com

module load cuda12.4/toolkit
nvidia-smi
NUM_GPUS=4  # Set the number of GPUs to use on this node
source activate Reasoning360
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES


#### MY Parameters
export USE_Thinking=0
#export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"
export PYTHONPATH="/export/home/asifali/Reasoning360:${PYTHONPATH:-}"

nvidia-smi
# =================== User-Configurable Settings ===================
# --- Execution Environment ---

# --- Resuming & Logging ---
RESUME_CKPT_DIR_NAME=""  # Fill in the W&B experiment name to resume from, otherwise leave empty to start from scratch
WANDB_PROJECT="Reasoning360_3K_0_6B" # Your wandb project name

# --- External Services ---
export STEM_LLM_JUDGE_URL="<STEM_LLM_JUDGE_URL>"  # Optional: Fill in the llm-as-judge hosted URL for 'STEM' domain evaluation
export WANDB_API_KEY="64305b88cc27033d4132d6ce147ecce132e6955d"

# =================== Environment Setup ===================
export NCCL_DEBUG=info
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_LAUNCH_BLOCKING=1 # Uncomment for easier debugging of CUDA errors

export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0

# =================== Data Mixture ===================
SHARED_DATA_PATH=/export/home/asifali/HF_cache/guru_data_3K
#SHARED_DATA_PATH=./data
TRAIN_DATA_DIR=${SHARED_DATA_PATH}/train/
TEST_DATA_DIR=${SHARED_DATA_PATH}/online_eval/

# Math (train)
math_train_path=${TRAIN_DATA_DIR}/math__combined_54.4k_3.0k.parquet
# Math (test)
math_test_path=${TEST_DATA_DIR}/math__math_500.parquet
aime_test_path=${TEST_DATA_DIR}/math__aime_repeated_8x_240.parquet
amc_test_path=${TEST_DATA_DIR}/math__amc_repeated_4x_332.parquet

## Code (train)
leetcode_train_path=${TRAIN_DATA_DIR}/codegen__leetcode2k_1.3k_216.parquet
livecodebench_train_path=${TRAIN_DATA_DIR}/codegen__livecodebench_440_73.parquet
primeintellect_train_path=${TRAIN_DATA_DIR}/codegen__primeintellect_7.5k_1.2k.parquet
taco_train_path=${TRAIN_DATA_DIR}/codegen__taco_8.8k_1.5k.parquet
## Code (test)
humaneval_test_path=${TEST_DATA_DIR}/codegen__humaneval_164.parquet
mbpp_test_path=${TEST_DATA_DIR}/codegen__mbpp_200.parquet
livecodebench_test_path=${TEST_DATA_DIR}/codegen__livecodebench_279.parquet

## Logic (train)
arcagi1_train_path=${TRAIN_DATA_DIR}/logic__arcagi1_111_52.parquet
arcagi2_train_path=${TRAIN_DATA_DIR}/logic__arcagi2_190_90.parquet
barc_train_path=${TRAIN_DATA_DIR}/logic__barc_1.6k_761.parquet
graph_train_path=${TRAIN_DATA_DIR}/logic__graph_logical_1.2k_571.parquet
ordering_train_path=${TRAIN_DATA_DIR}/logic__ordering_puzzle_1.9k_904.parquet
zebra_train_path=${TRAIN_DATA_DIR}/logic__zebra_puzzle_1.3k_618.parquet
## Logic (test)
ordering_puzzle_test_path=${TEST_DATA_DIR}/logic__ordering_puzzle_dataset_100.parquet
zebralogic_test_path=${TEST_DATA_DIR}/logic__zebra_puzzle_dataset_200.parquet
arcagi_test_path=${TEST_DATA_DIR}/logic__arcagi1_200.parquet

## Simulation (train)
codeio_train_path=${TRAIN_DATA_DIR}/simulation__codeio_3.7k_3k.parquet
## Simulation (test)
codeio_test_path=${TEST_DATA_DIR}/simulation__codeio_200.parquet

## Table (train)
hitab_train_path=${TRAIN_DATA_DIR}/table__hitab_4.3k_2.2k.parquet
multihier_train_path=${TRAIN_DATA_DIR}/table__multihier_1.5k_775.parquet
## Table (test)
multihier_test_path=${TEST_DATA_DIR}/table__multihier_200.parquet
hitab_test_path=${TEST_DATA_DIR}/table__hitab_200.parquet


## Stem (train)
webinstruct_train_path=${TRAIN_DATA_DIR}/stem__web_3.6k_3.0k.parquet
## Stem (test)
supergpqa_test_path=${TEST_DATA_DIR}/stem__supergpqa_200.parquet


train_files="['${math_train_path}']"  # Use math as example, add to more tasks as needed
test_files="['${math_test_path}','${aime_test_path}']"  # Use math as example, add to more tasks as needed

# =================== Model ===================
BASE_MODEL=Qwen/Qwen3-0.6B
#BASE_MODEL=/export/home/asifali/HF_cache/Qwen2.5-1.5B-Instruct
#BASE_MODEL=/export/home/asifali/HF_cache/Qwen3-1.7B

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
max_response_length=$((1024 * 8))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=512  # on-policy model update batchsize: train_prompt_bsz * rollout.n
gen_prompt_bsz=$((train_prompt_bsz * 1))
n_resp_per_prompt=16
train_prompt_mini_bsz=64  # model grad update batchsize

# Algorithm
temperature=0.7
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

python hello.py