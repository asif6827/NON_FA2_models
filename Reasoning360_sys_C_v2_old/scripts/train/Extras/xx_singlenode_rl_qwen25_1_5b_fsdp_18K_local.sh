#!/bin/bash -l

#SBATCH -J train-RL-qwen25-1.5B_18K #job name
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
source activate Reason360_v1

export CUDA_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES

export TRANSFORMERS_CACHE="/export/home/asifali/HF_cache"
export HF_HOME="/export/home/asifali/HF_cache"
export HF_DATASETS_CACHE="/export/home/asifali/HF_cache"

export PYTHONPATH="/export/home/asifali/Reasoning360:${PYTHONPATH:-}"

# =================== User-Configurable Settings ===================
# --- Execution Environment ---
NUM_GPUS=4  # Set the number of GPUs to use on this node

# --- Resuming & Logging ---
RESUME_CKPT_DIR_NAME=""  # Fill in the W&B experiment name to resume from, otherwise leave empty to start from scratch
WANDB_PROJECT="Reasoning360_18K_1_5B_local" # Your wandb project name

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


train_files="['${math_train_path}','${leetcode_train_path}','${livecodebench_train_path}','${primeintellect_train_path}','${taco_train_path}','${arcagi1_train_path}','${arcagi2_train_path}','${barc_train_path}','${graph_train_path}','${ordering_train_path}','${zebra_train_path}','${codeio_train_path}','${hitab_train_path}','${multihier_train_path}','${webinstruct_train_path}']"  # Use math as example, add to more tasks as needed
test_files="['${math_test_path}','${aime_test_path}']"  # Use math as example, add to more tasks as needed

# =================== Model ===================
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
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
temperature=1.0
top_p=1.0
top_k=0 # 0 for HF rollout, -1 for vLLM rollout

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
echo "Starting single-GPU training with HF local rollout..."
args=(
    --config-path=config # Hydra 配置目录（相对路径）
    --config-name="dapo_fsdp_config.yaml" # 选择具体的配置文件（FSDP 基线）
    algorithm.adv_estimator=${adv_estimator} # 优势估计器类型（如 GRPO）
    algorithm.use_kl_in_reward=${use_kl_in_reward} # 是否在奖励中加入 KL 正则项
    algorithm.kl_ctrl.kl_coef=${kl_coef} # KL 系数（奖励中的惩罚权重）
    algorithm.filter_groups.enable=${enable_filter_groups} # 是否启用基于指标的样本分组过滤
    # algorithm.filter_groups.metric=${filter_groups_metric} # 单卡且禁用过滤时不必要，使用默认
    # algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} # 单卡且禁用过滤时不必要
    data.train_files="$train_files" # 训练数据文件列表
    data.val_files="$test_files" # 验证/测试数据文件列表
    data.prompt_key=prompt # 数据集中提示字段的键名
    data.truncation='right' # 输入超长时的截断方向（右侧）
    data.max_prompt_length=${max_prompt_length} # 提示最大 token 数
    data.max_response_length=${max_response_length} # 生成最大 token 数
    data.train_batch_size=${train_prompt_bsz} # 训练阶段的 batch size（每步提示条数）
    data.gen_batch_size=${gen_prompt_bsz} # rollout 阶段的生成 batch size
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} # 是否在 actor 损失中显式加入 KL loss
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} # KL loss 系数
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} # PPO clip 下界
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} # PPO clip 上界
    actor_rollout_ref.actor.clip_ratio_c=10.0 # PPO clip 常数（额外约束幅度）
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} # 启用动态 batch size 以贴合显存
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} # 每 GPU 的 PPO 计算最大 token 长度
    actor_rollout_ref.actor.strategy="fsdp" # 训练策略（FSDP）
    actor_rollout_ref.actor.optim.lr=1e-6 # 学习率
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 # 学习率预热步数
    actor_rollout_ref.actor.optim.weight_decay=0.1 # 权重衰减
    actor_rollout_ref.actor.optim.warmup_style=constant # 预热策略（常数）
    actor_rollout_ref.actor.optim.min_lr_ratio=0. # 最小学习率比例（相对 base lr）
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} # PPO 的 mini-batch 提示条数
    actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} # 微批大小（null 表示由框架自适应）
    # actor_rollout_ref.actor.fsdp_config.param_offload=${offload} # 单卡通常不需要参数 offload
    # actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} # 单卡通常不需要优化器状态 offload
    actor_rollout_ref.actor.entropy_coeff=0 # 策略熵正则系数
    actor_rollout_ref.actor.grad_clip=1.0 # 梯度裁剪阈值（范数）
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} # 损失聚合方式（如 token-mean）
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} # Ulysses 序列并行规模
    # actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 # 单卡无需设置进程组大小，使用默认
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} # 计算 logprob 时是否使用动态 batch size
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} # 每 GPU 计算 logprob 的最大 token 长度
    # actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} # 单卡下可省略，使用自适应
    # actor_rollout_ref.ref.fsdp_config.param_offload=${offload} # 单卡通常不需要参考模型 offload
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} # 参考模型的序列并行规模
    actor_rollout_ref.rollout.name=hf # rollout 后端：HF（transformers）
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} # 每个提示生成的响应数量
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} # rollout 阶段计算 logprob 的动态 batch size
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} # rollout/logprob 阶段每 GPU 最大 token 长度
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 # 目标显存利用率（用于动态调度）
    # actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} # 单卡下可省略，使用自适应
    # actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} # 单卡不需要张量并行，默认 1
    # actor_rollout_ref.rollout.enable_chunked_prefill=True # 简化配置，单卡可按需打开，默认关闭
    # actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} # 单卡保守长度与 batch 已足够，可省略
    # actor_rollout_ref.rollout.max_num_seqs=${gen_max_num_seqs} # 单卡可使用默认并发上限
    actor_rollout_ref.rollout.temperature=${temperature} # 采样温度
    actor_rollout_ref.rollout.top_p=${top_p} # nucleus 采样阈值
    actor_rollout_ref.rollout.top_k=${top_k} # top-k 采样阈值（HF 用 0）
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} # 验证时的 top-k
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} # 验证时的 top-p
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} # 验证时的温度
    actor_rollout_ref.rollout.val_kwargs.n=1 # 验证每个提示生成 1 个响应
    actor_rollout_ref.rollout.val_kwargs.do_sample=True # 验证阶段启用采样
    actor_rollout_ref.model.path=$BASE_MODEL # 基础模型权重路径
    actor_rollout_ref.model.use_remove_padding=True # 移除填充以提升计算效率
    actor_rollout_ref.rollout.multi_turn.enable=False # 关闭多轮对话生成
    actor_rollout_ref.rollout.mode="sync" # 同步模式 rollout
    +actor_rollout_ref.model.override_config.attention_dropout=0. # 覆盖注意力 dropout 为 0
    +actor_rollout_ref.model.override_config.embd_pdrop=0. # 覆盖 embedding dropout 为 0
    +actor_rollout_ref.model.override_config.resid_pdrop=0. # 覆盖 residual dropout 为 0
    actor_rollout_ref.model.enable_gradient_checkpointing=True # 启用梯度检查点以省显存
    reward_model.reward_manager=async_multi_process # 奖励模型管理器：多进程异步
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} # 是否启用过长输出的惩罚缓冲机制
    reward_model.overlong_buffer.len=${overlong_buffer_len} # 过长缓冲的长度阈值
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} # 过长输出的惩罚系数
    trainer.logger=['console','wandb'] # 日志后端（控制台与 W&B）
    trainer.project_name=${WANDB_PROJECT} # W&B 项目名
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} # 实验名（用于区分 run）
    trainer.val_before_train=True # 训练前先跑一次验证
    trainer.n_gpus_per_node=${NUM_GPUS} # 每节点 GPU 数
    # trainer.nnodes=1 # 单卡默认 1，可省略
    trainer.save_freq=10 # 模型保存频率（epoch 间隔）
    trainer.test_freq=10 # 验证/测试频率（epoch 间隔）
    trainer.total_epochs=10 # 总训练轮数
    trainer.log_val_generations=50 # 验证阶段记录的生成条数
    trainer.resume_mode=auto # 断点恢复模式（自动）
)

python -m recipe.dapo.main_dapo "${args[@]}"
