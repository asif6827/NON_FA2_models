#!/bin/bash

#echo " I am only this..!"

# Shared reward mode: 1 = first segment, 2 = continuation, 3 = inference only.
export PUZZLE_TASK="${PUZZLE_TASK:-1}"
# Human-readable label; the child script appends job ID, time, and model name.
export PUZZLE_RUN_NAME="${PUZZLE_RUN_NAME:-task-${PUZZLE_TASK}_runA}"
if [[ "$PUZZLE_TASK" == "1" ]]; then
    unset PUZZLE_SOURCE_RUN PUZZLE_SOURCE_JOB_ID
fi
case "$PUZZLE_TASK" in
    1|2|3) ;;
    *) echo "PUZZLE_TASK must be 1, 2, or 3" >&2; exit 1 ;;
esac
# Keep these exported when submitting later stages:
# PUZZLE_SOURCE_RUN = source label (e.g. task-1_runA), or exact directory name.
# PUZZLE_SOURCE_JOB_ID = source Slurm job ID when selecting by label.
# PUZZLE_RUN_NAME   = destination label; job ID/time are appended automatically.
export PUZZLE_MODEL_ONLY="${PUZZLE_MODEL_ONLY:-1}"
#if false; then
mkdir -p ./all_logs

ACRONYM="MLXL"
echo "Submitting Training job H200 + GT"
TRAIN_TEMP_LIST=(0.8)
TEST_TEMP_LIST=(0.0)
SCORING_LIST=("gt")
EPOCH_LIST=(150)
TEST_LIST=(3)
ACC_W_LIST=(0.8)
Z3_W_LIST=(0.2)
SWITCH_EPOCH_LIST=(150)
SYSTEM_NAME_LIST=("Reasoning360_sys_B_v29_3task_case5_from_Chkpt_v1")
EVAL_PATH_LIST=("mlxl_train_mlxl_test_1_parsed_v3task_v1_${ACRONYM}")
DATA_PATH_LIST=("ZebraPuzzle_to_guru_parsed_v3task_v1_${ACRONYM}/mlxl_train_mlxl_test")



SLURM_SCRIPT_H200="./scripts_Qwen3_4B/train/Parsed_v2/${ACRONYM}_1_1shot_H200_zebra_rl_fsdp_parsed_v2.sh"
for i in "${!TRAIN_TEMP_LIST[@]}"; do
    TRAIN_TEMP=${TRAIN_TEMP_LIST[$i]}
    TEST_TEMP=${TEST_TEMP_LIST[$i]}
    SCORING_METHOD=${SCORING_LIST[$i]}
    EPOCH=${EPOCH_LIST[$i]}
    TEST_FREQUENCY=${TEST_LIST[$i]}
    ACC_W=${ACC_W_LIST[$i]}
    Z3_W=${Z3_W_LIST[$i]}
    SWITCH_EPOCH=${SWITCH_EPOCH_LIST[$i]}
    SYSTEM_NAME=${SYSTEM_NAME_LIST[$i]}
    EVAL_PATH=${EVAL_PATH_LIST[$i]}
    DATA_PATH=${DATA_PATH_LIST[$i]}

    echo "Submitting job: PUZZLE_TASK=$PUZZLE_TASK, TRAIN-TEMP=$TRAIN_TEMP, TEST-TEMP=$TEST_TEMP, SCORING-METHOD=$SCORING_METHOD, EPOCH=$EPOCH, TEST-FREQUENCY=$TEST_FREQUENCY,
    ACC_W=$ACC_W, Z3_W=$Z3_W, EPOCH_SWITCH=$SWITCH_EPOCH SYSTEM_NAME=$SYSTEM_NAME EVAL_PATH=$EVAL_PATH DATA_PATH=$DATA_PATH"
    sbatch --export=ALL,PUZZLE_TASK="$PUZZLE_TASK" $SLURM_SCRIPT_H200 $TRAIN_TEMP $TEST_TEMP $SCORING_METHOD $EPOCH $TEST_FREQUENCY $ACC_W $Z3_W $SWITCH_EPOCH $SYSTEM_NAME $EVAL_PATH $DATA_PATH
done
echo "All jobs submitted H200."
#fi


