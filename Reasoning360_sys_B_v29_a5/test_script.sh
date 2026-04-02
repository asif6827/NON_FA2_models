#!/bin/bash -l


# ==================== All INPUTS =================================
TRAIN_TEMP=$1
TEST_TEMP=$2
SCORING_METHOD="${3}"
EPOCH=$4
TEST_FREQUENCY=$5
ACC_W=$6
Z3_W=$7
SWITCH_EPOCH=$8
SYSTEM_NAME="${9}"

# ================================================================
#SYSTEM_NAME="Reasoning360_sys_B_v2"
export PYTHONPATH="/export/home/asifali/${SYSTEM_NAME}:${PYTHONPATH:-}"
echo "Python Path = ${PYTHONPATH}"
