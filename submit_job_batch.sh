#!/bin/bash
set -euo pipefail

BASE_DIR="$HOME/NON_FA2_models"


if false; then

    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job.sh


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job.sh


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job.sh


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job.sh

    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job.sh


    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job.sh




fi



#if false; then


echo "Submitting Reasoning360_sys_B_v1..."
cd "$BASE_DIR/Reasoning360_sys_B_v1"
bash submit_job.sh


echo "Submitting Reasoning360_sys_B_v5..."
cd "$BASE_DIR/Reasoning360_sys_B_v5"
bash submit_job.sh

echo "Submitting Reasoning360_sys_B_v7..."
cd "$BASE_DIR/Reasoning360_sys_B_v7"
bash submit_job.sh


echo "Submitting Reasoning360_sys_B_v8..."
cd "$BASE_DIR/Reasoning360_sys_B_v8"
bash submit_job.sh


echo "Submitting Reasoning360_sys_B_v9..."
cd "$BASE_DIR/Reasoning360_sys_B_v9"
bash submit_job.sh





echo "Submitting Reasoning360_sys_B_v2..."
cd "$BASE_DIR/Reasoning360_sys_B_v2"
bash submit_job.sh


echo "Submitting Reasoning360_sys_B_v3..."
cd "$BASE_DIR/Reasoning360_sys_B_v3"
bash submit_job.sh


echo "Submitting Reasoning360_sys_B_v4..."
cd "$BASE_DIR/Reasoning360_sys_B_v4"
bash submit_job.sh




#fi

echo "All submission commands executed."