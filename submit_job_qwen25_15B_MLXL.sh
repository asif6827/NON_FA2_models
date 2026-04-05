#!/bin/bash
set -euo pipefail


BASE_DIR="$HOME/NON_FA2_models"

echo "Submitting Reasoning360_sys_A..."
cd "$BASE_DIR/Reasoning360_sys_A"
bash submit_job_qwen25_15B_MLXL.sh



echo "Submitting Reasoning360_sys_B1..."
cd "$BASE_DIR/Reasoning360_sys_B1"
bash submit_job_qwen25_15B_MLXL.sh




echo "Submitting Reasoning360_sys_B_v29..."
cd "$BASE_DIR/Reasoning360_sys_B_v29"
bash submit_job_qwen25_15B_MLXL.sh




if false; then


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job_qwen25_15B_MLXL.sh


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job_qwen25_15B_MLXL.sh


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job_qwen25_15B_MLXL.sh




    echo "Submitting Reasoning360_sys_B_v29..."
    cd "$BASE_DIR/Reasoning360_sys_B_v29"
    bash submit_job_qwen25_15B_MLXL.sh
fi


echo "All submission commands executed."