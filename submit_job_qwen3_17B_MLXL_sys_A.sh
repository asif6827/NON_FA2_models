#!/bin/bash
set -euo pipefail


echo "Submitting Reasoning360_sys_A_AR_LSAT_Assignment..."
cd "$BASE_DIR/Reasoning360_sys_A_AR_LSAT_Assignment"
bash submit_job_qwen3_17B_MLXL_A100.sh


echo "Submitting Reasoning360_sys_A_AR_LSAT_Grouping..."
cd "$BASE_DIR/Reasoning360_sys_A_AR_LSAT_Grouping"
bash submit_job_qwen3_17B_MLXL_A100.sh



echo "Submitting Reasoning360_sys_A_AR_LSAT_Ordering..."
cd "$BASE_DIR/Reasoning360_sys_A_AR_LSAT_Ordering"
bash submit_job_qwen3_17B_MLXL_A100.sh



if false; then


    echo "Submitting Reasoning360_sys_B_v29..."
    cd "$BASE_DIR/Reasoning360_sys_B_v29"
    bash submit_job_qwen3_17B_MLXL.sh

    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job_qwen3_17B_MLXL.sh


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job_qwen3_17B_MLXL.sh


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job_qwen3_17B_MLXL.sh


    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job_qwen3_17B_MLXL.sh

    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job_qwen3_17B_MLXL.sh

    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job_qwen3_17B_MLXL.sh

    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job_qwen3_17B_MLXL.sh




    echo "Submitting Reasoning360_sys_B_v29..."
    cd "$BASE_DIR/Reasoning360_sys_B_v29"
    bash submit_job_qwen3_17B_MLXL.sh

    echo "Submitting Reasoning360_sys_B_v29..."
    cd "$BASE_DIR/Reasoning360_sys_B_v29"
    bash submit_job_qwen3_17B_MLXL.sh

    echo "Submitting Reasoning360_sys_B_v29..."
    cd "$BASE_DIR/Reasoning360_sys_B_v29"
    bash submit_job_qwen3_17B_MLXL.sh

    echo "Submitting Reasoning360_sys_B_v29..."
    cd "$BASE_DIR/Reasoning360_sys_B_v29"
    bash submit_job_qwen3_17B_MLXL.sh



    echo "Submitting Reasoning360_sys_B_v29..."
    cd "$BASE_DIR/Reasoning360_sys_B_v29"
    bash submit_job_qwen3_17B_MLXL.sh
fi


echo "All submission commands executed."