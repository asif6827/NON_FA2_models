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



    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job.sh


    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job.sh


    echo "Submitting Reasoning360_sys_B1..."
    cd "$BASE_DIR/Reasoning360_sys_B1"
    bash submit_job.sh


    echo "Submitting Reasoning360_sys_A..."
    cd "$BASE_DIR/Reasoning360_sys_A"
    bash submit_job.sh


fi



#if false; then

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


#fi

echo "All submission commands executed."