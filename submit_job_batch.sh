#!/bin/bash
set -euo pipefail

BASE_DIR="$HOME/NON_FA2_models"


echo "Submitting Reasoning360_sys_B1..."
cd "$BASE_DIR/Reasoning360_sys_B1"
bash submit_job.sh


#echo "Submitting Reasoning360_sys_B_v9..."
#cd "$BASE_DIR/Reasoning360_sys_B_v9"
#bash submit_job.sh


echo "All submission commands executed."