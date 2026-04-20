#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG (edit these) ----
REMOTE_USER="asifali"
REMOTE_HOST="panlogin"
SSH_PORT="22"
# -----------------------------



####################################################
###  NON-FA2 - HP
###

MODELS=(
  "Reasoning360_sys_A"
  "Reasoning360_sys_B1"
  "Reasoning360_sys_B_v1"
  "Reasoning360_sys_B_v2"
  "Reasoning360_sys_B_v3"
  "Reasoning360_sys_B_v4"
  "Reasoning360_sys_B_v5"
  "Reasoning360_sys_B_v6"
  "Reasoning360_sys_B_v7"
  "Reasoning360_sys_B_v8"
  "Reasoning360_sys_B_v9"
  "Reasoning360_sys_B_v10"
  "Reasoning360_sys_B_v11"
  "Reasoning360_sys_B_v12"
  "Reasoning360_sys_B_v13"
  "Reasoning360_sys_B_v14"
  "Reasoning360_sys_B_v15"
  "Reasoning360_sys_B_v16"
  "Reasoning360_sys_B_v17"
  "Reasoning360_sys_B_v18"
  "Reasoning360_sys_B_v19"
  "Reasoning360_sys_B_v20"
  "Reasoning360_sys_B_v21"
  "Reasoning360_sys_B_v22"
  "Reasoning360_sys_B_v23"
  "Reasoning360_sys_B_v24"
  "Reasoning360_sys_B_v25"
  "Reasoning360_sys_B_v26"
  "Reasoning360_sys_B_v27"
  "Reasoning360_sys_B_v28"
  "Reasoning360_sys_B_v29"
  "Reasoning360_sys_B_v29_qwen3_17"
  "Reasoning360_sys_B_v29_llama3_3B"
  "Reasoning360_sys_B_v29_a1"
  "Reasoning360_sys_B_v29_a1_004"
  "Reasoning360_sys_B_v29_a2"
  "Reasoning360_sys_B_v29_a3"
  "Reasoning360_sys_B_v29_a4"
  "Reasoning360_sys_B_v29_a4_004"
  "Reasoning360_sys_B_v29_a5"
  "Reasoning360_sys_B_v29_a6"
  "Reasoning360_sys_B_v29_a6_004"
  "Reasoning360_sys_B_v29_a7"
  "Reasoning360_sys_B_v29_a7_004"
  "Reasoning360_sys_B_v29_a8"
  "Reasoning360_sys_B_v29_a9"
  "Reasoning360_sys_B_v29_a10"
  "Reasoning360_sys_B_v30"
  "Reasoning360_sys_B_v31"
  "Reasoning360_sys_B_v32"
  "Reasoning360_sys_B_v33"
  "Reasoning360_sys_tester"
  "Reasoning360_sys_C_v1" 
  "Reasoning360_sys_C_v2" 
  "Reasoning360_sys_C_v3" 
  "Reasoning360_sys_C_v4" 
  "Reasoning360_sys_C_v5" 
)

REMOTE_BASE="/export/home/asifali/NON_FA2_models"
LOCAL_BASE="/home/asif/data3/Codes_QCRI/NON_FA2_models"

for MODEL in "${MODELS[@]}"; do
  REMOTE_PATH="${REMOTE_BASE}/${MODEL}/all_logs"
  LOCAL_PATH="${LOCAL_BASE}/${MODEL}"

  if [[ ! -d "$LOCAL_PATH" ]]; then
    continue
  fi

  rsync -az --checksum \
    -e "ssh -p ${SSH_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
    "${LOCAL_PATH}"
done

####################################################













####################################################
###  NON-FA2 - Dropbox
###

MODELS=(
  "Reasoning360_sys_A"
  "Reasoning360_sys_B1"
  "Reasoning360_sys_B_v1"
  "Reasoning360_sys_B_v2"
  "Reasoning360_sys_B_v3"
  "Reasoning360_sys_B_v4"
  "Reasoning360_sys_B_v5"
  "Reasoning360_sys_B_v6"
  "Reasoning360_sys_B_v7"
  "Reasoning360_sys_B_v8"
  "Reasoning360_sys_B_v9"
  "Reasoning360_sys_B_v10"
  "Reasoning360_sys_B_v11"
  "Reasoning360_sys_B_v12"
  "Reasoning360_sys_B_v13"
  "Reasoning360_sys_B_v14"
  "Reasoning360_sys_B_v15"
  "Reasoning360_sys_B_v16"
  "Reasoning360_sys_B_v17"
  "Reasoning360_sys_B_v18"
  "Reasoning360_sys_B_v19"
  "Reasoning360_sys_B_v20"
  "Reasoning360_sys_B_v21"
  "Reasoning360_sys_B_v22"
  "Reasoning360_sys_B_v23"
  "Reasoning360_sys_B_v24"
  "Reasoning360_sys_B_v25"
  "Reasoning360_sys_B_v26"
  "Reasoning360_sys_B_v27"
  "Reasoning360_sys_B_v28"
  "Reasoning360_sys_B_v29"
  "Reasoning360_sys_B_v29_a1"
  "Reasoning360_sys_B_v29_a2"
  "Reasoning360_sys_B_v29_a3"
  "Reasoning360_sys_B_v29_a4"
  "Reasoning360_sys_B_v29_a5"
  "Reasoning360_sys_B_v30"
  "Reasoning360_sys_B_v31"
  "Reasoning360_sys_B_v32"
  "Reasoning360_sys_B_v33"
  "Reasoning360_sys_C_v1"
  "Reasoning360_sys_C_v2"
  "Reasoning360_sys_C_v3"
  "Reasoning360_sys_C_v4"
  "Reasoning360_sys_C_v5"
)

REMOTE_BASE="/export/home/asifali/NON_FA2_models"
LOCAL_BASE="/home/asif/data3/Dropbox/shared_MRaza/25th_Feb_2026_Qwen3_4B_MTMT"

for MODEL in "${MODELS[@]}"; do
  REMOTE_PATH="${REMOTE_BASE}/${MODEL}/all_logs"
  LOCAL_PATH="${LOCAL_BASE}/${MODEL}"

  if [[ ! -d "$LOCAL_PATH" ]]; then
    continue
  fi

  rsync -az --checksum \
    -e "ssh -p ${SSH_PORT}" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
    "${LOCAL_PATH}"
done
####################################################


REMOTE_BASE="/export/home/asifali/Noise_math_data"
LOCAL_BASE="/home/asif/data3/Dropbox/Shared_Wenqing/Noise_math_data"


REMOTE_PATH="${REMOTE_BASE}/all_logs"
LOCAL_PATH="${LOCAL_BASE}/"

if [[ ! -d "$LOCAL_PATH" ]]; then
continue
fi

rsync -az --checksum \
-e "ssh -p ${SSH_PORT}" \
"${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
"${LOCAL_PATH}"




####################################################
####################################################


REMOTE_BASE="/export/home/asifali/Noise_math_data"
LOCAL_BASE="/home/asif/data3/Codes_QCRI/Noise_math_data"


REMOTE_PATH="${REMOTE_BASE}/all_logs"
LOCAL_PATH="${LOCAL_BASE}/"

if [[ ! -d "$LOCAL_PATH" ]]; then
continue
fi

rsync -az --checksum \
-e "ssh -p ${SSH_PORT}" \
"${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
"${LOCAL_PATH}"




####################################################  
