#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG (edit these) ----
REMOTE_USER="asifali"
REMOTE_HOST="panlogin"
SSH_PORT="22"
# -----------------------------



# --- Reasoning360 Dropbox-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/Shared_Wenqing/Reasoning360"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"



# --- Reasoning360_NL Dropbox-----

 
REMOTE_PATH="/export/home/asifali/Reasoning360_NL/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/Shared_Wenqing/Reasoning360_NL"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"


# --- Reasoning360-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"




# --- Reasoning360_NL-HP-----

 
REMOTE_PATH="/export/home/asifali/Reasoning360_NL/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_NL"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"




# --- Reasoning360_parsed_Asif-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_parsed_Asif/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_parsed_Asif"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"



# --- Reasoning360_sys_A-HP-----

 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_A/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_A"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"

  
  
# --- Reasoning360_sys_B-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  
  
# --- Reasoning360_sys_B_Parsed_v3-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_Parsed_V3/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_Parsed_V3"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  
  
  # --- Reasoning360_sys_B_v1-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v1/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_v1"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  
  # --- Reasoning360_sys_B_v2-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v2/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_v2"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  
  # --- Reasoning360_sys_B_v3-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v3/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_v3"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  

  # --- Reasoning360_sys_B_v3_ranking-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v3_ranking/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_v3_ranking"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"


  # --- Reasoning360_sys_B_v3_r1-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v3_r1/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_v3_r1"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"


  # --- Reasoning360_sys_B_v3_r2-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v3_r2/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_v3_r2"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"

  
  
  # --- Reasoning360_sys_B_v4-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v4/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_v4"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"


  
  # --- Reasoning360_sys_B_v5-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v5/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B_v5"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  
  # --- Reasoning360_sys_B1-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B1/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B1"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  # --- Reasoning360_sys_B3-HP-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B3/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/Reasoning360_sys_B3"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  # --- Reasoning360_sys_B_v4-Dropbox-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v4/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/10thFeb_2026_Qwen3_4B_MTMT/Case_5_0_1_contradiction"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  
  
  
  
  
  
  
  
  # --- Reasoning360_sys_B_v3-Dropbox-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v3/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/11thFeb_2026_Qwen3_4B_MTMT/Case1_outcome_only_kl_reg"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  









  
  # --- Reasoning360_sys_A-Dropbox-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_A/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/11thFeb_2026_Qwen3_4B_MTMT/Case2_outcome_only_NL_kl_reg"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
 
  
