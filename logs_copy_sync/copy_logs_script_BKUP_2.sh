#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG (edit these) ----
REMOTE_USER="asifali"
REMOTE_HOST="panlogin"
SSH_PORT="22"
# -----------------------------



  # --- Reasoning360_sys_A-HP-----

 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_A/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Codes_QCRI/FA2_models/Reasoning360_sys_A"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Codes_QCRI/FA2_models/Reasoning360_sys_B1"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Codes_QCRI/FA2_models/Reasoning360_sys_B_v1"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Codes_QCRI/FA2_models/Reasoning360_sys_B_v2"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Codes_QCRI/FA2_models/Reasoning360_sys_B_v3"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Codes_QCRI/FA2_models/Reasoning360_sys_B_v4"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Codes_QCRI/FA2_models/Reasoning360_sys_B_v5"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/19thFeb_2026_Qwen3_4B_MTMT/Outcome_v1_NL_60p"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  





  
    # --- Reasoning360_sys_B1-Dropbox-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B1/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/19thFeb_2026_Qwen3_4B_MTMT/Outcome_v2_Interleaved_60p"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  






  
    # --- Reasoning360_sys_B_v1-Dropbox-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v1/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/19thFeb_2026_Qwen3_4B_MTMT/PRM_v1_60p_10n_5f_5c"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  





  
    # --- Reasoning360_sys_B_v2-Dropbox-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v2/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/19thFeb_2026_Qwen3_4B_MTMT/PRM_v2_60p_10n_5c"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/19thFeb_2026_Qwen3_4B_MTMT/PRM_v3_60p_10n_5f"        # trailing slash recommended


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
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/19thFeb_2026_Qwen3_4B_MTMT/PRM_v4_60p_5c"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  

  
  
  
  
  
  
  # --- Reasoning360_sys_B_v5-Dropbox-----
 
REMOTE_PATH="/export/home/asifali/Reasoning360_sys_B_v5/all_logs"      # trailing slash = copy contents
LOCAL_PATH="/home/asif/data3/Dropbox/shared_MRaza/19thFeb_2026_Qwen3_4B_MTMT/PRM_v5_60p_5n_5c"        # trailing slash recommended


if [[ ! -d "$LOCAL_PATH" ]]; then
  echo "ERROR: Local directory does not exist: $LOCAL_PATH" >&2
  exit 1
fi

rsync -az --checksum \
  -e "ssh -p ${SSH_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
  "${LOCAL_PATH}"
  
  
  
  
  

  
