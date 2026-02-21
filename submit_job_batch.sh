#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$BASE_DIR/Reasoning360_sys_B_v1/submit_job.sh"
bash "$BASE_DIR/Reasoning360_sys_B_v2/submit_job.sh"
bash "$BASE_DIR/Reasoning360_sys_B_v3/submit_job.sh"
bash "$BASE_DIR/Reasoning360_sys_B_v5/submit_job.sh"
bash "$BASE_DIR/Reasoning360_sys_B_v4/submit_job.sh"