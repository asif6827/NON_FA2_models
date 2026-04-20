#!/usr/bin/env bash
set -euo pipefail

chmod +x ./copy_logs_script.sh

nohup bash -c '
while true; do
  /bin/bash ./copy_logs_script.sh >> ./log_copy.log 2>&1
  sleep 2
done
' >/dev/null 2>&1 &



# ps aux | grep run_copy_loop.sh
