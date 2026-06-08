#!/usr/bin/env bash
set -euo pipefail

cd /home/liujia/RF_Image
mkdir -p _logs

log_path="_logs/tiny_random64_full1500_resume.log"
pid_path="_logs/tiny_random64_full1500_resume.pid"

nohup /home/liujia/miniconda3/envs/augan_5090/bin/python -u resume_tiny_random64_full1500.py \
  > "${log_path}" 2>&1 &

echo "$!" > "${pid_path}"
echo "pid=$(cat "${pid_path}")"
echo "log=${log_path}"
