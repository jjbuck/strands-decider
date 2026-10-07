#!/bin/bash
# usage: bg.sh LOG cmd...   (detached; env set up)
LOG=$1; shift
cd ~/work/h5
source ~/venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup setsid "$@" > "$LOG" 2>&1 < /dev/null &
echo "pid $!"
