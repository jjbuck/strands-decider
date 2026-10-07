#!/bin/bash
cd ~/work/j1
source ~/venv/bin/activate
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup python "$@" < /dev/null > "$(basename $1 .py).log" 2>&1 &
