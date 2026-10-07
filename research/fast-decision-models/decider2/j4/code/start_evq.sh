#!/bin/bash
cd ~/work/j4; source ~/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python evq.py >> evq.log 2>&1 &
