#!/bin/bash
cd ~/work/j5; source ~/venv/bin/activate; export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f codes_gptq_w8_rot.pt ] || python w8eval.py codes
python w8eval.py eval h2
python w8eval.py eval j5
echo W8CHAIN DONE
