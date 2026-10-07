#!/bin/bash
# decision-level evals (all 3,227 questions), sequential on the GPU lock
cd ~/work/q4; source ~/venv/bin/activate; export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p preds
C=w8=~/work/h2/codes_gptq_w8.pt
flock gpu.lock python bench_pk.py 140,256,512 > pk.log 2>&1
flock gpu.lock python q4eval.py preds/b8gptq.jsonl --codes $C > ev_b8gptq.log 2>&1
flock gpu.lock python q4eval.py preds/b8gptq_b13.jsonl --codes $C --b13 l0+read > ev_b8gptq_b13.log 2>&1
flock gpu.lock python q4eval.py preds/b8rtn8.jsonl > ev_b8rtn8.log 2>&1
flock gpu.lock python q4eval.py preds/b8rtn7.jsonl --qmax8 63 > ev_b8rtn7.log 2>&1
flock gpu.lock python q4eval.py preds/w4rtn4.jsonl --prec w4a4 > ev_w4rtn4.log 2>&1
flock gpu.lock python q4eval.py preds/w4rtn3.jsonl --prec w4a4 --qmax4 3 > ev_w4rtn3.log 2>&1
echo EVALS_DONE > evals.done
