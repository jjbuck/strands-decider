#!/bin/bash
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"
while ! grep -q BENCH2DONE bench2.log 2>/dev/null; do sleep 10; done
SLOT8=1 BA16=1 python h6eval.py preds/plainq8_k64_ba16.jsonl --prec map:~/work/h6/precmap_w4a4_k64.json --codes $C --layout plainqb > preds/plainq8_k64_ba16.log 2>&1
echo "$(date +%T) plainq8_k64_ba16 $(tail -1 preds/plainq8_k64_ba16.log)" >> queue.log
echo CHAIN8DONE >> queue.log
