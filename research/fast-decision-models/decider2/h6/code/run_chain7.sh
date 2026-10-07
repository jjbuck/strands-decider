#!/bin/bash
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"
while ! grep -q CHAIN6DONE queue.log; do sleep 15; done
sleep 25; while pgrep -f "h6eval.py" > /dev/null; do sleep 10; done
rm -f preds/plainq8_k48_ba16.*
SLOT8=1 BA16=1 python h6eval.py preds/plainq8_k48_ba16.jsonl --prec map:~/work/h2/precmap_w4a4_k48.json --codes $C --layout plainqb --suites JB-all,REAL-agree,CF,CF-probe > preds/plainq8_k48_ba16.log 2>&1
echo "$(date +%T) plainq8_k48_ba16 $(tail -1 preds/plainq8_k48_ba16.log)" >> queue.log
echo CHAIN7DONE >> queue.log
