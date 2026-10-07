#!/bin/bash
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"
while ! grep -q CHAIN5DONE queue.log; do sleep 15; done
sleep 25; while pgrep -f "h6eval.py" > /dev/null; do sleep 10; done
BA16=1 python h6eval.py preds/plainqbr_k64_ba16.jsonl --prec map:~/work/h6/precmap_w4a4_k64.json --codes $C --layout plainqbr > preds/plainqbr_k64_ba16.log 2>&1
echo "$(date +%T) plainqbr_k64_ba16 $(tail -1 preds/plainqbr_k64_ba16.log)" >> queue.log
echo CHAIN6DONE >> queue.log
