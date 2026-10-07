#!/bin/bash
# frontier evals (serialized): k64+qb16+ba16; k48+ba16 with question rows at W8A8 (SLOT8) instead of bf16
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"
w() { sleep $((RANDOM % 20)); while pgrep -f "h6eval.py" > /dev/null; do sleep 10; done; }
ev() { out=$1; shift; w; rm -f preds/$out.jsonl preds/$out.log; python h6eval.py preds/$out.jsonl "$@" >> preds/$out.log 2>&1; echo "$(date +%T) $out $(tail -1 preds/$out.log)" >> queue.log; }
BA16=1 ev plainqb_k64_ba16 --prec map:~/work/h6/precmap_w4a4_k64.json --codes $C --layout plainqb
SLOT8=1 BA16=1 ev plainq8_k48_ba16 --prec map:~/work/h2/precmap_w4a4_k48.json --codes $C --layout plainqb
echo CHAIN5DONE >> queue.log
