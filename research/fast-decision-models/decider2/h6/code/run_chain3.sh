#!/bin/bash
# after chain2: run-d s300 checkpoint evals (head-load fix)
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q CHAIN2DONE queue.log 2>/dev/null; do sleep 15; done
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"; SUB="JB-all,REAL-agree,CF,CF-probe"
ev() { out=$1; shift; rm -f preds/$out.jsonl preds/$out.log; python h6eval.py preds/$out.jsonl "$@" >> preds/$out.log 2>&1; echo "$(date +%T) $out $(tail -1 preds/$out.log)" >> queue.log; }
ev qatp_d_s300 --prec $K48 --codes $C --layout plainqb --lrot lora_d_qatp_s300.pt --suites $SUB
ev ctl_d_s300 --prec bf16 --layout schema --lrot lora_d_ctl_s300.pt --force_rot --suites $SUB
echo CHAIN3DONE >> queue.log
