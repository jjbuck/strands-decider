#!/bin/bash
# priority evals (coordinator 02:25): H1's best 4-bit point in the deployed kernels, all suites, hobson layout; then run-d s300 checkpoints (subset)
source ~/venv/bin/activate; cd ~/work/h6; mkdir -p preds
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"; SUB="JB-all,REAL-agree,CF,CF-probe"
ev() { out=$1; shift; if ! grep -q "^done" preds/$out.log 2>/dev/null; then python h6eval.py preds/$out.jsonl "$@" >> preds/$out.log 2>&1; echo "$(date +%T) $out $(tail -1 preds/$out.log)" >> queue.log; fi; }
BA16=1 ev plainqb_k48_ba16 --prec $K48 --codes $C --layout plainqb
BA16=1 AC4=0.85 ev plainqb_k48_ba16_c85 --prec $K48 --codes $C --layout plainqb
ev qatp_d_s300 --prec $K48 --codes $C --layout plainqb --lrot lora_d_qatp_s300.pt --suites $SUB
ev ctl_d_s300 --prec bf16 --layout schema --lrot lora_d_ctl_s300.pt --force_rot --suites $SUB
echo PRIODONE >> queue.log
