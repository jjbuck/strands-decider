#!/bin/bash
# evals of run e (arm qatq: plain layout, LoRA only in low-bit state rows, KL + residual MSE to the dense teacher, lr 5e-5)
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"; SUB="JB-all,REAL-agree,CF,CF-probe"
ev() { out=$1; shift; if ! grep -q "^done" preds/$out.log 2>/dev/null; then rm -f preds/$out.jsonl; python h6eval.py preds/$out.jsonl "$@" >> preds/$out.log 2>&1; echo "$(date +%T) $out $(tail -1 preds/$out.log)" >> queue.log; fi; }
for s in 600 1200 2000; do
  while [ ! -f lora_e_qatq_s$s.pt ]; do sleep 30; if [ -f STOPQ ]; then exit; fi; done
  sleep 5
  while pgrep -f "h6eval.py" > /dev/null; do sleep 10; done
  if [ $s = 2000 ]; then S=all; else S=$SUB; fi
  ev qatq_e_s$s --prec $K48 --codes $C --layout plainqb --lrot lora_e_qatq_s$s.pt --suites $S
done
echo CHAIN4DONE >> queue.log
