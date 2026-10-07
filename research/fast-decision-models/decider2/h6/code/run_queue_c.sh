#!/bin/bash
# eval queue (GPU-time budgeted): untrained schema baseline, then ckpts 600..3000 of run b (both arms); LONG only at 1200/2400/3000
source ~/venv/bin/activate; cd ~/work/h6; mkdir -p preds
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "h6eval.py" > /dev/null; do sleep 10; done
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"; SUB="JB-all,REAL-agree,CF,CF-probe"
ev() { out=$1; shift; if ! grep -q "^done" preds/$out.log 2>/dev/null; then python h6eval.py preds/$out.jsonl "$@" >> preds/$out.log 2>&1; echo "$(date +%T) $out $(tail -1 preds/$out.log)" >> queue.log; fi; }
ev schema_bf16_s0 --prec bf16 --layout schema --suites $SUB
for s in 600 1200 1800 2400 3000; do
  while [ ! -f lora_b_qat_s$s.pt ] || [ ! -f lora_b_ctl_s$s.pt ]; do sleep 30; if [ -f STOPQ ]; then break 2; fi; done
  sleep 5
  if [ $s = 600 ] || [ $s = 1800 ]; then S=$SUB; else S=all; fi
  ev qat_b_s$s --prec $K48 --codes $C --layout schemamix --lrot lora_b_qat_s$s.pt --suites $S
  ev ctl_b_s$s --prec bf16 --layout schema --lrot lora_b_ctl_s$s.pt --force_rot --suites $S
done
echo QDONE >> queue.log
