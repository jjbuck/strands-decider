#!/bin/bash
# eval queue for run c (3 arms): ckpts 600, 1200 on the cheap subset (no LONG), 1800 on all suites; untrained schema baseline first
source ~/venv/bin/activate; cd ~/work/h6; mkdir -p preds
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "h6eval.py" > /dev/null; do sleep 10; done
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"; SUB="JB-all,REAL-agree,CF,CF-probe"
ev() { out=$1; shift; if ! grep -q "^done" preds/$out.log 2>/dev/null; then python h6eval.py preds/$out.jsonl "$@" >> preds/$out.log 2>&1; echo "$(date +%T) $out $(tail -1 preds/$out.log)" >> queue.log; fi; }
ev schema_bf16_s0 --prec bf16 --layout schema --suites $SUB
for s in 600 1200 1800; do
  while [ ! -f lora_c_qatp_s$s.pt ]; do sleep 30; if [ -f STOPQ ]; then break 2; fi; done
  sleep 5
  if [ $s = 1800 ]; then S=all; else S=$SUB; fi
  ev qatp_c_s$s --prec $K48 --codes $C --layout plainqb --lrot lora_c_qatp_s$s.pt --suites $S
  ev qat_c_s$s --prec $K48 --codes $C --layout schemamix --lrot lora_c_qat_s$s.pt --suites $S
  ev ctl_c_s$s --prec bf16 --layout schema --lrot lora_c_ctl_s$s.pt --force_rot --suites $S
done
echo QDONE >> queue.log
