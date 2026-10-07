#!/bin/bash
# eval queue for run 'b' (trainable head): baselines, then every checkpoint (qat: schemamix k48 + LoRA + head; ctl: schema bf16 + LoRA + head)
source ~/venv/bin/activate; cd ~/work/h6; mkdir -p preds
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "h6eval.py" > /dev/null; do sleep 10; done
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"
ev() { out=$1; shift; if ! grep -q "^done" preds/$out.log 2>/dev/null; then python h6eval.py preds/$out.jsonl "$@" > preds/$out.log 2>&1; echo "$(date +%T) $out $(tail -1 preds/$out.log)" >> queue.log; fi; }
ev plainqb_k48 --prec $K48 --codes $C --layout plainqb
ev schema_bf16_s0 --prec bf16 --layout schema
ev schemamix_k48_s0 --prec $K48 --codes $C --layout schemamix
for s in $(seq 300 300 3000); do
  while [ ! -f lora_b_qat_s$s.pt ] || [ ! -f lora_b_ctl_s$s.pt ]; do sleep 30; if [ -f STOPQ ]; then break 2; fi; done
  sleep 5
  ev qat_b_s$s --prec $K48 --codes $C --layout schemamix --lrot lora_b_qat_s$s.pt
  ev ctl_b_s$s --prec bf16 --layout schema --lrot lora_b_ctl_s$s.pt --force_rot
done
ev plainqb_w4a4 --prec w4a4 --codes $C --layout plainqb
echo QDONE >> queue.log
