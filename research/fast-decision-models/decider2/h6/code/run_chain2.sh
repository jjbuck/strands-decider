#!/bin/bash
# after run_prio: latency matrix (exclusive GPU), then the H7 'sets' schema layout evals (untrained hobson, bf16 vs k48+ba16 mixed rows)
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q PRIODONE queue.log 2>/dev/null; do sleep 15; done
while pgrep -f "h6eval.py" > /dev/null; do sleep 10; done
./run_bench.sh
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"; SUB="JB-all,REAL-agree,CF,CF-probe"
ev() { out=$1; shift; if ! grep -q "^done" preds/$out.log 2>/dev/null; then python h6eval.py preds/$out.jsonl "$@" >> preds/$out.log 2>&1; echo "$(date +%T) $out $(tail -1 preds/$out.log)" >> queue.log; fi; }
ev sets_bf16_s0 --prec bf16 --layout sets --suites $SUB
BA16=1 ev setsmix_k48ba_s0 --prec $K48 --codes $C --layout setsmix --suites $SUB
echo CHAIN2DONE >> queue.log
