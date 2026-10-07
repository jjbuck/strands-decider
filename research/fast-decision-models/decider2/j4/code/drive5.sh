#!/bin/bash
# J4 queue v5: after e, only the head-carry ablation (bf); seed repeats dropped for time. Then latency on an idle GPU.
source ~/venv/bin/activate; cd ~/work/j4
run() { local out=$1; shift; [ -f $out/done.json ] && { echo "skip $out"; return; }; echo "=== $out $(date +%T)"; python train.py --out $out "$@" > $out.log 2>&1; echo "=== $out end $(date +%T) rc=$?"; }
while [ ! -f ck/e/done.json ]; do sleep 20; done
run ck/bf --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --stack ck/dp/final.pt --fresh_head 1
echo ALLDONE5 $(date +%T)
