#!/bin/bash
# J4 queue v3 (replaces drive.sh/drive2.sh after arm c): NTP control, then d at 4k rows (the regime where DP differs), ablations.
source ~/venv/bin/activate; cd ~/work/j4
run() { local out=$1; shift; [ -f $out/done.json ] && { echo "skip $out"; return; }; echo "=== $out $(date +%T)"; python train.py --out $out "$@" > $out.log 2>&1; echo "=== $out end $(date +%T) rc=$?"; }
while [ ! -f ck/c/done.json ]; do sleep 20; done
run ck/ntp --mode ntp --data data/dp.jsonl --tok 4.4 --save_tok '' --qmax 40
run ck/d --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --stack ck/ntp/final.pt --fresh_head 1
run ck/bf --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --stack ck/dp/final.pt --fresh_head 1
run ck/a2 --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --seed 1
run ck/b2 --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --stack ck/dp/final.pt --seed 1
echo ALLDONE3 $(date +%T)
