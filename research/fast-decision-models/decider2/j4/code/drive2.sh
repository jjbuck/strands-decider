#!/bin/bash
# J4 extra arms after drive.sh: head-carry ablation and seed repeats at 4k rows.
source ~/venv/bin/activate; cd ~/work/j4
run() { local out=$1; shift; [ -f $out/done.json ] && { echo "skip $out"; return; }; echo "=== $out $(date +%T)"; python train.py --out $out "$@" > $out.log 2>&1; echo "=== $out end $(date +%T) rc=$?"; }
while ! grep -q ALLDONE drive.log; do sleep 30; done
run ck/bf --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --stack ck/dp/final.pt --fresh_head 1
run ck/a2 --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --seed 1
run ck/b2 --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --stack ck/dp/final.pt --seed 1
echo ALLDONE2 $(date +%T)
