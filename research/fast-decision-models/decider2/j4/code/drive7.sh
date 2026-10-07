#!/bin/bash
# J4 queue v7: third seed of a and b at 4k rows (seed noise turned out large), after the latency run.
source ~/venv/bin/activate; cd ~/work/j4
run() { local out=$1; shift; [ -f $out/done.json ] && { echo "skip $out"; return; }; echo "=== $out $(date +%T)"; python train.py --out $out "$@" > $out.log 2>&1; echo "=== $out end $(date +%T) rc=$?"; }
while pgrep -f '[j]4lat.py' > /dev/null; do sleep 10; done
run ck/b3 --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --stack ck/dp/final.pt --seed 2
run ck/a3 --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 4000 --save_rows 1000,4000 --seed 2
echo ALLDONE7 $(date +%T)
