#!/bin/bash
# J4 training queue (sequential on the GPU). Resumable: each step skips if its done.json exists.
source ~/venv/bin/activate; cd ~/work/j4
run() { local out=$1; shift; [ -f $out/done.json ] && { echo "skip $out"; return; }; echo "=== $out $(date +%T)"; python train.py --out $out "$@" > $out.log 2>&1; echo "=== $out end $(date +%T) rc=$?"; }
mkdir -p ck
DPTOK=${DPTOK:-4.4}
FTROWS=${FTROWS:-12000}
run ck/dp --mode dp --data data/dp.jsonl --tok $DPTOK --save_tok 1.1,2.2 --qmax 40
while [ ! -f teach.done ]; do grep -q '^done' teach.log && touch teach.done || sleep 30; done
run ck/a --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows $FTROWS --save_rows 1000,2000,4000,8000,$FTROWS
run ck/b --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows $FTROWS --save_rows 1000,2000,4000,8000,$FTROWS --stack ck/dp/final.pt
# (c): arm a continued (constant LR) until its total tokens = a's tokens + the DP phase's tokens
CT=$(python -c "import json;print((json.load(open('ck/a/done.json'))['tok']+json.load(open('ck/dp/done.json'))['tok'])/1e6)")
run ck/c --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows 99999 --tok_stop $CT --save_rows 16000,20000 --init_from ck/a/last.pt
run ck/ntp --mode ntp --data data/dp.jsonl --tok $DPTOK --save_tok '' --qmax 40
run ck/d --mode ft --data data/ft.jsonl --teacher data/teacher.jsonl --rows $FTROWS --save_rows 1000,2000,4000,8000,$FTROWS --stack ck/ntp/final.pt --fresh_head 1
echo ALLDONE $(date +%T)
