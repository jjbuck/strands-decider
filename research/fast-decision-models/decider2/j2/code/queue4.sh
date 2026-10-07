#!/bin/bash
# J2 queue v4 (takes over from queue3 during qag_early training): causal_mntp trains alongside qag_early; evals; exclusive latency last.
source ~/venv/bin/activate; cd ~/work/j2
STEPS=280
evs() { [ -f preds/$1.json ] || python j2eval.py preds/$1.json --ckpt ck/$1/final.pt --suites JB-all,REAL-agree,CF,CF-probe > logs/eval_$1.log 2>&1; }
tr() { name=$1; shift; [ -f ck/$name/final.pt ] || python j2train.py --ck ck/$name "$@" >> logs/train_$name.log 2>&1; }
while [ ! -f preds/qa_all.json ]; do sleep 20; done
( tr causal_mntp --mode causal --rev none --mntp 200 --steps $STEPS ) &
CM=$!
while [ ! -f ck/qag_early/final.pt ]; do sleep 20; done
sleep 30
evs qag_early
[ -f preds/qag_all_g0.json ] || python j2eval.py preds/qag_all_g0.json --ckpt ck/qag_all/final.pt --zero_gates 1 --suites REAL-agree,CF,CF-probe > logs/eval_g0.log 2>&1
wait $CM
while pgrep -f "j2train.py|j2eval.py" > /dev/null; do sleep 15; done
[ -f lat.json.done ] || { python j2lat.py time lat.json > logs/lat.log 2>&1 && touch lat.json.done; }
evs causal_mntp
echo ALLDONE > logs/queue4.done
