#!/bin/bash
# J2 queue v5 (replaces queue4's tail; causal_mntp is already training): qag_early eval, then exclusive latency, then causal_mntp eval.
source ~/venv/bin/activate; cd ~/work/j2
evs() { [ -f preds/$1.json ] || python j2eval.py preds/$1.json --ckpt ck/$1/final.pt --suites JB-all,REAL-agree,CF,CF-probe > logs/eval_$1.log 2>&1; }
while [ ! -f ck/qag_early/final.pt ]; do sleep 20; done
sleep 30
evs qag_early
while [ ! -f ck/causal_mntp/final.pt ]; do sleep 20; done
while pgrep -f "j2train.py|j2eval.py" > /dev/null; do sleep 15; done
[ -f lat.json.done ] || { python j2lat.py time lat.json > logs/lat.log 2>&1 && touch lat.json.done; }
evs causal_mntp
echo ALLDONE > logs/queue5.done
