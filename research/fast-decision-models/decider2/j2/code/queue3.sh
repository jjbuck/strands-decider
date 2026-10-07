#!/bin/bash
# J2 queue v3 (takes over from queue2 after qa_all training): qag_early, exclusive latency, then the reading-augmentation phase
# replaced by the MNTP-matched causal control (causal_mntp: same MNTP adaptation, causal) and the zero-gate ablation. Resumable.
source ~/venv/bin/activate; cd ~/work/j2
STEPS=280
ev() { [ -f preds/$1.json ] || python j2eval.py preds/$1.json --ckpt ck/$1/final.pt > logs/eval_$1.log 2>&1; }
evs() { [ -f preds/$1.json ] || python j2eval.py preds/$1.json --ckpt ck/$1/final.pt --suites JB-all,REAL-agree,CF,CF-probe > logs/eval_$1.log 2>&1; }
tr() { name=$1; shift; [ -f ck/$name/final.pt ] || python j2train.py --ck ck/$name "$@" >> logs/train_$name.log 2>&1; }
while [ ! -f ck/qa_all/final.pt ]; do sleep 30; done
sleep 20
ev qa_all &
tr qag_early --mode qag --rev early --mntp 200 --steps $STEPS
while [ ! -f preds/qa_all.json ] || [ ! -f preds/causal.json ]; do sleep 30; done
while pgrep -f "j2train.py|j2eval.py" > /dev/null; do sleep 15; done
[ -f lat.json.done ] || { python j2lat.py time lat.json > logs/lat.log 2>&1 && touch lat.json.done; }
evs qag_early &
tr causal_mntp --mode causal --rev none --mntp 200 --steps $STEPS
ev causal_mntp &
[ -f preds/qag_all_g0.json ] || python j2eval.py preds/qag_all_g0.json --ckpt ck/qag_all/final.pt --zero_gates 1 --suites REAL-agree,CF,CF-probe > logs/eval_g0.log 2>&1
wait
echo ALLDONE > logs/queue3.done
