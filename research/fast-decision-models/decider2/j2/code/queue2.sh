#!/bin/bash
# J2 run queue v2 (budget-cut): 280 FT steps x 32 rows per arm (rows <= 4096 tokens), one shared MNTP adaptation (ck/qag_all/mntp.pt) for every
# bidirectional arm. Sequential training; each eval runs in the background alongside the next training run. Resumable.
source ~/venv/bin/activate; cd ~/work/j2; mkdir -p logs preds ck
STEPS=${STEPS:-280}
ev() { [ -f preds/$1.json ] || python j2eval.py preds/$1.json --ckpt ck/$1/final.pt > logs/eval_$1.log 2>&1; }
tr() { name=$1; shift; [ -f ck/$name/final.pt ] || python j2train.py --ck ck/$name --steps $STEPS "$@" >> logs/train_$name.log 2>&1; }
for n in qa_all qag_early; do mkdir -p ck/$n; [ -f ck/$n/mntp.pt ] || cp ck/qag_all/mntp.pt ck/$n/mntp.pt; done
tr qag_all --mode qag --rev all --mntp 200
ev qag_all &
tr causal --mode causal --rev none
ev causal &
tr qa_all --mode qa --rev all --mntp 200
ev qa_all &
tr qag_early --mode qag --rev early --mntp 200
ev qag_early
wait
echo ALLDONE > logs/queue.done
