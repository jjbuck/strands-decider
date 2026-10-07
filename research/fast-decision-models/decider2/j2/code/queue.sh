#!/bin/bash
# J2 run queue (sequential training; each eval runs in the background alongside the next training run). Resumable: rerun the script.
source ~/venv/bin/activate; cd ~/work/j2; mkdir -p logs preds ck
STEPS=${STEPS:-500}; MNTP=${MNTP:-200}
ev() { [ -f preds/$1.json ] || python j2eval.py preds/$1.json --ckpt ck/$1/final.pt > logs/eval_$1.log 2>&1; }
tr() { name=$1; shift; [ -f ck/$name/final.pt ] || python j2train.py --ck ck/$name --steps $STEPS "$@" > logs/train_$name.log 2>&1; }
tr qag_all --mode qag --rev all --mntp $MNTP
ev qag_all &
tr causal --mode causal --rev none
ev causal &
mkdir -p ck/qa_all; [ -f ck/qa_all/mntp.pt ] || cp ck/qag_all/mntp.pt ck/qa_all/mntp.pt
tr qa_all --mode qa --rev all --mntp $MNTP
ev qa_all &
tr qag_early --mode qag --rev early --mntp $MNTP
ev qag_early
wait
echo ALLDONE > logs/queue.done
