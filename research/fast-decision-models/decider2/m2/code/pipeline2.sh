#!/bin/bash
# M2 pipeline part 2 (replaces pipeline.sh after N was stopped at a dev checkpoint): train C while N is evaluated, then evaluate C.
cd ~/work/m2 && source ~/venv/bin/activate
NCFG='gran=nat;iso=all;comp=docfirst'
CCFG='gran=const;iso=all;ro=1'
CH=${CH:-2.8}; CU=${CU:-620}
log() { echo "$(date +%T) $*" >> res/pipeline.log; }
train() {   # name cfg hours updates
  [ -f ck_$1/DONE ] && return
  log "train $1 start"
  python m2train.py --cfg "$2" --elastic 8,12 --ck ~/work/m2/ck_$1 --max_hours $3 --updates $4 --ndev 100 --p_real 0.55 --p_cf 0.08 \
     >> res/train_$1.log 2>&1 && touch ck_$1/DONE
  log "train $1 end rc=$?"
}
evalrun() {  # name cfg
  CK=$(ls -t ck_$1/s*.pt | head -1)
  log "eval $1 $CK"
  for k in 8 12; do
    [ -f res/full_$1_k$k/m.json ] || python m2tf.py res/full_$1_k$k --configs "m=$2;freeze=$k" --subset full --ckpt $CK > res/full_$1_k$k.log 2>&1
  done
  [ -f res/hold_$1.json ] || python m2hold.py res/hold_$1.json --ckpt $CK --configs "k8=$2;freeze=8|k12=$2;freeze=12|k8sent=${2/gran=nat/gran=sent};freeze=8" --tasks ruletaker_d3,ruletaker_d5,ruletaker_natlang > res/hold_$1.log 2>&1
  [ -f res/pre_$1.json ] || python m2pre.py res/pre_$1.json --ckpt $CK --cfg "$2;freeze=8" --n 60 > res/pre_$1.log 2>&1
  log "eval $1 done"
}
( train C "$CCFG" $CH $CU ) &
evalrun N "$NCFG"
wait
evalrun C "$CCFG"
log "pipeline2 done"
touch res/pipeline2.done
