#!/bin/bash
# M2 phase 3: wait for C, latency on an exclusive GPU, then N continuation alongside C's evals, then N2 evals.
cd ~/work/m2 && source ~/venv/bin/activate
NCFG='gran=nat;iso=all;comp=docfirst'
CCFG='gran=const;iso=all;ro=1'
log() { echo "$(date +%T) $*" >> res/pipeline.log; }
while [ ! -f ck_C/DONE ] && pgrep -f "m2train.py --cfg gran=const" > /dev/null; do sleep 30; done
sleep 5
NCK=$(ls -t ck_N/s*.pt | head -1); CCK=$(ls -t ck_C/s*.pt | head -1)
log "phase3 latency start N=$NCK C=$CCK"
[ -f res/lat_grid.json ] && grep -q '"Q4_T4000_m2k24_c0.55"' res/lat_grid.json || KS=8,12 CKPT=$NCK python m2lat.py grid > res/lat_grid.log 2>&1
KS=8,12 RO=1 GRAN=const CKPT=$CCK TAG=_C python m2lat.py grid > res/lat_grid_C.log 2>&1
KS=8,12 GRAN=nat CKPT=$NCK python m2lat.py real > res/lat_real.log 2>&1
KS=8,12 RO=1 GRAN=const CKPT=$CCK TAG=_C python m2lat.py real > res/lat_real_C.log 2>&1
log "phase3 latency done"
# N continuation (resumes ck_N/last.pt; LR schedule re-evaluated over 820 updates) alongside C's evals
( python m2train.py --cfg "$NCFG" --elastic 8,12 --ck ~/work/m2/ck_N --max_hours 4.1 --updates 820 --ndev 100 --p_real 0.55 --p_cf 0.08 >> res/train_N2.log 2>&1; log "N2 end rc=$?" ) &
CK=$CCK
for k in 8 12; do
  [ -f res/full_C_k$k/m.json ] || python m2tf.py res/full_C_k$k --configs "m=$CCFG;freeze=$k" --subset full --ckpt $CK > res/full_C_k$k.log 2>&1
done
[ -f res/hold_C.json ] || python m2hold.py res/hold_C.json --ckpt $CK --configs "k8=$CCFG;freeze=8|k12=$CCFG;freeze=12" --tasks ruletaker_d3,ruletaker_d5,ruletaker_natlang > res/hold_C.log 2>&1
[ -f res/pre_C.json ] || python m2pre.py res/pre_C.json --ckpt $CK --cfg "$CCFG;freeze=8" --n 60 > res/pre_C.log 2>&1
log "eval C done"
wait
N2=$(ls -t ck_N/s*.pt | head -1)
log "eval N2 $N2"
for k in 8 12; do
  python m2tf.py res/full_N2_k$k --configs "m=$NCFG;freeze=$k" --subset full --ckpt $N2 > res/full_N2_k$k.log 2>&1
done
python m2hold.py res/hold_N2.json --ckpt $N2 --configs "k8=$NCFG;freeze=8|k12=$NCFG;freeze=12" --tasks ruletaker_d3,ruletaker_d5,ruletaker_natlang > res/hold_N2.log 2>&1
log "phase3 done"; touch res/phase3.done
