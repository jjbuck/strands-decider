#!/bin/bash
# M2 phase 4 (after phase3c): latency with the fused single-question pass (QFUSED), exclusive GPU.
cd ~/work/m2 && source ~/venv/bin/activate
log() { echo "$(date +%T) $*" >> res/pipeline.log; }
while [ ! -f res/phase3c.done ]; do sleep 30; done
log "phase4 start"
QFUSED=1 KS=8 CKPT=ck_N/s512.pt python m2lat.py check res/full_N_k8/m.json > res/latcheck_qf.log 2>&1
QFUSED=1 Q1ONLY=1 KS=8,12 CKPT=ck_N/s512.pt TAG=_qf python m2lat.py grid > res/lat_grid_qf.log 2>&1
QFUSED=1 Q1ONLY=1 KS=8,12 RO=1 GRAN=const CKPT=ck_C/s470.pt TAG=_C_qf python m2lat.py grid > res/lat_grid_C_qf.log 2>&1
QFUSED=1 KS=8,12 GRAN=nat CKPT=ck_N/s512.pt TAG=_qf python m2lat.py real > res/lat_real_qf.log 2>&1
QFUSED=1 KS=8,12 RO=1 GRAN=const CKPT=ck_C/s470.pt TAG=_C_qf python m2lat.py real > res/lat_real_C_qf.log 2>&1
log "phase4 done"; touch res/phase4.done
