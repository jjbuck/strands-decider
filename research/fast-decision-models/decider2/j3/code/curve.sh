#!/bin/bash
# learning curve on the subset: evaluate ck_a812/s{300,500} at Ls 8 and 12 as they appear (runs alongside training)
cd ~/work/j3; source ~/venv/bin/activate
for s in 300 500; do
  while [ ! -f ck_a812/s$s.pt ] && pgrep -f "train_dt.py --Ls 8,8,12 --bridge A --updates 700" >/dev/null; do sleep 60; done
  [ -f ck_a812/s$s.pt ] || break
  sleep 20
  for L in 8 12; do python dt_eval.py eval curve/a812_s${s}_L$L.json --ckpt ck_a812/s$s.pt --Ls $L --subset 120 > logs/curve${s}_$L.log 2>&1; done
done
