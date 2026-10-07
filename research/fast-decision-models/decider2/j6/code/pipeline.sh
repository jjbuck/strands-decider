#!/bin/bash
# unattended pipeline after (a): each step independent; logs per step
source ~/venv/bin/activate; cd ~/work/j6
while pgrep -f "python train_a.py" > /dev/null; do sleep 30; done
CK=$(ls -t ck_a/s*.pt | head -1); echo "CK=$CK" > pipeline.state; date >> pipeline.state
# smoke tests (few updates, throwaway ck dirs) so bugs surface early
MINUTES=3 timeout 600 python train_h.py --updates 3 --ck ~/work/j6/ck_smoke_h > smoke_h.log 2>&1; echo "smoke_h rc=$? $(date)" >> pipeline.state
MINUTES=3 timeout 600 python train_e.py $CK --updates 2 --ck ~/work/j6/ck_smoke_e > smoke_e.log 2>&1; echo "smoke_e rc=$? $(date)" >> pipeline.state
[ -f SKIP_EVAL_A ] || python eval_a.py $CK preds_a.json --teacher > eval_a.log 2>&1; echo "eval_a done $(date)" >> pipeline.state
[ -f SKIP_LAT ] || python lat_j6.py check $CK preds_a.json > latcheck.log 2>&1; echo "latcheck done $(date)" >> pipeline.state
[ -f SKIP_LAT ] || python lat_j6.py time $CK > lat.log 2>&1; echo "lat done $(date)" >> pipeline.state
[ -f SKIP_H ] || MINUTES=${HMIN:-95} python train_h.py --updates 3200 > train_h.log 2>&1; echo "train_h done $(date)" >> pipeline.state
[ -f SKIP_E ] || MINUTES=${EMIN:-45} python train_e.py $CK --updates 800 --maxtok 2048 > train_e.log 2>&1; echo "train_e done $(date)" >> pipeline.state
CKH=$(ls -t ck_h/s*.pt 2>/dev/null | head -1)
[ -n "$CKH" ] && python eval_h.py $CKH preds_h.json > eval_h.log 2>&1; echo "eval_h done $(date)" >> pipeline.state
CKE=$(ls -t ck_e/early_s*.pt 2>/dev/null | head -1); CKL=$(ls -t ck_e/late_s*.pt 2>/dev/null | head -1)
[ -n "$CKE" ] && python eval_e.py $CKE preds_e.json > eval_e.log 2>&1; echo "eval_e done $(date)" >> pipeline.state
[ -n "$CKL" ] && python eval_a.py $CKL preds_late.json > eval_late.log 2>&1; echo "eval_late done $(date)" >> pipeline.state
echo ALLDONE >> pipeline.state
