#!/bin/bash
cd ~/work/m2 && source ~/venv/bin/activate
python m2tf.py res/smoke --configs 'df8=gran=nat;iso=all;comp=docfirst;freeze=8|ro8=gran=const;iso=all;ro=1;freeze=8|df24=gran=nat;iso=all;comp=docfirst' --subset full --suites REAL-agree,CF-probe --limit 60 > res/smoke_tf.log 2>&1; tail -n 2 res/smoke_tf.log
KS=8 python m2lat.py check res/smoke/df8.json 2>&1 | grep -v Loading | tail -n 3
KS=8 RO=1 GRAN=const python m2lat.py check res/smoke/ro8.json 2>&1 | grep -v Loading | tail -n 3
python m2train.py --cfg 'gran=nat;iso=all;comp=docfirst' --elastic 8,12 --ck ~/work/m2/ck_smoke --updates 2 --accum 2 --ndev 4 --dev_min 999 > res/smoke_train.log 2>&1; grep -v Loading res/smoke_train.log | tail -n 8
