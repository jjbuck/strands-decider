#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j9; mkdir -p preds logs
for cfg in "R affine" "S affine" "Rx affine" "R skip" "S skip"; do
  set -- $cfg
  python j9eval.py preds/d_$1_$2.json --layout $1 --comp $2 --suites REAL-agree,LONG,CF --stride 3 > logs/d_$1_$2.log 2>&1
done
echo done > logs/diag.done
