#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j9
CK=${1:-ckC2/s400.pt}
python j9eval.py preds/t4_R_affine.json --ckpt $CK --adapter compile --layout R --comp affine > logs/t4_R_affine.log 2>&1
python j9eval.py preds/t4_R_skip.json --ckpt $CK --adapter compile --layout R --comp skip --suites REAL-agree,LONG,CF,CF-probe > logs/t4_R_skip.log 2>&1
echo done > logs/final2.done
