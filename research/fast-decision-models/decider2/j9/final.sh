#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j9
CK=${1:-ckC/s200.pt}
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv > logs/lat_gpu_before.txt
COMP=skip REPS=15 python j9lat.py grid x > logs/lat_grid_skip.log 2>&1
COMP=affine REPS=15 python j9lat.py grid x > logs/lat_grid_affine.log 2>&1
COMP=skip NREAL=60 NLONG=24 REPS=15 python j9lat.py real x > logs/lat_real_skip.log 2>&1
COMP=affine NREAL=60 NLONG=24 REPS=15 python j9lat.py real x > logs/lat_real_affine.log 2>&1
python j9eval.py preds/t_R_affine.json --ckpt $CK --adapter compile --layout R --comp affine > logs/t_R_affine.log 2>&1
python j9eval.py preds/t_R_skip.json --ckpt $CK --adapter compile --layout R --comp skip --suites REAL-agree,LONG > logs/t_R_skip.log 2>&1
echo done > logs/final.done
