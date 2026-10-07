#!/bin/bash
# J7: one more budget point for the transplant: TL s700 + 700 updates at 64k (total ~1800 updates), then full eval.
cd ~/work/j7; source ~/venv/bin/activate
CK=/home/ubuntu/work/j7
step() { echo "$(date +%H:%M:%S) start $1"; }
[ -f ck_TL2/s700.pt ] || { step trainTL2; python train_j7.py --arm T --updates 700 --every 350 --lr 1e-4 --hlr 2e-4 --p_real 0.8 --levels 64k:1.0 --init ck_TL/s700.pt --seed 13 --ck $CK/ck_TL2 > logs/train_TL2.log 2>&1; }
[ -f preds/TL2_64k.json ] || { step evalTL2; ./evalpar.sh preds/TL2_64k.json --ckpt ck_TL2/s700.pt --level 64k > logs/eval_TL2_64k.log 2>&1; }
echo "$(date +%H:%M:%S) pipe6 done"
