#!/bin/bash
# J7: does the transplant keep improving with budget?  Continue T (s400) for 700 more updates at the 64k level only, then full eval.
cd ~/work/j7; source ~/venv/bin/activate
CK=/home/ubuntu/work/j7
while pgrep -f "^bash pipe4.sh" > /dev/null || pgrep -f "^bash pipe3.sh" > /dev/null; do sleep 20; done
step() { echo "$(date +%H:%M:%S) start $1"; }
[ -f ck_TL/s700.pt ] || { step trainTL; python train_j7.py --arm T --updates 700 --every 350 --lr 1e-4 --hlr 2e-4 --p_real 0.8 --levels 64k:1.0 --init ck_T/s400.pt --seed 11 --ck $CK/ck_TL > logs/train_TL.log 2>&1; }
[ -f preds/TL64k.json ] || { step evalTL; ./evalpar.sh preds/TL64k.json --ckpt ck_TL/s700.pt --level 64k > logs/eval_TL64k.log 2>&1; }
echo "$(date +%H:%M:%S) pipe5 done"
