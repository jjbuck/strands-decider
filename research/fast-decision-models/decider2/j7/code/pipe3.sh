#!/bin/bash
# J7 extras (continuation of pipe2 with runtime checks first).
cd ~/work/j7; source ~/venv/bin/activate
CK=/home/ubuntu/work/j7
while pgrep -f "^python eval_j7.py" > /dev/null || pgrep -f "^bash pipe2.sh" > /dev/null; do sleep 20; done
step() { echo "$(date +%H:%M:%S) start $1"; }
[ -f res/check_C.json ] || { step checkC; python lat_j7.py check ck_C/s400.pt - preds/C.json res/check_C.json > logs/check_C.log 2>&1; }
[ -f res/check_T.json ] || { step checkT; python lat_j7.py check ck_T/s400.pt 64k preds/T64k.json res/check_T.json > logs/check_T.log 2>&1; }
[ -f ck_T2/s400.pt ] || { step trainT2; python train_j7.py --arm T --updates 400 --every 100 --lr 1e-4 --hlr 2e-4 --p_real 0.8 --protect 1 --levels 16k:0.3,64k:0.7 --ck $CK/ck_T2 > logs/train_T2.log 2>&1; }
[ -f preds/T2_64k.json ] || { step evalT2; ./evalpar.sh preds/T2_64k.json --ckpt ck_T2/s400.pt --level 64k --protect 1 > logs/eval_T2_64k.log 2>&1; }
[ -f preds/Z16k.json ] || { step zero16; ./evalpar.sh preds/Z16k.json --level 16k > logs/eval_Z16k.log 2>&1; }
for u in 100 200 300; do [ -f preds/T64k_s$u.json ] || { step curveT$u; ./evalpar.sh preds/T64k_s$u.json --ckpt ck_T/s$u.pt --level 64k --suites REAL-agree > logs/eval_T64k_s$u.log 2>&1; }; done
echo "$(date +%H:%M:%S) pipe3 done"
