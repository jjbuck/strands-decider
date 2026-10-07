#!/bin/bash
# J7 extras after pipe.sh.
cd ~/work/j7; source ~/venv/bin/activate
CK=/home/ubuntu/work/j7
while pgrep -f "^bash pipe.sh" > /dev/null; do sleep 30; done
step() { echo "$(date +%H:%M:%S) start $1"; }
[ -f res/lat_single.json.done ] || { step latsingle; SINGLE=1 python lat_j7.py res/lat_single.json > logs/lat_single.log 2>&1 && touch res/lat_single.json.done; }
for m in C V; do [ -f preds/aug_$m.json ] || { step aug$m; ./evalpar.sh preds/aug_$m.json --ckpt ck_$m/s400.pt --items aug_heldout.jsonl > logs/eval_aug_$m.log 2>&1; }; done
[ -f preds/aug_hob.json ] || { step aughob; ./evalpar.sh preds/aug_hob.json --items aug_heldout.jsonl > logs/eval_aug_hob.log 2>&1; }
[ -f preds/T4k.json ] || { step evalT4; ./evalpar.sh preds/T4k.json --ckpt ck_T/s400.pt --level 4k > logs/eval_T4k.log 2>&1; }
[ -f res/check_TV.json ] || { step checkTV; python lat_j7.py check ck_TV/s400.pt 64k preds/TV64k.json res/check_TV.json > logs/check_TV.log 2>&1; }
[ -f preds/V_noval.json ] || { step ablV; ./evalpar.sh preds/V_noval.json --ckpt ck_V/s400.pt --suites CF,CF-probe --ablate val > logs/eval_V_noval.log 2>&1; }
[ -f ck_T2/s400.pt ] || { step trainT2; python train_j7.py --arm T --updates 400 --every 100 --lr 1e-4 --hlr 2e-4 --p_real 0.8 --protect 1 --levels 16k:0.3,64k:0.7 --ck $CK/ck_T2 > logs/train_T2.log 2>&1; }
[ -f preds/T2_64k.json ] || { step evalT2; ./evalpar.sh preds/T2_64k.json --ckpt ck_T2/s400.pt --level 64k --protect 1 > logs/eval_T2_64k.log 2>&1; }
[ -f preds/Z16k.json ] || { step zero16; ./evalpar.sh preds/Z16k.json --level 16k > logs/eval_Z16k.log 2>&1; }
for u in 100 200 300; do [ -f preds/T64k_s$u.json ] || { step curveT$u; ./evalpar.sh preds/T64k_s$u.json --ckpt ck_T/s$u.pt --level 64k --suites REAL-agree > logs/eval_T64k_s$u.log 2>&1; }; done
echo "$(date +%H:%M:%S) pipe2 done"
