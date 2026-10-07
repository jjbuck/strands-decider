#!/bin/bash
# J7 GPU pipeline (sequential, resumable: a step is skipped when its output exists).  Run: ./L.sh pipe bash pipe.sh
cd ~/work/j7; source ~/venv/bin/activate
CK=/home/ubuntu/work/j7
RECIPE="--lr 1e-4 --hlr 2e-4 --p_real 0.6 --p_aug 0.2 --w_aug 0.5"
while pgrep -f "^python train_j7.py --arm C" > /dev/null; do sleep 30; done
step() { echo "$(date +%H:%M:%S) start $1"; }
mkdir -p preds res
[ -f res/lat_j7.json.done ] || { step lat; python lat_j7.py res/lat_j7.json > logs/lat.log 2>&1 && touch res/lat_j7.json.done; }
[ -f ck_T/s400.pt ] || { step trainT; python train_j7.py --arm T --updates 400 --every 100 --lr 1e-4 --hlr 2e-4 --p_real 0.8 --ck $CK/ck_T > logs/train_T.log 2>&1; }
[ -f preds/C.json ] || { step evalC; ./evalpar.sh preds/C.json --ckpt ck_C/s400.pt > logs/eval_C.log 2>&1; }
[ -f ck_V/s400.pt ] || { step trainV; python train_j7.py --arm V --updates 400 --every 100 $RECIPE --ck $CK/ck_V > logs/train_V.log 2>&1; }
[ -f preds/T64k.json ] || { step evalT64; ./evalpar.sh preds/T64k.json --ckpt ck_T/s400.pt --level 64k > logs/eval_T64k.log 2>&1; }
[ -f ck_TV/s400.pt ] || { step trainTV; python train_j7.py --arm TV --updates 400 --every 100 $RECIPE --init ck_T/s400.pt --levels 16k:0.3,64k:0.7 --ck $CK/ck_TV > logs/train_TV.log 2>&1; }
[ -f preds/V.json ] || { step evalV; ./evalpar.sh preds/V.json --ckpt ck_V/s400.pt > logs/eval_V.log 2>&1; }
[ -f preds/TV64k.json ] || { step evalTV64; ./evalpar.sh preds/TV64k.json --ckpt ck_TV/s400.pt --level 64k > logs/eval_TV64k.log 2>&1; }
[ -f preds/T16k.json ] || { step evalT16; ./evalpar.sh preds/T16k.json --ckpt ck_T/s400.pt --level 16k > logs/eval_T16k.log 2>&1; }
echo "$(date +%H:%M:%S) pipeline done"
