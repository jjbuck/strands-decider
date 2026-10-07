#!/bin/bash
# J3 box pipeline v3: waits for the main run, then: evals (exclusive), order metrics, calibration fit, latency (exclusive, all kinds),
# option-set run + evals, training-free full 16A, a bold 4A run + eval, training-free full 12G / 12A.
cd ~/work/j3; source ~/venv/bin/activate; mkdir -p preds order logs calib
log() { echo "$(date -u +%H:%M:%S) $*" >> logs/pipeline.log; }
while pgrep -f "train_dt.py --Ls 8,8,12 --bridge A --updates 700" >/dev/null; do sleep 30; done
while pgrep -f "[b]ash curve.sh" >/dev/null; do sleep 20; done
CK=$(ls -v ck_a812/s*.pt | tail -1); log "main ckpt $CK"
python dt_eval.py eval preds/a812_L8.json --ckpt $CK --Ls 8 --bridge A > logs/ev_a812_L8.log 2>&1; log "ev L8 done"
python dt_eval.py eval preds/a812_L12.json --ckpt $CK --Ls 12 --bridge A > logs/ev_a812_L12.log 2>&1; log "ev L12 done"
python dt_order.py order/hobson.json > logs/or_hob.log 2>&1; log "order hobson done"
python dt_order.py order/a812_L8.json --ckpt $CK --Ls 8 > logs/or_a812_L8.log 2>&1; log "order L8 done"
python dt_order.py order/a812_L12.json --ckpt $CK --Ls 12 > logs/or_a812_L12.log 2>&1; log "order L12 done"
python dt_calib.py calib/a812_L8.json --ckpt $CK --Ls 8 > logs/calib8.log 2>&1
python dt_calib.py calib/a812_L12.json --ckpt $CK --Ls 12 > logs/calib12.log 2>&1; log "calib done"
python dt_lat.py check $CK preds/a812_L8.json 8 > logs/latcheck8.log 2>&1
python dt_lat.py check $CK preds/a812_L12.json 12 > logs/latcheck12.log 2>&1; log "latcheck done"
BRIDGE_G=1 KINDS=hob1,hobB,dt,dtG LSS=4,8,12,16 python dt_lat.py time $CK > logs/lat.log 2>&1; log "latency done"
SETARGS=$(cat set_args.txt 2>/dev/null || echo "--layout set --Ls 8,8,12 --bridge A --updates 400 --accum 16 --every 100 --max_hours 1.6"); log "set args $SETARGS"
python train_dt.py $SETARGS --ck ~/work/j3/ck_set812 > logs/train_set812.log 2>&1; log "set train done"
CS=$(ls -v ck_set812/s*.pt | tail -1); log "set ckpt $CS"
python dt_eval.py eval preds/set812_L8.json --ckpt $CS --Ls 8 --bridge A --layout set > logs/ev_set_L8.log 2>&1; log "ev set L8 done"
python dt_order.py order/set812_L8.json --ckpt $CS --Ls 8 --layout set > logs/or_set_L8.log 2>&1; log "order set L8 done"
LAYOUT=set python dt_lat.py check $CS preds/set812_L8.json 8 > logs/latcheck_set8.log 2>&1
KINDS=set LSS=8,12 python dt_lat.py time $CS > logs/lat_set.log 2>&1; log "set latency done"
python dt_eval.py eval preds/set812_L12.json --ckpt $CS --Ls 12 --bridge A --layout set > logs/ev_set_L12.log 2>&1; log "ev set L12 done"
python dt_eval.py eval preds/tf_full_16A.json --Ls 16 --bridge A > logs/ev_tf16.log 2>&1; log "tf full 16A done"
A4ARGS=$(cat a4_args.txt 2>/dev/null || echo "--Ls 4 --bridge A --updates 260 --accum 16 --every 100 --max_hours 1.0"); log "a4 args $A4ARGS"
python train_dt.py $A4ARGS --ck ~/work/j3/ck_a4 > logs/train_a4.log 2>&1; log "a4 train done"
C4=$(ls -v ck_a4/s*.pt | tail -1); log "a4 ckpt $C4"
python dt_eval.py eval preds/a4_L4.json --ckpt $C4 --Ls 4 --bridge A > logs/ev_a4.log 2>&1; log "ev a4 done"
python dt_order.py order/set812_L12.json --ckpt $CS --Ls 12 --layout set > logs/or_set_L12.log 2>&1; log "order set L12 done"
python dt_eval.py eval preds/tf_full_12G.json --Ls 12 --bridge G > logs/ev_tf12G.log 2>&1; log "tf full 12G done"
python dt_eval.py eval preds/tf_full_12A.json --Ls 12 --bridge A > logs/ev_tf12.log 2>&1; log "tf full 12A done"
log "pipeline done"
