#!/bin/bash
# J3 box pipeline v4 (continues v3 after the DT-set run): set evals, holdout, tf full 16A, ruletaker-heavy continuation of DT-A8 (decisive
# test: is the multi-hop loss data or structure?), its evals, then remaining training-free full evals.
cd ~/work/j3; source ~/venv/bin/activate; mkdir -p preds order logs holdout
log() { echo "$(date -u +%H:%M:%S) $*" >> logs/pipeline.log; }
while pgrep -f "train_dt.py --layout set" >/dev/null; do sleep 30; done
log "v4: set train done"
CS=$(ls -v ck_set812/s*.pt | tail -1); log "set ckpt $CS"
python dt_eval.py eval preds/set812_L8.json --ckpt $CS --Ls 8 --bridge A --layout set > logs/ev_set_L8.log 2>&1; log "ev set L8 done"
python dt_order.py order/set812_L8.json --ckpt $CS --Ls 8 --layout set > logs/or_set_L8.log 2>&1; log "order set L8 done"
LAYOUT=set python dt_lat.py check $CS preds/set812_L8.json 8 > logs/latcheck_set8.log 2>&1
KINDS=set LSS=8,12 python dt_lat.py time $CS > logs/lat_set.log 2>&1; log "set latency done"
python dt_eval.py eval preds/set812_L12.json --ckpt $CS --Ls 12 --bridge A --layout set > logs/ev_set_L12.log 2>&1; log "ev set L12 done"
python dt_holdout.py holdout/set812.json --ckpt $CS --splits 8,12 --layout set --n 150 > logs/holdout_set.log 2>&1; log "holdout set done"
python dt_eval.py eval preds/tf_full_16A.json --Ls 16 --bridge A > logs/ev_tf16.log 2>&1; log "tf full 16A done"
RTARGS=$(cat rt_args.txt 2>/dev/null || echo "--init ck_a812/s700.pt --Ls 8 --bridge A --v5_tasks ruletaker_d0,ruletaker_d1,ruletaker_d2 --p_real 0.35 --p_cf 0.15 --updates 240 --accum 16 --every 80 --max_hours 0.95"); log "rt args $RTARGS"
python train_dt.py $RTARGS --ck ~/work/j3/ck_rt8 > logs/train_rt8.log 2>&1; log "rt train done"
CR=$(ls -v ck_rt8/s*.pt | tail -1); log "rt ckpt $CR"
python dt_holdout.py holdout/rt8.json --ckpt $CR --splits 8 --n 150 > logs/holdout_rt8.log 2>&1; log "holdout rt done"
python dt_eval.py eval preds/rt8_L8.json --ckpt $CR --Ls 8 --bridge A > logs/ev_rt8.log 2>&1; log "ev rt done"
python dt_order.py order/set812_L12.json --ckpt $CS --Ls 12 --layout set > logs/or_set_L12.log 2>&1; log "order set L12 done"
python dt_eval.py eval preds/tf_full_12G.json --Ls 12 --bridge G > logs/ev_tf12G.log 2>&1; log "tf full 12G done"
python dt_eval.py eval preds/tf_full_12A.json --Ls 12 --bridge A > logs/ev_tf12.log 2>&1; log "tf full 12A done"
log "pipeline v4 done"
