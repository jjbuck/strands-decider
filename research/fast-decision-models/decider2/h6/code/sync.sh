#!/bin/bash
# copy H6 results (preds, logs, bench json) off g1 -- never model files
cd ~/decider2
mkdir -p h6/preds h6/box
./box.sh g1 get h6/preds/ h6/preds/ 2>/dev/null
for f in "train_*" queue.log teach.log "res_bench_*" "bench*.log"; do ./box.sh g1 get "h6/$f" h6/box/ 2>/dev/null; done
ls h6/preds | wc -l
