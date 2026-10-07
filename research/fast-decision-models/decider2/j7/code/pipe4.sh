#!/bin/bash
cd ~/work/j7; source ~/venv/bin/activate
while pgrep -f "^bash pipe3.sh" > /dev/null; do sleep 20; done
step() { echo "$(date +%H:%M:%S) start $1"; }
for m in T:64k:T64k TV:64k:TV64k; do IFS=: read a l p <<< "$m"; step check$a; python lat_j7.py check ck_$a/s400.pt $l preds/$p.json res/check2_$a.json > logs/check2_$a.log 2>&1; done
echo "$(date +%H:%M:%S) pipe4 done"
