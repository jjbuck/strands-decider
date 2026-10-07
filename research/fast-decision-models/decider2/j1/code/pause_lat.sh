#!/bin/bash
# exclusive-GPU latency window: pause training (STOP -> saves last.pt), copy the resume file, run checks + timings, resume training
source ~/venv/bin/activate; cd ~/work/j1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date +%T) pause requested" >> chain_lat.log
touch ck_main/STOP
while pgrep -f train_j1.py > /dev/null; do sleep 5; done
rm -f ck_main/STOP; cp ck_main/last.pt lat_ck.pt
echo "$(date +%T) training paused" >> chain_lat.log
python lat_enc.py check lat_ck.pt > latcheck_enc.log 2>&1
echo "$(date +%T) enc check done" >> chain_lat.log
python lat_enc.py time lat_ck.pt lat_enc.json > lat_enc.log 2>&1
echo "$(date +%T) enc timing done" >> chain_lat.log
python lat_hob.py lat_hob.json > lat_hob.log 2>&1
echo "$(date +%T) hob timing done" >> chain_lat.log
setsid nohup ./run_main.sh < /dev/null > /dev/null 2>&1 &
echo "$(date +%T) training resumed" >> chain_lat.log
