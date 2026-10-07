#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j13/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "queue.sh" >/dev/null; do sleep 20; done
echo "fullF start $(date)"
python route.py full2 ~/work/j13/full_F.jsonl --mode out --cfgs thin@13r64,thin@12r64,thin@13r128,thin@11r64 > ~/work/j13/full_F.log 2>&1
echo "lat2 start $(date)"
python lat.py bench ~/work/j13/lat.json 256,1000,4000 13:64:0.0,12:64:0.0 > ~/work/j13/lat2.log 2>&1
echo "queue2 done $(date)"
