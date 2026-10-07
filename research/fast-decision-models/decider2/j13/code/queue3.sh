#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j13/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "queue2.sh" >/dev/null; do sleep 20; done
echo "heal start $(date)"
LR=5e-5 ACC=4 python heal.py 10 128 1200 ~/work/j13/heal_10_128.pt > ~/work/j13/heal.log 2>&1
echo "heal eval start $(date)"
python route.py healE ~/work/j13/healE_base.jsonl --mode out --cfgs thin@10r128,thin@11r128 > ~/work/j13/healE_base.log 2>&1
python route.py healE ~/work/j13/healE_healed.jsonl --mode out --cfgs thin@10r128,thin@11r128 --heal ~/work/j13/heal_10_128.pt > ~/work/j13/healE_healed.log 2>&1
echo "queue3 done $(date)"
