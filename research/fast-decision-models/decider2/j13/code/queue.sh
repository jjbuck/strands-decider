#!/bin/bash
# sequential GPU queue after full_D
source ~/venv/bin/activate; cd ~/work/j13/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "route.py full2" >/dev/null; do sleep 20; done
echo "prT start $(date)"
python route.py prT ~/work/j13/prT.jsonl --mode out --cfgs orc20@8r256,or220@8r256,orc10@8r512,or210@8r512,orc30@4r256,or230@4r256,thin@13r64,thin@8r512+13r64,orc50@4r512+13r64,orc30@4r512+13r64 > ~/work/j13/prT.log 2>&1
echo "devE start $(date)"
python route.py devE ~/work/j13/dev_E.jsonl --mode out --cfgs thin@13r64,thin@13r128,thin@8r512+13r64,thin@4r512+13r64,orc30@4r512+13r64,orc50@4r512+13r64,orc30@8r512+13r64,orc30@4r256+13r64,orc50@4r256+13r64,rnd30@4r512+13r64 > ~/work/j13/dev_E.log 2>&1
echo "lat check start $(date)"
python lat.py check > ~/work/j13/lat_check.log 2>&1
echo "lat bench start $(date)"
python lat.py bench ~/work/j13/lat.json 256,1000,4000 dense,8:512:0.0,8:256:0.2,4:256:0.3,8:512:0.1,4:512:0.5:13r64,8:512:0.0:13r64,4:256:0.0 > ~/work/j13/lat.log 2>&1
echo "queue done $(date)"
