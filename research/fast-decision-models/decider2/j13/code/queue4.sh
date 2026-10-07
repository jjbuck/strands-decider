#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j13/code
while pgrep -f "queue3.sh" >/dev/null || pgrep -f "route.py full2" >/dev/null; do sleep 20; done
echo "lat bench start $(date)"
NREP=20 python lat.py bench ~/work/j13/lat.json 256,1000,4000 dense,13:64:0.0,12:64:0.0,10:128:0.0,8:512:0.0,8:256:0.2,4:256:0.3,8:512:0.1,4:512:0.5:13r64,8:512:0.0:13r64 > ~/work/j13/lat.log 2>&1
echo "queue4 done $(date)"
