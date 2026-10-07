#!/bin/bash
cd ~/work/g4; source ~/venv/bin/activate; export PYTHONUNBUFFERED=1; mkdir -p lmres
while [ ! -f runs/S60_moe/final.pt ]; do sleep 15; done; sleep 45
for r in S125_dense S125_slot S125_dense3 S60_moe; do
  [ -f lmres/zs_$r.json ] || python lmeval.py score $r lmres/zs_$r.json > lmres/zs_$r.log 2>&1
  rm -f lmres/lmft_$r.log; python lmft.py $r > lmres/lmft_$r.out 2>&1
done
echo done > lmres/q2_done
