#!/bin/bash
cd ~/work/g4; source ~/venv/bin/activate; export PYTHONUNBUFFERED=1; mkdir -p lmres
python lmeval.py build > lmres/build.log 2>&1
for r in S125_dense S125_slot S125_dense3; do
  python lmeval.py score $r lmres/zs_$r.json > lmres/zs_$r.log 2>&1
  python lmft.py $r > lmres/lmft_$r.out 2>&1
done
while [ ! -f runs/S60_moe/final.pt ]; do sleep 20; done; sleep 60
python lmeval.py score S60_moe lmres/zs_S60_moe.json > lmres/zs_S60_moe.log 2>&1
python lmft.py S60_moe > lmres/lmft_S60_moe.out 2>&1
echo done > lmres/q2_done
