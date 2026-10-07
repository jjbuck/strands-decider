#!/bin/bash
cd ~/work/g4; source ~/venv/bin/activate; export PYTHONUNBUFFERED=1; mkdir -p lmres
for r in S60_dense S60_dense3 S60_slot S60_mudd; do
  [ -f lmres/zs_$r.json ] || python lmeval.py score $r lmres/zs_$r.json > lmres/zs_$r.log 2>&1
  python lmft.py $r > lmres/lmft_$r.out 2>&1
done
while [ ! -f runs/S125_mudd/final.pt ]; do sleep 20; done; sleep 60
python lmeval.py score S125_mudd lmres/zs_S125_mudd.json > lmres/zs_S125_mudd.log 2>&1
python lmft.py S125_mudd > lmres/lmft_S125_mudd.out 2>&1
echo done > lmres/q2_done
