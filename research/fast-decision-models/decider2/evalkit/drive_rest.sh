#!/bin/bash
# remaining grid for CF-probe and LONG after the lite runs
cd ~/work/evalkit && source ~/venv/bin/activate
export PYTHONUNBUFFERED=1 BASE_GRID=rest
while [ ! -f logs/D2_DONE ]; do sleep 20; done
for S in CF-probe LONG; do
  for i in 0 1 2; do python kitrun.py base $S --shard $i/3 --out refs/$S.baserest.s$i.jsonl > logs/rest_${S}_s$i.log 2>&1 & done
  wait
  cat refs/$S.base.s*.jsonl refs/$S.baserest.s*.jsonl > refs/$S.base.jsonl
done
echo done > logs/REST_DONE
