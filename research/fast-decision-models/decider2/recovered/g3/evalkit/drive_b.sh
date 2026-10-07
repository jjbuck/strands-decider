#!/bin/bash
# sharded baseline runs: drive_b.sh SUITE NSHARDS
cd ~/work/evalkit && source ~/venv/bin/activate
export PYTHONUNBUFFERED=1
S=$1; N=$2
for i in $(seq 0 $((N-1))); do
  python kitrun.py base $S --shard $i/$N --out refs/$S.base.s$i.jsonl > logs/base_${S}_s$i.log 2>&1 &
done
wait
cat refs/$S.base.s*.jsonl > refs/$S.base.jsonl
echo done > logs/base_${S}.DONE
