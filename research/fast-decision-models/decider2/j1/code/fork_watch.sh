#!/bin/bash
# copy ck_main/last.pt when it holds step 1700 (the e1a fork point), before the 1800 overwrite
cd ~/work/j1
while true; do
  s=$(python3 - <<'PY' 2>/dev/null
import json
try:
    L=[json.loads(l) for l in open('ck_main/train_log.jsonl') if '"step"' in l and 'lc' not in l]
    print(L[-1]['step'])
except Exception: print(0)
PY
)
  if [ "$s" -ge 1700 ] && [ ! -f fork1700.pt ]; then
    sleep 20
    ~/venv/bin/python -c "import torch; c=torch.load('ck_main/last.pt', weights_only=False); print(c['step'])" > fork_step.txt 2>&1
    if grep -q '^1700$' fork_step.txt; then cp ck_main/last.pt fork1700.pt; echo "$(date +%T) forked at 1700" >> chain_main.log; exit 0; fi
    echo "$(date +%T) fork check saw $(cat fork_step.txt)" >> chain_main.log
    if [ "$s" -ge 1790 ]; then exit 1; fi
  fi
  sleep 30
done
