#!/bin/bash
# fetch only small result files (never weights) for the given runs: fetch.sh S_dec S_slot ...
cd ~/decider2
for r in "$@"; do
  mkdir -p ~/decider2/j11/runs/$r
  for f in meta.json log.jsonl eval.json preds_kit.json preds_kit_rot.json; do ./box.sh j11 get j11/runs/$r/$f ~/decider2/j11/runs/$r/ 2>/dev/null; done
done
