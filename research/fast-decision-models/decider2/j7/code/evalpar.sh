#!/bin/bash
# evalpar.sh OUT [eval_j7 args...] : 2 shards of eval_j7.py in parallel on the GPU, then merge into OUT (+ OUT.ntok.json)
cd ~/work/j7; source ~/venv/bin/activate
out=$1; shift
for i in 0 1; do python eval_j7.py $out.s$i "$@" --shard $i/2 > $out.s$i.log 2>&1 & done
wait
python - "$out" <<'PY'
import json, sys
o = sys.argv[1]; P = {}; N = {}
for i in (0, 1):
    P.update(json.load(open(f'{o}.s{i}'))); N.update(json.load(open(f'{o}.s{i}.ntok.json')))
json.dump(P, open(o, 'w')); json.dump(N, open(o + '.ntok.json', 'w')); print('merged', len(P))
PY
