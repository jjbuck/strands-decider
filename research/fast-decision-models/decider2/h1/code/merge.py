"""merge preds of several result files into one (per config, per item, per question); python merge.py out.json in1.json in2.json ..."""
import sys, json, os
out = sys.argv[1]; R = json.load(open(out))['preds'] if os.path.exists(out) else {}
for f in sys.argv[2:]:
    for c, P in json.load(open(f))['preds'].items():
        for iid, qs in P.items(): R.setdefault(c, {}).setdefault(iid, {}).update(qs)
json.dump(dict(preds=R, cfgs=list(R)), open(out, 'w')); print({c: sum(len(v) for v in P.values()) for c, P in R.items()})
