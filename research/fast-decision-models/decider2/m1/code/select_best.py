"""pick the dev-best checkpoint (lowest mean dev TV to hobson, 366 questions) from a run's curve.json; prints its path"""
import json, sys, os
ck = os.path.expanduser(sys.argv[1]); h = json.load(open(f'{ck}/curve.json'))
cand = [r for r in h if r['step'] > 0 and os.path.exists(f"{ck}/{r.get('tag', '')}.pt")]
best = min(cand, key=lambda r: r['tv'])
print(f"{ck}/{best['tag']}.pt")
