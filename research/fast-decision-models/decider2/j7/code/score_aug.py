"""held-out aug pairs (SEED 99, train-split states, 400 pairs): item and pair accuracy per kind.  python3 score_aug.py aug_heldout.jsonl name=preds.json ..."""
import sys, json, collections
items = [json.loads(l) for l in open(sys.argv[1])]
out = {}
for a in sys.argv[2:]:
    nm, p = a.split('='); P = json.load(open(p))
    byk = collections.defaultdict(list); pairs = collections.defaultdict(dict)
    for it in items:
        d = P[it['id']]['detail']; pred = max(d, key=d.get)
        byk[it['kind']].append(pred == it['expected']['detail']); pairs[it['pair']][it['id'][-1]] = (pred == it['expected']['detail'], it['kind'])
    pk = collections.defaultdict(list)
    for pr in pairs.values(): pk[pr['a'][1]].append(pr['a'][0] and pr['b'][0])
    out[nm] = {k: dict(item=round(sum(v) / len(v), 3), pair=round(sum(pk[k]) / len(pk[k]), 3)) for k, v in sorted(byk.items())}
    allp = [x for v in pk.values() for x in v]; out[nm]['ALL'] = dict(pair=round(sum(allp) / len(allp), 3))
    print(nm, {k: v['pair'] for k, v in out[nm].items()})
json.dump(out, open('~/decider2/j7/results/score_aug.json', 'w'), indent=1)
