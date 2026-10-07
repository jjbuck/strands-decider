"""synthetic dev scoring: per-kind item accuracy and pair (both-right) accuracy.  python3 score_syn.py dev.jsonl preds1.jsonl [preds2 ...]"""
import sys, json, collections
dev = [json.loads(l) for l in open(sys.argv[1])]
for fn in sys.argv[2:]:
    P = {}
    for l in open(fn):
        r = json.loads(l); P[r['id']] = r['p']
    acc = collections.defaultdict(list); pair = collections.defaultdict(dict)
    for it in dev:
        if it['id'] not in P: continue
        p = P[it['id']]; ok = max(p, key=p.get) == it['expected']['x']
        acc[it['kind']].append(ok); pair[(it['kind'], it['pair'])][it['id'][-1]] = ok
    pk = collections.defaultdict(list)
    for (k, _), d in pair.items():
        if len(d) == 2: pk[k].append(d['a'] and d['b'])
    allacc = [x for v in acc.values() for x in v]; allp = [x for v in pk.values() for x in v]
    print(fn, 'items %d acc %.3f pairs %d flip %.3f' % (len(allacc), sum(allacc) / max(1, len(allacc)), len(allp), sum(allp) / max(1, len(allp))))
    print('   ' + '  '.join(f'{k} {sum(pk[k]) / len(pk[k]):.2f}' for k in sorted(pk)))
