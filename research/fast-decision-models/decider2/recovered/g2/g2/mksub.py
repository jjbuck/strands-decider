"""laptop: pick evaluation subsets (pure json). Writes g2/code/sub_*.json = list of [suite, item_id, qname]"""
import sys, json, hashlib, collections
sys.path.insert(0, '/tmp/decider2/evalkit')
import evalkit as EK
H = lambda s: hashlib.sha1(s.encode()).hexdigest()
def amax(d): return max(d, key=d.get)
out = {}
for suite, n in (('REAL-agree', 400), ('LONG', 200)):
    refs = EK.load_refs(suite); rows = []
    for it in EK.load_suite(suite):
        r = refs.get(it['id'], {})
        for q in it['questions']:
            h = r.get('hobson', {}).get(q); ns = r.get('cfg', {}).get('nostate', {}).get(q)
            if h and ns and amax(EK._norm(h)) != amax(EK._norm(ns)):
                rows.append((suite, it['id'], q, it['n_state_tok']))
    rows.sort(key=lambda x: H(x[1] + x[2]))
    print(suite, 'state-dependent q', len(rows), 'median tok', sorted(r[3] for r in rows)[len(rows)//2])
    out[suite + '-SD'] = [list(r[:3]) for r in rows]
for suite in ('CF', 'CF-probe'):
    hp = EK._as_preds(suite, 'hobson'); tr = []; allp = []
    for pr in EK._pairs(suite):
        q = pr['q']
        ok = amax(EK._norm(hp[pr['a']][q])) == pr['ea'] and amax(EK._norm(hp[pr['b']][q])) == pr['eb']
        allp.append(pr)
        if ok: tr.append(pr)
    print(suite, 'pairs', len(allp), 'hobson-tracked', len(tr), collections.Counter(p['kind'] for p in tr))
    out[suite + '-T'] = [[suite, x, p['q']] for p in tr for x in (p['a'], p['b'])]
    out[suite + '-ALL'] = [[suite, x, p['q']] for p in allp for x in (p['a'], p['b'])]
jb = [['JB-hard', it['id'], q] for it in EK.load_suite('JB-hard') for q in it['questions']]
out['JB-hard'] = jb; print('JB-hard', len(jb))
ra = [['REAL-agree', it['id'], q] for it in EK.load_suite('REAL-agree') for q in it['questions']]
out['REAL-ALL'] = ra
lg = [['LONG', it['id'], q] for it in EK.load_suite('LONG') for q in it['questions']]
out['LONG-ALL'] = lg
print({k: len(v) for k, v in out.items()})
json.dump(out, open('/tmp/decider2/g2/code/subsets.json', 'w'))
