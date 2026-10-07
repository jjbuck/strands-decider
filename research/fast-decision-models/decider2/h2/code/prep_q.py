"""question bundles with exact hobson token ids + option offsets (plib.P.prep, the references' rendering)."""
import sys, os, json, collections, torch
sys.path[:0] = [os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from kitrun import load_P
P = load_P()
specs = json.load(open(os.path.expanduser('~/work/h2/qspecs.json')))
byname = {}
for s in specs:
    byname.setdefault(s['name'], s)
items = [json.loads(l) for l in open(os.path.expanduser('~/work/evalkit/suites/REAL-agree.jsonl'))]
c4 = collections.Counter(tuple(sorted(it['questions'])) for it in items if len(it['questions']) == 4)
print('top 4-question sets', c4.most_common(3))
q4 = list(c4.most_common(1)[0][0])
it4 = next(it for it in items if tuple(sorted(it['questions'])) == tuple(q4))
bank = [s['name'] for s in specs if s['domain'] == 'banking_knowledge']
q15 = []
for n in bank:
    if n not in q15: q15.append(n)
    if len(q15) == 15: break
B = {'1q': ['details_match'], '4q': q4, '15q': q15}
out = {}
for bn, names in B.items():
    qs = []
    for n in names:
        qd = it4['questions'][n] if (bn == '4q') else byname[n]['spec']
        pr = P.prep('x', qd)
        qs.append(dict(name=n, ids=pr['q'], opt=pr['opt'], kind=pr['rq'].kind, n_slots=pr['rq'].n_slots, temp=P.temp_for(pr['rq'].kind)))
    out[bn] = qs
    print(bn, [(q['name'], len(q['ids']), len(q['opt'])) for q in qs], 'total', sum(len(q['ids']) for q in qs))
torch.save(out, os.path.expanduser('~/work/h2/bundles.pt'))
