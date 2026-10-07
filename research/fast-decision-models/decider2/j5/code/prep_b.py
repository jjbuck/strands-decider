"""J5: token-length statistics of the evalkit suites (state / question tokens, exact hobson ids) and short-traffic question bundles.
-> ~/work/j5/lenstats.json, ~/work/j5/bundles.pt  (same record format as h2 bundles: name, ids, opt, kind, n_slots, temp)
bundles: jb1 = the JB-all question of median length; jb4 = 4 distinct JB-all questions nearest the median length;
         bk1 = details_match (h2's 1q, 125 tok); bk4 = 4 distinct REAL-agree questions nearest the REAL median question length."""
import os, sys, json, collections, statistics as stt, torch
sys.path[:0] = [os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import evalkit as EK
from kitrun import load_P
P = load_P()

def pct(a, ps=(10, 25, 50, 75, 90)):
    a = sorted(a); return {p: a[int(p / 100 * (len(a) - 1))] for p in ps}

stats = {}; qpool = {}
for suite in ['JB-all', 'JB-hard', 'REAL-agree', 'CF', 'CF-probe', 'LONG']:
    S, Qt, TOT = [], [], []
    for it in EK.load_suite(suite):
        for qn, qd in it['questions'].items():
            pr = P.prep(it['state'], qd)
            S.append(pr['q0']); Qt.append(len(pr['q'])); TOT.append(pr['L'])
            key = json.dumps(qd, sort_keys=True)
            if key not in qpool: qpool[key] = (suite, qn, qd, len(pr['q']))
    stats[suite] = dict(n=len(S), state=pct(S), question=pct(Qt), total=pct(TOT), frac_total_le256=sum(t <= 256 for t in TOT) / len(TOT),
                        frac_total_le400=sum(t <= 400 for t in TOT) / len(TOT))
    print(suite, json.dumps(stats[suite]), flush=True)
json.dump(stats, open(os.path.expanduser('~/work/j5/lenstats.json'), 'w'), indent=1)

def rec(name, qd):
    pr = P.prep('x', qd)
    return dict(name=name, ids=pr['q'], opt=pr['opt'], kind=pr['rq'].kind, n_slots=pr['rq'].n_slots, temp=P.temp_for(pr['rq'].kind))

B = {}
for src, bn in (('JB-all', 'jb'), ('REAL-agree', 'bk')):
    cands = [(v[3], v[1], v[2]) for v in qpool.values() if v[0] == src]
    med = stt.median([c[0] for c in cands])
    cands.sort(key=lambda c: abs(c[0] - med))
    B[bn + '1'] = [rec(cands[0][1], cands[0][2])]
    B[bn + '4'] = [rec(c[1], c[2]) for c in cands[:4]]
h2specs = json.load(open(os.path.expanduser('~/work/h2/qspecs.json'))) if os.path.exists(os.path.expanduser('~/work/h2/qspecs.json')) else []
for s in h2specs:
    if s['name'] == 'details_match': B['bk1'] = [rec('details_match', s['spec'])]; break
for bn, qs in B.items():
    print(bn, [(q['name'], len(q['ids']), len(q['opt'])) for q in qs], 'total', sum(len(q['ids']) for q in qs), flush=True)
torch.save(B, os.path.expanduser('~/work/j5/bundles.pt'))
