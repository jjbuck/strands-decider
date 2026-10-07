"""seed-aggregated comparison at 1k and 4k rows (laptop)."""
import sys, json, os
sys.path.insert(0, os.path.expanduser('~/decider2/j4/code'))
import numpy as np, score_j4 as S, evalkit as EK
P = os.path.expanduser('~/decider2/j4/preds/')
K = ['JB-all', 'JB-hard', 'REAL-label', 'CF acc', 'CF flip', 'CF-probe acc', 'CF-probe flip', 'SHUF both_right']
G = {'a@1k': ['a_r1000', 'a2_r1000'], 'b@1k': ['b_r1000', 'b2_r1000'], 'a@4k': ['a_r4000', 'a2_r4000', 'a3_r4000'], 'b@4k': ['b_r4000', 'b2_r4000', 'b3_r4000']}
out = {}
for g, names in G.items():
    vals = []
    for nm in names:
        if not os.path.exists(P + nm + '.json'): continue
        p = json.load(open(P + nm + '.json'))
        s = S.summary(p, have_long=False)
        d = {k: s[k] for k in K}
        for nmS in ('CF', 'CF-probe'):
            d[nmS + ' dmean'] = EK.score(nmS, p, baselines=False)['model']['dmean']
        vals.append(d)
    out[g] = {k: (float(np.mean([v[k] for v in vals])), float(min(v[k] for v in vals)), float(max(v[k] for v in vals)), len(vals)) for k in vals[0]}
    print(g, len(vals), 'seeds:', {k: f'{m:.3f} [{lo:.3f}-{hi:.3f}]' for k, (m, lo, hi, n) in out[g].items()})
json.dump(out, open(os.path.expanduser('~/decider2/j4/seeds.json'), 'w'), indent=1)
