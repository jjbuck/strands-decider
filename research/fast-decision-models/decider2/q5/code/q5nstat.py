"""B12 statistics from the cal passes (cpu): per layer, dead neurons at several tolerances, firing rates, concentration of decision saliency,
banking vs retail agreement of the removable sets, per-question saliency overlap. -> ~/work/q5/res_nstat.json
python q5nstat.py"""
import os, sys, json
sys.path[:0] = [os.path.expanduser('~/work/q5')]
import torch
import q5lib as Q5

out = {}
NB = Q5.neur_stats('bank'); NR = Q5.neur_stats('ret'); NG = Q5.neur_stats('gen')
for nm, N in (('bank', NB), ('ret', NR), ('gen', NG)):
    L = []
    for i in range(24):
        mx = N['max'][i]; s = N['S2m'][i]; v = N['var'][i]
        tot = float(s.sum()); ss = torch.sort(s, descending=True).values; c = torch.cumsum(ss, 0) / max(tot, 1e-300)
        r90 = int(torch.searchsorted(c, torch.tensor(0.9, dtype=c.dtype))) + 1; r99 = int(torch.searchsorted(c, torch.tensor(0.99, dtype=c.dtype))) + 1
        r999 = int(torch.searchsorted(c, torch.tensor(0.999, dtype=c.dtype))) + 1
        L.append({'layer': i, 'dead_1e-3': int((mx < 1e-3).sum()), 'dead_1e-2': int((mx < 1e-2).sum()), 'dead_3e-2': int((mx < 3e-2).sum()),
                  'dead_1e-1': int((mx < 1e-1).sum()), 'fire1e-2_lt_1pct': int((N['fire'][i] < 0.01).sum()), 'fire1e-1_lt_1pct': int((N['fire1'][i] < 0.01).sum()),
                  'sal_total': tot, 'sal_r90': r90, 'sal_r99': r99, 'sal_r999': r999, 'absmean_median': float(N['absmean'][i].median()),
                  'max_median': float(mx.median()), 'max_min': float(mx.min()), 'var_total': float(v.sum())})
    out[nm] = L
# layer totals of saliency (where the decision lives in the MLPs)
out['sal_layer_share_gen'] = [float(NG['S2m'][i].sum()) for i in range(24)]
t = sum(out['sal_layer_share_gen']); out['sal_layer_share_gen'] = [x / t for x in out['sal_layer_share_gen']]
# banking vs retail: overlap of the 25% / 50% least-salient neuron sets per layer, and Spearman of saliency
ov = []
for i in range(24):
    rb = torch.argsort(torch.argsort(NB['S2m'][i])).double(); rr = torch.argsort(torch.argsort(NR['S2m'][i])).double()
    sp = float(torch.corrcoef(torch.stack([rb, rr]))[0, 1])
    o = {}
    for f in (0.25, 0.5):
        n = int(f * 6144); a = set(torch.argsort(NB['S2m'][i])[:n].tolist()); b = set(torch.argsort(NR['S2m'][i])[:n].tolist()); o[str(f)] = len(a & b) / n
    # saliency mass that banking's removable 50% carries under retail traffic (and vice versa)
    nb50 = torch.argsort(NB['S2m'][i])[:3072]; nr50 = torch.argsort(NR['S2m'][i])[:3072]
    ov.append(dict(layer=i, spearman=sp, overlap=o, ret_mass_of_bank_low50=float(NR['S2m'][i][nb50].sum() / NR['S2m'][i].sum()),
                   bank_mass_of_bank_low50=float(NB['S2m'][i][nb50].sum() / NB['S2m'][i].sum()),
                   bank_mass_of_ret_low50=float(NB['S2m'][i][nr50].sum() / NB['S2m'][i].sum())))
out['bank_vs_ret'] = ov
# per question (banking cal pass): how many neurons per layer hold 99% of the zeroing saliency for one question vs all questions
pq = torch.load(f'{Q5.CAL}/bank/neurons.pt', map_location='cpu')['perq']
allA = sum(v['A2'] for v in pq.values())
res_q = {}
for q, v in sorted(pq.items(), key=lambda kv: -kv[1]['n']):
    if v['n'] < 5: continue
    r = []
    for i in (2, 6, 10, 14, 18, 22):
        def r99(x):
            ss = torch.sort(x, descending=True).values; c = torch.cumsum(ss, 0) / max(float(x.sum()), 1e-300)
            return int(torch.searchsorted(c, torch.tensor(0.99, dtype=c.dtype))) + 1
        top = set(torch.argsort(v['A2'][i], descending=True)[:r99(v['A2'][i])].tolist()); topall = set(torch.argsort(allA[i], descending=True)[:r99(allA[i])].tolist())
        r.append(dict(layer=i, r99_q=len(top), r99_all=len(topall), frac_in_all=len(top & topall) / max(len(top), 1)))
    res_q[q] = dict(n=v['n'], layers=r)
out['per_question'] = res_q
json.dump(out, open(f'{Q5.W5}/res_nstat.json', 'w'), indent=1)
for nm in ('bank', 'ret'):
    print(nm, [(d['layer'], d['dead_1e-2'], d['dead_1e-1'], d['sal_r90'], d['sal_r99']) for d in out[nm]])
print('layer share', [round(x, 3) for x in out['sal_layer_share_gen']])
print('bank vs ret', [(d['layer'], round(d['spearman'], 2), d['overlap']) for d in ov])
