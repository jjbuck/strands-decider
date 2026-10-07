"""J15 cascade analysis (laptop, pure numpy). tau fitted on train-split DEV only; applied unchanged to the eval suites.
cascade decision per question = verifier (b8) if signal < tau else draft."""
import json, sys, math, collections, os
import numpy as np
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK


def _binom_p(k, n):
    """exact two-sided binomial test p, p0 = 0.5"""
    if n == 0: return 1.0
    from math import comb
    pk = [comb(n, i) / 2 ** n for i in range(n + 1)]
    obs = pk[k]
    return min(1.0, sum(p for p in pk if p <= obs * (1 + 1e-9)))

PD = '~/decider2/j15/preds'
SU = ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']


def load(p):
    out = {}
    if not os.path.exists(p): return out
    for l in open(p):
        r = json.loads(l)
        out.setdefault(r['id'], {})[r['q']] = r['probs']
    return out


def meta(p):
    out = {}
    for l in open(p):
        r = json.loads(l); out[(r['id'], r['q'])] = r
    return out


def am(d): return max(d, key=d.get)


def margin(d):
    v = sorted(d.values(), reverse=True); s = sum(v)
    return (v[0] - (v[1] if len(v) > 1 else 0)) / s


def keys_of(P): return [(i, q) for i, qs in P.items() for q in qs]


def cascade(D, V, sig, tau):
    out = {}; nd = 0; n = 0
    for i, qs in D.items():
        for q, d in qs.items():
            n += 1
            if sig(i, q, d) < tau and i in V and q in V[i]:
                out.setdefault(i, {})[q] = V[i][q]; nd += 1
            else: out.setdefault(i, {})[q] = d
    return out, nd / max(n, 1)


_H = None
def hob():
    global _H
    if _H is None:
        _H = {}
        for s in SU:
            for k, v in EK._as_preds(s, 'hobson').items(): _H.setdefault(k, {}).update(v)
    return _H


def mcnemar(a, b):
    """a, b: lists of bools (correct/agree per item). returns (a_only, b_only, p)"""
    ao = sum(1 for x, y in zip(a, b) if x and not y); bo = sum(1 for x, y in zip(a, b) if y and not x)
    p = _binom_p(ao, ao + bo)
    return ao, bo, p


def suite_rows(s, P):
    rows = []
    H = hob()
    for it in EK.load_suite(s):
        for q in it['questions']:
            if it['id'] in P and q in P[it['id']] and it['id'] in H and q in H[it['id']]:
                rows.append((it['id'], q, (it.get('expected') or {}).get(q)))
    return rows


def brier(P, s):
    v = []
    for it in EK.load_suite(s):
        for q in it['questions']:
            e = (it.get('expected') or {}).get(q)
            if e is None or it['id'] not in P or q not in P[it['id']]: continue
            d = EK._norm(P[it['id']][q])
            v.append(sum((p - (1.0 if k == e else 0.0)) ** 2 for k, p in d.items()))
    return float(np.mean(v)) if v else float('nan')


def full_score(P, ref=None, floor=None):
    """all suites; paired tests vs ref (e.g. b8) and floor (bf16 runtime) on agreement with hobson (REAL+LONG) and JB-hard correctness"""
    H = hob(); r = {}
    for s in ('REAL-agree', 'LONG'):
        rows = suite_rows(s, P)
        ag = [am(EK._norm(P[i][q])) == am(EK._norm(H[i][q])) for i, q, _ in rows]
        r[f'{s}_flips'] = 1 - float(np.mean(ag)); r[f'{s}_n'] = len(rows)
        sc = EK.score(s, P, baselines=False)['model']; r[f'{s}_sd'] = sc['agree_sd']; r[f'{s}_tv'] = sc['tv']
        for nm, Rf in (('ref', ref), ('floor', floor)):
            if Rf is None: continue
            agr = [am(EK._norm(Rf[i][q])) == am(EK._norm(H[i][q])) if i in Rf and q in Rf[i] else True for i, q, _ in rows]
            r[f'{s}_paired_{nm}'] = mcnemar(ag, agr)    # (model-only agrees, ref-only agrees, p)
    for s in ('CF', 'CF-probe'):
        sc = EK.score(s, P, baselines=False)['model']
        r[f'{s}_fgh'] = sc.get('flip_given_hobson'); r[f'{s}_acc'] = sc.get('flip'); r[f'{s}_dir'] = sc.get('dir')
    for s in ('JB-hard', 'JB-all'):
        rows = suite_rows(s, P)
        c = [am(EK._norm(P[i][q])) == e for i, q, e in rows]; ch = [am(EK._norm(H[i][q])) == e for i, q, e in rows]
        r[f'{s}_acc'] = float(np.mean(c)); r[f'{s}_mcn_hob'] = mcnemar(c, ch)
    sc = EK.score('REAL-label', P, baselines=False)['model']; r['REAL-label_acc'] = sc['acc']
    r['brier_JB-all'] = brier(P, 'JB-all'); r['brier_REAL-label'] = brier(P, 'REAL-label')
    return r


def fit_tau(Dd, Vd, sig, eps_list=(0.0, 0.001, 0.002, 0.003), grid=None):
    """DEV: residual disagreement with the verifier among accepted questions (as a share of all) <= eps. returns {eps: (tau, f, resid)}"""
    ks = [(i, q) for i, qs in Dd.items() for q in qs if i in Vd and q in Vd[i]]
    s = np.array([sig(i, q, Dd[i][q]) for i, q in ks]); flip = np.array([am(Dd[i][q]) != am(Vd[i][q]) for i, q in ks])
    grid = grid if grid is not None else np.unique(np.concatenate([[0.0], np.sort(s) + 1e-9]))
    out = {}
    for eps in eps_list:
        for t in grid:
            res = (flip & (s >= t)).mean()
            if res <= eps:
                out[eps] = (float(t), float((s < t).mean()), float(res)); break
    return out, float(flip.mean()), len(ks)


def curve(Dd, Vd, sig, taus):
    ks = [(i, q) for i, qs in Dd.items() for q in qs if i in Vd and q in Vd[i]]
    s = np.array([sig(i, q, Dd[i][q]) for i, q in ks]); flip = np.array([am(Dd[i][q]) != am(Vd[i][q]) for i, q in ks])
    return [(t, float((s < t).mean()), float((flip & (s >= t)).mean()), float((flip & (s < t)).sum() / max(flip.sum(), 1))) for t in taus]
