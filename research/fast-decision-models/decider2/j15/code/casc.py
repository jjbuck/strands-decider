"""Offline cascade analysis on preds jsonl (pure numpy; no models). Draft/verifier preds: rows {suite,id,q,T,probs}."""
import json, sys, math, collections
import numpy as np
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK

def load(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        out.setdefault(r['id'], {})[r['q']] = r['probs']
    return out

def margin(d):
    v = sorted(d.values(), reverse=True)
    s = sum(v)
    return (v[0] - (v[1] if len(v) > 1 else 0)) / s

def hobson_all():
    H = {}
    for s in ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
        for k, v in EK._as_preds(s, 'hobson').items():
            H.setdefault(k, {}).update(v)
    return H

def suite_keys():
    S = collections.defaultdict(list)
    for s in ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
        for it in EK.load_suite(s):
            for q in it['questions']:
                S[s].append((it['id'], q))
    return S

def cascade(D, V, tau, sig=None):
    """per (id,q): use V if signal < tau else D. sig: dict id->q->value (default draft margin)."""
    out = {}; nd = 0; n = 0
    for iid, qs in D.items():
        for q, d in qs.items():
            s = sig[iid][q] if sig is not None else margin(d)
            n += 1
            if s < tau and iid in V and q in V[iid]:
                out.setdefault(iid, {})[q] = V[iid][q]; nd += 1
            else:
                out.setdefault(iid, {})[q] = d
    return out, nd / max(n, 1)

def summary(P, H=None):
    r = {}
    sc = EK.score('REAL-agree', P, baselines=False)['model']; r['real_agree'] = sc['agree']; r['real_sd'] = sc['agree_sd']; r['real_tv'] = sc['tv']
    sc = EK.score('LONG', P, baselines=False)['model']; r['long_agree'] = sc['agree']; r['long_sd'] = sc['agree_sd']
    sc = EK.score('CF', P, baselines=False)['model']; r['cf_fgh'] = sc['flip_given_hobson']; r['cf_flip'] = sc['flip']
    sc = EK.score('CF-probe', P, baselines=False)['model']; r['cfp_fgh'] = sc['flip_given_hobson']; r['cfp_flip'] = sc['flip']
    sc = EK.score('JB-hard', P, baselines=False)['model']; r['jbh'] = sc['acc']
    sc = EK.score('JB-all', P, baselines=False)['model']; r['jba'] = sc['acc']
    sc = EK.score('REAL-label', P, baselines=False)['model']; r['rlab'] = sc['acc']
    return r

if __name__ == '__main__':
    H = hobson_all()
    S = suite_keys()
    V = load(sys.argv[1])
    for dp in sys.argv[2:]:
        D = load(dp)
        # margin distributions: draft flips vs verifier decision and vs hobson
        rows = []
        for iid, qs in D.items():
            for q, d in qs.items():
                if iid not in V or q not in V[iid] or iid not in H or q not in H[iid]: continue
                a = EK._arg(d); v = EK._arg(V[iid][q]); h = EK._arg(EK._norm(H[iid][q]))
                rows.append((margin(d), a != v, a != h, v != h, margin(EK._norm(H[iid][q]))))
        m = np.array([r[0] for r in rows]); fv = np.array([r[1] for r in rows]); fh = np.array([r[2] for r in rows]); vh = np.array([r[3] for r in rows]); hm = np.array([r[4] for r in rows])
        print(f'\n### draft {dp.split("/")[-1]}  n={len(rows)} flips vs V {fv.mean():.4f} vs hobson {fh.mean():.4f} (V vs hobson {vh.mean():.4f})')
        for qq in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
            print(f'  flipped(vs V) margin quantile {qq}: {np.quantile(m[fv], qq):.3f}   nonflipped quantile: {np.quantile(m[~fv], qq):.3f}')
        print('  tau   f      catch(vsV)  resid_flip_vsV  resid_vs_hob(est)')
        for tau in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            dfr = m < tau
            resid = (fv & ~dfr).mean()
            # final decision = V if deferred else draft -> flips vs hobson
            fin_vs_h = np.where(dfr, vh, fh).mean()
            print(f'  {tau:.2f}  {dfr.mean():.3f}  {(fv & dfr).sum() / max(fv.sum(),1):.3f}       {resid:.4f}         {fin_vs_h:.4f}')
