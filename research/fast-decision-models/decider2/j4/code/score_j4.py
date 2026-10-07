"""J4 scorer (laptop, pure python + numpy): absolute accuracy on every evalkit suite, McNemar vs hobson and between arms, Brier / ECE,
CF / CF-probe by kind.  python score_j4.py preds/a_r12000.json preds/b_r12000.json ... [--vs a_r12000:b_r12000] [--json out.json]"""
import os, sys, json, math, argparse, collections
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import numpy as np
import evalkit as EK


def binom_two_sided(k, n):
    if n == 0: return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def mcnemar(x, y):
    """x, y: lists of bools (paired). -> (x_only, y_only, p)"""
    b = sum(1 for u, v in zip(x, y) if u and not v); c = sum(1 for u, v in zip(x, y) if v and not u)
    return b, c, binom_two_sided(min(b, c), b + c)


def correct_vec(name, preds, ids_q=None):
    rows = EK._qrows(name, preds)
    out = {}
    for r in rows:
        if r['exp'] is None or r['pred'] is None: continue
        out[(r['id'], r['q'])] = EK._arg(r['pred']) == r['exp']
    return out


def brier_ece(name, preds):
    rows = [r for r in EK._qrows(name, preds) if r['exp'] is not None and r['pred'] is not None]
    br = []; conf = []; ok = []
    for r in rows:
        p = r['pred']; br.append(sum((v - (1.0 if k == r['exp'] else 0.0)) ** 2 for k, v in p.items()))
        a = EK._arg(p); conf.append(p[a]); ok.append(a == r['exp'])
    conf = np.array(conf); ok = np.array(ok, dtype=float)
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any(): ece += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(np.mean(br)), float(ece)


def pair_vec(name, preds):
    out = {}
    for pr in EK._pairs(name):
        q = pr['q']
        try:
            pa = EK._norm(preds[pr['a']][q]); pb = EK._norm(preds[pr['b']][q])
        except KeyError: continue
        out[(pr['a'], pr['b'])] = EK._arg(pa) == pr['ea'] and EK._arg(pb) == pr['eb']
    return out


def kinds(name, preds):
    res = collections.defaultdict(list)
    for pr in EK._pairs(name):
        q = pr['q']
        try:
            pa = EK._norm(preds[pr['a']][q]); pb = EK._norm(preds[pr['b']][q])
        except KeyError: continue
        k = pr.get('kind', '?')
        if k.startswith('id_') and name == 'CF': k = 'identity'
        res[k].append(EK._arg(pa) == pr['ea'] and EK._arg(pb) == pr['eb'])
    return {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(res.items())}


def summary(preds, have_long=True):
    s = {}
    jb = EK.score('JB-all', preds, baselines=False); jh = EK.score('JB-hard', preds, baselines=False)
    s['JB-all'] = jb['model'].get('acc'); s['JB-hard'] = jh['model'].get('acc')
    hp_jb = EK._as_preds('JB-all', 'hobson')
    for nm in ('JB-all', 'JB-hard'):
        x = correct_vec(nm, preds); y = correct_vec(nm, hp_jb)
        ks = [k for k in x if k in y]
        b, c, p = mcnemar([x[k] for k in ks], [y[k] for k in ks]); s[nm + ' mcnemar(model_only,hob_only,p)'] = (b, c, round(p, 4))
    s['JB-all brier,ece'] = tuple(round(v, 4) for v in brier_ece('JB-all', preds))
    ra = EK.score('REAL-agree', preds, baselines=False)['model']
    s['REAL agree'] = ra.get('agree'); s['REAL agree_sd'] = ra.get('agree_sd')
    rl = EK.score('REAL-label', preds, baselines=False)['model']
    s['REAL-label'] = rl.get('acc'); s['REAL-label consensus'] = rl.get('acc_consensus')
    s['REAL-label brier,ece'] = tuple(round(v, 4) for v in brier_ece('REAL-label', preds))
    hp_rl = EK._as_preds('REAL-label', 'hobson')
    x = correct_vec('REAL-label', preds); y = correct_vec('REAL-label', {k: v for k, v in EK._as_preds('REAL-agree', 'hobson').items()})
    ks = [k for k in x if k in y]; b, c, p = mcnemar([x[k] for k in ks], [y[k] for k in ks]); s['REAL-label mcnemar'] = (b, c, round(p, 4))
    if have_long:
        lo = EK.score('LONG', preds, baselines=False)['model']
        if lo.get('coverage', 0) > 0.9: s['LONG agree'] = lo.get('agree'); s['LONG agree_sd'] = lo.get('agree_sd')
    for nm in ('CF', 'CF-probe'):
        m = EK.score(nm, preds, baselines=False)['model']
        s[nm + ' acc'] = m.get('acc'); s[nm + ' flip'] = m.get('flip'); s[nm + ' dir'] = m.get('dir'); s[nm + ' fgh'] = m.get('flip_given_hobson')
        hp = EK._as_preds(nm, 'hobson'); x = pair_vec(nm, preds); y = pair_vec(nm, hp); ks = [k for k in x if k in y]
        b, c, p = mcnemar([x[k] for k in ks], [y[k] for k in ks]); s[nm + ' flip mcnemar'] = (b, c, round(p, 4))
        s[nm + ' kinds'] = kinds(nm, preds)
    sh = EK.score('SHUF', preds, baselines=False)['model']; s['SHUF both_right'] = sh.get('both_right')
    return s


def compare(pa, pb):
    out = {}
    for nm in ('JB-all', 'JB-hard', 'REAL-label'):
        x = correct_vec(nm, pa); y = correct_vec(nm, pb); ks = [k for k in x if k in y]
        out[nm] = mcnemar([x[k] for k in ks], [y[k] for k in ks])
    for nm in ('CF', 'CF-probe'):
        x = pair_vec(nm, pa); y = pair_vec(nm, pb); ks = [k for k in x if k in y]
        out[nm + ' pairs'] = mcnemar([x[k] for k in ks], [y[k] for k in ks])
        x = correct_vec(nm, pa); y = correct_vec(nm, pb); ks = [k for k in x if k in y]
        out[nm + ' items'] = mcnemar([x[k] for k in ks], [y[k] for k in ks])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('preds', nargs='*'); ap.add_argument('--vs', action='append', default=[]); ap.add_argument('--json', default='')
    ap.add_argument('--hobson', type=int, default=1)
    a = ap.parse_args()
    allres = {}
    P = {}
    if a.hobson:
        hp = {}
        for nm in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe'):
            hp.update(EK._as_preds(nm, 'hobson'))
        P['hobson'] = hp
    for f in a.preds:
        P[os.path.basename(f).replace('.json', '')] = json.load(open(f))
    for k, p in P.items():
        allres[k] = summary(p)
        print('==', k)
        for kk, v in allres[k].items():
            print(f'  {kk}: {round(v, 4) if isinstance(v, float) else v}')
    for v in a.vs:
        x, y = v.split(':')
        c = compare(P[x], P[y]); allres[f'{x}_vs_{y}'] = c
        print('== paired', x, '(only, ) vs', y, {k: (b, cc, round(p, 4)) for k, (b, cc, p) in c.items()})
    if a.json: json.dump(allres, open(a.json, 'w'), indent=1, default=str)
