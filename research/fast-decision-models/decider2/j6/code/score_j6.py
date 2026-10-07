"""Laptop-side scoring (pure python + numpy): every evalkit suite + Brier/ECE + paired McNemar vs hobson (or vs another arm).
python3 score_j6.py NAME=preds.json[:key] ... [--vs NAME] -> prints a table, writes j6/scores.json"""
import sys, json, math, collections, os
sys.path.insert(0, '~/decider2/evalkit')
import numpy as np
import evalkit as EK

SUITES = ['JB-all', 'JB-hard', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']


def load(spec):
    nm, path = spec.split('=', 1)
    key = None
    if ':' in path: path, key = path.split(':', 1)
    d = json.load(open(path))
    if key: d = d[key]
    return nm, d


def mcnemar(a, b):
    """a, b: lists of bools (paired). exact two-sided binomial on discordant pairs -> (n01, n10, p)"""
    n01 = sum(1 for x, y in zip(a, b) if not x and y); n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0: return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return n01, n10, min(1.0, p)


def gt_rows(suite, preds):
    """(item, q, pred_dist, expected) for suites with ground truth"""
    out = []
    if suite in ('CF', 'CF-probe'):
        for pr in EK._pairs(suite):
            for x, e in ((pr['a'], pr['ea']), (pr['b'], pr['eb'])):
                p = preds.get(x, {}).get(pr['q'])
                out.append((x, pr['q'], EK._norm(p) if p else None, e))
        return out
    for it in EK.load_suite(suite):
        for q in it['questions']:
            e = (it.get('expected') or {}).get(q)
            if e is None: continue
            p = preds.get(it['id'], {}).get(q)
            out.append((it['id'], q, EK._norm(p) if p else None, e))
    return out


def brier_ece(rows):
    rows = [r for r in rows if r[2] is not None]
    if not rows: return float('nan'), float('nan')
    br = []; conf = []; corr = []
    for _, _, p, e in rows:
        br.append(sum((v - (1.0 if k == e else 0.0)) ** 2 for k, v in p.items()) + (0.0 if e in p else 1.0))
        a = max(p, key=p.get); conf.append(p[a]); corr.append(a == e)
    conf = np.array(conf); corr = np.array(corr, float); ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any(): ece += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(np.mean(br)), float(ece)


def correct_vec(suite, preds, ref_preds):
    """paired correctness against ground truth on rows covered by both"""
    a = {(i, q): (max(p, key=p.get) == e) for i, q, p, e in gt_rows(suite, preds) if p}
    b = {(i, q): (max(p, key=p.get) == e) for i, q, p, e in gt_rows(suite, ref_preds) if p}
    ks = sorted(set(a) & set(b))
    return [a[k] for k in ks], [b[k] for k in ks]


def agree_vec(suite, preds, other, sd_only=True):
    """paired agreement-with-hobson vectors for two prediction sets (state-dependent questions only)"""
    rows = EK._qrows(suite, preds); rows2 = {(r['id'], r['q']): r for r in EK._qrows(suite, other)}
    a, b = [], []
    for r in rows:
        r2 = rows2[(r['id'], r['q'])]
        if r['pred'] is None or r2['pred'] is None or r['hob'] is None: continue
        if sd_only and (r['nost'] is None or EK._arg(r['nost']) == EK._arg(r['hob'])): continue
        a.append(EK._arg(r['pred']) == EK._arg(r['hob'])); b.append(EK._arg(r2['pred']) == EK._arg(r['hob']))
    return a, b


def main():
    args = [x for x in sys.argv[1:] if not x.startswith('--')]
    vs = None
    if '--vs' in sys.argv: vs = sys.argv[sys.argv.index('--vs') + 1]; args = [x for x in args if x != vs]
    models = dict(load(x) for x in args)
    hob = {}
    for s in SUITES: hob.update(EK._as_preds(s, 'hobson'))
    inc = set(x for x in os.environ.get('ONLY_Q', '').split(',') if x); exc = set(x for x in os.environ.get('EXCL_Q', '').split(',') if x)
    if inc or exc:
        def filt(p):
            return {i: {q: v for q, v in d.items() if (not inc or q in inc) and q not in exc} for i, d in p.items()}
        models = {k: filt(v) for k, v in models.items()}; hob = filt(hob)
        # hobson restricted to the same (item, question) keys the first model covers
        first = next(iter(models.values()))
        hob = {i: {q: v for q, v in d.items() if q in first.get(i, {})} for i, d in hob.items()}
    res = {}
    for nm, pr in list(models.items()) + [('hobson', hob)]:
        r = {}
        for s in SUITES:
            m = EK.score(s, pr, baselines=False)['model']
            if s not in ('SHUF',):
                b, e = brier_ece(gt_rows(s, pr)) if s in ('JB-all', 'JB-hard', 'CF', 'CF-probe', 'REAL-label') else (float('nan'), float('nan'))
                m['brier'] = b; m['ece'] = e
            if nm != 'hobson' and s in ('JB-all', 'JB-hard', 'REAL-label', 'CF', 'CF-probe'):
                x, y = correct_vec(s, pr, hob); m['mcnemar_vs_hobson'] = mcnemar(y, x)    # (hobson wrong & model right, hobson right & model wrong, p)
            r[s] = m
        res[nm] = r
    if vs:
        for nm, pr in models.items():
            if nm == vs: continue
            for s in ('REAL-agree', 'LONG'):
                x, y = agree_vec(s, pr, models[vs]); res[nm][s][f'mcnemar_sd_vs_{vs}'] = mcnemar(y, x)
            for s in ('CF', 'CF-probe', 'REAL-label', 'JB-hard'):
                if s in res[nm]:
                    x, y = correct_vec(s, pr, models[vs]); res[nm][s][f'mcnemar_acc_vs_{vs}'] = mcnemar(y, x)
    cols = [('JB-all', 'acc'), ('JB-hard', 'acc'), ('JB-hard', 'brier'), ('REAL-agree', 'agree'), ('REAL-agree', 'agree_sd'), ('REAL-agree', 'tv'),
            ('LONG', 'agree'), ('LONG', 'agree_sd'), ('REAL-label', 'acc'), ('REAL-label', 'brier'), ('CF', 'acc'), ('CF', 'flip'), ('CF', 'flip_given_hobson'), ('CF', 'dir'),
            ('CF-probe', 'acc'), ('CF-probe', 'flip'), ('CF-probe', 'flip_given_hobson'), ('SHUF', 'both_right'), ('CF', 'brier')]
    print('| model | ' + ' | '.join(f'{s} {k}' for s, k in cols) + ' |')
    print('|' + '---|' * (len(cols) + 1))
    for nm, r in res.items():
        vals = []
        for s, k in cols:
            v = r.get(s, {}).get(k)
            vals.append('-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.3f}')
        print(f'| {nm} | ' + ' | '.join(vals) + ' |')
    for nm, r in res.items():
        if nm == 'hobson': continue
        print(nm, {s: r[s].get('mcnemar_vs_hobson') for s in r if 'mcnemar_vs_hobson' in r[s]},
              {f'{s}:{k}': v for s in r for k, v in r[s].items() if k.startswith('mcnemar_sd_vs') or k.startswith('mcnemar_acc_vs')},
              'coverage', {s: round(r[s].get('coverage', r[s].get('n_pairs', 0)), 3) for s in r})
    out = os.environ.get('SCORES', '~/decider2/j6/scores.json')
    old = json.load(open(out)) if os.path.exists(out) else {}
    old.update({k: v for k, v in res.items() if k != 'hobson' or 'hobson' not in old}); json.dump(old, open(out, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
