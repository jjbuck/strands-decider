"""J9 final scoring (laptop, pure python): absolute accuracy on every evalkit suite, paired McNemar against hobson, Brier and ECE,
agreement metrics. python3 final_score.py NAME=preds.json [NAME2=preds2.json ...]  -> prints a table, writes final_scores.json"""
import sys, os, json, math
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import numpy as np
import evalkit as EK


def load(f):
    if f.endswith('.jsonl'):
        p = {}
        for l in open(f):
            try: r = json.loads(l); p[r['id']] = r['p']
            except Exception: pass
        return p
    return json.load(open(f))


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def brier(p, y):
    return sum((p.get(k, 0.0) - (1.0 if k == y else 0.0)) ** 2 for k in set(p) | {y})


def ece(rows):
    bins = np.linspace(0, 1, 11); conf = np.array([r[0] for r in rows]); ok = np.array([r[1] for r in rows], float)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def labeled(name, preds):
    rows = [r for r in EK._qrows(name, preds) if r['exp'] is not None and r['pred'] is not None and r['hob'] is not None]
    if not rows: return None
    a = [EK._arg(r['pred']) == r['exp'] for r in rows]; h = [EK._arg(r['hob']) == r['exp'] for r in rows]
    b = sum(1 for x, y in zip(a, h) if x and not y); c = sum(1 for x, y in zip(a, h) if y and not x)
    return dict(n=len(rows), acc=float(np.mean(a)), hob=float(np.mean(h)), b=b, c=c, p=mcnemar(b, c),
                brier=float(np.mean([brier(r['pred'], r['exp']) for r in rows])), brier_hob=float(np.mean([brier(r['hob'], r['exp']) for r in rows])),
                ece=ece([(max(r['pred'].values()), EK._arg(r['pred']) == r['exp']) for r in rows]),
                ece_hob=ece([(max(r['hob'].values()), EK._arg(r['hob']) == r['exp']) for r in rows]))


def pairs(name, preds):
    hp = EK._as_preds(name, 'hobson'); out = []; ok_m = []; ok_h = []
    for pr in EK._pairs(name):
        q = pr['q']
        try:
            pa, pb = EK._norm(preds[pr['a']][q]), EK._norm(preds[pr['b']][q]); ha, hb = EK._norm(hp[pr['a']][q]), EK._norm(hp[pr['b']][q])
        except KeyError: continue
        ok_m.append(EK._arg(pa) == pr['ea'] and EK._arg(pb) == pr['eb']); ok_h.append(EK._arg(ha) == pr['ea'] and EK._arg(hb) == pr['eb'])
    if not ok_m: return None
    b = sum(1 for x, y in zip(ok_m, ok_h) if x and not y); c = sum(1 for x, y in zip(ok_m, ok_h) if y and not x)
    tr = [x for x, y in zip(ok_m, ok_h) if y]
    return dict(n=len(ok_m), pair_acc=float(np.mean(ok_m)), hob=float(np.mean(ok_h)), b=b, c=c, p=mcnemar(b, c), fgh=float(np.mean(tr)) if tr else float('nan'))


def agree(name, preds):
    m = EK._agree_metrics([r for r in EK._qrows(name, preds)])
    return dict(n=m.get('n_q'), coverage=m.get('coverage'), agree=m.get('agree'), agree_sd=m.get('agree_sd'), n_sd=m.get('n_sd'), tv=m.get('tv'))


def score_all(preds):
    r = {}
    for s in ('JB-all', 'JB-hard', 'REAL-label', 'CF', 'CF-probe'):
        x = labeled(s, preds)
        if x: r[s] = x
    for s in ('CF', 'CF-probe'):
        x = pairs(s, preds)
        if x: r[s + ' pairs'] = x
    for s in ('REAL-agree', 'LONG', 'JB-hard'):
        r[s + ' agree'] = agree(s, preds)
    sh = EK._shuf_metrics(preds)
    r['SHUF'] = sh
    return r


if __name__ == '__main__':
    allr = {}
    for a in sys.argv[1:]:
        nm, f = a.split('=', 1)
        r = score_all(load(f)); allr[nm] = r
        print('==', nm)
        for k, v in r.items():
            if v is None: continue
            print(f'  {k:<18}', ' '.join(f'{kk}={vv:.3f}' if isinstance(vv, float) else f'{kk}={vv}' for kk, vv in v.items()))
    json.dump(allr, open(os.path.expanduser('~/decider2/j9/final_scores.json'), 'w'), indent=1)
