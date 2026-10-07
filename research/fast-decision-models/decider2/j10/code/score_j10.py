"""Score J10 eval outputs with evalkit (laptop, pure python).  python3 score_j10.py TAG [TAG...] -> prints table, writes ../scores_j10.json
Adds: paired McNemar (exact) vs hobson on JB-hard / JB-all / REAL-label / CF items / CF-probe items; multi-class Brier and 10-bin ECE on every
ground-truth suite; CF / CF-probe flip by kind."""
import sys, os, json, math, collections
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
RUN = ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']


def load(tag):
    preds = {}
    for su in RUN:
        f = f'{R}/{su}.{tag}.jsonl'
        if not os.path.exists(f): continue
        for l in open(f):
            r = json.loads(l); preds.setdefault(r['id'], {}).update(r['q'])
    return preds


def _arg(p): return max(p, key=p.get)


def gt_rows(name, preds):
    """(pred dist, hobson dist, expected) per ground-truth question"""
    rows = []
    if name in ('CF', 'CF-probe'):
        hp = EK._as_preds(name, 'hobson')
        for pr in EK._pairs(name):
            for x, e in ((pr['a'], pr['ea']), (pr['b'], pr['eb'])):
                q = pr['q']
                if x in preds and q in preds[x] and x in hp and q in hp[x]:
                    rows.append((EK._norm(preds[x][q]), EK._norm(hp[x][q]), e))
        return rows
    for r in EK._qrows(name, preds):
        if r['exp'] is not None and r['pred'] is not None and r['hob'] is not None:
            rows.append((r['pred'], r['hob'], r['exp']))
    return rows


def mcnemar(rows):
    b = sum(1 for p, h, e in rows if _arg(p) == e and _arg(h) != e)
    c = sum(1 for p, h, e in rows if _arg(p) != e and _arg(h) == e)
    n = b + c; k = min(b, c)
    pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return b, c, pv


def brier(rows, which=0):
    out = []
    for r in rows:
        p = r[which]; e = r[2]
        out.append(sum((v - (1.0 if k == e else 0.0)) ** 2 for k, v in p.items()) + (0 if e in p else 1.0))
    return float(np.mean(out)) if out else float('nan')


def ece(rows, which=0, bins=10):
    conf = np.array([max(r[which].values()) for r in rows]); ok = np.array([_arg(r[which]) == r[2] for r in rows], dtype=float)
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def score_tag(tag):
    p = load(tag)
    if not p: return None
    out = {'coverage_items': len(p)}
    for su in ['JB-hard', 'JB-all', 'JB-long', 'REAL-agree', 'REAL-label', 'LONG', 'CF', 'CF-probe', 'SHUF']:
        m = EK.score(su, p, baselines=False)
        out[su] = dict(model=m['model'], hobson=m.get('hobson', {}))
        if su in ('JB-hard', 'JB-all', 'REAL-label', 'CF', 'CF-probe'):
            rows = gt_rows(su, p)
            b, c, pv = mcnemar(rows)
            out[su]['paired'] = dict(n=len(rows), model_only=b, hobson_only=c, mcnemar_p=pv, acc_model=float(np.mean([_arg(r[0]) == r[2] for r in rows])) if rows else None,
                                     acc_hobson=float(np.mean([_arg(r[1]) == r[2] for r in rows])) if rows else None,
                                     brier_model=brier(rows, 0), brier_hobson=brier(rows, 1), ece_model=ece(rows, 0) if rows else None, ece_hobson=ece(rows, 1) if rows else None)
        if su in ('CF', 'CF-probe'):
            out[su]['by_kind'] = EK.breakdown(su, p, 'kind', rows=('hobson',))
    return out


def fmt(x, d=3):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


if __name__ == '__main__':
    tags = sys.argv[1:]
    allres = {}
    path = os.path.join(R, '..', 'scores_j10.json')
    if os.path.exists(path): allres = json.load(open(path))
    for t in tags:
        r = score_tag(t)
        if r: allres[t] = r
    json.dump(allres, open(path, 'w'), indent=1)
    hdr = ['tag', 'JBh acc', 'JBh p', 'JBall acc', 'JBall p', 'REALlab', 'RL p', 'REAL agr', 'REAL sd', 'LONG agr', 'LONG sd', 'CF acc', 'CF flip', 'CF fgh',
           'CFp acc', 'CFp flip', 'CFp fgh', 'SHUF ch', 'Brier JB', 'Brier RL', 'ECE RL']
    print(' | '.join(hdr))
    for t in tags:
        r = allres.get(t)
        if not r: continue
        g = lambda su, k: r.get(su, {}).get('model', {}).get(k)
        pp = lambda su, k: r.get(su, {}).get('paired', {}).get(k)
        row = [t, fmt(g('JB-hard', 'acc')), fmt(pp('JB-hard', 'mcnemar_p'), 2), fmt(g('JB-all', 'acc')), fmt(pp('JB-all', 'mcnemar_p'), 2), fmt(g('REAL-label', 'acc')),
               fmt(pp('REAL-label', 'mcnemar_p'), 3), fmt(g('REAL-agree', 'agree')), fmt(g('REAL-agree', 'agree_sd')), fmt(g('LONG', 'agree')), fmt(g('LONG', 'agree_sd')),
               fmt(g('CF', 'acc')), fmt(g('CF', 'flip')), fmt(g('CF', 'flip_given_hobson')), fmt(g('CF-probe', 'acc')), fmt(g('CF-probe', 'flip')),
               fmt(g('CF-probe', 'flip_given_hobson')), fmt(g('SHUF', 'change')), fmt(pp('JB-all', 'brier_model')), fmt(pp('REAL-label', 'brier_model')), fmt(pp('REAL-label', 'ece_model'))]
        print(' | '.join(row))
    if tags and allres.get(tags[0]):
        r = allres[tags[0]]
        print('hobson ref: JBh', fmt(r['JB-hard']['hobson'].get('acc')), 'JBall', fmt(r['JB-all']['hobson'].get('acc')), 'REAL-label', fmt(r['REAL-label']['hobson'].get('acc')),
              'CF acc/flip', fmt(r['CF']['hobson'].get('acc')), fmt(r['CF']['hobson'].get('flip')), 'CFp acc/flip', fmt(r['CF-probe']['hobson'].get('acc')), fmt(r['CF-probe']['hobson'].get('flip')),
              'SHUF', fmt(r['SHUF']['hobson'].get('change')), 'Brier JB/RL', fmt(r['JB-all']['paired']['brier_hobson']), fmt(r['REAL-label']['paired']['brier_hobson']),
              'ECE RL', fmt(r['REAL-label']['paired']['ece_hobson']))
