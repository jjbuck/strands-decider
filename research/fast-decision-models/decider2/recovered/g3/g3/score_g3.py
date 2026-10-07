"""Score F7 eval outputs with F0's evalkit (pure python). usage: python3 score.py TAG [TAG...]  (reads results/<SUITE>.<TAG>.jsonl)"""
import sys, os, json, glob, math
sys.path.insert(0, '/tmp/decider2/evalkit')
import evalkit as EK
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
F7R = '/tmp/decider2/f7/results'
RUN = ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']
COLS = [('JB-hard', 'acc'), ('JB-hard', 'agree'), ('JB-long', 'acc'), ('JB-long', 'agree'), ('JB-all', 'acc'),
        ('REAL-agree', 'agree'), ('REAL-agree', 'agree_sd'), ('REAL-agree', 'tv'), ('REAL-label', 'acc'),
        ('LONG', 'agree'), ('LONG', 'agree_sd'), ('CF', 'acc'), ('CF', 'flip'), ('CF', 'flip_rel'), ('CF-probe', 'acc'), ('CF-probe', 'flip'),
        ('CF-probe', 'flip_rel'), ('CF', 'flip_given_hobson'), ('CF-probe', 'flip_given_hobson'), ('CF-probe', 'fgh_distract'),
        ('CF', 'dir'), ('CF-probe', 'dir'), ('SHUF', 'change'), ('SHUF', 'both_right'), ('JB-hard', 'mcnemar_b'), ('JB-hard', 'mcnemar_c'), ('JB-hard', 'mcnemar_p')]
REFROWS = ['hobson', 'nostate', 'drop@7', 'rand10@7', 'rand25@7', 'rand50@7', 'qattn10@7', 'drop@3', 'rand10@3']


def load(tag):
    preds = {}
    for su in RUN:
        f = f'{R}/{su}.{tag}.jsonl'
        if not os.path.exists(f): f = f'{F7R}/{su}.{tag}.jsonl'
        if not os.path.exists(f): continue
        for l in open(f):
            r = json.loads(l); preds.setdefault(r['id'], {}).update(r['q'])
    return preds


def _arg(p): return max(p, key=p.get)


def fgh_sub(name, preds, pred_kind):
    hp = EK._as_preds(name, 'hobson'); tr = []
    for pr in EK._pairs(name):
        if not pred_kind(pr.get('kind', '')): continue
        q = pr['q']
        try:
            ho = _arg(EK._norm(hp[pr['a']][q])) == pr['ea'] and _arg(EK._norm(hp[pr['b']][q])) == pr['eb']
            po = _arg(EK._norm(preds[pr['a']][q])) == pr['ea'] and _arg(EK._norm(preds[pr['b']][q])) == pr['eb']
        except KeyError: continue
        if ho: tr.append(po)
    return (sum(tr) / len(tr)) if tr else None


def mcnemar(name, preds, ref='hobson'):
    rp = EK._as_preds(name, ref) if isinstance(ref, str) else ref
    b = c = 0
    for it in EK.load_suite(name):
        for q, e in (it.get('expected') or {}).items():
            if e is None: continue
            try: m = _arg(EK._norm(preds[it['id']][q])) == e; h = _arg(EK._norm(rp[it['id']][q])) == e
            except KeyError: continue
            b += int(m and not h); c += int(h and not m)
    n = b + c; k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return b, c, p


def extra(p):
    out = {'CF-probe': {'fgh_distract': fgh_sub('CF-probe', p, lambda k: k.endswith('_distract'))}}
    b, c, pv = mcnemar('JB-hard', p); out['JB-hard'] = dict(mcnemar_b=float(b), mcnemar_c=float(c), mcnemar_p=pv)
    return out


def table(tags, refrows=REFROWS, out=sys.stdout):
    res = {}
    for t in tags:
        p = load(t)
        if p:
            res[t] = {su: EK.score(su, p, baselines=False)['model'] for su in sorted({s for s, _ in COLS})}
            for su, d in extra(p).items(): res[t][su].update(d)
    full = {su: EK.score(su, {}, baselines=True) for su in sorted({s for s, _ in COLS})}
    for r in refrows:
        res[r] = {su: dict(full[su].get(r, {})) for su in full}
        rp = {}
        for su in ('CF-probe', 'JB-all'):
            rp.update(EK._as_preds(su, r))
        if rp:
            for su, d in extra(rp).items(): res[r].setdefault(su, {}).update(d)
    hdr = ''.join(f'{(s + ":" + m)[:15]:>16s}' for s, m in COLS)
    print(f'{"":22s}' + hdr, file=out)
    for k, v in res.items():
        cells = []
        for s, m in COLS:
            x = v.get(s, {}).get(m)
            cells.append('            -   ' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:16.3f}')
        print(f'{k:22s}' + ''.join(cells), file=out)
    n = {s: (full[s].get('hobson', {}).get('n_q') or full[s].get('hobson', {}).get('n_pairs')) for s in full}
    print('n:', n, file=out)
    return res


if __name__ == '__main__':
    tags = sys.argv[1:] or sorted({os.path.basename(f).split('.', 1)[1].rsplit('.', 1)[0] for f in glob.glob(f'{R}/*.jsonl')})
    res = table(tags)
    json.dump(res, open(f'{R}/scores.json', 'w'), indent=1)
