"""laptop: score a g2map output with evalkit.  python3 score.py res/map_pilot.json [more.json ...]"""
import sys, os, json, math, collections
sys.path.insert(0, '/tmp/decider2/evalkit')
import evalkit as EK


def amax(d): return max(d, key=d.get)


def mcnemar(preds):
    its = EK.load_suite('JB-hard'); hp = EK._as_preds('JB-hard', 'hobson'); b = c = n = 0
    for it in its:
        for q, e in (it.get('expected') or {}).items():
            p = preds.get(it['id'], {}).get(q); h = hp.get(it['id'], {}).get(q)
            if p is None or h is None: continue
            n += 1; pm = amax(EK._norm(p)) == e; hm = amax(EK._norm(h)) == e
            b += pm and not hm; c += hm and not pm
    k = min(b, c); m = b + c
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return n, b, c, p


def fgh_kind(suite, preds, kinds):
    hp = EK._as_preds(suite, 'hobson'); tr = []
    for pr in EK._pairs(suite):
        if not any(k in pr['kind'] for k in kinds): continue
        q = pr['q']
        try:
            ho = amax(EK._norm(hp[pr['a']][q])) == pr['ea'] and amax(EK._norm(hp[pr['b']][q])) == pr['eb']
            po = amax(EK._norm(preds[pr['a']][q])) == pr['ea'] and amax(EK._norm(preds[pr['b']][q])) == pr['eb']
        except KeyError: continue
        if ho: tr.append(po)
    return (sum(tr) / len(tr), len(tr)) if tr else (float('nan'), 0)


def flips_vs(preds, ref, suites=('REAL-agree', 'LONG')):
    out = {}
    for su in suites:
        ids = {it['id'] for it in EK.load_suite(su)}
        n = f = 0; tv = 0.0
        for i in ids:
            for q, p in preds.get(i, {}).items():
                r = ref.get(i, {}).get(q)
                if r is None: continue
                n += 1; f += amax(EK._norm(p)) != amax(EK._norm(r)); tv += EK._tv(EK._norm(p), EK._norm(r))
        out[su] = (f, n, tv / max(n, 1))
    return out


def fgh_ref(suite, preds, ref):
    tr = []
    for pr in EK._pairs(suite):
        q = pr['q']
        try:
            ho = amax(EK._norm(ref[pr['a']][q])) == pr['ea'] and amax(EK._norm(ref[pr['b']][q])) == pr['eb']
            po = amax(EK._norm(preds[pr['a']][q])) == pr['ea'] and amax(EK._norm(preds[pr['b']][q])) == pr['eb']
        except KeyError: continue
        if ho: tr.append(po)
    return (sum(tr) / len(tr), len(tr)) if tr else (float('nan'), 0)


DENSE = None


def row(name, preds, cost):
    out = dict(cfg=name, cost=cost)
    if DENSE is not None:
        out['flips'] = flips_vs(preds, DENSE)
        out['fgh_dense'] = (fgh_ref('CF', preds, DENSE), fgh_ref('CF-probe', preds, DENSE))
    for s, key in (('REAL-agree', 'agree_sd'), ('LONG', 'agree_sd')):
        m = EK.score(s, preds, baselines=False)['model']
        out[s] = (m.get(key), m.get('n_sd', 0)); out[s + '_agree'] = m.get('agree')
    for s in ('CF', 'CF-probe'):
        m = EK.score(s, preds, baselines=False)['model']
        out[s] = (m.get('flip_given_hobson'), m.get('n_pairs', 0)); out[s + '_flip'] = m.get('flip')
    out['CFP-distract'] = fgh_kind('CF-probe', preds, ['_distract'])
    m = EK.score('JB-hard', preds, baselines=False)['model']
    out['JB-hard'] = (m.get('acc'), m.get('n_acc', 0))
    out['mcnemar'] = mcnemar(preds)
    return out


def fmt(v):
    if isinstance(v, tuple): return '   -   ' if v[0] is None or (isinstance(v[0], float) and math.isnan(v[0])) else f'{v[0]:.3f}/{v[1]:<3d}'
    return '  -  ' if v is None else f'{v:.3f}'


if __name__ == '__main__':
    allrows = []
    for f in sys.argv[1:]:
        d = json.load(open(f))
        if os.environ.get('REF'): DENSE = json.load(open(os.environ['REF']))['preds']['dense']
        elif 'dense' in d['preds']: DENSE = d['preds']['dense']
        for c, pr in d['preds'].items():
            if not pr: continue
            allrows.append(row(c, pr, d['meta'][c]['cost']))
        rc = collections.defaultdict(list)
        for k, v in d.get('recall', {}).items():
            for s, x in v['recall10'].items(): rc[s].append(x)
        print(f, 'locator recall of exact qa7 top-10%:', {s: round(sum(v) / len(v), 3) for s, v in rc.items()})
    if allrows:
        d0 = json.load(open(sys.argv[1]))['preds']; keyset = {(i, q) for c in d0.values() for i, qs in c.items() for q in qs}
        for b in ('merged_full', 'nostate', 'drop@7', 'rand50@7', 'qattn10@7', 'qattn10@3', 'rand50@0', 'rand25@0', 'rand10@0', 'rand50@3'):
            bp = {}
            for s_ in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe'):
                for i, qs in EK._as_preds(s_, b).items():
                    for q, v in qs.items():
                        if (i, q) in keyset: bp.setdefault(i, {})[q] = v
            allrows.append(row('[base] ' + b, bp, float('nan')))
    print(f'{"config":18s} {"cost":>5s} {"REAL sd":>11s} {"LONG sd":>11s} {"CF fgh":>11s} {"CFP fgh":>11s} {"CFP-dis":>11s} {"JB-hard":>11s}  mcnemar(n,b,c,p)')
    for r in allrows:
        print(f'{r["cfg"]:18s} {r["cost"]:5.3f} {fmt(r["REAL-agree"]):>11s} {fmt(r["LONG"]):>11s} {fmt(r["CF"]):>11s} {fmt(r["CF-probe"]):>11s} {fmt(r["CFP-distract"]):>11s} {fmt(r["JB-hard"]):>11s}  {r["mcnemar"]}')
    print('\nvs my dense runtime: flips (n_flip/n, mean TV) on REAL-agree / LONG; fgh relative to the pairs my dense tracks (CF, CF-probe); agreement vs hobson (all q)')
    for r in allrows:
        if 'flips' not in r: continue
        fl = r['flips']; fd = r['fgh_dense']
        tot_f = sum(v[0] for v in fl.values()); tot_n = sum(v[1] for v in fl.values())
        print(f'{r["cfg"]:18s} REAL {fl["REAL-agree"][0]}/{fl["REAL-agree"][1]} tv {fl["REAL-agree"][2]:.4f} | LONG {fl["LONG"][0]}/{fl["LONG"][1]} tv {fl["LONG"][2]:.4f} | all {tot_f}/{tot_n} = {tot_f / max(tot_n, 1):.4f}'
              f' | fgh_dense CF {fmt(fd[0])} CFP {fmt(fd[1])} | agree_hob REAL {fmt(r["REAL-agree_agree"])} LONG {fmt(r["LONG_agree"])}')
