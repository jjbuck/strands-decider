"""laptop: score H5 suite predictions (h5suite.py JSONL) with evalkit against the bf16 hobson references, plus flips against the
same-runtime dense bf16 run (isolates quantization from runtime noise) and the BRIEF7 low-bit kill criterion.
python3 h5score.py res/preds_dense.jsonl res/preds_X.jsonl [...]   (first file = same-runtime dense reference)"""
import sys, os, json, math, collections
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK


def load(p):
    preds = {}
    for l in open(p):
        r = json.loads(l); preds.setdefault(r['id'], {})[r['q']] = r['p']
    return preds


def amax(d): return max(d, key=d.get)


def mcnemar(preds, ref=None):
    its = EK.load_suite('JB-hard'); hp = ref if ref is not None else EK._as_preds('JB-hard', 'hobson'); b = c = n = 0
    for it in its:
        for q, e in (it.get('expected') or {}).items():
            p = preds.get(it['id'], {}).get(q); h = hp.get(it['id'], {}).get(q)
            if p is None or h is None: continue
            n += 1; pm = amax(EK._norm(p)) == e; hm = amax(EK._norm(h)) == e
            b += pm and not hm; c += hm and not pm
    k = min(b, c); m = b + c
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return dict(n=n, model_only=b, ref_only=c, p=round(p, 4))


def flips(preds, ref, suite):
    ids = {it['id'] for it in EK.load_suite(suite)}
    n = f = 0; tv = 0.0
    for i in ids:
        for q, p in preds.get(i, {}).items():
            r = ref.get(i, {}).get(q)
            if r is None: continue
            n += 1; f += amax(EK._norm(p)) != amax(EK._norm(r)); tv += EK._tv(EK._norm(p), EK._norm(r))
    return dict(flips=f, n=n, rate=round(f / max(n, 1), 4), tv=round(tv / max(n, 1), 4))


def fgh(suite, preds, ref, kinds=None):
    tr = []
    for pr in EK._pairs(suite):
        if kinds and not any(k in pr['kind'] for k in kinds): continue
        q = pr['q']
        try:
            ho = amax(EK._norm(ref[pr['a']][q])) == pr['ea'] and amax(EK._norm(ref[pr['b']][q])) == pr['eb']
            po = amax(EK._norm(preds[pr['a']][q])) == pr['ea'] and amax(EK._norm(preds[pr['b']][q])) == pr['eb']
        except KeyError: continue
        if ho: tr.append(po)
    return (round(sum(tr) / len(tr), 4), len(tr)) if tr else (float('nan'), 0)


def full(name, preds, dense):
    out = dict(cfg=name)
    for s in ('JB-hard', 'JB-long', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label'):
        try: out[s] = EK.score(s, preds, baselines=False)['model']
        except Exception as e: out[s] = dict(err=str(e))
    hob = {}
    for s in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe'):
        for i, qs in EK._as_preds(s, 'hobson').items(): hob.setdefault(i, {}).update(qs)
    out['flips_vs_hobson'] = {s: flips(preds, hob, s) for s in ('REAL-agree', 'LONG', 'CF', 'CF-probe', 'JB-all')}
    if dense is not None:
        out['flips_vs_dense'] = {s: flips(preds, dense, s) for s in ('REAL-agree', 'LONG', 'CF', 'CF-probe', 'JB-all')}
        out['fgh_vs_dense'] = {'CF': fgh('CF', preds, dense), 'CF-probe': fgh('CF-probe', preds, dense)}
        out['mcnemar_vs_dense'] = mcnemar(preds, dense)
    out['fgh_vs_hobson'] = {'CF': fgh('CF', preds, hob), 'CF-probe': fgh('CF-probe', preds, hob), 'CF-probe_distract': fgh('CF-probe', preds, hob, ['_distract'])}
    out['mcnemar_vs_hobson'] = mcnemar(preds)
    fr = out['flips_vs_hobson']['REAL-agree']['rate']
    out['kill'] = dict(real_flips_lt_0005=fr < 0.005, cf_fgh_ge_097=out['fgh_vs_hobson']['CF'][0] >= 0.97,
                       cfp_fgh_ge_097=out['fgh_vs_hobson']['CF-probe'][0] >= 0.97, jb_mcnemar_ns=not (out['mcnemar_vs_hobson']['p'] < 0.05 and out['mcnemar_vs_hobson']['ref_only'] > out['mcnemar_vs_hobson']['model_only']))
    return out


def brief(o):
    g = lambda s, k: o.get(s, {}).get(k)
    f = lambda v: '  -  ' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.3f}'
    fh = o['flips_vs_hobson']; fd = o.get('flips_vs_dense', {})
    s = (f"{o['cfg']:28s} JBh {f(g('JB-hard','acc'))} | REAL agree {f(g('REAL-agree','agree'))} sd {f(g('REAL-agree','agree_sd'))} flips {fh['REAL-agree']['rate']:.4f}"
         f"{(' (vs dense %.4f)' % fd['REAL-agree']['rate']) if fd else ''} | LONG sd {f(g('LONG','agree_sd'))} flips {fh['LONG']['rate']:.4f} | CF fgh {f(o['fgh_vs_hobson']['CF'][0])}"
         f" flip {f(g('CF','flip'))} | CFP fgh {f(o['fgh_vs_hobson']['CF-probe'][0])} dis {f(o['fgh_vs_hobson']['CF-probe_distract'][0])} | SHUF ch {f(g('SHUF','change'))} | RL {f(g('REAL-label','acc'))}"
         f" | McN {o['mcnemar_vs_hobson']} | kill {o['kill']}")
    return s


if __name__ == '__main__':
    dense = load(sys.argv[1]) if sys.argv[1] != '-' else None
    outs = []
    for p in sys.argv[1:]:
        if p == '-': continue
        pr = load(p); o = full(os.path.basename(p).replace('preds_', '').replace('.jsonl', ''), pr, dense if p != sys.argv[1] else None)
        outs.append(o); print(brief(o))
    json.dump(outs, open(os.environ.get('OUT', '~/decider2/h5/scores.json'), 'w'), indent=1)
