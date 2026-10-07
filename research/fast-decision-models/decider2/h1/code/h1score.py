"""H1 laptop scorer: every evalkit suite for each config in one or more preds files, against bf16 hobson refs (and the in-runtime dense, if present).
python3 h1score.py file.json [file2.json ...]  [--ref dense.json]   (files: {'preds': {cfg: {id: {q: {lab: p}}}}, 'meta': ...})"""
import sys, os, json, math, collections
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import evalkit as EK

am = lambda d: max(d, key=d.get)
_ids = {}
def ids_of(s):
    if s not in _ids: _ids[s] = {it['id'] for it in EK.load_suite(s)}
    return _ids[s]


def mcnemar(preds):
    its = EK.load_suite('JB-hard'); hp = EK._as_preds('JB-hard', 'hobson'); b = c = n = 0
    for it in its:
        for q, e in (it.get('expected') or {}).items():
            p = preds.get(it['id'], {}).get(q); h = hp.get(it['id'], {}).get(q)
            if p is None or h is None: continue
            n += 1; pm = am(EK._norm(p)) == e; hm = am(EK._norm(h)) == e
            b += pm and not hm; c += hm and not pm
    k = min(b, c); m = b + c
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return dict(n=n, gain=b, loss=c, p=round(p, 3))


def flips(preds, suite, ref=None):
    """argmax flips vs hobson refs (ref=None) or vs another preds dict; over the questions covered"""
    hp = EK._as_preds(suite, 'hobson') if ref is None else ref
    n = f = 0; tv = 0.0
    for i in ids_of(suite):
        for q, p in preds.get(i, {}).items():
            r = hp.get(i, {}).get(q)
            if r is None: continue
            p, r = EK._norm(p), EK._norm(r); n += 1; f += am(p) != am(r); tv += EK._tv(p, r)
    return dict(flips=f, n=n, rate=(f / n if n else float('nan')), tv=(tv / n if n else float('nan')))


def paired(preds, dense, suite):
    """McNemar: questions where dense agrees with hobson but the model does not (b) vs the reverse (c)"""
    hp = EK._as_preds(suite, 'hobson'); b = c = 0
    for i in ids_of(suite):
        for q, p in preds.get(i, {}).items():
            r = hp.get(i, {}).get(q); dd = dense.get(i, {}).get(q)
            if r is None or dd is None: continue
            a_ = am(EK._norm(r)); pm = am(EK._norm(p)) == a_; dm = am(EK._norm(dd)) == a_
            b += dm and not pm; c += pm and not dm
    k = min(b, c); m = b + c
    pv = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
    return dict(lost=b, gained=c, p=round(pv, 3))


def fgh_vs(suite, preds, ref):
    tr = []
    for pr in EK._pairs(suite):
        q = pr['q']
        try:
            ho = am(EK._norm(ref[pr['a']][q])) == pr['ea'] and am(EK._norm(ref[pr['b']][q])) == pr['eb']
            po = am(EK._norm(preds[pr['a']][q])) == pr['ea'] and am(EK._norm(preds[pr['b']][q])) == pr['eb']
        except KeyError: continue
        if ho: tr.append(po)
    return (sum(tr) / len(tr), len(tr)) if tr else (float('nan'), 0)


def row(preds, dense=None):
    o = {}
    for s in ('REAL-agree', 'LONG'):
        fl = flips(preds, s)
        if not fl['n']: continue
        m = EK.score(s, preds, baselines=False)['model']
        o[s] = dict(flip_rate=fl['rate'], flips=fl['flips'], n=fl['n'], tv=fl['tv'], agree_sd=m.get('agree_sd'), n_sd=m.get('n_sd'))
        if dense is not None:
            fd = flips(preds, s, dense); o[s].update(flip_rate_vs_dense=fd['rate'], flips_vs_dense=fd['flips'], n_vs_dense=fd['n'])
            o[s]['paired'] = paired(preds, dense, s)
    for s in ('CF', 'CF-probe'):
        m = EK.score(s, preds, baselines=False)['model']
        if not m.get('n_pairs'): continue
        hp = EK._as_preds(s, 'hobson'); fg = fgh_vs(s, preds, hp)
        o[s] = dict(fgh=m.get('flip_given_hobson'), n_tracked=fg[1], flip=m.get('flip'), flip_rel=m.get('flip_rel'), dir=m.get('dir'), dmean_rel=m.get('dmean_rel'), n_pairs=m['n_pairs'],
                    item_flips=1 - m.get('agree', float('nan')))
        if dense is not None:
            o[s]['fgh_vs_dense'] = fgh_vs(s, preds, dense)[0]
    m = EK.score('JB-hard', preds, baselines=False)
    if m['model'].get('n_acc'):
        o['JB-hard'] = dict(acc=m['model']['acc'], hobson=m['hobson']['acc'], n=m['model']['n_acc'], agree=m['model'].get('agree'), mcnemar=mcnemar(preds))
    m = EK.score('JB-long', preds, baselines=False)['model']
    if m.get('coverage', 0) > 0.5: o['JB-long'] = dict(acc=m.get('acc'), agree=m.get('agree'))
    m = EK.score('SHUF', preds, baselines=False)
    if m['model'].get('n_pairs'): o['SHUF'] = dict(change=m['model']['change'], both_right=m['model']['both_right'], hob_change=m['hobson'].get('change'), hob_both=m['hobson'].get('both_right'), n=m['model']['n_pairs'])
    m = EK.score('REAL-label', preds, baselines=False)
    if m['model'].get('coverage', 0) > 0.5: o['REAL-label'] = dict(acc=m['model'].get('acc'), hobson=m['hobson'].get('acc'), n=m['model'].get('n_acc'))
    return o


def _f(v, n=3):
    return '-' if v is None or (isinstance(v, float) and math.isnan(v)) else f'{v:.{n}f}'


def fmt(o):
    out = []
    for s, d in o.items():
        if s in ('REAL-agree', 'LONG'):
            t = f"{s}: flips {d['flips']}/{d['n']}={100*d['flip_rate']:.2f}% tv {d['tv']:.4f} agree_sd {_f(d['agree_sd'])}({d['n_sd']})"
            if 'flip_rate_vs_dense' in d: t += f" | vs dense {d['flips_vs_dense']}/{d['n_vs_dense']}={100*d['flip_rate_vs_dense']:.2f}% | vs-hobson paired w/ dense {d['paired']}"
        elif s in ('CF', 'CF-probe'):
            t = f"{s}: fgh {_f(d['fgh'])}({d['n_tracked']}) flip {_f(d['flip'])} rel {_f(d['flip_rel'])} dir {_f(d['dir'])} [pairs {d['n_pairs']}]"
            if 'fgh_vs_dense' in d: t += f" | fgh_vs_dense {_f(d['fgh_vs_dense'])}"
        elif s == 'JB-hard':
            t = f"JB-hard: acc {d['acc']:.3f} (hob {d['hobson']:.3f}) agree {d['agree']:.3f} mcnemar {d['mcnemar']}"
        else:
            t = f"{s}: " + ' '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in d.items())
        out.append(t)
    return '\n   '.join(out)


if __name__ == '__main__':
    args = sys.argv[1:]; ref = None
    if '--ref' in args:
        j = args.index('--ref'); ref = json.load(open(args[j + 1]))['preds']['dense']; del args[j:j + 2]
    allres = {}
    for f in args:
        d = json.load(open(f)); P = d['preds']
        dense = P.get('dense') or ref
        for c, pr in P.items():
            if not pr: continue
            o = row(pr, dense if c != 'dense' else None)
            allres[f'{os.path.basename(f)}::{c}'] = o
            print(f'### {os.path.basename(f)} :: {c}  (n_items {len(pr)})\n   ' + fmt(o), flush=True)
    if os.environ.get('OUT'): json.dump(allres, open(os.environ['OUT'], 'w'), indent=1)
