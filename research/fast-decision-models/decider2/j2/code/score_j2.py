"""Score J2 preds on the laptop (pure python + numpy): every evalkit suite, absolute accuracy, Brier/ECE, paired exact McNemar vs hobson
and vs a control arm.  python3 score_j2.py OUT.json name=preds.json [name=preds.json ...] [--control causal]"""
import sys, json, math, collections
sys.path.insert(0, '~/decider2/evalkit')
import numpy as np
import evalkit as EK


def mcnemar(a, b):
    """a, b: lists of bools (paired). -> (n_a_only, n_b_only, exact two-sided p)"""
    n01 = sum(1 for x, y in zip(a, b) if x and not y); n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0: return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return n01, n10, min(1.0, p)


def brier(p, lab):
    return sum((v - (1.0 if k == lab else 0.0)) ** 2 for k, v in p.items())


def ece(conf, ok, nb=10):
    conf = np.array(conf); ok = np.array(ok, dtype=float); e = 0.0
    for b in range(nb):
        m = (conf > b / nb) & (conf <= (b + 1) / nb)
        if m.sum(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def per_q_correct(name, preds):
    """{(id, q): bool} for suites with ground truth, plus (p, label) for Brier"""
    out = {}; pb = {}
    for r in EK._qrows(name, preds):
        if r['exp'] is None or r['pred'] is None: continue
        out[(r['id'], r['q'])] = EK._arg(r['pred']) == r['exp']; pb[(r['id'], r['q'])] = (r['pred'], r['exp'])
    return out, pb


def pair_correct(name, preds):
    out = {}
    for pr in EK._pairs(name):
        q = pr['q']
        try:
            pa = EK._norm(preds[pr['a']][q]); pb = EK._norm(preds[pr['b']][q])
        except KeyError: continue
        out[(pr['a'], pr['b'])] = (EK._arg(pa) == pr['ea'] and EK._arg(pb) == pr['eb'], pr)
    return out


def item_correct(name, preds):
    out = {}; pbs = {}
    for pr in EK._pairs(name):
        q = pr['q']
        for it, e in ((pr['a'], pr['ea']), (pr['b'], pr['eb'])):
            try: p = EK._norm(preds[it][q])
            except KeyError: continue
            out[(it, q)] = EK._arg(p) == e; pbs[(it, q)] = (p, e)
    return out, pbs


def summarize(preds):
    S = {}
    for s in ('JB-all', 'JB-hard', 'JB-long', 'REAL-agree', 'LONG', 'REAL-label'):
        S[s] = EK.score(s, preds, baselines=False)['model']
    for s in ('CF', 'CF-probe'):
        S[s] = EK.score(s, preds, baselines=False)['model']
        bk = collections.defaultdict(list)
        for (a, b), (ok, pr) in pair_correct(s, preds).items(): bk[pr['kind']].append(ok)
        S[s]['by_kind'] = {k: round(float(np.mean(v)), 3) for k, v in sorted(bk.items())}
        S[s]['by_pos'] = {g: round(d['model'], 3) if d['model'] is not None else None for g, d in EK.breakdown(s, preds, by='pos', rows=()).items()}
        S[s]['by_len'] = {g: (round(d['model'], 3) if d['model'] is not None else None, d['n']) for g, d in EK.breakdown(s, preds, by='len', rows=()).items()}
    S['SHUF'] = EK.score('SHUF', preds, baselines=False)['model']
    # calibration on every ground-truth question
    conf = []; ok = []; br = []
    for s in ('JB-all', 'REAL-label'):
        _, pb = per_q_correct(s, preds)
        for p, lab in pb.values():
            br.append(brier(p, lab)); a = EK._arg(p); conf.append(p[a]); ok.append(a == lab)
    for s in ('CF', 'CF-probe'):
        _, pb = item_correct(s, preds)
        for p, lab in pb.values():
            br.append(brier(p, lab)); a = EK._arg(p); conf.append(p[a]); ok.append(a == lab)
    S['calib'] = dict(brier=float(np.mean(br)), ece=ece(conf, ok), n=len(br))
    for s in ('JB-all', 'REAL-label'):
        _, pb = per_q_correct(s, preds)
        S[s]['brier'] = float(np.mean([brier(p, l) for p, l in pb.values()]))
    return S


def paired(pa, pb):
    """paired tests of model a vs model b (a-only correct, b-only correct, p)"""
    R = {}
    for s in ('JB-all', 'JB-hard', 'REAL-label'):
        ca, _ = per_q_correct(s, pa); cb, _ = per_q_correct(s, pb)
        ks = [k for k in ca if k in cb]
        R[s] = mcnemar([ca[k] for k in ks], [cb[k] for k in ks])
    for s in ('CF', 'CF-probe'):
        ca = pair_correct(s, pa); cb = pair_correct(s, pb)
        ks = [k for k in ca if k in cb]
        R[s + ' pairs'] = mcnemar([ca[k][0] for k in ks], [cb[k][0] for k in ks])
        ia, _ = item_correct(s, pa); ib, _ = item_correct(s, pb)
        ks = [k for k in ia if k in ib]
        R[s + ' items'] = mcnemar([ia[k] for k in ks], [ib[k] for k in ks])
    for s in ('REAL-agree', 'LONG'):    # agreement between the two models' decisions
        ra = {(r['id'], r['q']): r['pred'] for r in EK._qrows(s, pa) if r['pred'] is not None}
        rb = {(r['id'], r['q']): r['pred'] for r in EK._qrows(s, pb) if r['pred'] is not None}
        ks = [k for k in ra if k in rb]
        R[s + ' mutual_agree'] = round(float(np.mean([EK._arg(ra[k]) == EK._arg(rb[k]) for k in ks])), 4) if ks else None
    return R


if __name__ == '__main__':
    out = sys.argv[1]; ctl = None; args = sys.argv[2:]
    if '--control' in args:
        i = args.index('--control'); ctl = args[i + 1]; args = args[:i] + args[i + 2:]
    P = {}
    for a in args:
        n, p = a.split('=', 1); P[n] = json.load(open(p))
    P['hobson'] = {}
    for s in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe'):
        for iid, d in EK._as_preds(s, 'hobson').items(): P['hobson'].setdefault(iid, {}).update(d)
    res = {n: summarize(p) for n, p in P.items()}
    res['_paired_vs_hobson'] = {n: paired(p, P['hobson']) for n, p in P.items() if n != 'hobson'}
    if ctl: res['_paired_vs_' + ctl] = {n: paired(p, P[ctl]) for n, p in P.items() if n not in (ctl, 'hobson')}
    json.dump(res, open(out, 'w'), indent=1, default=str)
    cols = [('JB-all', 'acc'), ('JB-hard', 'acc'), ('REAL-label', 'acc'), ('REAL-agree', 'agree_sd'), ('LONG', 'agree'), ('LONG', 'agree_sd'),
            ('CF', 'acc'), ('CF', 'flip'), ('CF', 'flip_given_hobson'), ('CF-probe', 'acc'), ('CF-probe', 'flip'), ('CF-probe', 'flip_given_hobson'),
            ('SHUF', 'both_right'), ('calib', 'brier'), ('calib', 'ece')]
    print('| model | ' + ' | '.join(f'{a} {b}' for a, b in cols) + ' |')
    print('|---' * (len(cols) + 1) + '|')
    for n in P:
        print(f'| {n} | ' + ' | '.join(f"{res[n][a].get(b, float('nan')):.3f}" for a, b in cols) + ' |')
    for k in [k for k in res if k.startswith('_paired')]:
        print(k)
        for n, r in res[k].items(): print('  ', n, r)
    for n in P:
        print(n, 'CF by kind', res[n]['CF']['by_kind']); print(n, 'CF-probe by kind', res[n]['CF-probe']['by_kind'])
