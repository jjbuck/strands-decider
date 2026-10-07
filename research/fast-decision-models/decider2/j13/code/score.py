"""Laptop scorer (pure python): records jsonl -> per-config evalkit metrics + FLOPs.
python score.py OUT.json rec1.jsonl [rec2.jsonl ...]"""
import sys, json, math, collections
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(paths):
    recs = []
    for p in paths:
        for l in open(p):
            try: recs.append(json.loads(l))
            except Exception: pass
    return recs


def score_cfg(recs, name, minlen=0):
    """minlen: requests whose state is shorter than minlen tokens run dense (policy applied at scoring time)"""
    preds = {}
    use = lambda r: name == 'dense' or r['n_s'] < minlen
    for r in recs:
        d = r['dense'] if use(r) else r['cfg'].get(name)
        if d is None: continue
        preds.setdefault(r['id'], {})[r['q']] = d
    out = {}
    m = EK.score('REAL-agree', preds, baselines=False)['model']
    out['REAL'] = {k: m.get(k) for k in ('coverage', 'agree', 'agree_sd', 'n_sd', 'tv')}
    m = EK.score('REAL-label', preds, baselines=False)['model']
    if m.get('n_acc'): out['REAL-label'] = {k: m.get(k) for k in ('acc', 'acc_hobson_same', 'n_acc')}
    m = EK.score('LONG', preds, baselines=False)['model']
    out['LONG'] = {k: m.get(k) for k in ('coverage', 'agree', 'agree_sd', 'n_sd', 'tv')}
    for s in ('CF', 'CF-probe'):
        m = EK.score(s, preds, baselines=False)
        mm = m['model']; hh = m['hobson']
        out[s] = {k: mm.get(k) for k in ('n_pairs', 'acc', 'flip', 'flip_rel', 'flip_given_hobson', 'dir', 'dmean_rel', 'agree', 'tv')}
    # JB-hard paired with hobson
    hp = EK._as_preds('JB-all', 'hobson'); b = c = n = 0; acc = []; hacc = []
    for it in EK.load_suite('JB-hard'):
        for q in it['questions']:
            p = preds.get(it['id'], {}).get(q); h = hp.get(it['id'], {}).get(q)
            if p is None or h is None: continue
            e = it['expected'][q]; pm = EK._arg(EK._norm(p)) == e; hm = EK._arg(EK._norm(h)) == e
            acc.append(pm); hacc.append(hm); n += 1
            if hm and not pm: b += 1
            if pm and not hm: c += 1
    if n: out['JB-hard'] = {'n': n, 'acc': sum(acc) / n, 'hobson_acc': sum(hacc) / n, 'lost': b, 'gained': c, 'mcnemar_p': mcnemar(b, c)}
    # FLOPs (token weighted), per suite
    fl = collections.defaultdict(lambda: [0.0, 0])
    for r in recs:
        f = 1.0 if use(r) else r['fl'].get(name)
        if f is None: continue
        fl[r['suite']][0] += f * r['T']; fl[r['suite']][1] += r['T']
        fl['all'][0] += f * r['T']; fl['all'][1] += r['T']
    out['flops'] = {s: round(v[0] / v[1], 4) for s, v in fl.items() if v[1]}
    # agreement vs this runtime's own dense output (runtime floor removed)
    ag = []; tvs = []
    for r in recs:
        if r['suite'] not in ('REAL-agree', 'LONG'): continue
        d = r['dense'] if use(r) else r['cfg'].get(name)
        if d is None: continue
        ag.append(EK._arg(d) == EK._arg(r['dense'])); tvs.append(EK._tv(EK._norm(d), EK._norm(r['dense'])))
    if ag: out['vs_dense'] = {'agree': sum(ag) / len(ag), 'tv': sum(tvs) / len(tvs), 'n': len(ag)}
    return out


def row(name, s):
    g = lambda d, k: (('%.3f' % d[k]) if isinstance(d.get(k), float) else str(d.get(k, '-'))) if d else '-'
    return (f"| {name} | {s['flops'].get('all', 1):.3f} | {g(s['REAL'], 'agree_sd')} | {g(s['LONG'], 'agree_sd')} | {g(s['CF'], 'flip_given_hobson')} | "
            f"{g(s['CF-probe'], 'flip_given_hobson')} | {g(s['CF'], 'acc')} | {g(s['CF-probe'], 'acc')} | {g(s['CF'], 'flip')} | {g(s['CF-probe'], 'flip')} | "
            f"{g(s.get('JB-hard'), 'acc')} ({s.get('JB-hard', {}).get('lost', '-')}/{s.get('JB-hard', {}).get('gained', '-')}, p {g(s.get('JB-hard'), 'mcnemar_p')}) | "
            f"{g(s.get('REAL-label'), 'acc')} | {g(s['REAL'], 'tv')} | {g(s.get('vs_dense'), 'agree')} |")


HDR = ("| config | FLOPs | REAL agree_sd | LONG agree_sd | CF fgh | CF-probe fgh | CF acc | CF-probe acc | CF flip | CF-probe flip | JB-hard acc (lost/gained, McNemar p) | REAL-label | REAL tv | agree vs own dense |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

if __name__ == '__main__':
    outp = sys.argv[1]; args = sys.argv[2:]
    minlens = [0]
    if args and args[0].startswith('--minlen='): minlens = [int(x) for x in args[0].split('=')[1].split(',')]; args = args[1:]
    recs = load(args)
    names = ['dense'] + sorted({c for r in recs for c in r['cfg']} - {'dense'}, key=lambda c: (c.split('@')[1] if '@' in c else '', c))
    res = {}
    print(HDR)
    for ml in minlens:
        for nm in names:
            key = nm if ml == 0 else f'{nm}|min{ml}'
            if nm == 'dense' and ml: continue
            res[key] = score_cfg(recs, nm, ml); print(row(key, res[key]))
    hob = {'REAL': EK.score('REAL-agree', {}, baselines=False)}
    json.dump(res, open(outp, 'w'), indent=1)
