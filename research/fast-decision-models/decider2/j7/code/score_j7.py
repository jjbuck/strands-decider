"""J7 scoring (laptop, pure python): every evalkit suite in absolute accuracy, paired McNemar vs hobson, Brier + ECE, CF / CF-probe by kind.
python3 score_j7.py OUT.json name1=preds1.json [name2=preds2.json ...]"""
import sys, json, math, collections
sys.path.insert(0, '~/decider2/evalkit')
import numpy as np
import evalkit as EK
from evalkit import _norm, _arg


def mcnemar(a, b):
    """exact two-sided binomial test on discordant pairs; a, b lists of bools (model, hobson)"""
    n01 = sum(1 for x, y in zip(a, b) if x and not y); n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0: return dict(model_only=0, hobson_only=0, p=1.0)
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return dict(model_only=n01, hobson_only=n10, p=round(min(1.0, p), 4))


def labeled_rows(suite, preds):
    hp = EK._as_preds(suite, 'hobson')
    rows = []
    for it in EK.load_suite(suite):
        for q in it['questions']:
            e = (it.get('expected') or {}).get(q)
            if e is None: continue
            p = preds.get(it['id'], {}).get(q); h = hp.get(it['id'], {}).get(q)
            if p is None or h is None: continue
            rows.append((_norm(p), _norm(h), str(e), it))
    return rows


def brier_ece(rows, which=0):
    br = []; conf = []; corr = []
    for r in rows:
        p = r[which]; e = r[2]
        br.append(sum((p.get(k, 0) - (1.0 if k == e else 0.0)) ** 2 for k in set(p) | {e}))
        a = _arg(p); conf.append(p[a]); corr.append(a == e)
    conf = np.array(conf); corr = np.array(corr, dtype=float)
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf >= lo) & (conf < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
        if m.any(): ece += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(np.mean(br)), float(ece)


def pair_kind(suite, preds):
    pairs = EK._pairs(suite); hp = EK._as_preds(suite, 'hobson')
    out = collections.defaultdict(lambda: dict(n=0, model=0, hobson=0))
    for pr in pairs:
        q = pr['q']
        try:
            mo = _arg(_norm(preds[pr['a']][q])) == pr['ea'] and _arg(_norm(preds[pr['b']][q])) == pr['eb']
            ho = _arg(_norm(hp[pr['a']][q])) == pr['ea'] and _arg(_norm(hp[pr['b']][q])) == pr['eb']
        except KeyError: continue
        k = pr['kind']
        if k.startswith('id_'): k = 'identity(' + k + ')'
        out[k]['n'] += 1; out[k]['model'] += mo; out[k]['hobson'] += ho
        g = 'identity_all' if pr['kind'].startswith('id_') else None
        if g: out[g]['n'] += 1; out[g]['model'] += mo; out[g]['hobson'] += ho
    return {k: dict(n=v['n'], model=round(v['model'] / v['n'], 3), hobson=round(v['hobson'] / v['n'], 3)) for k, v in sorted(out.items())}


def pair_mcnemar(suite, preds):
    pairs = EK._pairs(suite); hp = EK._as_preds(suite, 'hobson'); a = []; b = []
    for pr in pairs:
        q = pr['q']
        try:
            a.append(_arg(_norm(preds[pr['a']][q])) == pr['ea'] and _arg(_norm(preds[pr['b']][q])) == pr['eb'])
            b.append(_arg(_norm(hp[pr['a']][q])) == pr['ea'] and _arg(_norm(hp[pr['b']][q])) == pr['eb'])
        except KeyError: continue
    return mcnemar(a, b)


def score_one(preds):
    R = {}
    for s in ('JB-all', 'JB-hard', 'REAL-label'):
        rows = labeled_rows(s, preds)
        acc = float(np.mean([_arg(r[0]) == r[2] for r in rows])); hacc = float(np.mean([_arg(r[1]) == r[2] for r in rows]))
        bm, em = brier_ece(rows, 0); bh, eh = brier_ece(rows, 1)
        R[s] = dict(n=len(rows), acc=round(acc, 4), hobson_acc=round(hacc, 4), mcnemar=mcnemar([_arg(r[0]) == r[2] for r in rows], [_arg(r[1]) == r[2] for r in rows]),
                    brier=round(bm, 4), hobson_brier=round(bh, 4), ece=round(em, 4), hobson_ece=round(eh, 4))
    for s in ('REAL-agree', 'LONG', 'JB-long'):
        m = EK.score(s, preds, baselines=False)['model']
        R[s] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    for s in ('CF', 'CF-probe'):
        m = EK.score(s, preds, baselines=False)['model']
        R[s] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
        R[s]['mcnemar_pairs'] = pair_mcnemar(s, preds)
        R[s]['by_kind'] = pair_kind(s, preds)
        rows = labeled_rows(s, preds)
        bm, em = brier_ece(rows, 0); bh, eh = brier_ece(rows, 1)
        R[s].update(brier=round(bm, 4), hobson_brier=round(bh, 4), ece=round(em, 4), hobson_ece=round(eh, 4))
    m = EK.score('SHUF', preds, baselines=False)['model']
    R['SHUF'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    # REAL-label by state length
    rows = labeled_rows('REAL-label', preds); bins = collections.defaultdict(list)
    for p, h, e, it in rows:
        b = 'state>=2k' if it.get('n_state_tok', 0) >= 2000 else 'state<2k'
        bins[b].append((_arg(p) == e, _arg(h) == e))
    R['REAL-label']['by_len'] = {k: dict(n=len(v), acc=round(float(np.mean([x[0] for x in v])), 3), hobson=round(float(np.mean([x[1] for x in v])), 3)) for k, v in bins.items()}
    return R


if __name__ == '__main__':
    out = sys.argv[1]; res = {}
    for a in sys.argv[2:]:
        nm, path = a.split('=', 1)
        preds = json.load(open(path))
        res[nm] = score_one(preds)
        r = res[nm]
        print(f"== {nm}: JB-all {r['JB-all']['acc']:.3f} (h {r['JB-all']['hobson_acc']:.3f}, p {r['JB-all']['mcnemar']['p']}) | JB-hard {r['JB-hard']['acc']:.3f} (h {r['JB-hard']['hobson_acc']:.3f}, p {r['JB-hard']['mcnemar']['p']}) | "
              f"REAL-label {r['REAL-label']['acc']:.3f} (p {r['REAL-label']['mcnemar']['p']}) | REAL agree {r['REAL-agree'].get('agree')} sd {r['REAL-agree'].get('agree_sd')} | "
              f"LONG agree {r['LONG'].get('agree')} sd {r['LONG'].get('agree_sd')} | CF acc {r['CF'].get('acc')} flip {r['CF'].get('flip')} fgh {r['CF'].get('flip_given_hobson')} | "
              f"CF-probe acc {r['CF-probe'].get('acc')} flip {r['CF-probe'].get('flip')} fgh {r['CF-probe'].get('flip_given_hobson')} | SHUF both {r['SHUF'].get('both_right')}")
        print('   CF kinds', {k: (v['model'], v['hobson']) for k, v in r['CF']['by_kind'].items()})
        print('   CF-probe kinds', {k: (v['model'], v['hobson']) for k, v in r['CF-probe']['by_kind'].items()})
    json.dump(res, open(out, 'w'), indent=1)
