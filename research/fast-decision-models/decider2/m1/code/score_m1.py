"""M1 scoring (laptop, pure python; copied from J3 score_dt.py + the relaxed bar): python3 score_m1.py OUT.json preds/a.json [preds/b.json ...]
Per preds file: absolute accuracy on every suite against hobson on the SAME covered items, McNemar (exact, paired) on JB-all / JB-hard,
REAL-label (also split by state length), CF / CF-probe pair accuracy (flip) and acc, agree / agree_sd, SHUF, Brier and ECE, and the brief's bar:
no significant drop on JB-all or JB-hard (McNemar p < .05 with fewer right), REAL-label >= .78, CF and CF-probe flip >= hobson's."""
import sys, os, json, math, collections
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))  # laptop copy of the kit
import evalkit as EK
import numpy as np


def mcnemar(rows):
    b = sum(1 for r in rows if r[0] and not r[1]); c = sum(1 for r in rows if r[1] and not r[0]); n = b + c
    if n == 0: return 1.0, b, c
    k = min(b, c); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p), b, c


def brier(p, e):
    return sum((v - (1.0 if k == e else 0.0)) ** 2 for k, v in p.items()) + (0.0 if e in p else 1.0)


def ece(rows, bins=10):
    """rows: (confidence, correct)"""
    if not rows: return float('nan')
    tot = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [r for r in rows if (lo < r[0] <= hi) or (b == 0 and r[0] == 0)]
        if sel: tot += len(sel) / len(rows) * abs(np.mean([r[0] for r in sel]) - np.mean([r[1] for r in sel]))
    return float(tot)


def labelled(name, preds):
    """(model dist, hobson dist, expected, item) for labelled questions covered by preds"""
    out = []
    for r in EK._qrows(name, preds):
        if r['pred'] is None or r['exp'] is None or r['hob'] is None: continue
        out.append((r['pred'], r['hob'], r['exp'], r['item']))
    return out


def summarize(preds):
    preds = EK._flat(preds, None); out = {}
    for s in ('JB-all', 'JB-hard', 'REAL-label', 'CF', 'CF-probe'):
        L = labelled(s, preds)
        if not L: continue
        ok = [(EK._arg(p) == e, EK._arg(h) == e) for p, h, e, _ in L]
        out[f'{s} n'] = len(L)
        out[f'{s} acc'] = float(np.mean([o[0] for o in ok])); out[f'{s} hob'] = float(np.mean([o[1] for o in ok]))
        pv, b, c = mcnemar(ok); out[f'{s} mcnemar p'] = pv; out[f'{s} model-only/hob-only'] = f'{b}/{c}'
        out[f'{s} brier'] = float(np.mean([brier(p, e) for p, h, e, _ in L])); out[f'{s} brier hob'] = float(np.mean([brier(h, e) for p, h, e, _ in L]))
        out[f'{s} ece'] = ece([(max(p.values()), EK._arg(p) == e) for p, h, e, _ in L]); out[f'{s} ece hob'] = ece([(max(h.values()), EK._arg(h) == e) for p, h, e, _ in L])
    L = labelled('REAL-label', preds)
    for nm, f in (('short<2000', lambda n: n < 2000), ('long>=2000', lambda n: n >= 2000)):
        sel = [(EK._arg(p) == e, EK._arg(h) == e) for p, h, e, it in L if f(it.get('n_state_tok', 0))]
        if sel: out[f'REAL-label {nm}'] = (len(sel), float(np.mean([x[0] for x in sel])), float(np.mean([x[1] for x in sel])))
    for s in ('JB-long', 'REAL-agree', 'LONG'):
        m = EK.score(s, preds, baselines=False)['model']
        for k in ('coverage', 'agree', 'agree_sd', 'tv'): out[f'{s} {k}'] = m.get(k)
    for s in ('CF', 'CF-probe'):
        sc = EK.score(s, preds, baselines=False)
        m = sc['model']
        for k in ('n_pairs', 'acc', 'flip', 'flip_rel', 'flip_given_hobson', 'dir', 'agree'): out[f'{s} {k}'] = m.get(k)
        # hobson on the same covered pairs
        cov = {x for pr in EK._pairs(s) if pr['a'] in preds and pr['b'] in preds for x in (pr['a'], pr['b'])}
        hm = EK._pair_metrics(s, {k: v for k, v in EK._as_preds(s, 'hobson').items() if k in cov})
        out[f'{s} flip hob'] = hm.get('flip'); out[f'{s} acc_pair hob'] = hm.get('acc')
        # paired test on pairs: tracked (both items right) by the model vs by hobson
        hp = EK._as_preds(s, 'hobson'); rows = []
        for pr in EK._pairs(s):
            q = pr['q']; pa, pb = preds.get(pr['a'], {}).get(q), preds.get(pr['b'], {}).get(q)
            ha, hb = hp.get(pr['a'], {}).get(q), hp.get(pr['b'], {}).get(q)
            if None in (pa, pb, ha, hb): continue
            tm = EK._arg(EK._norm(pa)) == pr['ea'] and EK._arg(EK._norm(pb)) == pr['eb']
            th = EK._arg(EK._norm(ha)) == pr['ea'] and EK._arg(EK._norm(hb)) == pr['eb']
            rows.append((tm, th))
        if rows:
            pv, b, c = mcnemar(rows); out[f'{s} pair mcnemar p'] = pv; out[f'{s} pair model-only/hob-only'] = f'{b}/{c}'; out[f'{s} pair n'] = len(rows)
        bd = EK.breakdown(s, preds, 'kind', rows=('hobson',))
        out[f'{s} by kind'] = {g: (round(d['model'], 3) if d.get('model') is not None else None, round(d['hobson'], 3) if d.get('hobson') is not None else None, d['n']) for g, d in bd.items()}
    out['SHUF both_right'] = EK.score('SHUF', preds, baselines=False)['model'].get('both_right')
    def drop(s): return out.get(f'{s} mcnemar p', 1) < 0.05 and out.get(f'{s} acc', 0) < out.get(f'{s} hob', 0)
    bar = dict(jb_all_ok=not drop('JB-all'), jb_hard_ok=not drop('JB-hard'), real_label_ok=(out.get('REAL-label acc') or 0) >= 0.78,
               cf_ok=(out.get('CF flip') or 0) >= (out.get('CF flip hob') or 1), cfp_ok=(out.get('CF-probe flip') or 0) >= (out.get('CF-probe flip hob') or 1))
    out['bar'] = bar; out['PASS'] = all(bar.values())
    rel = dict(jb_all_ok=(out.get('JB-all acc') or 0) >= 0.693, real_label_ok=(out.get('REAL-label acc') or 0) >= 0.765,
               real_agree_sd_ok=(out.get('REAL-agree agree_sd') or 0) >= 0.85, cf_ok=bar['cf_ok'], cfp_ok=bar['cfp_ok'])
    out['bar_relaxed'] = rel; out['PASS_relaxed'] = all(rel.values())
    return out


KEYS = ['JB-all n', 'JB-all acc', 'JB-all hob', 'JB-all mcnemar p', 'JB-hard acc', 'JB-hard hob', 'JB-hard mcnemar p', 'REAL-label n', 'REAL-label acc',
        'REAL-label hob', 'REAL-agree agree', 'REAL-agree agree_sd', 'LONG agree', 'LONG agree_sd', 'JB-long agree', 'CF n_pairs', 'CF flip', 'CF flip hob',
        'CF acc', 'CF flip_given_hobson', 'CF-probe n_pairs', 'CF-probe flip', 'CF-probe flip hob', 'CF-probe acc', 'CF-probe flip_given_hobson',
        'CF pair mcnemar p', 'CF-probe pair mcnemar p', 'SHUF both_right', 'JB-all brier', 'JB-all brier hob', 'JB-all ece', 'JB-all ece hob', 'REAL-label brier', 'REAL-label brier hob',
        'REAL-label ece', 'REAL-label ece hob']

if __name__ == '__main__':
    sp = sys.argv[1]
    res = json.load(open(sp)) if os.path.exists(sp) else {}
    for f in sys.argv[2:]:
        res[os.path.basename(f).replace('.json', '')] = summarize(json.load(open(f)))
    json.dump(res, open(sp, 'w'), indent=1)
    names = [os.path.basename(f).replace('.json', '') for f in sys.argv[2:]] or list(res)
    print(f'{"":26s}' + ''.join(f'{n[:11]:>12s}' for n in names))
    for k in KEYS:
        vals = [res[n].get(k) for n in names]
        print(f'{k:26s}' + ''.join(f'{v:12.3f}' if isinstance(v, (int, float)) and v is not None else f'{str(v):>12s}' for v in vals))
    print(f'{"PASS (strict bar)":26s}' + ''.join(f'{str(res[n]["PASS"]):>12s}' for n in names))
    print(f'{"PASS (relaxed bar)":26s}' + ''.join(f'{str(res[n]["PASS_relaxed"]):>12s}' for n in names))
    for n in names:
        print(n, 'bar', res[n]['bar'], 'relaxed', res[n]['bar_relaxed'], 'REAL-label by len', res[n].get('REAL-label short<2000'), res[n].get('REAL-label long>=2000'))
        print('   CF-probe by kind (model, hobson, n):', res[n]['CF-probe by kind'])
        print('   CF by kind:', res[n]['CF by kind'])
