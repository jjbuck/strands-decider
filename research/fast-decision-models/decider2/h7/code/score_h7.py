"""H7 scoring (laptop): python3 score_h7.py preds/a.json [preds/b.json ...] -> one summary row per file + the architecture co-design bar.
Also writes scores.json (merged)."""
import sys, os, json, math
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import evalkit as EK
import numpy as np


def mcnemar(rows):
    b = sum(1 for r in rows if r[0] and not r[1]); c = sum(1 for r in rows if r[1] and not r[0]); n = b + c
    if n == 0: return 1.0, b, c
    k = min(b, c); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p), b, c


def summarize(preds):
    preds = EK._flat(preds, None); out = {}
    jr = EK._qrows('JB-hard', preds)
    ok = [(EK._arg(EK._norm(r['pred'])) == r['exp'] if r['pred'] else False, EK._arg(r['hob']) == r['exp']) for r in jr]
    out['JB-hard acc'] = float(np.mean([o[0] for o in ok])); out['JB-hard hob'] = float(np.mean([o[1] for o in ok]))
    p, b, c = mcnemar(ok); out['JB-hard mcnemar p'] = p; out['JB-hard model-only/hob-only'] = f'{b}/{c}'
    for s in ('JB-long', 'REAL-agree', 'LONG'):
        m = EK.score(s, preds, baselines=False)['model']
        out[f'{s} agree'] = m.get('agree'); out[f'{s} agree_sd'] = m.get('agree_sd'); out[f'{s} cov'] = m.get('coverage')
    for s in ('CF', 'CF-probe'):
        m = EK.score(s, preds, baselines=False)['model']
        for k in ('acc', 'flip', 'flip_given_hobson', 'dir', 'agree'):
            out[f'{s} {k}'] = m.get(k)
        bd = EK.breakdown(s, preds, 'kind', rows=('hobson',))
        out[f'{s} by kind'] = {g: (round(d['model'], 3) if d.get('model') is not None else None, round(d['hobson'], 3) if d.get('hobson') is not None else None, d['n']) for g, d in bd.items()}
    out['SHUF both_right'] = EK.score('SHUF', preds, baselines=False)['model'].get('both_right')
    out['REAL-label acc'] = EK.score('REAL-label', preds, baselines=False)['model'].get('acc')
    bar = dict(real_sd=(out['REAL-agree agree_sd'] or 0) >= 0.95, long_sd=(out['LONG agree_sd'] or 0) >= 0.95,
               cf_fgh=(out['CF flip_given_hobson'] or 0) >= 0.90, cfp_fgh=(out['CF-probe flip_given_hobson'] or 0) >= 0.90,
               jb_flag=(out['JB-hard acc'] - out['JB-hard hob']) < -0.03)
    out['bar'] = bar; out['PASS'] = bar['real_sd'] and bar['long_sd'] and bar['cf_fgh'] and bar['cfp_fgh']
    return out


KEYS = ['JB-hard acc', 'JB-hard mcnemar p', 'JB-long agree', 'REAL-agree agree', 'REAL-agree agree_sd', 'LONG agree', 'LONG agree_sd', 'CF acc', 'CF flip',
        'CF flip_given_hobson', 'CF-probe acc', 'CF-probe flip', 'CF-probe flip_given_hobson', 'SHUF both_right', 'REAL-label acc']

if __name__ == '__main__':
    res = {}
    sp = os.path.expanduser('~/decider2/h7/scores.json')
    if os.path.exists(sp): res = json.load(open(sp))
    hob = summarize(EK._as_preds_all() if hasattr(EK, '_as_preds_all') else {k: v for s in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe') for k, v in EK._as_preds(s, 'hobson').items()})
    res['hobson'] = hob
    for f in sys.argv[1:]:
        nm = os.path.basename(f).replace('.json', '')
        res[nm] = summarize(json.load(open(f)))
    json.dump(res, open(sp, 'w'), indent=1)
    names = list(res)
    print(f'{"":28s}' + ''.join(f'{n[:14]:>15s}' for n in names))
    for k in KEYS:
        print(f'{k:28s}' + ''.join(f'{(res[n].get(k) if isinstance(res[n].get(k), (int, float)) else float("nan")):15.3f}' for n in names))
    print(f'{"PASS (co-design bar)":28s}' + ''.join(f'{str(res[n]["PASS"]):>15s}' for n in names))
    for n in names[1:]:
        if n in [x.replace('.json', '') for x in map(os.path.basename, sys.argv[1:])]:
            print(n, 'CF-probe by kind (model, hobson, n):', res[n]['CF-probe by kind'])
            print(n, 'CF by kind:', res[n]['CF by kind'])
