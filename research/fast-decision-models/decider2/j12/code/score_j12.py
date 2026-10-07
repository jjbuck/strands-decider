"""J12 laptop-side scoring (pure python + numpy): preds jsonl(s) -> every evalkit suite + breakdowns + McNemar vs hobson + Brier.
python3 score_j12.py NAME preds1.jsonl [preds2.jsonl ...]  -> j12/scores_NAME.json and a printed summary"""
import sys, json, os, math, collections
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK
import numpy as np

name = sys.argv[1]; files = sys.argv[2:]
preds = collections.defaultdict(dict)
for fn in files:
    for l in open(fn):
        r = json.loads(l); preds[r['id']][r['q']] = r['p']
preds = dict(preds)


def binom_two_sided(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def mcnemar(suite):
    refs = EK.load_refs(suite)
    b = c = n = 0; acc_m = acc_h = 0
    fam = collections.defaultdict(lambda: [0, 0, 0])
    for it in EK.load_suite(suite):
        for q, exp in (it.get('expected') or {}).items():
            if it['id'] not in preds or q not in preds[it['id']]: continue
            pm = preds[it['id']][q]; ph = refs[it['id']]['hobson'][q]
            am = max(pm, key=pm.get) == exp; ah = max(ph, key=ph.get) == exp
            n += 1; acc_m += am; acc_h += ah
            if am and not ah: b += 1
            if ah and not am: c += 1
            f = fam[it.get('family', '?')]; f[0] += 1; f[1] += am; f[2] += ah
    return dict(n=n, acc=acc_m / max(1, n), acc_hobson=acc_h / max(1, n), model_only=b, hobson_only=c, p=binom_two_sided(b, c),
                fam={k: dict(n=v[0], model=v[1], hobson=v[2]) for k, v in sorted(fam.items())})


def brier(suite):
    """multi-class Brier (sum over options of (p - onehot)^2), model and hobson, on labelled items"""
    refs = EK.load_refs(suite) if suite != 'REAL-label' else EK.load_refs('REAL-agree')
    bm = []; bh = []; conf = []
    for it in EK.load_suite(suite):
        for q, exp in (it.get('expected') or {}).items():
            if it['id'] not in preds or q not in preds[it['id']]: continue
            pm = preds[it['id']][q]; ph = refs[it['id']]['hobson'][q]
            bm.append(sum((pm[k] - (k == exp)) ** 2 for k in pm)); bh.append(sum((ph[k] - (k == exp)) ** 2 for k in ph))
            top = max(pm, key=pm.get); conf.append((pm[top], top == exp))
    ece = 0.0
    if conf:
        bins = collections.defaultdict(list)
        for p, ok in conf: bins[min(9, int(p * 10))].append((p, ok))
        ece = sum(len(v) * abs(np.mean([x[0] for x in v]) - np.mean([x[1] for x in v])) for v in bins.values()) / len(conf)
    return dict(n=len(bm), brier=float(np.mean(bm)) if bm else None, brier_hobson=float(np.mean(bh)) if bh else None, ece=ece)


out = {}
for s in EK.SUITES:
    try:
        sc = EK.score(s, preds, baselines=False)
        out[s] = sc.get('model', sc)
        out[s + '_hobson'] = sc.get('hobson')
    except Exception as e:
        out[s] = {'err': repr(e)[:200]}
for s in ('JB-all', 'JB-hard'):
    out['mcnemar_' + s] = mcnemar(s)
for s in ('JB-all', 'REAL-label', 'CF', 'CF-probe'):
    out['brier_' + s] = brier(s)
for s, by in (('CF', 'kind'), ('CF-probe', 'kind'), ('CF', 'len'), ('CF', 'pos'), ('CF-probe', 'domain')):
    try:
        out[f'bd_{s}_{by}'] = EK.breakdown(s, preds, by=by, rows=('hobson',))
    except Exception as e:
        out[f'bd_{s}_{by}'] = {'err': repr(e)[:200]}
json.dump(out, open(f'~/decider2/j12/scores_{name}.json', 'w'), indent=1, default=str)
print('coverage items', len(preds))
for s in EK.SUITES:
    m = out[s]; h = out.get(s + '_hobson') or {}
    keys = [k for k in ('acc', 'flip', 'agree', 'agree_sd', 'dir', 'change', 'both_right', 'coverage') if isinstance(m, dict) and k in m]
    print(f'{s:11s}', '  '.join(f'{k} {m[k]:.3f}' + (f' (h {h[k]:.3f})' if k in h and isinstance(h[k], float) else '') for k in keys))
for s in ('JB-all', 'JB-hard'):
    mc = out['mcnemar_' + s]
    print(s, 'acc %.3f vs hobson %.3f  model-only %d hobson-only %d  p %.3f' % (mc['acc'], mc['acc_hobson'], mc['model_only'], mc['hobson_only'], mc['p']))
fam = out['mcnemar_JB-hard']['fam']
print('JB-hard families:', ', '.join(f"{k} {v['model']}/{v['n']} (h {v['hobson']})" for k, v in fam.items()))
for s in ('JB-all', 'REAL-label', 'CF', 'CF-probe'):
    b = out['brier_' + s]; print('brier', s, b)
for k in ('bd_CF_kind', 'bd_CF-probe_kind', 'bd_CF_len'):
    print(k); print(json.dumps(out[k], default=str)[:3000])
