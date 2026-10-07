"""score J5 weight-only preds (laptop, pure python): every suite vs hobson refs + flips vs the in-runtime bf16 floor (paired McNemar)
+ JB-all / JB-hard McNemar vs hobson + Brier / ECE on labelled suites + the function-preserving fidelity bar."""
import sys, os, json, math
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import numpy as np, evalkit as EK
D = os.path.expanduser('~/decider2/j5/preds')
tags = sys.argv[1].split(',')


def load(tag):
    P = {}
    for l in open(f'{D}/preds_{tag}.jsonl'):
        r = json.loads(l); P.setdefault(r['id'], {})[r['q']] = r['probs']
    return P


def arg(p): return max(p, key=p.get)


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def brier_ece(P, suite):
    br = []; conf = []; corr = []
    for it in EK.load_suite(suite):
        for q in it['questions']:
            e = (it.get('expected') or {}).get(q)
            if e is None or it['id'] not in P or q not in P[it['id']]: continue
            p = EK._norm(P[it['id']][q])
            br.append(sum((p.get(k, 0) - (1.0 if k == e else 0.0)) ** 2 for k in set(p) | {e}))
            a = arg(p); conf.append(p[a]); corr.append(a == e)
    conf = np.array(conf); corr = np.array(corr, dtype=float); ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf > lo) & (conf <= lo + 0.1)
        if m.any(): ece += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(np.mean(br)), float(ece), len(br)


def hobson_preds(suite):
    return EK._as_preds(suite, 'hobson')


SU = ['JB-all', 'JB-hard', 'JB-long', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']
res = {}
base = load('bf16') if os.path.exists(f'{D}/preds_bf16.jsonl') else None
hob_all = {}
for s in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe'):
    hob_all.update(hobson_preds(s))
for tag in tags:
    if not os.path.exists(f'{D}/preds_{tag}.jsonl'): print('missing', tag); continue
    P = load(tag); r = {}
    for s in SU:
        sc = EK.score(s, P, baselines=False); r[s] = sc['model']; r[s + '_hob'] = sc['hobson']
    # flips vs hobson and vs the in-runtime bf16, all questions and REAL-agree, with a paired McNemar on agreement-with-hobson
    def flips(ids_q, ref):
        return [arg(P[i][q]) != arg(ref[i][q]) for i, q in ids_q if i in ref and q in ref[i] and i in P and q in P[i]]
    allq = [(i, q) for i in P for q in P[i]]
    ra = [(it['id'], q) for it in EK.load_suite('REAL-agree') for q in it['questions']]
    r['flips_hob_all'] = float(np.mean(flips(allq, hob_all)))
    r['flips_hob_real'] = float(np.mean(flips(ra, hob_all)))
    if base is not None:
        r['flips_bf16_all'] = float(np.mean(flips(allq, base)))
        r['flips_bf16_real'] = float(np.mean(flips(ra, base)))
        b = c = 0
        for i, q in allq:
            if i not in hob_all or q not in hob_all[i] or i not in base: continue
            h = arg(hob_all[i][q]); fa = arg(P[i][q]) == h; fb = arg(base[i][q]) == h
            b += fb and not fa; c += fa and not fb
        r['mcnemar_vs_bf16'] = dict(bf16_only=b, fmt_only=c, p=mcnemar(b, c))
        tv = [0.5 * sum(abs(P[i][q][k] - base[i][q][k]) for k in P[i][q]) for i, q in allq if i in base and q in base[i]]
        r['tv_bf16'] = float(np.mean(tv))
    for s in ('JB-all', 'JB-hard'):
        refs = EK.load_refs(s); b = c = 0
        for it in EK.load_suite(s):
            for q in it['questions']:
                e = it['expected'][q]; h = refs[it['id']]['hobson'][q]
                hr = arg(EK._norm(h)) == e; pr = arg(EK._norm(P[it['id']][q])) == e
                b += hr and not pr; c += pr and not hr
        r['mcnemar_' + s] = dict(hobson_only=b, model_only=c, p=mcnemar(b, c))
    for s in ('JB-all', 'REAL-label'):
        r['brier_' + s], r['ece_' + s], _ = brier_ece(P, s)
    res[tag] = r
# hobson's own Brier / ECE
hb = {}
for s in ('JB-all', 'REAL-label'):
    hp = hobson_preds('JB-all' if s == 'JB-all' else 'REAL-agree'); hb[s] = brier_ece(hp, s)
json.dump(dict(res=res, hobson_brier=hb), open(os.path.expanduser('~/decider2/j5/scores_wq.json'), 'w'), indent=1)
rows = [('JB-all acc', 'JB-all', 'acc'), ('JB-hard acc', 'JB-hard', 'acc'), ('JB-long agree', 'JB-long', 'agree'), ('REAL-agree agree', 'REAL-agree', 'agree'),
        ('REAL agree_sd', 'REAL-agree', 'agree_sd'), ('LONG agree', 'LONG', 'agree'), ('LONG agree_sd', 'LONG', 'agree_sd'), ('CF acc', 'CF', 'acc'), ('CF flip', 'CF', 'flip'),
        ('CF fgh', 'CF', 'flip_given_hobson'), ('CF dir', 'CF', 'dir'), ('CF-probe acc', 'CF-probe', 'acc'), ('CF-probe flip', 'CF-probe', 'flip'), ('CF-probe fgh', 'CF-probe', 'flip_given_hobson'),
        ('SHUF change', 'SHUF', 'change'), ('REAL-label acc', 'REAL-label', 'acc')]
T = list(res)
print('| metric | hobson | ' + ' | '.join(T) + ' |'); print('|---|---|' + '---|' * len(T))
for nm, s, k in rows:
    h = res[T[0]][s + '_hob'].get(k, float('nan'))
    print(f'| {nm} | {h:.3f} | ' + ' | '.join(f"{res[t][s].get(k, float('nan')):.3f}" for t in T) + ' |')
print('| flips vs hobson, all 3227 q | 0 | ' + ' | '.join(f"{res[t]['flips_hob_all']:.2%}" for t in T) + ' |')
print('| flips vs hobson, REAL-agree | 0 | ' + ' | '.join(f"{res[t]['flips_hob_real']:.2%}" for t in T) + ' |')
if base is not None:
    print('| flips vs in-runtime bf16, all q | - | ' + ' | '.join(f"{res[t]['flips_bf16_all']:.2%}" for t in T) + ' |')
    print('| TV vs in-runtime bf16 | - | ' + ' | '.join(f"{res[t]['tv_bf16']:.4f}" for t in T) + ' |')
    print('| McNemar vs bf16 on hobson-agreement (bf16-only/fmt-only, p) | - | ' + ' | '.join(f"{res[t]['mcnemar_vs_bf16']['bf16_only']}/{res[t]['mcnemar_vs_bf16']['fmt_only']} p={res[t]['mcnemar_vs_bf16']['p']:.3f}" for t in T) + ' |')
for s in ('JB-all', 'JB-hard'):
    print(f'| McNemar {s} vs hobson (hob-only/model-only, p) | - | ' + ' | '.join(f"{res[t]['mcnemar_' + s]['hobson_only']}/{res[t]['mcnemar_' + s]['model_only']} p={res[t]['mcnemar_' + s]['p']:.2f}" for t in T) + ' |')
for s in ('JB-all', 'REAL-label'):
    print(f'| Brier / ECE {s} | {hb[s][0]:.3f} / {hb[s][1]:.3f} | ' + ' | '.join(f"{res[t]['brier_' + s]:.3f} / {res[t]['ece_' + s]:.3f}" for t in T) + ' |')
