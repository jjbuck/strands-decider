"""score H2 runtime preds (laptop): every suite next to hobson refs + flips vs H2's own bf16 runtime + JB-hard McNemar + low-bit kill check."""
import sys, os, json, math, glob
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import numpy as np, evalkit as EK
D = os.path.expanduser('~/decider2/h2/preds')
tags = sys.argv[1].split(',') if len(sys.argv) > 1 else ['bf16', 'w8a8', 'w4a8', 'w4a4']
def load(tag):
    P = {}; L = {}
    for l in open(f'{D}/preds_{tag}.jsonl'):
        r = json.loads(l); P.setdefault(r['id'], {})[r['q']] = r['probs']; L[(r['id'], r['q'])] = r['logits']
    return P, L
def arg(p): return max(p, key=p.get)
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
SU = ['JB-hard', 'JB-long', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']
res = {}
base = load('bf16') if os.path.exists(f'{D}/preds_bf16.jsonl') else None
for tag in tags:
    if not os.path.exists(f'{D}/preds_{tag}.jsonl'): continue
    P, L = load(tag); r = {}
    for s in SU:
        sc = EK.score(s, P, baselines=False); r[s] = sc['model']; r[s + '_hob'] = sc['hobson']
    # vs H2 bf16 runtime
    if base is not None:
        bp, bl = base; fl = []; tv = []; cs = []
        for (iid, q), lg in L.items():
            if (iid, q) not in bl: continue
            p = P[iid][q]; pb = bp[iid][q]
            fl.append(arg(p) != arg(pb)); tv.append(0.5 * sum(abs(p[k] - pb[k]) for k in p))
            a = np.array(lg) - np.mean(lg); b = np.array(bl[(iid, q)]) - np.mean(bl[(iid, q)])
            if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0: cs.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
        r['vs_bf16'] = dict(n=len(fl), flips=float(np.mean(fl)), tv=float(np.mean(tv)), logit_cos=float(np.mean(cs)))
        # flips vs bf16 on REAL-agree only
        ra = [it['id'] for it in EK.load_suite('REAL-agree')]
        f2 = [arg(P[i][q]) != arg(bp[i][q]) for i in ra for q in P.get(i, {}) if i in bp and q in bp[i]]
        r['vs_bf16']['REAL_flips'] = float(np.mean(f2))
    # McNemar JB-hard vs hobson refs
    refs = EK.load_refs('JB-hard'); b = c = 0
    for it in EK.load_suite('JB-hard'):
        for q in it['questions']:
            e = it['expected'][q]; h = refs[it['id']]['hobson'][q]; p = P[it['id']][q]
            hr = arg(EK._norm(h)) == e; pr = arg(EK._norm(p)) == e
            b += hr and not pr; c += pr and not hr
    r['mcnemar_jbhard'] = dict(hobson_only=b, model_only=c, p=mcnemar(b, c))
    res[tag] = r
json.dump(res, open(os.path.expanduser('~/decider2/h2/scores.json'), 'w'), indent=1)
# table
rows = [('JB-hard acc', 'JB-hard', 'acc'), ('JB-long agree', 'JB-long', 'agree'), ('REAL-agree agree', 'REAL-agree', 'agree'), ('REAL agree_sd', 'REAL-agree', 'agree_sd'),
        ('LONG agree', 'LONG', 'agree'), ('LONG agree_sd', 'LONG', 'agree_sd'), ('CF flip', 'CF', 'flip'), ('CF fgh', 'CF', 'flip_given_hobson'), ('CF dir', 'CF', 'dir'),
        ('CF-probe flip', 'CF-probe', 'flip'), ('CF-probe fgh', 'CF-probe', 'flip_given_hobson'), ('SHUF change', 'SHUF', 'change'), ('REAL-label acc', 'REAL-label', 'acc')]
print('| metric | hobson ref | ' + ' | '.join(res) + ' |'); print('|---|---|' + '---|' * len(res))
for nm, s, k in rows:
    h = res[tags[0]][s + '_hob'].get(k, float('nan'))
    print(f'| {nm} | {h:.3f} | ' + ' | '.join(f"{res[t][s].get(k, float('nan')):.3f}" for t in res) + ' |')
print('| REAL flips vs hobson | 0 | ' + ' | '.join(f"{1 - res[t]['REAL-agree']['agree']:.2%}" for t in res) + ' |')
if base is not None:
    print('| REAL flips vs H2 bf16 | - | ' + ' | '.join(f"{res[t]['vs_bf16']['REAL_flips']:.2%}" for t in res) + ' |')
    print('| all-q flips vs H2 bf16 | - | ' + ' | '.join(f"{res[t]['vs_bf16']['flips']:.2%}" for t in res) + ' |')
    print('| logit cos vs H2 bf16 | - | ' + ' | '.join(f"{res[t]['vs_bf16']['logit_cos']:.4f}" for t in res) + ' |')
print('| JB-hard McNemar p (hob-only/model-only) | - | ' + ' | '.join(f"{res[t]['mcnemar_jbhard']['p']:.2f} ({res[t]['mcnemar_jbhard']['hobson_only']}/{res[t]['mcnemar_jbhard']['model_only']})" for t in res) + ' |')
for t in res:
    r = res[t]; ok = (1 - r['REAL-agree']['agree'] < 0.005) and r['CF']['flip_given_hobson'] >= 0.97 and r['CF-probe']['flip_given_hobson'] >= 0.97 and not (r['mcnemar_jbhard']['p'] < 0.05 and r['mcnemar_jbhard']['hobson_only'] > r['mcnemar_jbhard']['model_only'])
    print(t, 'LOW-BIT KILL CRITERION:', 'PASS' if ok else 'FAIL')
