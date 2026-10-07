"""J1 scorer (laptop, pure python): evalkit suites for a preds file vs hobson, plus
- JB-all / JB-hard paired McNemar (exact binomial on discordant tasks) vs hobson;
- Brier (multi-class, 0..2) and top-label ECE (10 equal-width bins) on every suite with ground truth (JB-all, JB-hard, REAL-label, CF, CF-probe);
- option-order sensitivity from a .rot.json (and rot_hob.json): share of decisions unchanged, mean |dp| of the original argmax label, mean TV.
usage: python3 score_j1.py PREDS.json [ROT.json] [--hobrot rot_hob.json] [--out scores.json] [--name tag]"""
import sys, os, json, math, argparse, collections
sys.path.insert(0, '~/decider2/evalkit')
import numpy as np
import evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('preds'); ap.add_argument('rot', nargs='?', default='')
ap.add_argument('--hobrot', default=''); ap.add_argument('--out', default=''); ap.add_argument('--name', default='model')
a = ap.parse_args()
P = json.load(open(a.preds))


def norm(p):
    s = sum(p.values()); return {k: v / s for k, v in p.items()}


def arg(p): return max(p, key=p.get)


def binom_two_sided(k, n):
    if n == 0: return 1.0
    from math import comb
    pk = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    obs = pk[k]
    return min(1.0, sum(x for x in pk if x <= obs + 1e-15))


def labelled(suite, preds):
    """(pred dist, hobson dist, expected) per question with ground truth"""
    out = []
    if suite in ('CF', 'CF-probe'):
        hp = EK._as_preds(suite, 'hobson')
        for pr in EK._pairs(suite):
            for side, e in (('a', 'ea'), ('b', 'eb')):
                iid = pr[side]; q = pr['q']
                if iid in preds and q in preds[iid] and iid in hp and q in hp[iid]:
                    out.append((norm(preds[iid][q]), norm(hp[iid][q]), pr[e], (iid, q)))
        return out
    for r in EK._qrows(suite, preds):
        if r['exp'] is not None and r['pred'] is not None and r['hob'] is not None:
            out.append((r['pred'], r['hob'], r['exp'], (r['id'], r['q'])))
    return out


def brier(rows, which):
    return float(np.mean([sum((p.get(k, 0) - (1.0 if k == e else 0.0)) ** 2 for k in set(p) | {e}) for p, e in ((r[which], r[2]) for r in rows)]))


def ece(rows, which, bins=10):
    conf = np.array([max(r[which].values()) for r in rows]); ok = np.array([arg(r[which]) == r[2] for r in rows], dtype=float)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi) if b else (conf >= lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


res = {'name': a.name, 'preds': a.preds, 'suites': {}}
for s in ['JB-all', 'JB-hard', 'JB-long', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']:
    sc = EK.score(s, P, baselines=False)
    res['suites'][s] = {'model': sc['model'], 'hobson': sc.get('hobson', {})}
for s in ['JB-all', 'JB-hard', 'REAL-label', 'CF', 'CF-probe']:
    rows = labelled(s, P)
    if not rows: continue
    mo = [arg(r[0]) == r[2] for r in rows]; ho = [arg(r[1]) == r[2] for r in rows]
    b = sum(1 for x, y in zip(mo, ho) if x and not y); c = sum(1 for x, y in zip(mo, ho) if y and not x)
    res['suites'][s]['calib'] = dict(n=len(rows), acc_model=float(np.mean(mo)), acc_hobson=float(np.mean(ho)),
                                     brier_model=brier(rows, 0), brier_hobson=brier(rows, 1), ece_model=ece(rows, 0), ece_hobson=ece(rows, 1),
                                     mcnemar_model_only=b, mcnemar_hobson_only=c, mcnemar_p=binom_two_sided(min(b, c), b + c))
# REAL-label by state length (LONG has no labels; this is the length trend on labelled real traffic)
lab = {r['id']: r for r in EK.load_suite('REAL-label')}
bk = collections.defaultdict(lambda: [[], []])
for r in EK._qrows('REAL-label', P):
    if r['exp'] is None or r['pred'] is None or r['hob'] is None: continue
    n = r['item'].get('n_state_tok', 0); g = '<1k' if n < 1000 else '1-2k' if n < 2000 else '2-4k'
    bk[g][0].append(arg(r['pred']) == r['exp']); bk[g][1].append(arg(r['hob']) == r['exp'])
res['suites']['REAL-label']['by_len'] = {g: dict(n=len(v[0]), model=float(np.mean(v[0])), hobson=float(np.mean(v[1]))) for g, v in sorted(bk.items())}
for s in ('CF', 'CF-probe'):
    res['suites'][s]['by_kind'] = EK.breakdown(s, P, 'kind', rows=('hobson',))
    res['suites'][s]['by_len'] = EK.breakdown(s, P, 'len', rows=('hobson',))


def rot_stats(R):
    same = []; dp = []; tv = []; per = collections.defaultdict(list)
    for iid, qs in R.items():
        for q, d in qs.items():
            base = norm(d['r0']); a0 = arg(base)
            for k, v in d.items():
                if k == 'r0': continue
                v = norm(v); same.append(arg(v) == a0); dp.append(abs(v[a0] - base[a0]))
                tv.append(0.5 * sum(abs(v.get(x, 0) - base.get(x, 0)) for x in base))
                per[len(base)].append(arg(v) == a0)
    return dict(n=len(same), unchanged=float(np.mean(same)), mean_abs_dp=float(np.mean(dp)), mean_tv=float(np.mean(tv)),
                unchanged_by_K={str(k): (len(v), round(float(np.mean(v)), 4)) for k, v in sorted(per.items())})


if a.rot and os.path.exists(a.rot):
    res['rot_model'] = rot_stats(json.load(open(a.rot)))
if a.hobrot and os.path.exists(a.hobrot):
    res['rot_hobson'] = rot_stats(json.load(open(a.hobrot)))

# ---- print
f = lambda v: '  -  ' if v is None or (isinstance(v, float) and math.isnan(v)) else (f'{v:.3f}' if isinstance(v, float) else str(v))
print(f'== {a.name}: {a.preds}')
for s, d in res['suites'].items():
    m, h = d['model'], d['hobson']
    keys = [k for k in ('acc', 'acc_consensus', 'agree', 'agree_sd', 'tv', 'flip', 'flip_given_hobson', 'dir', 'change', 'both_right') if k in m]
    print(f'{s:11s} ' + '  '.join(f'{k} {f(m.get(k))} (hob {f(h.get(k))})' for k in keys))
    if 'calib' in d:
        c = d['calib']
        print(f'{"":11s} n {c["n"]} acc {c["acc_model"]:.3f} vs {c["acc_hobson"]:.3f} | McNemar {c["mcnemar_model_only"]}/{c["mcnemar_hobson_only"]} p {c["mcnemar_p"]:.3f} | '
              f'Brier {c["brier_model"]:.3f} vs {c["brier_hobson"]:.3f} | ECE {c["ece_model"]:.3f} vs {c["ece_hobson"]:.3f}')
    if s == 'REAL-label' and 'by_len' in d:
        print(f'{"":11s} by state len: ' + '; '.join(f'{g} {v["model"]:.3f}/{v["hobson"]:.3f} (n{v["n"]})' for g, v in d['by_len'].items()))
    if 'by_kind' in d:
        print(f'{"":11s} by kind: ' + '; '.join(f'{g} {f(v.get("model"))}/{f(v.get("hobson"))} (n{v["n"]})' for g, v in d['by_kind'].items()))
        print(f'{"":11s} by len : ' + '; '.join(f'{g} {f(v.get("model"))}/{f(v.get("hobson"))} (n{v["n"]})' for g, v in d['by_len'].items()))
for k in ('rot_model', 'rot_hobson'):
    if k in res: print(k, res[k])
if a.out: json.dump(res, open(a.out, 'w'), indent=1)
