"""H6 laptop scorer. python3 h6score.py TAG[=REFTAG] ...   (preds in ~/decider2/h6/preds/TAG.jsonl)
Every suite vs bf16 hobson refs (EK.score), JB-hard McNemar vs hobson, and, if REFTAG is given, the same low-bit numbers against that
reference run (flips, CF / CF-probe 'fg_ref' = of the pairs the reference tracks, the fraction the model tracks, McNemar vs the reference).
Writes ~/decider2/h6/scores.json (merged) and prints a markdown table."""
import sys, os, json, math
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import numpy as np, evalkit as EK
D = os.path.expanduser('~/decider2/h6/preds')
SU = ['JB-hard', 'JB-long', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']
am = lambda d: max(d, key=d.get)


def load(tag):
    P = {}
    for l in open(f'{D}/{tag}.jsonl'):
        r = json.loads(l); P.setdefault(r['id'], {})[r['q']] = r['probs']
    return P


def mcn(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def jb_mcnemar(P, R):
    b = c = 0
    for it in EK.load_suite('JB-hard'):
        for q, e in (it.get('expected') or {}).items():
            p = P.get(it['id'], {}).get(q); h = R.get(it['id'], {}).get(q)
            if p is None or h is None: continue
            hr = am(EK._norm(h)) == e; pr = am(EK._norm(p)) == e
            b += hr and not pr; c += pr and not hr
    return dict(ref_only=b, model_only=c, p=round(mcn(b, c), 3))


def vs_ref(P, R):
    out = {}
    for s in ('REAL-agree', 'LONG'):
        ids = {it['id'] for it in EK.load_suite(s)}
        f = [am(EK._norm(P[i][q])) != am(EK._norm(R[i][q])) for i in ids for q in P.get(i, {}) if q in R.get(i, {})]
        out[s + '_flips'] = float(np.mean(f)) if f else float('nan'); out[s + '_n'] = len(f)
    for s in ('CF', 'CF-probe'):
        tr = []
        for pr in EK._pairs(s):
            q = pr['q']
            try:
                ro = am(EK._norm(R[pr['a']][q])) == pr['ea'] and am(EK._norm(R[pr['b']][q])) == pr['eb']
                po = am(EK._norm(P[pr['a']][q])) == pr['ea'] and am(EK._norm(P[pr['b']][q])) == pr['eb']
            except KeyError: continue
            if ro: tr.append(po)
        out[s + '_fg_ref'] = float(np.mean(tr)) if tr else float('nan'); out[s + '_n_ref_tracked'] = len(tr)
    # TV to hobson (model vs ref) and paired agreement-with-hobson test on REAL-agree + LONG
    for s in ('REAL-agree', 'LONG'):
        ids = {it['id'] for it in EK.load_suite(s)}; tvm = []; tvr = []; b = c = 0
        for i in ids:
            for q in P.get(i, {}):
                if q not in R.get(i, {}) or q not in HOB.get(i, {}): continue
                p_, r_, h_ = EK._norm(P[i][q]), EK._norm(R[i][q]), EK._norm(HOB[i][q])
                tvm.append(EK._tv(p_, h_)); tvr.append(EK._tv(r_, h_))
                pm, rm = am(p_) == am(h_), am(r_) == am(h_); b += rm and not pm; c += pm and not rm
        out[s + '_tv_hob'] = float(np.mean(tvm)) if tvm else float('nan'); out[s + '_tv_hob_ref'] = float(np.mean(tvr)) if tvr else float('nan')
        out[s + '_paired'] = dict(ref_only=b, model_only=c, p=round(mcn(b, c), 3))
    allf = [am(EK._norm(P[i][q])) != am(EK._norm(R[i][q])) for i in P for q in P[i] if q in R.get(i, {})]
    out['all_flips'] = float(np.mean(allf)); out['all_n'] = len(allf)
    out['jb_mcnemar'] = jb_mcnemar(P, R)
    return out


HOB = {}
for s in SU:
    if s != 'SHUF': HOB.update({k: dict(v) for k, v in EK._as_preds(s, 'hobson').items()})
res = json.load(open(os.path.expanduser('~/decider2/h6/scores.json'))) if os.path.exists(os.path.expanduser('~/decider2/h6/scores.json')) else {}
tags = []
for arg in sys.argv[1:]:
    tag, ref = (arg.split('=') + [None])[:2]
    if not os.path.exists(f'{D}/{tag}.jsonl'): print('missing', tag); continue
    P = load(tag); r = {}
    for s in SU:
        sc = EK.score(s, P, baselines=False); r[s] = sc['model']; r[s + '_hob'] = sc['hobson']
    r['jb_mcnemar_hob'] = jb_mcnemar(P, HOB)
    r['n'] = sum(len(v) for v in P.values())
    if ref and os.path.exists(f'{D}/{ref}.jsonl'):
        r['ref'] = ref; r['vs_ref'] = vs_ref(P, load(ref))
    lb = (1 - r['REAL-agree']['agree'] < 0.005) and r['CF'].get('flip_given_hobson', 0) >= 0.97 and r['CF-probe'].get('flip_given_hobson', 0) >= 0.97 \
        and not (r['jb_mcnemar_hob']['p'] < 0.05 and r['jb_mcnemar_hob']['ref_only'] > r['jb_mcnemar_hob']['model_only'])
    cd = r['REAL-agree'].get('agree_sd', 0) >= 0.95 and r['LONG'].get('agree_sd', 0) >= 0.95 and r['CF'].get('flip_given_hobson', 0) >= 0.90 \
        and r['CF-probe'].get('flip_given_hobson', 0) >= 0.90
    r['lowbit_pass_vs_hobson'] = bool(lb); r['codesign_pass'] = bool(cd); r['jb_hard_drop_pts'] = round(100 * (r['JB-hard_hob']['acc'] - r['JB-hard']['acc']), 1)
    if 'vs_ref' in r:
        v = r['vs_ref']
        r['lowbit_pass_vs_ref'] = bool(v['REAL-agree_flips'] < 0.005 and v['CF_fg_ref'] >= 0.97 and v['CF-probe_fg_ref'] >= 0.97
                                       and not (v['jb_mcnemar']['p'] < 0.05 and v['jb_mcnemar']['ref_only'] > v['jb_mcnemar']['model_only']))
    res[tag] = r; tags.append(tag)
json.dump(res, open(os.path.expanduser('~/decider2/h6/scores.json'), 'w'), indent=1)
rows = [('n questions', None, 'n'), ('JB-hard acc', 'JB-hard', 'acc'), ('JB-long agree', 'JB-long', 'agree'), ('REAL flips vs hobson', 'REAL-agree', 'flips'),
        ('REAL agree_sd', 'REAL-agree', 'agree_sd'), ('LONG agree', 'LONG', 'agree'), ('LONG agree_sd', 'LONG', 'agree_sd'), ('CF acc', 'CF', 'acc'), ('CF flip', 'CF', 'flip'),
        ('CF fgh', 'CF', 'flip_given_hobson'), ('CF-probe acc', 'CF-probe', 'acc'), ('CF-probe flip', 'CF-probe', 'flip'), ('CF-probe fgh', 'CF-probe', 'flip_given_hobson'),
        ('SHUF both_right', 'SHUF', 'both_right'), ('REAL-label acc', 'REAL-label', 'acc')]
print('| metric | hobson | ' + ' | '.join(tags) + ' |'); print('|---|---|' + '---|' * len(tags))
for nm, s, k in rows:
    if s is None: print(f'| {nm} | 3227 | ' + ' | '.join(str(res[t]['n']) for t in tags) + ' |'); continue
    if k == 'flips':
        print(f'| {nm} | 0 | ' + ' | '.join(f"{1 - res[t][s]['agree']:.2%}" for t in tags) + ' |'); continue
    h = res[tags[0]][s + '_hob'].get(k, float('nan')) if tags else float('nan')
    print(f'| {nm} | {h:.3f} | ' + ' | '.join(f"{res[t][s].get(k, float('nan')):.3f}" for t in tags) + ' |')
print('| JB-hard McNemar vs hobson p (hob-only/model-only) | - | ' + ' | '.join(f"{res[t]['jb_mcnemar_hob']['p']:.2f} ({res[t]['jb_mcnemar_hob']['ref_only']}/{res[t]['jb_mcnemar_hob']['model_only']})" for t in tags) + ' |')
if any('vs_ref' in res[t] for t in tags):
    for nm, k in [('ref', None), ('REAL flips vs ref', 'REAL-agree_flips'), ('LONG flips vs ref', 'LONG_flips'), ('CF fg_ref', 'CF_fg_ref'), ('CF-probe fg_ref', 'CF-probe_fg_ref'), ('all-q flips vs ref', 'all_flips')]:
        if k is None: print('| reference | - | ' + ' | '.join(res[t].get('ref', '-') for t in tags) + ' |'); continue
        print(f'| {nm} | - | ' + ' | '.join((f"{res[t]['vs_ref'][k]:.2%}" if 'flips' in k else f"{res[t]['vs_ref'][k]:.3f}") if 'vs_ref' in res[t] else '-' for t in tags) + ' |')
    for s_ in ('REAL-agree', 'LONG'):
        print(f'| {s_} TV to hobson (ref) | - | ' + ' | '.join(f"{res[t]['vs_ref'][s_ + '_tv_hob']:.4f} ({res[t]['vs_ref'][s_ + '_tv_hob_ref']:.4f})" if 'vs_ref' in res[t] and s_ + '_tv_hob' in res[t]['vs_ref'] else '-' for t in tags) + ' |')
        print(f'| {s_} paired vs ref: agree-with-hobson lost/gained (p) | - | ' + ' | '.join(f"{res[t]['vs_ref'][s_ + '_paired']['ref_only']}/{res[t]['vs_ref'][s_ + '_paired']['model_only']} ({res[t]['vs_ref'][s_ + '_paired']['p']:.2f})" if 'vs_ref' in res[t] and s_ + '_paired' in res[t]['vs_ref'] else '-' for t in tags) + ' |')
    print('| JB-hard McNemar vs ref | - | ' + ' | '.join(f"{res[t]['vs_ref']['jb_mcnemar']['p']:.2f} ({res[t]['vs_ref']['jb_mcnemar']['ref_only']}/{res[t]['vs_ref']['jb_mcnemar']['model_only']})" if 'vs_ref' in res[t] else '-' for t in tags) + ' |')
print('| low-bit bar vs hobson | - | ' + ' | '.join('PASS' if res[t]['lowbit_pass_vs_hobson'] else 'fail' for t in tags) + ' |')
print('| low-bit bar vs ref | - | ' + ' | '.join(('PASS' if res[t]['lowbit_pass_vs_ref'] else 'fail') if 'lowbit_pass_vs_ref' in res[t] else '-' for t in tags) + ' |')
print('| co-design bar (sd>=.95, fgh>=.90) | - | ' + ' | '.join('PASS' if res[t]['codesign_pass'] else 'fail' for t in tags) + ' |')
