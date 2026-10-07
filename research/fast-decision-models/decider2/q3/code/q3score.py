"""Q3 laptop scorer (pure python + numpy). python3 q3score.py TAG [TAG ...]   preds in ~/decider2/q3/preds/TAG.jsonl
Brief-10 bar, against hobson-v19's references on all 3,227 evalkit questions:
  REAL flips <= 0.7% with McNemar against the bf16 runtime not significant (paired agree-with-hobson, REAL-agree; bf16 runtime = H2's QRT bf16
  preds, ~/decider2/h6/preds/h2_bf16.jsonl); CF retention (flip_given_hobson) >= .99; CF-probe retention >= .95; JB-hard McNemar vs hobson
  not significant; REAL-label >= .78. Also TV to hobson on REAL-agree (W8A8-b8: .0048).
Writes ~/decider2/q3/scores.json and prints a markdown table."""
import sys, os, json, math
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import numpy as np, evalkit as EK
D = os.path.expanduser('~/decider2/q3/preds')
BF16 = os.path.expanduser(os.environ.get('Q3BF16', '~/decider2/q3/preds/qrt_bf16.jsonl'))
SU = ['JB-hard', 'JB-long', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'SHUF', 'REAL-label']
am = lambda d: max(d, key=d.get)


def load(path):
    P = {}
    for l in open(path):
        r = json.loads(l); P.setdefault(r['id'], {})[r['q']] = r['probs']
    return P


def mcn(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


HOB = {}
for s in SU:
    if s != 'SHUF': HOB.update({k: dict(v) for k, v in EK._as_preds(s, 'hobson').items()})


def jb_mcnemar(P, R):
    b = c = 0
    for it in EK.load_suite('JB-hard'):
        for q, e in (it.get('expected') or {}).items():
            p = P.get(it['id'], {}).get(q); h = R.get(it['id'], {}).get(q)
            if p is None or h is None: continue
            hr = am(EK._norm(h)) == e; pr = am(EK._norm(p)) == e
            b += hr and not pr; c += pr and not hr
    return dict(ref_only=b, model_only=c, p=round(mcn(b, c), 3))


def paired_vs(P, R, suites=('REAL-agree',)):
    """agree-with-hobson, model vs reference runtime: lost = ref agrees & model not; gained = model agrees & ref not"""
    b = c = n = 0
    for s in suites:
        for it in EK.load_suite(s):
            i = it['id']
            for q in P.get(i, {}):
                if q not in R.get(i, {}) or q not in HOB.get(i, {}): continue
                h = am(EK._norm(HOB[i][q])); pm = am(EK._norm(P[i][q])) == h; rm = am(EK._norm(R[i][q])) == h
                b += rm and not pm; c += pm and not rm; n += 1
    return dict(lost=b, gained=c, n=n, p=round(mcn(b, c), 4))


def score(tag, path=None):
    P = load(path or f'{D}/{tag}.jsonl'); r = {}
    for s in SU:
        sc = EK.score(s, P, baselines=False); r[s] = sc['model']; r[s + '_hob'] = sc['hobson']
    r['n'] = sum(len(v) for v in P.values())
    r['jb_mcnemar_hob'] = jb_mcnemar(P, HOB)
    if os.path.exists(BF16):
        R = load(BF16)
        r['vs_bf16_real'] = paired_vs(P, R, ('REAL-agree',)); r['vs_bf16_real_long'] = paired_vs(P, R, ('REAL-agree', 'LONG'))
    fl = 1 - r['REAL-agree']['agree']
    bar = dict(real_flips=fl <= 0.007, mcnemar_vs_bf16=r.get('vs_bf16_real', {}).get('p', 0) > 0.05,
               cf_fgh=r['CF'].get('flip_given_hobson', 0) >= 0.99, cfprobe_fgh=r['CF-probe'].get('flip_given_hobson', 0) >= 0.95,
               jb_hard=r['jb_mcnemar_hob']['p'] > 0.05, real_label=r['REAL-label']['acc'] >= 0.78)
    r['bar'] = bar; r['bar_all'] = all(bar.values())
    return r


if __name__ == '__main__':
    res = json.load(open(os.path.expanduser('~/decider2/q3/scores.json'))) if os.path.exists(os.path.expanduser('~/decider2/q3/scores.json')) else {}
    tags = []
    for t in sys.argv[1:]:
        if not os.path.exists(f'{D}/{t}.jsonl'): print('missing', t); continue
        res[t] = score(t); tags.append(t)
    json.dump(res, open(os.path.expanduser('~/decider2/q3/scores.json'), 'w'), indent=1)
    rows = [('n questions', None, 'n'), ('REAL flips vs hobson (bar <= 0.70%)', 'REAL-agree', 'flips'), ('REAL TV vs hobson (b8 .0048)', 'REAL-agree', 'tv'),
            ('REAL agree_sd', 'REAL-agree', 'agree_sd'), ('LONG agree_sd', 'LONG', 'agree_sd'), ('CF fgh (bar >= .99)', 'CF', 'flip_given_hobson'),
            ('CF-probe fgh (bar >= .95)', 'CF-probe', 'flip_given_hobson'), ('JB-hard acc', 'JB-hard', 'acc'), ('REAL-label acc (bar >= .78)', 'REAL-label', 'acc')]
    print('| metric | hobson | ' + ' | '.join(tags) + ' |'); print('|---|---|' + '---|' * len(tags))
    for nm, s, k in rows:
        if s is None: print(f'| {nm} | 3227 | ' + ' | '.join(str(res[t]['n']) for t in tags) + ' |'); continue
        if k == 'flips':
            print(f'| {nm} | 0 | ' + ' | '.join(f"{1 - res[t][s]['agree']:.2%}" for t in tags) + ' |'); continue
        h = res[tags[0]][s + '_hob'].get(k, float('nan')) if tags else float('nan')
        print(f'| {nm} | {h:.3f} | ' + ' | '.join(f"{res[t][s].get(k, float('nan')):.4f}" for t in tags) + ' |')
    print('| McNemar vs bf16 runtime, REAL (lost/gained, p) | - | ' + ' | '.join(
        f"{res[t]['vs_bf16_real']['lost']}/{res[t]['vs_bf16_real']['gained']} (p {res[t]['vs_bf16_real']['p']:.3g})" if 'vs_bf16_real' in res[t] else '-' for t in tags) + ' |')
    print('| JB-hard McNemar vs hobson (hob-only/model-only, p) | - | ' + ' | '.join(
        f"{res[t]['jb_mcnemar_hob']['ref_only']}/{res[t]['jb_mcnemar_hob']['model_only']} (p {res[t]['jb_mcnemar_hob']['p']:.2f})" for t in tags) + ' |')
    print('| all bars met | - | ' + ' | '.join(str(res[t]['bar_all']) for t in tags) + ' |')
