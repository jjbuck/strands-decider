"""Score Q2 runtime preds on the laptop (numpy only): the BRIEF10 fidelity bar for each tag, paired against the bf16 runtime and hobson.
python score_q2.py tag[,tag..] [base_tag=h2_bf16]   reads ~/decider2/q2/preds/preds_<tag>.jsonl, writes ~/decider2/q2/res/scores.json"""
import sys, os, json, math
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import numpy as np, evalkit as EK
D = os.path.expanduser('~/decider2/q2/preds')
tags = sys.argv[1].split(','); base_tag = sys.argv[2] if len(sys.argv) > 2 else 'h2_bf16'


def load(tag):
    P = {}
    for l in open(f'{D}/preds_{tag}.jsonl'):
        r = json.loads(l); P.setdefault(r['id'], {})[r['q']] = r['probs']
    return P


def arg(p): return max(p, key=p.get)


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


SU = ['JB-hard', 'REAL-agree', 'LONG', 'CF', 'CF-probe', 'REAL-label']
base = load(base_tag) if os.path.exists(f'{D}/preds_{base_tag}.jsonl') else None
refs_real = EK.load_refs('REAL-agree')
real_items = EK.load_suite('REAL-agree')
res = {}
for tag in tags:
    P = load(tag); r = {}
    for s in SU:
        sc = EK.score(s, P, baselines=False); r[s] = sc['model']; r[s + '_hob'] = sc['hobson']
    # REAL flips vs hobson, TV vs hobson, McNemar vs the bf16 runtime (both judged against hobson's answer)
    fl = []; tv = []; b = c = 0
    for it in real_items:
        for q in it['questions']:
            if q not in P.get(it['id'], {}): continue
            h = EK._norm(refs_real[it['id']]['hobson'][q]); p = P[it['id']][q]
            f = arg(p) != arg(h); fl.append(f)
            tv.append(0.5 * sum(abs(p.get(k, 0) - h.get(k, 0)) for k in set(p) | set(h)))
            if base is not None and q in base.get(it['id'], {}):
                fb = arg(base[it['id']][q]) != arg(h)
                b += f and not fb; c += fb and not f
    r['real'] = dict(n=len(fl), flips=float(np.mean(fl)), tv=float(np.mean(tv)), model_only_flips=b, base_only_flips=c, mcnemar_vs_base=mcnemar(b, c))
    refs = EK.load_refs('JB-hard'); b = c = 0
    for it in EK.load_suite('JB-hard'):
        for q in it['questions']:
            e = it['expected'][q]; h = refs[it['id']]['hobson'][q]; p = P[it['id']][q]
            hr = arg(EK._norm(h)) == e; pr = arg(EK._norm(p)) == e
            b += hr and not pr; c += pr and not hr
    r['jbhard_mcnemar'] = dict(hobson_only=b, model_only=c, p=mcnemar(b, c))
    bar = dict(real_flips=r['real']['flips'] <= 0.007, mcnemar_vs_bf16=r['real']['mcnemar_vs_base'] > 0.05,
               cf_ret=r['CF'].get('flip_given_hobson', 0) >= 0.99, cfprobe_ret=r['CF-probe'].get('flip_given_hobson', 0) >= 0.95,
               jbhard=not (r['jbhard_mcnemar']['p'] < 0.05 and b > c), real_label=r['REAL-label'].get('acc', 0) >= 0.78)
    r['bar'] = bar; r['passes'] = all(bar.values())
    res[tag] = r
    print(f"{tag}: REAL flips {r['real']['flips']:.2%} (n {r['real']['n']}; vs {base_tag}: model-only {r['real']['model_only_flips']} / base-only "
          f"{r['real']['base_only_flips']}, McNemar p {r['real']['mcnemar_vs_base']:.3f}), TV {r['real']['tv']:.4f}, CF ret {r['CF'].get('flip_given_hobson', float('nan')):.3f}, "
          f"CF-probe ret {r['CF-probe'].get('flip_given_hobson', float('nan')):.3f}, JB-hard {r['JB-hard'].get('acc', float('nan')):.3f} "
          f"(McNemar vs hobson {b}/{c} p {r['jbhard_mcnemar']['p']:.2f}), REAL-label {r['REAL-label'].get('acc', float('nan')):.3f}, "
          f"LONG agree_sd {r['LONG'].get('agree_sd', float('nan')):.3f} -> {'PASSES' if r['passes'] else 'misses: ' + ','.join(k for k, v in bar.items() if not v)}")
out = os.path.expanduser('~/decider2/q2/res/scores.json')
old = json.load(open(out)) if os.path.exists(out) else {}
old.update(res)
json.dump(old, open(out, 'w'), indent=1)
