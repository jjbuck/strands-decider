"""score_j8.py (laptop, pure python): score j8 preds files against the evalkit refs.  python3 score_j8.py preds1.jsonl [preds2.jsonl ...]"""
import sys, json, math
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK

SUITES = ['JB-all', 'JB-hard', 'REAL-agree', 'CF', 'CF-probe', 'REAL-label', 'SHUF']


def load(fn):
    P = {}
    for l in open(fn):
        r = json.loads(l); P.setdefault(r['id'], {})[r['qn']] = r['p']
    return P


def flat_ref():
    R = {}
    for s in ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
        for l in open(f'~/decider2/evalkit/refs/{s}.hobson.jsonl'):
            r = json.loads(l); R[r['id']] = r['hobson']
    return R


def compare(P, R):
    n = ag = 0; tv = 0.0; dmax = 0.0; flips = []
    for iid, qs in P.items():
        for q, p in qs.items():
            if iid not in R or q not in R[iid]: continue
            r = R[iid][q]; n += 1
            a = max(p, key=p.get); b = max(r, key=r.get); ag += a == b
            d = 0.5 * sum(abs(p[k] - r[k]) for k in r); tv += d; dmax = max(dmax, max(abs(p[k] - r[k]) for k in r))
            if a != b: flips.append((iid, q, round(max(r.values()), 3)))
    return dict(n=n, agree=round(ag / n, 4), flips=n - ag, tv=round(tv / n, 5), max_abs_dp=round(dmax, 4)), flips


def mcnemar(b, c):
    if b + c == 0: return 1.0
    k = min(b, c); n = b + c
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


if __name__ == '__main__':
    R = flat_ref()
    out = {}
    for fn in sys.argv[1:]:
        P = load(fn)
        c, flips = compare(P, R)
        res = {'vs_hobson_all': c, 'flip_examples': flips[:8]}
        for s in SUITES:
            try:
                sc = EK.score(s, P, baselines=False)
                res[s] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc['model'].items()}
                res[s + '_hobson_same'] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sc['hobson'].items()}
            except Exception as e:
                res[s] = f'ERR {type(e).__name__}: {e}'[:200]
        # JB-hard McNemar vs hobson on covered tasks
        try:
            items = {json.loads(l)['id']: json.loads(l) for l in open('~/decider2/evalkit/suites/JB-hard.jsonl')}
            b = cc = 0
            for iid, it in items.items():
                for q, spec in it['questions'].items():
                    if iid in P and q in P[iid] and iid in R:
                        exp = it['expected'][q] if isinstance(it.get('expected'), dict) else None
                        if exp is None: continue
                        m_ok = max(P[iid][q], key=P[iid][q].get) == str(exp); h_ok = max(R[iid][q], key=R[iid][q].get) == str(exp)
                        b += m_ok and not h_ok; cc += h_ok and not m_ok
            res['JB-hard_mcnemar'] = dict(model_only=b, hobson_only=cc, p=round(mcnemar(b, cc), 4))
        except Exception as e:
            res['JB-hard_mcnemar'] = f'ERR {e}'[:200]
        out[fn] = res
        print('=====', fn); print(json.dumps(res, indent=1))
    json.dump(out, open('~/decider2/n1/results/scores.json', 'w'), indent=1)
