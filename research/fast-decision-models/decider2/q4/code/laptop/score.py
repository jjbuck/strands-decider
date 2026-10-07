"""Q4 (laptop): score q4eval preds on all 3,227 evalkit questions against the bar, with J15's scorer (casc2.full_score, read-only import).
python score.py name [name ...]   (preds in ~/decider2/q4/preds/<name>.jsonl)
Also: decision agreement between pairs (b13 exactness at scale)."""
import sys, json, os
sys.path[:0] = ['~/decider2/j15/code', '~/decider2/evalkit']
import casc2 as C
import evalkit as EK
PD = '~/decider2/q4/preds'
FL = C.load('~/decider2/j15/preds/bf16_eval.jsonl')   # J15's bf16 runtime (H2 kernels) = the runtime floor


def tv(P, H):
    import numpy as np
    v = []
    for s in ('REAL-agree', 'LONG'):
        for i, q, _ in C.suite_rows(s, P):
            a = EK._norm(P[i][q]); b = EK._norm(H[i][q])
            v.append(0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)))
    return float(np.mean(v))


def main(names):
    H = C.hob(); out = {}
    for n in names:
        P = C.load(f'{PD}/{n}.jsonl')
        r = C.full_score(P, floor=FL)
        r['n_items'] = sum(len(v) for v in P.values())
        r['tv_real_long_vs_hobson'] = tv(P, H)
        out[n] = r
        print(n, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    if 'b8gptq' in names and 'b8gptq_b13' in names:
        A = C.load(f'{PD}/b8gptq.jsonl'); B = C.load(f'{PD}/b8gptq_b13.jsonl')
        ks = [(i, q) for i in A for q in A[i] if i in B and q in B[i]]
        dis = sum(C.am(A[i][q]) != C.am(B[i][q]) for i, q in ks)
        mx = max(max(abs(A[i][q][k] - B[i][q].get(k, 0)) for k in A[i][q]) for i, q in ks)
        out['b13_vs_base'] = dict(n=len(ks), decisions_differ=dis, max_abs_dp=mx)
        print('b13 vs base: questions', len(ks), 'decisions differ', dis, 'max |dp|', mx)
    json.dump(out, open('~/decider2/q4/res/scores.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:])
