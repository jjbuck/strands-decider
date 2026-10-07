"""Mean / p50 / p95 decision latency on the real length distributions (A10G, interpolated at each question's exact row count from the
measured grid), for b8 alone, the W4A4->b8 precision cascade and the depth-exit cascade.  python costs.py"""
import json, sys, numpy as np
import casc2 as C, lat as LT, exit_an as XA
import evalkit as EK

GROUPS = {'REAL-agree': ['REAL-agree'], 'LONG': ['LONG'], 'JevBench': ['JB-all'], 'REAL+LONG+JB': ['REAL-agree', 'LONG', 'JB-all']}


def rows_of(path):
    return {(r['id'], r['q']): r['T'] for r in map(json.loads, open(path))}


def qkeys(suites):
    out = []
    for s in suites:
        for it in EK.load_suite(s):
            out += [(it['id'], q) for q in it['questions']]
    return out


if __name__ == '__main__':
    cv = LT.curves()
    T = rows_of(f'{C.PD}/b8_eval.jsonl')
    V = C.load(f'{C.PD}/b8_eval.jsonl'); D = C.load(f'{C.PD}/w4a4_eval.jsonl')
    Fl = C.load(f'{C.PD}/bf16_eval.jsonl') or C.load('~/decider2/h6/preds/h2_bf16.jsonl'); Vd = C.load(f'{C.PD}/b8_dev.jsonl')
    an = json.load(open('~/decider2/j15/res/analysis.json'))
    res = {}
    S = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [12]
    eps_x = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    xr, XP, where = XA.run(S, eps_x, V, Vd, Fl)
    for g, su in GROUPS.items():
        ks = [k for k in qkeys(su) if k in T]
        r = np.array([T[k] for k in ks])
        b8 = LT.at(cv[('b8', 'full')], r); w4 = LT.at(cv[('w4a4', 'full')], r); bf = LT.at(cv[('bf16', 'full')], r)
        out = dict(n=len(ks), rows=LT.summ(r), b8=LT.summ(b8), w4a4=LT.summ(w4), bf16=LT.summ(bf))
        for eps, sc in an['w4a4']['casc'].items():
            tau = sc['tau']
            dfr = np.array([C.margin(D[i][q]) < tau for i, q in ks])
            c = w4 + dfr * b8
            out[f'prec_casc_eps{eps}'] = dict(LT.summ(c), defer=float(dfr.mean()), ratio_mean=float(c.mean() / b8.mean()))
        # depth cascade: exit at L costs pre:L; no exit costs pre:Lmax + post:Lmax (resume)
        Lmax = max(S)
        c = []
        for (i, q), rr in zip(ks, r):
            L = where[(i, q)]
            if L < 24: c.append(LT.at(cv[('b8', f'pre:{L}')], [rr])[0])
            else: c.append(LT.at(cv[('b8', f'pre:{Lmax}')], [rr])[0] + LT.at(cv[('b8', f'post:{Lmax}')], [rr])[0] if ('b8', f'post:{Lmax}') in cv else LT.at(cv[('b8', 'full')], [rr])[0])
        c = np.array(c)
        out['depth_casc'] = dict(LT.summ(c), exit=float(np.mean([where[k] < 24 for k in ks])), ratio_mean=float(c.mean() / b8.mean()))
        res[g] = out
        print(g, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) for kk, vv in v.items()}) for k, v in out.items()}))
    res['depth_cfg'] = {k: v for k, v in xr.items() if k != 'score'}
    json.dump(res, open('~/decider2/j15/res/costs.json', 'w'), indent=1)
