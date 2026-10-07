"""Depth-speculative cascade analysis (laptop). Draft = b8's own first L layers + trained exit head; verify = continue layers L..23.
Exit at the first L in S whose exit-head margin >= tau_L (tau_L fitted on train-split DEV: residual disagreement with the full b8 decision
among exits at L <= eps/|S| of DEV questions). python exit_an.py L1,L2,.. [eps]"""
import sys, json, numpy as np
import casc2 as C
PD = C.PD


def fit_layer(E, V, eps):
    ks = [(i, q) for i, qs in E.items() for q in qs if i in V and q in V[i]]
    s = np.array([C.margin(E[i][q]) for i, q in ks]); fl = np.array([C.am(E[i][q]) != C.am(V[i][q]) for i, q in ks])
    for t in np.unique(np.concatenate([np.sort(s) + 1e-9, [1.01]])):
        if (fl & (s >= t)).mean() <= eps: return float(t), float((s >= t).mean()), int((fl & (s >= t)).sum())
    return 1.01, 0.0, 0


def run(S, eps, V, Vd, Fl, quiet=False, xp='exit_'):
    E = {L: C.load(f'{PD}/{xp}L{L}_eval.jsonl') for L in S}; Ed = {L: C.load(f'{PD}/{xp}L{L}_dev.jsonl') for L in S}
    taus = {}
    for L in S:
        taus[L] = fit_layer(Ed[L], Vd, eps / len(S))
    def casc(Es, Vv):
        P = {}; where = {}
        for i, qs in Vv.items():
            for q, v in qs.items():
                for L in S:
                    e = Es[L].get(i, {}).get(q)
                    if e is not None and C.margin(e) >= taus[L][0]:
                        P.setdefault(i, {})[q] = e; where[(i, q)] = L; break
                else:
                    P.setdefault(i, {})[q] = v; where[(i, q)] = 24
        return P, where
    Pd, wd = casc(Ed, Vd)
    P, w = casc(E, V)
    sc = C.full_score(P, ref=V, floor=Fl)
    dist = {L: sum(1 for x in w.values() if x == L) / len(w) for L in list(S) + [24]}
    distd = {L: sum(1 for x in wd.values() if x == L) / len(wd) for L in list(S) + [24]}
    resd = sum(1 for (i, q), L in wd.items() if L < 24 and C.am(Pd[i][q]) != C.am(Vd[i][q])) / len(wd)
    return dict(S=S, eps=eps, taus={str(k): v for k, v in taus.items()}, exit_dist_eval=dist, exit_dist_dev=distd, dev_resid=resd, score=sc), P, w


if __name__ == '__main__':
    S = [int(x) for x in sys.argv[1].split(',')]; eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
    base = sys.argv[3] if len(sys.argv) > 3 else 'b8'
    V = C.load(f'{PD}/{base}_eval.jsonl'); Vd = C.load(f'{PD}/{base}_dev.jsonl'); Fl = C.load(f'{PD}/bf16_eval.jsonl') or C.load('~/decider2/h6/preds/h2_bf16.jsonl')
    r, P, w = run(S, eps, V, Vd, Fl, xp='exit_' if base == 'b8' else f'exit_{base}_')
    sc = r['score']
    print(json.dumps({k: v for k, v in r.items() if k != 'score'}))
    print(f"REAL flips {sc['REAL-agree_flips']:.4f} paired vs b8 {sc['REAL-agree_paired_ref']} vs bf16 {sc['REAL-agree_paired_floor']} LONG {sc['LONG_flips']:.4f} TV {sc['REAL-agree_tv']:.4f} "
          f"CF {sc['CF_fgh']:.3f} CFp {sc['CF-probe_fgh']:.3f} JBh {sc['JB-hard_acc']:.3f} {sc['JB-hard_mcn_hob']} lab {sc['REAL-label_acc']:.3f}")
