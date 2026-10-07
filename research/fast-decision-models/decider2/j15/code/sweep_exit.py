"""DEV-only selection of the exit set: every subset of the trained exit layers, tau per layer at zero DEV residual; DEV cost in layer units
(exit at L costs L/24, no exit costs 1). Reports the DEV-optimal sets and their eval scores."""
import itertools, json, numpy as np
import casc2 as C, exit_an as XA
LAY = [4, 6, 8, 10, 12, 14, 16, 18, 20]
V = C.load(f'{C.PD}/b8_eval.jsonl'); Vd = C.load(f'{C.PD}/b8_dev.jsonl'); Fl = C.load(f'{C.PD}/bf16_eval.jsonl')
Ed = {L: C.load(f'{C.PD}/exit_L{L}_dev.jsonl') for L in LAY}
ks = [(i, q) for i, qs in Vd.items() for q in qs]
tau0 = {L: XA.fit_layer(Ed[L], Vd, 0.0)[0] for L in LAY}
M = {L: np.array([C.margin(Ed[L][i][q]) for i, q in ks]) for L in LAY}
res = []
for r in range(1, 5):
    for S in itertools.combinations(LAY, r):
        left = np.ones(len(ks), bool); cost = np.zeros(len(ks))
        for L in S:
            ex = left & (M[L] >= tau0[L]); cost[ex] = L / 24; left &= ~ex
        cost[left] = 1.0 + 0.0
        res.append((cost.mean(), S))
res.sort()
print('DEV-optimal exit sets (eps 0 per layer):')
for c, S in res[:8]: print(f'  {S}: DEV layer-cost {c:.3f}')
out = []
for c, S in res[:5] + [(None, (16,)), (None, (8, 16)), (None, (14,)), (None, (14, 16)), (None, (8, 14, 16))]:
    r_, P, w = XA.run(list(S), 0.0, V, Vd, Fl)
    sc = r_['score']
    lc = float(np.mean([(L / 24 if L < 24 else 1.0) for L in w.values()]))
    out.append(dict(S=S, dev_cost=c, eval_layer_cost=lc, exit_eval=r_['exit_dist_eval'], taus={k: v[0] for k, v in r_['taus'].items()}, score=sc))
    print(f"S={S} dev-cost {c if c is None else round(c,3)} EVAL layer-cost {lc:.3f} exits {json.dumps({k: round(v,3) for k,v in r_['exit_dist_eval'].items()})} | REAL {sc['REAL-agree_flips']:.4f} paired-b8 {sc['REAL-agree_paired_ref']} LONG {sc['LONG_flips']:.4f} {sc['LONG_paired_ref']} TV {sc['REAL-agree_tv']:.4f} CF {sc['CF_fgh']:.3f} CFp {sc['CF-probe_fgh']:.3f} JBh {sc['JB-hard_acc']:.3f} {sc['JB-hard_mcn_hob']} lab {sc['REAL-label_acc']:.3f}")
json.dump(out, open('~/decider2/j15/res/exit_sweep.json', 'w'), indent=1, default=str)
