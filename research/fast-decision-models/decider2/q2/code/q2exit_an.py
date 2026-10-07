"""Laptop: depth cascade base -> layer-L exit (J15 exit_an.fit_layer): tau fitted on train-split DEV for zero residual disagreement with the
full base decision among exits (eps = 0), applied unchanged to eval. Writes ~/decider2/q2/preds/preds_<out>.jsonl (cascade probs) and
prints exit shares. python q2exit_an.py BASE L OUT [eps]"""
import sys, os, json, collections
sys.path[:0] = ['~/decider2/j15/code', '~/decider2/evalkit']
import numpy as np
import casc2 as C
D = '~/decider2/q2/preds'
base, L, outn = sys.argv[1], int(sys.argv[2]), sys.argv[3]; eps = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
V = C.load(f'{D}/{base}_eval.jsonl'); Vd = C.load(f'{D}/{base}_dev.jsonl')
E = C.load(f'{D}/exit_{base}_L{L}_eval.jsonl'); Ed = C.load(f'{D}/exit_{base}_L{L}_dev.jsonl')
ks = [(i, q) for i, qs in Ed.items() for q in qs if i in Vd and q in Vd[i]]
s = np.array([C.margin(Ed[i][q]) for i, q in ks]); fl = np.array([C.am(Ed[i][q]) != C.am(Vd[i][q]) for i, q in ks])
tau = 1.01
for t in np.unique(np.concatenate([np.sort(s) + 1e-9, [1.01]])):
    if (fl & (s >= t)).mean() <= eps: tau = float(t); break
print(f'DEV: {len(ks)} questions, exit-head disagreement with full {base} {fl.mean():.4f}; tau {tau:.6f} -> DEV exit share {(s >= tau).mean():.4f}, '
      f'residual changes {(fl & (s >= tau)).sum()}')
meta = {}
for l in open(f'{D}/{base}_eval.jsonl'):
    r = json.loads(l); meta[(r['id'], r['q'])] = r
P = {}; ex = collections.Counter(); tot = collections.Counter(); chg = collections.Counter()
with open(f'{D}/preds_{outn}.jsonl', 'w') as f:
    for (i, q), r in meta.items():
        e = E.get(i, {}).get(q)
        take = e is not None and C.margin(e) >= tau
        p = e if take else r['probs']
        tot[r['suite']] += 1; ex[r['suite']] += take; chg[r['suite']] += take and C.am(e) != C.am(r['probs'])
        f.write(json.dumps(dict(suite=r['suite'], id=i, q=q, T=r['T'], probs=p, exit=16 if take else 24)) + '\n')
print('eval exit share by suite:', {k: round(ex[k] / tot[k], 4) for k in tot}, 'overall', round(sum(ex.values()) / sum(tot.values()), 4))
print('eval decisions changed vs full base by exits:', dict(chg), 'total', sum(chg.values()))
json.dump(dict(base=base, L=L, eps=eps, tau=tau, dev_n=len(ks), dev_exit=float((s >= tau).mean()), dev_head_disagree=float(fl.mean()),
               eval_exit={k: ex[k] / tot[k] for k in tot}, eval_changed=dict(chg)), open(f'~/decider2/q2/res/exit_{outn}.json', 'w'), indent=1)
