"""Laptop: merge spec/grp*.json (+ heldout.json) into ~/decider2/q1/sensitivity.json and print a plain table. Pure python."""
import json, glob, os, sys, statistics as S
D = os.path.expanduser('~/decider2/q1/res/spec')
grps = sorted(glob.glob(f'{D}/grp*.json'))
G = {}; rows = {}; meta = None; n = None
for f in grps:
    d = json.load(open(f)); G.update(d['gemms']); rows.update(d['rows']); meta = meta or d['meta']; n = d['n']
ho = json.load(open(f'{D}/heldout.json')) if os.path.exists(f'{D}/heldout.json') else None
keys = sorted(G, key=lambda s: (int(s.split('.')[0]), ['Win', 'Wo', 'Wgu', 'Wd'].index(s.split('.')[1])))
tot_in = {r: sum(G[k][f'in_{r}']['trace'] for k in keys) for r in ('s', 'q', 'all')}
out = dict(
    what='Eigen-spectrum of the decision-loss gradient second moment G = E[g^T g] at each GEMM of hobson-v19, output side (G_out, N x N) and '
         'input side (G_in, K x K, wrt the input a quantized kernel sees: gain-free RMSNorm for Win/Wgu, mixer output for Wo, silu(g)*u for Wd). '
         'Loss: KL(p || p_perturbed) of the pointer-head decision distribution, i.e. G = sum over Fisher eigen-directions of the softmax (lam_j, u_j) of '
         'lam_j g_j^T g_j (exact for binary questions). bf16 dense forward and backward through the H1/G2 lean runtime (merged LoRA). '
         'Rows: s = state rows [0, q0), q = question rows [q0, T) (question text, options, readout row), all = both. '
         'rXX = number of eigen-directions holding XX% of the trace. pr = participation ratio tr^2/tr(G^2). Traces are per request (sum over rows).',
    n_requests=n, requests='train-split real requests (evalkit/train_pool.jsonl, eval tasks excluded), one random question each, T <= 2600 tokens, '
                           'at most 4 requests per tau task; held-out set B uses disjoint tau tasks.',
    note_isotropic='For isotropic input rounding error (rotated int4 is close to white within a token), the first-order decision error variance of a GEMM '
                   'is sigma^2 tr(G_in), and the best rank-r correction (input-side Q or oblique output-side) removes exactly the top-r eigenvalues of G_in. '
                   'The output-side spectrum weighted by W W^T has the same nonzero eigenvalues as G_in.',
    gemms={}, totals=dict(in_trace=tot_in))
for k in keys:
    o = G[k]; e = dict(N=o['N'], K=o['K'], macs=o['macs'])
    for side in ('out', 'in'):
        for r in ('s', 'q', 'all'):
            s = o[f'{side}_{r}']
            e[f'{side}_{r}'] = dict(trace=s['trace'], r50=s['r50'], r90=s['r90'], r99=s['r99'], r999=s['r999'], pr=round(s['pr'], 1), top_share=s['top_share'])
    e['share_of_total_in_trace'] = {r: (o[f'in_{r}']['trace'] / tot_in[r] if tot_in[r] else 0) for r in ('s', 'q', 'all')}
    e['question_share_of_in_trace'] = o['in_q']['trace'] / o['in_all']['trace'] if o['in_all']['trace'] else 0
    if k in rows: e['row_energy_out'] = rows[k]
    if ho and k in ho['gemms']: e['heldout'] = ho['gemms'][k]
    out['gemms'][k] = e
if meta:
    out['request_stats'] = dict(T_median=S.median(m['T'] for m in meta), q_rows_median=S.median(m['T'] - m['q0'] for m in meta),
                                n_slots={str(v): sum(1 for m in meta if m['n'] == v) for v in sorted({m['n'] for m in meta})},
                                fisher_trace_median=S.median(m['fisher_tr'] for m in meta))
    ft = sorted((m['fisher_tr'] for m in meta), reverse=True); t = sum(ft)
    out['request_stats']['fisher_share_top10pct_requests'] = sum(ft[:max(1, len(ft) // 10)]) / t if t else 0
json.dump(out, open(os.path.expanduser('~/decider2/q1/sensitivity.json'), 'w'), indent=1)
# plain table
print(f"{'gemm':8s} {'N':>6s} {'K':>5s} | out r90 s/q/all | out r99 s/q/all | in r90 s/q/all | in r99 s/q/all | in-trace share | q share")
for k in keys:
    e = out['gemms'][k]
    f = lambda side, q: '/'.join(str(e[f'{side}_{r}'][q]) for r in ('s', 'q', 'all'))
    print(f"{k:8s} {e['N']:6d} {e['K']:5d} | {f('out','r90'):>15s} | {f('out','r99'):>15s} | {f('in','r90'):>14s} | {f('in','r99'):>14s} | "
          f"{100*e['share_of_total_in_trace']['all']:5.2f}% | {100*e['question_share_of_in_trace']:5.1f}%")
