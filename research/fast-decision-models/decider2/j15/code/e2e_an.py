"""summaries of the end-to-end cascade measurements (res_casc*.jsonl, res_dcasc*.jsonl): per-request medians -> mean / p50 / p95."""
import json, sys, numpy as np, glob
def S(v): v = np.asarray(v); return f'mean {v.mean():7.2f} p50 {np.median(v):7.2f} p95 {np.quantile(v, .95):7.2f}'
for f in sorted(glob.glob('~/decider2/j15/res/res_*casc*.jsonl')):
    R = [json.loads(l) for l in open(f)]
    if not R: continue
    print(f'\n{f.split("/")[-1]}  n={len(R)}')
    base = 'verifier' if 'draft' in R[0] else 'b8'
    for grp in ('REAL-agree', 'LONG', 'JB-all', None):
        rr = [r for r in R if grp is None or r['suite'] == grp]
        if not rr: continue
        c = [r['casc'] for r in rr]; b = [r[base] for r in rr]
        extra = f"deferred {np.mean([r['deferred'] for r in rr]):.2f}" if 'deferred' in rr[0] else f"exit<24 {np.mean([r['exit'] < 24 for r in rr]):.2f}"
        print(f'  {grp or "ALL":10s} n {len(rr):3d} | cascade {S(c)} | b8 {S(b)} | ratio of means {np.mean(c)/np.mean(b):.3f} | {extra}')
