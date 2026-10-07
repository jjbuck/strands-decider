"""Projection of measured A10G latencies to RTX 3090 / 4090 / 5090 (arithmetic, not measured).
Per (question set, arm): least-squares t = a + b*rows on the T >= 1000 points; the row term scales with the card's dense half-precision tensor rate
(fp16 accumulation on GeForce; ratios from the doc's own projection of hobson at 1000 tokens: 27.4/52.7, 13.7/52.7, 10.0/52.7), the intercept with
DRAM bandwidth (600 GB/s A10G vs 936 / 1008 / 1792). Points below the fit range are projected as min(measured*bw ratio, ...) -> a*bw + b*rows*cmp."""
import json, sys, numpy as np
CMP = {'3090': 27.4 / 52.7, '4090': 13.7 / 52.7, '5090': 10.0 / 52.7}
BW = {'3090': 600 / 936, '4090': 600 / 1008, '5090': 600 / 1792}
d = json.load(open(sys.argv[1]))
out = {}
keys = [k for k in d if k != 'meta']
qsets = sorted(set(k.split('_T')[0] for k in keys))
for qs in qsets:
    Ts = sorted(int(k.split('_T')[1]) for k in keys if k.startswith(qs + '_T'))
    arms = sorted(set(a for T in Ts for a in d[f'{qs}_T{T}']))
    for arm in arms:
        pts = [(d[f'{qs}_T{T}'][arm]['rows'], d[f'{qs}_T{T}'][arm]['median'], T) for T in Ts if arm in d[f'{qs}_T{T}']]
        fit = [(r, t) for r, t, T in pts if T >= 1000]
        if len(fit) < 2: continue
        A = np.array([[1, r] for r, _ in fit]); y = np.array([t for _, t in fit])
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        a = max(a, 0.0)
        for r, t, T in pts:
            comp = b * r; fixed = max(t - comp, 0.0)          # measured split: everything not explained by the row term is 'fixed' (bandwidth/launch)
            out.setdefault(f'{qs}_T{T}', {})[arm] = {c: round(fixed * BW[c] + comp * CMP[c], 1) for c in CMP} | {'A10G': t}
json.dump(out, open(sys.argv[2], 'w'), indent=1)
for k, v in out.items(): print(k, v)
