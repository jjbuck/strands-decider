"""latency curves (A10G, measured) encoder vs hobson for Q1/Q4/Q15 -> results/lat_curves.png"""
import json, sys, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = '~/decider2/j1/results'
enc = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f'{R}/lat_enc_v1.json'))
hob = json.load(open(sys.argv[2] if len(sys.argv) > 2 else f'{R}/lat_hob.json'))
loc = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else {}
TS = [64, 128, 256, 400, 1000, 2000, 4000]
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for ax, q in zip(axs, ('Q1', 'Q4', 'Q15')):
    g = lambda d, k: [d.get(f'{q}_T{t}{k}', {}).get('median') for t in TS]
    ax.plot(TS, g(enc, ''), 'o-', color='#c0392b', label='T5Gemma-2B encoder (masked cache, fused)')
    if loc: ax.plot(TS, g(loc, ''), 'o:', color='#e67e22', label='encoder, 21/26 layers local-512')
    ax.plot(TS, g(hob, '_single'), 's-', color='#2c3e50', label='hobson fused, one pass' + (' (packed LB)' if q != 'Q1' else ''))
    if q != 'Q1': ax.plot(TS, g(hob, '_plain'), 's--', color='#7f8c8d', label='hobson fused, branch per question')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(TS); ax.set_xticklabels([str(t) for t in TS], fontsize=8)
    ax.set_title(f'{q[1:]} question(s)'); ax.set_xlabel('state tokens'); ax.grid(alpha=.3, which='both')
axs[0].set_ylabel('median ms (A10G)'); axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{R}/lat_curves.png', dpi=130)
print('saved')
