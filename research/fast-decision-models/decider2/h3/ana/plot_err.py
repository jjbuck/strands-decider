"""Paper fig 2c/5c analogue for hobson-v19: residual-stream relative error at the end of each layer after quantising ONE layer (W4A4),
normalised to the error the quantised layer injected; plus per-layer decision TV."""
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
FA = (3, 7, 11, 15, 19, 23)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, (f, lab) in enumerate((('int4', 'int4 per-token W4A4'), ('int4_rot', 'int4 W4A4 + Hadamard'), ('nvfp4', 'NVFP4 W4A4'))):
    R = json.load(open(f'~/decider2/h3/res/errA_{f}.json'))['res']
    for ql in (0, 3, 7, 12, 18):
        inj = R[str(ql)][str(ql)]['layer_out']['rel']
        ax[k].plot(range(ql, 24), [R[str(ql)][str(i)]['layer_out']['rel'] / inj for i in range(ql, 24)], marker='o', ms=3, label=f'layer {ql} quantised')
    ax[k].axhline(1.0, color='grey', lw=0.8, ls='--'); ax[k].set_ylim(0, 1.25); ax[k].set_xlabel('layer'); ax[k].set_title(lab)
    if k == 0: ax[k].set_ylabel('residual rel. error / injected error')
    for a_ in FA: ax[k].axvline(a_, color='0.9', lw=3, zorder=0)
ax[0].legend(fontsize=8)
fig.suptitle('hobson-v19: quantisation error injected at one layer decays through depth (grey bands = attention layers); no submodule amplification')
fig.tight_layout(); fig.savefig('~/decider2/h3/fig_errprop.png', dpi=120)
fig, ax = plt.subplots(figsize=(8, 3.5))
for f, lab in (('int4', 'int4'), ('int4_rot', 'int4 + rot'), ('nvfp4', 'NVFP4')):
    R = json.load(open(f'~/decider2/h3/res/errA_{f}.json'))['res']
    ax.plot(range(24), [R[str(l)]['dec']['tv'] for l in range(24)], marker='o', ms=3, label=lab)
ax.set_xlabel('quantised layer (only this one)'); ax.set_ylabel('decision TV vs bf16 (12 q)'); ax.legend(); ax.set_xticks(range(24))
fig.tight_layout(); fig.savefig('~/decider2/h3/fig_layer_tv.png', dpi=120)
print('ok')
