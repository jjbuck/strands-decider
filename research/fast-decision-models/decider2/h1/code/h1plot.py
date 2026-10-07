"""H1 plots: QAT learning curves (train KL / hid / train flips vs step) and per-checkpoint eval metrics. python3 h1plot.py"""
import json, os, sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.expanduser('~/decider2/h1/res/')
runs = [('train_qat_w8a8_A.log.jsonl', 'QAT-A W8A8 lr1e-4 (diverged)'), ('train_qat_w4a8_B.log.jsonl', 'QAT-B W4A8 lr1e-4'), ('train_qat_w4a4_C.log.jsonl', 'QAT-C W4A4-k48')]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for f, lab in runs:
    if not os.path.exists(R + f): continue
    rs = [json.loads(l) for l in open(R + f)]
    st = [r['step'] for r in rs if 'kl' in r]
    ax[0].plot(st, [r['kl'] for r in rs if 'kl' in r], label=lab); ax[1].plot([r['step'] for r in rs], [r['hid'] for r in rs], label=lab)
    ax[2].plot([r['step'] for r in rs], [r['flip'] for r in rs], label=lab)
for a_, t in zip(ax, ('train KL(teacher||student), real states', 'relative residual MSE @5/11/17/23', 'train argmax flips (50-step mean)')):
    a_.set_title(t, fontsize=9); a_.set_xlabel('step'); a_.set_yscale('log' if 'flips' not in t else 'linear')
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(R + '../qat_train_curves.png', dpi=130)
ev = R + 'curve_eval.json'
if os.path.exists(ev):
    d = json.load(open(ev)); fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for run, pts in d.items():
        st = [p['step'] for p in pts]
        ax[0].plot(st, [100 * p['real_flips'] for p in pts], 'o-', label=run); ax[1].plot(st, [p['cf_fgh'] for p in pts], 'o-', label=run); ax[2].plot(st, [p['cfp_fgh'] for p in pts], 'o-', label=run)
    for a_, t in zip(ax, ('REAL-agree-SD flips vs hobson (%)', 'CF fgh', 'CF-probe fgh')): a_.set_title(t, fontsize=9); a_.set_xlabel('QAT step')
    ax[1].axhline(0.97, ls='--', c='k', lw=.8); ax[2].axhline(0.97, ls='--', c='k', lw=.8); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(R + '../qat_eval_curves.png', dpi=130)
print('ok')
