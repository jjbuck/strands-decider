"""sample-efficiency curves: metric vs fine-tune rows for each arm (laptop).  python curves.py  -> ../curves.json, ../curves.png"""
import os, sys, json, glob, re
sys.path.insert(0, os.path.expanduser('~/decider2/j4/code'))
import score_j4 as S
import evalkit as EK

P = os.path.expanduser('~/decider2/j4/preds/')
METRICS = [('JB-all', lambda s: s['JB-all']), ('JB-hard', lambda s: s['JB-hard']), ('REAL-label', lambda s: s['REAL-label']),
           ('REAL agree_sd', lambda s: s['REAL agree_sd']), ('CF acc', lambda s: s['CF acc']), ('CF flip', lambda s: s['CF flip']),
           ('CF-probe acc', lambda s: s['CF-probe acc']), ('CF-probe flip', lambda s: s['CF-probe flip'])]
res = {}
for f in sorted(glob.glob(P + '*.json')):
    nm = os.path.basename(f)[:-5]
    m = re.match(r'([a-z]+)_r(\d+)$', nm) or re.match(r'(c)_(final)$', nm)
    if m and m.group(1) == 'c' and m.group(2) == 'final': m = re.match(r'(c)_(24416)', 'c_24416')
    if not m: continue
    arm, rows = m.group(1), m.group(2)
    p = json.load(open(f))
    try:
        s = S.summary(p, have_long=False)
    except Exception as e:
        print('skip', nm, e); continue
    res.setdefault(arm, {})[rows] = {k: fn(s) for k, fn in METRICS}
json.dump(res, open(os.path.expanduser('~/decider2/j4/curves.json'), 'w'), indent=1)
for arm, d in res.items():
    for rows, v in sorted(d.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 10 ** 9):
        print(arm, rows, {k: round(x, 3) for k, x in v.items()})
try:
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    hob = S.summary({k: v for nm in ('JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe') for k, v in EK._as_preds(nm, 'hobson').items()})
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (mn, fn) in zip(axs.flat, METRICS):
        for arm, col in (('a', 'tab:blue'), ('c', 'tab:cyan'), ('b', 'tab:red'), ('d', 'tab:green'), ('e', 'tab:purple'), ('bf', 'tab:orange')):
            if mn == 'REAL agree_sd' and arm in ('d', 'e', 'bf'): continue   # those runs were scored on REAL-label only
            if arm not in res: continue
            xs = sorted(int(r) for r in res[arm] if r.isdigit())
            ax.plot(xs, [res[arm][str(x)][mn] for x in xs], 'o-', color=col, label={'a': '(a) FT only', 'c': '(c) = (a) continued (x = rows)', 'b': '(b) DP + FT', 'd': '(d) NTP + FT', 'e': '(e) DP mixed into FT', 'bf': '(b) fresh head'}[arm])
        ax.axhline(fn(hob), color='k', ls='--', lw=1, label='hobson-v19 (115k rows)')
        ax.set_xscale('log'); ax.set_title(mn); ax.set_xlabel('fine-tune rows')
    axs.flat[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.expanduser('~/decider2/j4/curves.png'), dpi=110)
except Exception as e:
    print('plot failed', e)
