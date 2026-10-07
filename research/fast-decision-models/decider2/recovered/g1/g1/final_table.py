"""G1: kill-criterion table for G1 arms (reads /tmp/decider2/g1/res/<SUITE>.<TAG>.jsonl) next to hobson and the evalkit baselines."""
import sys, os, json, math
sys.path.insert(0, '/tmp/decider2/evalkit'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit as EK
import score_g1 as SC

tags = sys.argv[1:]
COLS = [('JB-hard', 'acc'), ('JB-all', 'acc'), ('REAL-agree', 'agree'), ('REAL-agree', 'agree_sd'), ('LONG', 'agree'), ('LONG', 'agree_sd'),
        ('CF', 'acc'), ('CF', 'flip'), ('CF', 'flip_given_hobson'), ('CF-probe', 'acc'), ('CF-probe', 'flip_given_hobson'), ('CF-probe', 'fgh_distract'),
        ('SHUF', 'change'), ('REAL-label', 'acc'), ('JB-hard', 'mcnemar_b'), ('JB-hard', 'mcnemar_c'), ('JB-hard', 'mcnemar_p')]
res = {}
for t in tags:
    p = SC.load(t)
    if not p: continue
    res[t] = {su: EK.score(su, p, baselines=False)['model'] for su in sorted({s for s, _ in COLS})}
    for su, d in SC.extra(p).items(): res[t].setdefault(su, {}).update(d)
    if 'denseft' in ' '.join(tags) and t != 'denseft':
        ctl = SC.load('denseft')
        if ctl:
            b, c, pv = SC.mcnemar('JB-hard', p, ref=ctl); res[t]['JB-hard'].update(mc_ctl_b=b, mc_ctl_c=c, mc_ctl_p=pv)
full = {su: EK.score(su, {}, baselines=True) for su in sorted({s for s, _ in COLS})}
for r in ['hobson', 'merged_full', 'nostate', 'qattn10@7', 'rand50@7', 'rand25@7']:
    res[r] = {su: dict(full[su].get(r, {})) for su in full}
    rp = {}
    for su in ('CF-probe', 'JB-all'): rp.update(EK._as_preds(su, r))
    if rp:
        for su, d in SC.extra(rp).items(): res[r].setdefault(su, {}).update(d)
hdr = ''.join(f'{(s + ":" + m)[:14]:>15s}' for s, m in COLS + [('JB-hard', 'mc_ctl_p')])
print(f'{"":16s}' + hdr)
for k, v in res.items():
    cells = []
    for s, m in COLS + [('JB-hard', 'mc_ctl_p')]:
        x = v.get(s, {}).get(m)
        cells.append('           -   ' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:15.3f}')
    print(f'{k:16s}' + ''.join(cells))
cov = {t: {su: res[t][su].get('coverage') for su in res[t]} for t in tags if t in res}
print('coverage', json.dumps(cov))
json.dump(res, open('/tmp/decider2/g1/res/final_scores.json', 'w'), indent=1, default=str)
