"""compact table across result files: python3 h1table.py file:cfg[:label] ...  (dense ref = union of dense preds found in e1_main / e3_dense)"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1score as S
R = os.path.expanduser('~/decider2/h1/res/')
dense = {}
for f in ('e1_main.json', 'e3_dense.json'):
    if os.path.exists(R + f):
        for iid, qs in json.load(open(R + f))['preds'].get('dense', {}).items(): dense.setdefault(iid, {}).update(qs)
cache = {}
print('| config | REAL flips vs hob | vs dense | paired(lost/gained,p) | TV | agree_sd | LONG flips | LONG agree_sd | CF fgh | CF-probe fgh (vs dense) | JB-hard (McNemar) | JB-long agree | SHUF chg/both | REAL-label |')
print('|' + '---|' * 14)
for spec in sys.argv[1:]:
    f, c, *lab = spec.split('::'); lab = lab[0] if lab else c
    if f not in cache: cache[f] = json.load(open(R + f))['preds']
    P = cache[f].get(c)
    if not P: print('|', lab, '| missing |'); continue
    o = S.row(P, dense if c != 'dense' else None)
    g = lambda s, k, fm='{:.3f}': (fm.format(o[s][k]) if s in o and o[s].get(k) is not None else '-')
    ra = o.get('REAL-agree', {}); lo = o.get('LONG', {})
    pr = ra.get('paired'); prs = f"{pr['lost']}/{pr['gained']},{pr['p']}" if pr else '-'
    jb = o.get('JB-hard'); jbs = f"{jb['acc']:.3f} ({jb['mcnemar']['loss']}/{jb['mcnemar']['gain']},p{jb['mcnemar']['p']})" if jb else '-'
    sh = o.get('SHUF'); shs = f"{sh['change']:.3f}/{sh['both_right']:.3f} (n{sh['n']})" if sh else '-'
    print(f"| {lab} | {ra.get('flips','-')}/{ra.get('n','-')} = {100*ra.get('flip_rate',float('nan')):.2f}% | {ra.get('flips_vs_dense','-')} | {prs} | {g('REAL-agree','tv','{:.4f}')} | {g('REAL-agree','agree_sd')} | "
          f"{lo.get('flips','-')}/{lo.get('n','-')} | {g('LONG','agree_sd')} | {g('CF','fgh')} ({o.get('CF',{}).get('n_tracked','-')}) | {g('CF-probe','fgh')} ({g('CF-probe','fgh_vs_dense')}) | {jbs} | {g('JB-long','agree')} | {shs} | {g('REAL-label','acc')} |")
