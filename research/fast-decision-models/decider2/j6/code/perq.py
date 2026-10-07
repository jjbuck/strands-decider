"""per-question agreement / agree_sd breakdown on REAL-agree + LONG for one or more prediction sets.  python3 perq.py NAME=file[:key] ..."""
import sys, json, collections
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK

sets = {}
for x in sys.argv[1:]:
    nm, p = x.split('=', 1); key = None
    if ':' in p: p, key = p.split(':', 1)
    d = json.load(open(p)); sets[nm] = d[key] if key else d
for suite in ('REAL-agree', 'LONG'):
    stat = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0, 0]))
    for nm, pr in sets.items():
        for r in EK._qrows(suite, pr):
            if r['pred'] is None or r['hob'] is None: continue
            s = stat[r['q']][nm]
            ok = EK._arg(r['pred']) == EK._arg(r['hob'])
            s[0] += ok; s[1] += 1
            if r['nost'] is not None and EK._arg(r['nost']) != EK._arg(r['hob']): s[2] += ok; s[3] += 1
    print(f'== {suite}: question  ' + '  '.join(f'{nm}: agree (n) | agree_sd (n_sd)' for nm in sets))
    for q in sorted(stat, key=lambda q: -max(v[1] for v in stat[q].values())):
        print(f'{q:55s}', '  '.join(f'{v[0] / max(1, v[1]):.2f} ({v[1]:3d}) | {v[2] / max(1, v[3]):.2f} ({v[3]:3d})' for nm, v in stat[q].items()))
