"""Laptop scorer for sens.py output: per config, mean TV and argmax flips against hobson (refs) and against this runtime's dense pass.
python sscore.py sens.jsonl OUT.json"""
import sys, json, collections
sys.path.insert(0, '~/decider2/evalkit')
import evalkit as EK
recs = [json.loads(l) for l in open(sys.argv[1])]
H = {s: EK._as_preds(s, 'hobson') for s in {r['suite'] for r in recs}}
acc = collections.defaultdict(lambda: collections.defaultdict(list))
for r in recs:
    h = EK._norm(H[r['suite']][r['id']][r['q']]); d = EK._norm(r['dense'])
    for nm, p in [('dense', r['dense'])] + list(r['cfg'].items()):
        p = EK._norm(p)
        a = acc[nm]
        a['tv_h'].append(EK._tv(p, h)); a['flip_h'].append(EK._arg(p) != EK._arg(h))
        a['tv_d'].append(EK._tv(p, d)); a['flip_d'].append(EK._arg(p) != EK._arg(d))
        if 'fl' in r and nm in r['fl']: a['fl'].append((r['fl'][nm], r['T']))
# CF pairs present
pairs = [p for p in EK._pairs('CF') if any(r['id'] == p['a'] for r in recs)]
byid = {(r['id'], r['q']): r for r in recs}
res = {}
for nm, a in acc.items():
    o = {k: sum(v) / len(v) for k, v in a.items() if k != 'fl'}
    if a.get('fl'): o['flops'] = sum(f * t for f, t in a['fl']) / sum(t for _, t in a['fl'])
    tr = []
    for p in pairs:
        ra, rb = byid.get((p['a'], p['q'])), byid.get((p['b'], p['q']))
        if not ra or not rb: continue
        pa = ra['dense'] if nm == 'dense' else ra['cfg'][nm]; pb = rb['dense'] if nm == 'dense' else rb['cfg'][nm]
        tr.append(EK._arg(EK._norm(pa)) == p['ea'] and EK._arg(EK._norm(pb)) == p['eb'])
    if tr: o['cf_pairs_tracked'] = sum(tr) / len(tr); o['n_pairs'] = len(tr)
    res[nm] = o
json.dump(res, open(sys.argv[2], 'w'), indent=1)
# per-layer table
print('n questions', len(recs), 'CF pairs', len(pairs))
print('dense', {k: round(v, 4) for k, v in res['dense'].items()})
import re
modes = sorted({re.match(r'L\d+([io]\d+)$', k).group(1) for k in res if re.match(r'L\d+([io]\d+)$', k)}, key=lambda s: (s[0], int(s[1:])))
print('layer | ' + ' | '.join(f'{m} TV/flip%' for m in modes))
for l in range(24):
    cells = []
    for m in modes:
        o = res.get(f'L{l}{m}')
        cells.append('%.3f/%.1f' % (o['tv_h'], 100 * o['flip_h']) if o else '-')
    print(f'{l} | ' + ' | '.join(cells))
for k in sorted(k for k in res if k.startswith('thin')):
    o = res[k]; print(k, 'flops %.3f' % o.get('flops', 0), 'TV_h %.3f flip_h %.1f%% TV_d %.3f flip_d %.1f%% cf_tracked %.2f' % (o['tv_h'], 100 * o['flip_h'], o['tv_d'], 100 * o['flip_d'], o.get('cf_pairs_tracked', float('nan'))))
