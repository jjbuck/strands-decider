"""score J9 preds (laptop): python3 score_j9.py preds.json [more.json ...]  -> main metrics per suite (covered items only)"""
import sys, json, os
sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import evalkit as EK
KEYS = {'JB-hard': ['acc', 'agree'], 'JB-all': ['acc', 'agree'], 'REAL-agree': ['agree', 'agree_sd', 'tv', 'coverage'], 'LONG': ['agree', 'agree_sd', 'tv', 'coverage'],
        'CF': ['acc', 'flip', 'flip_given_hobson', 'dir', 'coverage'], 'CF-probe': ['acc', 'flip', 'flip_given_hobson', 'dir', 'coverage'],
        'SHUF': ['change', 'both_right'], 'REAL-label': ['acc', 'acc_hobson_same', 'coverage']}
out = {}
for f in sys.argv[1:]:
    if f.endswith('.jsonl'):
        p = {}
        for l in open(f):
            try: r = json.loads(l); p[r['id']] = r['p']
            except Exception: pass
    else:
        p = json.load(open(f))
    row = {}
    for s in KEYS:
        try:
            sc = EK.score(s, p, baselines=False)
        except Exception as e:
            continue
        mm = sc.get('model', {})
        if not mm or mm.get('coverage', 1) == 0: continue
        row[s] = {k: mm.get(k) for k in KEYS[s] if k in mm}
        row[s]['hobson'] = {k: sc['hobson'].get(k) for k in KEYS[s] if k in sc.get('hobson', {})}
    out[os.path.basename(f)] = row
    print('==', os.path.basename(f))
    for s, r in row.items():
        print(f'  {s:<11}', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items() if k != 'hobson'), '| hobson', ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in r['hobson'].items()))
json.dump(out, open(os.path.expanduser('~/decider2/j9/scores_last.json'), 'w'), indent=1)
