"""Build data/cf.jsonl from H7's CF-style augmentation pairs (train-split states only; labels by construction): token ids exactly as hobson renders
them, rows <= 4096 tokens (pairs dropped whole if either item is longer). No teacher (CE-only rows). python j2cf.py ~/work/j2/data/cf_aug.jsonl"""
import os, sys, json, collections, random
sys.path[:0] = [os.path.expanduser('~/work/j2'), os.path.expanduser('~/work/evalkit')]
from j2data import prep_item
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
from kitrun import load_P
P = load_P(); eng = P.eng
pairs = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); assert r['task'] not in EV
    pairs[(r['kind'], r['pair'])].append(r)
keys = sorted(pairs); random.Random(1).shuffle(keys)
out = open(os.path.expanduser('~/work/j2/data/cf.jsonl'), 'w'); n = 0; kinds = collections.Counter()
for k in keys:
    its = []
    for r in pairs[k]:
        (qn, qd), = r['questions'].items()
        it = prep_item(eng, r['state'], qd)
        if len(it['s']) + len(it['q']) > 4096: its = None; break
        g = r['expected'][qn]
        its.append(dict(src='cf', task=r['task'], kind=it['kind'], cfkind=r['kind'], s=it['s'], q=it['q'], opt=it['opt'], n_slots=it['n_slots'],
                        labels=it['labels'], gold=it['labels'].index(g), tp=None))
    if not its or len(its) != 2: continue
    for it in its: out.write(json.dumps(it) + '\n')
    n += 1; kinds[k[0]] += 1
print('pairs', n, kinds)
