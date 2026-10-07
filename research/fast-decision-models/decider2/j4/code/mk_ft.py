"""J4: the fine-tune row list (the F7 / v19-mix recipe, one fixed order shared by every arm; arm (c) simply reads further down the list).
  gold rows : train_v5 (train_v5.holdout is a separate file, unused) + generated_v16 + generated_v18 + adequacy_gen, label = gold option name
  real rows : evalkit/train_pool.jsonl (train-split tasks only; eval tasks asserted absent), one random question per request, KL-only to hobson
  proportion: real rows are 5.5% of rows (F7: 6,000 of 108,547), interleaved uniformly
python mk_ft.py OUT.jsonl --n 60000
"""
import os, json, random, argparse, collections, sys

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--n', type=int, default=60000); ap.add_argument('--p_real', type=float, default=0.055)
ap.add_argument('--seed', type=int, default=4)
a = ap.parse_args()
H = os.path.expanduser('~/decider2') if os.path.exists(os.path.expanduser('~/decider2/evalkit')) else os.path.expanduser('~/work')
SD = os.path.expanduser('~/code/strands-decider/data/synthetic') if os.path.exists(os.path.expanduser('~/code/strands-decider')) else os.path.expanduser('~/work/sd/data/synthetic')
EV = set(json.load(open(f'{H}/evalkit/split.json'))['eval_tasks'])
R = random.Random(a.seed)


def conv(r):
    ins = r['instructions']
    if r['kind'] == 'choice':
        return {'type': 'choice', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    if r['kind'] == 'noul':
        return {'type': 'noul', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


gold = []
for path in [f'{H}/training/data/train_v5.jsonl', f'{SD}/generated_v16.jsonl', f'{SD}/generated_v18.jsonl', f'{SD}/adequacy_gen.jsonl']:
    for l in open(path):
        r = json.loads(l)
        q, lab = conv(r)
        gold.append(dict(kind='gold', src=r.get('task', '?'), state=r['state'], q=q, label=lab, variants=r.get('instruction_variants') or []))
real = []
for l in open(f'{H}/evalkit/train_pool.jsonl'):
    r = json.loads(l)
    assert r['task'] not in EV
    qn = R.choice(sorted(r['questions']))
    real.append(dict(kind='real', src=r['domain'] + '/' + qn, state=r['state'], q=r['questions'][qn], label=None, variants=[], task=r['task']))
R.shuffle(gold); R.shuffle(real)
print('gold', len(gold), 'real', len(real), file=sys.stderr)
out = []; gi = ri = 0
while len(out) < a.n and (gi < len(gold) or ri < len(real)):
    if R.random() < a.p_real and ri < len(real): out.append(real[ri]); ri += 1
    elif gi < len(gold): out.append(gold[gi]); gi += 1
with open(a.out, 'w') as f:
    for i, r in enumerate(out):
        r['i'] = i; f.write(json.dumps(r) + '\n')
c = collections.Counter(r['kind'] for r in out)
print('rows', len(out), dict(c), file=sys.stderr)
