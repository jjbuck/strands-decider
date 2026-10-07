import os, json, random


def samples(n, minT, maxT, seed=7):
    """train-split real states (eval tasks excluded), one question each (same rule as errprop.py)"""
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    rows = []
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for li, l in enumerate(f):
            if li % 7 != 3: continue
            r = json.loads(l)
            if r['task'] in EV or not (minT <= r['n_state_tok'] <= maxT): continue
            rows.append(r)
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        qn = sorted(r['questions'])[len(out) % len(r['questions'])]
        out.append((r['rid'], qn, r['state'], r['questions'][qn]))
        if len(out) >= n: break
    return out
