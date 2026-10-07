"""J12: run a checkpoint (or base hobson) on every evalkit question (EK.all_question_items: 3227), one question per sequence,
hobson's layout, exact reference token ids (max_length 16384, no truncation). Resumable; shardable.
python eval_xr.py --ck PATH|base --out OUT.jsonl [--shard i/n] [--limit N] [--suites A,B]"""
import os, sys, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from xrlib import HX, Prep

ap = argparse.ArgumentParser(); ap.add_argument('--ck', default='base'); ap.add_argument('--out', required=True)
ap.add_argument('--shard', default='0/1'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--suites', default='')
ap.add_argument('--noxr', action='store_true'); ap.add_argument('--syn', default='')
a = ap.parse_args()
si, sn = map(int, a.shard.split('/'))
m = HX(); m.free_hf()
if a.ck != 'base':
    meta = m.load_all(os.path.expanduser(a.ck)); print('loaded', a.ck, meta, flush=True)
if a.noxr: m.xr_on = False
prep = Prep(m.p.eng)
if a.syn:
    items = [('SYN', r['id'], 'x', r['state'], r['questions']['x']) for r in map(json.loads, open(os.path.expanduser(a.syn)))]
else:
    items = list(EK.all_question_items(a.suites.split(',') if a.suites else None))
items = items[si::sn]
if a.limit: items = items[:a.limit]
done = set()
if os.path.exists(a.out):
    good = []
    for l in open(a.out):
        try: r = json.loads(l); done.add((r['id'], r['q'])); good.append(l)
        except Exception: pass
    open(a.out, 'w').writelines(good)
t0 = time.time(); n = 0
with open(a.out, 'a') as f, torch.inference_mode():
    for suite, iid, q, state, spec in items:
        if (iid, q) in done: continue
        pr = prep(state, spec)
        h = m.fwd(pr)
        p = torch.softmax(m.logits_pr(h, pr).float(), -1).tolist()
        rec = {'id': iid, 'q': q, 'suite': suite, 'p': {lab: p[i] for i, lab in enumerate(pr['rq'].slot_labels)}, 'T': pr['L']}
        if m.xr is not None and m.xr_on: rec['N'] = len(pr['s_lits']) + len(pr['q_lits'])
        f.write(json.dumps(rec) + '\n'); n += 1
        if n % 100 == 0:
            f.flush(); print(n, len(items), '%.0fs' % (time.time() - t0), flush=True)
print('done', n, time.time() - t0, flush=True)
