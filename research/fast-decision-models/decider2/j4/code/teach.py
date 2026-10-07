"""hobson-v19 (merged, its own head, calibrated logits / T_kind) on the FT rows, canonical option order, max_length 4096 (= the student's ids).
python teach.py ft.jsonl teacher.jsonl --n 40000"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/j4')]
import torch
from j4lib import J4
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--n', type=int, default=40000); ap.add_argument('--mtok', type=int, default=16384)
a = ap.parse_args()
m = J4('hobson'); m.head = m.hob_head; m.eng.model.config.max_length = 4096
done = set()
if os.path.exists(a.out):
    for l in open(a.out):
        try: done.add(json.loads(l)['i'])
        except Exception: pass
rows = [json.loads(l) for l in open(a.src)][:a.n]
todo = [r for r in rows if r['i'] not in done]
print('todo', len(todo), flush=True)
t0 = time.time(); n = 0; ntok = 0
with open(a.out, 'a') as f, torch.no_grad():
    j = 0
    while j < len(todo):
        mb = []; cur = 0
        while j < len(todo):
            p = m.prep(todo[j]['state'], todo[j]['q']); L = len(p['s']) + len(p['q'])
            if mb and cur + L > a.mtok: break
            mb.append((todo[j], p)); cur += L; j += 1
        h, rsp, _ = m.forward([p['s'] + p['q'] for _, p in mb])
        for (r, p), (r0, r1) in zip(mb, rsp):
            lg = m.pointer_logits(h, r1 - 1, [r0 + len(p['s']) + o for o in p['opt']])[:p['rq'].n_slots] / m.temp(p['rq'].kind)
            pr = torch.softmax(lg.float(), -1).tolist()
            f.write(json.dumps(dict(i=r['i'], labels=list(p['rq'].slot_labels), p=pr, T=len(p['s']) + len(p['q']))) + '\n')
        n += len(mb); ntok += cur
        if n % 2000 < len(mb): f.flush(); print(n, len(todo), f'{time.time() - t0:.0f}s', round(ntok / (time.time() - t0)), 'tok/s', flush=True)
print('done', n, f'{time.time() - t0:.0f}s', flush=True)
