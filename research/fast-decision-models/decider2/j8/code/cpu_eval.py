"""cpu_eval.py (SPR): hob.py bf16 (eager, exact lengths, no padding) on evalkit items -> ~/work/j8/cpreds_<tag>.jsonl
  python cpu_eval.py TAG [subset_keys.json|all]"""
import os, sys, json, time
sys.path.insert(0, os.path.expanduser('~/work/j8'))
import torch
torch.set_num_threads(int(os.environ.get('NT', '16')))
import hob

D = os.path.expanduser('~/work/j8')
tag = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else f'{D}/subset_keys.json'
items = [json.loads(l) for l in open(f'{D}/ids.jsonl')]
if sub != 'all':
    keep = set(json.load(open(sub))); items = [it for it in items if f"{it['id']}|{it['qn']}" in keep]
W, hs, cfg = hob.load_weights()
m = hob.Hob(W, dtype=torch.bfloat16, C=int(os.environ.get('C', '64'))).eval(); del W
H = hob.Head(hs, cfg)
out = f'{D}/cpreds_{tag}.jsonl'
done = set()
if os.path.exists(out):
    for l in open(out):
        try: done.add(json.loads(l)['k'])
        except Exception: pass
t0 = time.time(); n = 0
with open(out, 'a') as f:
    for it in items:
        k = f"{it['id']}|{it['qn']}"
        if k in done: continue
        ids = torch.tensor(it['s'] + it['q'])
        sel = torch.tensor([len(it['s']) + o for o in it['opt']] + [len(ids) - 1])
        t = time.perf_counter()
        with torch.inference_mode():
            p = H.probs(m(ids, sel), it['kind'])
        f.write(json.dumps(dict(k=k, suite=it['suite'], id=it['id'], qn=it['qn'], p=dict(zip(it['labels'], p.tolist())), L=len(ids), ms=round((time.perf_counter() - t) * 1000, 1))) + '\n'); f.flush()
        n += 1
        if n % 50 == 0: print(n, len(items), '%.0fs' % (time.time() - t0), flush=True)
print('done', n, flush=True)
