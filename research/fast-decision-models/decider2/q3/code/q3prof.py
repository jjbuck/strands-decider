"""profile one student fwd+bwd (qad mode) on a ~2k-row request; prints the top CUDA ops"""
import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/q3')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import q3lib as QL
from torch.profiler import profile, ProfilerActivity
m = QL.Q3(); QL.FAST = True
pool, devset, v5 = QL.load_data(5000, 5, 0)
init = torch.load(os.path.expanduser('~/work/q3/gptq4.pt'), map_location='cpu')
params = m.set_student('lat', {k: dict(q=e['q'], s=e['s'], Wc=e['Wc']) for k, e in init.items()}); del init
rs = [r for r in pool[:60] if len(r['questions']) >= 3 and 1500 < r['n'] < 2500][:3]
rqs = [m.prep(r['state'], [r['questions'][n] for n in sorted(r['questions'])[:4]]) for r in rs]


def step(rq):
    with torch.no_grad(): (lt, kt), _ = m.run(rq, student=False)
    (ls, ks), lay = m.run(rq, student=True)
    loss, kl, hid = QL.loss_fn(lt, kt, ls, ks, 0.5, None); loss.backward()
    for p_ in params: p_.grad = None
    return lay['T']


import time
for rq in rqs[:2]: step(rq)
for rq in rqs:
    torch.cuda.synchronize(); t0 = time.time(); T = step(rq); torch.cuda.synchronize(); print('step T', T, f'{time.time()-t0:.3f}s', f'{T/(time.time()-t0):.0f} tok/s', flush=True)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
    T = step(rqs[2]); torch.cuda.synchronize()
print('T', T)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=40))
