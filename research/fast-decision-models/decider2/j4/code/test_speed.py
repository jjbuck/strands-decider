import os, sys, time
sys.path[:0] = [os.path.expanduser('~/work/j4')]
import torch
from j4lib import J4
m = J4('base')
def bench(tag, ids, ckpt, lora=True, reps=3):
    if lora and m.lora is None: m.add_lora(16, 32); m.set_head()
    if not lora: m.lora = None
    for rep in range(reps):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        h, rsp, _ = m.forward(ids, ckpt=ckpt)
        loss = h.float().pow(2).mean()
        if lora: loss.backward()
        torch.cuda.synchronize(); dt = time.time() - t0
    n = sum(len(x) for x in ids)
    print(tag, 'tok', n, round(dt, 2), 's', round(n / dt), 'tok/s', 'mem', round(torch.cuda.max_memory_allocated() / 1e9, 1), flush=True)
ids8 = [list(range(1000, 1400))] * 20
ids4 = [list(range(1000, 1400))] * 10
bench('ckpt 8k', ids8, True)
bench('ckpt 4k', ids4, True)
bench('ckpt 16k', ids8 * 2, True)
bench('nockpt 4k', ids4, False)
with torch.no_grad():
    bench('fwd-only nolora 8k', ids8, False, lora=False)
m.lora = None; m.add_lora(16, 32)
with torch.no_grad():
    bench('fwd-only lora 8k', ids8, False, lora=True)
