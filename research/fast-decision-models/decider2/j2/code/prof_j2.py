import os, sys, time, random
sys.path[:0] = [os.path.expanduser('~/work/j2')]
import torch
from j2lib import J2
from h3lib import StdHead
m = J2(); m.detach_inference()
for d in m.L:
    for k, v in d.items():
        if torch.is_tensor(v): v.requires_grad_(False)
m.add_lora(r=16, alpha=32, seed=0); m.head = StdHead(m.head0).to(m.dev)
mode = sys.argv[1] if len(sys.argv) > 1 else 'qag'
m.setup_bidir(mode)
rng = random.Random(0)
L = int(sys.argv[2]) if len(sys.argv) > 2 else 1024; n = 8192 // L
rows = [dict(s=[rng.randrange(1000, 50000) for _ in range(L - 40)], q=[rng.randrange(1000, 50000) for _ in range(40)], opt=[5, 10], kind='noul', n_slots=2) for _ in range(n)]
def step():
    out = m.decide(rows, ckpt=True); sum(o.float().logsumexp(-1) for o in out).backward()
step(); torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    step(); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=70))
