import os, sys, json, torch
sys.path.insert(0, os.path.expanduser('~/work/j8'))
torch.set_num_threads(16)
import hob
import torch._inductor.config as ic
ic.freezing = True
from torch.profiler import profile, ProfilerActivity
T = int(os.environ.get('T', '1000'))
LI = json.load(open(os.path.expanduser('~/work/j8/lat_inputs.json')))
W, hs, cfg = hob.load_weights()
m = hob.Hob(W, dtype=torch.bfloat16, C=int(os.environ.get('C', '64'))).eval(); del W
f = torch.compile(m.forward, dynamic=False)
q = LI['questions'][0]; ids = torch.tensor(LI['states'][str(T)][0] + q['q']); sel = torch.tensor([len(ids) - 1])
with torch.inference_mode():
    for _ in range(3): f(ids, sel)
    with profile(activities=[ProfilerActivity.CPU]) as p:
        f(ids, sel)
print(p.key_averages().table(sort_by='self_cpu_time_total', row_limit=22))
