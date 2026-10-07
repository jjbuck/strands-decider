import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe as M
from torch.profiler import profile, ProfilerActivity
g = M.load_cfg("st4b"); W = M.load_weights(g, layers=4)
t = M.MoETorso(g, W).cuda(); t.train(); t.tg = True
ids = torch.randint(0, 150000, (4, 512), device="cuda"); am = torch.ones_like(ids)
for _ in range(2):
    out = t(ids, am).last_hidden_state; out.float().pow(2).mean().backward()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    out = t(ids, am).last_hidden_state; out.float().pow(2).mean().backward(); torch.cuda.synchronize()
ka = p.key_averages()
print(ka.table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=50)[:4000])
tot_cuda = sum(e.self_device_time_total for e in ka) / 1e3
print("total self cuda ms", round(tot_cuda, 2))
torch.cuda.synchronize(); t0 = time.time()
out = t(ids, am).last_hidden_state; out.float().pow(2).mean().backward(); torch.cuda.synchronize(); print("wall ms", round((time.time() - t0) * 1e3, 2))
