import os, sys, time, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch._dynamo; torch._dynamo.config.recompile_limit = 64; torch._dynamo.config.cache_size_limit = 64
import bitnet_j10 as BJ
cfg, W = BJ.load(); m = BJ.BitNetTorso(cfg, W).cuda().train(); del W
B, T = int(sys.argv[1]), int(sys.argv[2])
ids = torch.randint(0, 120000, (B, T), device='cuda')
def step():
    h = m(ids); h.float().pow(2).mean().backward()
for _ in range(3): step()
torch.cuda.synchronize(); t = time.time()
for _ in range(3): step()
torch.cuda.synchronize(); dt = (time.time() - t) / 3
print(f'B{B} T{T}: {dt*1000:.0f} ms/step, {B*T/dt:.0f} tok/s, {B*T*8.34e9/dt/1e12:.1f} TFLOPS (2x fwd GEMM)', flush=True)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as p:
    step(); torch.cuda.synchronize()
print(p.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=70))
t0 = time.time(); m.refresh(); torch.cuda.synchronize(); print('refresh ms', (time.time() - t0) * 1000)
