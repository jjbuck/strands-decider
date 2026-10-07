"""OVL correctness: schemamix k48 forward with the bf16 slot rows on a side stream (eager and CUDA graph) vs serial."""
import sys, os, torch
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
from prof_d1 import capture
from lean2 import Lean2
import qrt6 as Q6
from qrt import Lay
from bench import load_model
torso, head = load_model(); ln = Lean2(torso, fuse=''); del torso
for d in ln.layers:
    for k_ in ('Wgu_il', 'Win_f', 'Wgu_f'): d.pop(k_, None)
torch.cuda.empty_cache()
m = Q6.QRT6(ln, head=head, prec='map:~/work/h2/precmap_w4a4_k48.json', wcodes=Q6.load_codes('w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt'), split=True, slim=True)
m.tune = False
bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt')); q = bundles['1q'][0]
pre = torch.tensor(q['ids'][:-3], device='cuda'); Ts = 1000
cache = m.compile_prefix(pre, Ts + 3, r0=0)
ids = torch.cat([torch.randint(1000, 100000, (Ts,), device='cuda'), torch.tensor(q['ids'][-3:], device='cuda')])
lay = Lay('schema', Ts, nslots=3, P=pre.shape[0])
with torch.inference_mode():
    h0 = m.forward(ids, lay, cache, r0=Ts).float().clone()
    m.ovl = True; m.s2 = torch.cuda.Stream()
    h1 = m.forward(ids, lay, cache, r0=Ts).float().clone()
print('eager ovl vs serial max|d|', (h0 - h1).abs().max().item(), flush=True)
def run(): return m.forward(ids, lay, cache, r0=Ts)
g, out = capture(run)
g.replay(); torch.cuda.synchronize()
print('graph ovl vs serial max|d|', (out.float() - h0).abs().max().item(), 'slot rows', (out.float()[-3:] - h0[-3:]).abs().max().item(), flush=True)
import time
for ov in (True, False):
    m.ovl = ov
    g, out = capture(run)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); print('ovl', ov, f'{(time.perf_counter() - t0) / 10 * 1000:.2f} ms (shared GPU: indicative only)', flush=True)
