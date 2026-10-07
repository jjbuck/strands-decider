"""ltest.py (inf2): in-situ cost of the GDN kernel inside a truncated hobson graph (first NL layers, real weights).
  python ltest.py MODEL NL L [tag]    MODEL: nl (J8 HobNL, kernel via HOB_KER) | n5 (HobN5, kernel via HOB_KMOD)
  env HOB_FAKEGDN=1 skips the GDN core.  Output rows = all L rows' final hidden state -> compared across runs (saved .pt).
"""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import torch, torch.nn as nn, torch_neuronx
import hob, hob5

MODEL, NL, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tag = sys.argv[4] if len(sys.argv) > 4 else f'{MODEL}_NL{NL}_L{L}' + ('_fake' if os.environ.get('HOB_FAKEGDN') == '1' else '')
OUT = os.path.expanduser('~/work/n1/ltest'); os.makedirs(OUT, exist_ok=True)
W, _, _ = hob.load_weights()
h = (hob5.HobN5(W) if MODEL == 'n5' else hob.HobNL(W)).eval(); del W
h.L = h.L[:NL]; h.types = h.types[:NL]


class Single(nn.Module):
    def __init__(s, h): super().__init__(); s.h = h
    def forward(s, ids, sel): return s.h(ids, sel)


LI = json.load(open(os.path.expanduser('~/work/n1/lat_inputs.json')))
ids = (LI['states']['1000'][0] + LI['questions'][0]['q'])[:L]
ids = ids + [0] * (L - len(ids))
ex = (torch.tensor(ids), torch.arange(0, L, 4))
t0 = time.time()
INL = os.environ.get('INLINE_W', '1') == '1'
tr = torch_neuronx.trace(Single(h), ex, compiler_args=['--model-type', 'transformer', '--auto-cast', 'none'],
                         compiler_workdir=f'{OUT}/wd_{tag}', inline_weights_to_neff=INL)
if not INL:
    torch_neuronx.move_trace_to_device(tr, int(os.environ.get('NDEV', '0')))
tc = time.time() - t0
for _ in range(3): y = tr(*ex)
ts = []
for _ in range(20):
    t = time.perf_counter(); y = tr(*ex); ts.append((time.perf_counter() - t) * 1000)
torch.save(y, f'{OUT}/y_{tag}.pt')
res = dict(tag=tag, NL=NL, L=L, compile_s=round(tc), median_ms=round(st.median(ts), 3), min_ms=round(min(ts), 3))
for other in sorted(os.listdir(OUT)):
    if other.startswith('y_') and other != f'y_{tag}.pt' and f'_NL{NL}_L{L}' in other and 'fake' not in other and 'fake' not in tag:
        z = torch.load(f'{OUT}/{other}')
        res['vs_' + other[2:-3]] = float((z - y).abs().max())
print(json.dumps(res), flush=True)
open(f'{OUT}/ltest.jsonl', 'a').write(json.dumps(res) + '\n')
