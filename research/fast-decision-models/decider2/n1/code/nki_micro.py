"""nki_micro.py (inf2): time the NKI GDN kernel inside a traced graph vs CPU reference.  python nki_micro.py T"""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import numpy as np, torch, torch.nn as nn
import torch_neuronx
import gdn_nki as G
import hob

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1152
H = 16
torch.manual_seed(0)
k = hob.l2n(torch.randn(H, T, 128) + 2 * torch.randn(H, 1, 128)); q = hob.l2n(torch.randn(H, T, 128)) * 128 ** -0.5
v = torch.randn(H, T, 128); g = -torch.rand(H, T, 1) * 0.5; b = torch.rand(H, T, 1)


class M(nn.Module):
    def __init__(s):
        super().__init__(); s.register_buffer('cst', torch.from_numpy(G.consts_np()))
    def forward(s, q, k, v, g, b):
        KER = {'2': G.gdn_kernel2, '3': G.gdn_kernel3}.get(os.environ.get('KER', '1'), G.gdn_kernel)
        return KER[H](q, k, v, g, b, s.cst, torch.zeros(H, 128, 128, device=q.device))[0]


t0 = time.time()
tr = torch_neuronx.trace(M(), (q, k, v, g, b), compiler_args=['--model-type', 'transformer', '--auto-cast', 'none'],
                         compiler_workdir=os.path.expanduser(f'~/work/n1/neff/wdnki_{T}_k' + os.environ.get('KER', '1')))
tc = time.time() - t0
for _ in range(3): tr(q, k, v, g, b)
ts = []
for _ in range(20):
    t = time.perf_counter(); o = tr(q, k, v, g, b); ts.append((time.perf_counter() - t) * 1000)
hob.TRI = 'bd'; hob.SCAN = False
with torch.inference_mode():
    ref, _ = hob.gdn_chunk(q[None], k[None], v[None], g[None, :, :, 0], b[None, :, :, 0], 128)
err = float((o - ref[0]).abs().max())
out = dict(part='nki_gdn' + os.environ.get('KER', '1'), T=T, compile_s=round(tc), median_ms=round(st.median(ts), 3), min_ms=round(min(ts), 3), max_abs_err=err, ref_absmax=float(ref.abs().max()))
print(json.dumps(out), flush=True)
open(os.path.expanduser('~/work/n1/nmicro.jsonl'), 'a').write(json.dumps(out) + '\n')
