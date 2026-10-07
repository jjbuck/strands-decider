"""mlptest.py (inf2): NKI fused MLP vs the XLA MLP (HobN5.mlpT) at one shape, random weights.  python mlptest.py T"""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
import mlp_nki as MN

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1152
torch.manual_seed(0)
x = (torch.randn(T, 2048) * 0.5).to(torch.bfloat16)
w1 = (1 + 0.1 * torch.randn(1, 2048)).float()
WguT = (torch.randn(2048, 12288) * 0.02).to(torch.bfloat16)
Wd = (torch.randn(6144, 2048) * 0.02).to(torch.bfloat16)


class Mk(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer('w1', w1); s.register_buffer('WguP', MN.tile_wgu(WguT)); s.register_buffer('Wd', Wd)
        s.register_buffer('eye', torch.eye(128, dtype=torch.bfloat16))
    def forward(s, x): return MN.mlp_kernel(x, s.w1, s.WguP, s.Wd, s.eye)


class Mx(nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer('w1', w1); s.register_buffer('WguT', WguT); s.register_buffer('Wd', Wd)
    def forward(s, x):
        xf = x.float(); h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * s.w1).to(x.dtype)
        gu = h @ s.WguT
        return x + (F.silu(gu[:, :6144]) * gu[:, 6144:]) @ s.Wd


def ref():
    xf = x.float(); h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w1)
    gu = h @ WguT.float()
    return xf + (F.silu(gu[:, :6144]) * gu[:, 6144:]) @ Wd.float()


r = ref()
res = dict(T=T)
for name, M in [('nki', Mk), ('xla', Mx)]:
    t0 = time.time()
    tr = torch_neuronx.trace(M(), (x,), compiler_args=['--model-type', 'transformer', '--auto-cast', 'none'],
                             compiler_workdir=os.path.expanduser(f'~/work/n1/mlpt/wd_{name}_{T}'))
    tc = time.time() - t0
    for _ in range(3): y = tr(x)
    ts = []
    for _ in range(20):
        t = time.perf_counter(); y = tr(x); ts.append((time.perf_counter() - t) * 1000)
    err = (y.float() - r).abs(); rel = float(err.max() / r.abs().max())
    res[name] = dict(compile_s=round(tc), median_ms=round(st.median(ts), 3), max_abs_err=float(err.max()), rel_to_absmax=rel)
    print(json.dumps({name: res[name]}), flush=True)
print(json.dumps(res))
open(os.path.expanduser('~/work/n1/mlptest.jsonl'), 'a').write(json.dumps(res) + '\n')
