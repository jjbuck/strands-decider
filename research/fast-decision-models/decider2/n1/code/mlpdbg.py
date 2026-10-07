import os, sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import mlp_nki as MN
T = 256
torch.manual_seed(0)
x = (torch.randn(T, 2048) * 0.5).to(torch.bfloat16)
w1 = (1 + 0.1 * torch.randn(1, 2048)).float()
WguT = (torch.randn(2048, 12288) * 0.02).to(torch.bfloat16)
Wd = (torch.randn(6144, 2048) * 0.02).to(torch.bfloat16)
class Mk(nn.Module):
    def __init__(s, mode):
        super().__init__(); s.mode = mode
        s.register_buffer('w1', w1); s.register_buffer('WguP', MN.tile_wgu(WguT)); s.register_buffer('Wd', Wd)
        s.register_buffer('eye', torch.eye(128, dtype=torch.bfloat16))
    def forward(s, x):
        if s.mode == 'in': return MN.mlp_kernel(x, w1, MN.tile_wgu(WguT), Wd, torch.eye(128, dtype=torch.bfloat16))
        return MN.mlp_kernel(x, s.w1, s.WguP, s.Wd, s.eye)
xf = x.float(); h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w1)
gu = h @ WguT.float(); r = xf + (F.silu(gu[:, :6144]) * gu[:, 6144:]) @ Wd.float()
class Ma(nn.Module):
    def forward(s, x, w1, WguP, Wd, eye): return MN.mlp_kernel(x, w1, WguP, Wd, eye)
class Mc(nn.Module):   # constants passed through an op (forces materialisation in the default layout?)
    def __init__(s):
        super().__init__()
        s.register_buffer('w1', w1); s.register_buffer('WguP', MN.tile_wgu(WguT)); s.register_buffer('Wd', Wd)
        s.register_buffer('eye', torch.eye(128, dtype=torch.bfloat16))
    def forward(s, x): return MN.mlp_kernel(x, s.w1 * 1.0, s.WguP.contiguous(), s.Wd.contiguous(), s.eye.contiguous())
args = (x, w1, MN.tile_wgu(WguT), Wd, torch.eye(128, dtype=torch.bfloat16))
for mode in ['args', 'inline_false', 'const_op']:
    if mode == 'args':
        tr = torch_neuronx.trace(Ma(), args, compiler_args=['--model-type', 'transformer', '--auto-cast', 'none']); y = tr(*args).float()
    elif mode == 'inline_false':
        tr = torch_neuronx.trace(Mk('buf'), (x,), compiler_args=['--model-type', 'transformer', '--auto-cast', 'none'], inline_weights_to_neff=False); y = tr(x).float()
    else:
        tr = torch_neuronx.trace(Mc(), (x,), compiler_args=['--model-type', 'transformer', '--auto-cast', 'none']); y = tr(x).float()
    print(mode, 'err', float((y - r).abs().max()), 'vs x', float((y - xf).abs().max()), 'mlp part absmax', float((r - xf).abs().max()),
          'corr', float(torch.corrcoef(torch.stack([(y - xf).flatten(), (r - xf).flatten()]))[0, 1]), flush=True)
