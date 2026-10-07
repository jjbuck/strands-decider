import os, sys, torch, torch.nn as nn, torch.nn.functional as F, torch_neuronx
sys.path.insert(0, os.path.expanduser('~/work/n1'))
import mlp_nki as MN
T = 256
torch.manual_seed(0)
x = (torch.randn(T, 2048) * 0.5).to(torch.bfloat16)
w1 = (1 + 0.1 * torch.randn(1, 2048)).float()
WguT = (torch.randn(2048, 12288) * 0.02).to(torch.bfloat16)
Wd = (torch.randn(6144, 2048) * 0.02).to(torch.bfloat16)
xf = x.float(); h = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w1)
gu = h @ WguT.float(); a = F.silu(gu[:, :6144]) * gu[:, 6144:]
for w2d, perk in [(True, False), (False, True), (True, True)]:
    MN.OPT['w2d'] = w2d; MN.OPT['hT_perk'] = perk
    Wg = MN.tile_wgu(WguT)
    if w2d: Wg = Wg.reshape(96 * 128, 16 * 128).contiguous()
    args = (x, w1, Wg, Wd, torch.eye(128, dtype=torch.bfloat16))
    class Ma(nn.Module):
        def forward(s, *a): return MN.mlp_kernel_aT(*a)
    tr = torch_neuronx.trace(Ma(), args, compiler_args=['--model-type', 'transformer', '--auto-cast', 'none'])
    y = tr(*args).float()
    print('w2d', w2d, 'perk', perk, 'aT err', float((y - a[:, :2048]).abs().max()), flush=True)
