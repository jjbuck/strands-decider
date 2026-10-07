import os, sys, torch
sys.path.insert(0, os.path.expanduser('~/work/j8'))
import hob
torch.manual_seed(0)
H, T, D = 2, 256, 128
k = hob.l2n(torch.randn(1, H, T, D)); q = hob.l2n(torch.randn(1, H, T, D)) * D ** -0.5; v = torch.randn(1, H, T, D)
beta = torch.rand(1, H, T); g = -torch.rand(1, H, T) * 0.5
S0 = torch.randn(1, H, D, D) * 0.1
qd, kd, vd, bd_, gd = (x.double() for x in (q, k, v, beta, g))
S = S0.double().clone(); o2r = []
for t in range(T):
    S = S * gd[..., t].exp()[..., None, None]
    pred = torch.einsum('bhk,bhkv->bhv', kd[:, :, t], S)
    S = S + torch.einsum('bhk,bhv->bhkv', kd[:, :, t], (vd[:, :, t] - pred) * bd_[..., t, None])
    o2r.append(torch.einsum('bhk,bhkv->bhv', qd[:, :, t], S))
o2r = torch.stack(o2r, 2)
for scan in (False, True):
    for C in (64, 128):
        hob.SCAN = scan
        o, S2 = hob.gdn_chunk(q, k, v, g, beta, C, S0=S0)
        e = (o.double() - o2r).abs().amax(dim=(0, 1, 3))
        print('scan', scan, 'C', C, 'err by chunk', [round(float(e[i:i + C].max()), 6) for i in range(0, T, C)], 'S err', float((S2.double() - S).abs().max()))
