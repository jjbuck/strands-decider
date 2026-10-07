import os, sys, json, time, torch
sys.path.insert(0, os.path.expanduser('~/work/j8'))
torch.set_num_threads(16)
import hob
torch.manual_seed(0)
# random but realistic GDN inputs: unit keys, beta in (0,1), decays
H, T, D = 16, 1024, 128
k = hob.l2n(torch.randn(1, H, T, D)); q = hob.l2n(torch.randn(1, H, T, D)) * D ** -0.5; v = torch.randn(1, H, T, D)
beta = torch.rand(1, H, T); g = -torch.rand(1, H, T) * 0.5
# make keys correlated (worst case for the inverse)
k = hob.l2n(k + 2 * torch.randn(1, H, 1, D))
res = {}
for tri in ('bd', 'scan'):
    hob.TRI = 'bd'; hob.SCAN = tri == 'scan'
    for C in (64, 128):
        hob.gdn_chunk(q, k, v, g, beta, C)
        t = time.perf_counter(); o, S = hob.gdn_chunk(q, k, v, g, beta, C); ms = (time.perf_counter() - t) * 1000
        S0 = torch.randn(1, H, D, D) * 0.1
        o2, S2 = hob.gdn_chunk(q[:, :, :256], k[:, :, :256], v[:, :, :256], g[:, :, :256], beta[:, :, :256], C, S0=S0)
        res[(tri, C, 'S0')] = (o2, S2)
        res[(tri, C)] = o
        print(tri, C, 'ms %.1f' % ms, 'absmax', float(o.abs().max()), flush=True)
# exact recurrent reference in fp64
qd, kd, vd, bd_, gd = (x.double() for x in (q, k, v, beta, g))
S = torch.zeros(1, H, D, D, dtype=torch.float64); outs = []
for t in range(T):
    S = S * gd[..., t].exp()[..., None, None]
    kt = kd[:, :, t]; vt = vd[:, :, t]
    pred = torch.einsum('bhk,bhkv->bhv', kt, S)
    S = S + torch.einsum('bhk,bhv->bhkv', kt, (vt - pred) * bd_[..., t, None])
    outs.append(torch.einsum('bhk,bhkv->bhv', qd[:, :, t], S))
ref = torch.stack(outs, 2)
S = torch.zeros(1, H, D, D, dtype=torch.float64) + S0.double(); o2r = []
for t in range(256):
    S = S * gd[..., t].exp()[..., None, None]
    kt = kd[:, :, t]; vt = vd[:, :, t]
    pred = torch.einsum('bhk,bhkv->bhv', kt, S)
    S = S + torch.einsum('bhk,bhv->bhkv', kt, (vt - pred) * bd_[..., t, None])
    o2r.append(torch.einsum('bhk,bhkv->bhv', qd[:, :, t], S))
o2r = torch.stack(o2r, 2)
for kk, o in list(res.items()):
    if kk[-1] == 'S0':
        print(kk, 'S0 case: o err', float((o[0].double() - o2r).abs().max()), 'S err', float((o[1].double() - S).abs().max())); continue
    print(kk, 'max abs err vs fp64 recurrent', float((o.double() - ref).abs().max()), 'ref absmax', float(ref.abs().max()))
