import os, sys, numpy as np, torch, torch.nn as nn, torch_neuronx
sys.path.insert(0, os.path.expanduser('~/work/j8'))
import neuronxcc.nki as nki, neuronxcc.nki.language as nl, neuronxcc.nki.isa as nisa
import gdn_nki as G
C = 128

@nki.jit
def dbg_kernel(q, k, v, g, beta, cst):
    out = nl.ndarray((9, C, C), dtype=nl.float32, buffer=nl.shared_hbm)
    U = nl.load(cst[0]); incl = nl.load(cst[1]); strict = nl.load(cst[2]); ones = nl.load(cst[4]); eye = nl.load(cst[5]); m1 = nl.load(cst[12])
    kc = nl.load(k[0, 0:C, :]); qc = nl.load(q[0, 0:C, :]); vc = nl.load(v[0, 0:C, :])
    gc = nl.load(g[0, 0:C, :]); bc = nl.load(beta[0, 0:C, :])
    gcum = G._mm(U, gc)
    Gb = G._mm(nl.multiply(ones, gc), U)
    dg = nl.subtract(gcum, Gb)
    decay = nl.multiply(nl.exp(nl.multiply(dg, incl)), incl)
    kT = G._tr(kc)
    KK = G._mm(kT, kT)
    L = nl.multiply(nl.multiply(nl.multiply(KK, decay), strict), bc)
    masks = nl.ndarray((6, nl.par_dim(C), C), dtype=nl.float32, buffer=nl.sbuf)
    for lv in nl.static_range(6):
        masks[lv] = nl.load(cst[6 + lv])
    Dinv = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    DinvT = nl.ndarray((C, C), dtype=nl.float32, buffer=nl.sbuf)
    Dinv[...] = nl.subtract(eye, nl.multiply(L, m1))
    DinvT[...] = G._tr(Dinv)
    nl.store(out[0], value=Dinv)
    for lv in nl.static_range(6):
        Lb = nl.multiply(L, masks[lv])
        LbT = G._tr(Lb)
        Z = G._mm(LbT, Dinv)
        Y = G._mm(DinvT, Z)
        Dinv[...] = nl.subtract(Dinv, Y)
        DinvT[...] = G._tr(Dinv)
        if lv == 0:
            nl.store(out[1], value=Lb)
            nl.store(out[2], value=Z)
            nl.store(out[3], value=Y)
    nl.store(out[4], value=Dinv)
    nl.store(out[5], value=DinvT)
    nl.store(out[6], value=nl.multiply(L, masks[0]))
    nl.store(out[7], value=masks[1])
    nl.store(out[8], value=L)
    return out

rng = np.random.default_rng(0)
def l2(x): return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
k = l2(rng.standard_normal((1, C, C))).astype(np.float32); q = k.copy(); v = rng.standard_normal((1, C, C)).astype(np.float32)
g = (-rng.random((1, C, 1)) * 0.5).astype(np.float32); b = rng.random((1, C, 1)).astype(np.float32)
cst = G.consts_np()
sim = nki.simulate_kernel(dbg_kernel, q, k, v, g, b, cst)
class M(nn.Module):
    def __init__(s): super().__init__(); s.register_buffer('cst', torch.from_numpy(cst))
    def forward(s, q, k, v, g, b): return dbg_kernel(q, k, v, g, b, s.cst)
tt = [torch.from_numpy(x) for x in (q, k, v, g, b)]
tr = torch_neuronx.trace(M(), tuple(tt), compiler_args=['--auto-cast', 'none'], compiler_workdir=os.path.expanduser('~/work/j8/neff/wdnkidbg'))
hw = tr(*tt).numpy()
names = ['Dinv0', 'Lb0', 'Z0', 'Y0', 'Dinv', 'DinvT', 'L*mask0', 'mask1', 'L']
for i, nm in enumerate(names):
    print(nm, 'max|hw-sim|', float(np.abs(hw[i] - sim[i]).max()), 'sim absmax', float(np.abs(sim[i]).max()))
