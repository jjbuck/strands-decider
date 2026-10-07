"""d1 megakernel prototype: ONE persistent Triton launch executing L consecutive (folded-RMSNorm) SwiGLU MLP blocks
   x <- x + Wd_l( swiglu( rs_l(x) * (x @ Wgu'_l^T) ) )     (rs_l = rsqrt(mean(x^2)+eps), weights pre-scaled by (1+w))
with an on-GPU ticket scheduler over the task graph {A(l,r,n): gate_up tile, B(l,r,n): down tile} and per-(layer,row-block)
dependency counters, so layer l+1 tiles start as soon as THEIR rows are ready (no kernel boundary, no global barrier).
Baseline: the same Triton tiles as separate launches (2 per layer) in a CUDA graph.
usage: python mega.py T1,T2,.. L [grid_mult]
"""
import sys, os, json, time, statistics as st, torch, triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/work/d1"))
from lean2 import tgemm

@triton.jit
def _wait(ptr, target):
    c = tl.atomic_add(ptr, 0, sem="acquire")
    while c < target:
        c = tl.atomic_add(ptr, 0, sem="acquire")

@triton.jit
def mlp_chain_k(X, MB, WGU, WD, SS, CA, CB, TICK, T, eps, L,
                H: tl.constexpr, I2: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    I: tl.constexpr = I2 // 2
    nrb = tl.cdiv(T, BM)
    nNA: tl.constexpr = I2 // BN
    nNB: tl.constexpr = H // BN
    nA = nrb * nNA; nB = nrb * nNB; tpl = nA + nB
    total = L * tpl
    rk = tl.arange(0, BK)
    t = tl.atomic_add(TICK, 1)
    while t < total:
        l = t // tpl; u = t % tpl
        if u < nA:
            # grouped (GROUP row-blocks, n-major inside the group) order -> weight tile reuse in L2
            gsz = GROUP * nNA; g = u // gsz; fm = g * GROUP; gm = tl.minimum(nrb - fm, GROUP)
            r = fm + (u % gsz) % gm; n = (u % gsz) // gm
            if l > 0:
                _wait(CB + (l - 1) * nrb + r, nNB)
            tl.debug_barrier()
            rm = r * BM + tl.arange(0, BM); rn = n * BN + tl.arange(0, BN)
            a_ptr = X + rm[:, None].to(tl.int64) * H + rk[None, :]
            b_ptr = WGU + l.to(tl.int64) * I2 * H + rn[None, :].to(tl.int64) * H + rk[:, None]
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            for k in range(0, H, BK):
                a = tl.load(a_ptr, mask=rm[:, None] < T, other=0., cache_modifier=".cg")
                b = tl.load(b_ptr)
                acc = tl.dot(a, b, acc)
                a_ptr += BK; b_ptr += BK
            ss = tl.load(SS + l * T + rm, mask=rm < T, other=1.0, cache_modifier=".cg")
            acc = acc * tl.rsqrt(ss / H + eps)[:, None]
            gg, uu = tl.split(tl.reshape(acc, [BM, BN // 2, 2]))
            s = (gg * tl.sigmoid(gg)).to(tl.bfloat16).to(tl.float32)
            out = (s * uu.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
            cn = n * (BN // 2) + tl.arange(0, BN // 2)
            tl.store(MB + rm[:, None].to(tl.int64) * I + cn[None, :], out, mask=rm[:, None] < T)
            tl.debug_barrier()
            tl.atomic_add(CA + l * nrb + r, 1, sem="release")
        else:
            v = u - nA
            gsz = GROUP * nNB; g = v // gsz; fm = g * GROUP; gm = tl.minimum(nrb - fm, GROUP)
            r = fm + (v % gsz) % gm; n = (v % gsz) // gm
            _wait(CA + l * nrb + r, nNA)
            tl.debug_barrier()
            rm = r * BM + tl.arange(0, BM); rn = n * BN + tl.arange(0, BN)
            a_ptr = MB + rm[:, None].to(tl.int64) * I + rk[None, :]
            b_ptr = WD + l.to(tl.int64) * H * I + rn[None, :].to(tl.int64) * I + rk[:, None]
            acc = tl.zeros([BM, BN], dtype=tl.float32)
            for k in range(0, I, BK):
                a = tl.load(a_ptr, mask=rm[:, None] < T, other=0., cache_modifier=".cg")
                b = tl.load(b_ptr)
                acc = tl.dot(a, b, acc)
                a_ptr += BK; b_ptr += BK
            cp = rm[:, None].to(tl.int64) * H + rn[None, :]
            res = tl.load(X + cp, mask=rm[:, None] < T, other=0., cache_modifier=".cg").to(tl.float32)
            sx = (res + acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
            tl.store(X + cp, sx, mask=rm[:, None] < T)
            sf = sx.to(tl.float32)
            tl.atomic_add(SS + (l + 1) * T + rm, tl.sum(sf * sf, 1), mask=rm < T, sem="relaxed")
            tl.debug_barrier()
            tl.atomic_add(CB + l * nrb + r, 1, sem="release")
        t = tl.atomic_add(TICK, 1)

class Chain:
    def __init__(self, T, L, H=2048, I2=12288, BM=128, BN=128, BK=32, nw=4, ns=4, grid=80, dev="cuda", group=8):
        self.group = group
        self.T, self.L, self.H, self.I2 = T, L, H, I2
        self.cfg = (BM, BN, BK, nw, ns); self.grid = grid
        g = torch.Generator(device=dev).manual_seed(1)
        self.Wgu = (torch.randn(L, I2, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        self.Wd = (torch.randn(L, H, I2 // 2, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        self.x = torch.randn(T, H, device=dev).to(torch.bfloat16)
        self.m = torch.empty(T, I2 // 2, device=dev, dtype=torch.bfloat16)
        nrb = triton.cdiv(T, BM)
        self.ss = torch.zeros(L + 1, T, device=dev); self.ca = torch.zeros(L, nrb, device=dev, dtype=torch.int32)
        self.cb = torch.zeros(L, nrb, device=dev, dtype=torch.int32); self.tick = torch.zeros(1, device=dev, dtype=torch.int32)
    def persistent(self, x0):
        self.x.copy_(x0); self.ss.zero_(); self.ca.zero_(); self.cb.zero_(); self.tick.zero_()
        self.ss[0] = self.x.float().pow(2).sum(-1)
        BM, BN, BK, nw, ns = self.cfg
        mlp_chain_k[(self.grid,)](self.x, self.m, self.Wgu, self.Wd, self.ss, self.ca, self.cb, self.tick, self.T, 1e-6, self.L,
                                  H=self.H, I2=self.I2, BM=BM, BN=BN, BK=BK, GROUP=self.group, num_warps=nw, num_stages=ns)
        return self.x
    def sequential(self, x0):
        self.x.copy_(x0); self.ss.zero_()
        self.ss[0] = self.x.float().pow(2).sum(-1)
        for l in range(self.L):
            m = tgemm(self.x, self.Wgu[l], epi=1, ss=self.ss[l], cfg=self.cfg)
            tgemm(m, self.Wd[l], epi=3, res=self.x, ssout=self.ss[l + 1], cfg=self.cfg)
        return self.x

def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    return g, out

def tgraph(g, reps=20):
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return dict(median=round(st.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3))

if __name__ == "__main__":
    toks = [int(x) for x in sys.argv[1].split(",")]; L = int(sys.argv[2])
    grids = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "80").split(",")]
    groups = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "8").split(",")]
    res = {}
    for T in toks:
        for grid, group in [(a, b) for a in grids for b in groups]:
            c = Chain(T, L, grid=grid, group=group)
            x0 = (torch.randn(T, 2048, device="cuda") * 1.0).to(torch.bfloat16)
            ys = c.sequential(x0).clone(); yp = c.persistent(x0).clone(); torch.cuda.synchronize()
            rel = ((yp.float() - ys.float()).norm() / ys.float().norm()).item()
            gs, _ = capture(lambda: c.sequential(x0)); gp, _ = capture(lambda: c.persistent(x0))
            rs = tgraph(gs); rp = tgraph(gp)
            fl = 2 * T * L * (12288 * 2048 + 2048 * 6144)
            r = dict(seq=rs, pers=rp, rel_err=round(rel, 6), seq_tflops=round(fl / rs["median"] / 1e9, 1), pers_tflops=round(fl / rp["median"] / 1e9, 1),
                     speedup=round(rs["median"] / rp["median"], 4))
            res[f"T{T}_L{L}_g{grid}_G{group}"] = r
            print(f"T={T} L={L} grid={grid} group={group}", json.dumps(r), flush=True)
            del c, gs, gp; torch.cuda.empty_cache()
    json.dump(res, open(os.path.expanduser(f"~/work/d1/mega_L{L}.json"), "w"), indent=1)
