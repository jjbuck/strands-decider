"""G2 latency: fused 2B runtime (d1/lean2, all fusions incl. folded RMSNorm) with PER-TOKEN NESTED WIDTH.
Every projection GEMM is replaced by one launch per width class of a row-indirect Triton GEMM (gather A rows / scatter C rows by an index list,
leading sub-block of the weight: in_proj N-segments limited to the class's heads, out_proj / down K limited, gate_up N limited).
Non-GEMM kernels (conv, GDN recurrence, norms, attention) run over all rows unchanged. Narrow rows get beta = 0 and no decay on skipped heads.
python g2lat.py   (exclusive --timing). CUDA graph per configuration, fresh ids each rep, 20 warm reps, median / p95."""
import os, sys, json, time, statistics
sys.path[:0] = [os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ['FUSE'] = 'addrms,gnorm,silu,prep,conv,gemm_swiglu,fold'
import torch, torch.nn.functional as F, triton, triton.language as tl
import lean2 as L2
import lean as LM
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


@triton.jit
def _ggemm_k(A, B, C, R, SS, SSOUT, RIDX, M, N, K, lda, ldb, eps, Kd, SEGEND, SEGW, NW,
             EPI: tl.constexpr, PRO_RS: tl.constexpr, SEG: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    gsz = GROUP * npn
    gi = pid // gsz; fm = gi * GROUP; gm = min(npm - fm, GROUP)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    rmask = rm < M
    rows = tl.load(RIDX + rm, mask=rmask, other=0).to(tl.int64)
    col0 = pn * BN
    Keff = K
    if SEG:
        Keff = tl.where((col0 < SEGEND) & ((col0 % SEGW) >= NW), 0, K)   # tile outside the class's heads: zeros
    a_ptr = A + rows[:, None] * lda + rk[None, :]
    b_ptr = B + rn[None, :].to(tl.int64) * ldb + rk[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, Keff, BK):
        a = tl.load(a_ptr, mask=rmask[:, None], other=0.)
        b = tl.load(b_ptr, mask=rn[None, :] < N, other=0.)
        acc = tl.dot(a, b, acc)
        a_ptr += BK; b_ptr += BK
    if PRO_RS:
        ss = tl.load(SS + rows, mask=rmask, other=1.0)
        acc = acc * tl.rsqrt(ss / Kd + eps)[:, None]
    if EPI == 1:
        gg, uu = tl.split(tl.reshape(acc, [BM, BN // 2, 2]))
        s = (gg * tl.sigmoid(gg)).to(tl.bfloat16).to(tl.float32)
        out = (s * uu.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        cn = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(C + rows[:, None] * NW + cn[None, :], out, mask=rmask[:, None] & (cn[None, :] < N // 2))
    elif EPI == 3:
        cp = rows[:, None] * N + rn[None, :]
        r = tl.load(R + cp, mask=rmask[:, None], other=0.).to(tl.float32)
        s = (r + acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(R + cp, s, mask=rmask[:, None])
        sf = s.to(tl.float32)
        tl.atomic_add(SSOUT + rows, tl.sum(sf * sf, 1), mask=rmask, sem="relaxed")
    else:
        tl.store(C + rows[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=rmask[:, None] & (rn[None, :] < N))


def ggemm(a, b, ridx, M, N, K, epi=0, out=None, res=None, ss=None, ssout=None, eps=1e-6, Kd=2048, seg=None, ldc_swiglu=None):
    """rows ridx[:M] of a (row stride a.stride(0)) times leading block b[:N, :K] (row stride b.stride(0)). out: full-T output buffer."""
    if M == 0: return
    BM, BN, BK, nw, ns = L2.pick_cfg(M, N, K)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    dummy = out if out is not None else res
    segend, segw, nwid = seg if seg else (0, 1, 0)
    NWarg = ldc_swiglu if epi == 1 else nwid
    _ggemm_k[grid](a, b, out if out is not None else dummy, res if res is not None else dummy, ss if ss is not None else dummy,
                   ssout if ssout is not None else dummy, ridx, M, N, K, a.stride(0), b.stride(0), eps, float(Kd), segend, segw, NWarg,
                   EPI=epi, PRO_RS=ss is not None, SEG=seg is not None, BM=BM, BN=BN, BK=BK, GROUP=8, num_warps=nw, num_stages=ns)


class Nest(L2.Lean2):
    """classes: list of (n_rows, w). Rows are assigned to classes by a fixed random permutation of the T rows (question rows in class 0)."""

    def setup(self, T, classes, nq):
        g = torch.Generator(device='cpu'); g.manual_seed(0)
        perm = torch.randperm(T - nq, generator=g)
        self.cls = []
        start = 0
        for j, (n, w) in enumerate(classes):
            idx = perm[start:start + n]; start += n
            if j == 0: idx = torch.cat([idx, torch.arange(T - nq, T)])
            self.cls.append((idx.to(torch.int32).to(self.dev).contiguous(), w))
        assert start == T - nq

    @torch.no_grad()
    def forward(self, ids):
        B, T = ids.shape
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.arange(T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        ss = torch.zeros(T, device=self.dev, dtype=torch.float32); ss += x.float().pow(2).sum(-1)
        for i, d in enumerate(self.layers):
            gdn = d["type"] == "linear_attention"
            Nin = d["Win_f"].shape[0]
            proj = torch.empty(T, Nin, device=self.dev, dtype=torch.bfloat16)
            for idx, w in self.cls:
                if gdn: seg = (8192, 2048, int(2048 * w))
                else: seg = (4096, 4096, int(4096 * w))
                ggemm(x, d["Win_f"], idx, idx.numel(), Nin, 2048, epi=0, out=proj, ss=ss, seg=None if w >= 1 else seg)
                if gdn and w < 1:   # skipped heads: beta = 0 (b = -inf) and no decay (a = -inf)
                    nh = int(16 * w)
                    proj[idx.long(), 8192 + nh:8208] = -1e4; proj[idx.long(), 8208 + nh:8224] = -1e4
            if gdn:
                qkv3 = L2.conv_l2(proj, d["conv_w"])
                q, k, v = qkv3[0][None], qkv3[1][None], qkv3[2][None]
                a = proj[:, 8208:8224].reshape(1, T, 16); b = proj[:, 8192:8208].reshape(1, T, 16)
                o, _ = chunk_gated_delta_rule(q, k, v, a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                              A_log=d["A_log"], dt_bias=d["dt_bias"], use_beta_sigmoid_in_kernel=True)
                o = L2.gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d["gn_w"], self.eps)
            else:
                q, k, gate = L2.attn_prep(proj, d["qn"], d["kn"], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                qh = q.reshape(1, T, 8, 256).transpose(1, 2); kh = k.reshape(1, T, 2, 256).transpose(1, 2); vh = v.reshape(1, T, 2, 256).transpose(1, 2)
                o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
                o = (o.transpose(1, 2).reshape(T, 2048) * gate).contiguous()
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            for idx, w in self.cls:
                ggemm(o, d["Wo"], idx, idx.numel(), 2048, int(2048 * w), epi=3, res=x, ssout=ss)
            I = d["I"]
            m = torch.empty(T, I, device=self.dev, dtype=torch.bfloat16)
            for idx, w in self.cls:
                ggemm(x, d["Wgu_f"], idx, idx.numel(), int(2 * I * w), 2048, epi=1, out=m, ss=ss, ldc_swiglu=I)
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            for idx, w in self.cls:
                ggemm(m, d["Wd"], idx, idx.numel(), 2048, int(I * w), epi=3, res=x, ssout=ss)
        h = LM._rms_zc(x, self.norm_w, self.eps)
        return h.reshape(1, T, -1)


def bench(fwd, T, reps=20, vocab=150000):
    pool = [torch.randint(1000, vocab, (1, T), device='cuda') for _ in range(4)]
    st = torch.empty(1, T, dtype=torch.long, device='cuda'); st.copy_(pool[0])
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fwd(st)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr): out = fwd(st)
    ts = []
    for r in range(reps + 5):
        st.copy_(pool[r % 4]); torch.cuda.synchronize(); t0 = time.perf_counter(); gr.replay(); torch.cuda.synchronize()
        if r >= 5: ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))], out


if __name__ == '__main__':
    from plib import P
    p = P()
    nest = Nest(p.tm)
    res = []
    NQ = 125
    for N in (1000, 4000):
        T = N + NQ
        # dense reference: lean2 fused forward
        md, p95, ref = bench(L2.Lean2.forward.__get__(nest), T)
        print(f'N={N} T={T} dense fused lean2: {md:.2f} ms (p95 {p95:.2f})', flush=True); res.append(dict(N=N, cfg='dense_lean2', ms=md, p95=p95))
        nest.setup(T, [(N, 1.0)], NQ)
        md2, p952, out = bench(nest.forward, T)
        print(f'  nested runtime, one full class: {md2:.2f} ms (p95 {p952:.2f})', flush=True); res.append(dict(N=N, cfg='nest_full', ms=md2, p95=p952))
        for name, split in (('10/90@1/4', [(.10, 1.0), (.90, .25)]), ('25/75@1/4', [(.25, 1.0), (.75, .25)]), ('10/20/70', [(.10, 1.0), (.20, .5), (.70, .25)]),
                            ('20/80@1/8', [(.20, 1.0), (.80, .125)]), ('10/90@1/8', [(.10, 1.0), (.90, .125)]), ('all@1/4', [(1.0, .25)])):
            cl = []; tot = 0
            for j, (f, w) in enumerate(split):
                n = N - tot if j == len(split) - 1 else int(round(f * N)); tot += n; cl.append((n, w))
            if split[0][1] < 1: cl = [(0, 1.0)] + cl
            nest.setup(T, cl, NQ)
            md3, p953, _ = bench(nest.forward, T)
            print(f'  nested {name}: {md3:.2f} ms (p95 {p953:.2f}) = {md3 / md:.3f} of dense', flush=True)
            res.append(dict(N=N, cfg=name, ms=md3, p95=p953, rel=md3 / md))
    json.dump(res, open(os.path.expanduser('~/work/g2/res_lat.json'), 'w'), indent=1)
