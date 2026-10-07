"""J13 step 4: mixed-width layers in the d1 lean2 fused runtime (norm folding, SwiGLU epilogue, fused glue kernels).

Layers < k run exactly as lean2 (fold).  At layer k the residual is permuted ONCE into sorted order [full rows | thin rows]
(full = question rows + sinks + routed chunks, each block in sequence order).  Every layer >= k then runs:
  Win : grouped GEMM {full rows @ Win_f -> scattered into sequence-order proj ; thin rows @ Bin_f -> z}  then  z @ Ain -> scattered
  mixer (conv/GDN or attention) unchanged on sequence-order proj
  Wo  : grouped GEMM {o[perm full] @ Wo -> residual(+sumsq) ; o[perm thin] @ Bo -> z}  then  z @ Ao -> residual(+sumsq)
  MLP : grouped GEMM {full @ Wgu_f (SwiGLU epi) -> m ; thin @ Bgu_f -> z}  then  z @ Agu_il (SwiGLU epi) -> m[thin]
        grouped GEMM {m[full] @ Wd -> residual ; m[thin] @ Bd -> z}  then  z @ Ad -> residual
Row gather (A operand) and row scatter (C operand) happen inside the GEMM kernels through index arrays: no extra memory passes.
RMSNorm row scales are applied in the GEMM epilogue (rsqrt(sum sq / K + eps)), so the thin first-stage GEMMs also read the
raw residual.  Thin-path weights: B_f = (U^T W) diag(1 + norm_w)  [r x K],  A = U [N x r]  (out-mode bases)."""
import os, sys, math, torch, torch.nn.functional as F
import triton, triton.language as tl
sys.path[:0] = [os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import lean as LM
from lean2 import Lean2, tgemm, conv_l2, gnorm, attn_prep, chunk_gated_delta_rule


@triton.jit
def _tile(pid, A, B, C, SS, SSOUT, AIDX, CIDX, M, N, K, a_off, c_off, eps, Kd,
          EPI: tl.constexpr, PRO_RS: tl.constexpr, GA: tl.constexpr, SC: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    gsz = GROUP * npn
    g = pid // gsz; fm = g * GROUP; gm = min(npm - fm, GROUP)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    mm = rm < M
    if GA:
        ar = tl.load(AIDX + rm, mask=mm, other=0).to(tl.int64)
    else:
        ar = (rm + a_off).to(tl.int64)
    a_ptr = A + ar[:, None] * K + rk[None, :]
    b_ptr = B + rn[None, :].to(tl.int64) * K + rk[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr, mask=mm[:, None], other=0.)
        b = tl.load(b_ptr, mask=rn[None, :] < N, other=0.)
        acc = tl.dot(a, b, acc)
        a_ptr += BK; b_ptr += BK
    if PRO_RS:
        ss = tl.load(SS + rm + a_off, mask=mm, other=1.0)
        acc = acc * tl.rsqrt(ss / Kd + eps)[:, None]
    if SC:
        cr = tl.load(CIDX + rm, mask=mm, other=0).to(tl.int64)
    else:
        cr = (rm + c_off).to(tl.int64)
    if EPI == 1:
        gg, uu = tl.split(tl.reshape(acc, [BM, BN // 2, 2]))
        s = (gg * tl.sigmoid(gg)).to(tl.bfloat16).to(tl.float32)
        out = (s * uu.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        cn = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(C + cr[:, None] * (N // 2) + cn[None, :], out, mask=mm[:, None])
    elif EPI == 3:
        cp = cr[:, None] * N + rn[None, :]
        r = tl.load(C + cp, mask=mm[:, None], other=0.).to(tl.float32)
        s = (r + acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(C + cp, s, mask=mm[:, None])
        sf = s.to(tl.float32)
        tl.atomic_add(SSOUT + cr, tl.sum(sf * sf, 1), mask=mm, sem="relaxed")
    else:
        tl.store(C + cr[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=mm[:, None] & (rn[None, :] < N))


@triton.jit
def _ggemm_k(n1,
             A1, B1, C1, S1, SO1, AI1, CI1, M1, N1, K1, ao1, co1,
             A2, B2, C2, S2, SO2, AI2, CI2, M2, N2, K2, ao2, co2,
             eps, Kd1, Kd2,
             EPI1: tl.constexpr, PRS1: tl.constexpr, GA1: tl.constexpr, SC1: tl.constexpr,
             EPI2: tl.constexpr, PRS2: tl.constexpr, GA2: tl.constexpr, SC2: tl.constexpr, TWO: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    if pid < n1:
        _tile(pid, A1, B1, C1, S1, SO1, AI1, CI1, M1, N1, K1, ao1, co1, eps, Kd1, EPI1, PRS1, GA1, SC1, BM, BN, BK, GROUP)
    else:
        if TWO:
            _tile(pid - n1, A2, B2, C2, S2, SO2, AI2, CI2, M2, N2, K2, ao2, co2, eps, Kd2, EPI2, PRS2, GA2, SC2, BM, BN, BK, GROUP)


class Prob:
    __slots__ = ('A', 'B', 'C', 'M', 'epi', 'ss', 'ssout', 'aidx', 'cidx', 'a_off', 'c_off', 'Kd')

    def __init__(self, A, B, C, M, epi=0, ss=None, ssout=None, aidx=None, cidx=None, a_off=0, c_off=0, Kd=2048.0):
        self.A, self.B, self.C, self.M, self.epi, self.ss, self.ssout = A, B, C, M, epi, ss, ssout
        self.aidx, self.cidx, self.a_off, self.c_off, self.Kd = aidx, cidx, a_off, c_off, Kd

    def args(self):
        d = self.C
        N, K = self.B.shape
        return [self.A, self.B, self.C, self.ss if self.ss is not None else d, self.ssout if self.ssout is not None else d,
                self.aidx if self.aidx is not None else d, self.cidx if self.cidx is not None else d,
                self.M, N, K, self.a_off, self.c_off]

    def flags(self):
        return [self.epi, self.ss is not None, self.aidx is not None, self.cidx is not None]

    def tiles(self, BM, BN):
        return triton.cdiv(self.M, BM) * triton.cdiv(self.B.shape[0], BN)


CFGS = [(128, 128, 32, 4, 4), (128, 64, 64, 4, 4), (64, 128, 64, 4, 3), (256, 128, 32, 8, 3), (64, 64, 64, 4, 4)]


def ggemm(p1, p2=None, cfg=(128, 128, 32, 4, 4), eps=1e-6):
    BM, BN, BK, nw, ns = cfg
    n1 = p1.tiles(BM, BN); n2 = p2.tiles(BM, BN) if p2 is not None and p2.M > 0 else 0
    if p1.M == 0 and p2 is not None:
        return ggemm(p2, None, cfg, eps)
    two = n2 > 0
    if n1 + n2 == 0: return
    q = p2 if two else p1
    _ggemm_k[(n1 + n2,)](n1, *p1.args(), *q.args(), eps, float(p1.Kd), float(q.Kd),
                         *p1.flags(), *q.flags(), two, BM=BM, BN=BN, BK=BK, GROUP=8, num_warps=nw, num_stages=ns)


class Tuner:
    """pick the fastest tile config per call-site signature (measured once, outside graph capture)"""
    def __init__(self, on=True):
        self.best = {}; self.on = on

    def __call__(self, key, p1, p2=None):
        if key not in self.best:
            if not self.on:
                self.best[key] = CFGS[0]
            else:
                ts = []
                for c in CFGS:
                    if c[1] % 2 and p1.epi == 1: continue
                    try:
                        for _ in range(2): ggemm(p1, p2, c)
                        torch.cuda.synchronize()
                        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                        e0.record()
                        for _ in range(5): ggemm(p1, p2, c)
                        e1.record(); torch.cuda.synchronize(); ts.append((e0.elapsed_time(e1), c))
                    except Exception as ex:
                        pass
                self.best[key] = min(ts)[1]
        ggemm(p1, p2, self.best[key])


class Mixed:
    def __init__(self, R: Lean2, k, r, U=None, seed=0, segs=None):
        """U: per-layer dict {'in','o','gu','d'} -> [N, >=r] orthonormal output bases (None: random orthonormal, latency only).
        segs: optional depth schedule [(start, r), ...] (first start = k)"""
        self.R = R; self.k = k; dev = R.dev
        segs = segs or [(k, r)]
        self.rl = {i: [rr for st, rr in segs if i >= st][-1] for i in range(k, len(R.layers))}
        self.thin = {}
        g = torch.Generator(device='cpu').manual_seed(seed)
        with torch.no_grad():
            for i in range(k, len(R.layers)):
                d = R.layers[i]; r = self.rl[i]
                Wl = {'in': d['Win'], 'o': d['Wo'], 'gu': torch.cat([d['Wgu'][:d['I']], d['Wgu'][d['I']:]], 0), 'd': d['Wd']}
                t = {}
                for nm, W in Wl.items():
                    if U is not None:
                        Ur = U[i][nm][:, :r].to(dev, torch.float32)
                    else:
                        Ur, _ = torch.linalg.qr(torch.randn(W.shape[0], r, generator=g).to(dev))
                    B = Ur.t() @ W.float()                                   # [r, K]
                    if nm == 'in': B = B * d['in1'][None, :]
                    if nm == 'gu': B = B * d['post1'][None, :]
                    A = Ur
                    if nm == 'gu':
                        I = d['I']; A = torch.stack([Ur[:I], Ur[I:]], 1).reshape(2 * I, r)   # interleaved g0,u0,g1,u1 for the SwiGLU epilogue
                    t['B' + nm] = B.to(torch.bfloat16).contiguous(); t['A' + nm] = A.to(torch.bfloat16).contiguous()
                self.thin[i] = t
        self.tune = Tuner()

    def dense_layer(self, i, x, ss, cos, sin, T):
        R = self.R; d = R.layers[i]
        proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
        o = self.mixer(i, proj, cos, sin, T)
        ss = torch.zeros(T, device=R.dev, dtype=torch.float32)
        tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
        m = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
        ss = torch.zeros(T, device=R.dev, dtype=torch.float32)
        tgemm(m, d['Wd'], epi=3, res=x, ssout=ss)
        return x, ss

    def mixer(self, i, proj, cos, sin, T):
        R = self.R; d = R.layers[i]
        if d['type'] == 'linear_attention':
            qkv3 = conv_l2(proj, d['conv_w'])
            q, k, v = qkv3[0][None], qkv3[1][None], qkv3[2][None]
            aa = proj[:, 8208:8224].reshape(1, T, 16); bb = proj[:, 8192:8208].reshape(1, T, 16)
            o, _ = chunk_gated_delta_rule(q, k, v, aa, bb, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                          A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True)
            return gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], R.eps)
        q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, R.eps)
        v = proj[:, 4608:5120].reshape(T, 2, 256)
        qh = q.reshape(1, T, 8, 256).transpose(1, 2); kh = k.reshape(1, T, 2, 256).transpose(1, 2); vh = v.reshape(1, T, 2, 256).transpose(1, 2)
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
        return o.transpose(1, 2).reshape(T, 2048) * gate

    def mixed_layer(self, i, xs, ss, cos, sin, T, perm, Mf):
        R = self.R; d = R.layers[i]; t = self.thin[i]; dev = R.dev; r = self.rl[i]
        Mt = T - Mf; pf, pt = perm[:Mf], perm[Mf:]
        bf = torch.bfloat16
        Nin = d['Win_f'].shape[0]
        proj = torch.empty(T, Nin, device=dev, dtype=bf); z = torch.empty(Mt, r, device=dev, dtype=bf)
        self.tune(('in', i % 4 == 3, T, Mf, r), Prob(xs, d['Win_f'], proj, Mf, ss=ss, cidx=pf), Prob(xs, t['Bin'], z, Mt, ss=ss, a_off=Mf))
        self.tune(('in2', i % 4 == 3, T, Mf, r), Prob(z, t['Ain'], proj, Mt, cidx=pt))
        o = self.mixer(i, proj, cos, sin, T)
        ss = torch.zeros(T, device=dev, dtype=torch.float32); z = torch.empty(Mt, r, device=dev, dtype=bf)
        self.tune(('o', T, Mf, r), Prob(o, d['Wo'], xs, Mf, epi=3, ssout=ss, aidx=pf), Prob(o, t['Bo'], z, Mt, aidx=pt))
        self.tune(('o2', T, Mf, r), Prob(z, t['Ao'], xs, Mt, epi=3, ssout=ss, c_off=Mf))
        m = torch.empty(T, d['I'], device=dev, dtype=bf); z = torch.empty(Mt, r, device=dev, dtype=bf)
        self.tune(('gu', T, Mf, r), Prob(xs, d['Wgu_f'], m, Mf, epi=1, ss=ss), Prob(xs, t['Bgu'], z, Mt, ss=ss, a_off=Mf))
        self.tune(('gu2', T, Mf, r), Prob(z, t['Agu'], m, Mt, epi=1, c_off=Mf))
        ss = torch.zeros(T, device=dev, dtype=torch.float32); z = torch.empty(Mt, r, device=dev, dtype=bf)
        self.tune(('d', T, Mf, r), Prob(m, d['Wd'], xs, Mf, epi=3, ssout=ss), Prob(m, t['Bd'], z, Mt, a_off=Mf))
        self.tune(('d2', T, Mf, r), Prob(z, t['Ad'], xs, Mt, epi=3, ssout=ss, c_off=Mf))
        return xs, ss

    def forward(self, ids, perm=None, Mf=None, rows=None, srows=None):
        """ids [T]; perm: LongTensor [T] sorted order (full rows first), Mf full count; rows: sequence positions to return (normed)"""
        R = self.R; T = ids.shape[0]
        x = F.embedding(ids, R.embed)
        pos = torch.arange(T, device=R.dev, dtype=torch.float32)
        fr = pos[:, None] * R.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        ss = x.float().pow(2).sum(-1)
        n = len(R.layers)
        kk = n if perm is None else self.k
        for i in range(kk):
            x, ss = self.dense_layer(i, x, ss, cos, sin, T)
        if perm is not None:
            xs = x[perm].contiguous(); ss = ss[perm].contiguous()
            for i in range(self.k, n):
                xs, ss = self.mixed_layer(i, xs, ss, cos, sin, T, perm, Mf)
            return LM._rms_zc(xs[srows], R.norm_w, R.eps)      # srows: sorted positions of the readout rows
        if rows is None:
            return LM._rms_zc(x, R.norm_w, R.eps)
        return LM._rms_zc(x[rows], R.norm_w, R.eps)
