"""d1 lean2: Lean 2B runtime with hand-written Triton fusions for the long (compute-bound) prefill path.
FUSE env (comma list, cumulative steps): addrms, gnorm, silu, prep, conv, gemm_swiglu, fold
"""
import os, sys, math, torch, torch.nn.functional as F
import triton, triton.language as tl
sys.path.insert(0, os.path.expanduser("~/work/systems/g"))
import lean as LM
from lean import Lean
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

# ---------------------------------------------------------------- kernels
@triton.jit
def _add_rms_k(X, D, W, H, eps, N: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, N)
    x = tl.load(X + r * N + offs).to(tl.float32)
    d = tl.load(D + r * N + offs).to(tl.float32)
    s = (x + d).to(tl.bfloat16)
    tl.store(X + r * N + offs, s)
    sf = s.to(tl.float32)
    rs = tl.rsqrt(tl.sum(sf * sf, 0) / N + eps)
    w = tl.load(W + offs)
    tl.store(H + r * N + offs, (sf * rs * w).to(tl.bfloat16))

def add_rms(x, d, w1, eps):  # in-place x += d ; returns rms(x)*(w1) ; w1 = 1+w fp32
    T, N = x.shape
    h = torch.empty_like(x)
    _add_rms_k[(T,)](x, d, w1, h, eps, N=N, num_warps=8)
    return x, h

@triton.jit
def _gnorm_k(O, Z, W, Y, zs, eps, NH: tl.constexpr, HD: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    hh = tl.arange(0, NH)[:, None]; cc = tl.arange(0, HD)[None, :]
    o = tl.load(O + r * NH * HD + hh * HD + cc).to(tl.float32)
    rs = tl.rsqrt(tl.sum(o * o, 1) / HD + eps)
    on = (o * rs[:, None]).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cc).to(tl.float32)
    y = (w * on).to(tl.bfloat16).to(tl.float32)
    z = tl.load(Z + r * zs + hh * HD + cc).to(tl.float32)
    y = y * z * tl.sigmoid(z)
    tl.store(Y + r * NH * HD + hh * HD + cc, y.to(tl.bfloat16))

def gnorm(o, z, w, eps):  # o [T,16,128] contiguous, z [T,2048] strided view
    T = z.shape[0]
    y = torch.empty(T, 2048, device=o.device, dtype=torch.bfloat16)
    _gnorm_k[(T,)](o, z, w, y, z.stride(0), eps, NH=16, HD=128, num_warps=8)
    return y

@triton.jit
def _silu_mul_k(GU, M, I: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); cb = tl.program_id(1)
    c = cb * BLOCK + tl.arange(0, BLOCK); cm = c < I
    g = tl.load(GU + r * 2 * I + c, mask=cm, other=0.).to(tl.float32)
    u = tl.load(GU + r * 2 * I + I + c, mask=cm, other=0.).to(tl.float32)
    s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
    tl.store(M + r * I + c, (s * u).to(tl.bfloat16), mask=cm)

def silu_mul(gu, I):
    T = gu.shape[0]
    m = torch.empty(T, I, device=gu.device, dtype=gu.dtype)
    _silu_mul_k[(T, triton.cdiv(I, 1024))](gu, m, I=I, BLOCK=1024, num_warps=4)
    return m

@triton.jit
def _attn_prep_k(P, QN, KN, COS, SIN, Q, K, G, ps, eps, D: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); s = tl.program_id(1)
    c = tl.arange(0, D)
    pc = tl.where(c < 32, c + 32, tl.where(c < 64, c - 32, c))
    if s < 8:
        base = P + r * ps + s * 512
        w = tl.load(QN + c); wp = tl.load(QN + pc)
        gt = tl.load(base + 256 + c).to(tl.float32)
        tl.store(G + r * 2048 + s * 256 + c, tl.sigmoid(gt).to(tl.bfloat16))
        out = Q + r * 8 * D + s * D
    else:
        base = P + r * ps + 4096 + (s - 8) * 256
        w = tl.load(KN + c); wp = tl.load(KN + pc)
        out = K + r * 2 * D + (s - 8) * D
    x = tl.load(base + c).to(tl.float32)
    xp = tl.load(base + pc).to(tl.float32)
    rs = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    xn = (x * rs * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
    xpn = (xp * rs * (1.0 + wp)).to(tl.bfloat16).to(tl.float32)
    cm = c < 64
    cs = tl.load(COS + r * 64 + c, mask=cm, other=1.0).to(tl.float32)
    sn = tl.load(SIN + r * 64 + c, mask=cm, other=0.0).to(tl.float32)
    sign = tl.where(c < 32, -1.0, 1.0)
    y = tl.where(cm, xn * cs + sign * xpn * sn, xn)
    tl.store(out + c, y.to(tl.bfloat16))

def attn_prep(proj, qn, kn, cos, sin, eps):
    T = proj.shape[0]
    q = torch.empty(T, 8, 256, device=proj.device, dtype=torch.bfloat16)
    k = torch.empty(T, 2, 256, device=proj.device, dtype=torch.bfloat16)
    g = torch.empty(T, 2048, device=proj.device, dtype=torch.bfloat16)
    _attn_prep_k[(T, 10)](proj, qn, kn, cos, sin, q, k, g, proj.stride(0), eps, D=256, num_warps=4)
    return q, k, g

@triton.jit
def _conv_l2_k(P, W, OUT, T, ps, BT: tl.constexpr, HD: tl.constexpr):
    tb = tl.program_id(0); h = tl.program_id(1)
    t = (tb * BT + tl.arange(0, BT)[:, None]).to(tl.int64); c = tl.arange(0, HD)[None, :]
    ch = h * HD + c
    acc = tl.zeros([BT, HD], tl.float32)
    for j in tl.static_range(4):
        tt = t - 3 + j
        x = tl.load(P + tt * ps + ch, mask=(tt >= 0) & (tt < T), other=0.).to(tl.float32)
        w = tl.load(W + ch * 4 + j).to(tl.float32)
        acc += x * w
    y = acc * tl.sigmoid(acc)
    y = y.to(tl.bfloat16).to(tl.float32)
    if h < 32:
        y = y / tl.sqrt(tl.sum(y * y, 1)[:, None] + 1e-6)
    grp = h // 16; hh = h % 16
    tl.store(OUT + grp * T * 2048 + t * 2048 + hh * HD + c, y.to(tl.bfloat16), mask=t < T)

def conv_l2(proj, w):  # proj [T, 8224] -> qkv3 [3, T, 16, 128] (q,k l2-normalized)
    T = proj.shape[0]
    out = torch.empty(3, T, 16, 128, device=proj.device, dtype=torch.bfloat16)
    BT = 32
    _conv_l2_k[(triton.cdiv(T, BT), 48)](proj, w, out, T, proj.stride(0), BT=BT, HD=128, num_warps=4)
    return out

# ---------------------------------------------------------------- Triton GEMM with fused epilogues
# C[M,N] = A[M,K] @ B[N,K]^T ; EPI: 0 plain, 1 swiglu (B rows interleaved g0,u0,g1,u1..., C has N/2 cols),
# 2 row-scale (C *= rs[m], rs = rsqrt(ss[m]/Kd+eps)) for folded RMSNorm, 3 residual add + row sumsq atomics
@triton.jit
def _gemm_k(A, B, C, R, SS, SSOUT, M, N, K, eps, Kd, EPI: tl.constexpr, PRO_RS: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    gsz = GROUP * npn
    g = pid // gsz; fm = g * GROUP; gm = min(npm - fm, GROUP)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = A + rm[:, None].to(tl.int64) * K + rk[None, :]
    b_ptr = B + rn[None, :].to(tl.int64) * K + rk[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr, mask=rm[:, None] < M, other=0.)
        b = tl.load(b_ptr, mask=rn[None, :] < N, other=0.)
        acc = tl.dot(a, b, acc)
        a_ptr += BK; b_ptr += BK
    if PRO_RS:
        ss = tl.load(SS + rm, mask=rm < M, other=1.0)
        acc = acc * tl.rsqrt(ss / Kd + eps)[:, None]
    if EPI == 1:
        gg, uu = tl.split(tl.reshape(acc, [BM, BN // 2, 2]))
        s = (gg * tl.sigmoid(gg)).to(tl.bfloat16).to(tl.float32)
        out = (s * uu.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        cn = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(C + rm[:, None].to(tl.int64) * (N // 2) + cn[None, :], out, mask=rm[:, None] < M)
    elif EPI == 3:
        cp = rm[:, None].to(tl.int64) * N + rn[None, :]
        r = tl.load(R + cp, mask=rm[:, None] < M, other=0.).to(tl.float32)
        s = (r + acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(R + cp, s, mask=rm[:, None] < M)
        sf = s.to(tl.float32)
        tl.atomic_add(SSOUT + rm, tl.sum(sf * sf, 1), mask=rm < M, sem="relaxed")
    else:
        tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))

def pick_cfg(M, N, K):   # from gemm_tune.py on A10G
    if M <= 384: return (128, 64, 64, 4, 4)
    if N == 2048 and K == 6144 and M <= 2048: return (256, 128, 32, 8, 3)
    return (128, 128, 32, 4, 4)

def tgemm(a, b, epi=0, res=None, ss=None, ssout=None, eps=1e-6, Kd=2048, cfg=None):
    M, K = a.shape; N = b.shape[0]
    BM, BN, BK, nw, ns = cfg or pick_cfg(M, N, K)
    if epi == 1: c = torch.empty(M, N // 2, device=a.device, dtype=torch.bfloat16)
    elif epi == 3: c = res
    else: c = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    dummy = c
    _gemm_k[grid](a, b, c, res if res is not None else dummy, ss if ss is not None else dummy, ssout if ssout is not None else dummy,
                  M, N, K, eps, float(Kd), EPI=epi, PRO_RS=ss is not None, BM=BM, BN=BN, BK=BK, GROUP=8, num_warps=nw, num_stages=ns)
    return c

# ---------------------------------------------------------------- runtime
class Lean2(Lean):
    def __init__(self, torso, fuse=None):
        super().__init__(torso)
        self.fuse = set((fuse if fuse is not None else os.environ.get("FUSE", "")).split(",")) - {""}
        for i, d in enumerate(self.layers):
            d["in1"] = (1.0 + d["in_norm"]).contiguous(); d["post1"] = (1.0 + d["post_norm"]).contiguous()
            if True:
                I = d["I"]; W = d["Wgu"]
                d["Wgu_il"] = torch.stack([W[:I], W[I:]], 1).reshape(2 * I, -1).contiguous()   # g0,u0,g1,u1,...
            if True:   # fold (1+w) of the pre-norm into the following GEMM weights (row scale applied in epilogue)
                d["Win_f"] = (d["Win"].float() * d["in1"][None, :]).to(torch.bfloat16).contiguous()
                Wg = d["Wgu_il"]
                d["Wgu_f"] = (Wg.float() * d["post1"][None, :]).to(torch.bfloat16).contiguous()
        self.norm1 = (1.0 + self.norm_w).contiguous()
        self.gcfg = {}

    def _mm(self, a, W):
        return a @ W.t()

    @torch.no_grad()
    def forward(self, ids):
        f = self.fuse
        B, T = ids.shape
        assert B == 1
        x = F.embedding(ids, self.embed).reshape(T, -1)
        pos = torch.arange(T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        fold = "fold" in f
        if fold:
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            ss += x.float().pow(2).sum(-1)
        else:
            h = LM._rms_zc(x, self.layers[0]["in_norm"], self.eps)
        n = len(self.layers)
        for i, d in enumerate(self.layers):
            if fold:
                proj = tgemm(x, d["Win_f"], epi=0, ss=ss, Kd=x.shape[1])
            else:
                proj = self._mm(h, d["Win"])
            if d["type"] == "linear_attention":
                if "conv" in f:
                    qkv3 = conv_l2(proj, d["conv_w"])
                    q, k, v = qkv3[0][None], qkv3[1][None], qkv3[2][None]
                    a = proj[:, 8208:8224].reshape(1, T, 16); b = proj[:, 8192:8208].reshape(1, T, 16)
                    o, _ = chunk_gated_delta_rule(q, k, v, a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True,
                                                  A_log=d["A_log"], dt_bias=d["dt_bias"], use_beta_sigmoid_in_kernel=True)
                    z = proj[:, 6144:8192]
                else:
                    z, beta, g = LM._lin_glue(proj, d["A_log"], d["dt_bias"])
                    qkv = proj[:, :6144].reshape(B, T, 6144)
                    qkv = LM.fla_conv(qkv, d["conv_w"], None, activation="silu")
                    qkv = qkv[0] if isinstance(qkv, tuple) else qkv
                    q, k, v = qkv.split(2048, dim=-1)
                    q = q.reshape(B, T, 16, 128); k = k.reshape(B, T, 16, 128); v = v.reshape(B, T, 16, 128)
                    o, _ = chunk_gated_delta_rule(q, k, v, g.reshape(B, T, 16), beta.reshape(B, T, 16), use_qk_l2norm_in_kernel=True)
                if "gnorm" in f:
                    o = gnorm(o.reshape(T, 16, 128), z, d["gn_w"], self.eps)
                else:
                    o = LM._gated_norm(o.reshape(-1, 128), z.reshape(-1, 128), d["gn_w"], self.eps).reshape(T, 2048)
            else:
                if "prep" in f:
                    q, k, gate = attn_prep(proj, d["qn"], d["kn"], cos, sin, self.eps)
                    v = proj[:, 4608:5120].reshape(T, 2, 256)
                else:
                    q, k, v, gate = LM._attn_prep(proj, d["qn"], d["kn"], cos, sin, self.eps)
                qh = q.reshape(1, T, 8, 256).transpose(1, 2); kh = k.reshape(1, T, 2, 256).transpose(1, 2); vh = v.reshape(1, T, 2, 256).transpose(1, 2)
                o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
            if fold:
                ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
                tgemm(o, d["Wo"], epi=3, res=x, ssout=ss)
                m = tgemm(x, d["Wgu_f"], epi=1, ss=ss, Kd=x.shape[1])
                ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
                tgemm(m, d["Wd"], epi=3, res=x, ssout=ss)
                continue
            d_out = self._mm(o, d["Wo"])
            if "addrms" in f:
                x, h2 = add_rms(x, d_out, d["post1"], self.eps)
            else:
                x, h2 = LM._add_rms(x, d_out, d["post_norm"], self.eps)
            if "gemm_swiglu" in f:
                m = tgemm(h2, d["Wgu_il"], epi=1, cfg=self.gcfg.get("swiglu"))
            else:
                gu = self._mm(h2, d["Wgu"])
                m = silu_mul(gu, d["I"]) if "silu" in f else LM._silu_mul(gu, d["I"])
            d_mlp = self._mm(m, d["Wd"])
            nw = self.layers[i + 1]["in_norm"] if i + 1 < n else self.norm_w
            if "addrms" in f:
                x, h = add_rms(x, d_mlp, (self.layers[i + 1]["in1"] if i + 1 < n else self.norm1), self.eps)
            else:
                x, h = LM._add_rms(x, d_mlp, nw, self.eps)
        if fold:
            h = LM._rms_zc(x, self.norm_w, self.eps)
        return h.reshape(1, T, -1)
