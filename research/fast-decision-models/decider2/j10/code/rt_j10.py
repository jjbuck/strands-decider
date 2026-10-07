"""J10 inference runtime for the ternary BitNet decider on the A10G: fused Triton glue + low-bit GEMMs + CUDA graph.

GEMM precisions (per GEMM class qkv | o | gu | down):
  'bf16' : cuBLAS bf16 with the ternary*gamma weights (same function, bf16 tensor cores; activations still fake-quantised to int8 levels)
  'i8'   : int8 tensor cores, weights stored as int8 ternary codes [K, N] (Triton, fused dequant epilogue -> bf16)
  'w2'   : int8 tensor cores, weights stored PACKED 2 bits/weight (4 per byte, unpacked in registers) -> 4x fewer weight bytes than 'i8'
  's4'   : int4 tensor cores (CUTLASS sm80 s4 x s4, mma k64), weights = ternary codes in s4, activations int4 (per-token absmax) -> fp16 alpha*acc,
           dequantised by the consuming glue kernel.  (4-bit activations change the function: see the A4 study.)
Glue (one Triton kernel each, per token row): [residual add +] RMSNorm + per-token absmax quantisation; RoPE + head split; attn_sub_norm + quant;
ReLU^2 * up + ffn_sub_norm + quant.  Attention = torch SDPA (flash, causal, GQA); multi-question = state causal + per-question block mask.
"""
import os, sys, math, ctypes, json, time, statistics
import torch, torch.nn.functional as F
import triton, triton.language as tl
from triton.language.extra import libdevice

# ------------------------------------------------------------------ GEMMs
_CFG = [triton.Config(dict(BM=bm, BN=bn, GROUP_M=8), num_warps=w, num_stages=s)
        for bm, bn, w, s in [(128, 128, 8, 3), (128, 256, 8, 3), (64, 128, 4, 4), (64, 256, 8, 3), (128, 64, 4, 4), (32, 128, 4, 4),
                             (64, 64, 4, 4), (32, 64, 4, 4), (16, 128, 4, 4), (16, 64, 4, 4)]]


@triton.autotune(configs=_CFG, key=['MB', 'N', 'K'])
@triton.jit
def _i8mm_k(A, B, SA, SW, C, M, N, K, MB, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN); npg = GROUP_M * npn
    gid = pid // npg; fm = gid * GROUP_M; gsm = min(npm - fm, GROUP_M)
    pm = fm + ((pid % npg) % gsm); pn = (pid % npg) // gsm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = A + rm[:, None].to(tl.int64) * K + rk[None, :]
    b_ptr = B + rk[:, None].to(tl.int64) * N + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr, mask=rm[:, None] < M, other=0)
        b = tl.load(b_ptr, mask=rn[None, :] < N, other=0)
        acc = tl.dot(a, b, acc)
        a_ptr += BK; b_ptr += BK * N
    sa = tl.load(SA + rm, mask=rm < M, other=0.)
    sw = tl.load(SW + rn, mask=rn < N, other=0.)
    out = acc.to(tl.float32) * sa[:, None] * sw[None, :]
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], out.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.autotune(configs=_CFG, key=['MB', 'N', 'K'])
@triton.jit
def _w2mm_k(A, B, SA, SW, C, M, N, K, MB, BM: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr, SB: tl.constexpr):
    """B packed [K/4, N] uint8: within each 4*SB block of K, byte row kk holds codes (+1) of k = j*SB + kk in bits 2j..2j+1."""
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN); npg = GROUP_M * npn
    gid = pid // npg; fm = gid * GROUP_M; gsm = min(npm - fm, GROUP_M)
    pm = fm + ((pid % npg) % gsm); pn = (pid % npg) // gsm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, SB)
    a_ptr = A + rm[:, None].to(tl.int64) * K + rk[None, :]
    b_ptr = B + rk[:, None].to(tl.int64) * N + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    am = rm[:, None] < M; bm = rn[None, :] < N
    for k0 in range(0, K, 4 * SB):
        p = tl.load(b_ptr, mask=bm, other=0x55).to(tl.int32)
        for j in tl.static_range(4):
            a = tl.load(a_ptr + j * SB, mask=am, other=0)
            w = (((p >> (2 * j)) & 3) - 1).to(tl.int8)
            acc = tl.dot(a, w, acc)
        a_ptr += 4 * SB; b_ptr += SB * N
    sa = tl.load(SA + rm, mask=rm < M, other=0.)
    sw = tl.load(SW + rn, mask=rn < N, other=0.)
    out = acc.to(tl.float32) * sa[:, None] * sw[None, :]
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], out.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


def mbucket(M):
    """autotune key: exact M is used for timing runs (EXACTM=1), else a coarse bucket so variable-length checks do not re-tune"""
    if os.environ.get('EXACTM') == '1': return M
    return min(4096, triton.next_power_of_2(max(16, M)))


def i8mm(a, bt, sa, sw, out):
    M, K = a.shape; N = bt.shape[1]
    _i8mm_k[lambda m: (triton.cdiv(M, m['BM']) * triton.cdiv(N, m['BN']),)](a, bt, sa, sw, out, M, N, K, mbucket(M), BK=128)
    return out


def w2mm(a, bp, sa, sw, out):
    M, K = a.shape; N = bp.shape[1]
    _w2mm_k[lambda m: (triton.cdiv(M, m['BM']) * triton.cdiv(N, m['BN']),)](a, bp, sa, sw, out, M, N, K, mbucket(M), SB=32)
    return out


def pack2(codes):
    """codes int8 [N, K] in {-1,0,1} -> packed uint8 [K/4, N] (layout of _w2mm_k, SB = 32)"""
    N, K = codes.shape; SB = 32
    c = (codes.t().to(torch.int32) + 1).reshape(K // (4 * SB), 4, SB, N)     # [kb, j, kk, N]
    p = c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)
    return p.reshape(K // 4, N).to(torch.uint8).contiguous()


_S4 = None


def s4lib():
    global _S4
    if _S4 is None:
        lib = ctypes.CDLL(os.path.expanduser('~/work/j10/libg2s4.so'))
        for fn in (lib.s4_gemm, lib.s8_gemm):
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        _S4 = lib
    return _S4


def c8mm(a8, b8, alpha, out, cfg=None):
    """CUTLASS sm80 s8 x s8 (G2's g2s4.cu s8_gemm): a8 [M, K] int8, b8 [N, K] int8 -> out fp16 [M, N] = alpha * acc"""
    M, K = a8.shape; N = b8.shape[0]
    if cfg is None: cfg = 3 if M <= 1500 else 1
    rc = s4lib().s8_gemm(a8.data_ptr(), b8.data_ptr(), out.data_ptr(), M, N, K, float(alpha), int(cfg), torch.cuda.current_stream().cuda_stream)
    if rc: raise RuntimeError(f's8 rc {rc} M{M} N{N} K{K} cfg{cfg}')
    return out


def s4mm(a4, b4, alpha, out, cfg):
    """a4 [M, K/2] uint8 (int4 low nibble first), b4 [N, K/2] -> out fp16 [M, N] = alpha * acc"""
    M = a4.shape[0]; K = a4.shape[1] * 2; N = b4.shape[0]
    rc = s4lib().s4_gemm(a4.data_ptr(), b4.data_ptr(), out.data_ptr(), M, N, K, float(alpha), int(cfg), torch.cuda.current_stream().cuda_stream)
    if rc: raise RuntimeError(f's4 rc {rc} M{M} N{N} K{K} cfg{cfg}')
    return out


def pack4(codes):
    """int8 [R, K] values in [-8, 7] -> uint8 [R, K/2] (low nibble = even column)"""
    c = codes.to(torch.int32) & 15
    return (c[:, 0::2] | (c[:, 1::2] << 4)).to(torch.uint8).contiguous()


# ------------------------------------------------------------------ glue
@triton.jit
def _quant_store(y, r, Q, SA, D: tl.constexpr, BLOCK: tl.constexpr, BITS: tl.constexpr):
    offs = tl.arange(0, BLOCK); m = offs < D
    amax = tl.max(tl.abs(y), 0)
    if BITS == 4:
        QM: tl.constexpr = 7.0
    else:
        QM: tl.constexpr = 127.0
    s = QM / tl.maximum(amax, 1e-5)
    q = tl.minimum(tl.maximum(libdevice.rint(y * s), -QM - 1), QM)
    if BITS == 16:
        tl.store(Q + r * D + offs, (q / s).to(tl.bfloat16), mask=m)
    elif BITS == 8:
        tl.store(Q + r * D + offs, q.to(tl.int8), mask=m)
        tl.store(SA + r, 1.0 / s)
    else:
        qi = q.to(tl.int32)
        lo, hi = tl.split(tl.reshape(qi, (BLOCK // 2, 2)))
        b = ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8)
        c = tl.arange(0, BLOCK // 2)
        tl.store(Q + r * (D // 2) + c, b, mask=c < D // 2)
        tl.store(SA + r, 1.0 / s)


@triton.jit
def _rmsn(x, W, eps, D: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK); m = offs < D
    ms = tl.sum(x * x, 0) / D
    xn = (x * tl.rsqrt(ms + eps)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=m, other=0.).to(tl.bfloat16).to(tl.float32)
    return (w * xn).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _addnormq_k(X, Y, RA, CS, W, Q, SA, eps, HAS_Y: tl.constexpr, DQ: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, BITS: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK); m = offs < D
    x = tl.load(X + r * D + offs, mask=m, other=0.).to(tl.float32)
    if HAS_Y:
        y = tl.load(Y + r * D + offs, mask=m, other=0.).to(tl.float32)
        if DQ:
            y = y * tl.load(RA + r) * tl.load(CS + offs, mask=m, other=0.)
        xb = (x + y.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(X + r * D + offs, xb, mask=m)
        x = xb.to(tl.float32)
    h = _rmsn(x, W, eps, D, BLOCK)
    _quant_store(h, r, Q, SA, D, BLOCK, BITS)


@triton.jit
def _relu2normq_k(GU, RA, CS, W, Q, SA, eps, DQ: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr, BITS: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK); m = offs < D
    g = tl.load(GU + r * 2 * D + offs, mask=m, other=0.).to(tl.float32)
    u = tl.load(GU + r * 2 * D + D + offs, mask=m, other=0.).to(tl.float32)
    if DQ:
        ra = tl.load(RA + r)
        g = (g * ra * tl.load(CS + offs, mask=m, other=0.)).to(tl.bfloat16).to(tl.float32)
        u = (u * ra * tl.load(CS + D + offs, mask=m, other=0.)).to(tl.bfloat16).to(tl.float32)
    a = tl.maximum(g, 0.)
    a = (a * a).to(tl.bfloat16).to(tl.float32)
    x = (a * u).to(tl.bfloat16).to(tl.float32)
    h = _rmsn(x, W, eps, D, BLOCK)
    _quant_store(h, r, Q, SA, D, BLOCK, BITS)


@triton.jit
def _attnnormq_k(O, W, Q, SA, eps, s_h, s_m, NH: tl.constexpr, NHP: tl.constexpr, HD: tl.constexpr, BITS: tl.constexpr):
    """O = SDPA output [1, NH, M, HD] with strides (s_h, s_m) -> row r of the [M, NH*HD] matrix: attn_sub_norm + quant"""
    r = tl.program_id(0).to(tl.int64)
    hh = tl.arange(0, NHP)[:, None]; cc = tl.arange(0, HD)[None, :]
    m2 = hh < NH
    x = tl.load(O + hh * s_h + r * s_m + cc, mask=m2, other=0.).to(tl.float32)
    D: tl.constexpr = NH * HD
    ms = tl.sum(tl.sum(x * x, 1), 0) / D
    xn = (x * tl.rsqrt(ms + eps)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + hh * HD + cc, mask=m2, other=0.).to(tl.bfloat16).to(tl.float32)
    h = (w * xn).to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(h), 1), 0)
    if BITS == 4:
        QM: tl.constexpr = 7.0
    else:
        QM: tl.constexpr = 127.0
    s = QM / tl.maximum(amax, 1e-5)
    q = tl.minimum(tl.maximum(libdevice.rint(h * s), -QM - 1), QM)
    if BITS == 16:
        tl.store(Q + r * D + hh * HD + cc, (q / s).to(tl.bfloat16), mask=m2)
    elif BITS == 8:
        tl.store(Q + r * D + hh * HD + cc, q.to(tl.int8), mask=m2)
        tl.store(SA + r, 1.0 / s)
    else:
        qi = q.to(tl.int32)
        lo, hi = tl.split(tl.reshape(qi, (NHP, HD // 2, 2)))
        b = ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8)
        c2 = tl.arange(0, HD // 2)[None, :]
        tl.store(Q + r * (D // 2) + hh * (HD // 2) + c2, b, mask=m2)
        tl.store(SA + r, 1.0 / s)


@triton.jit
def _rope_k(P, RA, CS, COS, SIN, Qo, Ko, Vo, Mrows, DQ: tl.constexpr, NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr):
    """P [Mrows, (NH+2NKV)*HD] (q | k | v) -> Qo [Mrows, NH, HD], Ko/Vo [Mrows, NKV, HD]; RoPE (rotate_half) on q, k"""
    r = tl.program_id(0).to(tl.int64); h = tl.program_id(1)
    W: tl.constexpr = (NH + 2 * NKV) * HD
    half: tl.constexpr = HD // 2
    c = tl.arange(0, half)
    base = P + r * W + h * HD
    x1 = tl.load(base + c).to(tl.float32); x2 = tl.load(base + half + c).to(tl.float32)
    if DQ:
        ra = tl.load(RA + r)
        x1 = (x1 * ra * tl.load(CS + h * HD + c)).to(tl.bfloat16).to(tl.float32)
        x2 = (x2 * ra * tl.load(CS + h * HD + half + c)).to(tl.bfloat16).to(tl.float32)
    if h < NH + NKV:
        c1 = tl.load(COS + r * HD + c).to(tl.float32); s1 = tl.load(SIN + r * HD + c).to(tl.float32)
        c2 = tl.load(COS + r * HD + half + c).to(tl.float32); s2 = tl.load(SIN + r * HD + half + c).to(tl.float32)
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
        if h < NH:
            o = Qo + r * NH * HD + h * HD
        else:
            o = Ko + r * NKV * HD + (h - NH) * HD
        tl.store(o + c, y1.to(tl.bfloat16)); tl.store(o + half + c, y2.to(tl.bfloat16))
    else:
        o = Vo + r * NKV * HD + (h - NH - NKV) * HD
        tl.store(o + c, x1.to(tl.bfloat16)); tl.store(o + half + c, x2.to(tl.bfloat16))


# ------------------------------------------------------------------ runtime
S4CFG = int(os.environ.get('S4CFG', '3'))


class RT:
    """prec: dict cls -> 'bf16' | 'i8' | 'w2' | 's4'  (or one string for all).  Built from a BitNetTorso (eval) + pointer head."""

    def __init__(self, torso, head, prec='i8', dev='cuda'):
        if isinstance(prec, str): prec = dict(qkv=prec, o=prec, gu=prec, down=prec)
        self.prec = prec; self.t = torso; self.head = head
        self.d, self.L, self.H, self.KV, self.hd, self.F, self.eps = torso.d, torso.L, torso.H, torso.KV, torso.hd, torso.ffn, torso.eps
        self.embed = torso.embed
        nr = torso.norms
        self.nw = {k: v.detach().float().contiguous() for k, v in nr.items()}
        self.W = []
        codes = torso.codes()
        groups = dict(qkv=('q', 'k', 'v'), o=('o',), gu=('gate', 'up'), down=('down',))
        for i in range(self.L):
            lw = {}
            for cls, parts in groups.items():
                c = torch.cat([codes[(i, p)][0] for p in parts], 0)                   # [N, K] int8
                sw = torch.cat([torch.full((codes[(i, p)][0].shape[0],), codes[(i, p)][1], device=dev) for p in parts]).float()
                p = prec[cls]
                if p == 'bf16':
                    lw[cls] = dict(w=(c.to(torch.bfloat16) * sw[:, None].to(torch.bfloat16)).t().contiguous())
                elif p == 'i8':
                    lw[cls] = dict(w=c.t().contiguous(), sw=sw)
                elif p == 'w2':
                    lw[cls] = dict(w=pack2(c), sw=sw)
                elif p == 's4':
                    alpha = 1.0 / 16
                    lw[cls] = dict(w=pack4(c), cs=(sw / alpha).contiguous(), alpha=alpha)
                elif p == 'c8':
                    alpha = 1.0 / 16
                    lw[cls] = dict(w=c.contiguous(), cs=(sw / alpha).contiguous(), alpha=alpha)
                lw[cls]['N'] = c.shape[0]
            self.W.append(lw)
        del codes
        self.abits = {cls: (4 if prec[cls] == 's4' else (16 if prec[cls] == 'bf16' else 8)) for cls in groups}

    def _gemm(self, cls, lw, a, sa, out):
        p = self.prec[cls]; w = lw[cls]
        if p == 'bf16': return torch.matmul(a, w['w'], out=out)
        if p == 'i8': return i8mm(a, w['w'], sa, w['sw'], out)
        if p == 'w2': return w2mm(a, w['w'], sa, w['sw'], out)
        if p == 'c8': return c8mm(a, w['w'], w['alpha'], out)
        return s4mm(a, w['w'], w['alpha'], out, S4CFG)

    def buffers(self, M):
        dev = self.embed.device; d = self.d
        B = {}
        def act(D, bits):
            if bits == 16: return torch.empty(M, D, device=dev, dtype=torch.bfloat16)
            if bits == 8: return torch.empty(M, D, device=dev, dtype=torch.int8)
            return torch.empty(M, D // 2, device=dev, dtype=torch.uint8)
        B['x'] = torch.empty(M, d, device=dev, dtype=torch.bfloat16)
        for cls, D in (('qkv', d), ('o', d), ('gu', d), ('down', self.F)):
            B['a_' + cls] = act(D, self.abits[cls]); B['s_' + cls] = torch.empty(M, device=dev, dtype=torch.float32)
        odt = lambda cls: torch.float16 if self.prec[cls] in ('s4', 'c8') else torch.bfloat16
        B['p'] = torch.empty(M, (self.H + 2 * self.KV) * self.hd, device=dev, dtype=odt('qkv'))
        B['q'] = torch.empty(1, M, self.H, self.hd, device=dev, dtype=torch.bfloat16)
        B['k'] = torch.empty(1, M, self.KV, self.hd, device=dev, dtype=torch.bfloat16)
        B['v'] = torch.empty(1, M, self.KV, self.hd, device=dev, dtype=torch.bfloat16)
        B['o_out'] = torch.empty(M, d, device=dev, dtype=odt('o'))
        B['gu'] = torch.empty(M, 2 * self.F, device=dev, dtype=odt('gu'))
        B['d_out'] = torch.empty(M, d, device=dev, dtype=odt('down'))
        return B

    def forward(self, ids, pos, B, attn):
        """ids [M] (state then questions), pos [M] rope positions, attn(q, k, v) -> o [1, H, M, hd]. Returns final-normed hidden [M, d] (bf16)."""
        M = ids.shape[0]; d = self.d; F_ = self.F
        cos, sin = self._cs(pos)
        x = B['x']; x.copy_(F.embedding(ids, self.embed))
        BL = lambda D: triton.next_power_of_2(D)
        prev = None
        for i in range(self.L):
            lw = self.W[i]; nw = self.nw
            # residual (+ previous down) + ln1 + quant
            if prev is None:
                _addnormq_k[(M,)](x, x, x, x, nw[f'ln1{i}'], B['a_qkv'], B['s_qkv'], self.eps, HAS_Y=False, DQ=False, D=d, BLOCK=BL(d), BITS=self.abits['qkv'], num_warps=8)
            else:
                pl, dq = prev
                _addnormq_k[(M,)](x, B['d_out'], B['s_down'], dq, nw[f'ln1{i}'], B['a_qkv'], B['s_qkv'], self.eps, HAS_Y=True, DQ=dq is not None,
                                  D=d, BLOCK=BL(d), BITS=self.abits['qkv'], num_warps=8)
            self._gemm('qkv', lw, B['a_qkv'], B['s_qkv'], B['p'])
            cs = lw['qkv'].get('cs')
            _rope_k[(M, self.H + 2 * self.KV)](B['p'], B['s_qkv'], cs if cs is not None else B['p'], cos, sin, B['q'], B['k'], B['v'], M,
                                               DQ=cs is not None, NH=self.H, NKV=self.KV, HD=self.hd, num_warps=2)
            o = attn(B['q'].transpose(1, 2), B['k'].transpose(1, 2), B['v'].transpose(1, 2))
            assert o.stride(3) == 1
            _attnnormq_k[(M,)](o, nw[f'sa{i}'], B['a_o'], B['s_o'], self.eps, o.stride(1), o.stride(2), NH=self.H, NHP=triton.next_power_of_2(self.H), HD=self.hd,
                               BITS=self.abits['o'], num_warps=8)
            self._gemm('o', lw, B['a_o'], B['s_o'], B['o_out'])
            cs = lw['o'].get('cs')
            _addnormq_k[(M,)](x, B['o_out'], B['s_o'], cs if cs is not None else x, nw[f'ln2{i}'], B['a_gu'], B['s_gu'], self.eps, HAS_Y=True,
                              DQ=cs is not None, D=d, BLOCK=BL(d), BITS=self.abits['gu'], num_warps=8)
            self._gemm('gu', lw, B['a_gu'], B['s_gu'], B['gu'])
            cs = lw['gu'].get('cs')
            _relu2normq_k[(M,)](B['gu'], B['s_gu'], cs if cs is not None else x, nw[f'sf{i}'], B['a_down'], B['s_down'], self.eps,
                                DQ=cs is not None, D=F_, BLOCK=BL(F_), BITS=self.abits['down'], num_warps=8)
            self._gemm('down', lw, B['a_down'], B['s_down'], B['d_out'])
            prev = (i, lw['down'].get('cs'))
        # final residual add (fp path) + final norm, in torch (tiny)
        pl, dq = prev
        y = B['d_out'].float()
        if dq is not None: y = y * B['s_down'][:, None] * dq[None, :]
        x = (x.float() + y.to(torch.bfloat16).float()).to(torch.bfloat16)
        return x

    def final(self, xrows):
        xf = xrows.float()
        xn = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16)
        return self.nw['final'].to(torch.bfloat16) * xn

    def _cs(self, pos):
        inv = 1.0 / (self.t.theta ** (torch.arange(0, self.hd, 2, device=pos.device, dtype=torch.float32) / self.hd))
        f = pos.float()[:, None] * inv[None, :]; e = torch.cat([f, f], -1)
        return e.cos().to(torch.bfloat16).contiguous(), e.sin().to(torch.bfloat16).contiguous()


def causal_attn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


class Packed:
    """state of T rows + n questions (each attends to the state and causally to itself), one pass."""

    def __init__(self, T, qlens, dev='cuda'):
        self.T = T; self.qlens = qlens; M = T + sum(qlens); self.M = M
        pos = list(range(T)); seg = [0] * T
        for j, L in enumerate(qlens):
            pos += list(range(T, T + L)); seg += [j + 1] * L
        self.pos = torch.tensor(pos, device=dev)
        if len(qlens) > 1:
            seg = torch.tensor(seg, device=dev)
            qs = seg[T:]; r = torch.arange(T, M, device=dev); c = torch.arange(M, device=dev)
            mask = (c[None, :] < T) | ((seg[None, :] == qs[:, None]) & (c[None, :] <= r[:, None]))
            self.mask = mask[None, None]
        else:
            self.mask = None

    def attn(self, q, k, v):
        if self.mask is None:
            return causal_attn(q, k, v)
        T = self.T
        o = torch.empty_like(q)
        o[:, :, :T] = F.scaled_dot_product_attention(q[:, :, :T], k[:, :, :T], v[:, :, :T], is_causal=True, enable_gqa=True)
        o[:, :, T:] = F.scaled_dot_product_attention(q[:, :, T:], k, v, attn_mask=self.mask, enable_gqa=True)
        return o
