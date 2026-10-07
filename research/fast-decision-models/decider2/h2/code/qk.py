"""H2 Triton glue kernels for the rotated low-bit runtime (H1 format v0).
Every kernel reads the previous GEMM's fp16 output C16 = alpha*acc and dequantizes on load: y = bf16(C16 * ra[row] * csa[col]), csa = s_w / alpha
(ra = the per-token activation scale of that GEMM's input). Then it does the bf16 model's op, the online rotation (if any), the per-token absmax and
the quantization of the NEXT GEMM's input (int8 codes, or int4 packed two per byte, low nibble first), so no extra pass touches memory.
Rotations use +-1 Sylvester / Paley matrices with tf32x3 tl.dot (fp32-accurate), scaled by 1/sqrt(n) afterwards.
DQ=False variants read bf16 inputs (the bf16 GEMM path / fold runtime).
"""
import math, torch, triton, triton.language as tl
from triton.language.extra import libdevice

# ------------------------------------------------------------------ rotation matrices (+-1)
def syl(n, dev):
    H = torch.ones(1, 1)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H.to(dev).contiguous()

def paley12_pm1():
    q = 11; sq = {(x * x) % q for x in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    H = torch.eye(12)
    H[0, 1:] += 1; H[1:, 0] -= 1
    for i in range(q):
        for j in range(q): H[1 + i, 1 + j] += chi(j - i)
    return H

_MATS = {}
def mats(dev='cuda'):
    if dev not in _MATS:
        P = torch.zeros(16, 16); P[:12, :12] = paley12_pm1()
        _MATS[dev] = dict(H16=syl(16, dev), H32=syl(32, dev), H64=syl(64, dev), H128=syl(128, dev), H256=syl(256, dev), P12=P.to(dev).contiguous(), P12T=P.t().to(dev).contiguous())
    return _MATS[dev]

# ------------------------------------------------------------------ device helpers
@triton.jit
def _ldm(Hp, n: tl.constexpr):
    i = tl.arange(0, n)
    return tl.load(Hp + i[:, None] * n + i[None, :])

@triton.jit
def _dot(a, b, PREC: tl.constexpr):
    if PREC == 0:
        return tl.dot(a, b, input_precision="tf32x3")
    elif PREC == 1:
        return tl.dot(a, b, input_precision="tf32")
    else:
        return tl.dot(a.to(tl.float16), b.to(tl.float16)).to(tl.float32)

@triton.jit
def _had2048(y, H32p, H64p, PREC: tl.constexpr):
    # y [2048] fp32 (one row) == X [32, 64]; returns vec(H32^T X H64) / sqrt(2048) = y @ (H32 (x) H64) / sqrt(2048)
    H64 = _ldm(H64p, 64); H32 = _ldm(H32p, 32)
    a = _dot(tl.reshape(y, (32, 64)), H64, PREC)
    a = _dot(H32, a, PREC)                     # Sylvester H is symmetric: H32^T = H32
    return tl.reshape(a, (2048,)) * 0.022097086912079608

@triton.jit
def _had128h(y, H128p, PREC: tl.constexpr):
    # per-head (16 x 128) Hadamard
    H = _ldm(H128p, 128)
    return tl.reshape(_dot(tl.reshape(y, (16, 128)), H, PREC), (2048,)) * 0.08838834764831843

@triton.jit
def _had256h(y, H16p, PREC: tl.constexpr):
    # per-head (8 x 256) Hadamard, H256 = H16 (x) H16: X_h [16,16] -> H16 X_h H16
    H16 = _ldm(H16p, 16)
    a = _dot(tl.reshape(y, (128, 16)), H16, PREC)
    a = tl.reshape(a, (8, 16, 16))
    Hb = tl.broadcast_to(H16[None, :, :], (8, 16, 16))
    a = _dot(Hb, a, PREC)
    return tl.reshape(a, (2048,)) * 0.0625

@triton.jit
def _had6144(y, P12Tp, H16p, H32p, PREC: tl.constexpr):
    # y [8192] (one row) == X [16 (12 valid), 16, 32]; returns y @ (P12 (x) H16 (x) H32) / sqrt(6144); padded rows 12..15 stay 0.
    # P12Tp holds P12^T zero-padded to 16x16 (left multiply).
    H32 = _ldm(H32p, 32); H16 = _ldm(H16p, 16); P12T = _ldm(P12Tp, 16)
    a = _dot(tl.reshape(y, (256, 32)), H32, PREC)                       # contract k
    a = tl.reshape(a, (16, 16, 32))
    a = _dot(tl.broadcast_to(H16[None, :, :], (16, 16, 16)), a, PREC)    # contract j (batched over i)
    a = _dot(P12T, tl.reshape(a, (16, 512)), PREC)                      # contract i
    return tl.reshape(a, (8192,)) * 0.012757759076995721

@triton.jit
def _qstore(y, rows, rmask, Q, SA, N: tl.constexpr, NP: tl.constexpr, R: tl.constexpr, QMAX: tl.constexpr, CLIP: tl.constexpr, BITS: tl.constexpr):
    # y [R, NP] fp32 (cols >= N are 0). Per-row symmetric absmax quantization exactly as torch: s = max(amax,1e-8)/qmax*clip, q = rint(y/s).clamp
    amax = tl.max(tl.abs(y), axis=1)
    s = tl.math.div_rn(tl.maximum(amax, 1e-8), QMAX) * CLIP
    q = libdevice.rint(tl.math.div_rn(y, s[:, None]))
    q = tl.minimum(tl.maximum(q, -QMAX), QMAX).to(tl.int32)
    tl.store(SA + rows, s, mask=rmask)
    if BITS == 8:
        c = tl.arange(0, NP)
        tl.store(Q + rows[:, None].to(tl.int64) * N + c[None, :], q.to(tl.int8), mask=rmask[:, None] & (c[None, :] < N))
    else:
        lo, hi = tl.split(tl.reshape(q, (R, NP // 2, 2)))
        b = ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8)
        c = tl.arange(0, NP // 2)
        tl.store(Q + rows[:, None].to(tl.int64) * (N // 2) + c[None, :], b, mask=rmask[:, None] & (c[None, :] < N // 2))

@triton.jit
def _qstore1(y, r, Q, SA, N: tl.constexpr, NP: tl.constexpr, QMAX: tl.constexpr, CLIP: tl.constexpr, BITS: tl.constexpr):
    # one row: y [NP] fp32
    amax = tl.max(tl.abs(y), axis=0)
    s = tl.math.div_rn(tl.maximum(amax, 1e-8), QMAX) * CLIP
    q = libdevice.rint(tl.math.div_rn(y, s))
    q = tl.minimum(tl.maximum(q, -QMAX), QMAX).to(tl.int32)
    tl.store(SA + r, s)
    if BITS == 8:
        c = tl.arange(0, NP)
        tl.store(Q + r * N + c, q.to(tl.int8), mask=c < N)
    else:
        lo, hi = tl.split(tl.reshape(q, (NP // 2, 2)))
        b = ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8)
        c = tl.arange(0, NP // 2)
        tl.store(Q + r * (N // 2) + c, b, mask=c < N // 2)

# ------------------------------------------------------------------ K5: residual add (+dequant) + gain-free RMSNorm + quantize (next Win / Wgu input)
@triton.jit
def _addq_k(X, D, RA, CS, Q, SA, HB, T, eps, HAS_D: tl.constexpr, DQ: tl.constexpr, OUTQ: tl.constexpr,
            QMAX: tl.constexpr, CLIP: tl.constexpr, BITS: tl.constexpr, R: tl.constexpr):
    c = tl.arange(0, 2048)
    if DQ:
        cs = tl.load(CS + c)
    for i in range(R):
        r = (tl.program_id(0) * R + i).to(tl.int64)
        if r < T:
            x = tl.load(X + r * 2048 + c).to(tl.float32)
            if HAS_D:
                d = tl.load(D + r * 2048 + c).to(tl.float32)
                if DQ:
                    d = (d * tl.load(RA + r)) * cs
                xb = (x + d.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
                tl.store(X + r * 2048 + c, xb)
                x = xb.to(tl.float32)
            rs = tl.rsqrt(tl.sum(x * x, 0) / 2048 + eps)
            xn = x * rs
            if OUTQ:
                _qstore1(xn, r, Q, SA, 2048, 2048, QMAX, CLIP, BITS)
            else:
                tl.store(HB + r * 2048 + c, xn.to(tl.bfloat16))

def addq(x, d, ra, cs, q, sa, hb=None, eps=1e-6, dq=True, outq=True, qmax=7., clip=.9, bits=4, R=1):
    T = x.shape[0]
    _addq_k[(triton.cdiv(T, R),)](x, d if d is not None else x, ra if ra is not None else x, cs if cs is not None else x,
                                  q if q is not None else x, sa if sa is not None else x, hb if hb is not None else x, T, eps,
                                  HAS_D=d is not None, DQ=dq, OUTQ=outq, QMAX=qmax, CLIP=clip, BITS=bits, R=R, num_warps=8)

# ------------------------------------------------------------------ K2: GDN short conv (+dequant, +prefix/branch tails) + SiLU + l2norm(q,k); b/a dequant
# PREV [T,3] int32: row index of logical positions t-3, t-2, t-1 in P (>=0), -1 -> zero, <=-2 -> row (-2-v) of TAIL (fp32 [nt, 6144], pre-conv values)
@triton.jit
def _conv_k(P, RA, CS, Wc, PREV, TAIL, OUT, AB, T, ps, DQ: tl.constexpr, BT: tl.constexpr, HD: tl.constexpr):
    tb = tl.program_id(0); h = tl.program_id(1)
    t = tb * BT + tl.arange(0, BT); tm = t < T
    c = tl.arange(0, HD)
    if h < 48:
        ch = h * HD + c
        if DQ:
            cs = tl.load(CS + ch)
        acc = tl.zeros([BT, HD], tl.float32)
        for j in tl.static_range(4):
            if j == 3:
                src = t
            else:
                src = tl.load(PREV + t * 3 + j, mask=tm, other=-1)
            ok = (src >= 0) & tm
            srcl = tl.where(ok, src, 0).to(tl.int64)
            x = tl.load(P + srcl[:, None] * ps + ch[None, :], mask=ok[:, None], other=0.).to(tl.float32)
            if DQ:
                ra = tl.load(RA + srcl, mask=ok, other=0.)
                x = ((x * ra[:, None]) * cs[None, :]).to(tl.bfloat16).to(tl.float32)
            if j < 3:
                tv = (src <= -2) & tm
                ti = tl.where(tv, -2 - src, 0).to(tl.int64)
                x += tl.load(TAIL + ti[:, None] * 6144 + ch[None, :], mask=tv[:, None], other=0.)
            w = tl.load(Wc + ch * 4 + j).to(tl.float32)
            acc += x * w[None, :]
        y = acc * tl.sigmoid(acc)
        y = y.to(tl.bfloat16).to(tl.float32)
        if h < 32:
            y = y / tl.sqrt(tl.sum(y * y, 1)[:, None] + 1e-6)
        grp = h // 16; hh = h % 16
        tl.store(OUT + grp * T * 2048 + t[:, None].to(tl.int64) * 2048 + hh * HD + c[None, :], y.to(tl.bfloat16), mask=tm[:, None])
    else:
        cm = c < 32
        x = tl.load(P + t[:, None].to(tl.int64) * ps + 8192 + c[None, :], mask=tm[:, None] & cm[None, :], other=0.).to(tl.float32)
        if DQ:
            ra = tl.load(RA + t, mask=tm, other=0.)
            cs = tl.load(CS + 8192 + c, mask=cm, other=0.)
            x = (x * ra[:, None]) * cs[None, :]
        tl.store(AB + t[:, None].to(tl.int64) * 32 + c[None, :], x.to(tl.bfloat16), mask=tm[:, None] & cm[None, :])

def conv(proj, ra, cs, w, prev, tail, dq=True):
    T = proj.shape[0]
    out = torch.empty(3, T, 16, 128, device=proj.device, dtype=torch.bfloat16)
    ab = torch.empty(T, 32, device=proj.device, dtype=torch.bfloat16)
    BT = 32
    _conv_k[(triton.cdiv(T, BT), 49)](proj, ra if ra is not None else proj, cs if cs is not None else proj, w, prev, tail, out, ab, T, proj.stride(0),
                                      DQ=dq, BT=BT, HD=128, num_warps=4)
    return out, ab

# ------------------------------------------------------------------ K3: GDN gated RMSNorm (o, z) + online R2 + quantize (Wo input); one row per program
# HAD: 1 = full 2048 (H32 (x) H64), 2 = per-head H128
@triton.jit
def _gnorm_k(O, Z, RA, CS, GW, SIGN, H32p, H64p, Hbp, Q, SA, YB, T, zs, eps, DQ: tl.constexpr, OUTQ: tl.constexpr, HAD: tl.constexpr,
             QMAX: tl.constexpr, CLIP: tl.constexpr, BITS: tl.constexpr, PREC: tl.constexpr = 3, ROWS: tl.constexpr = 4):
    c = tl.arange(0, 2048)
    gw = tl.load(GW + tl.arange(0, 128)).to(tl.float32)
    if DQ:
        cz = tl.load(CS + 6144 + c)
    if OUTQ:
        sg = tl.load(SIGN + c)
    for i in range(ROWS):
        r = (tl.program_id(0) * ROWS + i).to(tl.int64)
        if r < T:
            o3 = tl.reshape(tl.load(O + r * 2048 + c).to(tl.float32), (16, 128))
            rs = tl.rsqrt(tl.sum(o3 * o3, 1) / 128 + eps)
            on = (o3 * rs[:, None]).to(tl.bfloat16).to(tl.float32)
            y = tl.reshape((gw[None, :] * on).to(tl.bfloat16).to(tl.float32), (2048,))
            z = tl.load(Z + r * zs + c).to(tl.float32)
            if DQ:
                z = ((z * tl.load(RA + r)) * cz).to(tl.bfloat16).to(tl.float32)
            y = (y * (z * tl.sigmoid(z))).to(tl.bfloat16).to(tl.float32)
            if OUTQ:
                y = y * sg
                if HAD == 1:
                    if PREC == 3:
                        pre = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
                        y = _had2048(y / pre, H32p, H64p, 2) * pre
                    else:
                        y = _had2048(y, H32p, H64p, PREC)
                elif HAD == 2:
                    if PREC == 3:
                        pre = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
                        y = _had128h(y / pre, Hbp, 2) * pre
                    else:
                        y = _had128h(y, Hbp, PREC)
                _qstore1(y, r, Q, SA, 2048, 2048, QMAX, CLIP, BITS)
            else:
                tl.store(YB + r * 2048 + c, y.to(tl.bfloat16))

# ------------------------------------------------------------------ K3b: attention output * sigmoid(gate) + online R2 + quantize (Wo input)
# O: SDPA output [8, T, 256] with strides (osh, ost, 1); gate read from the attn proj (head h: cols h*512+256 .. +256)
@triton.jit
def _agate_k(O, P, RA, CS, SIGN, H32p, H64p, Hbp, Q, SA, YB, T, ps, osh, ost, DQ: tl.constexpr, OUTQ: tl.constexpr, HAD: tl.constexpr,
             QMAX: tl.constexpr, CLIP: tl.constexpr, BITS: tl.constexpr, PREC: tl.constexpr = 3, ROWS: tl.constexpr = 4):
    c = tl.arange(0, 2048)
    hh = c // 256; cc = c % 256
    gcol = hh * 512 + 256 + cc
    if DQ:
        cgt = tl.load(CS + gcol)
    if OUTQ:
        sg = tl.load(SIGN + c)
    for i in range(ROWS):
        r = (tl.program_id(0) * ROWS + i).to(tl.int64)
        if r < T:
            o = tl.load(O + hh.to(tl.int64) * osh + r * ost + cc).to(tl.float32)
            g = tl.load(P + r * ps + gcol).to(tl.float32)
            if DQ:
                g = ((g * tl.load(RA + r)) * cgt).to(tl.bfloat16).to(tl.float32)
            g = tl.sigmoid(g).to(tl.bfloat16).to(tl.float32)
            y = (o * g).to(tl.bfloat16).to(tl.float32)
            if OUTQ:
                y = y * sg
                if HAD == 1:
                    if PREC == 3:
                        pre = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
                        y = _had2048(y / pre, H32p, H64p, 2) * pre
                    else:
                        y = _had2048(y, H32p, H64p, PREC)
                else:
                    if PREC == 3:
                        pre = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30)
                        y = _had256h(y / pre, Hbp, 2) * pre
                    else:
                        y = _had256h(y, Hbp, PREC)
                _qstore1(y, r, Q, SA, 2048, 2048, QMAX, CLIP, BITS)
            else:
                tl.store(YB + r * 2048 + c, y.to(tl.bfloat16))

# ------------------------------------------------------------------ K4: SwiGLU (+dequant) + online R4 (6144) + quantize (Wd input); one row per program
@triton.jit
def _swiglu_k(GU, RA, CS, SIGN, P12Tp, H16p, H32p, Q, SA, MB, T, DQ: tl.constexpr, OUTQ: tl.constexpr,
              QMAX: tl.constexpr, CLIP: tl.constexpr, BITS: tl.constexpr, PREC: tl.constexpr = 3, ROWS: tl.constexpr = 4, MIN: tl.constexpr = False):
    # MIN: GU already holds m = SwiGLU output (bf16 [T, 6144], from the fused-epilogue GEMM); only R4 + quantize remain
    c = tl.arange(0, 8192); cm = c < 6144
    if DQ and not MIN:
        cg = tl.load(CS + c, mask=cm, other=0.); cu = tl.load(CS + 6144 + c, mask=cm, other=0.)
    if OUTQ:
        sg = tl.load(SIGN + c, mask=cm, other=0.)
    for i in range(ROWS):
        r = (tl.program_id(0) * ROWS + i).to(tl.int64)
        if r < T:
            if MIN:
                m = tl.load(GU + r * 6144 + c, mask=cm, other=0.).to(tl.float32)
            else:
                g = tl.load(GU + r * 12288 + c, mask=cm, other=0.).to(tl.float32)
                u = tl.load(GU + r * 12288 + 6144 + c, mask=cm, other=0.).to(tl.float32)
                if DQ:
                    ra = tl.load(RA + r)
                    g = ((g * ra) * cg).to(tl.bfloat16).to(tl.float32)
                    u = ((u * ra) * cu).to(tl.bfloat16).to(tl.float32)
                sl = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)
                m = (sl * u).to(tl.bfloat16).to(tl.float32)
            if OUTQ:
                m = m * sg
                if PREC == 3:
                    pre = tl.maximum(tl.max(tl.abs(m), axis=0), 1e-30)
                    y = _had6144(m / pre, P12Tp, H16p, H32p, 2) * pre
                else:
                    y = _had6144(m, P12Tp, H16p, H32p, PREC)
                _qstore1(y, r, Q, SA, 6144, 8192, QMAX, CLIP, BITS)
            else:
                tl.store(MB + r * 6144 + c, m.to(tl.bfloat16), mask=cm)

# ------------------------------------------------------------------ attention prep (+dequant): q/k RMSNorm(zero-centred gain) + RoPE; v copy; K/V into buffers at row offset
@triton.jit
def _aprep_k(P, RA, CS, QN, KN, COS, SIN, Qo, KB, VB, koff, ps, eps, DQ: tl.constexpr, D: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); s = tl.program_id(1)
    c = tl.arange(0, D)
    pc = tl.where(c < 32, c + 32, tl.where(c < 64, c - 32, c))
    if DQ:
        ra = tl.load(RA + r)
    if s < 10:
        if s < 8:
            col = s * 512
            w = tl.load(QN + c); wp = tl.load(QN + pc)
            out = Qo + r * 8 * D + s * D
        else:
            col = 4096 + (s - 8) * 256
            w = tl.load(KN + c); wp = tl.load(KN + pc)
            out = KB + (koff + r) * 2 * D + (s - 8) * D
        x = tl.load(P + r * ps + col + c).to(tl.float32)
        xp = tl.load(P + r * ps + col + pc).to(tl.float32)
        if DQ:
            x = ((x * ra) * tl.load(CS + col + c)).to(tl.bfloat16).to(tl.float32)
            xp = ((xp * ra) * tl.load(CS + col + pc)).to(tl.bfloat16).to(tl.float32)
        rs = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        xn = (x * rs * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
        xpn = (xp * rs * (1.0 + wp)).to(tl.bfloat16).to(tl.float32)
        cm = c < 64
        cs_ = tl.load(COS + r * 64 + c, mask=cm, other=1.0).to(tl.float32)
        sn = tl.load(SIN + r * 64 + c, mask=cm, other=0.0).to(tl.float32)
        sign = tl.where(c < 32, -1.0, 1.0)
        y = tl.where(cm, xn * cs_ + sign * xpn * sn, xn)
        tl.store(out + c, y.to(tl.bfloat16))
    else:
        col = 4608 + (s - 10) * 256
        x = tl.load(P + r * ps + col + c).to(tl.float32)
        if DQ:
            x = (x * ra) * tl.load(CS + col + c)
        tl.store(VB + (koff + r) * 2 * D + (s - 10) * D + c, x.to(tl.bfloat16))

def aprep(proj, ra, cs, qn, kn, cos, sin, kb, vb, koff, eps=1e-6, dq=True):
    T = proj.shape[0]
    q = torch.empty(T, 8, 256, device=proj.device, dtype=torch.bfloat16)
    _aprep_k[(T, 12)](proj, ra if ra is not None else proj, cs if cs is not None else proj, qn, kn, cos, sin, q, kb, vb, koff, proj.stride(0), eps,
                      DQ=dq, D=256, num_warps=4)
    return q
