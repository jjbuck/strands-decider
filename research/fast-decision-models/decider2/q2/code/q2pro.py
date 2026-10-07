"""Q2 Triton quantize prologues (FORMATS.md sections 0, 3-9) and the B3 nearest-prototype search.

_rowq_k: one row per program, input y [M, N] (bf16, standing in for the fp32 row a glue kernel holds in registers), modes:
  0 RTN int4 (H2 baseline)          1 SR int4 (section 4 hash)        2 RTN int4 + dX bf16 (B1)        3 group-scaled int4 (g = G)
  4 B5 / ResQ split: int8 cols [0, K8), int4 cols [K8, K8 + K4), rest skipped (0-bit)
  5 row-role: rows < q0 -> int4 (Q), rows >= q0 -> int8 (Q8, row r - q0)
  6 RTN int4 + B6 statistic: STAT[r % 64] += w_r * ||dX_r||^2 (fp32 atomics)
  7 RTN int4 + bf16 copy of the row (input of the B3 search)
  8 arbitrary row partition (B8): POS[r] >= 0 -> int8 row POS[r], else int4 row -POS[r]-1
_addq_x: H2's residual add + gain-free RMSNorm + quantize kernel with the same modes (K = 2048), for in-runtime cost.
_proto_k: B3 search (int8 scores, argmin) + residual int4 quantization, BM rows per program."""
import torch, triton, triton.language as tl
from triton.language.extra import libdevice

M32 = 0xFFFFFFFF


@triton.jit
def _hash_u(seed, r, c):
    # FORMATS section 4, in int64 with explicit 32-bit masks (bit-identical to q2k.hash_u)
    h = (seed ^ ((r * 0x9E3779B1) & 0xFFFFFFFF) ^ ((c * 0x85EBCA77) & 0xFFFFFFFF)) & 0xFFFFFFFF
    h = h ^ (h >> 16); h = (h * 0x85EBCA6B) & 0xFFFFFFFF; h = h ^ (h >> 13); h = (h * 0xC2B2AE35) & 0xFFFFFFFF; h = h ^ (h >> 16)
    return (h >> 8).to(tl.float32) * 5.9604644775390625e-08


@triton.jit
def _pack_store(q, ptr, NP: tl.constexpr, mask_pairs):
    lo, hi = tl.split(tl.reshape(q, (NP // 2, 2)))
    b = ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8)
    c = tl.arange(0, NP // 2)
    tl.store(ptr + c, b, mask=mask_pairs)


@triton.jit
def _quant(y, QMAX: tl.constexpr, CLIP: tl.constexpr):
    amax = tl.max(tl.abs(y), axis=0)
    s = tl.math.div_rn(tl.maximum(amax, 1e-8), QMAX) * CLIP
    return s


@triton.jit
def _qstore_x(y, r, Q, SA, Q8, S8, DX, XB, GS, STAT, W, POS, seed, q0, K8, K4,
              N: tl.constexpr, NP: tl.constexpr, MODE: tl.constexpr, G: tl.constexpr, CLIP4: tl.constexpr):
    c = tl.arange(0, NP); cm = c < N
    if MODE == 3:
        yg = tl.reshape(y, (NP // G, G))
        amax = tl.max(tl.abs(yg), axis=1)
        s = tl.math.div_rn(tl.maximum(amax, 1e-8), 7.0) * CLIP4
        q = libdevice.rint(tl.math.div_rn(yg, s[:, None]))
        q = tl.minimum(tl.maximum(q, -7.0), 7.0).to(tl.int32)
        gi = tl.arange(0, NP // G)
        tl.store(GS + r * (N // G) + gi, s, mask=gi < N // G)
        _pack_store(tl.reshape(q, (NP,)), Q + r * (N // 2), NP, tl.arange(0, NP // 2) < N // 2)
    elif MODE == 4:
        m8 = c < K8; m4 = (c >= K8) & (c < K8 + K4)
        s8 = tl.math.div_rn(tl.maximum(tl.max(tl.where(m8, tl.abs(y), 0.0), axis=0), 1e-8), 127.0)
        s4 = tl.math.div_rn(tl.maximum(tl.max(tl.where(m4, tl.abs(y), 0.0), axis=0), 1e-8), 7.0) * CLIP4
        q8 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(y, s8)), -127.0), 127.0).to(tl.int32)
        q4 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(y, s4)), -7.0), 7.0).to(tl.int32)
        tl.store(Q8 + r * K8 + c, q8.to(tl.int8), mask=m8)
        tl.store(S8 + r, s8); tl.store(SA + r, s4)
        j = tl.arange(0, NP // 2)
        lo, hi = tl.split(tl.reshape(q4, (NP // 2, 2)))
        b = ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8)
        tl.store(Q + r * (K4 // 2) + (j - K8 // 2), b, mask=(j >= K8 // 2) & (j < (K8 + K4) // 2))
    elif MODE == 5 or MODE == 8:
        if MODE == 5:
            is8 = r >= q0
            row = tl.where(is8, r - q0, r)
        else:
            pos = tl.load(POS + r)
            is8 = pos >= 0
            row = tl.where(is8, pos, -pos - 1)
        if is8:
            s = tl.math.div_rn(tl.maximum(tl.max(tl.abs(y), axis=0), 1e-8), 127.0)
            q = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(y, s)), -127.0), 127.0).to(tl.int32)
            tl.store(S8 + row, s)
            tl.store(Q8 + row * N + c, q.to(tl.int8), mask=cm)
        else:
            s = tl.math.div_rn(tl.maximum(tl.max(tl.abs(y), axis=0), 1e-8), 7.0) * CLIP4
            q = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(y, s)), -7.0), 7.0).to(tl.int32)
            tl.store(SA + row, s)
            _pack_store(q, Q + row * (N // 2), NP, tl.arange(0, NP // 2) < N // 2)
    else:
        s = tl.math.div_rn(tl.maximum(tl.max(tl.abs(y), axis=0), 1e-8), 7.0) * CLIP4
        v = tl.math.div_rn(y, s)
        if MODE == 1:
            u = _hash_u(seed, r.to(tl.int64), c.to(tl.int64))
            q = tl.floor(v + u)
        else:
            q = libdevice.rint(v)
        q = tl.minimum(tl.maximum(q, -7.0), 7.0)
        tl.store(SA + r, s)
        if MODE == 2 or MODE == 6:
            e = y - s * q
            if MODE == 2:
                tl.store(DX + r * N + c, e.to(tl.bfloat16), mask=cm)
            else:
                w = tl.load(W + r)
                tl.atomic_add(STAT + (r % 64), w * tl.sum(e * e, axis=0))
        if MODE == 7:
            tl.store(XB + r * N + c, y.to(tl.bfloat16), mask=cm)
        _pack_store(q.to(tl.int32), Q + r * (N // 2), NP, tl.arange(0, NP // 2) < N // 2)


@triton.jit
def _rowq_k(Y, Q, SA, Q8, S8, DX, XB, GS, STAT, W, POS, seed, q0, K8, K4, M,
            N: tl.constexpr, NP: tl.constexpr, MODE: tl.constexpr, G: tl.constexpr, CLIP4: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, NP)
    y = tl.load(Y + r * N + c, mask=c < N, other=0.).to(tl.float32)
    _qstore_x(y, r, Q, SA, Q8, S8, DX, XB, GS, STAT, W, POS, seed, q0, K8, K4, N, NP, MODE, G, CLIP4)


@triton.jit
def _addq_x(X, D, RA, CS, Q, SA, Q8, S8, DX, XB, GS, STAT, W, POS, seed, q0, K8, K4, T, eps,
            MODE: tl.constexpr, G: tl.constexpr, CLIP4: tl.constexpr):
    # H2 _addq_k (residual add of a dequantized GEMM output + gain-free RMSNorm) followed by the mode's quantizer
    c = tl.arange(0, 2048)
    cs = tl.load(CS + c)
    r = tl.program_id(0).to(tl.int64)
    x = tl.load(X + r * 2048 + c).to(tl.float32)
    d = tl.load(D + r * 2048 + c).to(tl.float32)
    d = (d * tl.load(RA + r)) * cs
    xb = (x + d.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(X + r * 2048 + c, xb)
    x = xb.to(tl.float32)
    rs = tl.rsqrt(tl.sum(x * x, 0) / 2048 + eps)
    _qstore_x(x * rs, r, Q, SA, Q8, S8, DX, XB, GS, STAT, W, POS, seed, q0, K8, K4, 2048, 2048, MODE, G, CLIP4)


class Bufs:
    """output buffers for one GEMM input of M rows x N cols"""
    def __init__(self, M, N, dev='cuda', K8=128, K4=None, G=64, q0=None):
        self.M, self.N = M, N
        self.K8 = K8; self.K4 = (N - K8) if K4 is None else K4; self.G = G; self.q0 = M if q0 is None else q0
        self.Q = torch.empty(M, N // 2, device=dev, dtype=torch.uint8); self.SA = torch.empty(M, device=dev)
        self.Q8 = torch.empty(M, N, device=dev, dtype=torch.int8); self.S8 = torch.empty(M, device=dev)
        self.DX = torch.empty(M, N, device=dev, dtype=torch.bfloat16); self.XB = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        self.GS = torch.empty(M, N // G, device=dev); self.STAT = torch.zeros(64, device=dev); self.W = torch.ones(M, device=dev)
        self.POS = torch.arange(M, device=dev, dtype=torch.int32)


def rowq(y, b, mode, seed=0, clip4=0.9):
    M, N = y.shape; NP = triton.next_power_of_2(N)
    _rowq_k[(M,)](y, b.Q, b.SA, b.Q8, b.S8, b.DX, b.XB, b.GS, b.STAT, b.W, b.POS, seed, b.q0, b.K8, b.K4, M,
                  N=N, NP=NP, MODE=mode, G=b.G, CLIP4=clip4, num_warps=8, enable_fp_fusion=False)


def addq(x, d, ra, cs, b, mode, seed=0, eps=1e-6, clip4=0.9):
    T = x.shape[0]
    _addq_x[(T,)](x, d, ra, cs, b.Q, b.SA, b.Q8, b.S8, b.DX, b.XB, b.GS, b.STAT, b.W, b.POS, seed, b.q0, b.K8, b.K4, T, eps,
                  MODE=mode, G=b.G, CLIP4=clip4, num_warps=8, enable_fp_fusion=False)


# ------------------------------------------------------------------ B3: nearest prototype (int8 scores) + residual int4
@triton.jit
def _proto_k(XB, CQ, SC, HN, CB, Q, SA, IDX, M, K: tl.constexpr, KC: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,
             CLIP4: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM); rm = rows < M
    rows64 = rows.to(tl.int64)
    kk = tl.arange(0, BK)
    # pass 0: per-row amax -> int8 search scale
    amax = tl.zeros([BM], tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(XB + rows64[:, None] * K + k0 + kk[None, :], mask=rm[:, None], other=0.).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x), axis=1))
    s8 = tl.math.div_rn(tl.maximum(amax, 1e-8), 127.0)
    # pass 1: scores over prototype chunks, running argmin
    best = tl.full([BM], float('inf'), tl.float32); bidx = tl.zeros([BM], tl.int32)
    cc = tl.arange(0, BC)
    for c0 in range(0, KC, BC):
        acc = tl.zeros([BM, BC], tl.int32)
        for k0 in range(0, K, BK):
            x = tl.load(XB + rows64[:, None] * K + k0 + kk[None, :], mask=rm[:, None], other=0.).to(tl.float32)
            q8 = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(x, s8[:, None])), -127.0), 127.0).to(tl.int8)
            cq = tl.load(CQ + (c0 + cc)[None, :].to(tl.int64) * K + k0 + kk[:, None])       # [BK, BC]
            acc += tl.dot(q8, cq, out_dtype=tl.int32)
        sc = tl.load(SC + c0 + cc); hn = tl.load(HN + c0 + cc)
        score = hn[None, :] - (s8[:, None] * sc[None, :]) * acc.to(tl.float32)
        mn = tl.min(score, axis=1)
        am = tl.argmin(score, axis=1).to(tl.int32) + c0
        upd = mn < best
        best = tl.where(upd, mn, best); bidx = tl.where(upd, am, bidx)
    tl.store(IDX + rows, bidx, mask=rm)
    # pass 2: residual amax; pass 3: quantize and store
    b64 = bidx.to(tl.int64)
    ra = tl.zeros([BM], tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(XB + rows64[:, None] * K + k0 + kk[None, :], mask=rm[:, None], other=0.).to(tl.float32)
        cv = tl.load(CB + b64[:, None] * K + k0 + kk[None, :]).to(tl.float32)
        ra = tl.maximum(ra, tl.max(tl.abs(x - cv), axis=1))
    s4 = tl.math.div_rn(tl.maximum(ra, 1e-8), 7.0) * CLIP4
    tl.store(SA + rows, s4, mask=rm)
    jj = tl.arange(0, BK // 2)
    for k0 in range(0, K, BK):
        x = tl.load(XB + rows64[:, None] * K + k0 + kk[None, :], mask=rm[:, None], other=0.).to(tl.float32)
        cv = tl.load(CB + b64[:, None] * K + k0 + kk[None, :]).to(tl.float32)
        q = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(x - cv, s4[:, None])), -7.0), 7.0).to(tl.int32)
        lo, hi = tl.split(tl.reshape(q, (BM, BK // 2, 2)))
        bq = ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8)
        tl.store(Q + rows64[:, None] * (K // 2) + k0 // 2 + jj[None, :], bq, mask=rm[:, None])


def proto(xb, cq, sc, hn, cb, Q, SA, IDX, BM=32, BK=128, BC=64, clip4=0.9):
    M, K = xb.shape; KC = cq.shape[0]
    _proto_k[(triton.cdiv(M, BM),)](xb, cq, sc, hn, cb, Q, SA, IDX, M, K=K, KC=KC, BM=BM, BK=BK, BC=min(BC, KC), CLIP4=clip4, num_warps=4, enable_fp_fusion=False)


def proto_ref(xb, cq, sc, hn, cb, clip4=0.9):
    """FORMATS section 5 in torch"""
    x = xb.float()
    a8 = x.abs().amax(1).clamp_min(1e-8); s8 = a8 / torch.full_like(a8, 127.0)
    q8 = torch.round(x / s8[:, None]).clamp(-127, 127)
    dot = (q8.double() @ cq.double().t()).round().float()
    score = hn[None, :] - (s8[:, None] * sc[None, :]) * dot
    idx = score.argmin(1)        # torch argmin returns the first minimal index
    r = x - cb.float()[idx]
    a4 = r.abs().amax(1).clamp_min(1e-8); s4 = (a4 / torch.full_like(a4, 7.0)) * clip4
    q4 = torch.round(r / s4[:, None]).clamp(-7, 7).to(torch.int8)
    return idx.to(torch.int32), q4, s4


# ------------------------------------------------------------------ B1 fused prologue: int4 RTN codes + Z = dX @ A in one pass (no dX round trip)
@triton.jit
def _qz_k(Y, A, Q, SA, Z, M, K: tl.constexpr, R: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, CLIP4: tl.constexpr, STAT, W, HAS_STAT: tl.constexpr):
    """rows [pid*BM, +BM): pass 1 per-row amax; pass 2 codes (packed int4), dX = y - s q in bf16, Z += dX @ A[k-chunk] (fp32 tensor-core sum).
    Z stored bf16 [M, R]. Optional B6 subspace statistic: STAT[pid % 64] += sum_t w_t ||Z_t||^2."""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM); rm = rows < M; r64 = rows.to(tl.int64)
    kk = tl.arange(0, BK); rr = tl.arange(0, R)
    amax = tl.zeros([BM], tl.float32)
    for k0 in range(0, K, BK):
        y = tl.load(Y + r64[:, None] * K + k0 + kk[None, :], mask=rm[:, None], other=0.).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(y), axis=1))
    s = tl.math.div_rn(tl.maximum(amax, 1e-8), 7.0) * CLIP4
    tl.store(SA + rows, s, mask=rm)
    acc = tl.zeros([BM, R], tl.float32)
    jj = tl.arange(0, BK // 2)
    for k0 in range(0, K, BK):
        y = tl.load(Y + r64[:, None] * K + k0 + kk[None, :], mask=rm[:, None], other=0.).to(tl.float32)
        q = tl.minimum(tl.maximum(libdevice.rint(tl.math.div_rn(y, s[:, None])), -7.0), 7.0)
        dx = (y - s[:, None] * q).to(tl.bfloat16)
        a = tl.load(A + (k0 + kk)[:, None].to(tl.int64) * R + rr[None, :])
        acc += tl.dot(dx, a)
        qi = q.to(tl.int32)
        lo, hi = tl.split(tl.reshape(qi, (BM, BK // 2, 2)))
        tl.store(Q + r64[:, None] * (K // 2) + k0 // 2 + jj[None, :], ((lo & 15) | ((hi & 15) << 4)).to(tl.uint8), mask=rm[:, None])
    tl.store(Z + r64[:, None] * R + rr[None, :], acc.to(tl.bfloat16), mask=rm[:, None])
    if HAS_STAT:
        w = tl.load(W + rows, mask=rm, other=0.)
        tl.atomic_add(STAT + (pid % 64), tl.sum(w * tl.sum(acc * acc, axis=1), axis=0))


def qz(y, A, Q, SA, Z, BM=32, BK=128, clip4=0.9, stat=None, w=None):
    M, K = y.shape; R = A.shape[1]
    _qz_k[(triton.cdiv(M, BM),)](y, A, Q, SA, Z, M, K=K, R=R, BM=BM, BK=BK, CLIP4=clip4, STAT=stat if stat is not None else SA,
                                 W=w if w is not None else SA, HAS_STAT=stat is not None, num_warps=4, enable_fp_fusion=False)
