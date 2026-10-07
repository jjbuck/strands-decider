"""J5 short-M GEMM family (Triton) for hobson's shapes: one template for bf16 / W8A16 / W4A16 (group) / W8A8 (+W4A8 group), with split-K
and a fused fix-up (the last split CTA of a tile reduces through an fp32 workspace and runs the epilogue), so a skinny GEMM can spread over
all SMs without a second kernel.  C[M,N] = A[M,K] @ W[N,K]^T.
  AM    0: A bf16            1: A int8 (per-row scale RA)
  BMODE 0: W bf16 [N,K]      1: W int8 [N,K] (+ per-channel scale SB[N])
        2: W int4, blocked packing [N,K/2]: in each 128-column block, byte j = q[k0+j] | q[k0+64+j] << 4 (unsigned, zero ZB[N,K/G]),
           scales SB[N,K/G], G in {64, 128}
  EPI   0: bf16 store   1: SwiGLU over interleaved (g0,u0,g1,u1..) columns -> bf16 [M,N/2]   3: residual add into R + row sum-of-squares atomics
        4: fp16 store of alpha*acc (raw, no scales: h2's QG.gemm interface)   5: SwiGLU after acc*RA[m]*SB[n] (h2's swiglu_gemm interface)
  PRO_RS: multiply rows by rsqrt(SS[m]/Kd + eps) (the folded RMSNorm of the fold runtime).
"""
import os, json, time, torch, triton, triton.language as tl


@triton.jit
def _sk(A, B, SB, ZB, C, R, SS, SSOUT, RA, WS, CNT, M, N, K, eps, Kd, alpha,
        AM: tl.constexpr, BMODE: tl.constexpr, G: tl.constexpr, EPI: tl.constexpr, PRO_RS: tl.constexpr, SCALE: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0); sp = tl.program_id(1)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    gsz = GROUP * npn
    g = pid // gsz; fm = g * GROUP; gm = min(npm - fm, GROUP)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
    mm = rm < M; nm = rn < N
    KS = K // SPLIT; k_lo = sp * KS
    if AM == 1:
        acc = tl.zeros([BM, BN], dtype=tl.int32)
    else:
        acc = tl.zeros([BM, BN], dtype=tl.float32)
    if BMODE == 2:
        accf = tl.zeros([BM, BN], dtype=tl.float32)
        rh = tl.arange(0, 64)
        a_ptr = A + rm[:, None].to(tl.int64) * K + (k_lo + rh)[None, :]
        b_ptr = B + rn[None, :].to(tl.int64) * (K // 2) + (k_lo // 2 + rh)[:, None]
        for k in range(0, KS, 128):
            kb = (k_lo + k) // G
            a0 = tl.load(a_ptr, mask=mm[:, None], other=0)
            a1 = tl.load(a_ptr + 64, mask=mm[:, None], other=0)
            bq = tl.load(b_ptr, mask=nm[None, :], other=0).to(tl.int32)
            z0 = tl.load(ZB + rn * (K // G) + kb, mask=nm, other=0).to(tl.int32)
            s0 = tl.load(SB + rn * (K // G) + kb, mask=nm, other=0.).to(tl.float32)
            if G == 64:
                z1 = tl.load(ZB + rn * (K // G) + kb + 1, mask=nm, other=0).to(tl.int32)
                s1 = tl.load(SB + rn * (K // G) + kb + 1, mask=nm, other=0.).to(tl.float32)
            else:
                z1 = z0; s1 = s0
            lo = (bq & 15) - z0[None, :]
            hi = (bq >> 4) - z1[None, :]
            if AM == 1:
                p0 = tl.dot(a0, lo.to(tl.int8)); p1 = tl.dot(a1, hi.to(tl.int8))
                if G == 64:
                    accf += p0.to(tl.float32) * s0[None, :] + p1.to(tl.float32) * s1[None, :]
                else:
                    accf += (p0 + p1).to(tl.float32) * s0[None, :]
            else:
                p0 = tl.dot(a0, lo.to(tl.bfloat16)); p1 = tl.dot(a1, hi.to(tl.bfloat16))
                if G == 64:
                    accf += p0 * s0[None, :] + p1 * s1[None, :]
                else:
                    accf += (p0 + p1) * s0[None, :]
            a_ptr += 128; b_ptr += 64
        res = accf
    else:
        rk = tl.arange(0, BK)
        a_ptr = A + rm[:, None].to(tl.int64) * K + (k_lo + rk)[None, :]
        b_ptr = B + rn[None, :].to(tl.int64) * K + (k_lo + rk)[:, None]
        for k in range(0, KS, BK):
            a = tl.load(a_ptr, mask=mm[:, None], other=0)
            b = tl.load(b_ptr, mask=nm[None, :], other=0)
            if BMODE == 1 and AM == 0:
                b = b.to(tl.bfloat16)
            acc = tl.dot(a, b, acc)
            a_ptr += BK; b_ptr += BK
        res = acc.to(tl.float32)
    if SCALE:
        if BMODE == 1:
            res = res * tl.load(SB + rn, mask=nm, other=0.)[None, :]
        if AM == 1:
            res = res * tl.load(RA + rm, mask=mm, other=0.)[:, None]
    if SPLIT > 1:      # deterministic split-K: store the partial tile; _skred sums the SPLIT partials and runs the epilogue
        wp = WS + sp.to(tl.int64) * M * N + rm[:, None].to(tl.int64) * N + rn[None, :]
        tl.store(wp, res, mask=mm[:, None] & nm[None, :])
    else:
        _epi(res, pn, rm, rn, mm, nm, C, R, SS, SSOUT, N, eps, Kd, alpha, EPI, PRO_RS, BM, BN)


@triton.jit
def _epi(res, pn, rm, rn, mm, nm, C, R, SS, SSOUT, N, eps, Kd, alpha, EPI: tl.constexpr, PRO_RS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    if PRO_RS:
        ss = tl.load(SS + rm, mask=mm, other=1.0)
        res = res * tl.rsqrt(ss / Kd + eps)[:, None]
    if EPI == 1 or EPI == 5:
        gg, uu = tl.split(tl.reshape(res, [BM, BN // 2, 2]))
        s = (gg * tl.sigmoid(gg)).to(tl.bfloat16).to(tl.float32)
        out = (s * uu.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        cn = pn * (BN // 2) + tl.arange(0, BN // 2)
        tl.store(C + rm[:, None].to(tl.int64) * (N // 2) + cn[None, :], out, mask=mm[:, None] & (cn[None, :] < N // 2))
    elif EPI == 3:
        cp = rm[:, None].to(tl.int64) * N + rn[None, :]
        r = tl.load(R + cp, mask=mm[:, None] & nm[None, :], other=0.).to(tl.float32)
        s = (r + res.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        tl.store(R + cp, s, mask=mm[:, None] & nm[None, :])
        sf = s.to(tl.float32)
        tl.atomic_add(SSOUT + rm, tl.sum(sf * sf, 1), mask=mm, sem="relaxed")
    elif EPI == 4:
        tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], (res * alpha).to(tl.float16), mask=mm[:, None] & nm[None, :])
    else:
        tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], res.to(tl.bfloat16), mask=mm[:, None] & nm[None, :])


@triton.jit
def _skred(WS, C, R, SS, SSOUT, M, N, eps, Kd, alpha, SPLIT: tl.constexpr, EPI: tl.constexpr, PRO_RS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN)
    mm = rm < M; nm = rn < N
    wp = WS + rm[:, None].to(tl.int64) * N + rn[None, :]
    res = tl.zeros([BM, BN], dtype=tl.float32)
    for s_ in tl.static_range(SPLIT):
        res += tl.load(wp + s_ * M * N, mask=mm[:, None] & nm[None, :], other=0.)
    _epi(res, pn, rm, rn, mm, nm, C, R, SS, SSOUT, N, eps, Kd, alpha, EPI, PRO_RS, BM, BN)


# ------------------------------------------------------------------ weights
class QW:
    """a weight for _sk. mode: 'bf16' | 'w8' (int8 per-channel) | 'w4' (int4 blocked, group G)"""
    def __init__(self, mode, w, s=None, z=None, G=128):
        self.mode = mode; self.w = w; self.s = s; self.z = z; self.G = G
        self.N = w.shape[0]; self.K = w.shape[1] * (2 if mode == 'w4' else 1)
        self.bmode = {'bf16': 0, 'w8': 1, 'w4': 2}[mode]

    def nbytes(self):
        n = self.w.numel() * self.w.element_size()
        for t in (self.s, self.z):
            if t is not None: n += t.numel() * t.element_size()
        return n


def pack_blocked4(q):
    """q [N,K] ints in [0,15] -> uint8 [N,K/2]: per 128-col block, byte j = q[k0+j] | q[k0+64+j] << 4"""
    N, K = q.shape
    qb = q.to(torch.int16).reshape(N, K // 128, 2, 64)
    return (qb[:, :, 0, :] | (qb[:, :, 1, :] << 4)).to(torch.uint8).reshape(N, K // 2).contiguous()


def unpack_blocked4(p):
    N, K2 = p.shape
    pb = p.to(torch.int16).reshape(N, K2 // 64, 64)
    return torch.stack([pb & 15, pb >> 4], 2).reshape(N, K2 * 2)


def from_codes(q, s, z, bits, g, sym):
    """j5 wq codes -> QW. w8 sym per-channel; w4 asym groups (sym: zero 8)."""
    if bits == 8 and g is None:
        return QW('w8', q.to(torch.int8).cuda().contiguous(), s.float().reshape(-1).cuda().contiguous())
    assert bits == 4 and g in (64, 128)
    if sym:
        q = (q.to(torch.int16) + 8); z = torch.full_like(s, 8, dtype=torch.uint8)
    return QW('w4', pack_blocked4(q.cuda()), s.float().cuda().contiguous(), z.to(torch.uint8).cuda().contiguous(), G=g)


# ------------------------------------------------------------------ launch + tuning
_WS = {}; _CNT = {}
CFGS = os.path.expanduser('~/work/j5/skcfg.json')
_TAB = json.load(open(CFGS)) if os.path.exists(CFGS) else {}
TUNE = os.environ.get('SKTUNE', '1') == '1'
SPLITS = tuple(int(x) for x in os.environ.get('SKSPLITS', '1,2,4,8').split(','))
_DUMMY = {}


def _dummy(dev):
    if dev not in _DUMMY: _DUMMY[dev] = torch.zeros(16, device=dev)
    return _DUMMY[dev]


def _ws(S, M, N, dev):
    k = (S, M, N)
    if k not in _WS:
        _WS[k] = torch.empty(S, M, N, device=dev, dtype=torch.float32)
    return _WS[k]


def launch(a, w, epi=0, res=None, ss=None, ssout=None, ra=None, alpha=1.0, eps=1e-6, Kd=2048, scale=True, cfg=None, out=None):
    M, K = a.shape; N = w.N
    am = 1 if a.dtype == torch.int8 else 0
    BM, BN, BK, SPLIT, nw, ns = cfg or pick(am, w, M, epi)
    if w.bmode == 2: BK = 128
    dev = a.device; d = _dummy(dev)
    if out is not None: c = out
    elif epi in (1, 5): c = torch.empty(M, N // 2, device=dev, dtype=torch.bfloat16)
    elif epi == 3: c = res
    elif epi == 4: c = torch.empty(M, N, device=dev, dtype=torch.float16)
    else: c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ws = _ws(SPLIT, M, N, dev) if SPLIT > 1 else d; cnt = d
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), SPLIT)
    _sk[grid](a, w.w, w.s if w.s is not None else d, w.z if w.z is not None else d, c, res if res is not None else d,
              ss if ss is not None else d, ssout if ssout is not None else d, ra if ra is not None else d, ws, cnt,
              M, N, K, eps, float(Kd), float(alpha),
              AM=am, BMODE=w.bmode, G=w.G, EPI=epi, PRO_RS=ss is not None, SCALE=scale,
              BM=BM, BN=BN, BK=BK, SPLIT=SPLIT, GROUP=8, num_warps=nw, num_stages=ns)
    if SPLIT > 1:
        RBM = 16 if M <= 16 else 32; RBN = 128
        _skred[(triton.cdiv(M, RBM), triton.cdiv(N, RBN))](ws, c, res if res is not None else d, ss if ss is not None else d, ssout if ssout is not None else d,
                                                          M, N, eps, float(Kd), float(alpha), SPLIT=SPLIT, EPI=epi, PRO_RS=ss is not None, BM=RBM, BN=RBN, num_warps=4)
    return c


def cands(am, w, M, epi):
    out = []
    bms = [b for b in (16, 32, 64, 128) if b <= max(16, triton.next_power_of_2(M))]
    BK = 128 if w.bmode in (1, 2) else 64
    for BM in bms:
        for BN in (64, 128, 256):
            if BM * BN > 128 * 128 and BM > 32: continue
            for SPLIT in SPLITS:
                if (w.K // SPLIT) % BK: continue
                nt = triton.cdiv(M, BM) * triton.cdiv(w.N, BN)
                if SPLIT > 1 and nt >= 160: continue
                nw = 4 if BM * BN <= 8192 else 8
                out.append((BM, BN, BK, SPLIT, nw, 4 if BM * BN <= 16384 else 3))
    return out


def _key(am, w, M, epi):
    return f'{am}|{w.mode}{w.G if w.mode == "w4" else ""}|{M}|{w.N}|{w.K}|{epi}'


def copies(w, min_bytes=48 << 20):
    n = max(1, -(-min_bytes // w.nbytes()))
    out = [w]
    for _ in range(n - 1):
        out.append(QW(w.mode, w.w.clone(), None if w.s is None else w.s.clone(), None if w.z is None else w.z.clone(), w.G))
    return out


def time_cfg(fn, ws, reps=3):
    for x in ws[:2]: fn(x)
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        for x in ws: fn(x)
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) / (reps * len(ws))


def pick(am, w, M, epi):
    k = _key(am, w, M, epi)
    if k in _TAB: return tuple(_TAB[k])
    if not TUNE:
        return (64 if M > 32 else 32, 128, 128 if w.bmode == 2 else 64, 1, 4, 4)
    dev = w.w.device
    a = (torch.randint(-20, 20, (M, w.K), device=dev, dtype=torch.int8) if am else torch.randn(M, w.K, device=dev, dtype=torch.bfloat16))
    res = torch.zeros(M, w.N, device=dev, dtype=torch.bfloat16) if epi == 3 else None
    ss = torch.ones(M, device=dev) if epi in (0, 1, 3) and not am else None
    sso = torch.zeros(M, device=dev) if epi == 3 else None
    ra = torch.ones(M, device=dev) if am else None
    best = None; wl = copies(w)
    for c in cands(am, w, M, epi):
        try:
            t = time_cfg(lambda x: launch(a, x, epi, res=res, ss=ss if epi != 3 else None, ssout=sso, ra=ra, cfg=c), wl)
        except Exception as ex:
            continue
        if best is None or t < best[0]: best = (t, c)
    _TAB[k] = list(best[1])
    return best[1]


def save_tab():
    json.dump(_TAB, open(CFGS, 'w'), indent=0)


# ------------------------------------------------------------------ multi-M-tile skinny GEMM: every CTA owns a BN column slice for ALL rows
# (row blocks B0..B3, powers of two, sum >= M), so each weight tile is read from DRAM once and multiplied against every row block.
@triton.jit
def _mt_blk(A, b, acc, r0, M, K, kk, BR: tl.constexpr, BK: tl.constexpr):
    rr = r0 + tl.arange(0, BR)
    a = tl.load(A + rr[:, None].to(tl.int64) * K + kk[None, :], mask=rr[:, None] < M, other=0)
    return tl.dot(a, b, acc)


@triton.jit
def _mt_out(acc, r0, pn, rn, nm, SB, RA, C, R, SS, SSOUT, WS, sp, M, N, eps, Kd, alpha, AM: tl.constexpr, BMODE: tl.constexpr, SCALE: tl.constexpr,
            EPI: tl.constexpr, PRO_RS: tl.constexpr, SPLIT: tl.constexpr, BR: tl.constexpr, BN: tl.constexpr):
    rm = r0 + tl.arange(0, BR); mm = rm < M
    res = acc.to(tl.float32)
    if SCALE:
        if BMODE == 1:
            res = res * tl.load(SB + rn, mask=nm, other=0.)[None, :]
        if AM == 1:
            res = res * tl.load(RA + rm, mask=mm, other=0.)[:, None]
    if SPLIT > 1:
        wp = WS + sp.to(tl.int64) * M * N + rm[:, None].to(tl.int64) * N + rn[None, :]
        tl.store(wp, res, mask=mm[:, None] & nm[None, :])
    else:
        _epi(res, pn, rm, rn, mm, nm, C, R, SS, SSOUT, N, eps, Kd, alpha, EPI, PRO_RS, BR, BN)


@triton.jit
def _mt(A, B, SB, C, R, SS, SSOUT, RA, WS, M, N, K, eps, Kd, alpha,
        AM: tl.constexpr, BMODE: tl.constexpr, EPI: tl.constexpr, PRO_RS: tl.constexpr, SCALE: tl.constexpr,
        B0: tl.constexpr, B1: tl.constexpr, B2: tl.constexpr, B3: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr):
    pn = tl.program_id(0); sp = tl.program_id(1)
    rn = pn * BN + tl.arange(0, BN); nm = rn < N
    KS = K // SPLIT; k_lo = sp * KS
    rk = tl.arange(0, BK)
    if AM == 1:
        a0 = tl.zeros([B0, BN], dtype=tl.int32)
        if B1 > 0: a1 = tl.zeros([B1, BN], dtype=tl.int32)
        if B2 > 0: a2 = tl.zeros([B2, BN], dtype=tl.int32)
        if B3 > 0: a3 = tl.zeros([B3, BN], dtype=tl.int32)
    else:
        a0 = tl.zeros([B0, BN], dtype=tl.float32)
        if B1 > 0: a1 = tl.zeros([B1, BN], dtype=tl.float32)
        if B2 > 0: a2 = tl.zeros([B2, BN], dtype=tl.float32)
        if B3 > 0: a3 = tl.zeros([B3, BN], dtype=tl.float32)
    b_ptr = B + rn[None, :].to(tl.int64) * K + (k_lo + rk)[:, None]
    for k in range(0, KS, BK):
        kk = k_lo + k + rk
        b = tl.load(b_ptr, mask=nm[None, :], other=0)
        if BMODE == 1 and AM == 0:
            b = b.to(tl.bfloat16)
        a0 = _mt_blk(A, b, a0, 0, M, K, kk, B0, BK)
        if B1 > 0: a1 = _mt_blk(A, b, a1, B0, M, K, kk, B1, BK)
        if B2 > 0: a2 = _mt_blk(A, b, a2, B0 + B1, M, K, kk, B2, BK)
        if B3 > 0: a3 = _mt_blk(A, b, a3, B0 + B1 + B2, M, K, kk, B3, BK)
        b_ptr += BK
    _mt_out(a0, 0, pn, rn, nm, SB, RA, C, R, SS, SSOUT, WS, sp, M, N, eps, Kd, alpha, AM, BMODE, SCALE, EPI, PRO_RS, SPLIT, B0, BN)
    if B1 > 0: _mt_out(a1, B0, pn, rn, nm, SB, RA, C, R, SS, SSOUT, WS, sp, M, N, eps, Kd, alpha, AM, BMODE, SCALE, EPI, PRO_RS, SPLIT, B1, BN)
    if B2 > 0: _mt_out(a2, B0 + B1, pn, rn, nm, SB, RA, C, R, SS, SSOUT, WS, sp, M, N, eps, Kd, alpha, AM, BMODE, SCALE, EPI, PRO_RS, SPLIT, B2, BN)
    if B3 > 0: _mt_out(a3, B0 + B1 + B2, pn, rn, nm, SB, RA, C, R, SS, SSOUT, WS, sp, M, N, eps, Kd, alpha, AM, BMODE, SCALE, EPI, PRO_RS, SPLIT, B3, BN)


def blocks(M, maxb=256):
    """power-of-two row blocks (>=16) covering M with at most 4 blocks and minimal padding"""
    Mp = -(-M // 16) * 16
    out = []; rest = Mp
    for b in (256, 128, 64, 32, 16):
        while b <= maxb and rest >= b and len(out) < 4:
            out.append(b); rest -= b
    if rest > 0:
        if len(out) < 4: out.append(16 if rest <= 16 else triton.next_power_of_2(rest))
        else: out[-1] = triton.next_power_of_2(out[-1] + rest)
    while len(out) < 4: out.append(0)
    return tuple(out)


def launch_mt(a, w, epi=0, res=None, ss=None, ssout=None, ra=None, alpha=1.0, eps=1e-6, Kd=2048, scale=True, cfg=None, out=None):
    M, K = a.shape; N = w.N
    am = 1 if a.dtype == torch.int8 else 0
    BN, BK, SPLIT, nw, ns, maxb = cfg or pick_mt(am, w, M, epi)
    bl = blocks(M, maxb)
    dev = a.device; d = _dummy(dev)
    if out is not None: c = out
    elif epi in (1, 5): c = torch.empty(M, N // 2, device=dev, dtype=torch.bfloat16)
    elif epi == 3: c = res
    elif epi == 4: c = torch.empty(M, N, device=dev, dtype=torch.float16)
    else: c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ws = _ws(SPLIT, M, N, dev) if SPLIT > 1 else d
    _mt[(triton.cdiv(N, BN), SPLIT)](a, w.w, w.s if w.s is not None else d, c, res if res is not None else d, ss if ss is not None else d,
                                     ssout if ssout is not None else d, ra if ra is not None else d, ws, M, N, K, eps, float(Kd), float(alpha),
                                     AM=am, BMODE=w.bmode, EPI=epi, PRO_RS=ss is not None, SCALE=scale, B0=bl[0], B1=bl[1], B2=bl[2], B3=bl[3],
                                     BN=BN, BK=BK, SPLIT=SPLIT, num_warps=nw, num_stages=ns)
    if SPLIT > 1:
        RBM = 16 if M <= 16 else 32; RBN = 128
        _skred[(triton.cdiv(M, RBM), triton.cdiv(N, RBN))](ws, c, res if res is not None else d, ss if ss is not None else d, ssout if ssout is not None else d,
                                                          M, N, eps, float(Kd), float(alpha), SPLIT=SPLIT, EPI=epi, PRO_RS=ss is not None, BM=RBM, BN=RBN, num_warps=4)
    return c


def cands_mt(am, w, M, epi):
    out = []
    Mp = -(-M // 16) * 16
    for BN in (32, 64, 128):
        if Mp * BN > 64 * 512: continue            # accumulator registers
        for SPLIT in (1, 2, 4, 8):
            ctas = -(-w.N // BN) * SPLIT
            if ctas < 60 or ctas > 800: continue
            for BK in ((64, 128) if am else (32, 64)):
                if (w.K // SPLIT) % BK: continue
                for ns in (2, 3):
                    if ns * (Mp + BN) * BK * (1 if am else 2) > 96 * 1024: continue
                    nw = 8 if Mp * BN >= 8192 else 4
                    out.append((BN, BK, SPLIT, nw, ns, 256))
    return out


def pick_mt(am, w, M, epi):
    k = 'mt|' + _key(am, w, M, epi)
    if k in _TAB: return tuple(_TAB[k][:6])
    dev = w.w.device
    a = (torch.randint(-20, 20, (M, w.K), device=dev, dtype=torch.int8) if am else torch.randn(M, w.K, device=dev, dtype=torch.bfloat16))
    res = torch.zeros(M, w.N, device=dev, dtype=torch.bfloat16) if epi == 3 else None
    ss = torch.ones(M, device=dev) if epi in (0, 1) and not am else None
    sso = torch.zeros(M, device=dev) if epi == 3 else None
    ra = torch.ones(M, device=dev) if am else None
    best = None; wl = copies(w)
    for c in cands_mt(am, w, M, epi):
        try:
            t = time_cfg(lambda x: launch_mt(a, x, epi, res=res, ss=ss, ssout=sso, ra=ra, alpha=1 / 4096, scale=(epi != 4), cfg=c), wl)
        except Exception as ex:
            continue
        if best is None or t < best[0]: best = (t, c)
    _TAB[k] = list(best[1]) + [best[0]]
    return best[1]
