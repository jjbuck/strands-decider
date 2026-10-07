"""Q4 B11: ctypes wrapper for libq4sp.so (q4sp.cu), 2:4 compression + lane-native metadata packing, the probe that pins the metadata
layout on the GPU, and exact torch references.
Conventions: weights W codes int8 [N, K] (values in [-qmax, qmax]); activations X codes int8 [T, K]; int4 packed 2/byte, low nibble = even k.
int8 2:4: each group of 4 consecutive k keeps <= 2 values.  int4 (hardware): each group of 8 consecutive k = 4 nibble PAIRS keeps <= 2 pairs."""
import os, ctypes, torch
W_ = os.path.expanduser('~/work/q4')
TY = dict(sp8=0, sp4=1, d8=2, d4=3)
EPI = dict(bf16=0, swiglu=1, fp16a=2, i32=3)
_lib = None
# metadata layout of one (16-row block, 64-byte k-step) as found by probe(): LAYOUT[lane] = list of 8 (row, group) for nibbles 0..7
LAYOUT = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(f'{W_}/libq4sp.so')
        f = _lib.q4sp_gemm; f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 6 + [ctypes.c_longlong] * 2 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        p = _lib.q4sp_probe; p.restype = ctypes.c_int; p.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
    return _lib


def pack4(q):
    q = q.to(torch.int16)
    return ((q[..., 0::2] & 0xF) | ((q[..., 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()


def unpack4(p):
    p = p.to(torch.int16)
    lo = p & 0xF; hi = (p >> 4) & 0xF
    lo = torch.where(lo > 7, lo - 16, lo); hi = torch.where(hi > 7, hi - 16, hi)
    return torch.stack([lo, hi], -1).reshape(*p.shape[:-1], p.shape[-1] * 2).to(torch.int8)


# ------------------------------------------------------------------ candidate metadata layouts: lane (g = lane>>2, t = lane&3) -> 8 (row, group)
def layout_cands():
    C = {}
    def mk(fn): return [fn(l >> 2, l & 3) for l in range(32)]
    # H2: 4 consecutive groups per thread, row g then row g+8
    C['consec'] = mk(lambda g, t: [(g, 4 * t + j) for j in range(4)] + [(g + 8, 4 * t + j) for j in range(4)])
    # H1: groups co-located with the thread's A data (2t, 2t+1, 8+2t, 9+2t)
    C['coloc'] = mk(lambda g, t: [(g, x) for x in (2 * t, 2 * t + 1, 8 + 2 * t, 9 + 2 * t)] + [(g + 8, x) for x in (2 * t, 2 * t + 1, 8 + 2 * t, 9 + 2 * t)])
    # H3: one row per thread pair: t even row g, t odd row g+8, 8 groups each (t>>1 selects k half)
    C['rowpair'] = mk(lambda g, t: [(g + 8 * (t & 1), 8 * (t >> 1) + j) for j in range(8)])
    C['rowpair2'] = mk(lambda g, t: [(g + 8 * (t >> 1), 8 * (t & 1) + j) for j in range(8)])
    # H4: thread holds 8 consecutive groups of one row: t in {0,1} -> row g, {2,3} -> row g+8 ... interleaved halves
    C['halves'] = mk(lambda g, t: [(g, 8 * (t & 1) + j) for j in range(4)] + [(g + 8, 8 * (t & 1) + j) for j in range(4)] if t < 2 else
                     [(g, 8 * (t & 1) + 4 + j) for j in range(4)] + [(g + 8, 8 * (t & 1) + 4 + j) for j in range(4)])
    return C


def _regs_from_bytes(M, rows, offs):
    """M uint8 [R, C]; returns per-lane u32 regs: for each (row_fn, byte_off_fn) pair"""
    out = torch.zeros(32, len(rows), dtype=torch.int64)
    for l in range(32):
        g, t = l >> 2, l & 3
        for j, (rf, of) in enumerate(zip(rows, offs)):
            r = rf(g, t); o = of(g, t)
            b = M[r, o:o + 4].to(torch.int64)
            out[l, j] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
    return out


def _to_u32_tensor(x):   # int64 values in [0, 2^32) -> int32 bit pattern tensor
    x = x.clone(); x[x >= 2 ** 31] -= 2 ** 32
    return x.to(torch.int32)


def probe(kind='sp8', trials=4, seed=0, verbose=True):
    """run one mma.sp per trial with random 2:4 A (16 x 64 logical bytes-steps) and random B; return the candidate layouts that match every trial."""
    g_ = torch.Generator().manual_seed(seed)
    cands = layout_cands(); ok = {k: True for k in cands}
    L = lib()
    for tr in range(trials):
        if kind == 'sp8':
            Kl = 64; qm = 127
            A = torch.zeros(16, Kl, dtype=torch.int64); idx = torch.zeros(16, 16, 2, dtype=torch.int64)
            for r in range(16):
                for G in range(16):
                    p = torch.randperm(4, generator=g_)[:2].sort().values
                    idx[r, G] = p
                    A[r, 4 * G + p] = torch.randint(1, qm + 1, (2,), generator=g_) * (torch.randint(0, 2, (2,), generator=g_) * 2 - 1)
            Ac = torch.zeros(16, 32, dtype=torch.int64)
            for r in range(16):
                for G in range(16):
                    Ac[r, 2 * G] = A[r, 4 * G + idx[r, G, 0]]; Ac[r, 2 * G + 1] = A[r, 4 * G + idx[r, G, 1]]
            Acb = (Ac & 0xFF).to(torch.uint8)
            B = torch.randint(-qm, qm + 1, (64, 8), generator=g_, dtype=torch.int64)
            Bb = (B.t().contiguous() & 0xFF).to(torch.uint8)          # [8 cols][64 k bytes]
            ref = A @ B
        else:
            Kl = 128; qm = 7
            A = torch.zeros(16, Kl, dtype=torch.int64); idx = torch.zeros(16, 16, 2, dtype=torch.int64)
            for r in range(16):
                for G in range(16):
                    p = torch.randperm(4, generator=g_)[:2].sort().values
                    idx[r, G] = p
                    for pp in p.tolist():
                        A[r, 8 * G + 2 * pp: 8 * G + 2 * pp + 2] = torch.randint(-qm, qm + 1, (2,), generator=g_)
            Ac = torch.zeros(16, 64, dtype=torch.int64)
            for r in range(16):
                for G in range(16):
                    for j in range(2):
                        pp = idx[r, G, j].item()
                        Ac[r, 4 * G + 2 * j: 4 * G + 2 * j + 2] = A[r, 8 * G + 2 * pp: 8 * G + 2 * pp + 2]
            Acb = pack4(Ac)                                           # [16, 32] bytes
            B = torch.randint(-qm, qm + 1, (Kl, 8), generator=g_, dtype=torch.int64)
            Bb = pack4(B.t().contiguous())                            # [8, 64] bytes
            ref = A @ B
        Ar = _regs_from_bytes(Acb, [lambda g, t: g, lambda g, t: g + 8, lambda g, t: g, lambda g, t: g + 8],
                              [lambda g, t: 4 * t, lambda g, t: 4 * t, lambda g, t: 16 + 4 * t, lambda g, t: 16 + 4 * t])
        Br = _regs_from_bytes(Bb, [lambda g, t: g] * 4, [lambda g, t, j=j: 16 * j + 4 * t for j in range(4)])
        Ad = _to_u32_tensor(Ar).cuda().contiguous(); Bd = _to_u32_tensor(Br).cuda().contiguous()
        for name, lay in cands.items():
            if not ok[name]: continue
            E = torch.zeros(32, dtype=torch.int64)
            for l in range(32):
                v = 0
                for j, (r, G) in enumerate(lay[l]):
                    v |= int(idx[r, G, 0] | (idx[r, G, 1] << 2)) << (4 * j)
                E[l] = v
            Ed = _to_u32_tensor(E).cuda().contiguous()
            D = torch.zeros(32, 4, dtype=torch.int32, device='cuda')
            rc = L.q4sp_probe(Ad.data_ptr(), Bd.data_ptr(), Ed.data_ptr(), D.data_ptr(), 0 if kind == 'sp8' else 1)
            assert rc == 0, rc
            D = D.cpu().to(torch.int64)
            good = True
            for l in range(32):
                g, t = l >> 2, l & 3
                exp = [ref[g, 2 * t], ref[g, 2 * t + 1], ref[g + 8, 2 * t], ref[g + 8, 2 * t + 1]]
                if [int(x) for x in D[l]] != [int(x) for x in exp]: good = False; break
            ok[name] = ok[name] and good
    res = [k for k, v in ok.items() if v]
    if verbose: print(kind, 'matching metadata layouts:', res, flush=True)
    return res


# ------------------------------------------------------------------ 2:4 masks and compression
def mask24_mag(Wq, kind):
    """keep-mask by magnitude: int8 -> top-2 |w| of each group of 4; int4 -> top-2 pairs (by |w0|+|w1|) of each group of 4 pairs"""
    N, K = Wq.shape
    a = Wq.abs().float()
    if kind == 'sp8':
        g = a.reshape(N, K // 4, 4)
        top = g.topk(2, -1).indices
        m = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, top, True)
        return m.reshape(N, K)
    g = a.reshape(N, K // 8, 4, 2).sum(-1)
    top = g.topk(2, -1).indices
    m = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, top, True)
    return m[..., None].expand(N, K // 8, 4, 2).reshape(N, K)


def compress(Wq, mask, kind, layout=None):
    """Wq int8 codes [N, K] (any values), mask bool [N, K] (2:4 for sp8; pair-2:4 for sp4). Returns (Wc uint8/int8 bytes [N, Kb/2], E int32 [nmb*ksg*32],
    Wdense = the masked codes [N, K] int8 for references). Groups with fewer than 2 kept positions are padded with unkept positions (value 0)."""
    lay = layout or LAYOUT
    assert lay is not None, 'run probe() first'
    dev = Wq.device
    N, K = Wq.shape
    Wm = torch.where(mask, Wq, torch.zeros_like(Wq))
    if kind == 'sp8':
        G = K // 4
        m = mask.reshape(N, G, 4)
        # choose 2 indices per group: kept ones first (in order), padded with the smallest unkept
        key = (~m).to(torch.int64) * 8 + torch.arange(4, device=dev)[None, None, :]
        sel = key.topk(2, -1, largest=False).indices.sort(-1).values          # [N, G, 2] increasing
        vals = torch.gather(Wm.reshape(N, G, 4), -1, sel)                    # [N, G, 2]
        Wc = vals.reshape(N, K // 2).to(torch.int8).contiguous()
        nib = (sel[..., 0] | (sel[..., 1] << 2)).to(torch.int64)            # [N, G] 4-bit per group
        gpk = 16                                                            # groups per 64-byte k-step
    else:
        G = K // 8
        m = mask.reshape(N, G, 4, 2)[..., 0]
        key = (~m).to(torch.int64) * 8 + torch.arange(4, device=dev)[None, None, :]
        sel = key.topk(2, -1, largest=False).indices.sort(-1).values
        pairs = Wm.reshape(N, G, 4, 2)
        vals = torch.gather(pairs, 2, sel[..., None].expand(N, G, 2, 2))      # [N, G, 2, 2]
        Wc = pack4(vals.reshape(N, K // 2))                                   # [N, K/4] bytes
        nib = (sel[..., 0] | (sel[..., 1] << 2)).to(torch.int64)
        gpk = 16
    Kb = K if kind == 'sp8' else K // 2
    ksg = Kb // 64
    nmb = (N + 15) // 16
    Np = nmb * 16
    if Np != N:
        nib = torch.cat([nib, torch.full((Np - N, nib.shape[1]), 4, device=dev, dtype=torch.int64)], 0)
    nib = nib.reshape(nmb, 16, ksg, gpk)                                     # [mb, row, ks, group]
    rows = torch.tensor([[r for (r, G_) in lay[l]] for l in range(32)], device=dev)
    grps = torch.tensor([[G_ for (r, G_) in lay[l]] for l in range(32)], device=dev)
    v = nib[:, rows, :, grps]                                               # [32, 8, mb, ks]  (advanced indexing moves dims first)
    v = v.permute(2, 3, 0, 1)                                               # [mb, ks, lane, 8]
    sh = (4 * torch.arange(8, device=dev, dtype=torch.int64))
    E = (v << sh).sum(-1)                                                   # [mb, ks, lane] in [0, 2^32)
    E = torch.where(E >= 2 ** 31, E - 2 ** 32, E).to(torch.int32).contiguous().reshape(-1)
    return Wc, E, Wm


def gemm(kind, cfg, Wc, E, X, out, N, T, K, sa=None, sb=None, epi='bf16', alpha=1.0, ldx=None, stream=None):
    Kb = K if kind in ('sp8', 'd8') else K // 2
    ldx = Kb if ldx is None else ldx
    st = torch.cuda.current_stream().cuda_stream if stream is None else stream
    rc = lib().q4sp_gemm(TY[kind], cfg, Wc.data_ptr(), 0 if E is None else E.data_ptr(), X.data_ptr(), 0 if sa is None else sa.data_ptr(),
                         0 if sb is None else sb.data_ptr(), out.data_ptr(), out.stride(0), ldx, N, T, Kb, EPI[epi], float(alpha), st)
    if rc: raise RuntimeError(f'q4sp {kind} cfg{cfg} rc {rc} N{N} T{T} K{K}')
    return out


def ref_i32(Wm, Xq):
    return (Xq.double() @ Wm.double().t()).round().to(torch.int64)
