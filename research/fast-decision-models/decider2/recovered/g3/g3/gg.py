"""G3 training-path grouped GEMM on the padded sorted layout (rows of each expert contiguous, padded to BM), Triton, with autograd.
Y[p, n] = sum_k X[p, k] * W[e(p), n, k]   (W strides generic: forward [E,N,K]; dX uses the transposed view; dW by a per-expert reduction)."""
import torch, triton, triton.language as tl

BM = 64


@triton.jit
def _gg_k(X, W, Y, C, BEXP, NPAD, N, K, swe, swn, swk, NT, ADD: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // NT; pid_n = pid % NT
    if pid_m * BM >= tl.load(NPAD):
        return
    e = tl.load(BEXP + pid_m).to(tl.int64)
    rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    acc = tl.zeros([BM, BN], tl.float32)
    for kk in range(0, K, BK):
        kmask = (kk + rk) < K
        a = tl.load(X + rm[:, None] * K + (kk + rk)[None, :], mask=kmask[None, :], other=0.)
        b = tl.load(W + e * swe + rn[None, :].to(tl.int64) * swn + (kk + rk)[:, None].to(tl.int64) * swk,
                    mask=kmask[:, None] & (rn[None, :] < N), other=0.)
        acc = tl.dot(a, b, acc)
    if ADD:
        acc += tl.load(C + rm[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0.).to(tl.float32)
    tl.store(Y + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=rn[None, :] < N)


@triton.jit
def _gw_k(DY, X, PST, PEN, DW, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """DW[e, n, k] = sum_{p in expert e} DY[p, n] * X[p, k]  (fp32 out)"""
    e = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    s = tl.load(PST + e); t = tl.load(PEN + e)
    rn = pid_n * BN + tl.arange(0, BN); rk = pid_k * BK + tl.arange(0, BK); rm = tl.arange(0, BM)
    acc = tl.zeros([BN, BK], tl.float32)
    for m in range(s, t, BM):
        rows = (m + rm).to(tl.int64)
        dy = tl.load(DY + rows[:, None] * N + rn[None, :], mask=rn[None, :] < N, other=0.)
        x = tl.load(X + rows[:, None] * K + rk[None, :], mask=rk[None, :] < K, other=0.)
        acc = tl.dot(tl.trans(dy), x, acc)
    tl.store(DW + e.to(tl.int64) * N * K + rn[:, None].to(tl.int64) * K + rk[None, :], acc, mask=(rn[:, None] < N) & (rk[None, :] < K))


def _cfg(N, K):
    BN = 128 if N >= 128 else max(16, triton.next_power_of_2(N))
    BK = 64 if K >= 64 else max(16, triton.next_power_of_2(K))
    return BN, BK


def gg_raw(X, W, meta, transpose=False, C=None):
    """X [P, K] bf16 contiguous (padded layout); W [E, N, K] (or [E, K, N] if transpose -> uses W[e]^T)"""
    bexp, npad, P = meta["bexp"], meta["npad"], X.shape[0]
    E = W.shape[0]
    if not transpose:
        N, K = W.shape[1], W.shape[2]; swe, swn, swk = W.stride(0), W.stride(1), W.stride(2)
    else:
        K, N = W.shape[1], W.shape[2]; swe, swn, swk = W.stride(0), W.stride(2), W.stride(1)
    assert X.shape[1] == K
    Y = torch.empty(P, N, device=X.device, dtype=torch.bfloat16)
    BN, BK = _cfg(N, K)
    NT = triton.cdiv(N, BN)
    _gg_k[((P // BM) * NT,)](X, W, Y, C if C is not None else Y, bexp, npad, N, K, swe, swn, swk, NT, ADD=C is not None, BM=BM, BN=BN, BK=BK,
                                         num_warps=4, num_stages=3)
    return Y


def gw_raw(dY, X, meta, E):
    N, K = dY.shape[1], X.shape[1]
    DW = torch.empty(E, N, K, device=X.device, dtype=torch.float32)
    BN = 64 if N >= 64 else max(16, triton.next_power_of_2(N)); BK = 64 if K >= 64 else max(16, triton.next_power_of_2(K))
    _gw_k[(E, triton.cdiv(N, BN), triton.cdiv(K, BK))](dY, X, meta["pstart"], meta["pend"], DW, N, K, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2)
    return DW


class GG(torch.autograd.Function):
    """Y = X @ W_e^T (+ C).  W may be fp32 (LoRA master) and is cast to bf16; `scale` multiplies W."""
    @staticmethod
    def forward(ctx, X, W, meta, C=None, scale=1.0):
        Wb = (W * scale).to(torch.bfloat16) if (W.dtype != torch.bfloat16 or scale != 1.0) else W
        ctx.meta = meta; ctx.wgrad = W.requires_grad; ctx.wdtype = W.dtype; ctx.scale = scale; ctx.hasC = C is not None
        ctx.save_for_backward(X if W.requires_grad else None, Wb)
        return gg_raw(X.contiguous(), Wb, meta, C=C.contiguous() if C is not None else None)

    @staticmethod
    def backward(ctx, dY):
        X, Wb = ctx.saved_tensors
        dY = dY.contiguous()
        dX = gg_raw(dY, Wb, ctx.meta, transpose=True) if ctx.needs_input_grad[0] else None
        dW = (gw_raw(dY, X.contiguous(), ctx.meta, Wb.shape[0]) * ctx.scale).to(ctx.wdtype) if ctx.wgrad else None
        return dX, dW, None, (dY if ctx.hasC else None), None


def align(fe, k, E, T):
    """fe [T*k] expert of each slot -> padded sorted layout: tok_pad [P] (token per padded row; T for pads), slot_pad [P] (slot id; -1 pads), meta"""
    S = fe.numel(); dev = fe.device
    counts = torch.bincount(fe, minlength=E)
    padded = (counts + BM - 1) // BM * BM
    pend = torch.cumsum(padded, 0); pstart = pend - padded
    cstart = torch.cumsum(counts, 0) - counts
    order = torch.argsort(fe, stable=True); se = fe[order]
    dest = pstart[se] + (torch.arange(S, device=dev) - cstart[se])
    P = (S // BM + E) * BM
    slot_pad = torch.full((P,), -1, device=dev, dtype=torch.long); slot_pad[dest] = order
    nblk = P // BM
    bexp = torch.searchsorted(pend, torch.arange(nblk, device=dev, dtype=pend.dtype) * BM, right=True).clamp_max_(E - 1).to(torch.int32)
    tok_pad = torch.where(slot_pad >= 0, slot_pad // k, torch.full_like(slot_pad, T))
    inv = torch.empty(S, device=dev, dtype=torch.long); inv[order] = dest   # slot id -> padded row
    meta = dict(bexp=bexp, npad=pend[-1:].to(torch.int32).contiguous(), pstart=pstart.to(torch.int32).contiguous(), pend=pend.to(torch.int32).contiguous())
    meta["inv"] = inv
    return tok_pad, slot_pad, meta, counts


class Combine(torch.autograd.Function):
    """out[t] = sum_j Y[inv[t*k + j]]  (each padded row used at most once; no atomics either way)"""
    @staticmethod
    def forward(ctx, Y, inv, T, k):
        ctx.save_for_backward(inv); ctx.P = Y.shape[0]; ctx.k = k
        return Y.index_select(0, inv).view(T, k, -1).float().sum(1).to(Y.dtype)

    @staticmethod
    def backward(ctx, g):
        inv, = ctx.saved_tensors
        gy = g.new_zeros(ctx.P, g.shape[1])
        gy.index_copy_(0, inv, g.repeat_interleave(ctx.k, 0))
        return gy, None, None, None


class Gather(torch.autograd.Function):
    """xs[p] = x[tok_pad[p]] (pads read the zero row T); backward = Combine forward (no atomics)"""
    @staticmethod
    def forward(ctx, x, tok_pad, inv, k):
        T = x.shape[0]
        ctx.save_for_backward(inv); ctx.T = T; ctx.k = k
        xe = torch.cat([x, x.new_zeros(1, x.shape[1])])
        return xe.index_select(0, tok_pad)

    @staticmethod
    def backward(ctx, g):
        inv, = ctx.saved_tensors
        return g.index_select(0, inv).view(ctx.T, ctx.k, -1).float().sum(1).to(g.dtype), None, None, None
