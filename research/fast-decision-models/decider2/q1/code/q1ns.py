"""Noise shaping of activation rounding along the token (row) axis: first-order error feedback (sigma-delta) per input column.
For rows t = 0..M-1 of one GEMM input (rotated, fp32) with per-token scales s_t (from the row's own absmax, as RTN):
    v_t = x_t + c * e_{t-1};  q_t = clamp(rint(v_t / s_t), -qmax, qmax);  e_t = v_t - s_t q_t
The carry restarts every `chunk` rows (parallel chunks in a kernel). The decision sees sum_t <g_t, e_t - c e_{t-1}> = sum_t <g_t - c g_{t+1}, e_t>,
so the error is suppressed where the decision gradient varies slowly from token to token.
"""
import torch, triton, triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _ns_kernel(X, S, Q, M, K, qmax, c, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pc = tl.program_id(0); pr = tl.program_id(1)
    cols = pc * BLOCK + tl.arange(0, BLOCK); m = cols < K
    e = tl.zeros([BLOCK], dtype=tl.float32)
    r0 = pr * CHUNK
    for j in range(CHUNK):
        t = r0 + j
        if t < M:
            s = tl.load(S + t)
            x = tl.load(X + t * K + cols, mask=m, other=0.0)
            v = x + c * e
            q = libdevice.rint(tl.math.div_rn(v, tl.zeros([BLOCK], dtype=tl.float32) + s))
            q = tl.minimum(tl.maximum(q, -qmax), qmax)
            e = v - q * s
            tl.store(Q + t * K + cols, q.to(tl.int8), mask=m)


def ns_quant(xr, s, qmax, c=1.0, chunk=64):
    """xr [M, K] fp32 contiguous, s [M] fp32 -> int8 codes [M, K]"""
    xr = xr.contiguous().float(); s = s.contiguous().float()
    M, K = xr.shape
    Q = torch.empty(M, K, device=xr.device, dtype=torch.int8)
    BLOCK = 128
    grid = (triton.cdiv(K, BLOCK), triton.cdiv(M, chunk))
    _ns_kernel[grid](xr, s, Q, M, K, float(qmax), float(c), CHUNK=chunk, BLOCK=BLOCK)
    return Q
