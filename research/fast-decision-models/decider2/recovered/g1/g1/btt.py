"""G1: Block Tensor-Train (BTT) / Monarch structured linear layers: kernels + reference math.

W (n_out x n_in) is split into b_out x b_in blocks; block (j, i) (k' x k, k = n_in/b_in, k' = n_out/b_out) equals L[j,i] @ R[i,j] with rank r.
  stage 1: z[m, j, i, :] = R[i, j] @ x[m, i-block]        (b_in batched GEMMs, M x k -> M x (b_out r))
  stage 2: y[m, j-block] = sum_i L[j, i] @ z[m, j, i, :]   (b_out batched GEMMs, M x (b_in r) -> M x k')
MACs/token = r (b_out n_in + b_in n_out). Monarch (Dao et al. 2022) is the case b_in = b_out = b, r = n/b^2 (middle width n).
b_in = b_out = 1 is plain low rank. Storage:
  R: (b_in, b_out*r, k)    row (j*r + r') of block i -> R[i, j] row r'   ("nn.Linear" layout, K contiguous)
  L: (b_out, k', b_in*r)   column (i*r + r') of block j -> L[j, i] column r'
The Triton stage-1 kernel stores z directly in the (M, b_out, b_in, r) layout stage 2 needs, so there is no permutation copy;
stage 2 is a strided-batched GEMM (cuBLAS or Triton) writing straight into the (M, n_out) output.
"""
import math, torch
import triton, triton.language as tl


def btt_dense(R, L):
    """reconstruct the dense W (n_out x n_in) from factors (reference / eval)."""
    b_in, N1, k = R.shape; b_out, kp, N2 = L.shape
    r = N1 // b_out
    Rb = R.view(b_in, b_out, r, k)            # [i, j, r, k]
    Lb = L.view(b_out, kp, b_in, r)           # [j, k', i, r]
    W = torch.einsum('jcir,ijrk->jcik', Lb.float(), Rb.float())  # [j, k', i, k]
    return W.reshape(b_out * kp, b_in * k)


def btt_ref(x, R, L):
    """reference forward with torch ops (bmm + permute copies). x: (M, n_in)."""
    M, n_in = x.shape
    b_in, N1, k = R.shape; b_out, kp, N2 = L.shape; r = N1 // b_out
    z = torch.bmm(x.view(M, b_in, k).transpose(0, 1), R.transpose(1, 2))         # (b_in, M, b_out r)
    z = z.view(b_in, M, b_out, r).permute(2, 1, 0, 3).reshape(b_out, M, b_in * r)  # copy
    y = torch.bmm(z, L.transpose(1, 2))                                              # (b_out, M, k')
    return y.transpose(0, 1).reshape(M, b_out * kp)                                  # copy


_CFGS = [triton.Config(dict(BM=bm, BN=bn, BK=bk), num_warps=w, num_stages=s) for bm, bn, bk, w, s in
         [(128, 128, 32, 4, 4), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4), (128, 128, 64, 8, 3), (64, 64, 64, 4, 4),
          (256, 128, 32, 8, 3), (128, 256, 32, 8, 3), (64, 64, 32, 4, 5), (32, 64, 64, 4, 4)]]


@triton.autotune(configs=_CFGS, key=['M', 'N1', 'K', 'NBI'])
@triton.jit
def _btt1_k(X, R, Z, RS, M, sxm, N1, K, NBI, RR, ACC16: tl.constexpr, ROWSCALE: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0); i = tl.program_id(1)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N1, BN)
    GROUP = 8
    gsz = GROUP * npn
    g = pid // gsz; fm = g * GROUP; gm = min(npm - fm, GROUP)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = X + rm[:, None].to(tl.int64) * sxm + i * K + rk[None, :]
    b_ptr = R + i.to(tl.int64) * N1 * K + rn[None, :].to(tl.int64) * K + rk[:, None]
    if ACC16:
        acc = tl.zeros([BM, BN], dtype=tl.float16)
    else:
        acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kk in range(0, K, BK):
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] + kk < K), other=0.)
        b = tl.load(b_ptr, mask=(rn[None, :] < N1) & (rk[:, None] + kk < K), other=0.)
        if ACC16:
            acc = tl.dot(a, b, acc, out_dtype=tl.float16)
        else:
            acc = tl.dot(a, b, acc)
        a_ptr += BK; b_ptr += BK
    if ROWSCALE:
        s = tl.load(RS + rm, mask=rm < M, other=1.0)
        acc = acc.to(tl.float32) * s[:, None]
    j = rn // RR; rp = rn % RR
    col = j * (NBI * RR) + i * RR + rp
    z_ptr = Z + rm[:, None].to(tl.int64) * (N1 * NBI) + col[None, :]
    tl.store(z_ptr, acc.to(Z.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N1))


@triton.autotune(configs=_CFGS, key=['M', 'N', 'K', 'NB'])
@triton.jit
def _bgemm_k(A, B, C, M, N, K, NB, sab, sam, sbb, sbn, scb, scm, ACC16: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[b] (M x N) = A[b] (M x K) @ B[b]^T (B[b] is N x K); unit inner strides; arbitrary batch / row strides."""
    pid = tl.program_id(0); bb = tl.program_id(1).to(tl.int64)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    GROUP = 8
    gsz = GROUP * npn
    g = pid // gsz; fm = g * GROUP; gm = min(npm - fm, GROUP)
    pm = fm + (pid % gsz) % gm; pn = (pid % gsz) // gm
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptr = A + bb * sab + rm[:, None].to(tl.int64) * sam + rk[None, :]
    b_ptr = B + bb * sbb + rn[None, :].to(tl.int64) * sbn + rk[:, None]
    if ACC16:
        acc = tl.zeros([BM, BN], dtype=tl.float16)
    else:
        acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kk in range(0, K, BK):
        a = tl.load(a_ptr, mask=(rm[:, None] < M) & (rk[None, :] + kk < K), other=0.)
        b = tl.load(b_ptr, mask=(rn[None, :] < N) & (rk[:, None] + kk < K), other=0.)
        if ACC16:
            acc = tl.dot(a, b, acc, out_dtype=tl.float16)
        else:
            acc = tl.dot(a, b, acc)
        a_ptr += BK; b_ptr += BK
    c_ptr = C + bb * scb + rm[:, None].to(tl.int64) * scm + rn[None, :]
    tl.store(c_ptr, acc.to(C.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def btt1(x, R, b_out, rowscale=None, acc16=False, out=None):
    """stage 1 -> z (M, b_out, b_in, r) contiguous."""
    M, n_in = x.shape
    b_in, N1, k = R.shape; r = N1 // b_out
    assert x.stride(1) == 1 and n_in == b_in * k
    z = out if out is not None else torch.empty(M, b_out, b_in, r, device=x.device, dtype=x.dtype)
    grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(N1, META['BN']), b_in)
    _btt1_k[grid](x, R, z, rowscale if rowscale is not None else x, M, x.stride(0), N1, k, b_in, r,
                  ACC16=acc16, ROWSCALE=rowscale is not None)
    return z


def btt2(z, L, out=None, backend='cublas', acc16=False):
    """stage 2: y (M, b_out*k') from z (M, b_out, b_in, r)."""
    M, b_out, b_in, r = z.shape
    _, kp, N2 = L.shape
    y = out if out is not None else torch.empty(M, b_out * kp, device=z.device, dtype=z.dtype)
    if backend == 'cublas':
        zv = z.view(M, b_out, b_in * r).transpose(0, 1)          # (b_out, M, b_in r), strides (b_in r, b_out b_in r, 1)
        yv = y.view(M, b_out, kp).transpose(0, 1)                # (b_out, M, k'),     strides (k', n_out, 1)
        torch.bmm(zv, L.transpose(1, 2), out=yv)
    else:
        grid = lambda META: (triton.cdiv(M, META['BM']) * triton.cdiv(kp, META['BN']), b_out)
        _bgemm_k[grid](z, L, y, M, kp, N2, b_out, b_in * r, b_out * b_in * r, kp * N2, N2, kp, y.stride(0), ACC16=acc16)
    return y


def btt(x, R, L, rowscale=None, backend='cublas', acc16=False):
    b_out = L.shape[0]
    return btt2(btt1(x, R, b_out, rowscale, acc16), L, backend=backend, acc16=acc16)


def btt_shapes(n_in, n_out, b, frac=None, r=None):
    """(b_in, b_out, r) for a FLOP fraction (dense = 1) or a given block rank."""
    if r is None:
        r = max(1, int(round(frac * n_in * n_out / (b * (n_in + n_out)))))
    return b, b, r


def init_factors(n_in, n_out, b_in, b_out, r, dev='cuda', dtype=torch.bfloat16):
    k = n_in // b_in; kp = n_out // b_out
    R = torch.randn(b_in, b_out * r, k, device=dev, dtype=dtype) / math.sqrt(k)
    L = torch.randn(b_out, kp, b_in * r, device=dev, dtype=dtype) / math.sqrt(b_in * r)
    return R, L


def macs(n_in, n_out, b_in, b_out, r):
    return r * (b_out * n_in + b_in * n_out)
