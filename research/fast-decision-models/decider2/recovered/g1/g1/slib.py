"""G1 structured-projection library: activation covariances, projections of dense W onto
  BTT/Monarch (block-rank-r), low-rank, low-rank + block-diagonal, Kronecker-sum,
by (a) Frobenius (Monarch paper: per-block truncated SVD), (b) activation-whitened per-block SVD, (c) (b) + Adam refinement on the exact
output error tr(E C E^T) (an ALS-like local fit with the full input covariance C).  Plus the trainable BTTLinear module."""
import math, torch, torch.nn as nn, torch.nn.functional as F

# ------------------------------------------------------------------ structure bookkeeping
def btt_r(n_in, n_out, b, frac):
    return max(1, int(round(frac * n_in * n_out / (b * (n_in + n_out)))))


def kron_dims(n_in, n_out):
    """(p, q, s, t): W (n_out x n_in) ~ sum A (p x q) kron B (s x t), n_out = p s, n_in = q t, near-square factors."""
    def split(n):
        a = int(2 ** round(math.log2(math.sqrt(n))))
        while n % a: a //= 2
        return a, n // a
    q, t = split(n_in); p, s = split(n_out)
    return p, q, s, t


# ------------------------------------------------------------------ helpers
def _chol(Cb, lam=1e-3):
    """Cb (..., k, k) -> lower S with S S^T = Cb + lam*mean(diag) I"""
    k = Cb.shape[-1]
    d = torch.diagonal(Cb, dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1)
    I = torch.eye(k, device=Cb.device, dtype=Cb.dtype)
    return torch.linalg.cholesky(Cb + lam * d * I)


def out_err(W, What, C):
    """relative output error sqrt(tr(E C E^T) / tr(W C W^T)), E = W - What (fp32)."""
    E = (W - What).float()
    num = (E @ C * E).sum(); den = (W.float() @ C * W.float()).sum()
    return float((num / den).clamp(min=0).sqrt())


def frob_err(W, What):
    return float((W.float() - What.float()).norm() / W.float().norm())


# ------------------------------------------------------------------ BTT
def btt_from_factors(R, L):
    b_in, N1, k = R.shape; b_out, kp, N2 = L.shape; r = N1 // b_out
    W = torch.einsum('jcir,ijrk->jcik', L.view(b_out, kp, b_in, r), R.view(b_in, b_out, r, k))
    return W.reshape(b_out * kp, b_in * k)


def btt_project(W, b_in, b_out, r, C=None, lam=1e-3):
    """per-block truncated SVD. C None -> Frobenius (Monarch paper's projection); else whitened by the per-input-block covariance."""
    W = W.float(); n_out, n_in = W.shape; k = n_in // b_in; kp = n_out // b_out
    Wb = W.view(b_out, kp, b_in, k).permute(0, 2, 1, 3)            # [j, i, k', k]
    if C is not None:
        Cb = torch.stack([C[i * k:(i + 1) * k, i * k:(i + 1) * k] for i in range(b_in)])  # [i, k, k]
        S = _chol(Cb.float(), lam)                                 # [i, k, k]
        A = Wb @ S[None]                                           # [j, i, k', k]
    else:
        A = Wb
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)            # [j,i,k',m], [j,i,m], [j,i,m,k]
    r = min(r, s.shape[-1])
    sq = s[..., :r].sqrt()
    Lb = U[..., :r] * sq[..., None, :]                              # [j, i, k', r]
    Rb = sq[..., :, None] * Vh[..., :r, :]                          # [j, i, r, k]
    if C is not None:
        # R S = Rb  ->  R = Rb S^{-1}  : solve S^T R^T = Rb^T
        Rb = torch.linalg.solve_triangular(S[None].transpose(-1, -2), Rb.transpose(-1, -2), upper=True).transpose(-1, -2)
    R = Rb.permute(1, 0, 2, 3).reshape(b_in, b_out * r, k)          # [i, (j r), k]
    L = Lb.permute(0, 2, 1, 3).reshape(b_out, kp, b_in * r)         # [j, k', (i r)]
    return R.contiguous(), L.contiguous()


def lowrank_project(W, r, C=None, lam=1e-3):
    R, L = btt_project(W, 1, 1, r, C, lam)
    return R, L


# ------------------------------------------------------------------ low-rank + block-diagonal
def lrbd_from(U, V, D):
    b, kp, k = D.shape
    return U @ V + torch.block_diag(*D)


def lrbd_project(W, b, r, C=None, iters=4, lam=1e-3):
    W = W.float(); n_out, n_in = W.shape; k = n_in // b; kp = n_out // b
    D = torch.stack([W[j * kp:(j + 1) * kp, j * k:(j + 1) * k] for j in range(b)])
    for _ in range(iters):
        Rr, Ll = lowrank_project(W - torch.block_diag(*D), r, C, lam)
        LR = Ll[0] @ Rr[0]
        Res = W - LR
        D = torch.stack([Res[j * kp:(j + 1) * kp, j * k:(j + 1) * k] for j in range(b)])
    return Ll[0].contiguous(), Rr[0].contiguous(), D.contiguous()


# ------------------------------------------------------------------ Kronecker sum (Van Loan-Pitsianis)
def kron_project(W, S, dims=None):
    W = W.float(); n_out, n_in = W.shape
    p, q, s, t = dims or kron_dims(n_in, n_out)
    Rw = W.view(p, s, q, t).permute(0, 2, 1, 3).reshape(p * q, s * t)
    U, sv, Vh = torch.linalg.svd(Rw, full_matrices=False)
    sq = sv[:S].sqrt()
    A = (U[:, :S] * sq).T.reshape(S, p, q)
    Bm = (sq[:, None] * Vh[:S]).reshape(S, s, t)
    return A, Bm


def kron_from(A, Bm):
    S, p, q = A.shape; _, s, t = Bm.shape
    return torch.einsum('npq,nst->psqt', A, Bm).reshape(p * s, q * t)


def kron_macs(A, Bm):
    S, p, q = A.shape; _, s, t = Bm.shape
    return S * (q * s * t + p * q * s)


# ------------------------------------------------------------------ refinement on the exact output error
def refine(params, recon, W, C, steps=300, lr=None, verbose=False):
    """Adam on tr(E C E^T)/tr(W C W^T); params: list of fp32 tensors (modified in place)."""
    W = W.float(); C = C.float()
    ps = [p.detach().clone().requires_grad_(True) for p in params]
    den = (W @ C * W).sum()
    scale = [float(p.detach().abs().mean()) for p in ps]
    opt = torch.optim.Adam([dict(params=[p], lr=(lr or 2e-3) * sc) for p, sc in zip(ps, scale)])
    best = None
    for it in range(steps):
        E = W - recon(*ps)
        loss = (E @ C * E).sum() / den
        if best is None or float(loss) < best[0]: best = (float(loss), [p.detach().clone() for p in ps])
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it % 50 == 0: print(it, float(loss) ** 0.5)
    return best[1]


# ------------------------------------------------------------------ trainable BTT module
class BTTLinear(nn.Module):
    """y = BTT(x) (+ bias). Factors stored like btt.py: R (b_in, b_out r, k), L (b_out, k', b_in r)."""
    def __init__(self, R, L, bias=None, dtype=torch.bfloat16):
        super().__init__()
        self.R = nn.Parameter(R.to(dtype).contiguous()); self.L = nn.Parameter(L.to(dtype).contiguous())
        self.bias = None if bias is None else nn.Parameter(bias.to(dtype), requires_grad=False)
        self.b_in, N1, self.k = R.shape; self.b_out, self.kp, _ = L.shape; self.r = N1 // self.b_out
        self.in_features = self.b_in * self.k; self.out_features = self.b_out * self.kp

    def forward(self, x):
        sh = x.shape; x2 = x.reshape(-1, sh[-1]); M = x2.shape[0]
        if self.b_in == 1 and self.b_out == 1:
            y = (x2 @ self.R[0].t()) @ self.L[0].t()
        else:
            z = torch.bmm(x2.view(M, self.b_in, self.k).transpose(0, 1), self.R.transpose(1, 2))      # (b_in, M, b_out r)
            z = z.view(self.b_in, M, self.b_out, self.r).permute(2, 1, 0, 3).reshape(self.b_out, M, self.b_in * self.r)
            y = torch.bmm(z, self.L.transpose(1, 2)).transpose(0, 1).reshape(M, self.out_features)
        if self.bias is not None: y = y + self.bias
        return y.reshape(*sh[:-1], self.out_features)

    def macs(self):
        return self.r * (self.b_out * self.in_features + self.b_in * self.out_features)


class LRBDLinear(nn.Module):
    def __init__(self, U, V, D, dtype=torch.bfloat16):
        super().__init__()
        self.U = nn.Parameter(U.to(dtype).contiguous()); self.V = nn.Parameter(V.to(dtype).contiguous()); self.D = nn.Parameter(D.to(dtype).contiguous())
        self.b, self.kp, self.k = D.shape
        self.in_features = self.b * self.k; self.out_features = self.b * self.kp

    def forward(self, x):
        sh = x.shape; x2 = x.reshape(-1, sh[-1]); M = x2.shape[0]
        y = (x2 @ self.V.t()) @ self.U.t()
        yb = torch.bmm(x2.view(M, self.b, self.k).transpose(0, 1), self.D.transpose(1, 2)).transpose(0, 1).reshape(M, -1)
        return (y + yb).reshape(*sh[:-1], self.out_features)

    def macs(self):
        return self.U.shape[1] * (self.in_features + self.out_features) + self.b * self.k * self.kp
