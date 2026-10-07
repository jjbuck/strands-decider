"""hob5.py -- HobN5: hobson-v19 for Neuron with the gdn5 NKI kernel and a token-major GDN layout.

Changes against J8's HobNL (same function):
  * GDN layers use plain token-major GEMMs: h [T,2048] @ WqkvzT [2048, 8192] (q|k|v|z) and h @ WabT [2048, 32]; the conv,
    l2-norm and gated RMSNorm run on [T, 16, 128] views; the output projection is one [T,2048] @ [2048,2048] GEMM.
    No head-major batched matmuls, no per-head einsum output projection, no token<->head transposes.
  * The GDN core is gdn5 (all 16 heads in one NKI program, exact fp32).  Its per-token columns / chunk rows (cumsum of g,
    e^g, beta products) are computed here on [T,16] tensors (tiny).
  * Attention layers and MLPs: plain [T, K] @ [K, N] GEMMs with pre-transposed weights (WqkvT, WoT, WguT, WdT).
HOB_FAKEGDN=1 replaces the GDN core by identity (decomposition only).
"""
import os, math
import torch, torch.nn as nn, torch.nn.functional as F
import hob
from hob import rms_zc, softplus, l2n, EPS

C = 128
FAKE = os.environ.get('HOB_FAKEGDN', '0')   # '1': skip GDN core; '2': skip it but keep all its XLA inputs alive
KMOD = os.environ.get('HOB_KMOD', 'gdn5')
_CST = {}


def _kernel():
    import importlib
    M = importlib.import_module(KMOD)
    return M, getattr(M, KMOD)


def gdn_prep(g, beta, valid=None):
    """g, beta [B, T, 16] fp32 (T % 128 == 0) -> cols [B, T, 16, 6], rows [3, B*4, n, 4, 128]."""
    B, T, H = g.shape
    n = T // C
    ar = torch.arange(C, device=g.device)
    U = (ar[:, None] <= ar[None, :]).float()
    gt = g.reshape(B, n, C, H).transpose(2, 3)                       # [B, n, H, C]
    gc = gt @ U                                                      # in-chunk cumsum
    gl = gc[..., -1:]
    bt = beta.reshape(B, n, C, H).transpose(2, 3)
    eg = torch.exp(gc)
    ge = torch.exp(gt)
    cols = torch.stack([bt, eg, -bt * eg, torch.exp(gl - gc), -gc, bt * ge], -1)     # [B, n, H, C, 6]
    cols = cols.transpose(2, 3).reshape(B, T, H, 6)
    logb = torch.clamp(torch.log(torch.clamp(bt, min=1e-38)), min=-1.0e4)
    el = torch.exp(gl).expand(B, n, H, C)
    rows = torch.stack([gc, gc + logb, el], 0)                       # [3, B, n, H, C]
    rows = rows.reshape(3, B, n, H // 4, 4, C).permute(0, 1, 3, 2, 4, 5).reshape(3, B * (H // 4), n, 4, C)
    return cols.contiguous(), rows.contiguous()


def gdn_prep2(g, beta):
    """lean prep for gdn11: token-major cumsum (one [C,C] matmul), cols [B,T,H,6] without permutes, rows [3, B*n, H, C]."""
    B, T, H = g.shape
    n = T // C
    ar = torch.arange(C, device=g.device)
    Lt = (ar[None, :] <= ar[:, None]).float()                        # [t, t'] = 1 if t' <= t
    g4 = g.reshape(B, n, C, H)
    gc = Lt @ g4                                                     # [B, n, C, H] in-chunk cumsum, token-major
    gl = gc[:, :, C - 1:C, :]
    b4 = beta.reshape(B, n, C, H)
    eg = torch.exp(gc)
    cols = torch.stack([b4, eg, -b4 * eg, torch.exp(gl - gc), -gc, b4 * torch.exp(g4)], -1).reshape(B, T, H, 6)
    logb = torch.clamp(torch.log(torch.clamp(b4, min=1e-38)), min=-1.0e4)
    rows = torch.stack([gc, gc + logb, torch.exp(gl).expand(B, n, C, H)], 0).transpose(3, 4).reshape(3, B * n, H, C)
    return cols, rows


def gdn_call(q, k, v, g, beta, S0=None):
    """q,k,v [B, T, 16, 128] fp32; g, beta [B, T, 16] -> o [B, T, 16, 128], S [B, 128, 16, 128]."""
    B, T, H, Dd = q.shape
    if FAKE == '1':
        return v * 1.0, torch.zeros(B, Dd, H, Dd, device=q.device, dtype=q.dtype)
    M, K = _kernel()
    key = str(q.device)
    if key not in _CST:
        _CST[key] = torch.from_numpy(M.consts_np()).to(q.device)
    if getattr(M, 'KPREP', False):                 # gdn10+: the kernel does the chunk prep from raw g and log(beta)
        lb = torch.clamp(torch.log(torch.clamp(beta, min=1e-38)), min=-1.0e4)
        if S0 is None:
            S0 = torch.zeros(B, Dd, H, Dd, device=q.device, dtype=q.dtype)
        if FAKE == '2':
            z = (q.sum() + k.sum() + g.sum() + lb.sum() + S0.sum()) * 0.0
            return v * 1.0 + z, torch.zeros(B, Dd, H, Dd, device=q.device, dtype=q.dtype) + z
        return K(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), lb.contiguous(), _CST[key], S0.contiguous())
    if getattr(M, 'ROWS2', False):
        cols, rows = gdn_prep2(g, beta)
        if S0 is None:
            S0 = torch.zeros(B, Dd, H, Dd, device=q.device, dtype=q.dtype)
        if FAKE == '2':
            z = (q.sum() + k.sum() + cols.sum() + rows.sum() + S0.sum()) * 0.0
            return v * 1.0 + z, torch.zeros(B, Dd, H, Dd, device=q.device, dtype=q.dtype) + z
        return K(q, k, v, cols, rows, _CST[key], S0)
    if os.environ.get('HOB_ABL') == 'noprep':     # ablation (timing only): skip the [T,16] column/row prep
        n = T // C
        cols = torch.zeros(B, T, H, 6, device=q.device) + g[..., None] * 0; rows = torch.zeros(3, B * (H // 4), n, 4, C, device=q.device)
    else:
        cols, rows = gdn_prep(g, beta)
    if S0 is None:
        S0 = torch.zeros(B, Dd, H, Dd, device=q.device, dtype=q.dtype)
    if FAKE == '2':
        z = (q.sum() + k.sum() + cols.sum() + rows.sum() + S0.sum()) * 0.0
        return v * 1.0 + z, torch.zeros(B, Dd, H, Dd, device=q.device, dtype=q.dtype) + z
    return K(q.contiguous(), k.contiguous(), v.contiguous(), cols, rows, _CST[key], S0.contiguous())


class HobN5(hob.Hob):
    def __init__(self, W, dtype=torch.bfloat16):
        super().__init__(W, dtype=dtype, C=128, attn='explicit')
        P = lambda t: nn.Parameter(t.to(dtype).contiguous(), requires_grad=False)
        for i, m in enumerate(self.L):
            m.WguT = P(m.Wgu.t()); m.WdT = P(m.Wd.t())
            if self.types[i] == 'gdn':
                m.WqkvzT = P(m.Win[:8192].t())                        # [2048, 8192] q|k|v|z
                m.WabT = P(m.Win[8192:8224].t())                      # [2048, 32]   b|a
                m.WoT = P(m.Wo.t())
            else:
                m.WinT = P(m.Win.t())                                 # [2048, 5120] (q|gate interleaved per head) | k | v
                m.WoT = P(m.Wo.t())
            del m.Win, m.Wo, m.Wgu, m.Wd

    def mlpT(self, m, x):
        h2 = rms_zc(x, m.post1)
        gu = h2 @ m.WguT
        I = gu.shape[-1] // 2
        return (F.silu(gu[:, :I]) * gu[:, I:]) @ m.WdT

    def conv_tok(self, m, xs, tail=None):
        """xs [B, T, 6144] fp32 -> silu(causal depthwise conv k=4) rounded to model dtype, as fp32."""
        B, T, Cc = xs.shape
        pre = torch.zeros(B, 3, Cc, device=xs.device, dtype=xs.dtype) if tail is None else tail
        xp = torch.cat([pre, xs], 1)
        w = m.conv
        y = xp[:, 0:T] * w[:, 0] + xp[:, 1:T + 1] * w[:, 1] + xp[:, 2:T + 2] * w[:, 2] + xp[:, 3:T + 3] * w[:, 3]
        return F.silu(y).to(self.dt).float()

    def gdn_qkv(self, y, B, T):
        y = y.reshape(B, T, 3, 16, 128)
        if getattr(_kernel()[0], 'RAW_QK', False) or os.environ.get('HOB_ABL') == 'nol2':   # kernel normalises q, k itself
            return y[:, :, 0], y[:, :, 1], y[:, :, 2]
        return l2n(y[:, :, 0]) * (128 ** -0.5), l2n(y[:, :, 1]), y[:, :, 2]

    def gdn_out(self, m, o, z):  # o [R, 16, 128] fp32, z [R, 2048]
        of = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + EPS)
        yo = (m.gnw * of.to(self.dt)).float() * F.silu(z.float().reshape(-1, 16, 128))
        return yo.to(self.dt).reshape(-1, 2048) @ m.WoT

    def attn_tok(self, m, h, cos, sin, Ls=None, M=0, Lq=0, smask=None, bmask=None):
        R = h.shape[0]
        proj = h @ m.WinT
        q, k, v, gate = self.attn_prep(m, proj, cos, sin)            # q [R,8,256], k,v [R,2,256]
        qh, kh, vh = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        if Ls is None:
            o = self.sdpa(qh[None], kh[None], vh[None], causal=True)[0]                     # [8, R, 256]
        else:
            o_s = self.sdpa(qh[None, :, :Ls], kh[None, :, :Ls], vh[None, :, :Ls], mask=smask, causal=True)[0]
            qb = qh[:, Ls:].reshape(8, M, Lq, 256).transpose(0, 1)
            kb = torch.cat([kh[None, :, :Ls].expand(M, -1, -1, -1), kh[:, Ls:].reshape(2, M, Lq, 256).transpose(0, 1)], 2)
            vb = torch.cat([vh[None, :, :Ls].expand(M, -1, -1, -1), vh[:, Ls:].reshape(2, M, Lq, 256).transpose(0, 1)], 2)
            o_b = self.sdpa(qb, kb, vb, mask=bmask)                                          # [M, 8, Lq, 256]
            o = torch.cat([o_s, o_b.transpose(0, 1).reshape(8, M * Lq, 256)], 1)
        o = o.transpose(0, 1).reshape(R, 2048) * gate
        return o @ m.WoT

    def forward(self, ids, sel):
        T = ids.shape[0]
        x = F.embedding(ids, self.embed)
        cos, sin = self.rope_tab(torch.arange(T, device=ids.device))
        for i, m in enumerate(self.L):
            h = rms_zc(x, m.in1)
            if self.types[i] == 'gdn':
                p = h @ m.WqkvzT                                                    # [T, 8192]
                ab = (h @ m.WabT).float()
                beta = torch.sigmoid(ab[:, :16]); g = m.negA * softplus(ab[:, 16:] + m.dtb)
                y = self.conv_tok(m, p[None, :, :6144].float())
                q, k, v = self.gdn_qkv(y, 1, T)
                o, _ = gdn_call(q, k, v, g[None], beta[None])
                x = x + self.gdn_out(m, o[0], p[:, 6144:])
            else:
                x = x + self.attn_tok(m, h, cos, sin)
            x = x + self.mlpT(m, x)
        return rms_zc(x[sel], self.norm1).float()

    def forward_packed(self, s_ids, b_ids, sel, n_s=None):
        dev = s_ids.device
        Ls = s_ids.shape[0]; M, Lq = b_ids.shape
        ns = Ls if n_s is None else n_s
        x = F.embedding(torch.cat([s_ids, b_ids.reshape(-1)]), self.embed)
        ar_s = torch.arange(Ls, device=dev); ar_q = torch.arange(Lq, device=dev)
        cos, sin = self.rope_tab(torch.cat([ar_s, (ns + ar_q).repeat(M)]))
        vs = (ar_s < ns).float()
        j = torch.arange(Ls + Lq, device=dev)[None, :]
        bmask = torch.where(j < Ls, j < ns, (j - Ls) <= ar_q[:, None])
        smask = (ar_s[None, :] <= ar_s[:, None]) if n_s is not None else None
        for li, m in enumerate(self.L):
            h = rms_zc(x, m.in1)
            if self.types[li] == 'gdn':
                p = h @ m.WqkvzT
                ab = (h @ m.WabT).float()
                beta = torch.sigmoid(ab[:, :16]); g = m.negA * softplus(ab[:, 16:] + m.dtb)
                xs = p[:Ls, :6144].float()
                ys = self.conv_tok(m, xs[None])
                tail = xs[ns - 3:ns]
                yb = self.conv_tok(m, p[Ls:, :6144].float().reshape(M, Lq, 6144), tail=tail[None].expand(M, 3, 6144))
                qs, ks, vv = self.gdn_qkv(ys, 1, Ls)
                ks = ks * vs[None, :, None, None]
                o_s, S = gdn_call(qs, ks, vv, (g[:Ls] * vs[:, None])[None], (beta[:Ls] * vs[:, None])[None])
                qb, kb, vb = self.gdn_qkv(yb, M, Lq)
                o_b, _ = gdn_call(qb, kb, vb, g[Ls:].reshape(M, Lq, 16), beta[Ls:].reshape(M, Lq, 16),
                                  S0=S.expand(M, -1, -1, -1))
                o = torch.cat([o_s[0], o_b.reshape(M * Lq, 16, 128)], 0)
                x = x + self.gdn_out(m, o, p[:, 6144:])
            else:
                x = x + self.attn_tok(m, h, cos, sin, Ls=Ls, M=M, Lq=Lq, smask=smask, bmask=bmask)
            x = x + self.mlpT(m, x)
        xb = x[Ls:].reshape(M, Lq, 2048)
        rows = torch.gather(xb, 1, sel[:, :, None].expand(-1, -1, 2048))
        return rms_zc(rows, self.norm1).float()
