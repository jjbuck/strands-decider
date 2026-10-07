"""m1.py -- agent M1's all-attention hobson (ARCH.md v1) for Inferentia2 timing.  Shapes only: the 18 former GDN layers keep
hobson's projections (random-free: hobson's own weights are used where shapes match; the new q/k gains are ones), and their
mixer is causal softmax attention, 16 heads x 128 (MHA), with RoPE on dims 0..63, q/k RMSNorm over dims 0..121, the GDN gates
as in-head biases in dims 122..127 (3-way bf16 split), scale 1.0, then hobson's gated RMSNorm and out-projection.
The 6 attention layers, MLPs and norms are hobson's (HobN5 token-major layout)."""
import torch, torch.nn as nn, torch.nn.functional as F
import hob5
from hob import rms_zc, softplus, EPS


def cumsum_T(g):
    """blocked cumulative sum over dim 1 of g [B, T, H] (T % 128 == 0) with two small triangular matmuls."""
    B, T, H = g.shape
    n = T // 128
    ar = torch.arange(128, device=g.device)
    U = (ar[:, None] >= ar[None, :]).float()                       # [i, j] = 1 if j <= i
    w = U @ g.reshape(B, n, 128, H)                                # within-chunk inclusive sums
    if n == 1:
        return w.reshape(B, T, H)
    an = torch.arange(n, device=g.device)
    Ux = (an[:, None] > an[None, :]).float()                       # exclusive over chunks
    pre = Ux @ w[:, :, -1, :]                                      # [B, n, H]
    return (w + pre[:, :, None, :]).reshape(B, T, H)


def split3(x):
    a = x.to(torch.bfloat16).float(); r = x - a
    b = r.to(torch.bfloat16).float(); c = (r - b).to(torch.bfloat16).float()
    return a, b, c


class M1(hob5.HobN5):
    def __init__(self, W, dtype=torch.bfloat16):
        super().__init__(W, dtype=dtype)
        for i, m in enumerate(self.L):
            if self.types[i] == 'gdn':
                m.qg1 = nn.Parameter(torch.ones(16, 122), requires_grad=False)
                m.kg1 = nn.Parameter(torch.ones(16, 122), requires_grad=False)

    def m1_mix(self, m, p, ab, cos, sin, B, T, tail=None, G0=None):
        """p [B*T, 8192] bf16 projections; returns attention inputs q, k, v [B, 16, T, 128] (bf16) and the per-head G."""
        y = self.conv_tok(m, p[:, :6144].float().reshape(B, T, 6144), tail)          # [B, T, 6144] fp32
        y = y.reshape(B, T, 3, 16, 128)
        q, k, v = y[:, :, 0], y[:, :, 1], y[:, :, 2]
        b = ab[:, :16].reshape(B, T, 16); g = (m.negA * softplus(ab[:, 16:] + m.dtb)).reshape(B, T, 16)
        G = cumsum_T(g) if G0 is None else G0[:, None] + cumsum_T(g)                      # [B, T, 16]
        Bk = F.logsigmoid(b) - G
        qc = q[..., :122] * torch.rsqrt(q[..., :122].pow(2).mean(-1, keepdim=True) + EPS) * m.qg1 * (128 ** -0.5)
        kc = k[..., :122] * torch.rsqrt(k[..., :122].pow(2).mean(-1, keepdim=True) + EPS) * m.kg1
        qc = self.rope(qc.reshape(B * T, 16, 122), cos, sin).reshape(B, T, 16, 122)
        kc = self.rope(kc.reshape(B * T, 16, 122), cos, sin).reshape(B, T, 16, 122)
        g1, g2, g3 = split3(G); b1, b2, b3 = split3(Bk)
        one = torch.ones_like(g1)
        qx = torch.stack([g1, g2, g3, one, one, one], -1)
        kx = torch.stack([one, one, one, b1, b2, b3], -1)
        qf = torch.cat([qc, qx], -1).to(self.dt).transpose(1, 2)
        kf = torch.cat([kc, kx], -1).to(self.dt).transpose(1, 2)
        return qf, kf, v.to(self.dt).transpose(1, 2), G

    @staticmethod
    def mattn(q, k, v, mask):  # q [B,16,Tq,128], k/v [B,16,Tk,128], mask [Tq,Tk] bool -> [B,16,Tq,128]
        s = (q @ k.transpose(-1, -2)).float()
        p = torch.softmax(s.masked_fill(~mask, -30000.0), -1).to(v.dtype)
        return p @ v

    def forward(self, ids, sel):
        T = ids.shape[0]
        x = F.embedding(ids, self.embed)
        cos, sin = self.rope_tab(torch.arange(T, device=ids.device))
        ar = torch.arange(T, device=ids.device); cmask = ar[None, :] <= ar[:, None]
        for i, m in enumerate(self.L):
            h = rms_zc(x, m.in1)
            if self.types[i] == 'gdn':
                p = h @ m.WqkvzT; ab = (h @ m.WabT).float()
                q, k, v, _ = self.m1_mix(m, p, ab, cos, sin, 1, T)
                o = self.mattn(q, k, v, cmask)[0].transpose(0, 1).float()                 # [T, 16, 128]
                x = x + self.gdn_out(m, o, p[:, 6144:])
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
        j = torch.arange(Ls + Lq, device=dev)[None, :]
        bmask = torch.where(j < Ls, j < ns, (j - Ls) <= ar_q[:, None])
        smask = ar_s[None, :] <= ar_s[:, None]
        for li, m in enumerate(self.L):
            h = rms_zc(x, m.in1)
            if self.types[li] == 'gdn':
                p = h @ m.WqkvzT; ab = (h @ m.WabT).float()
                qs, ks, vs, G = self.m1_mix(m, p[:Ls], ab[:Ls], cos[:Ls], sin[:Ls], 1, Ls)
                o_s = self.mattn(qs, ks, vs, smask)[0].transpose(0, 1).float()
                tail = p[ns - 3:ns, :6144].float()
                qb, kb, vb, _ = self.m1_mix(m, p[Ls:], ab[Ls:], cos[Ls:], sin[Ls:], M, Lq,
                                            tail=tail[None].expand(M, 3, 6144), G0=G[:, ns - 1].expand(M, 16))
                kf = torch.cat([ks.expand(M, -1, -1, -1), kb], 2); vf = torch.cat([vs.expand(M, -1, -1, -1), vb], 2)
                o_b = self.mattn(qb, kf, vf, bmask).transpose(1, 2).float().reshape(M * Lq, 16, 128)
                x = x + self.gdn_out(m, torch.cat([o_s, o_b], 0), p[:, 6144:])
            else:
                x = x + self.attn_tok(m, h, cos, sin, Ls=Ls, M=M, Lq=Lq, smask=smask, bmask=bmask)
            x = x + self.mlpT(m, x)
        xb = x[Ls:].reshape(M, Lq, 2048)
        rows = torch.gather(xb, 1, sel[:, :, None].expand(-1, -1, 2048))
        return rms_zc(rows, self.norm1).float()
