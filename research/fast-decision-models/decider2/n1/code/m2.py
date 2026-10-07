"""m2.py -- agent M2's decision-native layout (ARCH.md v1) for Inferentia2 timing (shapes only; hobson weights where shapes match,
random constants for the compiled U prefix).

Rows: U (4 constant tokens, compiled: K/V per attention layer, GDN state S_U + conv tail per GDN layer), state rows in fixed
blocks of `blk` tokens, question rows (M branches of Lq).
  * Layers 0..k-1, state rows: GEMMs over all live state rows; attention = per-block causal attention over [U + own block]
    (block-local positions); GDN = per-block recurrence from S_U (gdn kernel with batch = number of blocks) plus one native-order
    scan over all state rows from S_U whose final state the question rows read.
  * Layers k..23, state rows frozen at the layer-k residual: one GEMM per deep attention layer for its K/V, one GEMM per deep GDN
    layer for q|k|v|b|a + conv + scan (final state only).  No Wo / MLP for state rows.
  * Question rows: all 24 layers; attention to U + all state keys + own branch; GDN from the composed state.
forward_packed(s_ids [Ls], b_ids [M, Lq], sel [M, S], n_s) with Ls a multiple of blk (pads masked)."""
import os
import torch, torch.nn as nn, torch.nn.functional as F
import hob5
from hob import rms_zc, softplus, EPS


class M2(hob5.HobN5):
    def __init__(self, W, dtype=torch.bfloat16, k=8, blk=256):
        super().__init__(W, dtype=dtype)
        self.k, self.blk = k, blk
        g = torch.Generator().manual_seed(0)
        for i, m in enumerate(self.L):
            if self.types[i] == 'gdn':
                m.SU = nn.Parameter(torch.randn(1, 128, 16, 128, generator=g) * 0.02, requires_grad=False)
                m.tailU = nn.Parameter((torch.randn(3, 6144, generator=g) * 0.1).to(dtype), requires_grad=False)
                if i >= k:
                    m.WkvbaT = nn.Parameter(torch.cat([m.WqkvzT[:, :6144], m.WabT], 1).contiguous(), requires_grad=False)
            else:
                # U's 4 keys padded to 128 (masked) so every key length is a multiple of 128 (a 260-key block attention
                # failed Neuron SBUF allocation, measured)
                ku = torch.zeros(2, 128, 256); ku[:, :4] = torch.randn(2, 4, 256, generator=g) * 0.1
                vu = torch.zeros(2, 128, 256); vu[:, :4] = torch.randn(2, 4, 256, generator=g) * 0.1
                m.KU = nn.Parameter(ku.to(dtype), requires_grad=False); m.VU = nn.Parameter(vu.to(dtype), requires_grad=False)
        deep_attn = [i for i in range(24) if self.types[i] == 'attn' and i >= k]
        self.deep_attn = deep_attn
        if deep_attn:   # one GEMM for every deep attention layer's K|V
            self.WkvDeep = nn.Parameter(torch.cat([self.L[i].WinT[:, 4096:5120] for i in deep_attn], 1).contiguous(), requires_grad=False)

    def gdn_qkvbg(self, m, p, ab, B, T, tail):
        y = self.conv_tok(m, p[:, :6144].float().reshape(B, T, 6144), tail)
        q, k, v = self.gdn_qkv(y, B, T)
        beta = torch.sigmoid(ab[:, :16]).reshape(B, T, 16); g = (m.negA * softplus(ab[:, 16:] + m.dtb)).reshape(B, T, 16)
        return q, k, v, g, beta

    @staticmethod
    def mattn(q, k, v, mask):  # q [B,8,Tq,256], k/v [B,2,Tk,256]
        k = k.repeat_interleave(4, dim=1); v = v.repeat_interleave(4, dim=1)
        s = (q @ k.transpose(-1, -2)).float() * (1.0 / 16.0)
        p = torch.softmax(s.masked_fill(~mask, -30000.0), -1).to(v.dtype)
        return p @ v

    def forward_packed(self, s_ids, b_ids, sel, n_s=None):
        dev = s_ids.device
        Ls = s_ids.shape[0]; M, Lq = b_ids.shape; blk = self.blk; nb = Ls // blk
        ns = Ls if n_s is None else n_s
        xs = F.embedding(s_ids, self.embed); xq = F.embedding(b_ids.reshape(-1), self.embed)
        ar_s = torch.arange(Ls, device=dev); ar_q = torch.arange(Lq, device=dev); ar_b = torch.arange(blk, device=dev)
        cos_l, sin_l = self.rope_tab(4 + ar_s % blk)                       # block-local positions
        cos_q, sin_q = self.rope_tab((4 + ns + ar_q).repeat(M))
        vs = (ar_s < ns).float()
        # masks: block attention [blk, 4 + blk]; question attention [Lq, 4 + Ls + Lq]
        U = 128
        jb = torch.arange(U + blk, device=dev)[None, :]
        bmask_blk = (jb < 4) | ((jb >= U) & ((jb - U) <= ar_b[:, None]))
        jq = torch.arange(U + Ls + Lq, device=dev)[None, :]
        qmask = (jq < 4) | ((jq >= U) & (jq < U + Ls) & (jq - U < ns)) | ((jq >= U + Ls) & ((jq - U - Ls) <= ar_q[:, None]))
        Ksd = {}
        for li, m in enumerate(self.L):
            hq = rms_zc(xq, m.in1)
            live = li < self.k
            if li == self.k:          # freeze: deep K/V of the state rows in one GEMM
                hk = rms_zc(xs, self.L[li].in1)
                if self.deep_attn:
                    kv = hk @ self.WkvDeep
                    for j, a in enumerate(self.deep_attn):
                        Ksd[a] = kv[:, j * 1024:(j + 1) * 1024]
            if self.types[li] == 'attn':
                pq = hq @ m.WinT
                q_q, k_q, v_q, gate_q = self.attn_prep(m, pq, cos_q, sin_q)
                if live:
                    hs = rms_zc(xs, m.in1); ps = hs @ m.WinT
                    q_s, k_s, v_s, gate_s = self.attn_prep(m, ps, cos_l, sin_l)
                    qb = q_s.reshape(nb, blk, 8, 256).transpose(1, 2)
                    kb = torch.cat([m.KU[None].expand(nb, -1, -1, -1), k_s.reshape(nb, blk, 2, 256).transpose(1, 2)], 2)
                    vb = torch.cat([m.VU[None].expand(nb, -1, -1, -1), v_s.reshape(nb, blk, 2, 256).transpose(1, 2)], 2)
                    o_s = self.mattn(qb, kb, vb, bmask_blk).transpose(1, 2).reshape(Ls, 2048) * gate_s
                    xs = xs + o_s @ m.WoT
                    k_st, v_st = k_s, v_s
                else:
                    kvs = Ksd[li]
                    k_st = self.rope(rms_zc(kvs[:, :512].reshape(Ls, 2, 256), m.kn1), cos_l, sin_l); v_st = kvs[:, 512:].reshape(Ls, 2, 256)
                qb = q_q.reshape(M, Lq, 8, 256).transpose(1, 2)
                kf = torch.cat([m.KU[None].expand(M, -1, -1, -1), k_st.transpose(0, 1)[None].expand(M, -1, -1, -1),
                                k_q.reshape(M, Lq, 2, 256).transpose(1, 2)], 2)
                vf = torch.cat([m.VU[None].expand(M, -1, -1, -1), v_st.transpose(0, 1)[None].expand(M, -1, -1, -1),
                                v_q.reshape(M, Lq, 2, 256).transpose(1, 2)], 2)
                o_q = self.mattn(qb, kf, vf, qmask).transpose(1, 2).reshape(M * Lq, 2048) * gate_q
                xq = xq + o_q @ m.WoT
            else:
                if live:
                    hs = rms_zc(xs, m.in1); ps = hs @ m.WqkvzT; abs_ = (hs @ m.WabT).float()
                    qs, ks, vv, gs, bs = self.gdn_qkvbg(m, ps, abs_, nb, blk, m.tailU.float()[None].expand(nb, 3, 6144))
                    ks = ks * vs.reshape(nb, blk)[:, :, None, None]
                    gs = gs * vs.reshape(nb, blk)[:, :, None]; bs = bs * vs.reshape(nb, blk)[:, :, None]
                    o_s, _ = hob5.gdn_call(qs, ks, vv, gs, bs, S0=m.SU.expand(nb, -1, -1, -1))      # per-block recurrences
                    _, S = hob5.gdn_call(qs.reshape(1, Ls, 16, 128), ks.reshape(1, Ls, 16, 128), vv.reshape(1, Ls, 16, 128),
                                         gs.reshape(1, Ls, 16), bs.reshape(1, Ls, 16), S0=m.SU)          # composed state
                    xs = xs + self.gdn_out(m, o_s.reshape(Ls, 16, 128), ps[:, 6144:])
                    tail = ps[ns - 3:ns, :6144].float()
                else:
                    pk = hk @ m.WkvbaT; abk = pk[:, 6144:].float()
                    qs, ks, vv, gs, bs = self.gdn_qkvbg(m, pk, abk, 1, Ls, m.tailU.float()[None])
                    ks = ks * vs[None, :, None, None]; gs = gs * vs[None, :, None]; bs = bs * vs[None, :, None]
                    _, S = hob5.gdn_call(qs, ks, vv, gs, bs, S0=m.SU)
                    tail = pk[ns - 3:ns, :6144].float()
                pq = hq @ m.WqkvzT; abq = (hq @ m.WabT).float()
                qb, kb, vb, gb, bq = self.gdn_qkvbg(m, pq, abq, M, Lq, tail[None].expand(M, 3, 6144))
                o_b, _ = hob5.gdn_call(qb, kb, vb, gb, bq, S0=S.expand(M, -1, -1, -1))
                xq = xq + self.gdn_out(m, o_b.reshape(M * Lq, 16, 128), pq[:, 6144:])
            xq = xq + self.mlpT(m, xq)
            if live:
                xs = xs + self.mlpT(m, xs)
        xb = xq.reshape(M, Lq, 2048)
        rows = torch.gather(xb, 1, sel[:, :, None].expand(-1, -1, 2048))
        return rms_zc(rows, self.norm1).float()
