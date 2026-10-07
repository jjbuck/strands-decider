"""Lean Qwen3.5 inference runtime: fused weights, fla kernels, optional torch.compile'd glue, CUDA graphs."""
import math, torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv

def _rms_zc(x, w, eps):  # zero-centered RMSNorm, fp32 internals
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)).to(x.dtype)

def _add_rms(x, d, w, eps):  # residual add + norm
    x = x + d
    return x, _rms_zc(x, w, eps)

def _lin_glue(proj, A_log, dt_bias):
    # proj: [T, 8224] -> z [T,2048], b [T,16], g [T,16] fp32
    z = proj[:, 6144:8192]
    b = proj[:, 8192:8208]
    a = proj[:, 8208:8224]
    beta = torch.sigmoid(b)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    return z, beta, g

def _gated_norm(o, z, w, eps):  # o,z: [T*16,128]
    of = o.float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    y = w * of.to(o.dtype)
    y = y * F.silu(z.float())
    return y.to(o.dtype)

def _silu_mul(gu, I):
    return F.silu(gu[:, :I]) * gu[:, I:]

def _attn_prep(qkv, qn, kn, cos, sin, eps):
    # qkv [T, 5120]: q+gate 4096 (8 heads x 512: [q256|gate256]), k 512, v 512
    T = qkv.shape[0]
    qg = qkv[:, :4096].reshape(T, 8, 512)
    q, gate = qg[..., :256], qg[..., 256:]
    k = qkv[:, 4096:4608].reshape(T, 2, 256)
    v = qkv[:, 4608:5120].reshape(T, 2, 256)
    q = _rms_zc(q, qn, eps); k = _rms_zc(k, kn, eps)
    def rope(x):
        xr, xp = x[..., :64], x[..., 64:]
        x1, x2 = xr[..., :32], xr[..., 32:]
        c = cos[:, None, :]; s = sin[:, None, :]
        rot = torch.cat([x1 * c[..., :32] - x2 * s[..., :32], x2 * c[..., 32:] + x1 * s[..., 32:]], -1)
        return torch.cat([rot, xp], -1)
    return rope(q), rope(k), v, torch.sigmoid(gate.reshape(T, 2048))

class Lean:
    def __init__(self, torso, compile_glue=False, fuse=True):
        cfg = torso.config
        self.cfg = cfg
        self.dev = next(torso.parameters()).device
        self.eps = cfg.rms_norm_eps
        self.types = cfg.layer_types
        self.embed = torso.embed_tokens.weight
        self.norm_w = torso.norm.weight.float()
        self.layers = []
        for i, L in enumerate(torso.layers):
            d = {"type": self.types[i], "in_norm": L.input_layernorm.weight.float(), "post_norm": L.post_attention_layernorm.weight.float()}
            m = L.mlp
            d["Wgu"] = torch.cat([m.gate_proj.weight, m.up_proj.weight], 0).contiguous()
            d["Wd"] = m.down_proj.weight.contiguous()
            d["I"] = m.gate_proj.weight.shape[0]
            if d["type"] == "linear_attention":
                a = L.linear_attn
                d["Win"] = torch.cat([a.in_proj_qkv.weight, a.in_proj_z.weight, a.in_proj_b.weight, a.in_proj_a.weight], 0).contiguous()
                d["conv_w"] = a.conv1d.weight.squeeze(1).contiguous()
                d["A_log"] = a.A_log.data; d["dt_bias"] = a.dt_bias.data.float()
                d["gn_w"] = a.norm.weight.data; d["Wo"] = a.out_proj.weight.contiguous()
            else:
                a = L.self_attn
                d["Win"] = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous()
                d["qn"] = a.q_norm.weight.float(); d["kn"] = a.k_norm.weight.float(); d["Wo"] = a.o_proj.weight.contiguous()
            self.layers.append(d)
        rp = cfg.rope_parameters
        inv = 1.0 / (rp["rope_theta"] ** (torch.arange(0, 64, 2, dtype=torch.float32, device=self.dev) / 64))
        self.inv = inv
        if compile_glue:
            global _add_rms, _lin_glue, _gated_norm, _silu_mul, _attn_prep, _rms_zc
            _add_rms = torch.compile(_add_rms); _lin_glue = torch.compile(_lin_glue); _gated_norm = torch.compile(_gated_norm)
            _silu_mul = torch.compile(_silu_mul); _attn_prep = torch.compile(_attn_prep); _rms_zc = torch.compile(_rms_zc)

    @torch.no_grad()
    def forward(self, ids):  # ids [B, T] -> hidden [B, T, 2048]
        B, T = ids.shape
        x = F.embedding(ids, self.embed).reshape(B * T, -1)
        pos = torch.arange(T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]
        fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
        if B > 1:
            cos = cos.repeat(B, 1); sin = sin.repeat(B, 1)
        h = _rms_zc(x, self.layers[0]["in_norm"], self.eps)
        n = len(self.layers)
        for i, d in enumerate(self.layers):
            proj = h @ d["Win"].t()
            if d["type"] == "linear_attention":
                z, beta, g = _lin_glue(proj, d["A_log"], d["dt_bias"])
                qkv = proj[:, :6144].reshape(B, T, 6144)
                qkv = fla_conv(qkv, d["conv_w"], None, activation="silu")
                qkv = qkv[0] if isinstance(qkv, tuple) else qkv
                q, k, v = qkv.split(2048, dim=-1)
                q = q.reshape(B, T, 16, 128); k = k.reshape(B, T, 16, 128); v = v.reshape(B, T, 16, 128)
                o, _ = chunk_gated_delta_rule(q, k, v, g.reshape(B, T, 16), beta.reshape(B, T, 16), use_qk_l2norm_in_kernel=True)
                o = _gated_norm(o.reshape(-1, 128), z.reshape(-1, 128), d["gn_w"], self.eps).reshape(B * T, 2048)
            else:
                q, k, v, gate = _attn_prep(proj, d["qn"], d["kn"], cos, sin, self.eps)
                qh = q.reshape(B, T, 8, 256).transpose(1, 2); kh = k.reshape(B, T, 2, 256).transpose(1, 2); vh = v.reshape(B, T, 2, 256).transpose(1, 2)
                o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(B * T, 2048) * gate
            d_out = o @ d["Wo"].t()
            x, h2 = _add_rms(x, d_out, d["post_norm"], self.eps)
            gu = h2 @ d["Wgu"].t()
            m = _silu_mul(gu, d["I"])
            d_mlp = m @ d["Wd"].t()
            nw = self.layers[i + 1]["in_norm"] if i + 1 < n else self.norm_w
            x, h = _add_rms(x, d_mlp, nw, self.eps)
        return h.reshape(B, T, -1)   # final norm applied (h is norm(x) with norm_w)


def _conv(x, w):
    o = fla_conv(x, w, None, activation="silu")
    return o[0] if isinstance(o, tuple) else o

def _forward_packed(self, s_ids, n_s, b_ids):
    """One pass over [state (right-padded to Ls) | M question branches (each right-padded to Lq)].
    Branch m sees exactly state[:n_s] + its own tokens. Token-wise ops run once over all tokens."""
    dev = self.dev
    Ls = s_ids.shape[0]; M, Lq = b_ids.shape
    ids = torch.cat([s_ids, b_ids.reshape(-1)])
    x = F.embedding(ids, self.embed)
    ar_s = torch.arange(Ls, device=dev); ar_q = torch.arange(Lq, device=dev)
    pos = torch.cat([ar_s, (n_s + ar_q).repeat(M)]).float()
    fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
    valid_s = (ar_s < n_s)
    vs = valid_s[:, None]
    tail_idx = (n_s - 3 + torch.arange(3, device=dev)).clamp_min(0)
    # branch attention mask [Lq, Ls+Lq]
    j = torch.arange(Ls + Lq, device=dev)[None, :]
    i = ar_q[:, None]
    bmask = torch.where(j < Ls, j < n_s, (j - Ls) <= i)
    T = Ls + M * Lq
    h = _rms_zc(x, self.layers[0]["in_norm"], self.eps)
    nL = len(self.layers)
    for li, d in enumerate(self.layers):
        proj = h @ d["Win"].t()
        if d["type"] == "linear_attention":
            z, beta, g = _lin_glue(proj, d["A_log"], d["dt_bias"])
            beta_s = beta[:Ls] * vs; g_s = g[:Ls] * vs
            qkv = proj[:, :6144]
            qkv_s = qkv[:Ls]
            tail = qkv_s[tail_idx]
            qkv_b = qkv[Ls:].reshape(M, Lq, 6144)
            cs = _conv(qkv_s[None], d["conv_w"])
            cb = _conv(torch.cat([tail[None].expand(M, -1, -1), qkv_b], 1), d["conv_w"])[:, 3:]
            qs, ks, vs_ = cs.split(2048, -1); qb, kb, vb = cb.split(2048, -1)
            o_s, S = chunk_gated_delta_rule(qs.reshape(1, Ls, 16, 128), ks.reshape(1, Ls, 16, 128), vs_.reshape(1, Ls, 16, 128),
                                            g_s[None], beta_s[None], use_qk_l2norm_in_kernel=True, output_final_state=True)
            o_b, _ = chunk_gated_delta_rule(qb.reshape(M, Lq, 16, 128), kb.reshape(M, Lq, 16, 128), vb.reshape(M, Lq, 16, 128),
                                            g[Ls:].reshape(M, Lq, 16), beta[Ls:].reshape(M, Lq, 16), initial_state=S.expand(M, -1, -1, -1).contiguous(),
                                            use_qk_l2norm_in_kernel=True)
            o = torch.cat([o_s.reshape(Ls, 2048), o_b.reshape(M * Lq, 2048)], 0)
            o = _gated_norm(o.reshape(-1, 128), z.reshape(-1, 128), d["gn_w"], self.eps).reshape(T, 2048)
        else:
            q, k, v, gate = _attn_prep(proj, d["qn"], d["kn"], cos, sin, self.eps)
            qs = q[:Ls].reshape(1, Ls, 8, 256).transpose(1, 2); ks = k[:Ls].reshape(1, Ls, 2, 256).transpose(1, 2); vs2 = v[:Ls].reshape(1, Ls, 2, 256).transpose(1, 2)
            o_s = F.scaled_dot_product_attention(qs, ks, vs2, is_causal=True, enable_gqa=True)   # [1,8,Ls,256]
            qb = q[Ls:].reshape(M, Lq, 8, 256).transpose(1, 2)
            kbr = k[Ls:].reshape(M, Lq, 2, 256).transpose(1, 2); vbr = v[Ls:].reshape(M, Lq, 2, 256).transpose(1, 2)
            kf = torch.cat([ks.expand(M, -1, -1, -1), kbr], 2); vf = torch.cat([vs2.expand(M, -1, -1, -1), vbr], 2)
            kf = kf.repeat_interleave(4, dim=1); vf = vf.repeat_interleave(4, dim=1)
            o_b = F.scaled_dot_product_attention(qb, kf, vf, attn_mask=bmask)                  # [M,8,Lq,256]
            o = torch.cat([o_s.transpose(1, 2).reshape(Ls, 2048), o_b.transpose(1, 2).reshape(M * Lq, 2048)], 0) * gate
        d_out = o @ d["Wo"].t()
        x, h2 = _add_rms(x, d_out, d["post_norm"], self.eps)
        gu = h2 @ d["Wgu"].t()
        d_mlp = _silu_mul(gu, d["I"]) @ d["Wd"].t()
        nw = self.layers[li + 1]["in_norm"] if li + 1 < nL else self.norm_w
        x, h = _add_rms(x, d_mlp, nw, self.eps)
    return h[:Ls], h[Ls:].reshape(M, Lq, -1)

Lean.forward_packed = torch.no_grad()(_forward_packed)
