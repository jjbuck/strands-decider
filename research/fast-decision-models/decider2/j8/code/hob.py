"""hob.py -- pure-torch hobson-v19 forward (LoRA merged in fp32), device-agnostic and traceable.

No fla / triton / data-dependent control flow, so the same module runs on CPU (oneDNN/AMX) and traces with torch_neuronx.
GDN: chunked gated delta rule; the intra-chunk triangular inverse (I - N)^-1 is computed as prod_j (I + N^(2^j))
(N strictly lower => nilpotent), i.e. matmuls only, no row-by-row substitution. In-chunk cumsum is a matmul too.
"""
import os, glob, json, math
import torch, torch.nn as nn, torch.nn.functional as F

EPS = 1e-6
HUB = os.path.expanduser('~/.cache/huggingface/hub')


def ckpt_dirs():
    hc = glob.glob(f'{HUB}/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*')[0]
    bc = glob.glob(f'{HUB}/models--Qwen--Qwen3.5-2B-Base/snapshots/*')[0]
    return hc, bc


def load_weights():
    """Base Qwen3.5-2B text weights + hobson LoRA merged in fp32 (W + alpha/r * B@A), head weights, decider config."""
    from safetensors import safe_open
    from safetensors.torch import load_file
    hc, bc = ckpt_dirs()
    cfg = json.load(open(f'{hc}/hobson_config.json'))
    scale = cfg['lora_alpha'] / cfg['lora_r']
    lora = load_file(f'{hc}/lora/adapter_model.safetensors')
    fn = glob.glob(f'{bc}/model.safetensors*.safetensors')[0]
    base = safe_open(fn, 'pt')
    W, nmerged = {}, 0
    for k in base.keys():
        if not k.startswith('model.language_model.'):
            continue
        nk = k[len('model.language_model.'):]
        t = base.get_tensor(k)
        if nk.endswith('.weight'):
            lk = 'base_model.model.' + nk[:-len('.weight')]
            if lk + '.lora_A.weight' in lora:
                A = lora[lk + '.lora_A.weight'].float(); B = lora[lk + '.lora_B.weight'].float()
                t = (t.float() + scale * (B @ A)).to(t.dtype); nmerged += 1
        W[nk] = t
    assert nmerged * 2 == len(lora), (nmerged, len(lora))
    head = load_file(f'{hc}/head.safetensors')
    return W, head, cfg


def rms_zc(x, w1, eps=EPS):  # zero-centred RMSNorm; w1 = (1 + w) in fp32
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w1).to(x.dtype)


def softplus(x):  # fp32; avoids relying on a lowered softplus
    return torch.where(x > 20.0, x, torch.log(1.0 + torch.exp(torch.clamp(x, max=20.0))))


def l2n(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def tri_inv(kb, k, g, C):
    """Inverse of the unit lower-triangular Mtx = I + L, L_ij = beta_i k_i.k_j exp(g_i-g_j) (i>j), per chunk, by recursive 2x2 block
    inversion [[A,0],[B,D]]^-1 = [[A^-1,0],[-D^-1 B A^-1, D^-1]] -- exactly block forward substitution (stable), matmuls only.
    The off-diagonal blocks B are built directly from key sub-chunks (rows strictly after cols, so no mask and exp(.)<=1).
    kb,k [..., C, D]; g [..., C] (in-chunk cumsum) -> [..., C, C]."""
    lead = k.shape[:-2]; D = k.shape[-1]
    Dinv = torch.ones(*lead, C, 1, 1, device=k.device, dtype=k.dtype)
    s = 1
    while s < C:
        nb = C // (2 * s)
        kr = kb.reshape(*lead, nb, 2, s, D)[..., 1, :, :]; kc = k.reshape(*lead, nb, 2, s, D)[..., 0, :, :]
        gg = g.reshape(*lead, nb, 2, s)
        B = (kr @ kc.transpose(-1, -2)) * torch.exp(gg[..., 1, :].unsqueeze(-1) - gg[..., 0, :].unsqueeze(-2))
        Dv = Dinv.reshape(*lead, nb, 2, s, s)
        A_, D_ = Dv[..., 0, :, :], Dv[..., 1, :, :]
        X = -(D_ @ B) @ A_
        top = torch.cat([A_, torch.zeros_like(A_)], -1)
        bot = torch.cat([X, D_], -1)
        Dinv = torch.cat([top, bot], -2)                          # [..., nb, 2s, 2s]
        s *= 2
    return Dinv.reshape(*lead, C, C)


_BLK = {}


def _blkmask(C, s, dev):
    key = (C, s, str(dev))
    if key not in _BLK:
        ar = torch.arange(C, device=dev)
        m = ((ar[:, None] // (2 * s)) == (ar[None, :] // (2 * s))) & ((ar[:, None] // s) % 2 == 1) & ((ar[None, :] // s) % 2 == 0)
        _BLK[key] = m.float()
    return _BLK[key]


def tri_inv_bd(L, C):
    """(I + L)^-1 for strictly lower L [..., C, C]. Same recursive 2x2 block inversion as tri_inv, but each level is two full CxC
    matmuls with the block-diagonal running inverse: X = -Dinv (L o mask_s) Dinv lands exactly in the lower-left s-blocks of every
    2s super-block, so Dinv <- Dinv + X. Exact block forward substitution; big tiles instead of thousands of tiny matmuls."""
    Dinv = torch.eye(C, device=L.device, dtype=L.dtype).expand(L.shape).contiguous()
    s = 1
    while s < C:
        Lb = L * _blkmask(C, s, L.device)
        X = -Lb if s == 1 else -((Dinv @ Lb) @ Dinv)
        Dinv = Dinv + X
        s *= 2
    return Dinv


TRI = os.environ.get('HOB_TRI', 'bd')
SCAN = os.environ.get('HOB_SCAN', '0') == '1'
NKI = os.environ.get('HOB_NKI', '0') == '1'
_CST = {}


def gdn_nki_call(q, k, v, g, beta, S0=None):
    """q,k,v [B,H,T,128] fp32, g,beta [B,H,T] -> o [B,H,T,128], S [B,H,128,128] via the NKI kernel (T multiple of 128)."""
    import gdn_nki as G
    B, H, T, Dd = q.shape
    if os.environ.get('HOB_FAKEGDN') == '1':   # decomposition only: skip the GDN core (wrong function)
        return v * 1.0, torch.zeros(B, H, Dd, Dd, device=q.device, dtype=q.dtype)
    key = str(q.device)
    if key not in _CST:
        _CST[key] = torch.from_numpy(G.consts_np()).to(q.device)
    S0 = torch.zeros(B * H, Dd, Dd, device=q.device, dtype=q.dtype) if S0 is None else S0.reshape(B * H, Dd, Dd)
    KER = {'2': G.gdn_kernel2, '3': G.gdn_kernel3}.get(os.environ.get('HOB_KER', '1'), G.gdn_kernel)
    o, S = KER[B * H](q.reshape(B * H, T, Dd), k.reshape(B * H, T, Dd), v.reshape(B * H, T, Dd),
                               g.reshape(B * H, T, 1), beta.reshape(B * H, T, 1), _CST[key], S0)
    return o.reshape(B, H, T, Dd), S.reshape(B, H, Dd, Dd)
MARK = os.environ.get('HOB_MARK', '0') == '1'


def mark(x, start):
    if not MARK:
        return x
    from neuronx_distributed_inference.models.layer_boundary_marker import ModuleMarkerStartWrapper, ModuleMarkerEndWrapper
    return ModuleMarkerStartWrapper()(x) if start else ModuleMarkerEndWrapper()(x)


def gdn_chunk(q, k, v, g, beta, C=64, S0=None):
    """Gated delta rule, chunked. q,k,v [B,H,T,D] fp32 (q,k l2-normalised; q pre-scaled); g (log decay), beta [B,H,T].
    T must be a multiple of C (callers pad with k=0, beta=0, g=0 so the state is unchanged by pads).
    Returns o [B,H,T,D], final state S [B,H,D,D]."""
    Bt, H, T, D = q.shape
    n = T // C
    dev = q.device
    q = q.reshape(Bt, H, n, C, D); k = k.reshape(Bt, H, n, C, D); v = v.reshape(Bt, H, n, C, D)
    ar = torch.arange(C, device=dev)
    U = (ar[:, None] <= ar[None, :]).float()                      # U[j,i] = 1 if j <= i  -> cumsum as matmul
    g = (g.reshape(Bt, H, n, 1, C) @ U).reshape(Bt, H, n, C)      # in-chunk cumulative log decay
    b = beta.reshape(Bt, H, n, C, 1)
    kb = k * b; vb = v * b
    incl = (ar[:, None] >= ar[None, :]).float()                   # i >= j
    dg = g.unsqueeze(-1) - g.unsqueeze(-2)                        # g_i - g_j
    decay = torch.exp(dg * incl) * incl                           # exp(g_i-g_j) on/below diagonal, 0 above (no inf*0)
    if TRI == 'bd':
        strict = (ar[:, None] > ar[None, :]).float()
        P = tri_inv_bd((kb @ k.transpose(-1, -2)) * decay * strict, C)
    else:
        P = tri_inv(kb, k, g, C)                                  # (I + strict_lower(beta k k^T decay))^-1, block-recursive
    eg = torch.exp(g)
    u = P @ vb                                                    # [Bt,H,n,C,D]
    w = P @ (kb * eg.unsqueeze(-1))
    intra = (q @ k.transpose(-1, -2)) * decay                     # causal incl. diagonal
    qg = q * eg.unsqueeze(-1)
    glast = g[..., -1:]                                           # [Bt,H,n,1]
    kdec = k * torch.exp(glast - g).unsqueeze(-1)
    elast = torch.exp(glast).unsqueeze(-1)                        # [Bt,H,n,1,1]
    if SCAN:
        # chunk transition S_{i+1} = A_i S_i + B_i, A_i = e_i I - kdec_i^T w_i, B_i = kdec_i^T u_i; Hillis-Steele scan over chunks
        kt = kdec.transpose(-1, -2)
        A = elast * torch.eye(D, device=dev, dtype=q.dtype) - kt @ w
        Bm = kt @ u
        d = 1
        while d < n:
            Bm = torch.cat([Bm[:, :, :d], A[:, :, d:] @ Bm[:, :, :-d] + Bm[:, :, d:]], 2)
            A = torch.cat([A[:, :, :d], A[:, :, d:] @ A[:, :, :-d]], 2)
            d *= 2
        if S0 is None:
            Sb = torch.cat([torch.zeros_like(Bm[:, :, :1]), Bm[:, :, :-1]], 2)
            S = Bm[:, :, -1]
        else:
            S0e = S0.unsqueeze(2)
            Sb = torch.cat([S0e, A[:, :, :-1] @ S0e + Bm[:, :, :-1]], 2)
            S = A[:, :, -1] @ S0 + Bm[:, :, -1]
        vn = u - w @ Sb
        o = qg @ Sb + intra @ vn
        return o.reshape(Bt, H, T, D), S
    S = torch.zeros(Bt, H, D, D, device=dev, dtype=q.dtype) if S0 is None else S0
    outs = []
    for i in range(n):
        vn = u[:, :, i] - w[:, :, i] @ S
        outs.append(qg[:, :, i] @ S + intra[:, :, i] @ vn)
        S = S * elast[:, :, i] + kdec[:, :, i].transpose(-1, -2) @ vn
    o = torch.stack(outs, 2).reshape(Bt, H, T, D)
    return o, S


class Hob(nn.Module):
    """hobson torso + final norm. forward(ids [T], sel [S]) -> final hidden rows [S, 2048] fp32.
    forward_packed(s_ids [Ls], b_ids [M, Lq], sel [M, S]) -> [M, S, 2048]: state once, M question branches."""

    def __init__(self, W, dtype=torch.bfloat16, C=64, attn='sdpa', gdn_dtype=torch.float32):
        super().__init__()
        self.C, self.attn, self.dt, self.gdt = C, attn, dtype, gdn_dtype
        P = lambda t, d=dtype: nn.Parameter(t.to(d).contiguous(), requires_grad=False)
        self.embed = P(W['embed_tokens.weight'])
        self.norm1 = P(1.0 + W['norm.weight'].float(), torch.float32)
        self.types = []
        self.L = nn.ModuleList()
        for i in range(24):
            p = f'layers.{i}.'
            m = nn.Module()
            m.in1 = P(1.0 + W[p + 'input_layernorm.weight'].float(), torch.float32)
            m.post1 = P(1.0 + W[p + 'post_attention_layernorm.weight'].float(), torch.float32)
            m.Wgu = P(torch.cat([W[p + 'mlp.gate_proj.weight'], W[p + 'mlp.up_proj.weight']], 0))
            m.Wd = P(W[p + 'mlp.down_proj.weight'])
            if p + 'linear_attn.in_proj_qkv.weight' in W:
                a = p + 'linear_attn.'
                self.types.append('gdn')
                m.Win = P(torch.cat([W[a + 'in_proj_qkv.weight'], W[a + 'in_proj_z.weight'], W[a + 'in_proj_b.weight'], W[a + 'in_proj_a.weight']], 0))
                m.conv = P(W[a + 'conv1d.weight'].squeeze(1).float(), torch.float32)     # [6144, 4]
                m.negA = P(-W[a + 'A_log'].float().exp(), torch.float32)
                m.dtb = P(W[a + 'dt_bias'].float(), torch.float32)
                m.gnw = P(W[a + 'norm.weight'])
                m.Wo = P(W[a + 'out_proj.weight'])
            else:
                a = p + 'self_attn.'
                self.types.append('attn')
                m.Win = P(torch.cat([W[a + 'q_proj.weight'], W[a + 'k_proj.weight'], W[a + 'v_proj.weight']], 0))
                m.qn1 = P(1.0 + W[a + 'q_norm.weight'].float(), torch.float32)
                m.kn1 = P(1.0 + W[a + 'k_norm.weight'].float(), torch.float32)
                m.Wo = P(W[a + 'o_proj.weight'])
            self.L.append(m)
        inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        self.inv = P(inv, torch.float32)

    # ------------------------------------------------------------------ pieces
    def rope_tab(self, pos):
        fr = pos.float()[:, None] * self.inv[None, :]
        fr = torch.cat([fr, fr], -1)
        return fr.cos().to(self.dt), fr.sin().to(self.dt)

    @staticmethod
    def rope(x, cos, sin):  # x [T, h, 256]; partial rotary on the first 64 dims (rotate_half)
        xr, xp = x[..., :64], x[..., 64:]
        x1, x2 = xr[..., :32], xr[..., 32:]
        c, s = cos[:, None, :], sin[:, None, :]
        rot = xr * c + torch.cat([-x2, x1], -1) * s
        return torch.cat([rot, xp], -1)

    def conv(self, x, w, tail=None):  # x [B, T, 6144] (bf16) ; causal depthwise k=4 + SiLU, fp32 accumulate
        xf = x.float()
        B, T, Cc = xf.shape
        pre = torch.zeros(B, 3, Cc, dtype=xf.dtype, device=xf.device) if tail is None else tail.float()
        xp = torch.cat([pre, xf], 1)
        y = xp[:, 0:T] * w[:, 0] + xp[:, 1:T + 1] * w[:, 1] + xp[:, 2:T + 2] * w[:, 2] + xp[:, 3:T + 3] * w[:, 3]
        return F.silu(y).to(self.dt)

    def gdn_prep(self, m, proj, valid=None):
        """proj rows [R, 8224] -> z [R,2048], beta [R,16], g [R,16] (fp32). valid: [R] float mask (pads -> beta=0, g=0)."""
        z = proj[:, 6144:8192]
        beta = torch.sigmoid(proj[:, 8192:8208].float())
        g = m.negA * softplus(proj[:, 8208:8224].float() + m.dtb)
        if valid is not None:
            beta = beta * valid[:, None]; g = g * valid[:, None]
        return z, beta, g

    def qkv_heads(self, c, B, T):  # conv output [B,T,6144] -> q,k,v [B,16,T,128] in gdn dtype, q,k l2normed, q scaled
        c = c.to(self.gdt).reshape(B, T, 3, 16, 128).permute(2, 0, 3, 1, 4)
        return l2n(c[0]) * (128 ** -0.5), l2n(c[1]), c[2]

    def pad_T(self, T):
        return ((T + self.C - 1) // self.C) * self.C

    def chunk(self, q, k, v, g, beta, S0=None):  # q,k,v [B,H,T,D]; g,beta [B,H,T]; pads to C internally
        T = q.shape[2]; Tp = self.pad_T(T); pd = Tp - T
        if NKI and pd == 0 and self.C == 128:
            return gdn_nki_call(q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta.contiguous(), S0)
        if pd:
            q, k, v = (F.pad(x, (0, 0, 0, pd)) for x in (q, k, v))
            g, beta = (F.pad(x, (0, pd)) for x in (g, beta))
        o, S = gdn_chunk(q, k, v, g, beta, self.C, S0)
        return o[:, :, :T], S

    def gated_norm(self, o, z, w):  # o [R,16,128] , z [R,2048]
        of = o.float()
        of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + EPS)
        y = w * of.to(self.dt)
        y = y.float() * F.silu(z.float().reshape(-1, 16, 128))
        return y.to(self.dt).reshape(-1, 2048)

    def attn_prep(self, m, proj, cos, sin):
        R = proj.shape[0]
        qg = proj[:, :4096].reshape(R, 8, 512)
        q, gate = qg[..., :256], qg[..., 256:].reshape(R, 2048)
        k = proj[:, 4096:4608].reshape(R, 2, 256); v = proj[:, 4608:5120].reshape(R, 2, 256)
        q = self.rope(rms_zc(q, m.qn1), cos, sin); k = self.rope(rms_zc(k, m.kn1), cos, sin)
        return q, k, v, torch.sigmoid(gate.float()).to(self.dt)

    def sdpa(self, q, k, v, mask=None, causal=False):  # q [B,8,Tq,256], k/v [B,2,Tk,256]
        if self.attn == 'sdpa':
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None, enable_gqa=True)
        k = k.repeat_interleave(4, dim=1); v = v.repeat_interleave(4, dim=1)
        s = (q @ k.transpose(-1, -2)).float() * (1.0 / 16.0)
        if causal and mask is None:
            Tq, Tk = q.shape[2], k.shape[2]
            mask = torch.arange(Tk, device=q.device)[None, :] <= torch.arange(Tq, device=q.device)[:, None]
        if mask is not None:
            s = s.masked_fill(~mask, -30000.0)
        p = torch.softmax(s, -1).to(v.dtype)
        return p @ v

    def mlp(self, m, x):
        h2 = rms_zc(x, m.post1)
        gu = h2 @ m.Wgu.t()
        I = gu.shape[-1] // 2
        return (F.silu(gu[:, :I]) * gu[:, I:]) @ m.Wd.t()

    # ------------------------------------------------------------------ single sequence
    def forward(self, ids, sel):
        T = ids.shape[0]
        x = F.embedding(ids, self.embed)
        cos, sin = self.rope_tab(torch.arange(T, device=ids.device))
        for i, m in enumerate(self.L):
            x = mark(x, True)
            h = rms_zc(x, m.in1)
            proj = h @ m.Win.t()
            if self.types[i] == 'gdn':
                z, beta, g = self.gdn_prep(m, proj)
                c = self.conv(proj[None, :, :6144], m.conv)
                q, k, v = self.qkv_heads(c, 1, T)
                o, _ = self.chunk(q, k, v, g.t()[None], beta.t()[None])
                o = self.gated_norm(o[0].transpose(0, 1), z, m.gnw)
            else:
                q, k, v, gate = self.attn_prep(m, proj, cos, sin)
                o = self.sdpa(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], causal=True)
                o = o[0].transpose(0, 1).reshape(T, 2048) * gate
            x = x + o @ m.Wo.t()
            x = x + self.mlp(m, x)
            x = mark(x, False)
        return rms_zc(x[sel], self.norm1).float()

    # ------------------------------------------------------------------ state once + M question branches
    def forward_packed(self, s_ids, b_ids, sel, n_s=None):
        """s_ids [Ls] (all valid unless n_s given), b_ids [M, Lq] (right-padded; pads are causal-harmless), sel [M, S] branch-relative rows."""
        dev = s_ids.device
        Ls = s_ids.shape[0]; M, Lq = b_ids.shape
        ns = Ls if n_s is None else n_s
        ids = torch.cat([s_ids, b_ids.reshape(-1)])
        x = F.embedding(ids, self.embed)
        ar_s = torch.arange(Ls, device=dev); ar_q = torch.arange(Lq, device=dev)
        pos = torch.cat([ar_s, (ns + ar_q).repeat(M)])
        cos, sin = self.rope_tab(pos)
        vs = (ar_s < ns).float()
        j = torch.arange(Ls + Lq, device=dev)[None, :]
        bmask = torch.where(j < Ls, j < ns, (j - Ls) <= ar_q[:, None])          # [Lq, Ls+Lq]
        smask = (ar_s[None, :] <= ar_s[:, None])
        R = Ls + M * Lq
        for li, m in enumerate(self.L):
            x = mark(x, True)
            h = rms_zc(x, m.in1)
            proj = h @ m.Win.t()
            if self.types[li] == 'gdn':
                z, beta, g = self.gdn_prep(m, proj)
                beta_s = beta[:Ls] * vs[:, None]; g_s = g[:Ls] * vs[:, None]
                qkv = proj[:, :6144]
                tail = qkv[ns - 3:ns] if isinstance(ns, int) else qkv[:Ls][(ns - 3 + torch.arange(3, device=dev))]
                cs = self.conv(qkv[None, :Ls], m.conv)
                cb = self.conv(qkv[Ls:].reshape(M, Lq, 6144), m.conv, tail=tail[None].expand(M, 3, 6144))
                qs, ks, vv = self.qkv_heads(cs, 1, Ls)
                ks = ks * vs[None, None, :, None]
                o_s, S = self.chunk(qs, ks, vv, g_s.t()[None], beta_s.t()[None])
                qb, kb, vb = self.qkv_heads(cb, M, Lq)
                o_b, _ = self.chunk(qb, kb, vb, g[Ls:].reshape(M, Lq, 16).transpose(1, 2), beta[Ls:].reshape(M, Lq, 16).transpose(1, 2),
                                    S0=S.expand(M, -1, -1, -1))
                o = torch.cat([o_s[0].transpose(0, 1).reshape(Ls, 16, 128), o_b.transpose(1, 2).reshape(M * Lq, 16, 128)], 0)
                o = self.gated_norm(o, z, m.gnw)
            else:
                q, k, v, gate = self.attn_prep(m, proj, cos, sin)
                qs = q[:Ls].transpose(0, 1)[None]; ks = k[:Ls].transpose(0, 1)[None]; vv = v[:Ls].transpose(0, 1)[None]
                o_s = self.sdpa(qs, ks, vv, mask=smask if n_s is not None else None, causal=True)
                qb = q[Ls:].reshape(M, Lq, 8, 256).transpose(1, 2)
                kbr = k[Ls:].reshape(M, Lq, 2, 256).transpose(1, 2); vbr = v[Ls:].reshape(M, Lq, 2, 256).transpose(1, 2)
                kf = torch.cat([ks.expand(M, -1, -1, -1), kbr], 2); vf = torch.cat([vv.expand(M, -1, -1, -1), vbr], 2)
                o_b = self.sdpa(qb, kf, vf, mask=bmask)
                o = torch.cat([o_s[0].transpose(0, 1).reshape(Ls, 2048), o_b.transpose(1, 2).reshape(M * Lq, 2048)], 0) * gate
            x = x + o @ m.Wo.t()
            x = x + self.mlp(m, x)
            x = mark(x, False)
        xb = x[Ls:].reshape(M, Lq, 2048)
        rows = torch.gather(xb, 1, sel[:, :, None].expand(-1, -1, 2048))
        return rms_zc(rows, self.norm1).float()


class Head:
    """pointer head (fp32, host side): LN -> q(decide), k(options); logits/sqrt(256)/T_kind; softmax over options."""

    def __init__(self, hs, cfg):
        self.hs = {k: v.float() for k, v in hs.items()}
        self.tb = cfg.get('temperature_by_kind') or {}
        self.t0 = cfg['temperature']

    def probs(self, rows, kind):  # rows [K+1, 2048]: option rows then the <answer> row
        h = self.hs
        r = F.layer_norm(rows.float(), (rows.shape[-1],), h['norm.weight'], h['norm.bias'])
        dq = r[-1] @ h['q.weight'].t() + h['q.bias']
        ok = r[:-1] @ h['k.weight'].t() + h['k.bias']
        lg = (ok @ dq) * (256 ** -0.5) / float(self.tb.get(kind, self.t0))
        return torch.softmax(lg, -1)


class HobNL(Hob):
    """Neuron layout: head-major projections (no token->head transposes), pre-transposed [K, N] GEMM weights,
    NKI GDN kernel, per-head einsum output projections. Same function as Hob.forward (single sequence, T % 128 == 0)."""

    def __init__(self, W, dtype=torch.bfloat16):
        super().__init__(W, dtype=dtype, C=128, attn='explicit')
        P = lambda t: nn.Parameter(t.to(dtype).contiguous(), requires_grad=False)
        for i, m in enumerate(self.L):
            m.WguT = P(m.Wgu.t()); m.WdT = P(m.Wd.t())
            if self.types[i] == 'gdn':
                Win = m.Win
                m.Wqkvh = P(Win[:6144].reshape(48, 128, 2048).permute(0, 2, 1))       # [48, 2048, 128]
                m.Wzh = P(Win[6144:8192].reshape(16, 128, 2048).permute(0, 2, 1))    # [16, 2048, 128]
                m.WabT = P(Win[8192:8224].t())                                       # [2048, 32] (b | a)
                m.convh = nn.Parameter(m.conv.reshape(48, 128, 4).permute(0, 2, 1).contiguous(), requires_grad=False)  # [48, 4, 128]
                m.Woh = P(m.Wo.reshape(2048, 16, 128).permute(1, 2, 0))              # [16, 128, 2048]
            else:
                Win = m.Win
                qg = Win[:4096].reshape(8, 512, 2048)
                m.Wqh = P(qg[:, :256].permute(0, 2, 1)); m.Wgh = P(qg[:, 256:].permute(0, 2, 1))   # [8, 2048, 256]
                m.Wkh = P(Win[4096:4608].reshape(2, 256, 2048).permute(0, 2, 1))
                m.Wvh = P(Win[4608:5120].reshape(2, 256, 2048).permute(0, 2, 1))
                m.Woh = P(m.Wo.reshape(2048, 8, 256).permute(1, 2, 0))               # [8, 256, 2048]
            del m.Win, m.Wo, m.Wgu, m.Wd

    def mlpT(self, m, x):
        h2 = rms_zc(x, m.post1)
        gu = h2 @ m.WguT
        I = gu.shape[-1] // 2
        return (F.silu(gu[:, :I]) * gu[:, I:]) @ m.WdT

    def forward(self, ids, sel):
        T = ids.shape[0]
        x = F.embedding(ids, self.embed)
        cos, sin = self.rope_tab(torch.arange(T, device=ids.device))
        ar = torch.arange(T, device=ids.device)
        cmask = ar[None, :] <= ar[:, None]
        for i, m in enumerate(self.L):
            x = mark(x, True)
            h = rms_zc(x, m.in1)
            if self.types[i] == 'gdn':
                qkv = torch.matmul(h[None], m.Wqkvh)                     # [48, T, 128]
                z = torch.matmul(h[None], m.Wzh)                         # [16, T, 128]
                ab = (h @ m.WabT).float()                                # [T, 32]
                beta = torch.sigmoid(ab[:, :16]).t().contiguous()        # [16, T]
                g = (m.negA * softplus(ab[:, 16:] + m.dtb)).t().contiguous()
                xf = qkv.float()
                xp = torch.cat([torch.zeros(48, 3, 128, device=x.device), xf], 1)
                w = m.convh
                y = xp[:, 0:T] * w[:, 0:1] + xp[:, 1:T + 1] * w[:, 1:2] + xp[:, 2:T + 2] * w[:, 2:3] + xp[:, 3:T + 3] * w[:, 3:4]
                y = F.silu(y).to(self.dt).float()
                q = l2n(y[0:16]) * (128 ** -0.5); k = l2n(y[16:32]); v = y[32:48]
                o, _ = gdn_nki_call(q[None], k[None], v[None], g[None], beta[None])
                of = o[0]
                of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + EPS)
                yo = (m.gnw * of.to(self.dt)).float() * F.silu(z.float())
                x = x + torch.einsum('htd,hdo->to', yo.to(self.dt), m.Woh)
            else:
                q = torch.matmul(h[None], m.Wqh); gt = torch.matmul(h[None], m.Wgh)      # [8, T, 256]
                k = torch.matmul(h[None], m.Wkh); v = torch.matmul(h[None], m.Wvh)       # [2, T, 256]
                c2, s2 = cos[None], sin[None]
                q = self.rope(rms_zc(q, m.qn1).transpose(0, 1), cos, sin).transpose(0, 1)
                k = self.rope(rms_zc(k, m.kn1).transpose(0, 1), cos, sin).transpose(0, 1)
                kk = k.repeat_interleave(4, dim=0); vv = v.repeat_interleave(4, dim=0)
                s = (q @ kk.transpose(-1, -2)).float() * (1.0 / 16.0)
                s = s.masked_fill(~cmask, -30000.0)
                p = torch.softmax(s, -1).to(self.dt)
                o = (p @ vv) * torch.sigmoid(gt.float()).to(self.dt)                        # [8, T, 256]
                x = x + torch.einsum('htd,hdo->to', o, m.Woh)
            x = x + self.mlpT(m, x)
            x = mark(x, False)
        return rms_zc(x[sel], self.norm1).float()


def _nl_packed(self, s_ids, b_ids, sel, n_s=None):
    """HobNL: state once (Ls % 128 == 0, valid rows n_s) + M question branches (Lq % 128 == 0). Head-major throughout."""
    dev = s_ids.device
    Ls = s_ids.shape[0]; M, Lq = b_ids.shape
    ns = Ls if n_s is None else n_s
    ids = torch.cat([s_ids, b_ids.reshape(-1)])
    x = F.embedding(ids, self.embed)
    ar_s = torch.arange(Ls, device=dev); ar_q = torch.arange(Lq, device=dev)
    cos, sin = self.rope_tab(torch.cat([ar_s, (ns + ar_q).repeat(M)]))
    vs = (ar_s < ns).float()
    j = torch.arange(Ls + Lq, device=dev)[None, :]
    bmask = torch.where(j < Ls, j < ns, (j - Ls) <= ar_q[:, None])           # [Lq, Ls+Lq]
    smask = ar_s[None, :] <= ar_s[:, None]
    for li, m in enumerate(self.L):
        x = mark(x, True)
        h = rms_zc(x, m.in1)
        if self.types[li] == 'gdn':
            qkv = torch.matmul(h[None], m.Wqkvh)                                # [48, R, 128]
            z = torch.matmul(h[None], m.Wzh)                                    # [16, R, 128]
            ab = (h @ m.WabT).float()
            beta = torch.sigmoid(ab[:, :16]).t(); g = (m.negA * softplus(ab[:, 16:] + m.dtb)).t()   # [16, R]
            w = m.convh
            def conv(xp, Tn):
                y = xp[..., 0:Tn, :] * w[:, None, 0:1] + xp[..., 1:Tn + 1, :] * w[:, None, 1:2] + xp[..., 2:Tn + 2, :] * w[:, None, 2:3] + xp[..., 3:Tn + 3, :] * w[:, None, 3:4]
                return F.silu(y).to(self.dt).float()
            xs = qkv[:, :Ls].float()
            ys = conv(torch.cat([torch.zeros(48, 1, 3, 128, device=dev), xs[:, None]], 2), Ls)[:, 0]          # [48, Ls, 128]
            tail = qkv[:, ns - 3:ns].float()                                                                  # [48, 3, 128]
            xb = qkv[:, Ls:].float().reshape(48, M, Lq, 128)
            yb = conv(torch.cat([tail[:, None].expand(48, M, 3, 128), xb], 2), Lq)                           # [48, M, Lq, 128]
            qs = l2n(ys[0:16]) * (128 ** -0.5); ks = l2n(ys[16:32]) * vs[None, :, None]; vv = ys[32:48]
            o_s, S = gdn_nki_call(qs[None], ks[None], vv[None], (g[:, :Ls] * vs)[None], (beta[:, :Ls] * vs)[None])
            qb = (l2n(yb[0:16]) * (128 ** -0.5)).transpose(0, 1); kb = l2n(yb[16:32]).transpose(0, 1); vb = yb[32:48].transpose(0, 1)
            gb = g[:, Ls:].reshape(16, M, Lq).transpose(0, 1); bb = beta[:, Ls:].reshape(16, M, Lq).transpose(0, 1)
            o_b, _ = gdn_nki_call(qb.contiguous(), kb.contiguous(), vb.contiguous(), gb.contiguous(), bb.contiguous(),
                                  S0=S.expand(M, -1, -1, -1).contiguous())
            of = torch.cat([o_s[0], o_b.transpose(0, 1).reshape(16, M * Lq, 128)], 1)                        # [16, R, 128]
            of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + EPS)
            yo = (m.gnw * of.to(self.dt)).float() * F.silu(z.float())
            x = x + torch.einsum('htd,hdo->to', yo.to(self.dt), m.Woh)
        else:
            q = torch.matmul(h[None], m.Wqh); gt = torch.matmul(h[None], m.Wgh)
            k = torch.matmul(h[None], m.Wkh); v = torch.matmul(h[None], m.Wvh)
            q = self.rope(rms_zc(q, m.qn1).transpose(0, 1), cos, sin).transpose(0, 1)
            k = self.rope(rms_zc(k, m.kn1).transpose(0, 1), cos, sin).transpose(0, 1)
            ks_ = k[:, :Ls].repeat_interleave(4, dim=0); vs_ = v[:, :Ls].repeat_interleave(4, dim=0)
            s = (q[:, :Ls] @ ks_.transpose(-1, -2)).float() * (1.0 / 16.0)
            p = torch.softmax(s.masked_fill(~smask, -30000.0), -1).to(self.dt)
            o_s = p @ vs_                                                                                     # [8, Ls, 256]
            qb = q[:, Ls:].reshape(8, M, Lq, 256)
            kb = torch.cat([k[:, None, :Ls].expand(2, M, Ls, 256), k[:, Ls:].reshape(2, M, Lq, 256)], 2).repeat_interleave(4, dim=0)
            vb = torch.cat([v[:, None, :Ls].expand(2, M, Ls, 256), v[:, Ls:].reshape(2, M, Lq, 256)], 2).repeat_interleave(4, dim=0)
            sb = (qb @ kb.transpose(-1, -2)).float() * (1.0 / 16.0)
            pb = torch.softmax(sb.masked_fill(~bmask, -30000.0), -1).to(self.dt)
            o_b = (pb @ vb).reshape(8, M * Lq, 256)
            o = torch.cat([o_s, o_b], 1) * torch.sigmoid(gt.float()).to(self.dt)
            x = x + torch.einsum('htd,hdo->to', o, m.Woh)
        x = x + self.mlpT(m, x)
        x = mark(x, False)
    xb = x[Ls:].reshape(M, Lq, 2048)
    rows = torch.gather(xb, 1, sel[:, :, None].expand(-1, -1, 2048))
    return rms_zc(rows, self.norm1).float()


HobNL.forward_packed = _nl_packed
