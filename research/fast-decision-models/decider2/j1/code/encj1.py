"""J1: T5Gemma-2B (Gemma-2 2B UL2) bidirectional encoder torso for a decider, written out in plain torch.

- weights read straight from the safetensors of the encoder-only extraction (no HF modelling code at run time);
- LoRA (r, alpha) on q, k, v, o, gate, up, down of every layer (separate adapters, B = 0 at init), fp32 masters, bf16 compute;
- attention modes (per call):
    'masked'   (e1b): state rows attend to state rows only (bidirectional); question rows attend to state + own question (bidirectional)
    'full'     (e1a): every row attends to every valid row
  plus an optional local window for state->state attention on a chosen set of layers ('local' layers); question rows stay global.
- softcap: Gemma-2's tanh attention-logit cap (50). softcap=None drops it (needed for SDPA/flash kernels); measured below.
forward(ids [B,T], am [B,T], qs [B] question start) -> final-normed hidden [B,T,2304]
"""
import os, json, glob, math
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

D, NH, NKV, HD, FF = 2304, 8, 4, 256, 9216


def find_ckpt():
    c = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Minthy--t5gemma-2b-2b-ul2-encoder-only/snapshots/*'))
    return c[0]


def load_weights(path=None, dev='cuda', n_layers=26):
    from safetensors import safe_open
    path = path or find_ckpt()
    W = {}
    for f in sorted(glob.glob(os.path.join(path, '*.safetensors'))):
        with safe_open(f, framework='pt', device=dev) as fh:
            for k in fh.keys():
                kk = k.replace('encoder.', '', 1)
                if kk.startswith('layers.') and int(kk.split('.')[1]) >= n_layers: continue
                W[kk] = fh.get_tensor(k)
    cfg = json.load(open(os.path.join(path, 'config.json')))
    return W, cfg.get('encoder', cfg)


def rms(x, w1, eps=1e-6):  # Gemma RMSNorm: (x * rsqrt(mean x^2 + eps)) * (1 + w), fp32 internals, cast back
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w1).to(x.dtype)


def rope_cs(pos, dev, theta=10000.0):  # pos [B,T] -> cos,sin [B,T,256] bf16 (HF rotate_half convention)
    inv = 1.0 / (theta ** (torch.arange(0, HD, 2, device=dev, dtype=torch.float32) / HD))
    fr = pos.float()[..., None] * inv
    emb = torch.cat([fr, fr], -1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def rot(x, cos, sin):  # x [B,H,T,256]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * cos[:, None] + torch.cat([-x2, x1], -1) * sin[:, None]


class Lora(nn.Module):
    def __init__(self, din, douts, r, alpha):
        super().__init__()
        self.r, self.s = r, alpha / r
        self.A = nn.ParameterList([nn.Parameter(torch.empty(r, din).uniform_(-1 / math.sqrt(din), 1 / math.sqrt(din))) for _ in douts])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(o, r)) for o in douts])

    def forward(self, x):  # -> concatenated delta [.., sum(douts)] (block-diagonal B: one GEMM)
        A = torch.cat([a for a in self.A], 0).to(x.dtype)
        Bbd = (torch.block_diag(*[b for b in self.B]) if len(self.B) > 1 else self.B[0]).to(x.dtype)
        return ((x @ A.t()) * self.s) @ Bbd.t()


class EncTorso(nn.Module):
    def __init__(self, W, r=16, alpha=32, softcap=None, n_layers=26, local_layers=(), window=0, lora=True):
        super().__init__()
        self.embed = W['embed_tokens.weight']                       # [256000, 2304] bf16, frozen
        self.norm1 = (1.0 + W['norm.weight'].float())
        self.L = []
        for i in range(n_layers):
            p = f'layers.{i}.'
            g = lambda n: W[p + n]
            self.L.append(dict(
                Wqkv=torch.cat([g('self_attn.q_proj.weight'), g('self_attn.k_proj.weight'), g('self_attn.v_proj.weight')], 0).contiguous(),
                Wo=g('self_attn.o_proj.weight').contiguous(),
                Wgu=torch.cat([g('mlp.gate_proj.weight'), g('mlp.up_proj.weight')], 0).contiguous(),
                Wd=g('mlp.down_proj.weight').contiguous(),
                n_pa=1.0 + g('pre_self_attn_layernorm.weight').float(), n_poa=1.0 + g('post_self_attn_layernorm.weight').float(),
                n_pf=1.0 + g('pre_feedforward_layernorm.weight').float(), n_pof=1.0 + g('post_feedforward_layernorm.weight').float()))
        self.n_layers = n_layers
        self.softcap = softcap
        self.local = set(local_layers); self.window = window
        self.use_lora = lora
        if lora:
            self.lora = nn.ModuleList()
            for i in range(n_layers):
                self.lora.append(nn.ModuleDict(dict(
                    qkv=Lora(D, [NH * HD, NKV * HD, NKV * HD], r, alpha), o=Lora(NH * HD, [D], r, alpha),
                    gu=Lora(D, [FF, FF], r, alpha), d=Lora(FF, [D], r, alpha))))
        self.ckpt = False
        self.mode = 'masked'

    def merged(self):
        """copy of the per-layer weights with the LoRA folded in (bf16), for inference runtimes"""
        out = []
        for i, d in enumerate(self.L):
            e = dict(d)
            if self.use_lora:
                lo = self.lora[i]
                for key, mod in (('Wqkv', lo['qkv']), ('Wo', lo['o']), ('Wgu', lo['gu']), ('Wd', lo['d'])):
                    dl = torch.cat([b.float() @ a.float() for a, b in zip(mod.A, mod.B)], 0) * mod.s
                    e[key] = (d[key].float() + dl).to(torch.bfloat16).contiguous()
            out.append(e)
        return out

    # ------------------------------------------------------------ masks
    def build_mask(self, am, qs, local):
        """bool [B,1,T,T]: True = attend"""
        B, T = am.shape
        dev = am.device
        i = torch.arange(T, device=dev)[None, :, None]; j = torch.arange(T, device=dev)[None, None, :]
        valid = am.bool()[:, None, :]
        if self.mode == 'full':
            m = valid.expand(B, T, T)
            if local:
                m = m & ((i - j).abs() < self.window)
        else:
            q0 = qs[:, None, None]
            isq_i = i >= q0
            m = valid & ((j < q0) | isq_i)
            if local:   # state rows: window over state; question rows: global
                m = m & (isq_i | ((i - j).abs() < self.window))
        return m[:, None]

    # ------------------------------------------------------------ layer
    def _attn(self, q, k, v, mask):
        if self.softcap is None:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=HD ** -0.5, enable_gqa=True)
        k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
        s = (q @ k.transpose(-1, -2)).float() * HD ** -0.5
        s = torch.tanh(s / self.softcap) * self.softcap
        s = s.masked_fill(~mask, float('-inf'))
        return torch.softmax(s, -1).to(q.dtype) @ v

    def layer(self, i, x, mask, cos, sin):
        d = self.L[i]; lo = self.lora[i] if self.use_lora else None
        B, T, _ = x.shape
        h = rms(x, d['n_pa'])
        qkv = h @ d['Wqkv'].t()
        if lo is not None: qkv = qkv + lo['qkv'](h)
        q = qkv[..., :NH * HD].reshape(B, T, NH, HD).transpose(1, 2)
        k = qkv[..., NH * HD:NH * HD + NKV * HD].reshape(B, T, NKV, HD).transpose(1, 2)
        v = qkv[..., NH * HD + NKV * HD:].reshape(B, T, NKV, HD).transpose(1, 2)
        q = rot(q, cos, sin); k = rot(k, cos, sin)
        o = self._attn(q, k, v, mask).transpose(1, 2).reshape(B, T, NH * HD)
        a = o @ d['Wo'].t()
        if lo is not None: a = a + lo['o'](o)
        x = x + rms(a, d['n_poa'])
        h = rms(x, d['n_pf'])
        gu = h @ d['Wgu'].t()
        if lo is not None: gu = gu + lo['gu'](h)
        m = F.gelu(gu[..., :FF], approximate='tanh') * gu[..., FF:]
        dm = m @ d['Wd'].t()
        if lo is not None: dm = dm + lo['d'](m)
        return x + rms(dm, d['n_pof'])

    def forward(self, ids, am, qs):
        B, T = ids.shape
        x = F.embedding(ids, self.embed) * torch.tensor(D ** 0.5, dtype=torch.bfloat16)
        pos = torch.arange(T, device=ids.device)[None].expand(B, T)
        cos, sin = rope_cs(pos, ids.device)
        mg = self.build_mask(am, qs, False)
        ml = self.build_mask(am, qs, True) if (self.local and self.window) else mg
        for i in range(self.n_layers):
            mask = ml if i in self.local else mg
            if self.ckpt and torch.is_grad_enabled():
                x = checkpoint(self.layer, i, x, mask, cos, sin, use_reentrant=False)
            else:
                x = self.layer(i, x, mask, cos, sin)
        return rms(x, self.norm1)

    # ------------------------------------------------------------ masked state cache (deployment path, reference implementation)
    @torch.no_grad()
    def forward_cached(self, s_ids, q_list, Ls_valid=None):
        """state [Ls] once (self-attention over state only), then M questions in one batch, each attending to the cached state K/V + itself.
        -> (state hidden [Ls,d], list of question hidden [Lq_m, d])"""
        dev = s_ids.device
        Ls = s_ids.shape[0]; M = len(q_list); Lq = max(len(q) for q in q_list)
        qb = torch.zeros(M, Lq, dtype=torch.long, device=dev); qv = torch.zeros(M, Lq, dtype=torch.bool, device=dev)
        for m, q in enumerate(q_list):
            qb[m, :len(q)] = torch.as_tensor(q, device=dev); qv[m, :len(q)] = True
        xs = F.embedding(s_ids[None], self.embed) * torch.tensor(D ** 0.5, dtype=torch.bfloat16)
        xq = F.embedding(qb, self.embed) * torch.tensor(D ** 0.5, dtype=torch.bfloat16)
        cs, ss = rope_cs(torch.arange(Ls, device=dev)[None], dev)
        cq, sq = rope_cs((Ls + torch.arange(Lq, device=dev))[None].expand(M, Lq), dev)
        ii = torch.arange(Ls, device=dev)
        smask_l = ((ii[:, None] - ii[None, :]).abs() < self.window)[None, None] if (self.local and self.window) else None
        qmask = torch.cat([torch.ones(M, Ls, dtype=torch.bool, device=dev), qv], 1)[:, None, None, :]   # [M,1,1,Ls+Lq]
        for i in range(self.n_layers):
            d = self.L[i]; lo = self.lora[i] if self.use_lora else None
            outs = []
            # state rows
            ks_vs = None
            for which, x, c, s in (('s', xs, cs, ss), ('q', xq, cq, sq)):
                B, T, _ = x.shape
                h = rms(x, d['n_pa']); qkv = h @ d['Wqkv'].t()
                if lo is not None: qkv = qkv + lo['qkv'](h)
                q = rot(qkv[..., :2048].reshape(B, T, NH, HD).transpose(1, 2), c, s)
                k = rot(qkv[..., 2048:3072].reshape(B, T, NKV, HD).transpose(1, 2), c, s)
                v = qkv[..., 3072:].reshape(B, T, NKV, HD).transpose(1, 2)
                if which == 's':
                    ks_vs = (k, v)
                    mk = smask_l if (i in self.local and smask_l is not None) else torch.ones(1, 1, T, T, dtype=torch.bool, device=dev)
                    o = self._attn(q, k, v, mk)
                else:
                    kk = torch.cat([ks_vs[0].expand(M, -1, -1, -1), k], 2); vv = torch.cat([ks_vs[1].expand(M, -1, -1, -1), v], 2)
                    o = self._attn(q, kk, vv, qmask.expand(M, 1, T, Ls + T))
                o = o.transpose(1, 2).reshape(B, T, NH * HD)
                a = o @ d['Wo'].t()
                if lo is not None: a = a + lo['o'](o)
                x = x + rms(a, d['n_poa'])
                h = rms(x, d['n_pf']); gu = h @ d['Wgu'].t()
                if lo is not None: gu = gu + lo['gu'](h)
                m_ = F.gelu(gu[..., :FF], approximate='tanh') * gu[..., FF:]
                dm = m_ @ d['Wd'].t()
                if lo is not None: dm = dm + lo['d'](m_)
                outs.append(x + rms(dm, d['n_pof']))
            xs, xq = outs
        hs = rms(xs, self.norm1)[0]; hq = rms(xq, self.norm1)
        return hs, [hq[m, :len(q)] for m, q in enumerate(q_list)]


# ====================================================================== packed-stream path (training + eval): no padding, varlen flash
class _VarFlash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, wl, wr):
        out, lse, rng, unused, _ = torch.ops.aten._flash_attention_forward(q, k, v, cu_q, cu_k, max_q, max_k, 0.0, False, False, scale=HD ** -0.5,
                                                                          window_size_left=wl, window_size_right=wr)
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, rng, unused)
        ctx.mx = (max_q, max_k, wl, wr)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_q, cu_k, rng, unused = ctx.saved_tensors
        max_q, max_k, wl, wr = ctx.mx
        dq, dk, dv = torch.ops.aten._flash_attention_backward(dout.contiguous(), q, k, v, out, lse, cu_q, cu_k, max_q, max_k, 0.0, False, rng, unused,
                                                              scale=HD ** -0.5, window_size_left=wl, window_size_right=wr)
        return dq, dk, dv, None, None, None, None, None, None


def varflash(q, k, v, cu_q, cu_k, max_q, max_k, window=0):
    wl = wr = (window - 1) if window else None
    return _VarFlash.apply(q, k, v, cu_q, cu_k, max_q, max_k, wl, wr)


def pack_meta(lens, nstates, dev):
    """rows laid out contiguously [s_1 q_1 s_2 q_2 ...]; returns positions + index sets + cu_seqlens for the two attention calls"""
    import itertools
    N = sum(lens)
    starts = list(itertools.accumulate([0] + list(lens)))
    pos = torch.cat([torch.arange(L) for L in lens])
    s_idx = torch.cat([torch.arange(a, a + ns) for a, ns in zip(starts, nstates)])
    q_idx = torch.cat([torch.arange(a + ns, a + L) for a, ns, L in zip(starts, nstates, lens)])
    qlens = [L - ns for L, ns in zip(lens, nstates)]
    cu = lambda xs: torch.tensor(list(itertools.accumulate([0] + list(xs))), dtype=torch.int32)
    m = dict(N=N, pos=pos, s_idx=s_idx, q_idx=q_idx, cu_s=cu(nstates), cu_q=cu(qlens), cu_row=cu(lens),
             max_s=max(nstates), max_q=max(qlens), max_row=max(lens), last=torch.tensor([a + L - 1 for a, L in zip(starts, lens)]),
             starts=torch.tensor(starts[:-1]))
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in m.items()}


def _rms_impl(x, w1, eps: float = 1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w1).to(x.dtype)


def _add_rms_impl(x, a, w1, eps: float = 1e-6):
    af = a.float()
    return x + (af * torch.rsqrt(af.pow(2).mean(-1, keepdim=True) + eps) * w1).to(x.dtype)


def _geglu_impl(gu):
    return F.gelu(gu[..., :FF], approximate='tanh') * gu[..., FF:]


def _rope_impl(x, cos, sin):  # x [N,H,256], cos/sin [N,256]
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return x * cos[:, None] + torch.cat([-x2, x1], -1) * sin[:, None]


K_RMS, K_ADDRMS, K_GEGLU, K_ROPE = _rms_impl, _add_rms_impl, _geglu_impl, _rope_impl


def use_compiled():
    global K_RMS, K_ADDRMS, K_GEGLU, K_ROPE
    K_RMS = torch.compile(_rms_impl, dynamic=True); K_ADDRMS = torch.compile(_add_rms_impl, dynamic=True)
    K_GEGLU = torch.compile(_geglu_impl, dynamic=True); K_ROPE = torch.compile(_rope_impl, dynamic=True)


def _layer_packed(self, i, x, cos, sin, m):
    d = self.L[i]; lo = self.lora[i] if self.use_lora else None
    N = x.shape[0]
    h = K_RMS(x, d['n_pa'])
    qkv = torch.addmm(lo['qkv'](h), h, d['Wqkv'].t()) if lo is not None else h @ d['Wqkv'].t()
    q = K_ROPE(qkv[:, :NH * HD].reshape(N, NH, HD), cos, sin)
    k = K_ROPE(qkv[:, NH * HD:NH * HD + NKV * HD].reshape(N, NKV, HD), cos, sin)
    v = qkv[:, NH * HD + NKV * HD:].reshape(N, NKV, HD)
    win = self.window if (i in self.local and self.window) else 0
    if self.mode == 'full':
        o = varflash(q, k, v, m['cu_row'], m['cu_row'], m['max_row'], m['max_row'], win)
    else:
        si, qi = m['s_idx'], m['q_idx']
        o1 = varflash(q[si], k[si], v[si], m['cu_s'], m['cu_s'], m['max_s'], m['max_s'], win)
        o2 = varflash(q[qi], k, v, m['cu_q'], m['cu_row'], m['max_q'], m['max_row'], 0)
        o = torch.zeros_like(q).index_copy(0, si, o1).index_copy(0, qi, o2)
    o = o.reshape(N, NH * HD)
    a = torch.addmm(lo['o'](o), o, d['Wo'].t()) if lo is not None else o @ d['Wo'].t()
    x = K_ADDRMS(x, a, d['n_poa'])
    h = K_RMS(x, d['n_pf'])
    gu = torch.addmm(lo['gu'](h), h, d['Wgu'].t()) if lo is not None else h @ d['Wgu'].t()
    mm = K_GEGLU(gu)
    dm = torch.addmm(lo['d'](mm), mm, d['Wd'].t()) if lo is not None else mm @ d['Wd'].t()
    return K_ADDRMS(x, dm, d['n_pof'])


def _forward_packed(self, ids, m):
    """ids [N] packed rows -> final-normed hidden [N, 2304]"""
    x = F.embedding(ids, self.embed) * torch.tensor(D ** 0.5, dtype=torch.bfloat16)
    cos, sin = rope_cs(m['pos'], ids.device)
    for i in range(self.n_layers):
        if self.ckpt and torch.is_grad_enabled():
            x = checkpoint(self.layer_packed, i, x, cos, sin, m, use_reentrant=False)
        else:
            x = self.layer_packed(i, x, cos, sin, m)
    return K_RMS(x, self.norm1)


EncTorso.layer_packed = _layer_packed
EncTorso.forward_packed = _forward_packed
