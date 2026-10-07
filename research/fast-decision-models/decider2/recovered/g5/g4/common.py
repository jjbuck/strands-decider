"""G4: from-scratch architecture comparison.  Models: dense (Qwen-like), slot (asymmetric-depth reader/deep stack), monarch-MLP dense.
All models: RoPE, SwiGLU, RMSNorm (pre-norm), QK-norm, GQA, no biases, tied embeddings over a compact vocab (top-32k Qwen3.5 ids + OOV)."""
import math, torch, torch.nn as nn, torch.nn.functional as F

V = 32768
PRESETS = {
    'S60': dict(d=768, L=9, H=12, Hkv=2, hd=64, ffn=2304, mon_ffn=8704),
    'S125': dict(d=1024, L=11, H=16, Hkv=2, hd=64, ffn=3072, mon_ffn=11520),
}
ARMS = {
    'dense': dict(kind='dense'),
    'dense3': dict(kind='dense', L=3),           # equal-inference-FLOP control for the slot arm (~ its reader)
    'slot': dict(kind='slot', k=2, S=16, T=128),
    'mudd': dict(kind='mudd'),                   # MUDDFormer (2502.12170): multiway dynamic dense connections
    'moe': dict(kind='moe', E=16, topk=2),       # fine-grained MoE: 16 experts of ffn/2, top-2 (active MLP FLOPs = dense)
    'monarch': dict(kind='monarch', nb=4),
}


class Cfg:
    def __init__(self, scale, arm, **kw):
        self.scale, self.arm = scale, arm
        d = dict(PRESETS[scale]); d.update(ARMS[arm]); d.update(kw)
        for k, v in d.items(): setattr(self, k, v)
        self.nb = getattr(self, 'nb', 0) if self.kind == 'monarch' else 0

    def __repr__(self): return f'Cfg({self.__dict__})'


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x): return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def rope_cache(maxpos, hd, theta=10000.0):
    inv = 1.0 / theta ** (torch.arange(0, hd, 2).float() / hd)
    f = torch.outer(torch.arange(maxpos).float(), inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):
    """x (B, T, H, hd); cos/sin (T, hd/2) or (B, T, hd/2)"""
    if cos.dim() == 2: cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    else: cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


class MonarchLinear(nn.Module):
    """W = P2 B2 P B1 (Monarch, Dao et al. 2204.00595): B1 block-diag nb blocks (din/nb -> mid/nb), stride permutation,
    B2 block-diag nb blocks (mid/nb -> dout/nb).  MACs = din*mid/nb + mid*dout/nb, mid = min(din, dout)."""
    def __init__(self, din, dout, nb, std=0.02):
        super().__init__()
        mid = min(din, dout); self.nb, self.din, self.dout, self.mid = nb, din, dout, mid
        assert din % nb == 0 and dout % nb == 0 and mid % nb == 0 and (mid // nb) % nb == 0
        self.b1 = nn.Parameter(torch.randn(nb, din // nb, mid // nb) / math.sqrt(din // nb))
        self.b2 = nn.Parameter(torch.randn(nb, mid // nb, dout // nb) * std * math.sqrt(din) / math.sqrt(mid // nb))
        self.b1._lr_mult = 2.0 if din // nb <= 256 else 1.0
        self.b2._lr_mult = 2.0

    def forward(self, x):
        sh = x.shape[:-1]; nb, m = self.nb, self.mid // self.nb
        x = x.reshape(-1, nb, self.din // nb).transpose(0, 1)          # (nb, N, din/nb)
        y = torch.bmm(x, self.b1)                                         # (nb, N, m)
        y = y.permute(1, 2, 0).reshape(-1, nb, m).transpose(0, 1)        # stride permutation -> (nb, N, m)
        z = torch.bmm(y, self.b2)                                         # (nb, N, dout/nb)
        return z.transpose(0, 1).reshape(*sh, self.dout)


def lin(din, dout, std):
    l = nn.Linear(din, dout, bias=False); nn.init.normal_(l.weight, std=std); return l


class MLP(nn.Module):
    def __init__(self, c, out_std):
        super().__init__()
        if c.nb:
            f = c.mon_ffn
            self.gu = MonarchLinear(c.d, 2 * f, c.nb, 0.02); self.dn = MonarchLinear(f, c.d, c.nb, out_std)
        else:
            f = c.ffn
            self.gu = lin(c.d, 2 * f, 0.02); self.dn = lin(f, c.d, out_std)

    def forward(self, x):
        g, u = self.gu(x).chunk(2, -1)
        return self.dn(F.silu(g) * u)


def rep_kv(k, H):
    return k.repeat_interleave(H // k.shape[1], dim=1)


class Attn(nn.Module):
    def __init__(self, c, out_std):
        super().__init__()
        self.H, self.Hkv, self.hd = c.H, c.Hkv, c.hd
        self.qkv = lin(c.d, (c.H + 2 * c.Hkv) * c.hd, 0.02); self.o = lin(c.H * c.hd, c.d, out_std)
        self.qn = RMSNorm(c.hd); self.kn = RMSNorm(c.hd)

    def proj(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split([self.H * self.hd, self.Hkv * self.hd, self.Hkv * self.hd], -1)
        q = apply_rope(self.qn(q.view(B, T, self.H, self.hd)), cos, sin).transpose(1, 2)
        k = apply_rope(self.kn(k.view(B, T, self.Hkv, self.hd)), cos, sin).transpose(1, 2)
        return q, k, v.view(B, T, self.Hkv, self.hd).transpose(1, 2)

    def out(self, y):
        B, H, T, hd = y.shape
        return self.o(y.transpose(1, 2).reshape(B, T, H * hd))

    def forward(self, x, cos, sin):
        q, k, v = self.proj(x, cos, sin)
        return self.out(F.scaled_dot_product_attention(q, rep_kv(k, self.H), rep_kv(v, self.H), is_causal=True))


class Block(nn.Module):
    def __init__(self, c, nlayers_total):
        super().__init__()
        out_std = 0.02 / math.sqrt(2 * nlayers_total)
        self.n1 = RMSNorm(c.d); self.att = Attn(c, out_std); self.n2 = RMSNorm(c.d); self.mlp = MLP(c, out_std)

    def forward(self, x, cos, sin):
        x = x + self.att(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class DeepBlock(Block):
    """deep-stack layer: rows attend to [state K/V (per-layer projection of the reader output) ; own rows' K/V]"""
    def __init__(self, c, nlayers_total):
        super().__init__(c, nlayers_total)
        self.skv = lin(c.d, 2 * c.Hkv * c.hd, 0.02); self.skn = RMSNorm(c.hd)

    def state_kv(self, rn, cos, sin):
        B, N, _ = rn.shape; a = self.att
        k, v = self.skv(rn).split([a.Hkv * a.hd, a.Hkv * a.hd], -1)
        k = apply_rope(self.skn(k.view(B, N, a.Hkv, a.hd)), cos, sin).transpose(1, 2)
        return k, v.view(B, N, a.Hkv, a.hd).transpose(1, 2)

    def forward(self, x, cos, sin, kS, vS, attend):
        q, kD, vD = self.att.proj(self.n1(x), cos, sin)
        y = attend(q, torch.cat([kS, kD], 2), torch.cat([vS, vD], 2))
        x = x + self.att.out(y)
        return x + self.mlp(self.n2(x))


class DenseLM(nn.Module):
    def __init__(self, c, maxpos=4096):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(V, c.d); nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(c, c.L) for _ in range(c.L)])
        self.nf = RMSNorm(c.d)
        cos, sin = rope_cache(maxpos, c.hd)
        self.register_buffer('cos', cos, persistent=False); self.register_buffer('sin', sin, persistent=False)

    def hidden(self, ids):
        T = ids.shape[1]; x = self.emb(ids); cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks: x = b(x, cos, sin)
        return self.nf(x)

    def lm_loss(self, ids, tgt):
        h = self.hidden(ids)
        logits = h @ self.emb.weight.t()
        return F.cross_entropy(logits.float().view(-1, V), tgt.reshape(-1), reduction='none').view(tgt.shape)

    def dec_hidden(self, ids, last):
        """ids (B, T) = [Q][state][Q] right-padded; last (B,) index of the readout row -> (B, d)"""
        h = self.hidden(ids)
        return h[torch.arange(ids.shape[0], device=ids.device), last]


class SlotLM(nn.Module):
    """Asymmetric-depth slot transformer.  Reader: k full-width causal layers over every token.  Deep stack: L layers over
    [S learned slots ; a block of <= T rows] only; the block rows start from the reader output at their positions; at every
    deep layer all rows attend to state K/V made from the reader output by a per-layer K/V projection (~3.5% of a layer)."""
    def __init__(self, c, maxpos=4096):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(V, c.d); nn.init.normal_(self.emb.weight, std=0.02)
        self.reader = nn.ModuleList([Block(c, c.k + c.L) for _ in range(c.k)])
        self.rn = RMSNorm(c.d)
        self.deep = nn.ModuleList([DeepBlock(c, c.k + c.L) for _ in range(c.L)])
        self.slots = nn.Parameter(torch.randn(c.S, c.d) * 0.02)
        self.nf = RMSNorm(c.d)
        cos, sin = rope_cache(maxpos, c.hd)
        self.register_buffer('cos', cos, persistent=False); self.register_buffer('sin', sin, persistent=False)
        self._bm = None

    def read(self, ids):
        N = ids.shape[1]; x = self.emb(ids); cos, sin = self.cos[:N], self.sin[:N]
        for b in self.reader: x = b(x, cos, sin)
        return x

    # ---------------- pretraining: every T-token block of the sequence is a "question block"; state = all earlier tokens
    def _pt_setup(self, N, device):
        from torch.nn.attention.flex_attention import create_block_mask
        S, T = self.c.S, self.c.T; nb = N // T; R = nb * (S + T)
        o = torch.arange(S + T, device=device)
        pos = (torch.arange(nb, device=device)[:, None] * T + torch.clamp(o - S, min=0)[None, :]).reshape(-1)
        self._pos = pos

        def mask_mod(b, h, q, kv):
            qb = q // (S + T); qo = q % (S + T)
            st = kv < N
            kk = kv - N; kb = kk // (S + T); ko = kk % (S + T)
            state_ok = st & (kv < qb * T)
            deep_ok = (~st) & (kb == qb) & ((ko < S) | ((qo >= S) & (ko <= qo)))
            return state_ok | deep_ok
        self._bm = create_block_mask(mask_mod, None, None, R, N + R, device=device)
        self._ptN = N

    def lm_loss(self, ids, tgt):
        from torch.nn.attention.flex_attention import flex_attention
        B, N = ids.shape; c = self.c; S, T = c.S, c.T; nb = N // T
        if self._bm is None or self._ptN != N: self._pt_setup(N, ids.device)
        r = self.read(ids); rn = self.rn(r)
        x = torch.cat([self.slots.view(1, 1, S, c.d).expand(B, nb, S, c.d).to(r.dtype), r.view(B, nb, T, c.d)], 2).reshape(B, nb * (S + T), c.d)
        cos, sin = self.cos[self._pos], self.sin[self._pos]
        cs, ss = self.cos[:N], self.sin[:N]
        bm = self._bm
        att = lambda q, k, v: flex_attention(q, k, v, block_mask=bm, enable_gqa=True)
        for blk in self.deep:
            kS, vS = blk.state_kv(rn, cs, ss)
            x = blk(x, cos, sin, kS, vS, att)
        h = self.nf(x.view(B, nb, S + T, c.d)[:, :, S:].reshape(B, N, c.d))
        logits = h @ self.emb.weight.t()
        return F.cross_entropy(logits.float().view(-1, V), tgt.reshape(-1), reduction='none').view(tgt.shape)

    # ---------------- decision forward: reader over [Q][state][Q2]; deep over [slots ; Q2]
    def dec_hidden(self, ids, nctx, nq, nrows=None):
        """ids (B, Nmax) = [Q][state][Q2] right-padded; nctx (B,) = len(Q)+len(state); nq (B,) = len(Q2) (<= T).
        Returns the readout row (last Q2 row of the deep stack), (B, d).  nrows: deep question rows (default T; = q at inference)."""
        B, Nmax = ids.shape; c = self.c; S = c.S; T = nrows or c.T; dev = ids.device
        r = self.read(ids); rn = self.rn(r)
        ar = torch.arange(T, device=dev)
        gidx = (nctx[:, None] + ar[None, :]).clamp(max=Nmax - 1)                    # (B, T)
        q2 = torch.gather(r, 1, gidx[:, :, None].expand(B, T, c.d))
        x = torch.cat([self.slots.view(1, S, c.d).expand(B, S, c.d).to(r.dtype), q2], 1)   # (B, S+T, d)
        R = S + T
        pos = nctx[:, None] + torch.clamp(torch.arange(R, device=dev) - S, min=0)[None, :]
        pos = pos.clamp(max=self.cos.shape[0] - 1)
        cos, sin = self.cos[pos], self.sin[pos]
        cs, ss = self.cos[:Nmax], self.sin[:Nmax]
        kidx = torch.arange(Nmax + R, device=dev); qi = torch.arange(R, device=dev)
        st = kidx[None, None, :] < Nmax
        state_ok = st & (kidx[None, None, :] < nctx[:, None, None])
        ko = (kidx - Nmax)[None, None, :]
        deep_ok = (~st) & ((ko < S) | ((qi[None, :, None] >= S) & (ko <= qi[None, :, None])))
        mask = (state_ok | deep_ok)[:, None]                                          # (B, 1, R, Nmax+R)
        H = c.H
        att = lambda q, k, v: F.scaled_dot_product_attention(q, rep_kv(k, H), rep_kv(v, H), attn_mask=mask)
        for blk in self.deep:
            kS, vS = blk.state_kv(rn, cs, ss)
            x = blk(x, cos, sin, kS, vS, att)
        h = self.nf(x)
        return h[torch.arange(B, device=dev), S + nq - 1]


class MuddBlock(Block):
    """block with multiway inputs: q, k, v and residual streams are separate dynamic dense combinations of all earlier layer outputs"""
    def __init__(self, c, nlayers_total):
        super().__init__(c, nlayers_total)
        self.nk = RMSNorm(c.d); self.nv = RMSNorm(c.d)

    def forward(self, xq, xk, xv, xr, cos, sin):
        a = self.att; B, T, _ = xq.shape
        Wq, Wk, Wv = a.qkv.weight.split([a.H * a.hd, a.Hkv * a.hd, a.Hkv * a.hd], 0)
        q = F.linear(self.n1(xq), Wq).view(B, T, a.H, a.hd); k = F.linear(self.nk(xk), Wk).view(B, T, a.Hkv, a.hd)
        v = F.linear(self.nv(xv), Wv).view(B, T, a.Hkv, a.hd)
        q = apply_rope(a.qn(q), cos, sin).transpose(1, 2); k = apply_rope(a.kn(k), cos, sin).transpose(1, 2); v = v.transpose(1, 2)
        x = xr + a.out(F.scaled_dot_product_attention(q, rep_kv(k, a.H), rep_kv(v, a.H), is_causal=True))
        return x + self.mlp(self.n2(x))


class DynDense(nn.Module):
    """MUDD dynamic dense aggregation after layer i: per-token weights over the i+2 hidden states for C streams (static + dynamic)"""
    def __init__(self, d, nh, C):
        super().__init__()
        self.nh, self.C = nh, C
        self.norm = RMSNorm(d)
        self.w1 = lin(d, nh * C, 0.02); self.w2 = nn.Linear(nh * C, nh * C, bias=False); nn.init.zeros_(self.w2.weight)
        st = torch.zeros(C, nh); st[:, -1] = 1.0
        self.static = nn.Parameter(st)

    def forward(self, hs):
        x = hs[-1]
        dw = (self.w2(F.gelu(self.w1(self.norm(x)))).view(*x.shape[:-1], self.C, self.nh) + self.static).to(x.dtype)
        return [sum(dw[..., c, j, None] * hs[j] for j in range(self.nh)) for c in range(self.C)]


class MuddLM(DenseLM):
    def __init__(self, c, maxpos=4096):
        super().__init__(c, maxpos)
        self.blocks = nn.ModuleList([MuddBlock(c, c.L) for _ in range(c.L)])
        self.dd = nn.ModuleList([DynDense(c.d, i + 2, 4 if i < c.L - 1 else 1) for i in range(c.L)])

    def hidden(self, ids):
        T = ids.shape[1]; x = self.emb(ids); cos, sin = self.cos[:T], self.sin[:T]
        hs = [x]; xs = [x, x, x, x]
        for i, b in enumerate(self.blocks):
            hs.append(b(*xs, cos, sin))
            xs = self.dd[i](hs)
        return self.nf(xs[0])


class MoE(nn.Module):
    """fine-grained MoE MLP: E SwiGLU experts of width ffn/topk, top-k token choice (active MLP MACs = dense MLP), softmax router renormalised
    over the top-k, switch load-balancing loss (self.aux).  Static-shape capacity dispatch (capacity factor cf; overflow dropped)."""
    def __init__(self, c, out_std, cf=1.25):
        super().__init__()
        self.E, self.k, self.cf = c.E, c.topk, cf; f = c.ffn // c.topk
        self.f = f
        self.router = lin(c.d, c.E, 0.02)
        self.gu = nn.Parameter(torch.randn(c.E, c.d, 2 * f) * 0.02); self.dn = nn.Parameter(torch.randn(c.E, f, c.d) * out_std)
        self.aux = None

    def forward(self, x):
        sh = x.shape; d = sh[-1]; x = x.reshape(-1, d); N = x.shape[0]; E, k = self.E, self.k
        Cap = int(math.ceil(self.cf * N * k / E))
        p = torch.softmax(self.router(x).float(), -1)
        w, e = torch.topk(p, k, -1); w = w / w.sum(-1, keepdim=True)
        fe = e.reshape(-1); fw = w.reshape(-1)
        oh = F.one_hot(fe, E)
        self.aux = E * (oh.float().mean(0) * p.mean(0)).sum()
        pos = ((oh.cumsum(0) - 1) * oh).sum(-1)
        keep = pos < Cap
        dest = torch.where(keep, fe * Cap + pos, torch.full_like(pos, E * Cap))
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        buf = torch.zeros(E * Cap + 1, d, device=x.device, dtype=x.dtype).index_copy(0, dest, x[tok])
        h = torch.bmm(buf[:-1].view(E, Cap, d), self.gu.to(x.dtype))
        g, u = h.chunk(2, -1)
        out = torch.bmm(F.silu(g) * u, self.dn.to(x.dtype)).reshape(E * Cap, d)
        out = torch.cat([out, out.new_zeros(1, d)], 0)
        y = torch.zeros(N, d, device=x.device, dtype=x.dtype).index_add(0, tok, out[dest] * (fw * keep).to(x.dtype)[:, None])
        return y.view(sh)


def build(c):
    if c.kind == 'slot': return SlotLM(c)
    if c.kind == 'mudd': return MuddLM(c)
    if c.kind == 'moe':
        m = DenseLM(c)
        for b in m.blocks: b.mlp = MoE(c, 0.02 / math.sqrt(2 * c.L))
        return m
    return DenseLM(c)


# ---------------------------------------------------------------- analytic MAC counts (multiply-accumulates; FLOPs = 2 x MACs)
def layer_macs(c, monarch=None):
    att = c.d * (c.H + 2 * c.Hkv) * c.hd + c.H * c.hd * c.d
    if (c.nb if monarch is None else monarch):
        f = c.mon_ffn; d = c.d; nb = c.nb
        mlp = (d * d / nb + d * 2 * f / nb) + (f * d / nb + d * d / nb)
    else:
        mlp = 3 * c.d * c.ffn + (c.d * c.E if c.kind == 'moe' else 0)   # MoE: active expert MACs (top-k x ffn/k) + router
    return att + mlp


def attn_macs(c, nq, nk):
    return 2 * nq * nk * c.H * c.hd


def nonemb_params(c):
    p = layer_macs(c) + 4 * c.d
    if c.kind == 'slot':
        return c.k * p + c.L * (p + c.d * 2 * c.Hkv * c.hd) + c.S * c.d
    return c.L * p


def train_macs_per_token(c, N=1024):
    """forward MACs per training token (LM head incl.)"""
    head = c.d * V
    if c.kind == 'slot':
        S, T = c.S, c.T; nb = N // T
        lm = layer_macs(c)
        reader = c.k * (lm + attn_macs(c, 1, N / 2))
        kv = c.L * c.d * 2 * c.Hkv * c.hd
        avg_state = sum(j * T for j in range(nb)) / nb
        rows = (S + T) / T
        avg_keys_tok = avg_state + S + T / 2; avg_keys_slot = avg_state + S
        deep = c.L * (rows * lm + (T * attn_macs(c, 1, avg_keys_tok) + S * attn_macs(c, 1, avg_keys_slot)) / T)
        return reader + kv + deep + head
    return c.L * (layer_macs(c) + attn_macs(c, 1, N / 2)) + head + mudd_macs(c)


def mudd_macs(c):
    """per-token MACs of the MUDD aggregation (DA MLPs + weighted sums); 0 for other arms"""
    if c.kind != 'mudd': return 0
    t = 0
    for i in range(c.L):
        nh = i + 2; C_ = 4 if i < c.L - 1 else 1
        t += c.d * nh * C_ + (nh * C_) ** 2 + C_ * nh * c.d
    return t


def infer_macs(c, N=1000, q=64):
    """decision pass: [Q][state][Q] with a state of N tokens and a question of q tokens; no LM head."""
    if c.kind == 'slot':
        tot = 2 * q + N; nctx = q + N; R = c.S + q
        reader = c.k * (tot * layer_macs(c) + attn_macs(c, tot, tot / 2))
        kv = c.L * nctx * c.d * 2 * c.Hkv * c.hd
        deep = c.L * (R * layer_macs(c) + attn_macs(c, R, nctx + R / 2))
        return dict(total=reader + kv + deep, per_state_token=(c.k * layer_macs(c) + c.L * c.d * 2 * c.Hkv * c.hd))
    tot = 2 * q + N
    return dict(total=c.L * (tot * layer_macs(c) + attn_macs(c, tot, tot / 2)) + tot * mudd_macs(c),
                per_state_token=c.L * layer_macs(c) + mudd_macs(c))


def param_groups(model, wd):
    """weight decay on matrices (incl. tied embedding), none on norms/slots; Monarch factors carry an lr multiplier"""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        key = (getattr(p, '_lr_mult', 1.0), wd if (p.dim() >= 2 and n != 'slots') else 0.0)
        groups.setdefault(key, []).append(p)
    return [dict(params=ps, lr_mult=m, weight_decay=w) for (m, w), ps in groups.items()]


if __name__ == '__main__':
    for s in PRESETS:
        for a in ARMS:
            c = Cfg(s, a)
            im = infer_macs(c)
            print(s, a, f'nonemb {nonemb_params(c)/1e6:.1f}M  train GFLOP/tok {2*3*train_macs_per_token(c)/1e9:.3f}  '
                  f'infer GFLOP@1000+64 {2*im["total"]/1e9:.1f}  per-state-token MFLOP {2*im["per_state_token"]/1e6:.1f}')
