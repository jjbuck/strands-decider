"""M1 library: all-attention hobson-v19 (bf16, differentiable eager forward) next to the frozen hobson teacher, in hobson's shared-prefix
multi-question layout [state][q1][q2]... (each question sees the state and itself, as strands-decider's prefix-cache path does).

The 18 GDN layers' mixers are replaced by causal softmax attention built from the GDN layer's own parameters (C[i]):
  proj = h @ Win  (same 8224 rows: q 2048 | k 2048 | v 2048 | z 2048 | b 16 | a 16)
  q, k, v = SiLU(causal depthwise conv k=4 (q|k|v))            cfg conv (else SiLU only)
  q = rms(q) * qg[h], k = rms(k) * kg[h]  (per head, [16, 128]) qg init = tau (softmax temperature), kg init 1
  RoPE on dims 0..63 of each 128-dim head with hobson's frequencies (theta 1e7, 32 freqs; same pairing as hobson's attention layers)  cfg rope
  logit_ij = q_i.k_j / sqrt(128) + [decay] (G_i - G_j) + [beta] log sigmoid(b_j),   G = cumsum g,  g = -exp(A_log) softplus(a + dt_bias)
      the biases ride in 6 extra q/k dims (fp32-exact 3-way bf16 split), head dim 136 (v zero-padded), so plain SDPA / FlashAttention runs it
  o = softmax attention (16 heads, MHA, causal; question rows see state + own question)
  out = Wo (gn_w * rms(o) * SiLU(z))   (hobson's GDN gated-RMSNorm output gate)
Teacher = hobson exactly (fla chunk_gated_delta_rule for GDN; SDPA for attention), same weights, no LoRA.
Student = hobson weights + converted mixers (full-rank, fp32 master) on the layers in self.active + LoRA r32 on everything else.
"""
import os, sys, math, json, hashlib, random, collections, time
W = os.path.expanduser('~/work')
sys.path[:0] = [f'{W}/tokens', f'{W}/systems/g', f'{W}/evalkit', f'{W}/m1']
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

ATT = (3, 7, 11, 15, 19, 23)
GDN = tuple(i for i in range(24) if i not in ATT)
KEEP = (5, 11, 17, 23)
GEMMS = ('Win', 'Wo', 'Wgu', 'Wd')
BF = torch.bfloat16
DEFAULT_CFG = dict(conv=True, rope=True, beta=True, decay=True, slots=1)


# ------------------------------------------------------------------ elementwise helpers
def rms_zc(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)).to(x.dtype)


def rms_g(x, g, eps):
    """x [T, H, D] -> rms-normalised per head times a plain gain g [H, D] (fp32 math)"""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * g).to(x.dtype)


def rope(t, cos, sin):
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


def conv_e(raw, w, pv):
    """depthwise causal conv (k=4) + SiLU with explicit previous-row indices pv [T, 3] into [zero row; raw] (q3lib._conv_e)"""
    ext = torch.cat([torch.zeros(1, raw.shape[1], device=raw.device, dtype=raw.dtype), raw], 0)
    wf = w.float()
    acc = raw.float() * wf[:, 3] + ext[pv[:, 0]].float() * wf[:, 0] + ext[pv[:, 1]].float() * wf[:, 1] + ext[pv[:, 2]].float() * wf[:, 2]
    return F.silu(acc).to(raw.dtype)


def gnorm_e(o, z, gn_w, eps):
    of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    return ((gn_w * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(z.dtype)


def swiglu(gu, I):
    g, u = gu.split([I, gu.shape[1] - I], 1)
    return F.silu(g) * u


def split3(x):
    """fp32 x -> three bf16 tensors whose fp32 sum equals x to ~2^-24 relative (gradient flows through the first part: d(sum)/dx = 1)"""
    a = x.to(BF); r = x - a.float()
    b = r.to(BF); r = r - b.float()
    return a, b, r.to(BF)


def attn_core(q, k, v, lay, scale):
    """q [T, Hq, D], k/v [T, Hk, D] (Hk divides Hq). State rows: causal over the state. Question rows: state + own question (one masked call).
    -> [T, Hq, D]"""
    Ls = lay['Ls']; T = q.shape[0]; Hq, Hk = q.shape[1], k.shape[1]
    if Hk != Hq:
        k = k.repeat_interleave(Hq // Hk, 1); v = v.repeat_interleave(Hq // Hk, 1)
    qt = q.transpose(0, 1)[None]; kt = k.transpose(0, 1)[None]; vt = v.transpose(0, 1)[None]
    o1 = F.scaled_dot_product_attention(qt[:, :, :Ls], kt[:, :, :Ls], vt[:, :, :Ls], is_causal=True, scale=scale)[0]
    if T == Ls: return o1.transpose(0, 1)
    o2 = F.scaled_dot_product_attention(qt[:, :, Ls:], kt, vt, attn_mask=lay['bmask'][None, None], scale=scale)[0]
    return torch.cat([o1, o2], 1).transpose(0, 1)


# ------------------------------------------------------------------ pointer head (hobson's, frozen)
class StdHead(nn.Module):
    def __init__(self, h0):
        super().__init__()
        self.norm = nn.LayerNorm(h0.q.in_features); self.q = nn.Linear(h0.q.in_features, h0.q.out_features); self.k = nn.Linear(h0.k.in_features, h0.k.out_features)
        self.load_state_dict({k: v.detach().clone().float() for k, v in h0.state_dict().items()}, strict=False)
        self.scale = h0.q.out_features ** -0.5

    def forward(self, decide, options):
        d = self.q(self.norm(decide)).unsqueeze(-1)
        o = self.k(self.norm(options))
        return (o @ d).squeeze(-1) * self.scale


# ------------------------------------------------------------------ model
class M1:
    def __init__(self, cfg=None, maxlen=16384):
        from plib import P
        import lean as LN, gc
        p = P(); p.model.config.max_length = maxlen
        self.p = p; self.tok = p.tok
        lean = LN.Lean(p.tm)
        self.L = lean.layers; self.eps = lean.eps; self.dev = lean.dev
        for d in self.L:
            for k, v in list(d.items()):
                if torch.is_tensor(v): d[k] = v.detach().clone()
        self.embed = lean.embed.detach().clone(); self.norm_w = lean.norm_w.detach().clone(); self.inv = lean.inv.clone()
        self.head = StdHead(p.model.head).to(self.dev).eval()
        for q_ in self.head.parameters(): q_.requires_grad_(False)
        self.temps = {}
        p.model.torso = None; p.tm = None; del lean
        gc.collect(); torch.cuda.empty_cache()
        self.cfg = dict(DEFAULT_CFG, **(cfg or {}))
        self.C = nn.ModuleDict()          # str(i) -> converted mixer params
        self.active = set()               # converted layers the student uses
        self.lora = None; self.lora_scale = 2.0

    def temp(self, kind):
        if kind not in self.temps: self.temps[kind] = self.p.temp_for(kind)
        return self.temps[kind]

    # ---------- conversion ----------
    def convert(self, layers, tau=1.0):
        """create converted mixers for `layers` from each GDN layer's own parameters. tau: float or {i: float} (q-gain init = temperature)"""
        for i in layers:
            assert i in GDN, i
            d = self.L[i]; t = tau[i] if isinstance(tau, dict) else tau
            m = nn.Module()
            m.Win = nn.Parameter(d['Win'].float().clone()); m.Wo = nn.Parameter(d['Wo'].float().clone())
            m.conv_w = nn.Parameter(d['conv_w'].float().clone())
            m.qg = nn.Parameter(torch.full((16, 128), float(t), device=self.dev)); m.kg = nn.Parameter(torch.ones(16, 128, device=self.dev))
            m.gn_w = nn.Parameter(d['gn_w'].float().clone())
            m.A_log = nn.Parameter(d['A_log'].float().clone()); m.dt_bias = nn.Parameter(d['dt_bias'].float().clone())
            self.C[str(i)] = m
        self.active |= set(layers)

    def mixer_params(self, big=True):
        """big: the two projection matrices of every converted layer; else the small ones (gains, conv, gate params, gn_w)"""
        out = []
        for m in self.C.values():
            for n_, p_ in m.named_parameters():
                if (n_ in ('Win', 'Wo')) == big: out.append(p_)
        return out

    def add_lora(self, r=32, alpha=64, seed=0, layers_all=True):
        """LoRA on every GEMM that is not a converted mixer: Wgu, Wd of all layers; Win, Wo of attention layers and of unconverted GDN layers."""
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.lora = nn.ModuleDict()
        for i in range(24):
            for k in GEMMS:
                out, inp = self.L[i][k].shape
                mm = nn.Module()
                mm.A = nn.Parameter((torch.randn(r, inp, generator=g) / math.sqrt(inp)).to(self.dev))
                mm.B = nn.Parameter(torch.zeros(out, r, device=self.dev))
                self.lora[f'{i}_{k}'] = mm
        self.lora_scale = alpha / r
        return list(self.lora.parameters())

    def lora_params(self):
        """LoRA tensors that are live for the current self.active (converted layers' Win/Wo LoRA are unused)"""
        out = []
        for key, mm in self.lora.items():
            i, k = key.split('_'); i = int(i)
            if i in self.active and k in ('Win', 'Wo'): continue
            out += [mm.A, mm.B]
        return out

    # ---------- state io ----------
    def trainable_state(self, half=False):
        sd = {}
        for key, m in self.C.items():
            for n_, p_ in m.named_parameters():
                t = p_.detach()
                sd[f'C.{key}.{n_}'] = (t.to(BF) if half and n_ in ('Win', 'Wo') else t).cpu()
        if self.lora is not None:
            for key, mm in self.lora.items():
                sd[f'lora.{key}.A'] = mm.A.detach().cpu(); sd[f'lora.{key}.B'] = mm.B.detach().cpu()
        sd['_meta'] = dict(cfg=self.cfg, active=sorted(self.active), lora_scale=self.lora_scale)
        return sd

    def load_trainable(self, sd, active=None, lora=True):
        if isinstance(sd, str): sd = torch.load(os.path.expanduser(sd), map_location='cpu')
        meta = sd['_meta']; self.cfg = dict(DEFAULT_CFG, **meta['cfg'])
        layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('C.')})
        self.convert(layers)
        with torch.no_grad():
            for i in layers:
                m = self.C[str(i)]
                for n_, p_ in m.named_parameters(): p_.copy_(sd[f'C.{i}.{n_}'].to(self.dev).float())
        self.active = set(meta['active'] if active is None else active)
        if lora and any(k.startswith('lora.') for k in sd):
            r = sd['lora.0_Wgu.A'].shape[0]
            self.add_lora(r=r, alpha=meta.get('lora_scale', 2.0) * r)
            with torch.no_grad():
                for key, mm in self.lora.items():
                    mm.A.copy_(sd[f'lora.{key}.A'].to(self.dev)); mm.B.copy_(sd[f'lora.{key}.B'].to(self.dev))
        return meta

    # ---------- tokenization / layout ----------
    def prep(self, state, qdicts):
        """-> dict(s=state ids, qs=[dict(q ids, opt offsets, n, kind, temp, labels)])"""
        qs = []; s0 = None
        for qd in qdicts:
            pr = self.p.prep(state, qd)
            if s0 is None: s0 = pr['s']
            assert pr['s'] == s0
            qs.append(dict(q=pr['q'], opt=list(pr['opt']), n=pr['rq'].n_slots, kind=pr['rq'].kind, temp=self.temp(pr['rq'].kind), labels=pr['rq'].slot_labels))
        return dict(s=s0, qs=qs)

    def layout(self, rq):
        dev = self.dev; Ls = len(rq['s']); lens = [len(x['q']) for x in rq['qs']]
        T = Ls + sum(lens)
        pos = list(range(Ls)); prev = [[t - 3 + j if t - 3 + j >= 0 else -1 for j in range(3)] for t in range(Ls)]
        seg = []; s = Ls; rows = []; gsub = [0] * Ls; gadd = [0] * Ls
        for L, x in zip(lens, rq['qs']):
            seg.append((s, s + L)); pos += list(range(Ls, Ls + L))
            for tau in range(L):
                prev.append([(s + tau - 3 + j) if tau - 3 + j >= 0 else (Ls + tau - 3 + j if Ls + tau - 3 + j >= 0 else -1) for j in range(3)])
            gsub += [s] * L; gadd += [Ls] * L
            rows.append([s + L - 1] + [s + o for o in x['opt']])
            s += L
        ids = torch.tensor(rq['s'] + [t for x in rq['qs'] for t in x['q']], device=dev)
        posf = torch.tensor(pos, device=dev, dtype=torch.float32)
        fr = posf[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cu = torch.tensor([0] + [sum(lens[:i + 1]) for i in range(len(lens))], device=dev, dtype=torch.long)
        pv = torch.tensor(prev, device=dev, dtype=torch.long) + 1          # into [zero row; raw]
        allrows = torch.tensor([r for rr in rows for r in rr], device=dev)
        R = T - Ls
        bmask = None
        if R > 0:
            bmask = torch.zeros(R, T, dtype=torch.bool, device=dev); bmask[:, :Ls] = True
            for (a0, a1) in seg:
                bmask[a0 - Ls:a1 - Ls, a0:a1] = torch.tril(torch.ones(a1 - a0, a1 - a0, dtype=torch.bool, device=dev))
        return dict(Ls=Ls, T=T, lens=lens, seg=seg, rows=rows, allrows=allrows, ids=ids, cos=fr.cos().to(BF), sin=fr.sin().to(BF),
                    cu=cu, pv=pv, M=len(lens), bmask=bmask, gsub=torch.tensor(gsub, device=dev), gadd=torch.tensor(gadd, device=dev), qs=rq['qs'])

    # ---------- hobson mixers ----------
    def _gdn(self, i, d, proj, lay):
        T = lay['T']; Ls = lay['Ls']; M = lay['M']; eps = self.eps
        raw, z, b, a = proj.split([6144, 2048, 16, 16], 1)
        beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
        cv = conv_e(raw, d['conv_w'], lay['pv'])
        q, k, v = cv.split(2048, dim=-1)
        q = q.reshape(1, T, 16, 128); k = k.reshape(1, T, 16, 128); v = v.reshape(1, T, 16, 128)
        bt = beta.to(q.dtype)
        c_ = lambda t: t.contiguous()
        o1, S = chunk_gated_delta_rule(c_(q[:, :Ls]), c_(k[:, :Ls]), c_(v[:, :Ls]), c_(g[None, :Ls]), c_(bt[None, :Ls]), use_qk_l2norm_in_kernel=True,
                                       output_final_state=True)
        if M > 0:
            o2, _ = chunk_gated_delta_rule(c_(q[:, Ls:]), c_(k[:, Ls:]), c_(v[:, Ls:]), c_(g[None, Ls:]), c_(bt[None, Ls:]),
                                           initial_state=S.expand(M, -1, -1, -1).contiguous(), cu_seqlens=lay['cu'], use_qk_l2norm_in_kernel=True)
            o = torch.cat([o1[0], o2[0]], 0)
        else:
            o = o1[0]
        return gnorm_e(o, z, d['gn_w'], eps).reshape(T, 2048)

    def _attn(self, i, d, proj, lay):
        T = lay['T']; eps = self.eps
        qg, kk, v = proj.split([4096, 512, 512], 1)
        qh, gate = qg.reshape(T, 8, 512).split([256, 256], -1)
        kk = kk.reshape(T, 2, 256); v = v.reshape(T, 2, 256)
        qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
        qh, kk = rope(qh, lay['cos'], lay['sin']), rope(kk, lay['cos'], lay['sin'])
        o = attn_core(qh, kk, v, lay, 256 ** -0.5)
        return (o * torch.sigmoid(gate)).reshape(T, 2048)

    # ---------- the converted mixer ----------
    def _cmix(self, i, c, h, lay, cfg=None):
        """h: pre-normed input [T, 2048] bf16 -> gated-norm output [T, 2048] (before Wo)"""
        cf = cfg or self.cfg
        T = lay['T']; eps = self.eps
        proj = h @ c.Win.to(BF).t()
        raw, z, b, a = proj.split([6144, 2048, 16, 16], 1)
        cv = conv_e(raw, c.conv_w, lay['pv']) if cf['conv'] else F.silu(raw)
        q, k, v = cv.reshape(T, 3, 16, 128).unbind(1)
        slots = cf.get('slots', 1) and (cf['beta'] or cf['decay'])
        if slots:      # in-head bias slots: dims 0..121 carry content, 122..127 the gate biases (head dim stays 128)
            q = rms_g(q[..., :122], c.qg[:, :122], eps); k = rms_g(k[..., :122], c.kg[:, :122], eps)
        else:
            q = rms_g(q, c.qg, eps); k = rms_g(k, c.kg, eps)
        if cf['rope']:
            q = rope(q, lay['cos'], lay['sin']); k = rope(k, lay['cos'], lay['sin'])
        sc = 128 ** -0.5
        if cf['beta'] or cf['decay']:
            Aq = torch.zeros(T, 16, device=h.device, dtype=torch.float32); Bk = torch.zeros_like(Aq)
            if cf['decay']:
                g = -c.A_log.exp() * F.softplus(a.float() + c.dt_bias)
                cs = g.cumsum(0); cs0 = F.pad(cs, (0, 0, 1, 0))
                G = cs + cs0[lay['gadd']] - cs0[lay['gsub']]
                Aq = Aq + G; Bk = Bk - G
            if cf['beta']:
                Bk = Bk + F.logsigmoid(b.float())
            a1, a2, a3 = split3(Aq); b1, b2, b3 = split3(Bk)
            one = torch.ones(T, 16, 3, device=h.device, dtype=BF)
            if slots:
                q = torch.cat([q * sc, a1[..., None], a2[..., None], a3[..., None], one], -1)
                k = torch.cat([k, one, b1[..., None], b2[..., None], b3[..., None]], -1)
            else:
                zer = torch.zeros(T, 16, 2, device=h.device, dtype=BF)
                q = torch.cat([q * sc, a1[..., None], a2[..., None], a3[..., None], one, zer], -1)
                k = torch.cat([k, one, b1[..., None], b2[..., None], b3[..., None], zer], -1)
                v = torch.cat([v, torch.zeros(T, 16, 8, device=h.device, dtype=BF)], -1)
            sc = 1.0
        o = attn_core(q, k, v, lay, sc)[..., :128]
        return gnorm_e(o, z, c.gn_w, eps).reshape(T, 2048)

    # ---------- layers ----------
    def lin(self, x, i, k):
        y = x @ self.L[i][k].t()
        if self.lora is not None:
            mm = self.lora[f'{i}_{k}']
            y = y + ((x @ mm.A.t().to(x.dtype)) @ mm.B.t().to(x.dtype)) * self.lora_scale
        return y

    def _slayer(self, i, x, lay):
        d = self.L[i]; eps = self.eps
        h = rms_zc(x, d['in_norm'], eps)
        if i in self.active:
            c = self.C[str(i)]
            x = x + self._cmix(i, c, h, lay) @ c.Wo.to(BF).t()
        else:
            proj = self.lin(h, i, 'Win')
            o = self._gdn(i, d, proj, lay) if d['type'] == 'linear_attention' else self._attn(i, d, proj, lay)
            x = x + self.lin(o, i, 'Wo')
        gu = self.lin(rms_zc(x, d['post_norm'], eps), i, 'Wgu')
        return x + self.lin(swiglu(gu, int(d['I'])), i, 'Wd')

    def _tlayer(self, i, x, lay):
        d = self.L[i]; eps = self.eps
        proj = rms_zc(x, d['in_norm'], eps) @ d['Win'].t()
        o = self._gdn(i, d, proj, lay) if d['type'] == 'linear_attention' else self._attn(i, d, proj, lay)
        x = x + o @ d['Wo'].t()
        gu = rms_zc(x, d['post_norm'], eps) @ d['Wgu'].t()
        return x + swiglu(gu, int(d['I'])) @ d['Wd'].t()

    def forward(self, lay, student=True, keep=KEEP, ckpt=True):
        """-> (list of per-question log-probs [n] fp32, dict layer -> residual at readout rows [R, 2048] fp32)"""
        rows = lay['allrows']
        x = F.embedding(lay['ids'], self.embed)
        kept = {}
        fn = self._slayer if student else self._tlayer
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(fn, i, x, lay, use_reentrant=False)
            else:
                x = fn(i, x, lay)
            if i in keep: kept[i] = x.index_select(0, rows).float()
        h = rms_zc(x.index_select(0, rows), self.norm_w, self.eps)
        return self.readout(h, lay), kept

    def readout(self, h, lay):
        out = []; j = 0
        for rr, qx in zip(lay['rows'], lay['qs']):
            hb = h[j:j + len(rr)].float(); j += len(rr)
            lg = self.head(hb[:1], hb[1:][None])[0] / qx['temp']
            out.append(torch.log_softmax(lg[:qx['n']], -1))
        return out

    def run(self, rq, student=True, keep=KEEP, ckpt=True):
        lay = self.layout(rq)
        return self.forward(lay, student, keep, ckpt), lay

    # ---------- teacher-forced local fidelity / transfer ----------
    def local(self, lay, layers, cfgs=None, backward=False, taus=None):
        """teacher pass; at each layer i in `layers`, the converted mixer's residual contribution on the TEACHER's input vs the GDN's.
        cfgs: list of (name, cfg) evaluated (no grad) -> {name: {i: relMSE}}; or backward=True: train-mode loss on self.cfg, grads accumulate,
        -> {i: relMSE}. taus: optional {name: [tau values]} -> evaluates qg scaled by tau/current (no grad)."""
        x = F.embedding(lay['ids'], self.embed); res = {}
        for i in range(24):
            d = self.L[i]
            with torch.no_grad():
                h = rms_zc(x, d['in_norm'], self.eps)
                proj = h @ d['Win'].t()
                o = self._gdn(i, d, proj, lay) if d['type'] == 'linear_attention' else self._attn(i, d, proj, lay)
                yt = (o @ d['Wo'].t()).float(); den = yt.pow(2).sum()
            if i in layers:
                c = self.C[str(i)]
                if backward:
                    ys = self._cmix(i, c, h, lay) @ c.Wo.to(BF).t()
                    l = (ys.float() - yt).pow(2).sum() / den
                    l.backward(); res[i] = float(l.detach())
                else:
                    with torch.no_grad():
                        for nm, cf in (cfgs or [('cur', self.cfg)]):
                            for tau in (taus or {}).get(nm, [None]):
                                if tau is not None:
                                    q0 = c.qg.data.clone(); c.qg.data.fill_(tau)
                                ys = self._cmix(i, c, h, lay, cf) @ c.Wo.to(BF).t()
                                e = float((ys.float() - yt).pow(2).sum() / den)
                                cs_ = float(F.cosine_similarity(ys.float(), yt, dim=-1).mean())
                                res.setdefault(nm if tau is None else f'{nm}@{tau}', {})[i] = (e, cs_)
                                if tau is not None: c.qg.data.copy_(q0)
            with torch.no_grad():
                x = x + yt.to(x.dtype)
                gu = rms_zc(x, d['post_norm'], self.eps) @ d['Wgu'].t()
                x = x + swiglu(gu, int(d['I'])) @ d['Wd'].t()
        return res


# ------------------------------------------------------------------ data (train split only; dev = 10% of TRAIN tasks by hash, as J14 / Q3)
def is_dev(task): return int(hashlib.sha1(task.encode()).hexdigest()[:8], 16) % 10 == 0


def v5_q(r):
    ins = r['instructions']
    if r['kind'] in ('choice', 'noul'): return {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def load_data(maxtok=5000, ndev=100, seed=0, cf_path=None):
    EV = set(json.load(open(f'{W}/evalkit/split.json'))['eval_tasks'])
    pool, dev = [], []
    for l in open(f'{W}/evalkit/train_pool.jsonl'):
        r = json.loads(l)
        assert r['task'] not in EV
        if r['n_state_tok'] > maxtok or r['n_state_tok'] < 32: continue
        (dev if is_dev(r['task']) else pool).append(dict(state=r['state'], questions=r['questions'], task=r['task'], rid=r['rid'], n=r['n_state_tok']))
    R = random.Random(seed); R.shuffle(pool)
    Rd = random.Random(123); Rd.shuffle(dev); dev = dev[:ndev]
    v5 = [json.loads(l) for l in open(f'{W}/training/data/train_v5.jsonl')]
    R.shuffle(v5)
    cf = []
    if cf_path and os.path.exists(cf_path):
        by = collections.defaultdict(list)
        for l in open(cf_path):
            r = json.loads(l)
            if r['task'] in EV or is_dev(r['task']): continue
            by[r['pair']].append(r)
        cf = list(by.values()); R.shuffle(cf)
    return pool, dev, v5, cf


# ------------------------------------------------------------------ dev references and scoring (train-split dev set)
def dev_rq(m, r, maxq=8):
    names = sorted(r['questions'])[:maxq]
    return m.prep(r['state'], [r['questions'][n] for n in names]), names


def make_devref(m, devset, path, maxq=8):
    if os.path.exists(path): return torch.load(path)
    ref = {}
    with torch.no_grad():
        for r in devset:
            rq, names = dev_rq(m, r, maxq)
            (lt, kt), _ = m.run(rq, student=False, keep=KEEP)
            rq0 = m.prep('', [r['questions'][n] for n in names])
            (l0, _), _ = m.run(rq0, student=False, keep=())
            ref[r['rid']] = dict(lt=[x.cpu() for x in lt], ns=[int(x.argmax()) for x in l0], kt={i: v.cpu() for i, v in kt.items()})
    torch.save(ref, path)
    return ref


def dev_eval(m, devset, ref, maxq=8, limit=None):
    rows = []; hid = collections.defaultdict(list)
    with torch.no_grad():
        for r in devset[:limit]:
            rq, names = dev_rq(m, r, maxq)
            (ls, ks), _ = m.run(rq, student=True, keep=KEEP)
            rf = ref[r['rid']]
            for i, v in ks.items():
                xt = rf['kt'][i].to(v.device)
                hid[i].append(float(((v - xt).pow(2).sum(-1) / xt.pow(2).sum(-1).clamp_min(1e-6)).mean()))
            for lt, l_, ns in zip(rf['lt'], ls, rf['ns']):
                pt = lt.exp(); ps = l_.float().cpu().exp()
                rows.append(dict(t=int(pt.argmax()), s=int(ps.argmax()), ns=ns, tv=float(0.5 * (pt - ps).abs().sum()), kl=float((pt * (lt - l_.float().cpu())).sum())))
    n = len(rows); sd = [x for x in rows if x['t'] != x['ns']]
    return dict(n=n, agree=sum(x['t'] == x['s'] for x in rows) / n, tv=sum(x['tv'] for x in rows) / n, kl=sum(x['kl'] for x in rows) / n,
                n_sd=len(sd), agree_sd=sum(x['t'] == x['s'] for x in sd) / max(1, len(sd)), hid={i: sum(v) / len(v) for i, v in hid.items()})


# ------------------------------------------------------------------ evalkit (reporting only; never used for training or selection)
def eval_suites(m, suites, out, student=True, maxq=8, max_rows=12000, log=print, items=None):
    """every question of the given evalkit suites through the model; shared-prefix packing per item; appends {suite,id,q,T,probs} JSONL (resumable)"""
    import evalkit as EK
    by = collections.OrderedDict()
    for s, iid, q, st, spec in EK.all_question_items(suites):
        if items is not None and iid not in items: continue
        by.setdefault(iid, []).append((s, q, st, spec))
    done = set()
    if os.path.exists(out):
        for l in open(out): r = json.loads(l); done.add((r['id'], r['q']))
    t0 = time.time(); n = 0
    with open(out, 'a') as f, torch.no_grad():
        for iid, qs in by.items():
            qs = [x for x in qs if (iid, x[1]) not in done]
            if not qs: continue
            state = qs[0][2]
            for c0 in range(0, len(qs), maxq):
                ch = qs[c0:c0 + maxq]
                rq = m.prep(state, [x[3] for x in ch])
                groups = [list(range(len(ch)))]
                if len(rq['s']) + sum(len(x['q']) for x in rq['qs']) > max_rows: groups = [[j] for j in range(len(ch))]
                for g in groups:
                    rqg = dict(s=rq['s'], qs=[rq['qs'][j] for j in g])
                    (lp, _), _ = m.run(rqg, student=student, keep=())
                    for j, l_ in zip(g, lp):
                        p = l_.float().exp().tolist(); labs = rq['qs'][j]['labels']
                        f.write(json.dumps(dict(suite=ch[j][0], id=iid, q=ch[j][1], T=len(rq['s']) + len(rq['qs'][j]['q']),
                                                probs={lab: p[k] for k, lab in enumerate(labs)})) + '\n'); n += 1
            f.flush()
    log('eval_suites', suites, out, n, f'{time.time() - t0:.0f}s')
    return n


def jsonl_to_preds(path):
    preds = {}
    for l in open(path):
        r = json.loads(l); preds.setdefault(r['id'], {})[r['q']] = r['probs']
    return preds
