"""G2 library: hobson-v19 (merged LoRA, lean fused weights, fla kernels) with PER-TOKEN width and precision classes (emulation, inference only).

Per-token class c (rowcls[t]) has a spec: dict(mlp=w, heads=w, prec=p, drop=False, lo=0, hi=23). Class 0 = exact.
  mlp=w     : the token's MLP uses only the w*6144 most important neurons (MatFormer-style nesting after a fixed per-layer permutation).
  heads=w   : GDN layers: the token computes only the w*16 most important heads (q,k,v,z,a,b rows of in_proj; the skipped heads see beta=0 and no decay,
              i.e. the token is invisible to them, and its output channels for them are zero -> out_proj uses only w*2048 input columns).
              Attention layers: the token's query heads outside the leading w*8 are zero (o_proj input columns); its K/V are computed in full.
  drop=True : width 0 everywhere and removed from attention keys (true input-level removal, the selection comparator).
  prec      : 'bf16' | 'w8a8' (per-channel W, per-token A, RTN) | 'w8a8h' / 'w4a4h' / 'w4a8h' (same after a block Hadamard rotation of the GEMM input, QuaRot-style).
              Integer-exact emulation: integer codes are carried in bf16 (exact), products accumulate in fp32, then x per-token scale x per-channel scale.
  lo/hi     : layers where the class applies (outside: exact).
Question rows are always class 0. Signals for assigning state rows to classes are computed by the caller (g2map.py).
"""
import os, sys, json, math, glob
sys.path[:0] = [os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv

FULL_ATTN = (3, 7, 11, 15, 19, 23)


def nrm(x, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rms_zc(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)).to(x.dtype)


def paley12(dev):
    q = 11; sq = {(x * x) % q for x in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    H = torch.eye(12)
    H[0, 1:] += 1; H[1:, 0] -= 1
    for i in range(q):
        for j in range(q): H[1 + i, 1 + j] += chi(j - i)
    return (H / math.sqrt(12)).to(dev)


def hadamard(n, dev):
    H = torch.ones(1, 1, device=dev, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


class G2:
    def __init__(self, maxlen=16384):
        from plib import P
        import lean as LN
        p = P(); p.model.config.max_length = maxlen
        self.p = p
        lean = LN.Lean(p.tm)
        self.L = lean.layers; self.embed = lean.embed; self.norm_w = lean.norm_w; self.inv = lean.inv; self.eps = lean.eps
        self.dev = lean.dev
        self.head = p.model.head
        self._rot = {}
        self.qcache = {}
        for d in self.L:
            d['in1'] = (1.0 + d['in_norm']).float(); d['post1'] = (1.0 + d['post_norm']).float()
        self.rank_mlp = None; self.rank_gdn = None; self.rank_att = None
        self.tidx = []; cg = ca = 0
        for d in self.L:
            if d['type'] == 'linear_attention': self.tidx.append(cg); cg += 1
            else: self.tidx.append(ca); ca += 1
        self._calib = None
        try:
            a = torch.ones(16, 16, device=self.dev, dtype=torch.bfloat16); torch.mm(a, a, out_dtype=torch.float32); G2._mm_out = True
        except Exception: G2._mm_out = False

    # ------------------------------------------------------------------ quantization
    # prec codes: 'p8'  plain W8A8 (per-channel W, per-token A, RTN, activations = the weighted normed input as in d2)
    #             'r8' / 'r4' / 'r48': QuaRot-style rotated W8A8 / W4A4 / W4A8: RMSNorm weights folded into Win/Wgu (W' = W diag(1+w)), the GEMM input
    #             (unweighted normed residual for Win/Wgu = an offline residual rotation; GDN/attention output for Wo and the MLP hidden for Wd = online)
    #             rotated by a randomized Kronecker-Hadamard (2048 = H32 x H64, 6144 = H12(Paley) x H16 x H32); RTN, absmax scales.
    #             suffix 'c' = activation clip ratio 0.9 (4-bit only).
    def _rot_mats(self, K):
        if K not in self._rot:
            g = torch.Generator(device='cpu'); g.manual_seed(1234 + K)
            sign = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).float().to(self.dev)
            facs = [32, 64] if K == 2048 else [12, 16, 32]
            mats = [paley12(self.dev) if n == 12 else hadamard(n, self.dev) for n in facs]
            self._rot[K] = (sign, facs, mats)
        return self._rot[K]

    def rot_act(self, x):   # activations: same rotation in fp16 (as an online fast Hadamard kernel would compute it)
        K = x.shape[-1]
        if ('d', K) not in self._rot:
            sign, facs, mats = self._rot_mats(K)
            if K == 2048: self._rot[('d', K)] = (sign.half(), None, torch.kron(mats[0], mats[1]).half())
            else: self._rot[('d', K)] = (sign.half(), mats[0].half(), torch.kron(mats[1], mats[2]).half())
        sign, H12, Hb = self._rot[('d', K)]
        xs = x.half() * sign
        if H12 is None: return xs @ Hb
        y = torch.einsum('mjk,ji->mik', xs.reshape(x.shape[0], 12, 512), H12)
        return (y @ Hb).reshape(x.shape)

    def rot(self, x):   # x [M, K] -> (x * sign) @ (H_a x H_b x ...), computed along the Kronecker factors in fp32 (weights, offline)
        sign, facs, mats = self._rot_mats(x.shape[-1])
        y = (x.float() * sign).reshape(x.shape[0], *facs)
        for ax, Hm in enumerate(mats):
            y = torch.movedim(torch.tensordot(y, Hm, dims=([ax + 1], [0])), -1, ax + 1)
        return y.reshape(x.shape)

    @staticmethod
    def _bits(prec):
        if prec.startswith('r48'): return 4, 8
        if prec.startswith('r4'): return 4, 4
        return 8, 8

    def qweight(self, i, name, prec):
        key = (i, name, prec.rstrip('c'))
        if key not in self.qcache:
            d = self.L[i]; W = d[name].float()
            wb, _ = self._bits(prec); qmax = 7.0 if wb == 4 else 127.0
            if prec.startswith('r'):
                if name == 'Win': W = W * d['in1'][None, :]
                if name == 'Wgu': W = W * d['post1'][None, :]
                W = self.rot(W)
            s = W.abs().amax(1).clamp_min(1e-8) / qmax
            if wb == 4:   # per-channel MSE clip search (RTN grid), as in QuaRot's RTN baseline with clipping
                best = None
                for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
                    sr = s * r; err = ((torch.round(W / sr[:, None]).clamp(-qmax, qmax) * sr[:, None] - W) ** 2).sum(1)
                    if best is None: best = (err, sr.clone())
                    else:
                        m = err < best[0]; best[0][m] = err[m]; best[1][m] = sr[m]
                s = best[1]
            q = torch.round(W / s[:, None]).clamp(-qmax, qmax)
            self.qcache[key] = (q.to(torch.bfloat16).contiguous(), s.contiguous())
        return self.qcache[key]

    def qact(self, x, prec):
        _, ab = self._bits(prec); qmax = 7.0 if ab == 4 else 127.0
        xf = self.rot_act(x).float() if prec.startswith('r') else x.float()
        s = xf.abs().amax(1).clamp_min(1e-8) / qmax
        if prec.endswith('c') and ab == 4: s = s * 0.9
        return torch.round(xf / s[:, None]).clamp(-qmax, qmax).to(torch.bfloat16), s

    def qlin(self, x, i, name, prec):
        q, s = self.qact(x, prec)
        Wq, sw = self.qweight(i, name, prec)
        y = torch.mm(q, Wq.t(), out_dtype=torch.float32) if self._mm_out else (q @ Wq.t()).float()
        return (y * s[:, None] * sw[None, :]).to(torch.bfloat16)

    _mm_out = None

    def lin(self, x, i, name, groups, xn=None):
        """groups: list of (row_idx LongTensor, prec) for rows NOT in bf16. xn: the unweighted normed input (for rotated codes with folded norms)"""
        y = x @ self.L[i][name].t()
        for rows, prec in groups:
            if rows.numel() == 0 or prec == 'bf16': continue
            src = xn if (prec.startswith('r') and xn is not None) else x
            y[rows] = self.qlin(src[rows], i, name, prec)
        return y

    # ------------------------------------------------------------------ calibration of nesting order
    @torch.no_grad()
    def calibrate(self, seqs, path=None):
        """importance of MLP neurons, GDN heads and attention q-heads from dense passes over seqs (lists of ids); state rows only."""
        acc_m = [torch.zeros(self.L[i]['I'], device=self.dev) for i in range(24)]
        acc_h = [torch.zeros(16 if self.L[i]['type'] == 'linear_attention' else 8, device=self.dev) for i in range(24)]
        n = 0
        for ids in seqs:
            self._calib = (acc_m, acc_h)
            self.forward(ids, None)
            n += len(ids)
        self._calib = None
        self.rank_mlp, self.rank_gdn, self.rank_att = [], [], []
        for i in range(24):
            Wd = self.L[i]['Wd'].float()
            score_m = acc_m[i] / n * Wd.norm(dim=0)
            self.rank_mlp.append(torch.argsort(torch.argsort(score_m, descending=True)))     # rank 0 = most important
            r = torch.argsort(torch.argsort(acc_h[i] / n, descending=True))
            (self.rank_gdn if self.L[i]['type'] == 'linear_attention' else self.rank_att).append(r)
        if path: torch.save(dict(m=self.rank_mlp, g=self.rank_gdn, a=self.rank_att), path)

    def detach_inference(self):
        """clone every weight out of inference mode (needed for autograd through the frozen model) and free the HF copy"""
        import gc
        for d in self.L:
            for k, v in list(d.items()):
                if torch.is_tensor(v): d[k] = v.clone()
        self.embed = self.embed.clone(); self.norm_w = self.norm_w.clone(); self.inv = self.inv.clone()
        h0 = self.head
        h1 = type(h0)(h0.q.in_features, h0.q.out_features).to(self.dev)
        h1.load_state_dict({k: v.clone() for k, v in h0.state_dict().items()}); h1.to(next(h0.parameters()).dtype)
        self.head = h1
        self.p.model.torso = None; self.p.tm = None
        gc.collect(); torch.cuda.empty_cache()

    def merge_lora(self, path):
        sd = torch.load(path, map_location=self.dev)
        for i in range(24):
            for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                A, B = sd[f'{i}.{k}.A'].float(), sd[f'{i}.{k}.B'].float()
                self.L[i][k] = (self.L[i][k].float() + B @ A).to(torch.bfloat16).contiguous()
        self.qcache = {}

    def load_ranks(self, path):
        d = torch.load(path, map_location=self.dev)
        self.rank_mlp, self.rank_gdn, self.rank_att = d['m'], d['g'], d['a']

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, ids, ctl=None, stop=None, capture=(), q0=None, keep_x=(), mlp_mask=None):
        """ids: list[int]. ctl: dict(rowcls LongTensor [T], classes list of spec dicts) or None (exact).
        stop: run layers < stop only. capture: attention layers whose question->state attention to return (needs q0 = number of state rows).
        Returns (h_final_normed [T,2048] or None, {layer: attn over state rows [q0]})."""
        dev = self.dev; T = len(ids)
        ids_t = torch.tensor(ids, device=dev)
        x = F.embedding(ids_t, self.embed)
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
        h = rms_zc(x, self.L[0]['in_norm'], self.eps)
        caps = {}
        cls = ctl['rowcls'] if ctl else None
        specs = ctl['classes'] if ctl else None
        nL = 24 if stop is None else stop
        calib = self._calib
        for i in range(nL):
            d = self.L[i]; gdn = d['type'] == 'linear_attention'
            # per-layer row groups
            pgroups = []; wm = None; wh = None; alive = None
            if ctl is not None:
                keepm = torch.ones(T, device=dev); keeph = torch.ones(T, device=dev); dropr = torch.zeros(T, dtype=torch.bool, device=dev)
                for c, sp in enumerate(specs):
                    if c == 0 or not (sp.get('lo', 0) <= i <= sp.get('hi', 23)): continue
                    rows = torch.nonzero(cls == c).flatten()
                    if rows.numel() == 0: continue
                    if sp.get('prec', 'bf16') != 'bf16': pgroups.append((rows, sp['prec']))
                    if sp.get('drop'): dropr[rows] = True; keepm[rows] = 0; keeph[rows] = 0
                    else:
                        keepm[rows] = sp.get('mlp', 1.0); keeph[rows] = sp.get('heads', 1.0)
                if (keepm < 1).any():
                    wm = (self.rank_mlp[i][None, :].float() < (keepm[:, None] * d['I'])).to(x.dtype)            # [T, I]
                if (keeph < 1).any():
                    rk = (self.rank_gdn if gdn else self.rank_att)[self.tidx[i]]
                    nh = 16 if gdn else 8
                    wh = (rk[None, :].float() < (keeph[:, None] * nh)).to(x.dtype)                                  # [T, nh]
                if dropr.any(): alive = ~dropr
            proj = self.lin(h, i, 'Win', pgroups, nrm(x, self.eps) if pgroups else None)
            if gdn:
                qkv = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
                beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
                if wh is not None:
                    qkv = (qkv.reshape(T, 3, 16, 128) * wh[:, None, :, None]).reshape(T, 6144)
                    beta = beta * wh.float(); g = g * wh.float()
                qkv = fla_conv(qkv[None].contiguous(), d['conv_w'], None, activation='silu')
                qkv = qkv[0] if isinstance(qkv, tuple) else qkv
                q, k, v = qkv.split(2048, dim=-1)
                o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), g[None],
                                              beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
                of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)
                o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(h.dtype).reshape(T, 2048)
                if wh is not None: o = (o.reshape(T, 16, 128) * wh[..., None]).reshape(T, 2048)
                if calib is not None:
                    Wo = d['Wo'].float()
                    contrib = torch.stack([(o[:, hh * 128:(hh + 1) * 128].float() @ Wo[:, hh * 128:(hh + 1) * 128].t()).norm(dim=-1).sum() for hh in range(16)])
                    calib[1][i] += contrib
            else:
                qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
                kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
                qh = rms_zc(qh, d['qn'], self.eps); kk = rms_zc(kk, d['kn'], self.eps)
                def rope(t):
                    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
                    c = cos[:, None, :]; s = sin[:, None, :]
                    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s[..., :32], x2 * c[..., 32:] + x1 * s[..., 32:]], -1), xp], -1)
                qh, kk = rope(qh), rope(kk)
                if i in capture:
                    nq = T - q0
                    lg = torch.einsum('qhd,khd->hqk', qh[q0:].float(), kk.repeat_interleave(4, dim=1).float()) / 16.0
                    lg = lg.masked_fill(torch.arange(T, device=dev)[None, None, :] > (q0 + torch.arange(nq, device=dev))[None, :, None], float('-inf'))
                    if alive is not None: lg = lg.masked_fill(~alive[None, None, :], float('-inf'))
                    caps[i] = torch.softmax(lg, -1)[..., :q0].mean(dim=(0, 1))
                if alive is None:
                    o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
                else:
                    msk = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev)) & alive[None, :]
                    msk |= torch.eye(T, dtype=torch.bool, device=dev)
                    o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], attn_mask=msk[None, None], enable_gqa=True)
                o = o[0].transpose(0, 1) * torch.sigmoid(gate)                       # [T, 8, 256]
                if calib is not None:
                    Wo = d['Wo'].float()
                    calib[1][i] += torch.stack([(o[:, hh].float() @ Wo[:, hh * 256:(hh + 1) * 256].t()).norm(dim=-1).sum() for hh in range(8)])
                if wh is not None: o = o * wh[..., None]
                o = o.reshape(T, 2048)
            dout = self.lin(o, i, 'Wo', pgroups)
            if alive is not None: dout = dout * alive[:, None].to(dout.dtype)
            x = x + dout
            h2 = rms_zc(x, d['post_norm'], self.eps)
            gu = self.lin(h2, i, 'Wgu', pgroups, nrm(x, self.eps) if pgroups else None); I = d['I']
            m = F.silu(gu[:, :I]) * gu[:, I:]
            if calib is not None: calib[0][i] += m.float().abs().sum(0)
            if wm is not None: m = m * wm
            if mlp_mask is not None and mlp_mask.get(i) is not None: m = m * mlp_mask[i][None, :]
            x = x + self.lin(m, i, 'Wd', pgroups)
            if i in keep_x: caps[('x', i)] = x
            nw = self.L[i + 1]['in_norm'] if i + 1 < 24 else self.norm_w
            h = rms_zc(x, nw, self.eps)
        if stop is not None: return None, caps
        return h, caps

    # ------------------------------------------------------------------ decisions
    def prep(self, state, qd):
        return self.p.prep(state, qd)

    @torch.no_grad()
    def probs(self, h, pr):
        T = h.shape[0]
        opt_abs = [pr['q0'] + o for o in pr['opt']]
        pooled = h[T - 1].float()[None]; options = h[torch.tensor(opt_abs, device=self.dev)].float()[None]
        logits = self.head(pooled, options) / self.p.temp_for(pr['rq'].kind)
        from strands_decider.modeling import masked_log_softmax
        lp = masked_log_softmax(logits, torch.tensor([pr['rq'].n_slots], device=self.dev))
        return lp.exp()[0, :pr['rq'].n_slots].tolist()


def assign(score, q0, T, plan, sink=4):
    """score [q0] (higher = more relevant). plan: list of (class, fraction) in priority order (fractions of q0; the last class takes the rest).
    Returns rowcls LongTensor [T] (question rows and sinks = 0)."""
    dev = score.device
    cls = torch.zeros(T, dtype=torch.long, device=dev)
    sc = score.clone().float(); sc[:min(sink, q0)] = float('inf')
    order = torch.argsort(sc, descending=True)
    start = 0
    for j, (c, f) in enumerate(plan):
        n = q0 - start if j == len(plan) - 1 else int(round(f * q0))
        cls[order[start:start + n]] = c; start += n
    cls[:min(sink, q0)] = 0
    return cls


def cost(plan_specs, layers=24):
    """fraction of dense linear FLOPs per state token. plan_specs: list of (fraction, spec). GDN layer: in_proj 29%, out 7%, MLP 64% (attention layers similar)."""
    tot = 0.0
    for f, sp in plan_specs:
        if sp.get('drop'): w = 0.0
        else:
            hw, mw = sp.get('heads', 1.0), sp.get('mlp', 1.0)
            w = 0.36 * hw + 0.64 * mw
        pr_ = sp.get('prec', 'bf16'); pf = 1.0 if pr_ == 'bf16' else (0.25 if pr_.startswith('r4') and not pr_.startswith('r48') else 0.5)
        lo, hi = sp.get('lo', 0), sp.get('hi', 23); nl = hi - lo + 1
        tot += f * (w * pf * nl + (layers - nl)) / layers
    return tot
