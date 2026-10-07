"""J13 token-adaptive width on full 24-layer hobson-v19 (merged LoRA), plain-torch layer runtime with an optional thin path.

Thin path for GEMM y = W x (W: N x K) on a token: y_thin = A_r (B_r x), A_r = Afull[:, :r], B_r = Bfull[:r]
  mode 'in' : Bfull = P^T (top input-PCA directions of x), Afull = W P          -> y = W P_r P_r^T x
  mode 'out': Bfull = U^T W, Afull = U (top principal directions of W x)       -> y = U_r U_r^T W x  (data-aware SVD)
Bases fitted on state tokens of train-split states (fit.py).  Cost r (K + N) instead of K N.

Plans per layer:
  None                                  dense
  {'mode','r','ti','fi'}                thin rows ti (state background), full rows fi (everything else)
  {'mode','w','asum','s0','ns'}         oracle gradient pass: rows [s0, ns) get y + sum_r alpha_r (thin_r(x) - y);
                                        w[t, j] = sum_{r > j} alpha_r[t]  (nested ranks), asum = sum_r alpha_r
"""
import os, sys, json, math, glob, time
sys.path[:0] = [os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/systems/g')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from strands_decider.modeling import masked_log_softmax

GEMMS = ('in', 'o', 'gu', 'd')
RMAX = 512
SINK = 4
FULL_ATTN = (3, 7, 11, 15, 19, 23)


def rms_zc(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)).to(x.dtype)


def conv_silu(x, w):  # x [T, C] bf16, w [C, 4] -> causal depthwise conv + silu, [T, C]
    T, C = x.shape
    y = F.conv1d(x.t()[None], w[:, None, :], padding=3, groups=C)[0, :, :T]
    return F.silu(y).t()


def gated_norm(o, z, w, eps):  # o, z [T*16, 128]
    of = o.float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    y = w * of.to(o.dtype)
    y = y * F.silu(z.float())
    return y.to(o.dtype)


def attn_prep(qkv, qn, kn, cos, sin, eps):
    T = qkv.shape[0]
    qg = qkv[:, :4096].reshape(T, 8, 512)
    q, gate = qg[..., :256], qg[..., 256:]
    k = qkv[:, 4096:4608].reshape(T, 2, 256)
    v = qkv[:, 4608:5120].reshape(T, 2, 256)
    q = rms_zc(q, qn, eps); k = rms_zc(k, kn, eps)

    def rope(x):
        xr, xp = x[..., :64], x[..., 64:]
        x1, x2 = xr[..., :32], xr[..., 32:]
        c = cos[:, None, :]; s = sin[:, None, :]
        rot = torch.cat([x1 * c[..., :32] - x2 * s[..., :32], x2 * c[..., 32:] + x1 * s[..., 32:]], -1)
        return torch.cat([rot, xp], -1)
    return rope(q), rope(k), v, torch.sigmoid(gate.reshape(T, 2048))


def layer_flops(K_N):  # MACs per token
    return sum(K * N for K, N in K_N)


class TW:
    def __init__(self, dev='cuda', bases=None, p=None, modes=('in', 'out')):
        from plib import P
        self.p = p if p is not None else P(); self.p.model.config.max_length = 16384
        for prm in self.p.model.parameters(): prm.requires_grad_(False)
        tm = self.p.tm; cfg = tm.config
        self.tok = self.p.tok; self.dev = dev
        self.eps = cfg.rms_norm_eps; self.types = cfg.layer_types; self.NL = cfg.num_hidden_layers
        with torch.no_grad():
            self.embed = tm.embed_tokens.weight
            self.norm_w = tm.norm.weight.float()
            self.L = []
            for i, Ly in enumerate(tm.layers):
                d = {'type': self.types[i], 'in_norm': Ly.input_layernorm.weight.float(), 'post_norm': Ly.post_attention_layernorm.weight.float()}
                m = Ly.mlp
                W = {'gu': torch.cat([m.gate_proj.weight, m.up_proj.weight], 0).contiguous(), 'd': m.down_proj.weight.contiguous()}
                d['I'] = m.gate_proj.weight.shape[0]
                if d['type'] == 'linear_attention':
                    a = Ly.linear_attn
                    W['in'] = torch.cat([a.in_proj_qkv.weight, a.in_proj_z.weight, a.in_proj_b.weight, a.in_proj_a.weight], 0).contiguous()
                    d['conv_w'] = a.conv1d.weight.squeeze(1).contiguous()
                    d['A_log'] = a.A_log.data; d['dt_bias'] = a.dt_bias.data.float()
                    d['gn_w'] = a.norm.weight.data; W['o'] = a.out_proj.weight.contiguous()
                else:
                    a = Ly.self_attn
                    W['in'] = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0).contiguous()
                    d['qn'] = a.q_norm.weight.float(); d['kn'] = a.k_norm.weight.float(); W['o'] = a.o_proj.weight.contiguous()
                d['W'] = W
                self.L.append(d)
            # free the HF copies of the weights we duplicated (Win / Wgu are new tensors; keep o / d which alias)
            for Ly in tm.layers:
                for mod in (getattr(Ly, 'linear_attn', None), getattr(Ly, 'self_attn', None)):
                    if mod is None: continue
                    for nm in ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a', 'q_proj', 'k_proj', 'v_proj'):
                        if hasattr(mod, nm): getattr(mod, nm).weight = None
                Ly.mlp.gate_proj.weight = None; Ly.mlp.up_proj.weight = None
            # weights built from inference-mode tensors cannot be saved for backward (oracle pass): clone those
            cl = lambda t: t.clone() if isinstance(t, torch.Tensor) and t.is_inference() else t
            for d in self.L:
                for kk in list(d):
                    if kk == 'W': d['W'] = {g: cl(v) for g, v in d['W'].items()}
                    else: d[kk] = cl(d[kk])
            self.embed = cl(self.embed); self.norm_w = cl(self.norm_w)
            for Ly in tm.layers:   # drop HF references to o/out/down weights (now cloned or aliased in self.L)
                Ly.mlp.down_proj.weight = None
                mod = getattr(Ly, 'linear_attn', None) or getattr(Ly, 'self_attn', None)
                for nm in ('out_proj', 'o_proj'):
                    if hasattr(mod, nm): getattr(mod, nm).weight = None
            hd = self.p.model.head
            for mod in hd.modules():
                for nm, prm in list(mod._parameters.items()):
                    if prm is not None: mod._parameters[nm] = torch.nn.Parameter(prm.detach().clone(), requires_grad=False)
            torch.cuda.empty_cache()
            rp = cfg.rope_parameters
            self.inv = 1.0 / (rp['rope_theta'] ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64))
        self.KN = [{g: (self.L[l]['W'][g].shape[1], self.L[l]['W'][g].shape[0]) for g in GEMMS} for l in range(self.NL)]
        self.dense_macs = [sum(K * N for K, N in self.KN[l].values()) for l in range(self.NL)]
        self.A = {}; self.B = {}
        if bases: self.load_bases(bases, modes)

    # ------------------------------------------------------------ bases
    def load_bases(self, path, modes=('in', 'out')):
        bz = torch.load(path, map_location='cpu')
        for mode in modes:
            if mode not in bz: continue
            self.A[mode] = []; self.B[mode] = []
            for l in range(self.NL):
                Al = {}; Bl = {}
                for g in GEMMS:
                    W = self.L[l]['W'][g]
                    V = bz[mode][l][g].to(self.dev, torch.float32)       # in: P [K, RMAX]; out: U [N, RMAX]
                    if mode == 'in':
                        Bl[g] = V.t().contiguous().to(torch.bfloat16)                       # [RMAX, K]
                        Al[g] = (W.float() @ V).contiguous().to(torch.bfloat16)             # [N, RMAX]
                    else:
                        Al[g] = V.contiguous().to(torch.bfloat16)                           # [N, RMAX]
                        Bl[g] = (V.t() @ W.float()).contiguous().to(torch.bfloat16)         # [RMAX, K]
                    del V
                self.A[mode].append(Al); self.B[mode].append(Bl)
        torch.cuda.empty_cache()

    def thin_macs(self, l, r):
        return sum(r * (K + N) for K, N in self.KN[l].values())

    # ------------------------------------------------------------ prompt
    def prep(self, item, qn):
        pr = self.p.prep(item['state'], item['questions'][qn])
        pr['ids'] = pr['s'] + pr['q']
        pr['opt_abs'] = [pr['q0'] + o for o in pr['opt']]
        return pr

    def embed_rope(self, ids):
        t = torch.tensor(ids, device=self.dev)
        x = F.embedding(t, self.embed)
        T = len(ids)
        pos = torch.arange(T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        return x, fr.cos().to(x.dtype), fr.sin().to(x.dtype)

    # ------------------------------------------------------------ core
    def _gemm(self, l, g, a, pl):
        W = self.L[l]['W'][g]
        if pl is None:
            return a @ W.t()
        mode = pl['mode']
        if 'fac' in pl: Af, Bf = pl['fac'][g]                     # trainable factors (heal)
        else: Af, Bf = self.A[mode][l][g], self.B[mode][l][g]
        if 'w' in pl:   # oracle gradient pass
            y = a @ W.t()
            s0, ns = pl['s0'], pl['ns']; rm = pl['w'].shape[1]
            xs = a[s0:ns]; ys = y[s0:ns]
            c = xs @ Bf[:rm].t()
            ys2 = ys + (pl['w'] * c) @ Af[:, :rm].t() - pl['asum'][:, None] * ys
            return torch.cat([y[:s0], ys2, y[ns:]], 0)
        r, ti, fi = pl['r'], pl['ti'], pl['fi']
        if ti.numel() == 0:
            return a @ W.t()
        y = a.new_empty(a.shape[0], W.shape[0])
        y[fi] = a[fi] @ W.t()
        y[ti] = (a[ti] @ Bf[:r].t()) @ Af[:, :r].t()
        return y

    def layer(self, l, x, cos, sin, pl=None, collect=None):
        d = self.L[l]; T = x.shape[0]; eps = self.eps
        h = rms_zc(x, d['in_norm'], eps)
        if collect is not None: collect(l, 'in', h)
        proj = self._gemm(l, 'in', h, pl)
        if d['type'] == 'linear_attention':
            z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b)
            gg = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            qkv = conv_silu(proj[:, :6144], d['conv_w'])
            q, k, v = qkv.split(2048, dim=-1)
            o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128),
                                          gg.reshape(1, T, 16), beta.reshape(1, T, 16), use_qk_l2norm_in_kernel=True)
            o = gated_norm(o.reshape(-1, 128), z.reshape(-1, 128), d['gn_w'], eps).reshape(T, 2048)
        else:
            q, k, v, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, eps)
            qh = q.reshape(1, T, 8, 256).transpose(1, 2); kh = k.reshape(1, T, 2, 256).transpose(1, 2); vh = v.reshape(1, T, 2, 256).transpose(1, 2)
            o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
            o = o.transpose(1, 2).reshape(T, 2048) * gate
        if collect is not None: collect(l, 'o', o)
        x = x + self._gemm(l, 'o', o, pl)
        h2 = rms_zc(x, d['post_norm'], eps)
        if collect is not None: collect(l, 'gu', h2)
        gu = self._gemm(l, 'gu', h2, pl)
        I = d['I']
        m = F.silu(gu[:, :I]) * gu[:, I:]
        if collect is not None: collect(l, 'd', m)
        x = x + self._gemm(l, 'd', m, pl)
        return x

    def run(self, x, cos, sin, l0=0, l1=None, plans=None, keep=None, collect=None):
        """layers l0..l1-1; plans: {layer: plan}; keep: set of layer indices whose INPUT is returned in dict"""
        l1 = self.NL if l1 is None else l1
        kept = {}
        for l in range(l0, l1):
            if keep is not None and l in keep: kept[l] = x
            x = self.layer(l, x, cos, sin, (plans or {}).get(l), collect)
        return x, kept

    def head(self, x, pr):
        rows = pr['opt_abs'] + [x.shape[0] - 1]
        h = rms_zc(x[rows], self.norm_w, self.eps)
        pooled = h[-1:].float(); options = h[:-1].float()[None]
        rq = pr['rq']
        logits = self.p.model.head(pooled, options) / self.p.temp_for(rq.kind)
        lp = masked_log_softmax(logits, torch.tensor([rq.n_slots], device=self.dev))
        return lp[0, :rq.n_slots]

    def probdict(self, pr, lp):
        return {lab: float(v) for lab, v in zip(pr['rq'].slot_labels, lp.exp().tolist())}

    # ------------------------------------------------------------ plans
    def thin_plan(self, mode, r, thin_rows, T):
        ti = torch.as_tensor(sorted(thin_rows), dtype=torch.long, device=self.dev)
        mask = torch.ones(T, dtype=torch.bool, device=self.dev)
        if ti.numel(): mask[ti] = False
        fi = mask.nonzero().squeeze(1)
        return {'mode': mode, 'r': r, 'ti': ti, 'fi': fi}

    def flops_ratio(self, T, n_thin, k, r, segs=None):
        dense = T * sum(self.dense_macs)
        if r is None or n_thin == 0: return 1.0
        segs = segs or [(k, r)]
        cost = 0
        for l in range(self.NL):
            rl = None
            for st, rr in segs:
                if l >= st: rl = rr
            if rl is None: cost += T * self.dense_macs[l]
            else: cost += (T - n_thin) * self.dense_macs[l] + n_thin * self.thin_macs(l, rl)
        return cost / dense

    # ------------------------------------------------------------ oracle: d margin / d alpha at the dense point
    def oracle_pass(self, pr, mode, ranks=(128, 256, 512), k0=4, ks=(4, 8), ckpt_T=1800):
        """returns dense log-probs, per-token scores {(k, r): float tensor [q0]} (rows < SINK zero), x at layer inputs {k: tensor}"""
        ids = pr['ids']; q0 = pr['q0']; T = len(ids)
        with torch.no_grad():
            x, cos, sin = self.embed_rope(ids)
            x, kept = self.run(x, cos, sin, 0, k0, keep={k0} if k0 in ks else set())
        s0, ns = SINK, q0
        Ts = max(ns - s0, 0)
        rs = sorted(ranks)
        alphas = {l: torch.zeros(len(rs), Ts, device=self.dev, dtype=torch.float32, requires_grad=True) for l in range(k0, self.NL)}
        hk = {k0: x.detach()}

        def mkplan(al):
            # w[t, j] = sum over ranks r_i > j of alpha_i[t]
            cols = []; prev = 0
            for i, r in enumerate(rs):
                cols.append(al[i:].sum(0)[:, None].expand(-1, r - prev)); prev = r
            w = torch.cat(cols, 1).to(torch.bfloat16)
            return {'mode': mode, 'w': w, 'asum': al.sum(0).to(torch.bfloat16), 's0': s0, 'ns': ns}

        def lay(l, xx, al):  # al passed so checkpoint sees it as an input
            return self.layer(l, xx, cos, sin, mkplan(al))

        with torch.enable_grad():
            xx = x.detach()
            for l in range(k0, self.NL):
                if l in ks and l != k0: hk[l] = xx.detach()
                if T > ckpt_T:
                    xx = torch.utils.checkpoint.checkpoint(lay, l, xx, alphas[l], use_reentrant=False)
                else:
                    xx = lay(l, xx, alphas[l])
            lp = self.head(xx, pr)
            a = int(lp.argmax())
            others = torch.cat([lp[:a], lp[a + 1:]])
            Fm = lp[a] - torch.logsumexp(others, 0)
            Fm.backward()
        lp = lp.detach()
        sc = {}
        for k in ks:
            for i, r in enumerate(rs):
                g = torch.zeros(q0, device=self.dev)
                acc = sum(alphas[l].grad[i] for l in range(k, self.NL))
                g[s0:ns] = acc
                sc[(k, r)] = g
        del alphas
        return lp, sc, hk, (cos, sin)


    def refine_pass(self, pr, mode, k, r, thin_rows, hk, cos, sin, p_dense, ckpt_T=1800):
        """gradient of KL(p_dense || p_alpha) w.r.t. per-token alpha at the ROUTED point (alpha = 1 on thin rows, 0 elsewhere),
        thin path in layers >= k.  Positive = thinning that token (more) increases the divergence.  Returns [q0] tensor and lp at the routed point."""
        q0 = pr['q0']; T = len(pr['ids']); s0, ns = SINK, q0
        a0 = torch.zeros(ns - s0, device=self.dev)
        idx = torch.as_tensor([t - s0 for t in thin_rows if t >= s0], dtype=torch.long, device=self.dev)
        if idx.numel(): a0[idx] = 1.0
        al = a0.clone().requires_grad_(True)
        def plan():
            return {'mode': mode, 'w': al.to(torch.bfloat16)[:, None].expand(-1, r), 'asum': al.to(torch.bfloat16), 's0': s0, 'ns': ns}
        def lay(l, xx, aa):
            return self.layer(l, xx, cos, sin, plan())
        with torch.enable_grad():
            xx = hk[k].detach()
            for l in range(k, self.NL):
                xx = torch.utils.checkpoint.checkpoint(lay, l, xx, al, use_reentrant=False) if T > ckpt_T else lay(l, xx, al)
            lp = self.head(xx, pr)
            kl = (p_dense.exp() * (p_dense - lp)).sum()
            kl.backward()
        g = torch.zeros(q0, device=self.dev); g[s0:ns] = al.grad
        return g, lp.detach()


def chunks_of(q0, C=32):
    return [(s, min(s + C, q0)) for s in range(0, q0, C)]


def select_chunks(scores_chunk, f, rng=None):
    n = len(scores_chunk); m = int(math.floor(f * n + 0.5))
    if m <= 0: return []
    if rng is not None:
        idx = list(range(n)); rng.shuffle(idx); return sorted(idx[:m])
    order = sorted(range(n), key=lambda i: -scores_chunk[i])
    return sorted(order[:m])
