"""H1 library: hobson-v19 as a UNIFORM low-bit model (every row, state and question, goes through the same quantized GEMMs).

Built on G2's g2lib.G2 (weights, prep, head). Per-GEMM precision map  self.P[(layer, name)] in
   'bf16' | 'w8a8' | 'w4a4' | 'w4a8'      name in Win (GDN in_proj qkvzba / attn q+gate,k,v), Wo (out_proj / o_proj), Wgu (gate+up), Wd (down)
Quantized GEMM = QuaRot layout:
   Win / Wgu : input = RMSNorm(x) WITHOUT gain, rotated by R1 (= offline residual rotation: the residual stream is stored as xR1, the norm
               gains are folded into W, W' = (W diag(1+g)) R1).  Win's GDN b/a rows (beta / decay gates, 32 rows) optionally stay bf16 (opt ba16).
   Wo        : input = mixer output, online Hadamard R2 (2048 = H32 x H64, or per-head H128 for GDN with opt 'ohead').
   Wd        : input = SiLU(gate)*up, online Hadamard R4 (6144 = H12(Paley) x H16 x H32).
   Rotations: randomized (random signs) Kronecker Hadamards, seed opt 'rseed'.
   Activations: symmetric per-token absmax, int8 (127) or int4 (7, clip ratio opt 'aclip4'), dynamic.
   Weights: symmetric per-output-channel; RTN with MSE clip search (4-bit) / absmax (8-bit), or GPTQ (opt 'wq'='gptq', act-order, Hessians of
            the rotated dense inputs from train-split calibration states).
   GEMM: exact integer math (torch._int_mm, int8 x int8 -> int32; 4-bit codes live in int8 storage), then * s_token * s_channel -> bf16.
Everything else (embedding, norms, conv, l2norm, gates, delta rule, attention softmax, pointer head) is bf16/fp32 as in the dense runtime.
"""
import os, sys, math, json
sys.path[:0] = [os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
import g2lib as GL
from g2lib import nrm, rms_zc, paley12, hadamard
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv

GEMMS = ('Win', 'Wo', 'Wgu', 'Wd')
ATT = (3, 7, 11, 15, 19, 23)
QMAX = {4: 7.0, 8: 127.0}


def bits(prec):
    return {'w8a8': (8, 8), 'w4a4': (4, 4), 'w4a8': (4, 8), 'w8a4': (8, 4), 'w8a16': (8, 16), 'w4a16': (4, 16), 'w16a8': (16, 8), 'w16a4': (16, 4), 'w16a16': (16, 16)}[prec]


def flops_share(i, k):
    """dense GEMM MACs per token for (layer, gemm)"""
    N = {'Win': 8224 if i not in ATT else 5120, 'Wo': 2048, 'Wgu': 12288, 'Wd': 2048}[k]
    K = 6144 if k == 'Wd' else 2048
    return N * K


class Rot:
    """randomized Kronecker Hadamard acting on the last dim: x -> (x * sign) @ (H_a (x) H_b (x) ...)"""
    def __init__(self, K, seed, dev, block=None):
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.K = K; self.block = block
        self.sign = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).float().to(dev)
        if block:                                     # block-diagonal Hadamard (per head)
            self.facs = [K // block, block]; self.mats = [None, hadamard(block, dev)]
        else:
            self.facs = [32, 64] if K == 2048 else [12, 16, 32]
            self.mats = [paley12(dev) if n == 12 else hadamard(n, dev) for n in self.facs]

    def __call__(self, x):
        sh = x.shape
        y = (x.float() * self.sign).reshape(-1, *self.facs)
        for ax, Hm in enumerate(self.mats):
            if Hm is None: continue
            y = torch.movedim(torch.tensordot(y, Hm, dims=([ax + 1], [0])), -1, ax + 1)
        return y.reshape(sh)

    def inv(self, x):                                  # x -> x R^T
        sh = x.shape
        y = x.float().reshape(-1, *self.facs)
        for ax, Hm in enumerate(self.mats):
            if Hm is None: continue
            y = torch.movedim(torch.tensordot(y, Hm.t(), dims=([ax + 1], [0])), -1, ax + 1)
        return (y.reshape(sh) * self.sign)


def rtn_scales(W, qmax, search):
    s = W.abs().amax(1).clamp_min(1e-8) / qmax
    if not search: return s
    best_e = None; best = s.clone()
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):
        sr = s * r
        e = ((torch.round(W / sr[:, None]).clamp(-qmax, qmax) * sr[:, None] - W) ** 2).sum(1)
        if best_e is None: best_e = e
        else:
            m = e < best_e; best_e = torch.where(m, e, best_e); best = torch.where(m, sr, best)
    return best


def gptq(W, H, qmax, s=None, blocksize=128, percdamp=0.01, actorder=True):
    """GPTQ (Frantar et al.) with per-channel symmetric scales s [N]. W [N,K] fp32, H [K,K] fp32. Returns integer codes [N,K] (float) and s."""
    W = W.clone().float(); N, K = W.shape; H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    if s is None: s = rtn_scales(W, qmax, qmax < 100)
    perm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True); W = W[:, perm]; H = H[perm][:, perm]
    damp = percdamp * torch.mean(torch.diag(H))
    H[range(K), range(K)] += damp
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    sc = s[:, None]
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); n = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(n):
            w = W1[:, j]; d = Hi[j, j]
            q = torch.round(w / s).clamp(-qmax, qmax)
            Q1[:, j] = q
            e = (w - q * s) / d
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        inv = torch.argsort(perm); Q = Q[:, inv]
    return Q, s


class H1(GL.G2):
    def __init__(self, maxlen=16384, **opt):
        super().__init__(maxlen)
        self.opt = dict(ba16=False, aclip4=0.9, aclip8=1.0, wq='rtn', rseed=1234, ohead=False, rout=True, hdir=os.path.expanduser('~/work/h1/hess'))
        self.opt.update(opt)
        self.P = {}
        self.Qw = {}
        self.cap = None          # calibration capture: set of (i,k) -> accumulate Hessians into self.Hacc
        self.Hacc = {}
        self.wclip = {}          # optional per-(i,k) override of weight scales
        self.LR = None; self.LRtag = None   # rotated-space LoRA from h1qat: {(i,k): (A [r,K], B [N,r]) fp32}
        self.mk_rots()
        try:
            a = torch.randint(-5, 5, (32, 64), device=self.dev, dtype=torch.int8); b = torch.randint(-5, 5, (48, 64), device=self.dev, dtype=torch.int8)
            r = torch._int_mm(a, b.t()); ok = torch.equal(r, (a.float() @ b.float().t()).to(torch.int32)); H1._intmm = ok
        except Exception as e:
            print('int_mm unavailable', e); H1._intmm = False

    _intmm = False

    def mk_rots(self):
        self._rots = {}

    def rots(self):
        sd = self.opt['rseed']
        if sd not in self._rots:
            self._rots[sd] = dict(R1=Rot(2048, sd, self.dev), R2=Rot(2048, sd + 1, self.dev), R2h=Rot(2048, sd + 1, self.dev, block=128),
                                  R2a=Rot(2048, sd + 1, self.dev, block=256), R4=Rot(6144, sd + 2, self.dev))
        return self._rots[sd]

    def rot_for(self, i, k):
        R = self.rots()
        if k in ('Win', 'Wgu'): return R['R1']
        if k == 'Wd': return R['R4']
        if self.opt['ohead']: return R['R2a'] if i in ATT else R['R2h']
        return R['R2']

    # ------------------------------------------------------------- configuration
    def set_prec(self, P):
        """P: dict (i,k) -> prec, missing = bf16. Quantized weights are built lazily and cached per (i,k,prec,wq)."""
        self.P = {key: v for key, v in P.items() if v != 'bf16'}

    def uniform(self, prec, except_=None):
        P = {(i, k): prec for i in range(24) for k in GEMMS}
        for key, v in (except_ or {}).items(): P[key] = v
        return P

    def wfold(self, i, k):
        d = self.L[i]; W = d[k].float()
        if k == 'Win': W = W * d['in1'][None, :]
        if k == 'Wgu': W = W * d['post1'][None, :]
        if k == 'Win' and self.opt['ba16'] and d['type'] == 'linear_attention': W = W[:8192]
        W = self.rot_for(i, k)(W)                          # input side: W R_in
        if k in ('Wo', 'Wd') and self.opt['rout']:         # output side (residual writer): R1^T W R_in, as stored in the rotated runtime
            W = self.rots()['R1'](W.t().contiguous()).t().contiguous()
        return W

    def hess(self, i, k):
        H0 = torch.load(f"{self.opt['hdir']}/H_{i}_{k}.pt", map_location=self.dev).float()
        R = self.rot_for(i, k)
        return R(R(H0).t().contiguous())                 # R^T H0 R  (Rot applies x -> x R on the last dim)

    def qweight(self, i, k, prec):
        wb, _ = bits(prec); wq = self.opt['wq'] if wb == 4 or self.opt.get('wq8') == 'gptq' else 'rtn'
        key = (i, k, wb, wq, self.opt['ba16'], self.opt['ohead'], self.opt['rseed'], self.opt['rout'], self.LRtag, getattr(self, '_base_only', False))
        if key in self.Qw: return self.Qw[key]
        W = self.wfold(i, k); qmax = QMAX[wb]
        if wq == 'gptq':
            Hm = self.hess(i, k)
            q, s = gptq(W, Hm, qmax)
            del Hm
            latent = None
        else:
            s = rtn_scales(W, qmax, wb == 4)
            q = torch.round(W / s[:, None]).clamp(-qmax, qmax)
            latent = W
        if self.LR is not None and (i, k) in self.LR and not getattr(self, '_base_only', False):   # QAT LoRA (rotated space), same rule as h1qat
            A, B = self.LR[(i, k)]
            lat = (q * s[:, None]) if latent is None else latent
            q = torch.round((lat + B @ A) / s[:, None]).clamp(-qmax, qmax)
        out = (q.to(torch.int8).contiguous(), s.float().contiguous())
        self.Qw[key] = out
        return out

    def to_fp32(self):
        """fp32 reference mode (no bf16 rounding floor): all layer tensors fp32, HF copy freed. Used for clean sensitivity measurements."""
        import gc
        for d in self.L:
            for k, v in list(d.items()):
                if torch.is_tensor(v) and v.is_floating_point(): d[k] = v.float().contiguous()
        self.p.model.torso = None; self.p.tm = None; gc.collect(); torch.cuda.empty_cache(); self.fp32 = True

    def load_lrot(self, path):
        if path is None: self.LR = None; self.LRtag = None; return
        sd = torch.load(path, map_location=self.dev)
        self.LR = {(i, k): (sd[f'{i}.{k}.A'].float(), sd[f'{i}.{k}.B'].float()) for i in range(24) for k in GEMMS if f'{i}.{k}.A' in sd}
        self.LRtag = path

    def drop_cache(self):
        self.Qw = {}; torch.cuda.empty_cache()

    # ------------------------------------------------------------- GEMMs
    def qact(self, xr, ab):
        qmax = QMAX[ab]
        s = xr.abs().amax(-1).clamp_min(1e-8) / qmax * (self.opt['aclip4'] if ab == 4 else self.opt['aclip8'])
        q = torch.round(xr / s[:, None]).clamp(-qmax, qmax)
        return q.to(torch.int8), s

    def imm(self, qa, qw):
        M = qa.shape[0]
        if self._intmm and M > 16:
            return torch._int_mm(qa, qw.t()).float()
        return (qa.to(torch.bfloat16) @ qw.to(torch.bfloat16).t()).float() if M > 0 else torch.zeros(0, qw.shape[0], device=qa.device)

    def lin(self, i, k, x, xn=None):
        """x: the bf16 GEMM input of the dense model (for Win/Wgu: rms_zc(x, gain)); xn: fp32 unweighted normed residual (Win/Wgu)."""
        d = self.L[i]; prec = self.P.get((i, k), 'bf16')
        src = xn if k in ('Win', 'Wgu') else x
        if self.cap is not None and (i, k) in self.cap:      # UNROTATED input Hessian (rotated per config in hess())
            xs = src.float()
            Hk = self.Hacc.get((i, k))
            upd = xs.t() @ xs
            self.Hacc[(i, k)] = upd if Hk is None else Hk + upd
        if prec == 'bf16':
            y = x @ d[k].t()
            if self.LR is not None and (i, k) in self.LR:
                A, B = self.LR[(i, k)]
                yl = (self.rot_for(i, k)(src.float()) @ A.t()) @ B.t()
                if yl.shape[1] < y.shape[1]: yl = torch.cat([yl, torch.zeros(yl.shape[0], y.shape[1] - yl.shape[1], device=yl.device)], 1)
                y = (y.float() + yl).to(x.dtype)
            return y
        q0 = getattr(self, '_q0', None)
        if self.opt.get('qb16') and q0 is not None and q0 < x.shape[0]:     # question rows (>= q0) in bf16, state rows low-bit
            P0 = self.P; self.P = {key: v for key, v in P0.items() if key != (i, k)}
            yq = self.lin(i, k, x[q0:], None if xn is None else xn[q0:])
            self.P = P0; self._q0 = None
            ys = self.lin(i, k, x[:q0], None if xn is None else xn[:q0]) if q0 > 0 else yq[:0]
            self._q0 = q0
            return torch.cat([ys, yq], 0)
        wb, ab = bits(prec)
        xr = self.rot_for(i, k)(src.float())
        if wb == 16 or ab == 16:                                   # diagnostics: weight-only / activation-only quantization (fp32 GEMM)
            if wb == 16:
                kf = ('f32', i, k, self.opt['ba16'], self.opt['ohead'], self.opt['rseed'], self.opt['rout'])
                if kf not in self.Qw: self.Qw[kf] = self.wfold(i, k)
                Wf = self.Qw[kf]
            else: qw, sw = self.qweight(i, k, prec); Wf = qw.float() * sw[:, None]
            if ab != 16: qa, sa = self.qact(xr, ab); xr = qa.float() * sa[:, None]
            y = xr @ Wf.t()
        else:
            qw, sw = self.qweight(i, k, prec)
            qa, sa = self.qact(xr, ab)
            y = self.imm(qa, qw) * sa[:, None] * sw[None, :]
        if k in ('Wo', 'Wd') and self.opt['rout']: y = self.rots()['R1'].inv(y)   # back to the unrotated residual basis (emulation only)
        y = y.to(x.dtype)
        if k == 'Win' and self.opt['ba16'] and d['type'] == 'linear_attention':
            y = torch.cat([y, x @ d['Win'][8192:].t()], 1)
        return y

    # ------------------------------------------------------------- forward
    @torch.no_grad()
    def fwd(self, ids, keep_x=(), stop=None, q0=None):
        dev = self.dev; T = len(ids); eps = self.eps; self._q0 = q0
        ids_t = torch.tensor(ids, device=dev)
        x = F.embedding(ids_t, self.embed)
        if getattr(self, 'fp32', False): x = x.float()
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
        caps = {}
        nL = 24 if stop is None else stop
        for i in range(nL):
            d = self.L[i]; gdn = d['type'] == 'linear_attention'
            xf = x.float(); xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            h = (xn * d['in1']).to(x.dtype)
            proj = self.lin(i, 'Win', h, xn)
            if gdn:
                qkv = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
                beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
                qkv = fla_conv(qkv[None].contiguous(), d['conv_w'], None, activation='silu')
                qkv = qkv[0] if isinstance(qkv, tuple) else qkv
                q, k, v = qkv.split(2048, dim=-1)
                o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), g[None],
                                              beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
                of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
                o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(x.dtype).reshape(T, 2048)
            else:
                qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
                kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
                qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
                def rope(t):
                    xr_, xp = t[..., :64], t[..., 64:]; x1, x2 = xr_[..., :32], xr_[..., 32:]
                    c = cos[:, None, :]; s_ = sin[:, None, :]
                    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
                qh, kk = rope(qh), rope(kk)
                o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
                o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
            x = x + self.lin(i, 'Wo', o)
            xf = x.float(); xn2 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            h2 = (xn2 * d['post1']).to(x.dtype)
            gu = self.lin(i, 'Wgu', h2, xn2); I = d['I']
            m = F.silu(gu[:, :I]) * gu[:, I:]
            x = x + self.lin(i, 'Wd', m)
            if i in keep_x: caps[i] = x
        if stop is not None: return None, caps
        return rms_zc(x, self.norm_w, eps), caps

    @torch.no_grad()
    def logits(self, h, pr):
        T = h.shape[0]; opt_abs = [pr['q0'] + o for o in pr['opt']]
        lg = self.head(h[T - 1].float()[None], h[torch.tensor(opt_abs, device=self.dev)].float()[None])[0] / self.p.temp_for(pr['rq'].kind)
        return lg[:pr['rq'].n_slots].float()

    def pdict(self, lg, pr):
        p = torch.softmax(lg, -1).tolist()
        return {lab: p[j] for j, lab in enumerate(pr['rq'].slot_labels)}


def cal_items(n, seed=0, minT=300, maxT=3500, skip=0):
    """train-split real (state, question) pairs from evalkit/train_pool.jsonl (eval tasks excluded), deterministic"""
    import random
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    out = []; rng = random.Random(seed)
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for li, l in enumerate(f):
            if li % 97 != 13 + seed: continue
            r = json.loads(l)
            if r['task'] in EV or not minT <= r['n_state_tok'] <= maxT: continue
            qn = rng.choice(sorted(r['questions']))
            out.append((r['state'], r['questions'][qn]))
    out = out[skip:skip + n]
    return out
