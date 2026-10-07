"""H3 library: hobson-v19 (LoRA merged, fused lean weights, fla kernels) with
  * per-linear fake quantization (W and A separately): int4 / int8 per-channel W (MSE clip) x per-token A (absmax), NVFP4 (block 16, E4M3
    block scale + fp32 tensor scale), MXFP4 (block 32, E8M0 power-of-two scale); optional norm-gain folding and QuaRot-style rotation;
  * capture of every sub-layer tensor (error-propagation analysis, paper figs 2/5);
  * trainable LoRA + head variants ('std' = hobson's PointerHead, 'hyp' = hyperspherical pointer head, v1) and block-norm variants
    ('rms' = hobson's zero-centred RMSNorm, 'l2' = x/||x|| (v2 literal: no gain, no 1/sqrt(d)), 'l2s' = x/||x|| * learnable scalar
    initialised to sqrt(d)*mean(1+w) (v2 with the per-channel gain removed but the scale kept)).
Forward is differentiable (the caller sets no_grad / inference_mode).
"""
import os, sys, math, json, glob
sys.path[:0] = [os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/evalkit'),
                os.path.expanduser('~/work/j2')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv

FULL_ATTN = (3, 7, 11, 15, 19, 23)
NAMES = ('Win', 'Wo', 'Wgu', 'Wd')


def nrm(x, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rms_zc(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)).to(x.dtype)


def l2n(x, eps=1e-12):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(-1, keepdim=True) + eps)).to(x.dtype)


# ------------------------------------------------------------------ quantizers (fake quant; return dequantized fp32)
_FP4_THR = None
_FP4_VAL = None


def e2m1(a):
    """round |a| (<= 6) to the FP4 E2M1 magnitude grid {0,.5,1,1.5,2,3,4,6}; sign restored by caller"""
    global _FP4_THR, _FP4_VAL
    if _FP4_THR is None or _FP4_THR.device != a.device:
        _FP4_THR = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=a.device)
        _FP4_VAL = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=a.device)
    return _FP4_VAL[torch.bucketize(a.clamp(max=6.0), _FP4_THR, right=False)]


def q_rows(x, bits, clip=1.0):
    """symmetric per-row absmax (rows = tokens for activations); x [M, K] fp32"""
    qmax = 7.0 if bits == 4 else 127.0
    s = (x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax) * clip
    return torch.round(x / s).clamp(-qmax, qmax) * s


def q_rows_mse(W, bits):
    """per-output-channel symmetric RTN with per-channel MSE clip search (weights)"""
    qmax = 7.0 if bits == 4 else 127.0
    s0 = W.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
    best_e = None; best_s = s0.clone()
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7) if bits == 4 else (1.0,):
        s = s0 * r
        e = ((torch.round(W / s).clamp(-qmax, qmax) * s - W) ** 2).sum(1, keepdim=True)
        if best_e is None: best_e = e
        else:
            m = e < best_e; best_e = torch.where(m, e, best_e); best_s = torch.where(m, s, best_s)
    return torch.round(W / best_s).clamp(-qmax, qmax) * best_s


def q_nvfp4(x):
    """NVFP4: blocks of 16 along the last dim, block scale amax/6 stored as FP8 E4M3 relative to a per-tensor fp32 scale"""
    M, K = x.shape
    xb = x.reshape(M, K // 16, 16)
    amax_t = x.abs().amax().clamp_min(1e-12)
    gs = amax_t / (6.0 * 448.0)
    bs = (xb.abs().amax(-1, keepdim=True) / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float() * gs
    bs = torch.where(bs > 0, bs, torch.ones_like(bs))
    y = xb / bs
    return (e2m1(y.abs()) * torch.sign(y) * bs).reshape(M, K)


def q_mxfp4(x):
    """MXFP4 (OCP): blocks of 32, shared scale 2^(floor(log2 amax) - 2), E2M1 elements (clamped at 6)"""
    M, K = x.shape
    xb = x.reshape(M, K // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    sc = torch.exp2(torch.floor(torch.log2(amax)) - 2.0)
    y = xb / sc
    return (e2m1(y.abs()) * torch.sign(y) * sc).reshape(M, K)


def q_group_int4(x, g=128):
    M, K = x.shape
    xb = x.reshape(M, K // g, g)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.0
    return (torch.round(xb / s).clamp(-7, 7) * s).reshape(M, K)


def fq_act(x, fmt, clip=1.0):
    """x [M, K] -> fake-quantized fp32"""
    if fmt in (None, 'bf16'): return x.float()
    xf = x.float()
    if fmt == 'int4': return q_rows(xf, 4, clip)
    if fmt == 'int8': return q_rows(xf, 8)
    if fmt == 'nvfp4': return q_nvfp4(xf)
    if fmt == 'mxfp4': return q_mxfp4(xf)
    if fmt == 'g128': return q_group_int4(xf, 128)
    raise ValueError(fmt)


def fq_w(W, fmt):
    Wf = W.float()
    if fmt in (None, 'bf16'): return Wf
    if fmt == 'int4': return q_rows_mse(Wf, 4)
    if fmt == 'int8': return q_rows_mse(Wf, 8)
    if fmt == 'nvfp4': return q_nvfp4(Wf)
    if fmt == 'mxfp4': return q_mxfp4(Wf)
    if fmt == 'g128': return q_group_int4(Wf, 128)
    raise ValueError(fmt)


def ste(xq, x):
    return x + (xq - x).detach()


# ------------------------------------------------------------------ rotation (QuaRot-style randomized Kronecker Hadamard; as g2lib)
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


_ROT = {}


def _facs(K):
    if K == 2048: return [32, 64]
    if K == 6144: return [12, 16, 32]
    if K & (K - 1) == 0: return [K]
    assert K % 12 == 0 and (K // 12) & (K // 12 - 1) == 0, K
    return [12, K // 12]


def rot_mat(K, dev):
    """dense orthogonal [K, K] matrix: diag(sign) @ kron(H...) (fp32)"""
    if K not in _ROT:
        g = torch.Generator(device='cpu'); g.manual_seed(1234 + K)
        sign = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).float().to(dev)
        facs = _facs(K)
        mats = [paley12(dev) if n == 12 else hadamard(n, dev) for n in facs]
        H = mats[0]
        for m in mats[1:]: H = torch.kron(H, m)
        _ROT[K] = sign[:, None] * H
    return _ROT[K]


_RF = {}
_RB = {}


def rot_mat_bf16(K, dev):
    if K not in _RB: _RB[K] = rot_mat(K, dev).to(torch.bfloat16).contiguous()
    return _RB[K]


def rot_rows(x):
    """x [M, K] -> x @ rot_mat(K) via the Kronecker factors (fast); fp32"""
    K = x.shape[-1]; dev = x.device
    if K not in _RF:
        rot_mat(K, dev)
        g = torch.Generator(device='cpu'); g.manual_seed(1234 + K)
        sign = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).float().to(dev)
        facs = _facs(K)
        mats = [paley12(dev) if n == 12 else hadamard(n, dev) for n in facs]
        _RF[K] = (sign, facs, mats)
    sign, facs, mats = _RF[K]
    y = (x.float() * sign).reshape(x.shape[0], *facs)
    for ax, Hm in enumerate(mats):
        y = torch.movedim(torch.tensordot(y, Hm, dims=([ax + 1], [0])), -1, ax + 1)
    return y.reshape(x.shape)


# ------------------------------------------------------------------ quant config
class QCfg:
    """rules: list of dicts {w: fmt, a: fmt, layers: iterable|None, names: iterable|None, clip: float}; later rules override earlier.
    names may include GDN in_proj sub-blocks 'Win.qkv', 'Win.z', 'Win.ab' (and 'Win' = all three). fold: fold norm gains into Win/Wgu and feed the
    unweighted normed residual (function-preserving); rot: rotate GEMM inputs (implies fold)."""

    def __init__(self, rules=(), fold=False, rot=False, name='', qb16=False):
        self.rules = list(rules); self.fold = fold or rot; self.rot = rot; self.name = name
        self.qb16 = qb16      # question rows (>= q0, incl. option and answer readout rows) run the bf16 GEMM; state rows low-bit
        self._tab = {}

    def get(self, i, nm):
        k = (i, nm)
        if k not in self._tab:
            res = None
            for r in self.rules:
                if r.get('layers') is not None and i not in r['layers']: continue
                nms = r.get('names')
                if nms is not None and nm not in nms and not (nm.startswith('Win.') and 'Win' in nms): continue
                res = (r.get('w', 'bf16'), r.get('a', 'bf16'), r.get('clip', 1.0))
            if res is not None and res[0] == 'bf16' and res[1] == 'bf16': res = None
            self._tab[k] = res
        return self._tab[k]

    def any(self):
        return len(self.rules) > 0


def Q(w, a=None, layers=None, names=None, clip=1.0, fold=False, rot=False, name=''):
    return QCfg([dict(w=w, a=a if a is not None else w, layers=layers, names=names, clip=clip)], fold=fold, rot=rot, name=name)


DENSE = QCfg()


# ------------------------------------------------------------------ heads
class StdHead(nn.Module):
    """hobson's PointerHead (LayerNorm -> q / k Linear(2048 -> 256) -> dot / 16)"""

    def __init__(self, h0):
        super().__init__()
        self.norm = nn.LayerNorm(h0.q.in_features); self.q = nn.Linear(h0.q.in_features, h0.q.out_features); self.k = nn.Linear(h0.k.in_features, h0.k.out_features)
        self.load_state_dict({k: v.detach().clone().float() for k, v in h0.state_dict().items()}, strict=False)
        self.scale = h0.q.out_features ** -0.5

    def forward(self, decide, options):  # [B, d], [B, K, d]
        d = self.q(self.norm(decide)).unsqueeze(-1)
        o = self.k(self.norm(options))
        return (o @ d).squeeze(-1) * self.scale


class HypHead(nn.Module):
    """v1: hyperspherical pointer head (nGPT attention-logit form).  x_hat = N(LN(x)); q = N(row_unit(Wq) x_hat) * s_qk; k = N(row_unit(Wk) o_hat) * s_qk;
    logit = tau * <q, k>.  Every factor is bounded: |logit| <= tau * max(s_qk)^2.  Initialised from hobson's head (directions kept)."""

    def __init__(self, h0, tau=10.0):
        super().__init__()
        din, dim = h0.q.in_features, h0.q.out_features
        self.norm = nn.LayerNorm(din)
        self.norm.load_state_dict({k: v.detach().clone().float() for k, v in h0.norm.state_dict().items()})
        self.Wq = nn.Parameter(h0.q.weight.detach().clone().float()); self.Wk = nn.Parameter(h0.k.weight.detach().clone().float())
        self.s_qk = nn.Parameter(torch.ones(dim, device=h0.q.weight.device))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau), device=h0.q.weight.device))

    def forward(self, decide, options):
        xd = l2n(self.norm(decide).float()); xo = l2n(self.norm(options).float())
        Uq = l2n(self.Wq); Uk = l2n(self.Wk)
        q = l2n(xd @ Uq.t()) * self.s_qk; k = l2n(xo @ Uk.t()) * self.s_qk
        return (k @ q.unsqueeze(-1)).squeeze(-1) * self.log_tau.exp()


# ------------------------------------------------------------------ model
class H3:
    def __init__(self, maxlen=16384, dev='cuda'):
        from plib import P
        import lean as LN
        p = P(); p.model.config.max_length = maxlen
        self.p = p
        lean = LN.Lean(p.tm)
        self.L = lean.layers; self.embed = lean.embed; self.norm_w = lean.norm_w; self.inv = lean.inv; self.eps = lean.eps
        self.dev = lean.dev
        self.head0 = p.model.head
        for d in self.L:
            d['in1'] = (1.0 + d['in_norm']).float(); d['post1'] = (1.0 + d['post_norm']).float()
        self.wcache = {}
        self.lora = None; self.lora_scale = 2.0
        self.norm_mode = 'rms'; self.nscale = None
        self.head = None
        self.qat = None          # QCfg used with STE during training (QAT arm)

    # ---------- setup ----------
    def detach_inference(self):
        """clone weights out of inference mode and free the HF torso (needed before autograd)"""
        import gc
        for d in self.L:
            for k, v in list(d.items()):
                if torch.is_tensor(v): d[k] = v.clone()
        self.embed = self.embed.clone(); self.norm_w = self.norm_w.clone(); self.inv = self.inv.clone()
        self.head0 = StdHead(self.head0).to(self.dev)
        for p_ in self.head0.parameters(): p_.requires_grad_(False)
        self.p.model.torso = None; self.p.tm = None
        gc.collect(); torch.cuda.empty_cache()

    def free_hf(self):
        """inference-only: drop the HF torso (its q/k/v/gate/up copies are duplicated by the fused weights)"""
        import gc
        self.head0 = StdHead(self.head0).to(self.dev)
        for p_ in self.head0.parameters(): p_.requires_grad_(False)
        self.p.model.torso = None; self.p.tm = None
        gc.collect(); torch.cuda.empty_cache()

    def add_lora(self, r=16, alpha=32, seed=0):
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.lora = nn.ModuleList()
        for i in range(24):
            md = nn.ModuleDict()
            for k in NAMES:
                out, inp = self.L[i][k].shape
                m = nn.Module()
                m.A = nn.Parameter((torch.randn(r, inp, generator=g) / math.sqrt(inp)).to(self.dev))
                m.B = nn.Parameter(torch.zeros(out, r, device=self.dev))
                md[k] = m
            self.lora.append(md)
        self.lora_scale = alpha / r
        return list(self.lora.parameters())

    def set_head(self, kind, tau=None):
        if kind == 'std': self.head = StdHead(self.head0).to(self.dev)
        elif kind == 'hyp': self.head = HypHead(self.head0, tau or 10.0).to(self.dev)
        else: raise ValueError(kind)
        return list(self.head.parameters())

    def set_norm(self, mode):
        """block pre-norms (input_layernorm, post_attention_layernorm of every layer). Final norm untouched (it only feeds the fp32 head)."""
        self.norm_mode = mode
        if mode == 'l2s':
            sd = math.sqrt(2048.0)
            self.nscale = nn.ParameterList([nn.Parameter(torch.tensor([sd * float(self.L[i]['in1'].mean()), sd * float(self.L[i]['post1'].mean())], device=self.dev))
                                            for i in range(24)])
            return list(self.nscale.parameters())
        return []

    def init_qat(self, fmt):
        self.qat = Q(fmt); self.qat_clip = {}
        if fmt not in ('int4', 'int8'): return
        qmax = 7.0 if fmt == 'int4' else 127.0
        with torch.no_grad():
            for i in range(24):
                for nm in NAMES:
                    Wf = self.L[i][nm].float(); s0 = Wf.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
                    best_e = None; best_r = torch.ones_like(s0)
                    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7) if fmt == 'int4' else (1.0,):
                        e = ((torch.round(Wf / (s0 * r)).clamp(-qmax, qmax) * (s0 * r) - Wf) ** 2).sum(1, keepdim=True)
                        if best_e is None: best_e = e
                        else:
                            mm = e < best_e; best_e = torch.where(mm, e, best_e); best_r = torch.where(mm, torch.full_like(best_r, r), best_r)
                    self.qat_clip[(i, nm)] = best_r
                    del Wf

    def merge_lora_into_weights(self):
        """fold trained LoRA into the bf16 weights (for PTQ of the fine-tuned model)"""
        with torch.no_grad():
            for i in range(24):
                for k in NAMES:
                    lo = self.lora[i][k]
                    self.L[i][k] = (self.L[i][k].float() + self.lora_scale * lo.B.float() @ lo.A.float()).to(torch.bfloat16).contiguous()
        self.lora = None; self.wcache = {}

    # ---------- state io ----------
    def trainable_state(self):
        sd = {}
        if self.lora is not None:
            for i in range(24):
                for k in NAMES:
                    sd[f'lora.{i}.{k}.A'] = self.lora[i][k].A.detach().cpu(); sd[f'lora.{i}.{k}.B'] = self.lora[i][k].B.detach().cpu()
        if self.head is not None:
            for k, v in self.head.state_dict().items(): sd[f'head.{k}'] = v.detach().cpu()
        if self.nscale is not None:
            for i, p_ in enumerate(self.nscale): sd[f'nscale.{i}'] = p_.detach().cpu()
        sd['_meta'] = dict(norm_mode=self.norm_mode, head=type(self.head).__name__ if self.head is not None else None, lora_scale=self.lora_scale)
        return sd

    def load_trainable(self, path, merge=False):
        sd = torch.load(path, map_location=self.dev)
        meta = sd.get('_meta', {})
        hk = meta.get('head')
        self.set_head('hyp' if hk == 'HypHead' else 'std')
        self.head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('head.')})
        self.set_norm(meta.get('norm_mode', 'rms'))
        if self.nscale is not None:
            with torch.no_grad():
                for i in range(24): self.nscale[i].copy_(sd[f'nscale.{i}'])
        if any(k.startswith('lora.') for k in sd):
            r = sd['lora.0.Win.A'].shape[0]
            self.add_lora(r=r, alpha=meta.get('lora_scale', 2.0) * r)
            with torch.no_grad():
                for i in range(24):
                    for k in NAMES:
                        self.lora[i][k].A.copy_(sd[f'lora.{i}.{k}.A']); self.lora[i][k].B.copy_(sd[f'lora.{i}.{k}.B'])
            for p_ in self.lora.parameters(): p_.requires_grad_(False)
            if merge: self.merge_lora_into_weights()   # NB: bf16 rounding of W + BA loses part of a small update; default keeps LoRA unmerged
                                                        # (bf16 path = PEFT-style W x + B A x, as hobson itself is deployed; quantized path
                                                        # quantizes W + BA formed in fp32)
        for p_ in self.head.parameters(): p_.requires_grad_(False)
        if self.nscale is not None:
            for p_ in self.nscale.parameters(): p_.requires_grad_(False)
        return meta

    # ---------- norms ----------
    def bnorm(self, x, i, which):
        """block pre-norm. which 0 = input_layernorm (before mixer), 1 = post_attention_layernorm (before MLP)"""
        if self.norm_mode == 'rms':
            return rms_zc(x, self.L[i]['in_norm' if which == 0 else 'post_norm'], self.eps)
        if self.norm_mode == 'l2':
            return l2n(x)
        if self.norm_mode == 'l2s':
            return (l2n(x).float() * self.nscale[i][which]).to(x.dtype)
        raise ValueError(self.norm_mode)

    # ---------- linear with quant ----------
    WBUDGET = 7e9

    def _wq(self, i, nm, W, fmt, fold, rot, lo=None, rows=None):
        key = (i, nm, fmt, fold, rot)
        if key in self.wcache:
            self.wcache[key] = self.wcache.pop(key)       # LRU refresh
        else:
            tot = sum(v.numel() * 2 for v in self.wcache.values())
            while self.wcache and tot + W.numel() * 2 > self.WBUDGET:
                k0 = next(iter(self.wcache)); tot -= self.wcache[k0].numel() * 2; del self.wcache[k0]
            Wf = W.float()
            if lo is not None:
                B = lo.B if rows is None else lo.B[rows[0]:rows[1]]
                Wf = Wf + self.lora_scale * (B.float() @ lo.A.float())
            if fold:
                if nm.startswith('Win'): Wf = Wf * self.L[i]['in1'][None, :]
                if nm == 'Wgu': Wf = Wf * self.L[i]['post1'][None, :]
            if rot: Wf = Wf @ rot_mat(Wf.shape[1], self.dev)
            self.wcache[key] = fq_w(Wf, fmt).to(torch.bfloat16)
        return self.wcache[key]

    def lin(self, x, i, nm, qc, xn=None):
        """x: GEMM input [T, K] (bf16). nm in NAMES. xn: unweighted normed residual (for fold / rot of Win and Wgu)."""
        W = self.L[i][nm]
        lo = self.lora[i][nm] if self.lora is not None else None
        qat = self.qat
        if qat is not None and qat.get(i, nm) is not None:          # QAT: fake quant with STE on (W + BA) and on the activation
            wf, af, clip = qat.get(i, nm)
            Weff = W.float() + (self.lora_scale * lo.B.float() @ lo.A.float() if lo is not None else 0.0)
            if wf in ('int4', 'int8'):          # per-channel absmax x clip ratio fixed at init (MSE search on the bf16 weight)
                qmax = 7.0 if wf == 'int4' else 127.0
                s = (Weff.detach().abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax) * self.qat_clip[(i, nm)]
                Wq = ste(torch.round(Weff.detach() / s).clamp(-qmax, qmax) * s, Weff)
            else:
                Wq = ste(fq_w(Weff.detach(), wf), Weff) if wf != 'bf16' else Weff
            xq = ste(fq_act(x.detach(), af, clip), x.float()) if af != 'bf16' else x.float()
            return (xq.to(torch.bfloat16) @ Wq.to(torch.bfloat16).t())
        if nm == 'Win' and self.L[i]['type'] == 'linear_attention' and qc.any():
            parts = [('Win.qkv', 0, 6144), ('Win.z', 6144, 8192), ('Win.ab', 8192, 8224)]
            specs = [qc.get(i, p[0]) for p in parts]
            if len(set(specs)) > 1:
                outs = []
                for (pn, a0, a1), sp in zip(parts, specs):
                    outs.append(self._lin1(x, i, pn, W[a0:a1], sp, qc, xn, lo, (a0, a1)))
                return torch.cat(outs, -1)
            return self._lin1(x, i, 'Win', W, specs[0], qc, xn, lo, None)
        return self._lin1(x, i, nm, W, qc.get(i, nm), qc, xn, lo, None)

    def _lin1(self, x, i, nm, W, spec, qc, xn, lo, rows):
        if spec is None:
            y = x @ W.t()
        else:
            wf, af, clip = spec
            src = x
            fold = xn is not None and nm.startswith(('Win', 'Wgu'))
            if fold: src = xn
            if qc.rot: src = src.to(torch.bfloat16) @ rot_mat_bf16(src.shape[1], self.dev)   # online rotation as a bf16 tensor-core GEMM
            xq = fq_act(src, af, clip).to(torch.bfloat16)
            Wq = self._wq(i, nm, W, wf, fold, qc.rot, lo, rows)
            y = torch.mm(xq, Wq.t(), out_dtype=torch.float32).to(torch.bfloat16) if not torch.is_grad_enabled() else xq @ Wq.t()
            if qc.qb16 and self._q0 is not None:       # question rows: exact bf16 GEMM (+ unmerged LoRA)
                q0 = self._q0; yq = x[q0:] @ W.t()
                if lo is not None:
                    B = lo.B if rows is None else lo.B[rows[0]:rows[1]]
                    yq = yq + ((x[q0:] @ lo.A.t().to(x.dtype)) @ B.t().to(x.dtype)) * self.lora_scale
                y = torch.cat([y[:q0], yq.to(y.dtype)], 0)
            return y                                   # LoRA (if any) is inside the quantized weight
        if lo is not None:
            B = lo.B if rows is None else lo.B[rows[0]:rows[1]]
            y = y + ((x @ lo.A.t().to(x.dtype)) @ B.t().to(x.dtype)) * self.lora_scale
        return y

    # ---------- forward ----------
    def layer(self, i, x, cos, sin, qc, cap=None):
        d = self.L[i]; T = x.shape[0]; eps = self.eps
        gdn = d['type'] == 'linear_attention'
        need_xn = qc.fold and self.norm_mode == 'rms'
        h = self.bnorm(x, i, 0)
        if cap: cap(i, 'layer_in', x); cap(i, 'attn_in', h)
        proj = self.lin(h, i, 'Win', qc, nrm(x, eps) if need_xn else None)
        if cap: cap(i, 'in_proj', proj)
        if gdn:
            qkv = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            if cap: cap(i, 'beta', beta); cap(i, 'g', g)
            qkv = fla_conv(qkv[None].contiguous(), d['conv_w'], None, activation='silu')
            qkv = qkv[0] if isinstance(qkv, tuple) else qkv
            qkv = qkv.reshape(T, 6144) if qkv.dim() == 3 else qkv
            q, k, v = qkv.split(2048, dim=-1)
            o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), g[None],
                                          beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
            if cap: cap(i, 'core', o.reshape(T, 2048))
            of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(x.dtype).reshape(T, 2048)
        else:
            qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)

            def rope(t):
                xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
                c = cos[:, None, :]; s_ = sin[:, None, :]
                return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
            qh, kk = rope(qh), rope(kk)
            o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
            if cap: cap(i, 'core', o[0].transpose(0, 1).reshape(T, 2048))
            o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
        if cap: cap(i, 'o_in', o)
        dout = self.lin(o, i, 'Wo', qc)
        if cap: cap(i, 'attn_out', dout)
        x = x + dout
        h2 = self.bnorm(x, i, 1)
        if cap: cap(i, 'A+R', x); cap(i, 'mlp_in', h2)
        gu = self.lin(h2, i, 'Wgu', qc, nrm(x, eps) if need_xn else None); I = d['I']
        m = F.silu(gu[:, :I]) * gu[:, I:]
        if cap: cap(i, 'mlp_hid', m)
        mo = self.lin(m, i, 'Wd', qc)
        if cap: cap(i, 'mlp_out', mo)
        x = x + mo
        if cap: cap(i, 'layer_out', x)
        return x

    _q0 = None

    def forward(self, ids, qc=DENSE, cap=None, ckpt=False, keep=(), q0=None):
        dev = self.dev; T = len(ids); self._q0 = q0
        ids_t = torch.tensor(ids, device=dev)
        x = F.embedding(ids_t, self.embed)
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
        kept = {}
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self.layer, i, x, cos, sin, qc, None, use_reentrant=False)
            else:
                x = self.layer(i, x, cos, sin, qc, cap)
            if i in keep: kept[i] = x
        h = rms_zc(x, self.norm_w, self.eps)
        return (h, kept) if keep else h

    # ---------- decisions ----------
    def prep(self, state, qd):
        return self.p.prep(state, qd)

    def temp(self, kind):
        return self.p.temp_for(kind)

    def logits(self, h, pr, head=None):
        head = head or self.head or self.head0
        T = h.shape[0]
        opt_abs = torch.tensor([pr['q0'] + o for o in pr['opt']], device=self.dev)
        lg = head(h[T - 1].float()[None], h[opt_abs].float()[None])[0] / self.temp(pr['rq'].kind)
        return lg[:pr['rq'].n_slots]

    def probs(self, h, pr, head=None):
        return torch.softmax(self.logits(h, pr, head).float(), -1)
