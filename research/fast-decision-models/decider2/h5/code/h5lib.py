"""H5 library: full-depth hobson-v19 (24 layers, merged LoRA) as a fake-quant simulator for learned-rotation W4A4.

Basis convention (simulation keeps the residual stream in the ORIGINAL basis; deployment keeps it rotated, which is the same math):
  R1 [2048,2048] orthogonal residual rotation (learned, SpinQuant R1). Offline-absorbable:
      embed' = embed R1;  Win' = Win diag(1+in_norm) R1;  Wgu' = Wgu diag(1+post_norm) R1;  Wo' = R1^T Wo Ho;  Wd' = R1^T Wd Hd;
      final norm/head read x R1^T (only the pooled + option rows; or fold into the head's LayerNorm input).
  Ho [2048] / Hd [6144]: fixed randomized Kronecker-Hadamard on the ONLINE GEMM inputs (GDN out_proj / attention o_proj input after the
      gated norm / output gate; MLP down_proj input). Not absorbable (elementwise gates in between), as SpinQuant's R4.
Quantizers (match G2's CUTLASS s4/s8 kernels: symmetric codes, int32 accumulate, per-row x per-column dequant scales):
  activations: per-token symmetric absmax (optional clip ratio), weights: per-output-channel symmetric (MSE clip search) RTN or GPTQ.
Site classes (precision configurable per class): Win_g (GDN in_proj qkvz[+ba]), Win_a (attention q/gate/k/v), Wo_g, Wo_a, Wgu, Wd.
  ba_hp: GDN in_proj_b / in_proj_a rows (32 of 8224) computed as a separate bf16 GEMM from the unquantized normed input.
Modes: 'ref' = exact bf16 path of g2lib (the dense reference); 'learn' = weights 16-bit, activations fake-quantized in the rotated basis
  (differentiable in R1; SpinQuant's W16A4 rotation-learning setting); 'q' = materialized rotated (and fake-quantized) weights.
"""
import os, sys, json, math, random
sys.path[:0] = [os.path.expanduser('~/work/h5'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv
import g2lib as GL

SITES = ('Win', 'Wo', 'Wgu', 'Wd')


def ste(x):
    return (torch.round(x) - x).detach() + x


FMT = {'a': 'int', 'w': 'int'}     # 'nvfp4' switches every 4-bit quantizer to NVFP4 (16-element FP8 block scales)


def fq_act(x, bits, clip=1.0, detach_scale=True):
    """per-token symmetric fake quant with straight-through rounding. detach_scale=False lets the gradient reach the per-token absmax
    (needed to learn rotations: with STE the rounding error itself has no gradient, the outlier-driven scale does)"""
    if bits >= 16: return x
    if bits == 4 and FMT['a'] == 'nvfp4': return q_nvfp4(x)
    qm = 2 ** (bits - 1) - 1
    xf = x.float()
    s = (xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) * clip / qm)
    if detach_scale: s = s.detach()
    return (ste(xf / s).clamp(-qm, qm) * s).to(x.dtype)


_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def q_nvfp4(x, block=16):
    """NVFP4 fake quant (Blackwell): FP4 E2M1 values, one FP8 E4M3 scale per 16 consecutive elements along the last dim, one fp32 scale
    per tensor (dynamic here). Round to nearest, saturating."""
    sh = x.shape; xf = x.float().reshape(-1, sh[-1] // block, block)
    g = xf.abs().amax().clamp_min(1e-12) / (6.0 * 448.0)
    sb = (xf.abs().amax(-1, keepdim=True) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn).float() * g
    sb = sb.clamp_min(1e-30)
    v = xf / sb
    grid = torch.tensor(_E2M1, device=x.device); mids = torch.tensor(_MIDS, device=x.device)
    q = grid[torch.bucketize(v.abs(), mids)] * torch.sign(v)
    return (q * sb).reshape(sh).to(x.dtype)


class Ident:
    def fwd(self, x): return x
    def inv(self, x): return x
    def dense(self): raise NotImplementedError


def wscale(W, bits, search=True):
    """per-output-channel symmetric scale with MSE clip search over the RTN grid"""
    qm = 2 ** (bits - 1) - 1
    Wf = W.float(); s = Wf.abs().amax(1).clamp_min(1e-8) / qm
    if not search: return s
    best_e = None; best_s = s.clone()
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6):
        sr = s * r
        e = ((torch.round(Wf / sr[:, None]).clamp(-qm, qm) * sr[:, None] - Wf) ** 2).sum(1)
        if best_e is None: best_e = e; best_s = sr.clone()
        else:
            m = e < best_e; best_e = torch.where(m, e, best_e); best_s = torch.where(m, sr, best_s)
    return best_s


def rtn(W, bits, s=None):
    if bits == 4 and FMT['w'] == 'nvfp4':
        return q_nvfp4(W.float()), torch.ones(W.shape[0], device=W.device)
    qm = 2 ** (bits - 1) - 1
    if s is None: s = wscale(W, bits)
    return (torch.round(W.float() / s[:, None]).clamp(-qm, qm) * s[:, None]), s


def gptq(W, Hs, bits, s=None, blocksize=128, percdamp=0.01, act_order=True):
    """GPTQ (Frantar et al.) with fixed per-channel symmetric scales s (from the MSE clip search). W [N,K], Hs [K,K] = sum x^T x."""
    qm = 2 ** (bits - 1) - 1
    W = W.float().clone(); H = Hs.float().clone(); K = W.shape[1]
    fp4 = bits == 4 and FMT['w'] == 'nvfp4'
    if fp4:   # fixed NVFP4 block scales (FP8 E4M3 per 16 along K, fp32 per tensor) from the unquantized weight; E2M1 grid
        Wb = W.reshape(W.shape[0], K // 16, 16); g = Wb.abs().amax().clamp_min(1e-12) / (6.0 * 448.0)
        sb = (Wb.abs().amax(-1, keepdim=True) / 6.0 / g).clamp(max=448.0).to(torch.float8_e4m3fn).float() * g
        SB = sb.clamp_min(1e-30).expand(-1, -1, 16).reshape(W.shape[0], K)
        grid = torch.tensor(_E2M1, device=W.device); mids = torch.tensor(_MIDS, device=W.device)
        s = torch.ones(W.shape[0], device=W.device)
    if s is None: s = wscale(W, bits)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True); W = W[:, perm]; H = H[perm][:, perm]
        if fp4: SB = SB[:, perm]
    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(K, device=W.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    sc = s[:, None]
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(cnt):
            w = W1[:, j]; dj = Hi[j, j]
            if fp4:
                sj = SB[:, i1 + j]; v = w / sj
                q = grid[torch.bucketize(v.abs(), mids)] * torch.sign(v) * sj
            else:
                q = (torch.round(w[:, None] / sc).clamp(-qm, qm) * sc)[:, 0]
            Q1[:, j] = q
            e = (w - q) / dj
            W1[:, j:] -= e[:, None] @ Hi[j, j:][None, :]
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q, s


class KRot:
    """randomized Kronecker-Hadamard rotation x -> (x * sign) @ kron(mats) and its inverse (orthonormal)"""
    def __init__(self, K, dev, seed):
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.sign = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).float().to(dev)
        self.facs = [32, 64] if K == 2048 else [12, 16, 32]
        self.mats = [GL.paley12(dev) if n == 12 else GL.hadamard(n, dev) for n in self.facs]
        self.K = K

    def fwd(self, x):
        sh = x.shape; y = (x * self.sign.to(x.dtype)).reshape(-1, *self.facs)
        for ax, Hm in enumerate(self.mats):
            y = torch.movedim(torch.tensordot(y, Hm.to(x.dtype), dims=([ax + 1], [0])), -1, ax + 1)
        return y.reshape(sh)

    def inv(self, y):
        sh = y.shape; x = y.reshape(-1, *self.facs)
        for ax, Hm in enumerate(self.mats):
            x = torch.movedim(torch.tensordot(x, Hm.t().to(y.dtype), dims=([ax + 1], [0])), -1, ax + 1)
        return (x.reshape(sh) * self.sign.to(y.dtype))

    def dense(self):
        return self.fwd(torch.eye(self.K, device=self.sign.device))


def kron_apply(x, P1, P2):
    """row vectors x (..., k1*k2) -> x (P1 (x) P2), i.e. X -> P1^T X P2 on the (k1, k2) reshape"""
    k1, k2 = P1.shape[0], P2.shape[0]; sh = x.shape
    X = x.reshape(-1, k1, k2)
    Y = torch.einsum('nab,ac,bd->ncd', X, P1.to(x.dtype), P2.to(x.dtype))
    return Y.reshape(sh)


KFAC = {2048: (32, 64), 6144: (96, 64)}


def site_class(L, i, site):
    if site in ('Win', 'Wo'):
        return site + ('_g' if L[i]['type'] == 'linear_attention' else '_a')
    return site


DEFAULT = dict(Win_g=(4, 4), Win_a=(4, 4), Wo_g=(4, 4), Wo_a=(4, 4), Wgu=(4, 4), Wd=(4, 4))


class Q5:
    def __init__(self, maxlen=16384, lean=False):
        g = GL.G2(maxlen=maxlen); g.detach_inference()
        self.g = g; self.L = g.L; self.dev = g.dev; self.eps = g.eps
        self.Ho = KRot(2048, self.dev, 1235); self.Hd = KRot(6144, self.dev, 1236)   # = H1 FORMAT v0 R2 / R4
        self.R1 = KRot(2048, self.dev, 1234).dense()          # init: randomized Hadamard (QuaRot); replaced by a learned rotation
        self.cfg = dict(DEFAULT); self.aclip = 1.0; self.ba_hp = True; self.layer_cfg = {}
        self.Wr = None; self.lora = None
        self.K = {}; self.clipA = {}          # per-GEMM learned Kronecker rotation (P1, P2) on the GEMM input, per-GEMM activation clip
        self.lean = lean
        for d in self.L:
            if not lean:
                d['Wf_in'] = (d['Win'].float() * d['in1'][None, :]).to(torch.bfloat16)
                d['Wf_gu'] = (d['Wgu'].float() * d['post1'][None, :]).to(torch.bfloat16)
            if d['type'] == 'linear_attention':
                d['Wba'] = (d['Win'][8192:8224].float() * d['in1'][None, :]).to(torch.bfloat16).contiguous()

    def bits(self, i, site):
        c = site_class(self.L, i, site)
        return self.layer_cfg.get((i, c), self.cfg[c])

    # ---------------------------------------------------------------- materialized rotated weights
    def rotated_weight(self, i, site, R=None, use_k=True):
        d = self.L[i]; R = self.R1 if R is None else R
        if site == 'Win': W = (d['Win'].float() * d['in1'][None, :]) @ R
        elif site == 'Wgu': W = (d['Wgu'].float() * d['post1'][None, :]) @ R
        elif site == 'Wo': W = self.Ho.fwd(R.t() @ d['Wo'].float())        # R1^T Wo Ho : rows rotated by R1, columns by Ho
        else: W = self.Hd.fwd(R.t() @ d['Wd'].float())
        P = self.K.get((i, site)) if use_k else None
        return kron_apply(W, *P) if P is not None else W

    def gemm_input(self, i, site, v, use_k=True):
        """rotated (pre-quantization) GEMM input. v = unweighted normed residual (Win/Wgu) or the online input o / m (Wo/Wd)"""
        if site in ('Win', 'Wgu'): xr = v @ self.R1.to(v.dtype)
        else: xr = (self.Ho if site == 'Wo' else self.Hd).fwd(v)
        P = self.K.get((i, site)) if use_k else None
        return kron_apply(xr, *P) if P is not None else xr

    def aclip_of(self, i, site):
        return self.clipA.get((i, site), self.aclip)

    @torch.no_grad()
    def build_rtn(self):
        """materialize rotated weights; RTN-quantize sites with wbits < 16"""
        self.Wr = []; self.Ws = []
        for i in range(24):
            wr = {}; ws = {}
            for s in SITES:
                W = self.rotated_weight(i, s); wb, _ = self.bits(i, s)
                if wb < 16:
                    W, sc = rtn(W, wb); ws[s] = sc
                wr[s] = W.to(torch.bfloat16).contiguous()
            self.Wr.append(wr); self.Ws.append(ws)
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- sites
    def site_in(self, x, i, site, mode, R):
        """x: residual (original basis). Win / Wgu GEMM."""
        d = self.L[i]
        if mode == 'ref':
            w = d['in_norm'] if site == 'Win' else d['post_norm']
            return GL.rms_zc(x, w, self.eps) @ d[site].t()
        wb, ab = self.bits(i, site)
        xn = GL.nrm(x, self.eps)
        if mode == 'learn':
            xr = xn @ R.to(xn.dtype)
            xq = fq_act(xr, ab, self.aclip, detach_scale=False) @ R.t().to(xn.dtype)
            return xq @ (d['Wf_in'] if site == 'Win' else d['Wf_gu']).t()
        return fq_act(self.gemm_input(i, site, xn), ab, self.aclip_of(i, site)) @ self.weight(i, site).t()

    def site_out(self, o, i, site, mode, R):
        """o: online GEMM input (gated-norm / gated-attention output, or MLP hidden); returns the residual update in the ORIGINAL basis"""
        d = self.L[i]
        if mode == 'ref':
            return o @ d[site].t()
        wb, ab = self.bits(i, site)
        H = self.Ho if site == 'Wo' else self.Hd
        if mode == 'learn':
            return H.inv(fq_act(H.fwd(o), ab, self.aclip)) @ d[site].t()
        y = fq_act(self.gemm_input(i, site, o), ab, self.aclip_of(i, site)) @ self.weight(i, site).t()
        return y @ self.R1.t().to(y.dtype)

    def weight(self, i, site):
        W = self.Wr[i][site]
        if self.lora is not None:
            W = self.lora.apply(i, site, W, self)
        return W

    # ---------------------------------------------------------------- layer
    def layer(self, i, x, cos, sin, mode='ref', R=None):
        d = self.L[i]; T = x.shape[0]; eps = self.eps
        gdn = d['type'] == 'linear_attention'
        proj = self.site_in(x, i, 'Win', mode, R)
        if gdn:
            qkv = proj[:, :6144]; z = proj[:, 6144:8192]
            if mode != 'ref' and self.ba_hp:
                ba = GL.nrm(x, eps) @ d['Wba'].t()
                b = ba[:, :16]; a = ba[:, 16:]
            else:
                b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); gg = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            qkv = fla_conv(qkv[None].contiguous(), d['conv_w'], None, activation='silu')
            qkv = qkv[0] if isinstance(qkv, tuple) else qkv
            q, k, v = qkv.split(2048, dim=-1)
            o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), gg[None],
                                          beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
            of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(torch.bfloat16).reshape(T, 2048)
        else:
            qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = GL.rms_zc(qh, d['qn'], eps); kk = GL.rms_zc(kk, d['kn'], eps)

            def rope(t):
                xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
                c = cos[:, None, :]; s_ = sin[:, None, :]
                return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
            qh, kk = rope(qh), rope(kk)
            o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
            o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
        x = x + self.site_out(o, i, 'Wo', mode, R)
        gu = self.site_in(x, i, 'Wgu', mode, R); I = d['I']
        m = F.silu(gu[:, :I]) * gu[:, I:]
        x = x + self.site_out(m, i, 'Wd', mode, R)
        return x

    def rope_tables(self, T):
        pos = torch.arange(T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.g.inv[None, :]; fr = torch.cat([fr, fr], -1)
        return fr.cos().to(torch.bfloat16), fr.sin().to(torch.bfloat16)

    def forward(self, ids, mode='ref', R=None, keep=(), ckpt=False):
        """-> (final normed hidden [T,2048] (original basis), {layer: residual after layer})"""
        T = len(ids); ids_t = torch.tensor(ids, device=self.dev)
        x = F.embedding(ids_t, self.g.embed)
        cos, sin = self.rope_tables(T)
        caps = {}
        for i in range(24):
            if ckpt: x = checkpoint(self.layer, i, x, cos, sin, mode, R, use_reentrant=False)
            else: x = self.layer(i, x, cos, sin, mode, R)
            if i in keep: caps[i] = x
        return GL.rms_zc(x, self.g.norm_w, self.eps), caps

    def logits(self, h, pr):
        T = h.shape[0]; opt_abs = [pr['q0'] + o for o in pr['opt']]
        lg = self.g.head(h[T - 1].float()[None], h[torch.tensor(opt_abs, device=self.dev)].float()[None])[0] / self.g.p.temp_for(pr['rq'].kind)
        return lg[:pr['rq'].n_slots]

    def prep(self, state, qd):
        return self.g.prep(state, qd)

    # ---------------------------------------------------------------- GPTQ (sequential, layer by layer, site by site)
    @torch.no_grad()
    def build_gptq(self, seqs, log=print, act_order=True):
        """seqs: list of token-id lists (calibration, train split). Requires R1 set. Quantizes every site with wbits<16 by GPTQ on the
        rotated weight, using inputs from the already-quantized prefix of the network (rotated + activation-fake-quantized)."""
        self.Wr = [{s: self.rotated_weight(i, s).to(torch.bfloat16).contiguous() for s in SITES} for i in range(24)]   # unquantized; GPTQ'd in order
        self.Ws = [{} for _ in range(24)]
        torch.cuda.empty_cache()
        xs = []
        for ids in seqs:
            xs.append(F.embedding(torch.tensor(ids, device=self.dev), self.g.embed))
        tabs = {}
        for i in range(24):
            for s in SITES:
                wb, ab = self.bits(i, s)
                if wb >= 16: continue
                K = self.Wr[i][s].shape[1]
                Hs = torch.zeros(K, K, device=self.dev, dtype=torch.float32)
                cap = {}
                for x in xs:
                    T = x.shape[0]
                    if T not in tabs: tabs[T] = self.rope_tables(T)
                    cap.clear()
                    self._cap = cap
                    self.layer_capture(i, x, *tabs[T], s)
                    X = cap['x'][0].float()
                    Hs += X.t() @ X
                Q, sc = gptq(self.Wr[i][s].float(), Hs, wb, act_order=act_order)
                self.Wr[i][s] = Q.to(torch.bfloat16).contiguous(); self.Ws[i][s] = sc
                del Hs
            for j, x in enumerate(xs):
                T = x.shape[0]
                xs[j] = self.layer(i, x, *tabs[T], 'q')
            log(f'gptq layer {i} done')
        torch.cuda.empty_cache()

    def layer_capture(self, i, x, cos, sin, site):
        """run layer i in mode 'q' and record the quantized GEMM input of `site` in self._cap['x']"""
        cap = self._cap
        orig_in, orig_out = self.site_in, self.site_out

        def sin_(x_, i_, s_, mode, R):
            if s_ == site:
                xn = GL.nrm(x_, self.eps); _, ab = self.bits(i_, s_)
                xq = fq_act(self.gemm_input(i_, s_, xn), ab, self.aclip_of(i_, s_)); cap.setdefault('x', []).append(xq)
                return xq @ self.weight(i_, s_).t()
            return orig_in(x_, i_, s_, mode, R)

        def sout_(o, i_, s_, mode, R):
            if s_ == site:
                _, ab = self.bits(i_, s_)
                xq = fq_act(self.gemm_input(i_, s_, o), ab, self.aclip_of(i_, s_)); cap.setdefault('x', []).append(xq)
                return (xq @ self.weight(i_, s_).t()) @ self.R1.t().to(o.dtype)
            return orig_out(o, i_, s_, mode, R)
        self.site_in, self.site_out = sin_, sout_
        try:
            if site in ('Win',):
                # only the input GEMM is needed
                self.site_in(x, i, 'Win', 'q', None)
            else:
                self.layer(i, x, cos, sin, 'q')
        finally:
            self.site_in, self.site_out = orig_in, orig_out


# -------------------------------------------------------------------- data
def load_pool(min_tok=150, max_tok=100000):
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    out = []
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV: continue
            if not (min_tok <= r['n_state_tok'] <= max_tok): continue
            out.append(r)
    return out


def tv(p, q):
    return 0.5 * float((p - q).abs().sum())


def kl(p, q):
    return float((p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum())
