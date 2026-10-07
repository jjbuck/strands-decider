"""Q3 library: hobson-v19 as a differentiable W4A4 student in H1/H2's rotated format, plus the bf16 teacher, in hobson's shared-prefix
multi-question layout [state][q1][q2]... (each question sees the state and itself, exactly like strands-decider's prefix-cache path).

Format (H1 FORMAT v1 = H2 qrt.py, until Q2's FORMATS.md says otherwise):
  * residual stream stored rotated, x R1 (R1 = h1lib.Rot(2048, 1234)); embedding E R1; gain-free RMSNorm of the rotated residual feeds
    Win' = Win diag(1+in_norm) R1 and Wgu' = Wgu diag(1+post_norm) R1; Wo' = R1^T Wo R2 (online R2 = Rot(2048, 1235) on the mixer output),
    Wd' = R1^T Wd R4 (online R4 = Rot(6144, 1236) on silu(g)*u). Head reads R1^T-unrotated rows times (1 + norm_w).
  * activations: dynamic symmetric per token, s_t = amax_t / qmax * clip (int4: qmax 7, clip 0.9; int8: 127, 1.0), codes round-half-even, clamp.
  * weights: static symmetric per output channel, codes in [-7, 7] (int4) / [-127, 127] (int8).
  * GEMM: int8 x int8 -> int32 (torch._int_mm; exact), y = acc * s_t * s_ch in fp32 -> bf16.
Student weight modes (per GEMM, QLin):
  'lat'  : bf16 latent W' (trainable); codes = clamp(round(W'/s)); straight-through estimator to W'. Full-weight QAD.
  'code' : fixed int8 codes, trainable per-channel scales (exact gradient). Decision-level scale optimisation.
  'soft' : AdaRound relaxation: W' = s * clamp(cf + h(V), -qmax, qmax), h = clamp(sigmoid(V) * 1.2 - 0.1, 0, 1), trainable V.
Everything else (norms, conv, GDN delta rule, attention, gates, head) as in the lean bf16 runtime.
"""
import os, sys, math, json, hashlib, random
sys.path[:0] = [os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

GEMMS = ('Win', 'Wo', 'Wgu', 'Wd')
ATT = (3, 7, 11, 15, 19, 23)
KEEP = (5, 11, 17, 23)
ZETA, GAMMA = 1.1, -0.1
QMAX = {4: 7.0, 8: 127.0}
ACLIP = {4: 0.9, 8: 1.0}


# ------------------------------------------------------------------ rotations (identical to h1lib.Rot / g2lib)
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


class Rot:
    """randomized Kronecker Hadamard on the last dim: x -> (x * sign) @ (H_a (x) H_b (x) ...); inv: x -> x R^T"""
    def __init__(self, K, seed, dev):
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.K = K
        self.sign = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).float().to(dev)
        self.facs = [32, 64] if K == 2048 else [12, 16, 32]
        self.mats = [paley12(dev) if n == 12 else hadamard(n, dev) for n in self.facs]

    def __call__(self, x):
        sh = x.shape
        y = (x.float() * self.sign).reshape(-1, *self.facs)
        for ax, Hm in enumerate(self.mats):
            y = torch.movedim(torch.tensordot(y, Hm, dims=([ax + 1], [0])), -1, ax + 1)
        return y.reshape(sh)

    def inv(self, x):
        sh = x.shape
        y = x.float().reshape(-1, *self.facs)
        for ax, Hm in enumerate(self.mats):
            y = torch.movedim(torch.tensordot(y, Hm.t(), dims=([ax + 1], [0])), -1, ax + 1)
        return y.reshape(sh) * self.sign


def nrm32(x, eps):
    return _fast('nrm32', _nrm32_e)(x, float(eps))


def rms_zc(x, w, eps):
    return _fast('rmszc', _rmszc_e)(x, w, float(eps))


def rope(t, cos, sin):
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


# ------------------------------------------------------------------ quantizers
FAST = False      # training: torch.compile'd elementwise helpers (division may not be correctly rounded); eval keeps FAST = False (eager = FORMATS.md)
F1 = False        # eval only: emulate the deployed H2 kernel's fp16 accumulator stage (FORMATS.md section 1) on Win/Wo/Wd


def _qact_e(x, qmax: float, clip: float):
    s = x.abs().amax(-1).clamp_min(1e-8) / qmax * clip
    q = torch.round(x / s[:, None]).clamp(-qmax, qmax)
    return q.to(torch.int8), s


def _deq_e(qa, sa):
    return (qa.float() * sa[:, None]).to(torch.bfloat16)


def _wcode_e(A, s, wq: float):
    return torch.round(A.float() / s[:, None]).clamp(-wq, wq).to(torch.int8)


def _soft_e(cf, V, wq: float):
    return (cf.float() + (torch.sigmoid(V.float()) * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)).clamp(-wq, wq).to(torch.bfloat16)


def _gys_e(gy, s):
    return (gy.float() * s[None, :]).to(torch.bfloat16)


_C = {}


def _fast(name, fn):
    if not FAST: return fn
    if name not in _C:
        try: _C[name] = torch.compile(fn, dynamic=True)
        except Exception as e: print('compile failed', name, e); _C[name] = fn
    return _C[name]


def _epi_e(acc, sa, s):
    return (acc * sa[:, None] * s[None, :]).to(torch.bfloat16)


def _nrm32_e(x, eps: float):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def _rmszc_e(x, w, eps: float):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)).to(x.dtype)


def _glue_e(b, a, A_log, dt_bias):
    return torch.sigmoid(b.float()), -A_log.float().exp() * F.softplus(a.float() + dt_bias)


def _conv_e(raw, w, pv):
    ext = torch.cat([torch.zeros(1, raw.shape[1], device=raw.device, dtype=raw.dtype), raw], 0)
    wf = w.float()
    acc = raw.float() * wf[:, 3] + ext[pv[:, 0]].float() * wf[:, 0] + ext[pv[:, 1]].float() * wf[:, 1] + ext[pv[:, 2]].float() * wf[:, 2]
    return F.silu(acc).to(raw.dtype)


def _gnorm_e(o, z, gn_w, eps: float):
    of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    return ((gn_w * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(z.dtype)


def _swiglu_e(gu, I: int):
    g, u = gu.split([I, gu.shape[1] - I], 1)
    return F.silu(g) * u


def _agate_e(o, gate):
    return (o * torch.sigmoid(gate))


def qact(x, qmax, clip):
    """x [M, K] fp32 -> int8 codes, fp32 per-row scales (h1lib.H1.qact = FORMATS.md section 0, RTN).
    The compiled path closes over (qmax, clip) as constants: passing them as arguments to one dynamic=True graph gave wrong int8 rows
    after the graph had been compiled for int4 (measured: hidden rel. MSE 1.03 instead of .0065 in the w4q8 format)."""
    if not FAST or qmax > 7: return _qact_e(x, float(qmax), float(clip))     # int8 rows always eager (see q3dbg8.py)
    qm, cl = float(qmax), float(clip)
    return _fast(f'qact_{qm}_{cl}', lambda x_: _qact_e(x_, qm, cl))(x)


class _HadF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sign, H, P, c):
        T = x.shape[0]
        xs = x.to(torch.bfloat16) * sign
        if P is None:
            y = torch.mm(xs, H, out_dtype=torch.float32) * c
        else:
            z = torch.mm(xs.reshape(T * 12, 512), H, out_dtype=torch.float32).reshape(T, 12, 512)
            y = torch.einsum('tjk,ji->tik', z, P).reshape(T, 12 * 512) * c
        ctx.save_for_backward(sign, H, P if P is not None else torch.empty(0, device=x.device)); ctx.c = c; ctx.hasP = P is not None; ctx.xd = x.dtype
        return y

    @staticmethod
    def backward(ctx, gy):
        sign, H, P = ctx.saved_tensors; c = ctx.c; T = gy.shape[0]
        if ctx.hasP:
            gz = torch.einsum('tik,ji->tjk', gy.float().reshape(T, 12, 512), P)
            gx = torch.mm(gz.reshape(T * 12, 512).to(torch.bfloat16), H.t().contiguous(), out_dtype=torch.float32).reshape(T, 12 * 512)
        else:
            gx = torch.mm(gy.to(torch.bfloat16), H.t().contiguous(), out_dtype=torch.float32)
        return (gx * c * sign.float()).to(ctx.xd), None, None, None, None


class Had:
    """the same randomized Kronecker Hadamard as Rot, applied to bf16 activations by matmuls with +-1 matrices (exact products, fp32 sums),
    then scaled by 1/sqrt(K). 2048: one dense +-1 matmul; 6144: [T*12, 512] @ (H16 x H32) then P12 over the 12 axis."""
    def __init__(self, rot):
        self.K = rot.K; self.sign = rot.sign.to(torch.bfloat16)
        un = [Hm * math.sqrt(Hm.shape[0]) for Hm in rot.mats]           # +-1 entries
        if rot.K == 2048:
            self.H = torch.kron(un[0], un[1]).round().to(torch.bfloat16).contiguous(); self.P = None
        else:
            self.H = torch.kron(un[1], un[2]).round().to(torch.bfloat16).contiguous(); self.P = un[0].round().float().contiguous()
        self.c = 1.0 / math.sqrt(rot.K)

    def __call__(self, x):
        return _HadF.apply(x, self.sign, self.H, self.P, self.c)


def imm(qa, qw):
    """exact integer GEMM qa [M,K] int8 x qw [N,K] int8 -> fp32 [M,N]"""
    M = qa.shape[0]
    if M > 16:
        return torch._int_mm(qa, qw.t()).float()
    return torch.mm(qa.to(torch.bfloat16), qw.to(torch.bfloat16).t(), out_dtype=torch.float32) if M > 0 else torch.zeros(0, qw.shape[0], device=qa.device)


def hsoft(V):
    return (torch.sigmoid(V.float()) * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


class QLinF(torch.autograd.Function):
    """y = Q_a(x) @ Q_w(W)^T with exact integer arithmetic (modes 'lat', 'code') or the AdaRound relaxation (mode 'soft').
    x: fp32 [M, K] (already rotated). A: 'lat': bf16 latent [N,K]; 'code': int8 codes [N,K]; 'soft': bf16 V [N,K]. s: fp32 [N].
    aux: 'lat': cached int8 codes round(A/s) (refreshed after every optimizer step); 'soft': int8 floor codes. Returns bf16 [M, N].
    Backward (STE): dx = (dy * s) @ codes; dW_latent = dy^T @ deq(Q_a(x)); ds = sum_t dy * y / s; dV = s * dy^T @ deq(Q_a(x)) * dh/dV."""

    @staticmethod
    def forward(ctx, x, A, s, aux, mode, wq, aq, aclip, f1a):
        qa, sa = qact(x, aq, aclip)
        if mode == 'lat':
            acc = imm(qa, aux)
        elif mode == 'code':
            acc = imm(qa, A)
        else:                                            # soft: codes U = clamp(cf + h(V)) are fractional
            U = _fast('soft', _soft_e)(aux, A, float(wq))
            acc = torch.mm(qa.to(torch.bfloat16), U.t(), out_dtype=torch.float32); del U
        if f1a:                                          # deployed H2 kernel: C16 = fp16(alpha * acc); d = (C16 * s_t) * (sw / alpha)
            acc = (acc * f1a).half().float(); y = (acc * sa[:, None] * (s / f1a)[None, :]).to(torch.bfloat16)
        else:
            y = _fast('epi', _epi_e)(acc, sa, s) if aq <= 7 else _epi_e(acc, sa, s)
        ctx.mode = mode; ctx.wq = wq; ctx.xdtype = x.dtype
        e0 = torch.empty(0, device=x.device)
        ctx.save_for_backward(qa, sa, A if mode != 'lat' else e0, s, aux if aux is not None else e0, y if mode == 'code' else e0)
        return y

    @staticmethod
    def backward(ctx, gy):
        qa, sa, A, s, aux, y = ctx.saved_tensors
        mode = ctx.mode; wq = ctx.wq
        gy = gy.to(torch.bfloat16).contiguous()
        if mode == 'lat':
            gx = torch.mm(_fast('gys', _gys_e)(gy, s), aux.to(torch.bfloat16)).to(ctx.xdtype)
        elif mode == 'code':
            gx = torch.mm(_fast('gys', _gys_e)(gy, s), A.to(torch.bfloat16)).to(ctx.xdtype)
        else:
            U = _fast('soft', _soft_e)(aux, A, float(wq))
            gx = torch.mm(_fast('gys', _gys_e)(gy, s), U).to(ctx.xdtype); del U
        gA = gs = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            Xd = _fast('deq', _deq_e)(qa, sa)
            if mode == 'lat':
                gA = (gy.t() @ Xd)                                   # STE: d deq(Q(W)) / dW = 1
            elif mode == 'code':
                gs = (gy.float() * y.float()).sum(0) / s             # y = acc * s_t * s  -> dy/ds = y / s
            else:
                gU = (gy.t() @ Xd).float() * s[:, None]
                sg = torch.sigmoid(A.float())
                h = sg * (ZETA - GAMMA) + GAMMA
                u = aux.float() + h.clamp(0, 1)
                live = (h > 0) & (h < 1) & (u > -wq) & (u < wq)
                gA = (gU * sg * (1 - sg) * (ZETA - GAMMA) * live).to(A.dtype)
                del gU, sg, h, u, live
            del Xd
        return gx, gA, gs, None, None, None, None, None, None


# ------------------------------------------------------------------ GPTQ (h1lib.gptq, also returning the error-compensated weights)
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


def gptq_c(W, H, qmax, s=None, blocksize=128, percdamp=0.01, actorder=True):
    """h1lib.gptq exactly (same codes), plus Wc: the error-compensated weight each column had when it was rounded (Q = round(Wc / s))."""
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
    Q = torch.zeros_like(W); Wc = torch.zeros_like(W)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); n = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(n):
            w = W1[:, j]; d = Hi[j, j]
            Wc[:, i1 + j] = w
            q = torch.round(w / s).clamp(-qmax, qmax)
            Q1[:, j] = q
            e = (w - q * s) / d
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        inv = torch.argsort(perm); Q = Q[:, inv]; Wc = Wc[:, inv]
    return Q, s, Wc


# ------------------------------------------------------------------ head
class StdHead(nn.Module):
    """hobson's PointerHead (LayerNorm -> q / k Linear -> dot * d^-1/2), fp32 copy"""
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
class Q3:
    def __init__(self, maxlen=16384):
        from plib import P
        import lean as LN, gc
        p = P(); p.model.config.max_length = maxlen
        self.p = p; self.tok = p.tok
        lean = LN.Lean(p.tm)
        self.L = lean.layers; self.eps = lean.eps; self.dev = lean.dev
        for d in self.L:
            for k, v in list(d.items()):
                if torch.is_tensor(v): d[k] = v.detach().clone()
            d['in1'] = (1.0 + d['in_norm']).float(); d['post1'] = (1.0 + d['post_norm']).float()
        self.embed = lean.embed.detach().clone(); self.norm_w = lean.norm_w.detach().clone(); self.inv = lean.inv.clone()
        self.norm1 = (1.0 + self.norm_w).float()
        self.head = StdHead(p.model.head).to(self.dev).eval()
        for q_ in self.head.parameters(): q_.requires_grad_(False)
        self.temps = {}
        p.model.torso = None; p.tm = None; del lean
        gc.collect(); torch.cuda.empty_cache()
        self.R1 = Rot(2048, 1234, self.dev); self.R2 = Rot(2048, 1235, self.dev); self.R4 = Rot(6144, 1236, self.dev)
        self.H2 = Had(self.R2); self.H4 = Had(self.R4)
        # student state: per (i, k): dict(mode, A, s, cf, wb, ab) ; precision map default plain W4A4
        self.S = {}
        self.qrow_ab = None      # optional activation bits for question rows (row-role precision), None = same as state rows

    def temp(self, kind):
        if kind not in self.temps: self.temps[kind] = self.p.temp_for(kind)
        return self.temps[kind]

    # ---------- weights in the rotated (quantization) domain = h1lib.H1.wfold(rout=True, ohead=False, ba16=False) ----------
    def wfold(self, i, k):
        d = self.L[i]; W = d[k].float()
        if k == 'Win': W = W * d['in1'][None, :]
        if k == 'Wgu': W = W * d['post1'][None, :]
        if k in ('Win', 'Wgu'): W = self.R1(W)
        else:
            W = (self.R4 if k == 'Wd' else self.R2)(W)
            W = self.R1(W.t().contiguous()).t().contiguous()
        return W

    # ---------- student configuration ----------
    def set_student(self, mode, src, wb=4, ab=4, trainable=True, pmap=None, q8src=None):
        """src: dict (i,k) -> dict(q int8, s fp32, Wc bf16 latent). pmap: optional (i,k) -> 'w4a4'|'w8a8'|'bf16'.
        q8src: optional dict (i,k) -> dict(q int8 [-127,127], s fp32): row-role format w4q8 (FORMATS.md section 8, two weight copies):
        question rows (>= q0) use these fixed W8A8 codes, state rows the (trainable) int4 path."""
        self.S = {}
        params = []
        for i in range(24):
            for k in GEMMS:
                prec = (pmap or {}).get((i, k), f'w{wb}a{ab}')
                if prec == 'bf16':
                    self.S[(i, k)] = dict(mode='bf16', A=self.wfold(i, k).to(torch.bfloat16)); continue
                w_, a_ = int(prec[1]), int(prec[3])
                e = src[(i, k)]
                s = e['s'].to(self.dev).float()
                if mode == 'lat':
                    A = nn.Parameter(e['Wc'].to(self.dev).to(torch.bfloat16).contiguous(), requires_grad=trainable)
                    st = dict(mode='lat', A=A, s=s, cf=None, qc=None)
                elif mode == 'code':
                    st = dict(mode='code', A=e['q'].to(self.dev).to(torch.int8).contiguous(), s=s, cf=None,
                              rho=nn.Parameter(torch.zeros_like(s), requires_grad=trainable))
                elif mode == 'soft':
                    st = dict(mode='soft', A=nn.Parameter(e['V'].to(self.dev).to(torch.bfloat16).contiguous(), requires_grad=trainable), s=s,
                              cf=e['cf'].to(self.dev).to(torch.int8).contiguous())
                else: raise ValueError(mode)
                st.update(wq=QMAX[w_], aq=QMAX[a_], aclip=ACLIP[a_], ab=a_, wb=w_, K=st['A'].shape[1], k=k)
                if q8src is not None:
                    st['A8'] = q8src[(i, k)]['q'].to(self.dev).to(torch.int8).contiguous(); st['s8'] = q8src[(i, k)]['s'].to(self.dev).float()
                self.S[(i, k)] = st
                params.append(st['rho'] if mode == 'code' else st['A'])
        self.refresh_codes()
        return params

    @torch.no_grad()
    def refresh_codes(self):
        """lat mode: cache the int8 codes of the current latent (call after every optimizer step)"""
        for st in self.S.values():
            if st['mode'] == 'lat': st['qc'] = _wcode_e(st['A'], st['s'], float(st['wq']))      # eager = exactly the exported codes

    def scale(self, st):
        return st['s'] * st['rho'].float().exp() if st.get('rho') is not None else st['s']

    def codes(self, i, k):
        """exported int8 codes + fp32 scales of a quantized GEMM (what the deployed kernel loads)"""
        st = self.S[(i, k)]
        with torch.no_grad():
            if st['mode'] == 'lat': q = _wcode_e(st['A'], st['s'], float(st['wq'])).float()
            elif st['mode'] == 'code': q = st['A'].float()
            else: q = (st['cf'].float() + (hsoft(st['A']) >= 0.5).float()).clamp(-st['wq'], st['wq'])
            return q.to(torch.int8), self.scale(st).detach().float().clone()

    def harden(self):
        """soft -> code mode with hard rounding (h >= 0.5)"""
        for key, st in self.S.items():
            if st['mode'] != 'soft': continue
            q, s = self.codes(*key)
            st.update(mode='code', A=q.contiguous(), s=s, cf=None, rho=None, qc=None)

    def qlin(self, x, i, k, q0=None):
        st = self.S[(i, k)]
        if st['mode'] == 'bf16':
            return (x.to(torch.bfloat16) @ st['A'].t())
        aq, aclip = st['aq'], st['aclip']; sc = self.scale(st)
        aux = st['qc'] if st['mode'] == 'lat' else st['cf']
        f1a = 0.0
        if F1 and k != 'Wgu':
            mx = st['K'] * aq * st['wq']; f1a = 1.0
            while mx * f1a > 60000: f1a *= 0.5
        if st.get('A8') is not None and q0 is not None and q0 < x.shape[0]:          # w4q8 row role
            ys = QLinF.apply(x[:q0].contiguous(), st['A'], sc, aux, st['mode'], st['wq'], aq, aclip, f1a) if q0 > 0 else None
            yq = QLinF.apply(x[q0:].contiguous(), st['A8'], st['s8'], None, 'code', 127.0, 127.0, 1.0, 0.0)
            return yq if ys is None else torch.cat([ys, yq], 0)
        if self.qrow_ab is not None and q0 is not None and q0 < x.shape[0] and self.qrow_ab != st['ab']:
            ys = QLinF.apply(x[:q0].contiguous(), st['A'], sc, aux, st['mode'], st['wq'], aq, aclip, f1a) if q0 > 0 else None
            yq = QLinF.apply(x[q0:].contiguous(), st['A'], sc, aux, st['mode'], st['wq'], QMAX[self.qrow_ab], ACLIP[self.qrow_ab], 0.0)
            return yq if ys is None else torch.cat([ys, yq], 0)
        return QLinF.apply(x.contiguous(), st['A'], sc, aux, st['mode'], st['wq'], aq, aclip, f1a)

    # ---------- tokenization ----------
    def prep(self, state, qdicts):
        """-> dict(s=state ids, qs=[dict(q ids, opt offsets, n, kind, temp)])"""
        qs = []; s0 = None
        for qd in qdicts:
            pr = self.p.prep(state, qd)
            if s0 is None: s0 = pr['s']
            assert pr['s'] == s0
            qs.append(dict(q=pr['q'], opt=list(pr['opt']), n=pr['rq'].n_slots, kind=pr['rq'].kind, temp=self.temp(pr['rq'].kind), labels=pr['rq'].slot_labels))
        return dict(s=s0, qs=qs)

    # ---------- packed layout ----------
    def layout(self, rq):
        dev = self.dev; Ls = len(rq['s']); lens = [len(x['q']) for x in rq['qs']]
        T = Ls + sum(lens)
        pos = list(range(Ls)); prev = [[t - 3 + j if t - 3 + j >= 0 else -1 for j in range(3)] for t in range(Ls)]
        seg = []; s = Ls; rows = []
        for L, x in zip(lens, rq['qs']):
            seg.append((s, s + L)); pos += list(range(Ls, Ls + L))
            for tau in range(L):
                prev.append([(s + tau - 3 + j) if tau - 3 + j >= 0 else (Ls + tau - 3 + j if Ls + tau - 3 + j >= 0 else -1) for j in range(3)])
            rows.append([s + L - 1] + [s + o for o in x['opt']])
            s += L
        ids = torch.tensor(rq['s'] + [t for x in rq['qs'] for t in x['q']], device=dev)
        posf = torch.tensor(pos, device=dev, dtype=torch.float32)
        fr = posf[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cu = torch.tensor([0] + [sum(lens[:i + 1]) for i in range(len(lens))], device=dev, dtype=torch.long)
        pv = torch.tensor(prev, device=dev, dtype=torch.long) + 1          # into [zero row; raw]
        allrows = torch.tensor([r for rr in rows for r in rr], device=dev)
        return dict(Ls=Ls, T=T, lens=lens, seg=seg, rows=rows, allrows=allrows, ids=ids, cos=fr.cos().to(torch.bfloat16), sin=fr.sin().to(torch.bfloat16),
                    cu=cu, pv=pv, M=len(lens))

    def _conv(self, raw, w, lay):
        """depthwise causal conv (k=4) + SiLU with explicit previous-row indices (state rows: own history; branch rows: own + state tail)"""
        return _fast('conv', _conv_e)(raw, w, lay['pv'])

    def _gdn(self, i, d, proj, lay):
        T = lay['T']; Ls = lay['Ls']; M = lay['M']; eps = self.eps
        raw, z, b, a = proj.split([6144, 2048, 16, 16], 1)
        beta, g = _fast('glue', _glue_e)(b, a, d['A_log'], d['dt_bias'])
        cv = self._conv(raw, d['conv_w'], lay)
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
        return _fast('gnorm', _gnorm_e)(o, z, d['gn_w'], float(eps)).reshape(T, 2048)

    def _attn(self, i, d, proj, lay):
        T = lay['T']; Ls = lay['Ls']; eps = self.eps
        qg, kk, v = proj.split([4096, 512, 512], 1)
        qh, gate = qg.reshape(T, 8, 512).split([256, 256], -1)
        kk = kk.reshape(T, 2, 256); v = v.reshape(T, 2, 256)
        qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
        qh, kk = rope(qh, lay['cos'], lay['sin']), rope(kk, lay['cos'], lay['sin'])
        qt = qh.transpose(0, 1)[None]; kt = kk.transpose(0, 1).repeat_interleave(4, 0)[None]; vt = v.transpose(0, 1).repeat_interleave(4, 0)[None]
        outs = [F.scaled_dot_product_attention(qt[:, :, :Ls], kt[:, :, :Ls], vt[:, :, :Ls], is_causal=True)[0]]
        for (s, e) in lay['seg']:
            Lb = e - s
            kb = torch.cat([kt[:, :, :Ls], kt[:, :, s:e]], 2); vb = torch.cat([vt[:, :, :Ls], vt[:, :, s:e]], 2)
            jj = torch.arange(Ls + Lb, device=proj.device)[None, :]; ii = torch.arange(Lb, device=proj.device)[:, None]
            mask = (jj < Ls) | ((jj - Ls) <= ii)
            outs.append(F.scaled_dot_product_attention(qt[:, :, s:e], kb, vb, attn_mask=mask[None, None])[0])
        o = torch.cat(outs, 1).transpose(0, 1)
        return _fast('agate', _agate_e)(o, gate).reshape(T, 2048)

    # ---------- student layer (rotated residual) ----------
    def _slayer(self, i, x, lay):
        d = self.L[i]; eps = self.eps; Ls = lay['Ls']
        proj = self.qlin(nrm32(x, eps), i, 'Win', Ls)
        o = self._gdn(i, d, proj, lay) if d['type'] == 'linear_attention' else self._attn(i, d, proj, lay)
        x = x + self.qlin(self.H2(o), i, 'Wo', Ls)
        gu = self.qlin(nrm32(x, eps), i, 'Wgu', Ls)
        m = _fast('swiglu', _swiglu_e)(gu, int(d['I']))
        return x + self.qlin(self.H4(m), i, 'Wd', Ls)

    # ---------- teacher layer (plain bf16 hobson, unrotated) ----------
    def _tlayer(self, i, x, lay):
        d = self.L[i]; eps = self.eps
        proj = rms_zc(x, d['in_norm'], eps) @ d['Win'].t()
        o = self._gdn(i, d, proj, lay) if d['type'] == 'linear_attention' else self._attn(i, d, proj, lay)
        x = x + o @ d['Wo'].t()
        gu = rms_zc(x, d['post_norm'], eps) @ d['Wgu'].t()
        return x + _fast('swiglu', _swiglu_e)(gu, int(d['I'])) @ d['Wd'].t()

    def forward(self, lay, student=True, keep=KEEP, ckpt=True):
        """-> (list of per-question log-probs [n] fp32, dict layer -> residual at readout rows in the UNROTATED basis [R, 2048] fp32)"""
        ids = lay['ids']; rows = lay['allrows']
        x = F.embedding(ids, self.embed)
        if student: x = self.R1(x).to(torch.bfloat16)
        kept = {}
        fn = self._slayer if student else self._tlayer
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(fn, i, x, lay, use_reentrant=False)
            else:
                x = fn(i, x, lay)
            if i in keep:
                xr = x.index_select(0, rows)
                kept[i] = self.R1.inv(xr) if student else xr.float()
        xr = x.index_select(0, rows)
        if student: h = (self.R1.inv(nrm32(xr, self.eps)) * self.norm1).to(torch.bfloat16)
        else: h = rms_zc(xr, self.norm_w, self.eps)
        return self.readout(h, lay), kept

    def readout(self, h, lay):
        out = []; j = 0
        for rr, qx in zip(lay['rows'], self._qs):
            hb = h[j:j + len(rr)].float(); j += len(rr)
            lg = self.head(hb[:1], hb[1:][None])[0] / qx['temp']
            out.append(torch.log_softmax(lg[:qx['n']], -1))
        return out

    def run(self, rq, student=True, keep=KEEP, ckpt=True):
        lay = self.layout(rq); self._qs = rq['qs']
        return self.forward(lay, student, keep, ckpt), lay


# ------------------------------------------------------------------ loss
def loss_fn(lt, kt, ls, ks, w_hid, nrows):
    """lt/ls: lists of log-probs; kt/ks: layer -> [R, 2048] at readout rows (R = sum of 1 + options over questions)."""
    kl = sum((a.exp() * (a - b)).sum() for a, b in zip(lt, ls)) / len(lt)
    hid = torch.zeros((), device=ls[0].device)
    if w_hid > 0 and ks:
        for i in ks:
            xs = ks[i]; xt = kt[i]
            hid = hid + ((xs - xt).pow(2).sum(-1) / xt.pow(2).sum(-1).clamp_min(1e-6)).mean()
        hid = hid / len(ks)
    return kl + w_hid * hid, float(kl.detach()), float(hid.detach())


# ------------------------------------------------------------------ data (train split only; dev = 10% of train tasks by hash, as J14)
def is_dev(task): return int(hashlib.sha1(task.encode()).hexdigest()[:8], 16) % 10 == 0


def v5_q(r):
    ins = r['instructions']
    if r['kind'] == 'choice': return {'type': 'choice', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}
    if r['kind'] == 'noul': return {'type': 'noul', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}


def load_data(maxtok=6000, ndev=100, seed=0, v5_every=8):
    W = os.path.expanduser('~/work')
    EV = set(json.load(open(f'{W}/evalkit/split.json'))['eval_tasks'])
    pool, dev = [], []
    for l in open(f'{W}/evalkit/train_pool.jsonl'):
        r = json.loads(l)
        assert r['task'] not in EV
        if r['n_state_tok'] > maxtok or r['n_state_tok'] < 32: continue
        (dev if is_dev(r['task']) else pool).append(dict(state=r['state'], questions=r['questions'], task=r['task'], rid=r['rid'], n=r['n_state_tok']))
    R = random.Random(seed); R.shuffle(pool)
    Rd = random.Random(123); Rd.shuffle(dev); dev = dev[:ndev]
    v5 = []
    for li, l in enumerate(open(f'{W}/training/data/train_v5.jsonl')):
        if li % v5_every == 0: v5.append(json.loads(l))
    R.shuffle(v5)
    return pool, dev, v5


# ------------------------------------------------------------------ optimizer (J14 arm B's AdamSR, generalised)
class AdamSR(torch.optim.Optimizer):
    """Adam on bf16 parameters: bf16 first moment, factored (Adafactor-style row/col) or bf16 second moment, fp32 math, stochastic rounding
    of the updated weight back to bf16, optional decoupled L2-SP pull toward an anchor (W <- W - lr * l2sp * (W - W0))."""

    def __init__(self, params, anchors=None, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, l2sp=0.0, factored=True, clip_el=3.0, clip_rms=1.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, l2sp=l2sp)); self.factored = factored
        self.clip_el = clip_el; self.clip_rms = clip_rms      # factored v can under-estimate an element's second moment: clip the update
        self.anchor = {id(p): a for p, a in zip(self.param_groups[0]['params'], anchors)} if anchors is not None else {}

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g['betas']; lr = g['lr']
            for p in g['params']:
                if p.grad is None: continue
                st = self.state[p]
                fac = self.factored and p.dim() == 2
                if not st:
                    st['t'] = 0; st['m'] = torch.zeros_like(p, dtype=torch.bfloat16)
                    if fac:
                        st['vr'] = torch.zeros(p.shape[0], device=p.device); st['vc'] = torch.zeros(p.shape[1], device=p.device)
                    else:
                        st['v'] = torch.zeros_like(p, dtype=torch.bfloat16)
                st['t'] += 1; t = st['t']
                gr = p.grad.float()
                m_ = st['m'].float().mul_(b1).add_(gr, alpha=1 - b1); st['m'].copy_(m_)
                if fac:
                    g2 = gr * gr + 1e-30
                    st['vr'].mul_(b2).add_(g2.mean(1), alpha=1 - b2); st['vc'].mul_(b2).add_(g2.mean(0), alpha=1 - b2)
                    v_ = (st['vr'][:, None] * st['vc'][None, :]) / st['vr'].mean().clamp_min(1e-30); del g2
                else:
                    v_ = st['v'].float().mul_(b2).addcmul_(gr, gr, value=1 - b2); st['v'].copy_(v_)
                upd = (m_ / (1 - b1 ** t)) / ((v_ / (1 - b2 ** t)).sqrt_().add_(g['eps']))
                del m_, v_, gr
                if self.clip_el: upd.clamp_(-self.clip_el, self.clip_el)
                if self.clip_rms:
                    r = float(upd.pow(2).mean().sqrt())
                    if r > self.clip_rms: upd.mul_(self.clip_rms / r)
                w = p.float()
                if g['l2sp'] > 0 and id(p) in self.anchor:
                    w.sub_((w - self.anchor[id(p)].float()).mul_(lr * g['l2sp']))
                w.add_(upd, alpha=-lr); del upd
                wi = w.view(torch.int32)
                wi.add_(torch.randint_like(wi, 0, 1 << 16)).bitwise_and_(-65536)
                p.copy_(wi.view(torch.float32).to(torch.bfloat16))
                del w, wi


def adaround_reg_grad(V, beta):
    """d/dV of (1 - |2h - 1|^beta), h = clamp(sigmoid(V) * (zeta - gamma) + gamma, 0, 1)"""
    sg = torch.sigmoid(V.float()); h = sg * (ZETA - GAMMA) + GAMMA
    live = ((h > 0) & (h < 1)).float(); hc = h.clamp(0, 1); u = 2 * hc - 1
    dR = -2 * beta * u.abs().pow(beta - 1) * torch.sign(u)
    return dR * (ZETA - GAMMA) * sg * (1 - sg) * live


@torch.no_grad()
def adaround_reg(V, beta):
    hc = hsoft(V); return float((1 - (2 * hc - 1).abs().pow(beta)).mean())


# ------------------------------------------------------------------ evalkit scoring (reporting only; never used for training or selection)
def eval_suites(m, suites, out, student=True, maxq=8, max_rows=14000, log=print, fast=False):
    """every question of the given evalkit suites through the model; shared-prefix packing per item; writes {suite,id,q,T,probs} JSONL"""
    import collections, time
    sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
    import evalkit as EK
    global FAST
    fast0 = FAST; FAST = bool(fast)
    by = collections.OrderedDict()
    for s, iid, q, st, spec in EK.all_question_items(suites): by.setdefault(iid, []).append((s, q, st, spec))
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
    FAST = fast0
    log('eval_suites', suites, out, n, f'{time.time() - t0:.0f}s')
    return n
