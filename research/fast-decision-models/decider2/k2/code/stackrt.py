"""K2 stackrt: one fused inference runtime for the combined stack on H2's QRT kernels + J5's short-input GEMMs.

Flags (any combination):
  D  depth split: live state rows run layers 0..LS-1 (LS = 8); question rows run all 24. Deep attention layers read the state through a
     memory K/V computed from the state rows' layer-LS input with each deep layer's own K/V projection (+ rank-32 memory adapter), as one GEMM
     (folded norms; int8 on the same quantized codes as layer LS's Win input in W8A8). Deep GDN layers start question rows from a zero state.
  C  compiled deployment documents: the request's compiled rows (frame + constant blocks) are gathered from the deployment library inside the
     graph (post-norm pre-RoPE K and V per attention layer; under D also the deep memory K/V), RoPE'd at their runtime positions and placed
     ahead of the live rows. GDN sees the compiled prefix only through a per-request initial state and conv tail (K/V-only library: blocks are
     GDN-invisible, the frame's state is the start state) unless the plan supplies a composed state.
  V  16k super-token vocabulary: ids index an extended embedding table; RoPE positions are given per row (original Qwen positions).
  Q  deployed questions in the weights: each deployed question is a branch of K+1 slot rows (learned input vectors) with its stacked LoRA delta
     (shared r16 + own r8) on Win / Wo / Wd of its slot rows only; other questions stay in context as token branches.
  P  W8A8-b8 (H1 format: R1/R2/R4 Hadamard rotations, int8 codes, per-token absmax activations, the 8 GEMMs of precmap_w8a8_b8 in bf16).
  X  exit at layer 16: graph A = layers 0..15 + exit head on the layer-16 residual; graph B = layers 16..23 + hobson's head, run only for the
     questions that did not exit (exact: question branches are independent).
A CUDA graph per exact shape. Host side builds a Spec (rows, positions, branches); Req turns it into static device tensors and run functions.
"""
import os, sys, math, json, time, statistics as stt
for p in ('~/work/k2', '~/work/j5', '~/work/h2/code', '~/work/h2', '~/work/d1', '~/work/systems/g', '~/work/evalkit', '~/work/tokens'):
    p = os.path.expanduser(p)
    if p not in sys.path: sys.path.append(p)
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F
import triton, triton.language as tl
from torch.nn.attention.bias import causal_lower_right
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

dev = 'cuda'
NL = 24
ATT = (3, 7, 11, 15, 19, 23)
LS_DEFAULT = 8
XL_DEFAULT = 16
GDN_KW = dict(use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True)


# ====================================================================== small kernels
@triton.jit
def _kvprep_k(P, RA, CS, KN, COS, SIN, KB, VB, koff, ps, colk, colv, eps, DQ: tl.constexpr, D: tl.constexpr):
    """memory K/V of one deep attention layer from a [rows, *] projection: K = rope(rmsnorm_zc(k) ), V = v; written at rows koff + r."""
    r = tl.program_id(0).to(tl.int64); s = tl.program_id(1)
    c = tl.arange(0, D)
    pc = tl.where(c < 32, c + 32, tl.where(c < 64, c - 32, c))
    if DQ:
        ra = tl.load(RA + r)
    if s < 2:
        col = colk + s * D
        w = tl.load(KN + c); wp = tl.load(KN + pc)
        x = tl.load(P + r * ps + col + c).to(tl.float32)
        xp = tl.load(P + r * ps + col + pc).to(tl.float32)
        if DQ:
            x = ((x * ra) * tl.load(CS + col + c)).to(tl.bfloat16).to(tl.float32)
            xp = ((xp * ra) * tl.load(CS + col + pc)).to(tl.bfloat16).to(tl.float32)
        rs = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        xn = (x * rs * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
        xpn = (xp * rs * (1.0 + wp)).to(tl.bfloat16).to(tl.float32)
        cm = c < 64
        cs_ = tl.load(COS + r * 64 + c, mask=cm, other=1.0).to(tl.float32)
        sn = tl.load(SIN + r * 64 + c, mask=cm, other=0.0).to(tl.float32)
        sign = tl.where(c < 32, -1.0, 1.0)
        y = tl.where(cm, xn * cs_ + sign * xpn * sn, xn)
        tl.store(KB + (koff + r) * 2 * D + s * D + c, y.to(tl.bfloat16))
    else:
        col = colv + (s - 2) * D
        x = tl.load(P + r * ps + col + c).to(tl.float32)
        if DQ:
            x = (x * ra) * tl.load(CS + col + c)
        tl.store(VB + (koff + r) * 2 * D + (s - 2) * D + c, x.to(tl.bfloat16))


def kvprep(y, ra, cs, kn, cos, sin, kb, vb, koff, colk, colv, eps, dq):
    R = y.shape[0]
    if R == 0: return
    _kvprep_k[(R, 4)](y, ra if ra is not None else y, cs if cs is not None else y, kn, cos, sin, kb, vb, koff, y.stride(0), colk, colv, eps,
                      DQ=dq, D=256, num_warps=4)


@triton.jit
def _ropeg_k(KP, IDX, COS, SIN, KB, koff, D: tl.constexpr):
    """K buffer row koff + r <- rope(KP[IDX[r]]) (KP post-norm pre-RoPE bf16 [N, 2, D]); two heads per program row."""
    r = tl.program_id(0).to(tl.int64); h = tl.program_id(1)
    src = tl.load(IDX + r).to(tl.int64)
    c = tl.arange(0, D)
    pc = tl.where(c < 32, c + 32, tl.where(c < 64, c - 32, c))
    xn = tl.load(KP + src * 2 * D + h * D + c).to(tl.float32)
    xpn = tl.load(KP + src * 2 * D + h * D + pc).to(tl.float32)
    cm = c < 64
    cs_ = tl.load(COS + r * 64 + c, mask=cm, other=1.0).to(tl.float32)
    sn = tl.load(SIN + r * 64 + c, mask=cm, other=0.0).to(tl.float32)
    sign = tl.where(c < 32, -1.0, 1.0)
    y = tl.where(cm, xn * cs_ + sign * xpn * sn, xn)
    tl.store(KB + (koff + r) * 2 * D + h * D + c, y.to(tl.bfloat16))


def ropeg(kpre, idx, cos, sin, kb, koff=0):
    n = idx.shape[0]
    if n == 0: return
    _ropeg_k[(n, 2)](kpre, idx, cos, sin, kb, koff, D=256, num_warps=4)


def rope_t(t, cos, sin):
    """reference RoPE (rotate-half on the first 64 of 256 dims), t [N, H, 256], cos/sin [N, 64]"""
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


def cos_sin(pos, inv):
    fr = pos[:, None].float() * inv[None, :]; fr = torch.cat([fr, fr], -1)
    return fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()


# ====================================================================== base model (QRT + J5 kernels)
PREC_B8 = 'w8a8:' + os.path.expanduser('~/work/h1/precmap_w8a8_b8.json')


def load_base(prec='bf16', codes=None, lora=None):
    """-> (P, m): P = plib.P (engine, tokenizer, head, temps), m = J5 runtime ('bf16' FoldSK or 'w8a8' IntSK on the b8 map).
    codes: optional {(i, k): (q, s)} W8 codes (GPTQ); default RTN (speed does not depend on code values)."""
    from kitrun import load_P
    from lean2 import Lean2
    import j5rt, qrt as Q
    P = load_P()
    head = P.model.head.float().eval()
    ln = Lean2(P.tm, fuse='fold' if prec == 'bf16' else '')
    j5rt.slim(ln)
    if prec == 'bf16':
        m = j5rt.make(ln, head, 'bf16')
    else:
        j5rt.patch_qg()
        m = j5rt.IntSK(ln, head=head, prec='map:' + PREC_B8.split(':', 1)[1] + ':w8a8', wcodes=codes, lora=lora); m.tune = False
    m.prec = prec
    return P, m


def free_bf16_copies(m):
    """drop Lean2 weight copies the runtime no longer reads (after deployment objects that need raw weights are built)."""
    import gc
    for d in m.L:
        for k in ('Wgu_il', 'Win_f', 'Wgu_f', 'Win', 'Wgu', 'Wo', 'Wd'):
            d.pop(k, None)
    gc.collect(); torch.cuda.empty_cache()


# ====================================================================== deployment objects
class Mem:
    """D: memory K/V weights of the deep attention layers (ordered), concatenated per group (one GEMM per group).
    adapters: {i: (A [r, 2048], B [1024, r], scale)} rank-32 memory adapters on the K/V rows; lora_win: {i: dW [1024, 2048]} optional extra."""

    def __init__(self, m, LS=LS_DEFAULT, adapters=None, groups=None, dW=None):
        self.LS = LS
        deep = [i for i in ATT if i >= LS]
        self.groups = groups or [deep]
        self.e = []
        for grp in self.groups:
            Ws = []
            for i in grp:
                d = m.L[i]
                W = d['Win'][4096:5120].float()
                if dW is not None and i in dW: W = W + dW[i].float().to(W.device)
                if adapters is not None and i in adapters:
                    A, B, sc = adapters[i]; W = W + sc * (B.float().to(W.device) @ A.float().to(W.device))
                Ws.append(W * d['in1'][None, :])
            W = torch.cat(Ws, 0).contiguous()
            self.e.append(self._pack(m, W))
            del W
        torch.cuda.empty_cache()

    @staticmethod
    def _pack(m, W):
        import sk, qrt as Q
        if m.fold:
            return dict(qw=sk.QW('bf16', W.to(torch.bfloat16).contiguous()), N=W.shape[0])
        Wr = m.R1(W)
        s = Q.rtn_scales(Wr, 127.0, False); q = torch.round(Wr / s[:, None]).clamp(-127, 127)
        e = dict(codes=q.to(torch.int8).contiguous(), kind='s8', prec='w8a8', abits=8, N=W.shape[0])
        e['alpha'] = Q.alpha_for('s8', 2048); e['csa'] = (s / e['alpha']).float().contiguous()
        return e

    def set_codes(self, gi, q, s):
        """replace group gi's int8 codes / scales (GPTQ from a calibration)"""
        import qrt as Q
        e = self.e[gi]; e['codes'] = q.to(torch.int8).to(dev).contiguous(); e['csa'] = (s.float().to(dev) / e['alpha']).contiguous()


class QTab:
    """Q: deployed question table. Per question name: slot input vectors [K+1, 2048] (unrotated), K, kind/temp, and LoRA deltas per layer and
    module {(i, mod): (A [r, in], B [out, r])} with scale folded into B (shared and own parts concatenated along r)."""

    def __init__(self):
        self.q = {}

    def add(self, name, slots, K, temp, n_slots, lora, labels=None):
        self.q[name] = dict(slots=slots, K=K, temp=temp, n_slots=n_slots, lora=lora, labels=labels)

    @staticmethod
    def random(names_K, r=24, seed=0, scaleB=1e-3, embed=None, temp=1.0):
        """latency / fidelity stand-in: random adapters (non-zero B) and slot vectors"""
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        t = QTab()
        for name, K in names_K:
            lo = {}
            for i in range(NL):
                for mod, (out, inp) in (('Win', (8224 if i not in ATT else 5120, 2048)), ('Wo', (2048, 2048)), ('Wd', (2048, 6144))):
                    lo[(i, mod)] = ((torch.randn(r, inp, generator=g) / inp ** 0.5), torch.randn(out, r, generator=g) * scaleB)
            sl = torch.randn(K + 1, 2048, generator=g) * 0.02
            t.add(name, sl, K, temp, K, lo)
        return t


class ExitHead(nn.Module):
    """J15's exit head (j15learn.EH): pointer head + rank-512 residual adapter on the layer-16 residual."""

    def __init__(self, base, r=512):
        super().__init__()
        self.pre = nn.LayerNorm(2048, elementwise_affine=False)
        self.a1 = nn.Linear(2048, r); self.a2 = nn.Linear(r, 2048)
        nn.init.zeros_(self.a2.weight); nn.init.zeros_(self.a2.bias)
        self.norm = nn.LayerNorm(2048); self.q = nn.Linear(2048, base.q.out_features); self.k = nn.Linear(2048, base.k.out_features)
        self.norm.load_state_dict(base.norm.state_dict()); self.q.load_state_dict(base.q.state_dict()); self.k.load_state_dict(base.k.state_dict())
        self.scale = base.q.out_features ** -0.5
        self.gain = nn.Parameter(torch.ones(2048) * 8.0)

    def emb(self, x):
        x = self.pre(x) * self.gain
        return x + self.a2(F.gelu(self.a1(x)))

    def forward(self, d, o):
        dq = self.q(self.norm(self.emb(d)))
        ok = self.k(self.norm(self.emb(o)))
        return (ok @ dq.unsqueeze(-1)).squeeze(-1) * self.scale


class Lib:
    """C: deployment library. Per attention layer i a post-norm pre-RoPE K [N, 2, 256] and V [N, 2, 256] (bf16) over all compiled rows of the
    deployment (frame and blocks, at library row indices); under D, deep layers hold the memory K/V of the same rows. Per GDN layer, raw pre-conv
    rows [N, 6144] (fp32, true scale) for the conv tails. Requests reference library rows by index."""

    def __init__(self, K, V, raw=None):
        self.K = K; self.V = V; self.raw = raw or {}
        self.N = next(iter(K.values())).shape[0] if K else 0

    @staticmethod
    def random(N, layers, seed=0, gdn_layers=()):
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        K = {}; V = {}
        for i in layers:
            k = torch.randn(N, 2, 256, generator=g)
            K[i] = (k / k.pow(2).mean(-1, keepdim=True).sqrt()).to(dev, torch.bfloat16).contiguous()
            V[i] = (torch.randn(N, 2, 256, generator=g) * 0.5).to(dev, torch.bfloat16).contiguous()
        raw = {i: (torch.randn(min(N, 64), 6144, generator=g) * 0.5).to(dev) for i in gdn_layers}
        return Lib(K, V, raw)


# ====================================================================== host-side request description
class Branch:
    """one question. kind 'ctx': ids = question token ids (ends with '<answer>'), opt = option-end offsets; kind 'slot': name = deployed question.
    pos: runtime positions of its rows (len = rows)."""

    def __init__(self, kind, pos, opt, temp, n_slots, ids=None, name=None, label=None):
        self.kind = kind; self.ids = ids; self.name = name; self.pos = list(pos); self.opt = list(opt); self.temp = temp
        self.n_slots = n_slots; self.label = label
        self.L = len(ids) if kind == 'ctx' else len(pos)


class Spec:
    """live: state token ids computed per request and their positions; prefix: dict(rows=[library row idx], pos=[runtime positions],
    S0={gdn layer: [1,16,128,128] fp32} or None, tail_rows={gdn layer: [3 library raw-row idx or -1]}) or None; branches: list of Branch."""

    def __init__(self, live_ids, live_pos, branches, prefix=None, D=False, X=False, LS=LS_DEFAULT, XL=XL_DEFAULT, vocab_ext=False, npre=0):
        self.live_ids = list(live_ids); self.live_pos = list(live_pos); self.branches = branches; self.prefix = prefix
        self.npre = npre if prefix else 0      # C: live rows that precede the compiled blocks in runtime order (they do not see the blocks)
        self.D = D; self.X = X; self.LS = LS; self.XL = XL; self.vocab_ext = vocab_ext

    @property
    def Ls(self): return len(self.live_ids)

    @property
    def R(self): return sum(b.L for b in self.branches)


# ====================================================================== the runtime
class StackRT:
    def __init__(self, P, m, qtab=None, mem=None, lib=None, ext=None, eh=None):
        self.P = P; self.m = m; self.fold = m.fold
        self.qtab = qtab; self.mem = mem; self.lib = lib; self.eh = eh
        self.eps = m.eps; self.inv = m.ln2.inv; self.L = m.L
        self.head = m.head
        self.norm1 = m.norm1
        base = m.ln2.embed if self.fold else m.embed
        self.E = base
        if ext is not None:      # V: extended table rows (unrotated) appended after the base vocabulary rows
            V0, rows = ext
            rows = rows.to(dev).float()
            if not self.fold: rows = m.R1(rows)
            self.E = torch.cat([base[:V0], rows.to(torch.bfloat16)], 0).contiguous()
        self._lora_cache = {}

    # ------------------------------------------------------------------ rotations for W8A8 LoRA folding
    def _rot_in(self, i, mod, A):
        """A [r, in] -> A' such that delta = (input as the runtime holds it) @ A'^T. fold: unchanged except Win (norm gain folded)"""
        m = self.m; A = A.float().to(dev)
        if mod == 'Win':
            A = A * m.L[i]['in1'][None, :]
            return A if self.fold else m.R1(A)
        return A          # Wo / Wd deltas read the exact bf16 (unrotated) GEMM input in both runtimes

    def _rot_out(self, mod, B):
        """B [out, r] -> Bt' [r, out] in the runtime's output basis (residual writers rotate by R1 in W8A8)"""
        Bt = B.float().to(dev).t().contiguous()
        if mod in ('Wo', 'Wd') and not self.fold: Bt = self.m.R1(Bt)
        return Bt

    def lora_stack(self, names):
        """stacked per-question deltas for an ordered question list: {(i, mod): (A [n, in, r] bf16, Bt [n, r, out] bf16)}"""
        key = tuple(names)
        if key in self._lora_cache: return self._lora_cache[key]
        out = {}
        for i in range(NL):
            for mod in ('Win', 'Wo', 'Wd'):
                As = []; Bs = []
                for nm in names:
                    A, B = self.qtab.q[nm]['lora'][(i, mod)]
                    As.append(self._rot_in(i, mod, A).t()); Bs.append(self._rot_out(mod, B))
                out[(i, mod)] = (torch.stack(As).to(torch.bfloat16).contiguous(), torch.stack(Bs).to(torch.bfloat16).contiguous())
        self._lora_cache[key] = out
        return out

    def slot_rows(self, name):
        x = self.qtab.q[name]['slots'].to(dev).float()
        if not self.fold: x = self.m.R1(x)
        return x.to(torch.bfloat16)


class Req:
    """static device state + run functions for one exact request shape (a CUDA graph per instance).
    lib / mem: the deployment library (C) and memory weights (D) for this request's configuration (default rt.lib / rt.mem; rt.mem may be a dict
    {'full': Mem(groups=[[11,15,19,23]]), 'split': Mem(groups=[[11,15],[19,23]])}, picked by X)."""

    def __init__(self, rt, spec, lib=None, mem=None):
        self.rt = rt; self.s = spec; m = rt.m; self.m = m
        self.lib = lib if lib is not None else rt.lib
        mem = mem if mem is not None else rt.mem
        if isinstance(mem, dict): mem = mem['split' if spec.X else 'full']
        self.memo = mem
        Ls = spec.Ls; R = spec.R; T = Ls + R; self.Ls = Ls; self.R = R; self.T = T
        br = spec.branches; self.n = len(br)
        self.LS = spec.LS if spec.D else NL
        self.XL = spec.XL if spec.X else NL
        # ---- ids / input rows
        self.ids = torch.tensor(spec.live_ids + [t for b in br if b.kind == 'ctx' for t in b.ids], device=dev, dtype=torch.long)
        self.nid = self.ids.shape[0]
        # row order: [live][ctx branches][slot branches]  (branches are emitted in the order given; we require ctx first, slot last)
        kinds = [b.kind for b in br]
        assert kinds == sorted(kinds), 'ctx branches must come before slot branches'
        self.slot_names = [b.name for b in br if b.kind == 'slot']
        self.s0 = Ls + sum(b.L for b in br if b.kind == 'ctx')       # first slot row
        if self.slot_names:
            self.xslot = torch.cat([rt.slot_rows(nm) for nm in self.slot_names], 0).contiguous()
            self.lora = rt.lora_stack(self.slot_names)
            lens = [b.L for b in br if b.kind == 'slot']; Lm = max(lens); r0 = [0]
            for L_ in lens: r0.append(r0[-1] + L_)
            pad = torch.zeros(len(lens), Lm, dtype=torch.long); unpad = []
            for j, L_ in enumerate(lens):
                pad[j] = torch.arange(r0[j], r0[j] + Lm).clamp(max=r0[j] + L_ - 1); unpad += list(range(j * Lm, j * Lm + L_))
            self.lpad = pad.to(dev); self.lunpad = torch.tensor(unpad, device=dev); self.nslot = r0[-1]
        else:
            self.xslot = None; self.lora = None; self.nslot = 0
        # ---- positions
        pos = spec.live_pos + [p for b in br for p in b.pos]
        assert len(pos) == T
        self.pos = torch.tensor(pos, device=dev, dtype=torch.float32)
        self.cos, self.sin = cos_sin(self.pos, rt.inv)
        # ---- branch geometry
        self.br0 = []; r = Ls
        for b in br: self.br0.append(r); r += b.L
        self.cu = torch.tensor([x - Ls for x in self.br0] + [R], device=dev, dtype=torch.long)
        self.single = (self.n == 1)          # one branch: [state][branch] is one causal sequence in the shared layers
        # ---- prefix (C)
        pf = spec.prefix
        self.P = len(pf['rows']) if pf else 0
        if pf:
            self.prow = torch.tensor(pf['rows'], device=dev, dtype=torch.int32)
            ppos = torch.tensor(pf['pos'], device=dev, dtype=torch.float32)
            self.pcos, self.psin = cos_sin(ppos, rt.inv)
            self.S0 = pf.get('S0') or {}
            self.tail = {}
            for i, d in enumerate(m.L):
                if d['type'] == 'linear_attention' and pf.get('tail'):
                    tr = pf.get('tail_rows', {}).get(i)
                    if tr is not None and self.lib is not None and i in self.lib.raw:
                        t3 = torch.stack([self.lib.raw[i][j] if j >= 0 else torch.zeros(6144, device=dev) for j in tr], 0)
                    else:
                        t3 = pf.get('tail', {}).get(i, torch.zeros(3, 6144, device=dev))
                    self.tail[i] = t3.float().contiguous()
        else:
            self.S0 = {}; self.tail = {}
        self.npre = spec.npre
        has_tail = bool(pf and pf.get('tail'))
        # ---- conv history (shared layers): logical history = [tail(3) if prefix][live][branch]
        prev = []
        for t in range(Ls):
            prev.append([t - 3 + j if t - 3 + j >= 0 else ((-2 - (t + j)) if has_tail else -1) for j in range(3)])
        for k, b in enumerate(br):
            b0 = self.br0[k]
            for tau in range(b.L):
                pv = []
                for j in range(3):
                    pp = tau - 3 + j
                    if pp >= 0: pv.append(b0 + pp)
                    else:
                        sp = Ls + pp
                        pv.append(sp if sp >= 0 else ((-2 - (3 + sp)) if has_tail else -1))
                prev.append(pv)
        self.prev = torch.tensor(prev, device=dev, dtype=torch.int32).contiguous()
        # deep layers (D): branch rows only, zero history
        prevd = []
        for k, b in enumerate(br):
            b0 = self.br0[k] - Ls
            for tau in range(b.L):
                prevd.append([b0 + tau - 3 + j if tau - 3 + j >= 0 else -1 for j in range(3)])
        self.prevd = torch.tensor(prevd, device=dev, dtype=torch.int32).contiguous()
        self.ztail = torch.zeros(1, 6144, device=dev)
        # ---- attention masks
        P_ = self.P
        if not self.single:
            mk = torch.zeros(R, P_ + T, dtype=torch.bool, device=dev); mk[:, :P_ + Ls] = True
            for k, b in enumerate(br):
                a = self.br0[k] - Ls
                mk[a:a + b.L, P_ + Ls + a:P_ + Ls + a + b.L] = torch.tril(torch.ones(b.L, b.L, dtype=torch.bool, device=dev))
            self.bmask = mk[None, None]
        if spec.D:
            Pm = self.P; Kd = Pm + Ls + R
            if self.single:
                self.dmask = None
            else:
                mk = torch.zeros(R, Kd, dtype=torch.bool, device=dev); mk[:, :Pm + Ls] = True
                for k, b in enumerate(br):
                    a = self.br0[k] - Ls
                    mk[a:a + b.L, Pm + Ls + a:Pm + Ls + a + b.L] = torch.tril(torch.ones(b.L, b.L, dtype=torch.bool, device=dev))
                self.dmask = mk[None, None]
        # ---- readout rows (relative to branch rows) and temperatures
        Kmax = max(len(b.opt) for b in br)
        ans = []; opt = []; msk = []
        for k, b in enumerate(br):
            a = self.br0[k] - Ls
            ans.append(a + b.L - 1)
            o = [a + x for x in b.opt]; opt.append(o + [o[0]] * (Kmax - len(o))); msk.append([j >= b.n_slots for j in range(Kmax)])
        self.ans = torch.tensor(ans, device=dev); self.opt = torch.tensor(opt, device=dev)
        self.omask = torch.tensor(msk, device=dev); self.temps = torch.tensor([b.temp for b in br], device=dev, dtype=torch.float32)[:, None]
        self.Kmax = Kmax
        # ---- static buffers
        self.x = torch.empty(T, 2048, device=dev, dtype=torch.bfloat16)
        if spec.D:
            self.Mq = None; self.Ms = None; self.ssM = torch.zeros(max(Ls, 1), device=dev)
            # deep memory K/V buffers (per deep attention layer): [prefix mem | state mem | branch rows]
            self.dkb = {i: torch.zeros(self.P + T, 2, 256, device=dev, dtype=torch.bfloat16) for i in ATT if i >= self.LS}
            self.dvb = {i: torch.zeros(self.P + T, 2, 256, device=dev, dtype=torch.bfloat16) for i in ATT if i >= self.LS}
            if not self.fold:
                self.Mq = torch.empty(max(Ls, 1), 2048, device=dev, dtype=torch.int8); self.Ms = torch.empty(max(Ls, 1), device=dev)
        self.kb = {i: torch.zeros(self.P + T, 2, 256, device=dev, dtype=torch.bfloat16) for i in ATT}
        self.vb = {i: torch.zeros(self.P + T, 2, 256, device=dev, dtype=torch.bfloat16) for i in ATT}

    @property
    def fold(self): return self.rt.fold

    # ------------------------------------------------------------------ helpers
    def _bf16_input(self, cur, rows):
        """dequantized input rows of a GEMM (W8A8: int8 codes * per-token scale; bf16 GEMMs: the bf16 input)"""
        if cur['outq']: return (cur['q'][rows].float() * cur['s'][rows, None]).to(torch.bfloat16)
        return cur['hb'][rows]

    def _delta(self, i, mod, hin):
        """per-question LoRA delta on the slot rows: hin [nslot, in] -> [nslot, out]"""
        A, Bt = self.lora[(i, mod)]
        n = A.shape[0]
        hp = hin[self.lpad] if n > 1 else hin[None]
        y = torch.bmm(torch.bmm(hp, A), Bt)
        return y.reshape(-1, Bt.shape[2])[self.lunpad] if n > 1 else y[0]

    def _slots(self, T0):
        """slot-row slice in the current row set (T0 = first row of the current set in full-row numbering)"""
        return slice(self.s0 - T0, self.s0 - T0 + self.nslot)

    # ------------------------------------------------------------------ mixers
    def _gdn(self, i, d, proj, ra, cs, dq, deep):
        if deep:
            qkv3, ab = self.K.conv(proj, ra, cs, d['conv_w'], self.prevd, self.ztail, dq=dq)
            q, k, v = qkv3[0], qkv3[1], qkv3[2]; R = proj.shape[0]
            kw = dict(GDN_KW, A_log=d['A_log'], dt_bias=d['dt_bias'])
            o, _ = chunk_gated_delta_rule(q[None], k[None], v[None], ab[:, 16:32][None], ab[:, 0:16][None],
                                          cu_seqlens=None if self.single else self.cu, **kw)
            return o[0]
        T = proj.shape[0]; Ls = self.Ls
        tail = self.tail.get(i, self.ztail)
        qkv3, ab = self.K.conv(proj, ra, cs, d['conv_w'], self.prev, tail, dq=dq)
        q, k, v = qkv3[0], qkv3[1], qkv3[2]
        kw = dict(GDN_KW, A_log=d['A_log'], dt_bias=d['dt_bias'])
        S0 = self.S0.get(i)
        if self.single:
            o, _ = chunk_gated_delta_rule(q[None], k[None], v[None], ab[:, 16:32][None], ab[:, 0:16][None], initial_state=S0, **kw)
            return o[0]
        if Ls > 0:
            o1, S = chunk_gated_delta_rule(q[None, :Ls], k[None, :Ls], v[None, :Ls], ab[:Ls, 16:32][None], ab[:Ls, 0:16][None],
                                           initial_state=S0, output_final_state=True, **kw)
        else:
            o1 = None; S = S0
        init = S.expand(self.n, -1, -1, -1).contiguous() if S is not None else None
        o2, _ = chunk_gated_delta_rule(q[None, Ls:], k[None, Ls:], v[None, Ls:], ab[Ls:, 16:32][None], ab[Ls:, 0:16][None],
                                       initial_state=init, cu_seqlens=self.cu, **kw)
        return o2[0] if o1 is None else torch.cat([o1[0], o2[0]], 0)

    def _prefix_kv(self, i, kb, vb, deep):
        """gather + RoPE the request's compiled rows into the front of this layer's K/V buffer (inside the graph)"""
        lib = self.lib
        ropeg(lib.K[i], self.prow, self.pcos, self.psin, kb, 0)
        torch.index_select(lib.V[i], 0, self.prow.long(), out=vb[:self.P])

    def _attn(self, i, d, proj, ra, cs, dq, deep):
        P_ = self.P; Ls = self.Ls
        if deep:
            kb, vb = self.dkb[i], self.dvb[i]; R = proj.shape[0]
            cos, sin = self.cos[Ls:], self.sin[Ls:]
            q = self.K.aprep(proj, ra, cs, d['qn'], d['kn'], cos, sin, kb, vb, P_ + Ls, self.eps, dq=dq)
            qh = q.transpose(0, 1)[None]; Lk = P_ + Ls + R
            if self.single:
                o = F.scaled_dot_product_attention(qh, kb[:Lk].transpose(0, 1)[None], vb[:Lk].transpose(0, 1)[None],
                                                   attn_mask=causal_lower_right(R, Lk), enable_gqa=True)
            else:
                o = F.scaled_dot_product_attention(qh, kb[:Lk].transpose(0, 1)[None], vb[:Lk].transpose(0, 1)[None], attn_mask=self.dmask, enable_gqa=True)
            return o[0]
        T = proj.shape[0]
        kb, vb = self.kb[i], self.vb[i]
        if P_: self._prefix_kv(i, kb, vb, False)
        q = self.K.aprep(proj, ra, cs, d['qn'], d['kn'], self.cos, self.sin, kb, vb, P_, self.eps, dq=dq)
        qh = q.transpose(0, 1)[None]
        npre = self.npre
        sdpa = F.scaled_dot_product_attention
        def kv(a, b): return kb[a:b].transpose(0, 1)[None], vb[a:b].transpose(0, 1)[None]
        def lr(Tq, Lk): return dict(is_causal=True) if Tq == Lk else dict(attn_mask=causal_lower_right(Tq, Lk))
        outs = []
        if npre:       # live rows before the compiled blocks see only themselves
            outs.append(sdpa(qh[:, :, :npre], *kv(P_, P_ + npre), is_causal=True, enable_gqa=True)[0])
        if self.single:
            Tq = T - npre; Lk = P_ + T
            outs.append(sdpa(qh[:, :, npre:], *kv(0, Lk), enable_gqa=True, **lr(Tq, Lk))[0])
        else:
            if Ls > npre:
                Tq = Ls - npre; Lk = P_ + Ls
                outs.append(sdpa(qh[:, :, npre:Ls], *kv(0, Lk), enable_gqa=True, **lr(Tq, Lk))[0])
            outs.append(sdpa(qh[:, :, Ls:], *kv(0, P_ + T), attn_mask=self.bmask, enable_gqa=True)[0])
        return torch.cat(outs, 1) if len(outs) > 1 else outs[0]

    def _mem(self, gi):
        """D: deep memory K/V of group gi from the state rows' layer-LS input (+ the prefix's compiled memory rows)"""
        mem = self.memo; e = mem.e[gi]; Ls = self.Ls; P_ = self.P
        grp = mem.groups[gi]
        if Ls > 0:
            if self.fold:
                import sk
                y = sk.launch(self.x[:Ls], e['qw'], 0, ss=self.ssM[:Ls], Kd=2048); ra = cs = None; dq = False
            else:
                y = self.m.qgemm(self.Mq[:Ls], e); ra = self.Ms[:Ls]; cs = e['csa']; dq = True
        for j, i in enumerate(grp):
            kb, vb = self.dkb[i], self.dvb[i]
            if P_: self._prefix_kv(i, kb, vb, True)
            if Ls > 0:
                kvprep(y, ra, cs, self.L[i]['kn'], self.cos[:Ls], self.sin[:Ls], kb, vb, P_, j * 1024, j * 1024 + 512, self.eps, dq)

    # ------------------------------------------------------------------ layer loops
    def _layers_fold(self, i0, i1):
        """bf16 fold runtime (FoldSK sk GEMMs). Rows: full set for i < LS, branch rows after."""
        m = self.m; LS = self.LS; Ls = self.Ls
        for i in range(i0, i1):
            d = self.L[i]; deep = i >= LS
            T0 = Ls if deep else 0
            x = self.x[T0:]; T = x.shape[0]
            if i == i0 or i == LS:
                ss = x.float().pow(2).sum(-1)
            proj = m.g(x, i, 'Win', 0, ss=ss)
            sl = self._slots(T0) if self.nslot else None
            if sl is not None:
                xs = x[sl].float(); h = (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + self.eps)).to(torch.bfloat16)
                proj[sl] += self._delta(i, 'Win', h)
            y = torch.empty(T, 2048, device=dev, dtype=torch.bfloat16)
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, None, None, False, deep)
                self.K._gnorm_k[(triton.cdiv(T, m.rows),)](o, proj[:, 6144:], proj, proj, d['gn_w'], proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), self.eps,
                                                           DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=m.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, None, None, False, deep)
                self.K._agate_k[(triton.cdiv(T, m.rows),)](o, proj, proj, proj, proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), o.stride(0), o.stride(1),
                                                           DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=m.rows, num_warps=8)
            if sl is not None: x[sl] += self._delta(i, 'Wo', y[sl])
            ss = torch.zeros(T, device=dev, dtype=torch.float32)
            m.g(y, i, 'Wo', 3, res=x, ssout=ss)
            mm = m.g(x, i, 'Wgu', 1, ss=ss)
            if sl is not None: x[sl] += self._delta(i, 'Wd', mm[sl])
            ss = torch.zeros(T, device=dev, dtype=torch.float32)
            m.g(mm, i, 'Wd', 3, res=x, ssout=ss)
            if self.s.D and i + 1 == LS and Ls > 0:
                self.ssM[:Ls].copy_(ss[:Ls])
                self._mem(0)

    def _layers_q(self, i0, i1):
        """rotated W8A8-b8 runtime (IntSK). x holds x R1."""
        m = self.m; LS = self.LS; Ls = self.Ls; eps = self.eps; K = self.K; QG = self.QG
        for i in range(i0, i1):
            d = self.L[i]; deep = i >= LS
            T0 = Ls if deep else 0
            x = self.x[T0:]; T = x.shape[0]
            sl = self._slots(T0) if self.nslot else None
            if i == i0:
                cur = m._next_in(i, 'Win', T)
                K.addq(x, None, None, None, cur['q'], cur['s'], cur['hb'], eps, dq=False, outq=cur['outq'], qmax=cur['qmax'], clip=cur['clip'], bits=cur['bits'])
            elif i == LS:
                cur = dict(cur, q=cur['q'][Ls:] if cur['q'] is not None else None, s=cur['s'][Ls:] if cur['s'] is not None else None,
                           hb=cur['hb'][Ls:] if cur['hb'] is not None else None)
            e = m.qw[i]['Win']
            proj, ra, cs, dq = m._run(cur, e)
            if sl is not None:
                xs = x[sl].float(); hs = (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
                dl = self._delta(i, 'Win', hs)
                if dq:
                    sc = ra[sl, None] * cs[None, :]
                    proj[sl] = ((proj[sl].float() * sc + dl.float()) / sc).to(proj.dtype)
                else:
                    proj[sl] += dl.to(proj.dtype)
            nxt = m._next_in(i, 'Wo', T)
            R2 = m.r2_for(i)
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, ra, cs, dq, deep)
                K._gnorm_k[(triton.cdiv(T, m.rows),)](o, proj[:, 6144:], ra if dq else proj, cs if dq else proj, d['gn_w'], R2.sign, m.M['H32'], m.M['H64'], m.M['H128'],
                                                      nxt['q'], nxt['s'], nxt['hb'], T, proj.stride(0), eps, DQ=dq, OUTQ=nxt['outq'], HAD=2 if m.ohead else 1,
                                                      QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=m.hprec, ROWS=m.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, ra, cs, dq, deep)
                K._agate_k[(triton.cdiv(T, m.rows),)](o, proj, ra if dq else proj, cs if dq else proj, R2.sign, m.M['H32'], m.M['H64'], m.M['H16'],
                                                      nxt['q'], nxt['s'], nxt['hb'], T, proj.stride(0), o.stride(0), o.stride(1), DQ=dq, OUTQ=nxt['outq'], HAD=2 if m.ohead else 1,
                                                      QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=m.hprec, ROWS=m.rows, num_warps=8)
            cur = nxt
            if sl is not None:       # exact bf16 Wo input of the slot rows (the glue kernel again, bf16 output, slot rows only)
                if cur['outq']:
                    a = sl.start; n_ = sl.stop - sl.start
                    yb = torch.empty(n_, 2048, device=dev, dtype=torch.bfloat16)
                    if d['type'] == 'linear_attention':
                        K._gnorm_k[(n_,)](o[a:], proj[a:, 6144:], ra[a:] if dq else proj, cs if dq else proj, d['gn_w'], yb, yb, yb, yb, yb, yb, yb, n_, proj.stride(0), eps,
                                         DQ=dq, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=1, num_warps=8)
                    else:
                        K._agate_k[(n_,)](o[:, a:], proj[a:], ra[a:] if dq else proj, cs if dq else proj, yb, yb, yb, yb, yb, yb, yb, n_, proj.stride(0), o.stride(0), o.stride(1),
                                         DQ=dq, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=1, num_warps=8)
                else:
                    yb = cur['hb'][sl]
                x[sl] += self._delta(i, 'Wo', yb)
            D_, ra, cs, dq = m._run(cur, m.qw[i]['Wo'])
            nxt = m._next_in(i, 'Wgu', T)
            K.addq(x, D_, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            cur = nxt
            eg = m.qw[i]['Wgu']; nxt = m._next_in(i, 'Wd', T)
            if eg.get('evt'):
                GU = QG.swiglu_gemm(eg['kind'], cur['q'], eg['codes'], cur['s'], eg['cs_true'], cfg=(1 if (eg['kind'] == 's8' and T >= 2500) else 3)); dq = False; ra = cs = None; mi = True
            else:
                GU, ra, cs, dq = m._run(cur, eg); mi = False
            K._swiglu_k[(triton.cdiv(T, m.rows),)](GU, ra if dq else GU, cs if dq else GU, m.R4.sign, m.M['P12T'], m.M['H16'], m.M['H32'], nxt['q'], nxt['s'], nxt['hb'], T,
                                                   DQ=dq, OUTQ=nxt['outq'], QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=m.hprec, ROWS=m.rows, MIN=mi, num_warps=8)
            cur = nxt
            if sl is not None:       # exact bf16 Wd input (SwiGLU output) of the slot rows
                if not cur['outq']: mb = cur['hb'][sl]
                elif mi: mb = GU[sl]
                else:
                    a = sl.start; n_ = sl.stop - sl.start
                    mb = torch.empty(n_, 6144, device=dev, dtype=torch.bfloat16)
                    K._swiglu_k[(n_,)](GU[a:], ra[a:] if dq else GU, cs if dq else GU, mb, mb, mb, mb, mb, mb, mb, n_,
                                       DQ=dq, OUTQ=False, QMAX=7., CLIP=1., BITS=8, ROWS=1, MIN=False, num_warps=8)
                x[sl] += self._delta(i, 'Wd', mb)
            D_, ra, cs, dq = m._run(cur, m.qw[i]['Wd'])
            if i + 1 < i1:
                nxt = m._next_in(i + 1, 'Win', T)
            else:
                nxt = m._next_in(None, 'final', T)
            K.addq(x, D_, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            cur = nxt
            if self.s.D and i + 1 == LS and Ls > 0 and i + 1 < i1:
                self.Mq[:Ls].copy_(cur['q'][:Ls]); self.Ms[:Ls].copy_(cur['s'][:Ls])
                self._mem(0)
            elif self.s.D and i + 1 == LS and Ls > 0:
                raise RuntimeError('graph boundary at the depth split is not supported')

    @property
    def K(self):
        import qk
        return qk

    @property
    def QG(self):
        import qgemm
        return qgemm

    # ------------------------------------------------------------------ readouts
    def _rows_branch(self):
        return self.x[self.Ls:]

    def _unrot(self, h):
        return h.float() if self.fold else self.m.R1.inv(h.float())

    def readout_final(self):
        xb = self._rows_branch()
        rows = torch.cat([self.ans, self.opt.reshape(-1)])
        h = xb[rows].float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if not self.fold: h = h.to(torch.bfloat16)            # QRT: the final addq stores the gain-free norm in bf16 before unrotation
        h = (self._unrot(h) * self.norm1).to(torch.bfloat16).float()
        n = self.n
        dec = h[:n]; opts = h[n:].reshape(n, self.Kmax, -1)
        lg = self.rt.head(dec, opts) / self.temps
        return torch.softmax(lg.masked_fill(self.omask, float('-inf')), -1)

    def readout_exit(self):
        xb = self._rows_branch()
        rows = torch.cat([self.ans, self.opt.reshape(-1)])
        h = self._unrot(xb[rows])
        n = self.n
        lg = self.rt.eh(h[:n], h[n:].reshape(n, self.Kmax, -1)) / self.temps
        return torch.softmax(lg.masked_fill(self.omask, float('-inf')), -1)

    # ------------------------------------------------------------------ entry points
    def _embed(self):
        rt = self.rt
        nid = self.nid
        torch.index_select(rt.E, 0, self.ids, out=self.x[:nid])
        if self.xslot is not None: self.x[nid:].copy_(self.xslot)

    def run_full(self):
        self._embed()
        (self._layers_fold if self.fold else self._layers_q)(0, NL)
        return self.readout_final()

    def run_a(self):
        """X graph A: layers 0..XL-1 + exit head"""
        self._embed()
        (self._layers_fold if self.fold else self._layers_q)(0, self.XL)
        return self.readout_exit()

    def run_b(self):
        """X graph B: layers XL..23 (+ remaining deep memory groups) + hobson head. Reads self.x left by graph A."""
        if self.s.D and len(self.memo.groups) > 1 and self.Ls > 0:
            for gi in range(1, len(self.memo.groups)): self._mem(gi)
        (self._layers_fold if self.fold else self._layers_q)(self.XL, NL)
        return self.readout_final()


# ====================================================================== graphs and timing
def capture(fn, warm=3):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(warm): out = fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): out = fn()
    torch.cuda.synchronize()
    return g, out


def med(ts):
    ts = sorted(ts)
    return dict(median=round(stt.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3), min=round(ts[0], 3), n=len(ts))


def fresh_pool(req, n, lo=1000, hi=100000, hi_ext=None):
    """fresh input ids (same shape) per rep: live state ids random; question ids fixed (deployment constants)"""
    pool = []
    base = req.ids.cpu()
    for _ in range(n):
        x = base.clone()
        if req.Ls:
            x[:req.Ls] = torch.randint(lo, hi_ext or hi, (req.Ls,))
        pool.append(x.pin_memory())
    return pool


def wall_full(req, g, out, pool, warm=3):
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory(); ts = []
    for x in pool:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        req.ids.copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return med(ts[warm:])


def margins(p, nsl):
    out = []
    for j, row in enumerate(p.tolist()):
        v = sorted(row[:nsl[j]], reverse=True); out.append(v[0] - (v[1] if len(v) > 1 else 0.0))
    return out


def wall_casc(req, ga, oa, gb, ob, pool, decide, warm=3):
    """cascade: A -> D2H -> host margins -> decide(margins) -> B if any question continues. decide returns bool 'run B'."""
    ha = torch.empty(oa.shape, dtype=oa.dtype).pin_memory()
    hb = torch.empty(ob.shape, dtype=ob.dtype).pin_memory() if ob is not None else None
    nsl = [b.n_slots for b in req.s.branches]; ts = []; nb = 0
    for x in pool:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        req.ids.copy_(x, non_blocking=True); ga.replay(); ha.copy_(oa, non_blocking=True); torch.cuda.synchronize()
        mg = margins(ha, nsl)
        if decide(mg):
            gb.replay(); hb.copy_(ob, non_blocking=True); torch.cuda.synchronize(); nb += 1
        ts.append((time.perf_counter() - t0) * 1000)
    r = med(ts[warm:]); r['ran_b'] = nb
    return r


def kcat(n):
    nl = n.lower()
    if 'flash' in nl or 'fmha' in nl or 'attention' in nl or 'efficient' in nl: return 'attn'
    if 'cutlass' in nl or '_gemm_k' in nl or 'gemm' in nl or 'cublas' in nl or 'sm80_xmma' in nl or 'ampere_' in nl or '_tgemm' in nl or '_mm_k' in nl or '_sk' in nl[:4] or nl.startswith('_sk') or '_skred' in nl: return 'gemm'
    if 'chunk' in nl or 'recompute_w_u' in nl or 'kkt' in nl or 'solve' in nl or 'fwd_kernel_h' in nl or 'fwd_kernel_o' in nl: return 'gdn_core'
    if any(k in nl for k in ('_swiglu_k', '_gnorm_k', '_agate_k', '_addq_k', '_conv_k', '_aprep_k', '_kvprep_k', '_ropeg_k')): return 'glue'
    return 'other'


def kprof(g):
    from torch.profiler import profile, ProfilerActivity
    import collections
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    agg = collections.defaultdict(float); n = 0
    for e in p.events():
        if e.device_type.name != 'CUDA': continue
        dt = (e.device_time if hasattr(e, 'device_time') else e.cuda_time) / 1000
        agg[kcat(e.name)] += dt; n += 1
    r = {k: round(v, 4) for k, v in agg.items()}; r['kernels'] = n; r['busy'] = round(sum(agg.values()), 4)
    return r
