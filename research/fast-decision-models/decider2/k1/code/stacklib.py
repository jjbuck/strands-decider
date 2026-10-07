"""K1 stacklib (Brief 9): ONE differentiable hobson-v19 forward with composable flags D, C, V, Q.

All flags off (and adapters off) = hobson-v19 (LoRA merged, H3's lean fused weights, fla kernels), state-first, every question its own
branch over the shared state (hobson's own shared-prefix layout).

  D  depth split (J3 DT-A8, bridge 'A'). State rows (and C's compiled blocks) run layers 0..7 only. Question rows run all 24 layers.
     Memory M = residual of every state row after layer 7. Deep attention layers (11, 15, 19, 23) read M through their own K/V rows of Win
     (+ the main LoRA) + a rank-32 memory adapter (zero-init B), K roped at the rows' runtime positions. Deep GDN layers recur over each
     question branch alone from a zero state and a zero conv history.
  C  precompiled deployment documents (J9 layout R, K/V-only library). J9's loose segmentation (frame / blk / dyn; blocks >= 32 Qwen tokens).
     Every distinct block is computed once in the universal context U = '<state>\\n' (rows [U][block], causal, positions len(U)..), with the
     main LoRA plus a compile adapter (LoRA r16, all 24 layers, all four GEMMs) that acts on compile rows only. Runtime order:
     [pre = frame pieces (or the first dyn piece if there is no frame)][distinct blocks, first appearance][the other dyn pieces][questions];
     runtime positions are consecutive in that order. Runtime rows read the blocks only through attention: block K (normed) roped at its
     runtime position, block V. Blocks are invisible to every runtime GDN layer (no state composition, no conv history).
     Under D the compile pass stops after layer 7 and the blocks' layer-7 residuals join M (their deep K/V get the compile adapter's K/V rows).
  V  16k super-token vocabulary (J7 superbpe over Qwen ids; digits and newline tokens never merged). Embedding of super-token k =
     mean(E[constituents]) + sum_j A[j] * E[constituent j from the end] + delta[k]  (A, delta zero-init). Each row's RoPE position = the
     original (runtime-order) Qwen position of its last constituent. Applied to U, blocks, stream pieces (merged piece by piece) and every
     in-context question. Option rows = the rows ending at the option-end Qwen offsets.
  Q  deployed questions compiled into weights (J6 'late'). The 38 deployed question specs (train_pool; assets/deployed_q.json): a question
     whose name and spec match, rendered in canonical option order, becomes a branch of K+1 slot rows (K options, then the answer row) with
     learned input vectors, positions P_end .. P_end + K (P_end = runtime Qwen length of the state), and a shared r16 + per-question r8 LoRA
     on Win, Wo, Wd of all 24 layers acting on its slot rows only. All other questions stay in context.

P (inference only): W8A8 emulation in H1's format (FORMAT.md): QuaRot rotations R1/R2/R4 (seeds 1234/1235/1236), act-order GPTQ codes from
Hessians of the run's own GEMM inputs, per-token absmax int8 activations, the 8 GEMMs of precmap_w8a8_b8.json in bf16, integer-exact
torch._int_mm. The main LoRA is merged into the weights before GPTQ. Not quantized: the compile pass (offline, bf16), the C/Q adapter deltas
(bf16 side GEMMs on the bf16 GEMM input), everything H1 keeps in bf16/fp32. D's memory K/V GEMM (one per deep attention layer, Win K/V rows +
LoRA + memory adapter merged) is W8A8 with its own GPTQ Hessian.
X (inference only): exit head at layer 16 (J15's EH: pointer head + rank-512 adapter on the residual after 16 layers).

Host side:  base = m.prep(state_text, [(name, spec, perm|None), ...]) ; view = m.view(base, flags) ; out = m.forward(view, ...)
Convenience: m.predict_item(item, flags) -> {qname: {label: p}}.
"""
import os, sys, math, json, hashlib, collections, itertools, random
HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.expanduser('~/work')
sys.path[:0] = [HERE, f'{W}/tokens', f'{W}/systems/g', f'{W}/evalkit']
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import h3lib as H
from h3lib import rms_zc, StdHead, fla_conv, chunk_gated_delta_rule, paley12, hadamard

VERSION = '0.3'
ATT = (3, 7, 11, 15, 19, 23)
NAMES = ('Win', 'Wo', 'Wgu', 'Wd')
QMODS = ('Win', 'Wo', 'Wd')
KV = (4096, 5120)
LS_D = 8
HID_LAYERS = (5, 11, 17, 23)
EXIT_L = 16
ASSETS = os.path.join(HERE, 'assets')
FLAGSET = 'DCVQ'


def flagset(f):
    f = set((f or '').upper()) - {'P', 'X'}
    assert f <= set(FLAGSET), f
    return frozenset(f)


def _rope(t, cos, sin):
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


def _nrm(x, eps):
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def _conv_index(runs, T):
    """runs: (row0, length, history rows [<=3], -1 = zero). -> cin (conv input rows, -1 = zero row), cout (row -> index in the conv output)"""
    cin = []; cout = [0] * T
    for r0, L, hist in runs:
        if L == 0: continue
        hh = ([-1, -1, -1] + list(hist))[-3:]
        cin += hh
        for t in range(L):
            cout[r0 + t] = len(cin); cin.append(r0 + t)
    return cin, cout


def attn(q, k, v, mask):
    """q [Tq, 8, 256] k/v [Tk, 2, 256]; mask: 'causal' (Tq == Tk) | 'lr' (lower-right causal) | bool [Tq, Tk] (True = visible) -> [Tq, 8, 256]"""
    Tq, Tk = q.shape[0], k.shape[0]
    if isinstance(mask, str) and mask == 'causal' and Tq == Tk:
        return F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True,
                                              enable_gqa=True)[0].transpose(0, 1)
    if isinstance(mask, str):
        i = torch.arange(Tq, device=q.device)[:, None]; j = torch.arange(Tk, device=q.device)[None, :]
        mask = j <= i + (Tk - Tq)
    kr = k.repeat_interleave(4, dim=1); vr = v.repeat_interleave(4, dim=1)
    return F.scaled_dot_product_attention(q.transpose(0, 1)[None], kr.transpose(0, 1)[None], vr.transpose(0, 1)[None],
                                          attn_mask=mask[None, None])[0].transpose(0, 1)


# ====================================================================== W8A8 rotations (H1 FORMAT section 2, exact)
class Rot:
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


def gptq(W, Hm, qmax, s=None, blocksize=128, percdamp=0.01, actorder=True):
    """H1's GPTQ (act-order, damp .01 mean diag, block 128), per-channel scale fixed before (absmax / qmax). -> integer codes (float), s"""
    W = W.clone().float(); N, K = W.shape; Hm = Hm.clone().float()
    dead = torch.diag(Hm) == 0
    Hm[dead, dead] = 1; W[:, dead] = 0
    if s is None: s = W.abs().amax(1).clamp_min(1e-8) / qmax
    perm = None
    if actorder:
        perm = torch.argsort(torch.diag(Hm), descending=True); W = W[:, perm]; Hm = Hm[perm][:, perm]
    damp = percdamp * torch.mean(torch.diag(Hm))
    Hm[range(K), range(K)] += damp
    L = torch.linalg.cholesky(Hm)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
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
        Q = Q[:, torch.argsort(perm)]
    return Q, s


# ====================================================================== adapters
def lora_pair(out, inp, r, g, dev):
    A = nn.Parameter((torch.randn(r, inp, generator=g) / math.sqrt(inp)).to(dev))
    B = nn.Parameter(torch.zeros(out, r, device=dev))
    return A, B


class LoRASet(nn.Module):
    """per (layer, module) A [r, in], B [out, r]; y += (x A^T) B^T * scale"""

    def __init__(self, m, r, alpha, mods, seed, layers=range(24)):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.A = nn.ParameterDict(); self.B = nn.ParameterDict(); self.scale = alpha / r; self.r = r
        for i in layers:
            for k in mods:
                out, inp = m.L[i][k].shape
                A, B = lora_pair(out, inp, r, g, m.dev); self.A[f'{i}_{k}'] = A; self.B[f'{i}_{k}'] = B

    def has(self, i, k): return f'{i}_{k}' in self.A

    def delta(self, x, i, k, rows=None):
        A = self.A[f'{i}_{k}']; B = self.B[f'{i}_{k}']
        if rows is not None: B = B[rows[0]:rows[1]]
        return ((x @ A.t().to(x.dtype)) @ B.t().to(x.dtype)) * self.scale

    def dense(self, i, k):
        return self.B[f'{i}_{k}'].float() @ self.A[f'{i}_{k}'].float() * self.scale


class MemAdapters(nn.Module):
    """D: rank-32 adapters on the memory K/V projection of each deep attention layer (J3 add_mem: A random / sqrt(2048), B zero, scale alpha/r = 1)"""

    def __init__(self, m, r=32, alpha=32, seed=1):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.A = nn.ParameterDict(); self.B = nn.ParameterDict(); self.scale = alpha / r
        for i in range(LS_D, 24):
            if m.L[i]['type'] == 'linear_attention': continue
            self.A[str(i)] = nn.Parameter((torch.randn(r, 2048, generator=g) / math.sqrt(2048)).to(m.dev))
            self.B[str(i)] = nn.Parameter(torch.zeros(KV[1] - KV[0], r, device=m.dev))

    def delta(self, x, i):
        return ((x @ self.A[str(i)].t().to(x.dtype)) @ self.B[str(i)].t().to(x.dtype)) * self.scale

    def dense(self, i):
        return self.B[str(i)].float() @ self.A[str(i)].float() * self.scale


class SuperEmb(nn.Module):
    """V (J7 j7lib.SuperEmb): super-token k embedding = mean(E[c]) + sum_j A[j] E[c_j] + delta[k], constituents counted from the end."""

    def __init__(self, base_embed, toks, dev, maxc=24):
        super().__init__()
        N = len(toks); self.N = N
        C = torch.zeros(N, maxc, dtype=torch.long); M = torch.zeros(N, maxc)
        for i, t in enumerate(toks):
            t = t[-maxc:]; L = len(t)
            C[i, :L] = torch.tensor(t[::-1]); M[i, :L] = 1.0
        self.register_buffer('C', C.to(dev)); self.register_buffer('M', M.to(dev))
        self.A = nn.Parameter(torch.zeros(maxc, 2048, device=dev))
        self.delta = nn.Parameter(torch.zeros(N, 2048, device=dev))
        self.base = base_embed

    def rows(self, k):
        c = self.C[k]; m = self.M[k]
        e = self.base[c].float()
        mean = (e * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        comp = (e * m[..., None] * self.A[None]).sum(1)
        return mean + comp + self.delta[k]


class QAdapters(nn.Module):
    """Q (J6 QAdapters): shared LoRA r16 (alpha 32) + per-question LoRA r8 (alpha 16) on Win/Wo/Wd of all 24 layers + per-question slot inputs."""

    def __init__(self, m, qnames, slot_init, r_s=16, r_q=8, seed=0):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(seed); dev = m.dev
        self.qnames = list(qnames); self.qi = {q: j for j, q in enumerate(self.qnames)}
        self.sA = nn.ParameterDict(); self.sB = nn.ParameterDict()
        self.qA = nn.ModuleList(); self.qB = nn.ModuleList()
        for i in range(24):
            for nm in QMODS:
                out, inp = m.L[i][nm].shape
                A, B = lora_pair(out, inp, r_s, g, dev); self.sA[f'{i}_{nm}'] = A; self.sB[f'{i}_{nm}'] = B
        for q in self.qnames:
            pa = nn.ParameterDict(); pb = nn.ParameterDict()
            for i in range(24):
                for nm in QMODS:
                    out, inp = m.L[i][nm].shape
                    A, B = lora_pair(out, inp, r_q, g, dev); pa[f'{i}_{nm}'] = A; pb[f'{i}_{nm}'] = B
            self.qA.append(pa); self.qB.append(pb)
        self.slots = nn.ParameterList([nn.Parameter(slot_init[q].float().clone().to(dev)) for q in self.qnames])
        self.scale = 2.0     # alpha_s / r_s = alpha_q / r_q = 2

    def delta(self, h, i, nm, qi):
        k = f'{i}_{nm}'; hd = h.dtype
        A = torch.cat([self.sA[k], self.qA[qi][k]], 0).to(hd); B = torch.cat([self.sB[k], self.qB[qi][k]], 1).to(hd)
        return ((h @ A.t()) @ B.t()) * self.scale

    def shared_params(self): return list(self.sA.parameters()) + list(self.sB.parameters())

    def perq_params(self): return [p for j in range(len(self.qnames)) for p in list(self.qA[j].parameters()) + list(self.qB[j].parameters())]

    def slot_params(self): return list(self.slots)


class EH(nn.Module):
    """X: J15's exit head (j15learn.EH): LayerNorm(no affine) * gain(8) + rank-512 residual adapter (zero-init out), then hobson's pointer head."""

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


# ====================================================================== host-side layout
class Base:
    """flag-independent part of a request: rendered state text, native Qwen state ids, rendered questions"""
    pass


class View:
    """one layout of a request (teacher = flags '', student = the run's flags). Host lists only."""
    pass


class Stack(H.H3):
    """hobson-v19 + composable D/C/V/Q. Call free_hf() (inference) or detach_inference() (training) after construction."""
    C_CONV_BLOCKS = False

    def __init__(self, dev='cuda'):
        super().__init__(maxlen=16384)
        self.eng = self.p.eng; self.tok = self.eng.tok
        self.U = self.tok('<state>\n', add_special_tokens=False)['input_ids']
        self.ll = None; self.sup_tok = None; self.deployed = None
        self.lora = None          # main LoRA (LoRASet)
        self.cad = None           # C compile adapter (LoRASet)
        self.mad = None           # D memory adapters
        self.sup = None           # V super embeddings
        self.qad = None           # Q adapters
        self.head = None
        self.use = True           # student adapters on (False = teacher: hobson exactly)
        self.qz = None            # W8A8 state: dict(codes={(i,k): (q int8, s fp32)}, bf16=set) or None
        self.capture = None; self.Hacc = {}; self.Hn = {}
        self.rots = None
        self.flags_trained = frozenset()

    def cos_sin(self, pos):
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        return fr.cos().to(torch.bfloat16), fr.sin().to(torch.bfloat16)

    # ------------------------------------------------------------------ assets
    def load_assets(self, need=('C', 'V', 'Q')):
        if 'C' in need and self.ll is None:
            import j9lib as J
            self.J = J
            ll = J.LineLib.__new__(J.LineLib)
            lines = json.load(open(os.path.join(ASSETS, 'linelib.json')))
            ll.lib = set(lines); ll.libs = sorted(ll.lib); self.ll = ll
        if 'V' in need and self.sup_tok is None:
            from superbpe import Super
            self.sup_tok = Super(os.path.join(ASSETS, 'sb16k.json'), tok=self.tok)
        if 'Q' in need and self.deployed is None:
            self.deployed = json.load(open(os.path.join(ASSETS, 'deployed_q.json')))
            self.deployed_js = {q: json.dumps(s, sort_keys=True) for q, s in self.deployed.items()}

    # ------------------------------------------------------------------ student parameters
    def add_student(self, flags, seed=7, lora_r=32):
        """-> dict of parameter groups. Zero-init B everywhere (as the source code)."""
        flags = flagset(flags); self.flags_trained = flags
        self.load_assets(flags)
        g = {}
        self.lora = LoRASet(self, lora_r, 2 * lora_r, NAMES, seed); g['lora'] = list(self.lora.parameters())
        self.head = StdHead(self.head0).to(self.dev)
        for p_ in self.head.parameters(): p_.requires_grad_(True)
        g['head'] = list(self.head.parameters())
        if 'D' in flags:
            self.mad = MemAdapters(self, seed=seed + 1); g['mem'] = list(self.mad.parameters())
        if 'C' in flags:
            self.cad = LoRASet(self, 16, 32, NAMES, seed + 2); g['compile'] = list(self.cad.parameters())
        if 'V' in flags:
            S = self.sup_tok; self.sup = SuperEmb(self.embed, [list(t) for t in S.toks], self.dev)
            g['sup_A'] = [self.sup.A]; g['sup_delta'] = [self.sup.delta]
        if 'Q' in flags:
            qn = sorted(self.deployed)
            sinit = {q: self.slot_init(q) for q in qn}
            self.qad = QAdapters(self, qn, sinit, seed=seed + 3)
            g['q_shared'] = self.qad.shared_params(); g['q_perq'] = self.qad.perq_params(); g['q_slots'] = self.qad.slot_params()
        return g

    def slot_init(self, qname):
        """J6 qtab.slot_init: option k = mean embedding of its option line (<= 24 tokens back, rescaled), answer row = '<answer>' embedding"""
        pq = self.prep_q(self.deployed[qname])
        E = self.embed.float(); rows = []
        starts = [0] + [o + 1 for o in pq['opt'][:-1]]
        for k, o in enumerate(pq['opt']):
            a = max(starts[k], o - 23) if k > 0 else max(0, o - 23)
            e = E[torch.tensor(pq['q'][a:o + 1], device=E.device)]
            v = e.mean(0); v = v / v.norm() * e.norm(dim=-1).mean()
            rows.append(v)
        rows.append(E[pq['q'][-1]])
        return torch.stack(rows, 0).detach()

    def prep_q(self, spec, perm=None):
        from pydantic import TypeAdapter
        import strands_decider.schema as SC
        from strands_decider.prompting import render_question
        q = TypeAdapter(SC.Question).validate_python(spec)
        rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
        s, qs = self.eng._fit('S', [rq.text])
        opt = self.eng._option_idx([rq], 0)[0].tolist()
        return dict(q=qs[0], opt=opt, rq=rq, kind=rq.kind, K=len(opt))

    def randomize_adapters(self, std=0.02, seed=0):
        """fidelity tests (K2): give every zero-init adapter factor a random value so each flag's code path changes the output"""
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        with torch.no_grad():
            for mod in (self.lora, self.cad, self.mad, self.qad):
                if mod is None: continue
                for n_, p_ in mod.named_parameters():
                    if ('B' in n_.split('.')[0] or n_.startswith('sB') or n_.startswith('qB')) and p_.abs().max() == 0:
                        p_.copy_(torch.randn(p_.shape, generator=g).to(p_.device) * std)
            if self.sup is not None:
                self.sup.A.copy_(torch.randn(self.sup.A.shape, generator=g).to(self.dev) * 0.01)
                self.sup.delta.copy_(torch.randn(self.sup.delta.shape, generator=g).to(self.dev) * 0.002)

    # ------------------------------------------------------------------ checkpoints
    def student_state(self):
        sd = {}
        for nm, mod in (('lora', self.lora), ('cad', self.cad), ('mad', self.mad), ('qad', self.qad), ('head', self.head)):
            if mod is not None:
                for k, v in mod.state_dict().items(): sd[f'{nm}.{k}'] = v.detach().cpu()
        if self.sup is not None:
            sd['sup.A'] = self.sup.A.detach().cpu(); sd['sup.delta'] = self.sup.delta.detach().cpu()
        sd['_meta'] = dict(flags=''.join(sorted(self.flags_trained)), version=VERSION, lora_r=self.lora.r if self.lora else None)
        return sd

    def load_student(self, path_or_sd):
        sd = torch.load(path_or_sd, map_location=self.dev, weights_only=False) if isinstance(path_or_sd, str) else path_or_sd
        meta = sd['_meta']
        self.add_student(meta['flags'], lora_r=meta.get('lora_r') or 32)
        for nm, mod in (('lora', self.lora), ('cad', self.cad), ('mad', self.mad), ('qad', self.qad), ('head', self.head)):
            if mod is not None:
                mod.load_state_dict({k[len(nm) + 1:]: v for k, v in sd.items() if k.startswith(nm + '.')})
        if self.sup is not None:
            with torch.no_grad():
                self.sup.A.copy_(sd['sup.A']); self.sup.delta.copy_(sd['sup.delta'])
        for mod in (self.lora, self.cad, self.mad, self.qad, self.head, self.sup):
            if mod is not None:
                for p_ in mod.parameters(): p_.requires_grad_(False)
        return meta

    class _Teacher:
        def __init__(self, m): self.m = m
        def __enter__(self): self.s = self.m.use; self.m.use = False
        def __exit__(self, *e): self.m.use = self.s

    def teacher(self): return Stack._Teacher(self)

    # ------------------------------------------------------------------ host: request preparation
    def prep(self, state_text, qlist):
        """state_text = render_state(state) (possibly truncated); qlist = [(name, spec, perm or None)].  -> Base"""
        from pydantic import TypeAdapter
        import strands_decider.schema as SC
        from strands_decider.prompting import render_question
        ta = TypeAdapter(SC.Question)
        rqs = [render_question(ta.validate_python(sp), option_order=pm) if pm is not None else render_question(ta.validate_python(sp))
               for _, sp, pm in qlist]
        s, qs = self.eng._fit(state_text, [rq.text for rq in rqs])
        opt = self.eng._option_idx(rqs, 0).tolist()
        b = Base(); b.st_text = state_text; b.s = s
        b.qs = [dict(name=n, spec=sp, perm=pm, rq=rq, q=q, opt=[o for o in op if o >= 0][:rq.n_slots], kind=rq.kind)
                for (n, sp, pm), rq, q, op in zip(qlist, rqs, qs, opt)]
        return b

    def _merge(self, ids):
        """V: Qwen ids -> (row ids, ends[row] = index of its last Qwen token)"""
        return self.sup_tok.merge_ids(ids)

    def view(self, b, flags):
        flags = flagset(flags)
        if flags: self.load_assets(flags)
        v = View(); v.flags = flags; v.b = b
        V_ = 'V' in flags
        # ---- state pieces (C) in runtime order
        pre, blks, post = [b.s], [], []
        if 'C' in flags:
            J = self.J
            req = J.tokenize_pieces(self.tok, J.pieces(b.st_text, self.ll))
            if [t for _, ids, _ in req for t in ids] == list(b.s) and any(k == 'blk' for k, _, _ in req):
                has_frame = any(k == 'frame' for k, _, _ in req)
                first = None; pre = []
                for t, (k, ids, key) in enumerate(req):
                    if k == 'frame' or (not has_frame and k == 'dyn' and first is None):
                        pre.append(ids); first = t
                seen = set(); blks = []
                for k, ids, key in req:
                    if k == 'blk' and key not in seen: seen.add(key); blks.append((key, list(ids)))
                post = [ids for t, (k, ids, key) in enumerate(req) if k == 'dyn' and t != first]
            v.c_status = 'mismatch' if [t for _, ids, _ in req for t in ids] != list(b.s) else ('blocks' if blks else 'noblocks')
        # ---- rows and runtime positions
        qpos = 0; s_ids = []; s_pos = []; n_pre = 0

        def add_piece(ids, p0):
            if V_ and len(ids):
                mi, en = self._merge(list(ids)); return mi, [p0 + e for e in en]
            return list(ids), list(range(p0, p0 + len(ids)))
        for ids in pre:
            a_, p_ = add_piece(ids, qpos); s_ids += a_; s_pos += p_; qpos += len(ids)
        n_pre = len(s_ids)
        v.blocks = []
        if blks:
            if V_:
                ui, ue = self._merge(list(self.U)); upos = list(ue)
            else:
                ui = list(self.U); upos = list(range(len(ui)))
            v.U = (ui, upos); u = len(self.U)
            for key, ids in blks:
                if V_:
                    mi, en = self._merge(ids); cpos = [u + e for e in en]; rpos = [qpos + e for e in en]
                else:
                    mi = ids; cpos = list(range(u, u + len(ids))); rpos = list(range(qpos, qpos + len(ids)))
                v.blocks.append(dict(key=key, ids=mi, cpos=cpos, rpos=rpos, nq=len(ids)))
                qpos += len(ids)
        else:
            v.U = None
        for ids in post:
            a_, p_ = add_piece(ids, qpos); s_ids += a_; s_pos += p_; qpos += len(ids)
        v.s_ids = s_ids; v.s_pos = s_pos; v.n_pre = n_pre; v.P_end = qpos
        # ---- question branches
        v.br = []
        for qd in b.qs:
            slot = ('Q' in flags and qd['perm'] is None and qd['name'] in self.deployed
                    and json.dumps(qd['spec'], sort_keys=True) == self.deployed_js[qd['name']])
            if 'Q' in flags and qd['name'] in (self.deployed or {}) and qd['perm'] is not None:
                slot = False      # sampler guarantees deployed questions are unpermuted under Q
            if slot:
                K = len(qd['opt'])
                v.br.append(dict(kind='slot', qname=qd['name'], ids=[0] * (K + 1), pos=[qpos + k for k in range(K + 1)],
                                 opt=list(range(K)), ans=K, n=K + 1, rq=qd['rq']))
            else:
                q = qd['q']
                if V_:
                    mi, en = self._merge(list(q))
                    e2r = {e: r for r, e in enumerate(en)}
                    opt = [e2r[o] if o in e2r else next(r for r, e in enumerate(en) if e >= o) for o in qd['opt']]
                    pos = [qpos + e for e in en]
                else:
                    mi = list(q); opt = list(qd['opt']); pos = list(range(qpos, qpos + len(q)))
                v.br.append(dict(kind='ctx', qname=qd['name'], ids=mi, pos=pos, opt=opt, ans=len(mi) - 1, n=len(mi), rq=qd['rq']))
        return v

    # ------------------------------------------------------------------ device plan
    def plan(self, v):
        dev = self.dev; P = type('Plan', (), {})(); P.v = v
        D_ = 'D' in v.flags
        nb = len(v.blocks)
        nU = len(v.U[0]) if nb else 0
        bl = [len(bk['ids']) for bk in v.blocks]
        nB = sum(bl); nc = nU + nB
        NS = len(v.s_ids); R = sum(br['n'] for br in v.br)
        P.nU, P.nB, P.nc, P.NS, P.R = nU, nB, nc, NS, R
        P.b0, P.s0, P.q0 = nU, nc, nc + NS; P.T = nc + NS + R
        P.npre = v.n_pre; P.D = D_
        ids = []; pos = []
        if nb:
            ids += v.U[0]; pos += v.U[1]
            for bk in v.blocks: ids += bk['ids']; pos += bk['cpos']
        ids += v.s_ids; pos += v.s_pos
        P.bseg = []; r = nU
        for L in bl: P.bseg.append((r, L)); r += L
        P.qseg = []; r = 0; slot_rows = []; slot_q = []; slot_k = []
        for br in v.br:
            P.qseg.append((r, br['n']))
            ids += br['ids']; pos += br['pos']
            if br['kind'] == 'slot':
                qi = self.qad.qi[br['qname']]
                for k in range(br['n']): slot_rows.append(P.q0 + r + k); slot_q.append(qi); slot_k.append(k)
            r += br['n']
        P.ids = torch.tensor(ids, device=dev, dtype=torch.long)
        P.pos = torch.tensor(pos, device=dev, dtype=torch.float32)
        P.cs = self.cos_sin(P.pos)
        P.slot_rows = slot_rows; P.slot_q = slot_q; P.slot_k = slot_k
        P.slot_segs = [(r0, L, self.qad.qi[br['qname']]) for (r0, L), br in zip(P.qseg, v.br) if br['kind'] == 'slot']
        if nb:
            P.brpos = torch.tensor([p for bk in v.blocks for p in bk['rpos']], device=dev, dtype=torch.float32)
            P.cs_b = self.cos_sin(P.brpos)
            P.bcu = torch.tensor([0] + list(itertools.accumulate(bl)), device=dev, dtype=torch.long)
        P.qcu = torch.tensor([0] + list(itertools.accumulate(br['n'] for br in v.br)), device=dev, dtype=torch.long)
        # conv runs over all rows
        runs = []
        if nb:
            runs.append((0, nU, []))
            uh = list(range(max(0, nU - 3), nU))
            for (r0, L) in P.bseg: runs.append((r0, L, uh))
        if self.C_CONV_BLOCKS and nb and NS > P.npre:     # test only: J9's convention (block rows feed the stream's conv history)
            runs.append((P.s0, P.npre, []))
            runs.append((P.s0 + P.npre, NS - P.npre, (list(range(P.s0, P.s0 + P.npre)) + list(range(P.b0, P.nc)))[-3:]))
        else:
            runs.append((P.s0, NS, []))
        sh = list(range(P.s0 + max(0, NS - 3), P.s0 + NS))
        for (r0, L) in P.qseg: runs.append((P.q0 + r0, L, sh))
        cin, cout = _conv_index(runs, P.T)
        P.cin = torch.tensor(cin, device=dev, dtype=torch.long); P.cout = torch.tensor(cout, device=dev, dtype=torch.long)
        # question attention mask: keys [stream keys (pre, blocks, post) NK][question rows R]
        NK = NS + nB; P.NK = NK
        m = torch.zeros(R, NK + R, dtype=torch.bool, device=dev); m[:, :NK] = True
        for (r0, L) in P.qseg:
            m[r0:r0 + L, NK + r0:NK + r0 + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
        P.qmask = m
        # deep (D): question rows only, zero conv history
        if D_:
            cin, cout = _conv_index([(r0, L, []) for (r0, L) in P.qseg], R)
            P.dcin = torch.tensor(cin, device=dev, dtype=torch.long); P.dcout = torch.tensor(cout, device=dev, dtype=torch.long)
            mpos = list(v.s_pos[:P.npre]) + [p for bk in v.blocks for p in bk['rpos']] + list(v.s_pos[P.npre:])
            P.cs_m = self.cos_sin(torch.tensor(mpos, device=dev, dtype=torch.float32))
        # head rows (question-region relative)
        P.heads = []
        for (r0, L), br in zip(P.qseg, v.br):
            P.heads.append((r0 + br['ans'], [r0 + o for o in br['opt']], br['rq']))
        return P

    # ------------------------------------------------------------------ GEMMs
    def _base(self, i, nm, h, xn, rows=None, wkey=None, cap_from=0):
        """bf16 or W8A8 base GEMM for (layer i, module nm) on rows h (bf16 GEMM input) / xn (fp32 unweighted normed residual: Win/Wgu)."""
        W_ = self.L[i][nm] if rows is None else self.L[i][nm][rows[0]:rows[1]]
        key = wkey or (i, nm)
        if self.qz is not None and key in self.qz['codes']:
            src = xn if nm in ('Win', 'Wgu', 'Wmem') else h
            R = self.rot_for(nm, key)
            xr = R(src.float())
            sa = xr.abs().amax(-1).clamp_min(1e-8) / 127.0
            qa = torch.round(xr / sa[:, None]).clamp(-127, 127).to(torch.int8)
            qw, sw = self.qz['codes'][key]
            if qa.shape[0] > 16: acc = torch._int_mm(qa, qw.t()).float()
            else: acc = (qa.float() @ qw.float().t())
            y = acc * sa[:, None] * sw[None, :]
            if nm in ('Wo', 'Wd'): y = self.rots['R1'].inv(y)
            return y.to(torch.bfloat16), True
        if self.capture is not None and key in self.capture:
            src = (xn if nm in ('Win', 'Wgu', 'Wmem') else h)[cap_from:].float()
            self.Hacc[key] = self.Hacc.get(key, 0) + src.t() @ src; self.Hn[key] = self.Hn.get(key, 0) + src.shape[0]
        return h @ W_.t(), False

    def rot_for(self, nm, key):
        if self.rots is None:
            self.rots = dict(R1=Rot(2048, 1234, self.dev), R2=Rot(2048, 1235, self.dev), R4=Rot(6144, 1236, self.dev))
        return self.rots['R4'] if nm == 'Wd' else self.rots['R2'] if nm == 'Wo' else self.rots['R1']

    def proj(self, i, nm, h, xn, P, region):
        """region 'all': rows = the plan's all-rows tensor (compile rows [0, nc) first); 'q': question rows only."""
        nc = P.nc if region == 'all' else 0
        qoff = P.q0 if region == 'all' else 0
        use = self.use
        if nc and self.qz is not None:          # compile rows: offline bf16 pass (unquantized weights, unmerged LoRA)
            yc = h[:nc] @ self.L[i][nm].t()
            if use and self.lora is not None: yc = yc + self.lora.delta(h[:nc], i, nm)
            yr, qd = self._base(i, nm, h[nc:], None if xn is None else xn[nc:])
            if use and self.lora is not None and not qd: yr = yr + self.lora.delta(h[nc:], i, nm)
            y = torch.cat([yc, yr], 0)
        else:
            y, qd = self._base(i, nm, h, xn, cap_from=nc)
            if use and self.lora is not None and not qd: y = y + self.lora.delta(h, i, nm)
        if use and nc and self.cad is not None:
            y = torch.cat([y[:nc] + self.cad.delta(h[:nc], i, nm), y[nc:]], 0)
        if use and self.qad is not None and P.slot_segs and nm in QMODS:
            rows = []; ds = []
            for r0, L, qi in P.slot_segs:
                a0 = qoff + r0
                ds.append(self.qad.delta(h[a0:a0 + L], i, nm, qi)); rows.append(torch.arange(a0, a0 + L, device=h.device))
            y = y.index_add(0, torch.cat(rows), torch.cat(ds).to(y.dtype))
        return y

    def mem_kv(self, i, M, P):
        """D: memory K/V rows of deep attention layer i from M (runtime order [pre][blocks][post])."""
        hm = self.bnorm(M, i, 0)
        xnm = _nrm(M, self.eps) if (self.qz is not None or self.capture is not None) else None
        b0, b1 = P.npre, P.npre + P.nB
        if self.qz is not None and (i, 'Wmem') in self.qz['codes']:
            parts = []
            for a0, a1, comp in ((0, b0, False), (b0, b1, True), (b1, M.shape[0], False)):
                if a1 <= a0: continue
                if comp:     # compiled blocks: offline bf16
                    y = hm[a0:a1] @ self.L[i]['Win'][KV[0]:KV[1]].t()
                    if self.lora is not None: y = y + self.lora.delta(hm[a0:a1], i, 'Win', KV)
                    if self.mad is not None: y = y + self.mad.delta(hm[a0:a1], i)
                    if self.cad is not None: y = y + self.cad.delta(hm[a0:a1], i, 'Win', KV)
                else:
                    y, _ = self._base(i, 'Wmem', hm[a0:a1], xnm[a0:a1], wkey=(i, 'Wmem'))
                parts.append(y)
            return torch.cat(parts, 0)
        if self.capture is not None and (i, 'Wmem') in self.capture:      # runtime memory rows only (blocks are compiled offline)
            src = torch.cat([xnm[:b0], xnm[b1:]], 0).float(); key = (i, 'Wmem')
            self.Hacc[key] = self.Hacc.get(key, 0) + src.t() @ src; self.Hn[key] = self.Hn.get(key, 0) + src.shape[0]
        y = hm @ self.L[i]['Win'][KV[0]:KV[1]].t()
        if self.use:
            if self.lora is not None: y = y + self.lora.delta(hm, i, 'Win', KV)
            if self.mad is not None: y = y + self.mad.delta(hm, i)
            if self.cad is not None and P.nB:
                y = torch.cat([y[:b0], y[b0:b1] + self.cad.delta(hm[b0:b1], i, 'Win', KV), y[b1:]], 0)
        return y

    # ------------------------------------------------------------------ layers
    def _mixer_all(self, i, d, proj, P):
        T = proj.shape[0]; eps = self.eps
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()).to(torch.bfloat16); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            rp = torch.cat([raw, raw.new_zeros(1, raw.shape[1])], 0)
            xin = rp[torch.where(P.cin < 0, torch.full_like(P.cin, T), P.cin)]
            cy = fla_conv(xin[None].contiguous(), d['conv_w'], None, activation='silu')
            cy = cy[0] if isinstance(cy, tuple) else cy
            cv = cy[0][P.cout]
            q, k, v = cv.split(2048, dim=-1)
            q = q.reshape(T, 16, 128); k = k.reshape(T, 16, 128); v = v.reshape(T, 16, 128)
            outs = []
            if P.nc:
                sl = slice(0, P.nU)
                oU, SU = chunk_gated_delta_rule(q[None, sl], k[None, sl], v[None, sl], g[None, sl], beta[None, sl], use_qk_l2norm_in_kernel=True,
                                                output_final_state=True)
                outs.append(oU[0])
                sl = slice(P.b0, P.nc); nb = len(P.bseg)
                oB, _ = chunk_gated_delta_rule(q[None, sl], k[None, sl], v[None, sl], g[None, sl], beta[None, sl],
                                               initial_state=SU.expand(nb, -1, -1, -1).contiguous(), use_qk_l2norm_in_kernel=True, cu_seqlens=P.bcu)
                outs.append(oB[0])
            sl = slice(P.s0, P.q0)
            if P.NS:
                oS, S = chunk_gated_delta_rule(q[None, sl], k[None, sl], v[None, sl], g[None, sl], beta[None, sl], use_qk_l2norm_in_kernel=True,
                                               output_final_state=True)
                outs.append(oS[0])
            else:
                S = torch.zeros(1, 16, 128, 128, device=raw.device, dtype=torch.float32)
            if P.R:
                sl = slice(P.q0, P.T); n = len(P.qseg)
                oQ, _ = chunk_gated_delta_rule(q[None, sl], k[None, sl], v[None, sl], g[None, sl], beta[None, sl],
                                               initial_state=S.expand(n, -1, -1, -1).contiguous(), use_qk_l2norm_in_kernel=True, cu_seqlens=P.qcu)
                outs.append(oQ[0])
            o = torch.cat(outs, 0)
            of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            return ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(torch.bfloat16).reshape(T, 2048)
        qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
        kk = proj[:, 4096:4608].reshape(T, 2, 256); vv = proj[:, 4608:5120].reshape(T, 2, 256)
        qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
        cos, sin = P.cs
        qr = _rope(qh, cos, sin); kr = _rope(kk, cos, sin)
        outs = []
        s0, q0, npre = P.s0, P.q0, P.npre
        if P.nc:
            nU = P.nU
            outs.append(attn(qr[:nU], kr[:nU], vv[:nU], 'causal'))
            for (r0, L) in P.bseg:
                outs.append(attn(qr[r0:r0 + L], torch.cat([kr[:nU], kr[r0:r0 + L]], 0), torch.cat([vv[:nU], vv[r0:r0 + L]], 0), 'lr'))
            kb = _rope(kk[P.b0:P.nc], *P.cs_b); vb = vv[P.b0:P.nc]
            Ks = torch.cat([kr[s0:s0 + npre], kb, kr[s0 + npre:q0]], 0); Vs = torch.cat([vv[s0:s0 + npre], vb, vv[s0 + npre:q0]], 0)
            if npre: outs.append(attn(qr[s0:s0 + npre], kr[s0:s0 + npre], vv[s0:s0 + npre], 'causal'))
            if q0 - s0 - npre: outs.append(attn(qr[s0 + npre:q0], Ks, Vs, 'lr'))
        else:
            Ks = kr[s0:q0]; Vs = vv[s0:q0]
            if P.NS: outs.append(attn(qr[s0:q0], Ks, Vs, 'causal'))
        if P.R:
            outs.append(attn(qr[q0:], torch.cat([Ks, kr[q0:]], 0), torch.cat([Vs, vv[q0:]], 0), P.qmask))
        o = torch.cat(outs, 0)
        return (o * torch.sigmoid(gate)).reshape(T, 2048)

    def layer_all(self, i, x, P):
        d = self.L[i]; eps = self.eps
        h = self.bnorm(x, i, 0); xn = _nrm(x, eps) if (self.qz is not None or self.capture is not None) else None
        proj = self.proj(i, 'Win', h, xn, P, 'all')
        o = self._mixer_all(i, d, proj, P)
        x = x + self.proj(i, 'Wo', o, None, P, 'all')
        h2 = self.bnorm(x, i, 1); xn2 = _nrm(x, eps) if (self.qz is not None or self.capture is not None) else None
        gu = self.proj(i, 'Wgu', h2, xn2, P, 'all'); I = d['I']
        return x + self.proj(i, 'Wd', F.silu(gu[:, :I]) * gu[:, I:], None, P, 'all')

    def layer_deep(self, i, xq, M, P):
        d = self.L[i]; eps = self.eps; R = xq.shape[0]
        h = self.bnorm(xq, i, 0); xn = _nrm(xq, eps) if (self.qz is not None or self.capture is not None) else None
        proj = self.proj(i, 'Win', h, xn, P, 'q')
        if d['type'] == 'linear_attention':
            raw = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()).to(torch.bfloat16); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            rp = torch.cat([raw, raw.new_zeros(1, raw.shape[1])], 0)
            xin = rp[torch.where(P.dcin < 0, torch.full_like(P.dcin, R), P.dcin)]
            cy = fla_conv(xin[None].contiguous(), d['conv_w'], None, activation='silu')
            cy = cy[0] if isinstance(cy, tuple) else cy
            cv = cy[0][P.dcout]
            q, k, v = cv.split(2048, dim=-1)
            o, _ = chunk_gated_delta_rule(q.reshape(1, R, 16, 128), k.reshape(1, R, 16, 128), v.reshape(1, R, 16, 128), g[None], beta[None],
                                          use_qk_l2norm_in_kernel=True, cu_seqlens=P.qcu)
            of = o[0].reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            o = ((d['gn_w'] * of.to(torch.bfloat16)).float() * F.silu(z.reshape(-1, 128).float())).to(torch.bfloat16).reshape(R, 2048)
        else:
            qg = proj[:, :4096].reshape(R, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kq = proj[:, 4096:4608].reshape(R, 2, 256); vq = proj[:, 4608:5120].reshape(R, 2, 256)
            cos, sin = P.cs; cq, sq = cos[P.q0:], sin[P.q0:]
            qh = _rope(rms_zc(qh, d['qn'], eps), cq, sq); kq = _rope(rms_zc(kq, d['kn'], eps), cq, sq)
            pm = self.mem_kv(i, M, P)
            Tm = M.shape[0]
            km = _rope(rms_zc(pm[:, :512].reshape(Tm, 2, 256), d['kn'], eps), *P.cs_m); vm = pm[:, 512:].reshape(Tm, 2, 256)
            o = attn(qh, torch.cat([km, kq], 0), torch.cat([vm, vq], 0), P.qmask)
            o = (o * torch.sigmoid(gate)).reshape(R, 2048)
        x = xq + self.proj(i, 'Wo', o, None, P, 'q')
        h2 = self.bnorm(x, i, 1); xn2 = _nrm(x, eps) if (self.qz is not None or self.capture is not None) else None
        gu = self.proj(i, 'Wgu', h2, xn2, P, 'q'); I = d['I']
        return x + self.proj(i, 'Wd', F.silu(gu[:, :I]) * gu[:, I:], None, P, 'q')

    # ------------------------------------------------------------------ forward
    def embed_rows(self, P):
        ids = P.ids
        if self.sup is not None and 'V' in P.v.flags:
            V0 = self.sup_tok.V; new = ids >= V0
            x = F.embedding(torch.where(new, torch.zeros_like(ids), ids), self.embed).float()
            if bool(new.any()): x = x.index_put((new.nonzero()[:, 0],), self.sup.rows(ids[new] - V0))
        else:
            x = F.embedding(ids, self.embed).float()
        if P.slot_rows:
            sv = torch.stack([self.qad.slots[q][k] for q, k in zip(P.slot_q, P.slot_k)], 0)
            x = x.index_put((torch.tensor(P.slot_rows, device=self.dev),), sv.float())
        return x.to(torch.bfloat16)

    def forward(self, v, ckpt=False, keep=(), taps=(), stop=None):
        """-> dict(logits=[per branch [n_slots] (temperature applied)], kept={layer: [rows, 2048]} (option rows then answer row, per branch),
        taps={L: [per branch [1 + K, 2048]: answer row then option rows]} (residual after L layers). stop: run only the first `stop` layers."""
        P = self.plan(v)
        x = self.embed_rows(P)
        head = (self.head if (self.use and self.head is not None) else self.head0)
        krows = torch.tensor([r for a_, o_, _ in P.heads for r in (o_ + [a_])], device=self.dev)
        trows = [torch.tensor([a_] + o_, device=self.dev) for a_, o_, _ in P.heads]
        kept = {}; tapped = {}
        nL = 24 if stop is None else stop
        Ls = LS_D if P.D else 24
        cp = ckpt and torch.is_grad_enabled()
        for i in range(min(Ls, nL)):
            x = checkpoint(self.layer_all, i, x, P, use_reentrant=False) if cp else self.layer_all(i, x, P)
            if i in keep: kept[i] = x[P.q0:][krows]
            if i + 1 in taps: tapped[i + 1] = [x[P.q0:][t] for t in trows]
        xq = x[P.q0:]
        if P.D and nL > Ls:
            M = torch.cat([x[P.s0:P.s0 + P.npre], x[P.b0:P.nc], x[P.s0 + P.npre:P.q0]], 0)
            for i in range(Ls, nL):
                xq = checkpoint(self.layer_deep, i, xq, M, P, use_reentrant=False) if cp else self.layer_deep(i, xq, M, P)
                if i in keep: kept[i] = xq[krows]
                if i + 1 in taps: tapped[i + 1] = [xq[t] for t in trows]
        out = dict(kept=kept, taps=tapped, P=P)
        if stop is not None and stop < 24: return out
        h = rms_zc(xq, self.norm_w, self.eps)
        lg = []
        for a_, o_, rq in P.heads:
            l_ = head(h[a_].float()[None], h[torch.tensor(o_, device=self.dev)].float()[None])[0] / self.temp(rq.kind)
            lg.append(l_[:rq.n_slots])
        out['logits'] = lg
        return out

    # ------------------------------------------------------------------ convenience
    def item_base(self, item, qnames=None, max_state_tok=None):
        from strands_decider.prompting import render_state
        st = render_state(item['state'])
        if max_state_tok: st = trunc_text(self.tok, st, max_state_tok)
        names = list(qnames or item['questions'])
        return self.prep(st, [(n, item['questions'][n], None) for n in names])

    @torch.inference_mode()
    def predict_item(self, item, flags, qnames=None, taps=()):
        b = self.item_base(item, qnames)
        v = self.view(b, flags)
        out = self.forward(v, taps=taps)
        res = {}
        for qd, lg in zip(b.qs, out['logits']):
            p = torch.softmax(lg.float(), -1).tolist()
            res[qd['name']] = {lab: p[j] for j, lab in enumerate(qd['rq'].slot_labels)}
        return (res, out) if taps else res

    # ------------------------------------------------------------------ W8A8 (P)
    def gemm_keys(self, flags):
        flags = flagset(flags)
        ks = [(i, k) for i in range(24) for k in NAMES]
        if 'D' in flags: ks += [(i, 'Wmem') for i in range(LS_D, 24) if self.L[i]['type'] != 'linear_attention']
        return ks

    def merged_weight(self, i, k):
        """fp32 weight the deployed W8A8 GEMM implements: base + main LoRA (+ memory adapter for Wmem)"""
        if k == 'Wmem':
            Wf = self.L[i]['Win'][KV[0]:KV[1]].float()
            if self.lora is not None: Wf = Wf + self.lora.dense(i, 'Win')[KV[0]:KV[1]]
            if self.mad is not None: Wf = Wf + self.mad.dense(i)
            return Wf
        Wf = self.L[i][k].float()
        if self.lora is not None: Wf = Wf + self.lora.dense(i, k)
        return Wf

    def wfold(self, i, k):
        Wf = self.merged_weight(i, k)
        gk = 'in1' if k in ('Win', 'Wmem') else 'post1' if k == 'Wgu' else None
        if gk: Wf = Wf * self.L[i][gk][None, :]
        R = self.rot_for(k, (i, k))
        Wf = R(Wf)
        if k in ('Wo', 'Wd'): Wf = self.rots['R1'](Wf.t().contiguous()).t().contiguous()
        return Wf

    def quantize(self, hess, bf16_keys, flags):
        """hess: {(i,k): unrotated input Hessian (mean x^T x)}. -> self.qz (codes for every GEMM not in bf16_keys)"""
        codes = {}
        for key in self.gemm_keys(flags):
            if key in bf16_keys: continue
            i, k = key
            Wf = self.wfold(i, k)
            R = self.rot_for(k, key)
            H0 = hess[key].to(self.dev).float()
            Hm = R(R(H0).t().contiguous())
            q, s = gptq(Wf, Hm, 127.0)
            codes[key] = (q.to(torch.int8).contiguous(), s.float().contiguous())
            del Wf, Hm, H0
        self.qz = dict(codes=codes, bf16=set(bf16_keys))
        torch.cuda.empty_cache()
        return codes


def trunc_text(tok, text, k):
    """head 1/4 + tail 3/4 of a rendered state at line boundaries, by Qwen token count (flag-independent truncation)"""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    n = len(enc['input_ids'])
    if n <= k: return text
    offs = enc['offset_mapping']; h = k // 4
    c1 = offs[h][0]; c2 = offs[n - (k - h)][0]
    c1 = text.rfind('\n', 0, c1) + 1 or c1
    j = text.find('\n', c2)
    c2 = j + 1 if j >= 0 else c2
    if c2 <= c1: return text[:c1]
    return text[:c1] + '[...]\n' + text[c2:]


def load_bf16_map(path=None):
    path = path or os.path.join(ASSETS, 'precmap_w8a8_b8.json')
    return {(int(k.split('.')[0]), k.split('.')[1]) for k, v in json.load(open(path)).items() if v == 'bf16'}
