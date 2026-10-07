"""H2 runtime: hobson-v19 (24 layers) with low-bit tensor-core GEMMs, fused quantization glue and a compiled schema prefix.

prec: 'bf16' (= d1's fused fold runtime: all-Triton GEMMs, norm folded, SwiGLU / residual+sumsq epilogues) or a per-GEMM map
      {(layer, 'Win'|'Wo'|'Wgu'|'Wd'): 'bf16'|'w8a8'|'w4a8'|'w4a4'} / a uniform string.  Any non-bf16 GEMM => rotated runtime (H1 FORMAT v0):
      residual stored as x R1; Win/Wgu read gain-free RMSNorm(xR1) quantized per token by the residual-add kernel; Wo input rotated online by R2,
      Wd input by R4, both inside the producing kernel (gated norm / attention gate / SwiGLU) which also quantizes.
Layouts (Lay): 'single' one causal sequence; 'packed' state + M question branches (hobson's shared-prefix multi-question path);
      'schema' = this-that-model schema-first: [P = question bundle, cached][state][N answer slots]; per request only state + slots are computed.
"""
import os, sys, math, json, torch, torch.nn.functional as F
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import triton
import qk as K, qgemm as QG, rot as RT
import lean2 as L2
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from torch.nn.attention.bias import causal_lower_right

GEMMS = ('Win', 'Wo', 'Wgu', 'Wd')
QMAX = {4: 7.0, 8: 127.0}
KIND = {'w8a8': 's8', 'w4a8': 's8s4', 'w4a4': 's4'}
BITS = {'w8a8': (8, 8), 'w4a8': (4, 8), 'w4a4': (4, 4), 'bf16': (16, 16)}


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


def alpha_for(kind, K):
    qa, qw = {'s8': (127, 127), 's8s4': (127, 7), 's4': (7, 7)}[kind]
    mx = K * qa * qw
    a = 1.0
    while mx * a > 60000: a *= 0.5
    return a


class Lay:
    """host-side description of one forward's rows. kind: single | packed | schema."""
    def __init__(self, kind, Ls, qlens=(), nslots=0, P=0, dev='cuda'):
        self.kind = kind; self.Ls = Ls; self.qlens = list(qlens); self.N = nslots; self.P = P
        if kind == 'single':
            self.T = Ls; pos = list(range(Ls)); prev = [[t - 3 + j if t - 3 + j >= 0 else -1 for j in range(3)] for t in range(Ls)]
        elif kind == 'packed':
            self.T = Ls + sum(self.qlens); pos = list(range(Ls)); prev = [[t - 3 + j if t - 3 + j >= 0 else -1 for j in range(3)] for t in range(Ls)]
            self.seg = []; s = Ls
            for L in self.qlens:
                self.seg.append((s, s + L)); pos += list(range(Ls, Ls + L))
                for tau in range(L):
                    prev.append([(s + tau - 3 + j) if tau - 3 + j >= 0 else (Ls + tau - 3 + j) for j in range(3)])
                s += L
            self.cu = torch.tensor([0] + [sum(self.qlens[:i + 1]) for i in range(len(self.qlens))], device=dev, dtype=torch.long)
        elif kind == 'schema':
            self.T = Ls + nslots; pos = list(range(P, P + self.T))
            prev = [[t - 3 + j if t - 3 + j >= 0 else -2 - (t + j) for j in range(3)] for t in range(self.T)]   # tail row index = 3 + (t-3+j)
        self.pos = torch.tensor(pos, device=dev, dtype=torch.float32)
        self.prev = torch.tensor(prev, device=dev, dtype=torch.int32).contiguous()


class DenseRot:
    """learned dense orthogonal R1 (H5 FORMAT v0): x -> x R, inv: x -> x R^T (folded offline only)"""
    def __init__(self, R, dev):
        self.R = R.float().to(dev).contiguous()
    def __call__(self, x): return x.float() @ self.R
    def inv(self, x): return x.float() @ self.R.t()


def parse_prec(spec, nL=24):
    """'w4a4' | 'map:<json>[:<default>]' (keys 'i.k')"""
    if not spec.startswith('map:'): return spec
    parts = spec.split(':'); path = parts[1]; dflt = parts[2] if len(parts) > 2 else 'w4a4'
    mp = json.load(open(os.path.expanduser(path)))
    P = {(i, k): dflt for i in range(nL) for k in GEMMS}
    for key, v in mp.items():
        i, k = key.split('.'); P[(int(i), k)] = v
    return P


class QRT:
    def __init__(self, ln2, head=None, temps=None, prec='w4a4', ohead=False, rseed=1234, aclip4=0.9, aclip8=1.0, lora=None, wcodes=None, R1=None, lrot=None):
        self.ln2 = ln2; self.L = ln2.layers; self.dev = ln2.dev; self.eps = ln2.eps
        self.head = head; self.temps = temps or {}
        self.nL = len(self.L)
        if isinstance(prec, str): prec = parse_prec(prec, self.nL)
        if isinstance(prec, str): prec = {(i, k): prec for i in range(self.nL) for k in GEMMS}
        self.lrot = lrot
        self.pm = {key: prec.get(key, 'bf16') for key in [(i, k) for i in range(self.nL) for k in GEMMS]}
        self.fold = all(v == 'bf16' for v in self.pm.values())
        self.aclip = {4: aclip4, 8: aclip8}
        self.M = K.mats(self.dev)
        self.cfg_cache = {}
        self.tune = True
        self.ohead = ohead
        self.rows = int(os.environ.get('KROWS', '1'))
        self.evt = os.environ.get('EVT', '1') == '1'           # fused SwiGLU-epilogue gate_up GEMM (libh2evt.so)
        self.hprec = int(os.environ.get('HPREC', '3'))      # online-Hadamard dot precision: 0 tf32x3, 1 tf32, 2 fp16, 3 fp16 with row prescale
        if not self.fold:
            self.R1 = RT.Rot(2048, rseed, self.dev) if R1 is None else DenseRot(R1, self.dev); self.R4 = RT.Rot(6144, rseed + 2, self.dev)
            self.R2 = RT.Rot(2048, rseed + 1, self.dev)
            self.R2h = RT.Rot(2048, rseed + 1, self.dev, block=128); self.R2a = RT.Rot(2048, rseed + 1, self.dev, block=256)
            self.embed = self.R1(ln2.embed.float()).to(torch.bfloat16).contiguous()
            self.qw = [dict() for _ in range(self.nL)]
            for i in range(self.nL):
                for k in GEMMS:
                    self._build(i, k, lora, wcodes)
            torch.cuda.empty_cache()
        self.norm1 = (1.0 + ln2.norm_w).float()

    # ------------------------------------------------------------------ weights (H1 FORMAT v0 wfold + RTN, or external codes)
    def r2_for(self, i):
        if not self.ohead: return self.R2
        return self.R2a if self.L[i]['type'] != 'linear_attention' else self.R2h

    def wfold(self, i, k, lora=None):
        d = self.L[i]; W = d[k].float()
        if lora is not None and f'{i}.{k}.A' in lora:
            W = W + lora[f'{i}.{k}.B'].float().to(self.dev) @ lora[f'{i}.{k}.A'].float().to(self.dev)
        if k == 'Win': W = W * d['in1'][None, :]
        if k == 'Wgu': W = W * d['post1'][None, :]
        prec = self.pm[(i, k)]
        if k in ('Win', 'Wgu'): W = self.R1(W)
        elif prec != 'bf16': W = (self.R4 if k == 'Wd' else self.r2_for(i))(W)       # online input rotation only when the GEMM is quantized
        if k in ('Wo', 'Wd'): W = self.R1(W.t().contiguous()).t().contiguous()        # residual writer: R1^T W
        return W

    def _build(self, i, k, lora, wcodes):
        prec = self.pm[(i, k)]; e = {}
        if prec == 'bf16':
            W = self.wfold(i, k, lora)
            if k == 'Wgu':
                I = W.shape[0] // 2
                e['Wb_il'] = torch.stack([W[:I], W[I:]], 1).reshape(2 * I, -1).to(torch.bfloat16).contiguous()
            e['Wb'] = W.to(torch.bfloat16).contiguous()
        else:
            wb, ab = BITS[prec]; qmax = QMAX[wb]
            src = (wcodes.get(f'w{wb}') if ('w4' in wcodes or 'w8' in wcodes) else wcodes) if wcodes is not None else None
            if src is not None and (i, k) in src:
                q, s = src[(i, k)]; q = q.to(self.dev).float(); s = s.to(self.dev).float()
            else:
                W = self.wfold(i, k, lora)
                s = rtn_scales(W, qmax, wb == 4)
                q = torch.round(W / s[:, None]).clamp(-qmax, qmax)
                if self.lrot is not None and f'{i}.{k}.A' in self.lrot:    # H1 rotated-space QAT LoRA: codes = round((W_rot + B A) / s_base)
                    A = self.lrot[f'{i}.{k}.A'].float().to(self.dev); B = self.lrot[f'{i}.{k}.B'].float().to(self.dev)
                    q = torch.round((W + B @ A) / s[:, None]).clamp(-qmax, qmax)
                del W
            kind = KIND[prec]; Kd = q.shape[1]
            if k == 'Wgu' and self.evt and kind in ('s4', 's8'):      # fused SwiGLU epilogue: interleave gate/up rows and their scales
                I = q.shape[0] // 2
                q = torch.stack([q[:I], q[I:]], 1).reshape(2 * I, -1); s = torch.stack([s[:I], s[I:]], 1).reshape(2 * I)
                e['evt'] = True; e['cs_true'] = s.float().contiguous()
            e['codes'] = QG.pack4(q) if wb == 4 else q.to(torch.int8).contiguous()
            e['alpha'] = alpha_for(kind, Kd); e['kind'] = kind
            e['csa'] = (s / e['alpha']).float().contiguous()
            e['abits'] = ab
        e['prec'] = prec
        self.qw[i][k] = e

    # ------------------------------------------------------------------ GEMM dispatch
    @staticmethod
    def static_cfg(kind, M, N, K):
        # from res_gemm.json (A10G): best CUTLASS config per kind / shape
        if kind == 's4': return 0 if (K == 6144 and M < 2048) else 3
        if kind == 's8': return 3 if (M < 2500 or (N == 2048 and K == 2048)) else 1
        return 3 if (N == 2048 and M >= 2500) else 1

    def _cfg(self, kind, A, B, alpha):
        key = (kind, A.shape[0], B.shape[0], QG.kdim(kind, A))
        if not self.tune: return self.static_cfg(*key)
        if key not in self.cfg_cache:
            best = None
            C = torch.empty(A.shape[0], B.shape[0], device=A.device, dtype=torch.float16)
            for cfg in range(5):
                try:
                    QG.gemm(kind, A, B, alpha, cfg, out=C); torch.cuda.synchronize()
                except Exception: continue
                ts = []
                for _ in range(6):
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                    e0.record(); QG.gemm(kind, A, B, alpha, cfg, out=C); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
                t = sorted(ts)[3]
                if best is None or t < best[0]: best = (t, cfg)
            self.cfg_cache[key] = best[1]
        return self.cfg_cache[key]

    def qgemm(self, A, e):
        cfg = self._cfg(e['kind'], A, e['codes'], e['alpha'])
        return QG.gemm(e['kind'], A, e['codes'], e['alpha'], cfg)

    def _abuf(self, T, ncol, bits):
        return torch.empty(T, ncol if bits == 8 else ncol // 2, device=self.dev, dtype=torch.int8 if bits == 8 else torch.uint8)

    # ------------------------------------------------------------------ mixers (shared by fold and quantized paths)
    def _gdn(self, i, d, proj, ra, cs, lay, cache, dq):
        T = lay.T
        tail = cache['tail'][i] if lay.kind == 'schema' else self._ztail
        qkv3, ab = K.conv(proj, ra, cs, d['conv_w'], lay.prev, tail, dq=dq)
        a = ab[:, 16:32]; b = ab[:, 0:16]
        kw = dict(use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True)
        q, k, v = qkv3[0], qkv3[1], qkv3[2]
        if lay.kind == 'single':
            o, S = chunk_gated_delta_rule(q[None], k[None], v[None], a.reshape(1, T, 16), b.reshape(1, T, 16), output_final_state=cache is not None and cache.get('compile', False), **kw)
            if cache is not None and cache.get('compile', False):
                cache['S'][i] = S
                cache['tail'][i] = (proj[T - 3:, :6144].float() * (ra[T - 3:, None] * cs[None, :6144] if dq else 1.0)).to(torch.bfloat16).float().contiguous()
            return o.reshape(T, 16, 128)
        if lay.kind == 'schema':
            o, _ = chunk_gated_delta_rule(q[None], k[None], v[None], a.reshape(1, T, 16), b.reshape(1, T, 16), initial_state=cache['S'][i], **kw)
            return o.reshape(T, 16, 128)
        Ls = lay.Ls
        o1, S = chunk_gated_delta_rule(q[None, :Ls], k[None, :Ls], v[None, :Ls], a[:Ls].reshape(1, Ls, 16), b[:Ls].reshape(1, Ls, 16), output_final_state=True, **kw)
        Mq = len(lay.qlens); Tb = T - Ls
        o2, _ = chunk_gated_delta_rule(q[None, Ls:], k[None, Ls:], v[None, Ls:], a[Ls:].reshape(1, Tb, 16), b[Ls:].reshape(1, Tb, 16),
                                       initial_state=S.expand(Mq, -1, -1, -1).contiguous(), cu_seqlens=lay.cu, **kw)
        return torch.cat([o1[0], o2[0]], 0)

    def _attn(self, i, d, proj, ra, cs, lay, cache, dq, cos, sin):
        T = lay.T
        if lay.kind == 'schema':
            kb, vb = cache['kb'][i], cache['vb'][i]; koff = lay.P
        else:
            kb = torch.empty(T, 2, 256, device=self.dev, dtype=torch.bfloat16); vb = torch.empty_like(kb); koff = 0
        q = K.aprep(proj, ra, cs, d['qn'], d['kn'], cos, sin, kb, vb, koff, self.eps, dq=dq)
        qh = q.transpose(0, 1)[None]
        if lay.kind == 'single':
            o = F.scaled_dot_product_attention(qh, kb.transpose(0, 1)[None], vb.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
            if cache is not None and cache.get('compile', False):
                cache['kb'][i] = kb; cache['vb'][i] = vb
            return o[0]
        if lay.kind == 'schema':
            Lk = lay.P + T
            o = F.scaled_dot_product_attention(qh, kb[:Lk].transpose(0, 1)[None], vb[:Lk].transpose(0, 1)[None], attn_mask=causal_lower_right(T, Lk), enable_gqa=True)
            return o[0]
        Ls = lay.Ls
        ks = kb[:Ls]; vs = vb[:Ls]
        outs = [F.scaled_dot_product_attention(qh[:, :, :Ls], ks.transpose(0, 1)[None], vs.transpose(0, 1)[None], is_causal=True, enable_gqa=True)[0]]
        for (s, e) in lay.seg:
            kk = torch.cat([ks, kb[s:e]], 0); vv = torch.cat([vs, vb[s:e]], 0)
            outs.append(F.scaled_dot_product_attention(qh[:, :, s:e], kk.transpose(0, 1)[None], vv.transpose(0, 1)[None],
                                                       attn_mask=causal_lower_right(e - s, Ls + e - s), enable_gqa=True)[0])
        return torch.cat(outs, 1)

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, ids, lay, cache=None, keep_hn=False):
        """ids [T] (rows to compute); returns the gain-free normed final hidden nrm(x) in the UNROTATED basis for all rows if keep_hn,
        else the final residual buffer (rotated basis in the quantized runtime) plus the last kernel's normed output."""
        T = lay.T; dev = self.dev
        self._ztail = torch.zeros(1, 6144, device=dev)
        fr = lay.pos[:, None] * self.ln2.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
        if self.fold: return self._fwd_fold(ids, lay, cache, cos, sin)
        return self._fwd_q(ids, lay, cache, cos, sin)

    def _fwd_fold(self, ids, lay, cache, cos, sin):
        T = lay.T
        x = F.embedding(ids, self.ln2.embed).contiguous()
        ss = x.float().pow(2).sum(-1)
        for i, d in enumerate(self.L):
            proj = L2.tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=2048)
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, None, None, lay, cache, False)
                y = torch.empty(T, 2048, device=self.dev, dtype=torch.bfloat16)
                K._gnorm_k[(triton.cdiv(T, self.rows),)](o, proj[:, 6144:], proj, proj, d['gn_w'], proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), self.eps,
                                                 DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=self.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, None, None, lay, cache, False, cos, sin)
                y = torch.empty(T, 2048, device=self.dev, dtype=torch.bfloat16)
                K._agate_k[(triton.cdiv(T, self.rows),)](o, proj, proj, proj, proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), o.stride(0), o.stride(1),
                                                 DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=self.rows, num_warps=8)
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            L2.tgemm(y, d['Wo'], epi=3, res=x, ssout=ss)
            m = L2.tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=2048)
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            L2.tgemm(m, d['Wd'], epi=3, res=x, ssout=ss)
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))     # gain-free norm, unrotated basis

    def _fwd_q(self, ids, lay, cache, cos, sin):
        T = lay.T; dev = self.dev; eps = self.eps
        x = F.embedding(ids, self.embed).contiguous()
        nxt = self._next_in(0, 'Win', T)
        K.addq(x, None, None, None, nxt['q'], nxt['s'], nxt['hb'], eps, dq=False, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
        cur = nxt
        for i, d in enumerate(self.L):
            e = self.qw[i]['Win']
            proj, ra, cs, dq = self._run(cur, e)
            nxt = self._next_in(i, 'Wo', T)
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, ra, cs, lay, cache, dq)
                R2 = self.r2_for(i)
                K._gnorm_k[(triton.cdiv(T, self.rows),)](o, proj[:, 6144:], ra if dq else proj, cs if dq else proj, d['gn_w'], R2.sign, self.M['H32'], self.M['H64'], self.M['H128'],
                                                 nxt['q'], nxt['s'], nxt['hb'], T, proj.stride(0), eps, DQ=dq, OUTQ=nxt['outq'], HAD=2 if self.ohead else 1,
                                                 QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, ra, cs, lay, cache, dq, cos, sin)
                R2 = self.r2_for(i)
                K._agate_k[(triton.cdiv(T, self.rows),)](o, proj, ra if dq else proj, cs if dq else proj, R2.sign, self.M['H32'], self.M['H64'], self.M['H16'],
                                                 nxt['q'], nxt['s'], nxt['hb'], T, proj.stride(0), o.stride(0), o.stride(1), DQ=dq, OUTQ=nxt['outq'], HAD=2 if self.ohead else 1,
                                                 QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
            cur = nxt
            D, ra, cs, dq = self._run(cur, self.qw[i]['Wo'])
            nxt = self._next_in(i, 'Wgu', T)
            K.addq(x, D, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            cur = nxt
            eg = self.qw[i]['Wgu']; nxt = self._next_in(i, 'Wd', T)
            if eg.get('evt'):
                GU = QG.swiglu_gemm(eg['kind'], cur['q'], eg['codes'], cur['s'], eg['cs_true'], cfg=(1 if (eg['kind'] == 's8' and T >= 2500) else 3)); dq = False; ra = cs = None; mi = True
            else:
                GU, ra, cs, dq = self._run(cur, eg); mi = False
            K._swiglu_k[(triton.cdiv(T, self.rows),)](GU, ra if dq else GU, cs if dq else GU, self.R4.sign, self.M['P12T'], self.M['H16'], self.M['H32'], nxt['q'], nxt['s'], nxt['hb'], T,
                              DQ=dq, OUTQ=nxt['outq'], QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, MIN=mi, num_warps=8)
            cur = nxt
            D, ra, cs, dq = self._run(cur, self.qw[i]['Wd'])
            nxt = self._next_in(i + 1, 'Win', T) if i + 1 < self.nL else self._next_in(None, 'final', T)
            K.addq(x, D, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            cur = nxt
        return cur['hb']      # gain-free norm of x R1 (rotated basis)

    def _next_in(self, i, k, T):
        """buffers for the input of GEMM (i, k): codes+scale if quantized, else bf16."""
        if k == 'final' or self.pm[(i, k)] == 'bf16':
            return dict(outq=False, q=None, s=None, hb=torch.empty(T, 6144 if k == 'Wd' else 2048, device=self.dev, dtype=torch.bfloat16),
                        qmax=7., clip=1., bits=8)
        e = self.qw[i][k]; ab = e['abits']; ncol = 6144 if k == 'Wd' else 2048
        return dict(outq=True, q=self._abuf(T, ncol, ab), s=torch.empty(T, device=self.dev, dtype=torch.float32), hb=None,
                    qmax=QMAX[ab], clip=self.aclip[ab], bits=ab)

    def _run(self, cur, e):
        if e['prec'] == 'bf16':
            return cur['hb'] @ e['Wb'].t(), None, None, False
        return self.qgemm(cur['q'], e), cur['s'], e['csa'], True

    # ------------------------------------------------------------------ readout
    def unrot(self, hn):
        """gain-free normed rows -> the bf16 model's final hidden (unrotated, with gain)"""
        h = hn.float() if self.fold else self.R1.inv(hn.float())
        return (h * self.norm1).to(torch.bfloat16)

    # ------------------------------------------------------------------ compiled schema
    @torch.no_grad()
    def compile_prefix(self, prefix_ids, Tmax):
        """run the question bundle once; cache GDN states, conv tails, attention K/V (in buffers with room for Tmax more rows), final prefix rows."""
        P = prefix_ids.shape[0]
        lay = Lay('single', P, dev=self.dev)
        cache = dict(compile=True, S={}, tail={}, kb={}, vb={})
        hn = self.forward(prefix_ids, lay, cache)
        for i in list(cache['kb']):
            kb = torch.zeros(P + Tmax, 2, 256, device=self.dev, dtype=torch.bfloat16); kb[:P] = cache['kb'][i]; cache['kb'][i] = kb
            vb = torch.zeros(P + Tmax, 2, 256, device=self.dev, dtype=torch.bfloat16); vb[:P] = cache['vb'][i]; cache['vb'][i] = vb
        cache['compile'] = False; cache['P'] = P
        cache['hP'] = self.unrot(hn)        # final hidden of the prefix rows (option rows are read from here)
        return cache
