"""J5 short-length runtime: h2's QRT with every GEMM on sk.py's short-M kernels (tuned per (format, M, shape, epilogue)).
make(ln, head, spec):
  'bf16'            fold runtime, bf16 weights, sk GEMMs
  'w8a16' | 'w4a16' | 'w4a16g64' | 'mixw4w8'   fold runtime with weight-only codes (RTN codes built on the fly when WQ codes are absent;
                    speed does not depend on values; accuracy runs use wq.py's GPTQ codes through the same dequantized weights)
  'w8a8:<precmap>'  h2's rotated W8A8 runtime (H1 format) with QG.gemm / QG.swiglu_gemm replaced by sk int8 kernels
  suffix '+fr'      GDN core through fla fused_recurrent instead of chunk for single sequences (short-T variant)
"""
import os, sys, torch, triton
sys.path[:0] = [os.path.expanduser('~/work/j5'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1')]
import sk, qrt as Q, qgemm as QG, lean2 as L2, qk as K
import torch.nn.functional as F

GEMMS = ('Win', 'Wo', 'Wgu', 'Wd')
KEY = {'Win': 'Win_f', 'Wo': 'Wo', 'Wgu': 'Wgu_f', 'Wd': 'Wd'}


def slim(ln):
    """drop the non-fold weight copies Lean2 keeps (the fold runtime and h2's QRT read Win_f, Wo, Wgu_f, Wd)"""
    for d in ln.layers:
        for k in ('Wgu_il',):
            d.pop(k, None)
    torch.cuda.empty_cache()
    return ln


def rtn_qw(W, mode):
    W = W.float()
    if mode == 'bf16': return sk.QW('bf16', W.to(torch.bfloat16).contiguous())
    if mode == 'w8':
        s = (W.abs().amax(1) / 127).clamp_min(1e-9); q = torch.round(W / s[:, None]).clamp(-127, 127)
        return sk.QW('w8', q.to(torch.int8).contiguous(), s.contiguous())
    G = 64 if mode == 'w4g64' else 128
    N, Kd = W.shape; Wg = W.reshape(N, Kd // G, G); mx = Wg.amax(-1); mn = Wg.amin(-1)
    sc = ((mx - mn) / 15).clamp_min(1e-9); z = torch.round(-mn / sc).clamp(0, 15)
    q = (torch.round(Wg / sc[..., None]) + z[..., None]).clamp(0, 15).reshape(N, Kd)
    return sk.QW('w4', sk.pack_blocked4(q), sc.contiguous(), z.to(torch.uint8).contiguous(), G=G)



class FRMixin:
    """GDN core through fla fused_recurrent (one kernel per call) for short sequences: single = 1 call, packed = 2 (state, branches)."""
    fr = False

    def _gdn(self, i, d, proj, ra, cs, lay, cache, dq):
        if not self.fr or lay.kind == 'schema':
            return super()._gdn(i, d, proj, ra, cs, lay, cache, dq)
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as FR
        T = lay.T
        qkv3, ab = K.conv(proj, ra, cs, d['conv_w'], lay.prev, self._ztail, dq=dq)
        a = ab[:, 16:32].float(); b = ab[:, 0:16].float()
        gk = -d['A_log'].float().exp() * F.softplus(a + d['dt_bias']); beta = torch.sigmoid(b).to(qkv3.dtype)
        q, k, v = qkv3[0], qkv3[1], qkv3[2]
        if lay.kind == 'single':
            o, _ = FR(q[None], k[None], v[None], gk[None], beta[None], use_qk_l2norm_in_kernel=False)
            return o.reshape(T, 16, 128)
        Ls = lay.Ls; Mq = len(lay.qlens)
        o1, S = FR(q[None, :Ls], k[None, :Ls], v[None, :Ls], gk[None, :Ls], beta[None, :Ls], output_final_state=True, use_qk_l2norm_in_kernel=False)
        o2, _ = FR(q[None, Ls:], k[None, Ls:], v[None, Ls:], gk[None, Ls:], beta[None, Ls:], initial_state=S.expand(Mq, -1, -1, -1).contiguous(),
                   cu_seqlens=lay.cu, use_qk_l2norm_in_kernel=False)
        return torch.cat([o1[0], o2[0]], 0)


class FoldSK(FRMixin, Q.QRT):
    """fold runtime (bf16 activations) with per-GEMM weight formats on sk kernels"""
    def __init__(self, ln, head, fmt, fr=False):
        super().__init__(ln, head=head, prec='bf16'); self.tune = False
        self.fr = fr
        per = {'bf16': {k: 'bf16' for k in GEMMS}, 'w8a16': {k: 'w8' for k in GEMMS}, 'w4a16': {k: 'w4' for k in GEMMS},
               'w4a16g64': {k: 'w4g64' for k in GEMMS}, 'mixw4w8': {'Win': 'w8', 'Wo': 'w8', 'Wgu': 'w4', 'Wd': 'w4'}}[fmt]
        self.W = [{k: rtn_qw(d[KEY[k]], per[k]) for k in GEMMS} for d in self.L]
        torch.cuda.empty_cache()

    def g(self, a, i, k, epi, res=None, ss=None, ssout=None):
        return sk.launch(a, self.W[i][k], epi, res=res, ss=ss, ssout=ssout, Kd=2048)

    def _fwd_fold(self, ids, lay, cache, cos, sin):
        T = lay.T
        x = F.embedding(ids, self.ln2.embed).contiguous()
        ss = x.float().pow(2).sum(-1)
        for i, d in enumerate(self.L):
            proj = self.g(x, i, 'Win', 0, ss=ss)
            y = torch.empty(T, 2048, device=self.dev, dtype=torch.bfloat16)
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, None, None, lay, cache, False)
                K._gnorm_k[(triton.cdiv(T, self.rows),)](o, proj[:, 6144:], proj, proj, d['gn_w'], proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), self.eps,
                                                         DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=self.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, None, None, lay, cache, False, cos, sin)
                K._agate_k[(triton.cdiv(T, self.rows),)](o, proj, proj, proj, proj, proj, proj, proj, proj, proj, y, T, proj.stride(0), o.stride(0), o.stride(1),
                                                         DQ=False, OUTQ=False, HAD=1, QMAX=7., CLIP=1., BITS=8, ROWS=self.rows, num_warps=8)
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            self.g(y, i, 'Wo', 3, res=x, ssout=ss)
            m = self.g(x, i, 'Wgu', 1, ss=ss)
            ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
            self.g(m, i, 'Wd', 3, res=x, ssout=ss)
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps))


# ------------------------------------------------------------------ h2 int runtime on sk kernels
_ORIG = {}


def patch_qg():
    if _ORIG: return
    _ORIG['gemm'] = QG.gemm; _ORIG['swiglu'] = QG.swiglu_gemm
    cache = {}

    def wobj(B):
        key = (B.data_ptr(), tuple(B.shape))
        if key not in cache: cache[key] = sk.QW('w8', B, None)
        return cache[key]

    def gemm(kind, A, B, alpha, cfg, out=None):
        assert kind == 's8', kind
        return sk.launch(A, wobj(B), 4, alpha=alpha, scale=False, out=out)

    choice = {}
    def swiglu(kind, A, B, ra, cs, cfg=3, out=None):
        assert kind == 's8', kind
        key = (A.shape[0], B.shape[0], A.shape[1])
        if key not in choice:
            best = None
            for nm in [('c', 3), ('c', 1), ('s', 0)]:
                try:
                    if nm[0] == 'c': t = sk.time_cfg(lambda x: _ORIG['swiglu'](kind, A, B, ra, cs, cfg=nm[1]), [None] * 8)
                    else: t = sk.time_cfg(lambda x: sk.launch(A, sk.QW('w8', B, cs), 5, ra=ra, scale=True), [None] * 8)
                except Exception as ex:
                    print('swiglu cand fail', nm, key, repr(ex)[:300], flush=True); continue
                if best is None or t < best[0]: best = (t, nm)
            choice[key] = best[1]
        nm = choice[key]
        if nm[0] == 'c': return _ORIG['swiglu'](kind, A, B, ra, cs, cfg=nm[1], out=out)
        return sk.launch(A, sk.QW('w8', B, cs), 5, ra=ra, scale=True, out=out)
    QG.swiglu_gemm = swiglu


class IntSK(FRMixin, Q.QRT):
    """h2's rotated int runtime; each int8 GEMM runs on the faster of CUTLASS (5 configs) and sk (tuned), chosen per (M, N, K, epilogue)."""
    choice = {}

    def qgemm(self, A, e):
        key = ('g', A.shape[0], e['codes'].shape[0], A.shape[1])
        if key not in self.choice:
            def run_c(cfg): return _ORIG['gemm']('s8', A, e['codes'], e['alpha'], cfg)
            def run_s(): return sk.launch(A, sk.QW('w8', e['codes'], None), 4, alpha=e['alpha'], scale=False)
            best = None
            import cutsk
            cl = [(('c', c), (lambda c=c: run_c(c))) for c in range(11)] + [(('s', 0), run_s)]
            cl += [(('k', cs), (lambda cs=cs: cutsk.gemm(A, e['codes'], e['alpha'], cs[0], cs[1]))) for cs in cutsk.CANDS if (A.shape[1] // cs[1]) % 64 == 0]
            for nm, fn in cl:
                try:
                    t = sk.time_cfg(lambda x: fn(), [None] * 8)
                except Exception: continue
                if best is None or t < best[0]: best = (t, nm)
            self.choice[key] = best[1]
        nm = self.choice[key]
        if nm[0] == 'c': return _ORIG['gemm']('s8', A, e['codes'], e['alpha'], nm[1])
        if nm[0] == 'k':
            import cutsk
            return cutsk.gemm(A, e['codes'], e['alpha'], nm[1][0], nm[1][1])
        return sk.launch(A, sk.QW('w8', e['codes'], None), 4, alpha=e['alpha'], scale=False)


def make(ln, head, spec):
    fr = spec.endswith('+fr'); spec = spec.replace('+fr', '')
    if spec.startswith('w8a8'):
        patch_qg()
        prec = ('map:' + spec.split(':', 1)[1]) if ':' in spec else 'w8a8'
        m = IntSK(ln, head=head, prec=prec); m.tune = False; m.fr = fr
        return m
    return FoldSK(ln, head, spec, fr=fr)
