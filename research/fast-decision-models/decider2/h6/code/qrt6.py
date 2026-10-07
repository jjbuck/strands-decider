"""H6: H2's low-bit runtime (qrt.QRT) + mixed-row precision + H1's rotated-space LoRA rule on external (GPTQ) codes.

Row split: forward(ids, lay, cache, r0=R) runs rows [0, R) through the low-bit GEMMs and rows [R, T) through bf16 GEMMs, in ONE forward.
  No kernel changes: the producing glue kernels are launched on two row slices (OUTQ=True -> codes, OUTQ=False -> bf16), the int GEMM writes
  rows [0,R) of the fp16 'acc*alpha' buffer, and the bf16 rows write the same buffer in that domain (y / csa / ra_b with their own per-row
  scale ra_b), so every consumer kernel (conv, aprep, gnorm, agate, addq, swiglu) runs unchanged with DQ=True over all T rows.
  r0=0: every row bf16 (used to compile the question bundle in bf16 once).  r0=None: H2's uniform path.
bf16-row weights = latent rotated weight W'' + B A (H1 LoRA rule), un-rotated on the input side for Wo/Wd (the OUTQ=False glue output is unrotated).
LoRA rule (H1 FORMAT v1 / h1qat): codes = round((q*s + B A) / s) for GPTQ codes, round((W'' + B A)/s) for RTN; scales fixed.
"""
import os, sys, json, torch, triton, torch.nn.functional as F
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import qrt as Q
from qrt import K, QG, RT, GEMMS, QMAX, KIND, BITS, rtn_scales, alpha_for, parse_prec, Lay, DenseRot
import triton.language as tl
from qk import _qstore1


@triton.jit
def _addq2_k(X, D, RA, CS, Q, SA, HB, T, eps, HAS_D: tl.constexpr, DQ: tl.constexpr, OUTQ: tl.constexpr, BOTH: tl.constexpr,
             QMAX: tl.constexpr, CLIP: tl.constexpr, BITS: tl.constexpr, R: tl.constexpr):
    # = qk._addq_k, plus BOTH: also store the bf16 gain-free normed row (input of the bf16 GDN b/a gate GEMM, H1 'ba16')
    c = tl.arange(0, 2048)
    if DQ:
        cs = tl.load(CS + c)
    for i in range(R):
        r = (tl.program_id(0) * R + i).to(tl.int64)
        if r < T:
            x = tl.load(X + r * 2048 + c).to(tl.float32)
            if HAS_D:
                d = tl.load(D + r * 2048 + c).to(tl.float32)
                if DQ:
                    d = (d * tl.load(RA + r)) * cs
                xb = (x + d.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
                tl.store(X + r * 2048 + c, xb)
                x = xb.to(tl.float32)
            rs = tl.rsqrt(tl.sum(x * x, 0) / 2048 + eps)
            xn = x * rs
            if OUTQ:
                _qstore1(xn, r, Q, SA, 2048, 2048, QMAX, CLIP, BITS)
                if BOTH:
                    tl.store(HB + r * 2048 + c, xn.to(tl.bfloat16))
            else:
                tl.store(HB + r * 2048 + c, xn.to(tl.bfloat16))


def addq2(x, d, ra, cs, q, sa, hb, eps, dq, outq, both, qmax, clip, bits):
    T = x.shape[0]
    _addq2_k[(T,)](x, d if d is not None else x, ra if ra is not None else x, cs if cs is not None else x,
                   q if q is not None else x, sa if sa is not None else x, hb if hb is not None else x, T, eps,
                   HAS_D=d is not None, DQ=dq, OUTQ=outq, BOTH=both, QMAX=qmax, CLIP=clip, BITS=bits, R=1, num_warps=8)


class QRT6(Q.QRT):
    def __init__(self, ln2, head=None, prec='w4a4', lrot=None, wcodes=None, split=False, force_rot=False, ohead=False, rseed=1234,
                 aclip4=0.9, aclip8=1.0, R1=None, slim=False):
        self.ln2 = ln2; self.L = ln2.layers; self.dev = ln2.dev; self.eps = ln2.eps
        self.head = head; self.temps = {}; self.nL = len(self.L)
        if isinstance(prec, str): prec = parse_prec(prec, self.nL)
        if isinstance(prec, str): prec = {(i, k): prec for i in range(self.nL) for k in GEMMS}
        self.lrot = lrot; self.split = split
        self.lr_bf16 = not (lrot is not None and lrot.get('_lora_rows') == 'state')   # 'state': LoRA only in the low-bit rows' codes
        self.pm = {key: prec.get(key, 'bf16') for key in [(i, k) for i in range(self.nL) for k in GEMMS]}
        self.fold = all(v == 'bf16' for v in self.pm.values()) and not force_rot and lrot is None
        self.aclip = {4: aclip4, 8: aclip8}
        self.M = K.mats(self.dev); self.cfg_cache = {}; self.tune = True; self.ohead = ohead
        self.rows = int(os.environ.get('KROWS', '1')); self.evt = os.environ.get('EVT', '1') == '1'; self.hprec = int(os.environ.get('HPREC', '3'))
        self._r0 = None
        self.slot8 = os.environ.get('SLOT8', '0') == '1'  # high-precision rows (r0 > 0) at W8A8 (GPTQ8 codes) instead of bf16
        self.ba16 = os.environ.get('BA16', '0') == '1'    # H1: GDN b/a gate rows (Win rows 8192..8223) as a bf16 GEMM on the unquantized input
        self.ovl = os.environ.get('OVL', '0') == '1'      # bf16 rows on a side stream, overlapped with the int GEMM of the low-bit rows
        self.s2 = torch.cuda.Stream() if self.ovl else None
        if not self.fold:
            self.R1 = RT.Rot(2048, rseed, self.dev) if R1 is None else DenseRot(R1, self.dev); self.R4 = RT.Rot(6144, rseed + 2, self.dev)
            self.R2 = RT.Rot(2048, rseed + 1, self.dev)
            self.R2h = RT.Rot(2048, rseed + 1, self.dev, block=128); self.R2a = RT.Rot(2048, rseed + 1, self.dev, block=256)
            E = ln2.embed; self.embed = torch.empty(E.shape, device=self.dev, dtype=torch.bfloat16)
            for c in range(0, E.shape[0], 16384): self.embed[c:c + 16384] = self.R1(E[c:c + 16384].float()).to(torch.bfloat16)
            self.qw = [dict() for _ in range(self.nL)]
            for i in range(self.nL):
                for k in GEMMS:
                    self._build(i, k, None, wcodes)
                if slim:
                    d = self.L[i]
                    for k_ in [k_ for k_, v in d.items() if torch.is_tensor(v) and v.numel() > 1_000_000]: del d[k_]
                    torch.cuda.empty_cache()
            if slim: ln2.embed = None
            torch.cuda.empty_cache()
        self.norm1 = (1.0 + ln2.norm_w).float()

    # ------------------------------------------------------------------ weights
    def rin(self, i, k):
        if k in ('Win', 'Wgu'): return self.R1
        return self.R4 if k == 'Wd' else self.r2_for(i)

    def wrot(self, i, k):
        """H1 FORMAT rotated, gain-folded weight W'' (input side always rotated; residual writers R1^T W)."""
        d = self.L[i]; W = d[k].float()
        if k == 'Win': W = W * d['in1'][None, :]
        if k == 'Wgu': W = W * d['post1'][None, :]
        W = self.rin(i, k)(W)
        if k in ('Wo', 'Wd'): W = self.R1(W.t().contiguous()).t().contiguous()
        return W

    def delta(self, i, k):
        if self.lrot is None or f'{i}.{k}.A' not in self.lrot: return None
        A = self.lrot[f'{i}.{k}.A'].float().to(self.dev); B = self.lrot[f'{i}.{k}.B'].float().to(self.dev)
        return B @ A

    def bf16_weight(self, i, k):
        W = self.wrot(i, k); dl = self.delta(i, k) if (self.pm[(i, k)] == 'bf16' or getattr(self, 'lr_bf16', True)) else None
        if dl is not None: W[:dl.shape[0]] += dl
        if k in ('Wo', 'Wd'): W = self.rin(i, k).inv(W)
        return W

    def _build(self, i, k, lora, wcodes):
        prec = self.pm[(i, k)]; e = dict(prec=prec)
        if prec == 'bf16':
            e['Wb'] = self.bf16_weight(i, k).to(torch.bfloat16).contiguous()
        else:
            wb, ab = BITS[prec]; qmax = QMAX[wb]
            src = (wcodes.get(f'w{wb}') if ('w4' in wcodes or 'w8' in wcodes) else wcodes) if wcodes is not None else None
            dl = self.delta(i, k)
            if src is not None and (i, k) in src:
                q, s = src[(i, k)]; q = q.to(self.dev).float(); s = s.to(self.dev).float()
                if dl is not None:
                    lat = q * s[:, None]; lat[:dl.shape[0]] += dl
                    q = torch.round(lat / s[:, None]).clamp(-qmax, qmax); del lat
            else:
                W = self.wrot(i, k); s = rtn_scales(W, qmax, wb == 4)
                if dl is not None: W[:dl.shape[0]] += dl
                q = torch.round(W / s[:, None]).clamp(-qmax, qmax); del W
            kind = KIND[prec]; Kd = q.shape[1]
            if k == 'Wgu' and self.evt and kind in ('s4', 's8'):
                I = q.shape[0] // 2
                q = torch.stack([q[:I], q[I:]], 1).reshape(2 * I, -1); s = torch.stack([s[:I], s[I:]], 1).reshape(2 * I)
                e['evt'] = True; e['cs_true'] = s.float().contiguous()
            e['codes'] = QG.pack4(q) if wb == 4 else q.to(torch.int8).contiguous()
            e['alpha'] = alpha_for(kind, Kd); e['kind'] = kind
            e['csa'] = (s / e['alpha']).float().contiguous()
            e['abits'] = ab
            if self.ba16 and k == 'Win' and self.L[i]['type'] == 'linear_attention':
                e['Wba'] = self.bf16_weight(i, k)[8192:8224].to(torch.bfloat16).contiguous()
            if self.slot8 and wcodes is not None and 'w8' in wcodes:
                q8, s8w = wcodes['w8'][(i, k)]; q8 = q8.to(self.dev).float(); s8w = s8w.to(self.dev).float()
                dl8 = self.delta(i, k) if getattr(self, 'lr_bf16', True) else None
                if dl8 is not None:
                    lat = q8 * s8w[:, None]; lat[:dl8.shape[0]] += dl8; q8 = torch.round(lat / s8w[:, None]).clamp(-127, 127); del lat
                e['codes8'] = q8.to(torch.int8).contiguous(); e['alpha8'] = alpha_for('s8', q8.shape[1])
                if k in ('Wo', 'Wd'): e['rin8'] = self.rin(i, k)       # the OUTQ=False glue output is unrotated; W8 codes expect R2/R4 inputs
                e['csa8'] = (s8w / e['alpha8']).float().contiguous()
                if not e.get('evt'): e['ratio8'] = (e['csa8'] / e['csa']).contiguous()
                del q8
            if self.split:
                Wb = self.bf16_weight(i, k)
                if e.get('evt'): e['Wb'] = Wb.to(torch.bfloat16).contiguous()
                else: e['Wbs'] = (Wb / e['csa'][:, None]).to(torch.bfloat16).contiguous()
                del Wb
        self.qw[i][k] = e

    def slim(self, P=None):
        """after building a rotated runtime: drop the dense layer weights (only codes / bf16 copies are used) and the HF torso"""
        import gc
        assert not self.fold
        for d in self.L:
            for k_ in [k_ for k_, v in d.items() if torch.is_tensor(v) and v.numel() > 1_000_000]: del d[k_]
        self.ln2.embed = None
        if P is not None:
            P.model.torso = None; P.tm = None
            try: P.eng.model.torso = None
            except Exception: pass
        gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------ forward with row split
    @torch.no_grad()
    def forward(self, ids, lay, cache=None, keep_hn=False, r0=None, lowrows=None):
        self._r0 = r0; self._lowrows = lowrows
        if lowrows is not None:
            self._lowmask = torch.zeros(lay.T, dtype=torch.bool, device=self.dev); self._lowmask[lowrows] = True
        try:
            return super().forward(ids, lay, cache, keep_hn)
        finally:
            self._r0 = None; self._lowrows = None

    def _fwd_q(self, ids, lay, cache, cos, sin):
        r0 = self._r0
        if self.ba16 and (r0 is None or r0 >= lay.T): return self._fwd_split(ids, lay, cache, cos, sin, lay.T)
        if r0 is None or r0 >= lay.T: return super()._fwd_q(ids, lay, cache, cos, sin)
        assert self.split, 'row split needs split=True (bf16 copies of the quantized GEMMs)'
        return self._fwd_split(ids, lay, cache, cos, sin, r0)

    def _nin(self, i, k, T, r0):
        ncol = 6144 if k == 'Wd' else 2048
        if k == 'final' or self.pm[(i, k)] == 'bf16':
            return dict(rq=0, outq=False, q=None, s=None, hb=torch.empty(T, ncol, device=self.dev, dtype=torch.bfloat16), qmax=7., clip=1., bits=8)
        e = self.qw[i][k]; ab = e['abits']; rq = r0
        if getattr(self, '_lowrows', None) is not None and rq < T:
            bq = self.ba16 and 'Wba' in e and rq > 0
            return dict(rq=rq, full=True, outq=True, q=self._abuf(T, ncol, ab), s=torch.empty(T, device=self.dev, dtype=torch.float32),
                        hb=torch.empty(T - rq, ncol, device=self.dev, dtype=torch.bfloat16),
                        hq=torch.empty(rq, ncol, device=self.dev, dtype=torch.bfloat16) if bq else None,
                        qmax=QMAX[ab], clip=self.aclip[ab], bits=ab)
        both = self.ba16 and 'Wba' in e and rq > 0
        return dict(rq=rq, outq=True, q=self._abuf(rq, ncol, ab) if rq > 0 else None, s=torch.empty(T, device=self.dev, dtype=torch.float32),
                    hb=torch.empty(T - rq, ncol, device=self.dev, dtype=torch.bfloat16) if rq < T else None,
                    hq=torch.empty(rq, ncol, device=self.dev, dtype=torch.bfloat16) if both else None,
                    qmax=QMAX[ab], clip=self.aclip[ab], bits=ab)

    def _addq_parts(self, x, D, ra, cs, dq, nxt, T):
        if nxt.get('full'):
            rq = nxt['rq']; dq_ = dq if D is not None else False
            if rq > 0 and nxt.get('hq') is not None:
                addq2(x[:rq], D[:rq] if D is not None else None, ra[:rq] if dq_ else None, cs if dq_ else None, nxt['q'], nxt['s'], nxt['hq'], self.eps,
                      dq=dq_, outq=True, both=True, qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            elif rq > 0:
                K.addq(x[:rq], D[:rq] if D is not None else None, ra[:rq] if dq_ else None, cs if dq_ else None, nxt['q'], nxt['s'], None, self.eps,
                       dq=dq_, outq=True, qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            qv = nxt['q'][rq:]; sv = nxt['s'][rq:]
            addq2(x[rq:], D[rq:] if D is not None else None, ra[rq:] if dq_ else None, cs if dq_ else None, qv, sv, nxt['hb'], self.eps,
                  dq=dq_, outq=True, both=True, qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            return
        for (a, b, oq, q, s, hb) in self._parts(nxt, T):
            if oq and nxt.get('hq') is not None:
                addq2(x[a:b], D[a:b] if D is not None else None, ra[a:b] if dq else None, cs if dq else None, q, s, nxt['hq'], self.eps,
                      dq=dq if D is not None else False, outq=True, both=True, qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            else:
                K.addq(x[a:b], D[a:b] if D is not None else None, ra[a:b] if dq else None, cs if dq else None, q, s, hb, self.eps,
                       dq=dq if D is not None else False, outq=oq, qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])

    @staticmethod
    def _parts(nxt, T):
        rq = nxt['rq']; out = []
        if nxt.get('full'):
            return [(0, T, True, nxt['q'], nxt['s'], None), (rq, T, False, None, None, nxt['hb'])]
        if rq > 0: out.append((0, rq, True, nxt['q'], nxt['s'], None))
        if rq < T: out.append((rq, T, False, None, None, nxt['hb']))
        return out

    def _run6(self, cur, e, T):
        if e['prec'] == 'bf16':
            return cur['hb'] @ e['Wb'].t(), None, None, False
        rq = cur['rq']; N = e['codes'].shape[0]
        C = torch.empty(T, N, device=self.dev, dtype=torch.float16)
        if cur.get('full'):
            QG.gemm(e['kind'], cur['q'], e['codes'], e['alpha'], self._cfg(e['kind'], cur['q'], e['codes'], e['alpha']), out=C)
            yb = (cur['hb'] @ e['Wbs'].t()).float()
            rb = yb.abs().amax(-1).clamp_min(1e-20) / 30000.
            keep = self._lowmask[rq:]                      # True = keep the low-bit result
            Cb = (yb / rb[:, None]).to(torch.float16)
            C[rq:] = torch.where(keep[:, None], C[rq:], Cb)
            cur['s'][rq:] = torch.where(keep, cur['s'][rq:], rb)
            return C, cur['s'], e['csa'], True
        def bpart():
            if self.slot8 and rq > 0 and 'codes8' in e:
                hb = cur['hb'].float()
                if 'rin8' in e: hb = e['rin8'](hb)
                sa8 = hb.abs().amax(-1).clamp_min(1e-8) / 127.
                a8 = torch.round(hb / sa8[:, None]).clamp_(-127, 127).to(torch.int8)
                C8 = QG.gemm('s8', a8, e['codes8'], e['alpha8'], 3)
                yo = C8.float() * e['ratio8'][None, :]
                f8 = yo.abs().amax(-1).clamp_min(1e-20) / 30000.
                cur['s'][rq:] = sa8 * f8
                C[rq:] = (yo / f8[:, None]).to(torch.float16)
                return
            yb = (cur['hb'] @ e['Wbs'].t()).float()
            rb = yb.abs().amax(-1).clamp_min(1e-20) / 30000.
            cur['s'][rq:] = rb
            C[rq:] = (yb / rb[:, None]).to(torch.float16)
        ov = self.ovl and 0 < rq < T
        if ov:
            self.s2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.s2): bpart()
        if rq > 0:
            cfg = self._cfg(e['kind'], cur['q'], e['codes'], e['alpha'])
            QG.gemm(e['kind'], cur['q'], e['codes'], e['alpha'], cfg, out=C[:rq])
        if ov: torch.cuda.current_stream().wait_stream(self.s2)
        elif rq < T: bpart()
        return C, cur['s'], e['csa'], True

    def _fwd_split(self, ids, lay, cache, cos, sin, r0):
        T = lay.T; eps = self.eps; M = self.M
        x = F.embedding(ids, self.embed).contiguous()
        nxt = self._nin(0, 'Win', T, r0)
        self._addq_parts(x, None, None, None, False, nxt, T)
        cur = nxt
        for i, d in enumerate(self.L):
            proj, ra, cs, dq = self._run6(cur, self.qw[i]['Win'], T)
            nxt = self._nin(i, 'Wo', T, r0)
            R2 = self.r2_for(i)
            if d['type'] == 'linear_attention':
                eW = self.qw[i]['Win']
                self._ba_rows = (cur['hq'] @ eW['Wba'].t()) if (cur.get('hq') is not None) else None
                o = self._gdn(i, d, proj, ra, cs, lay, cache, dq)
                self._ba_rows = None
                for (a, b, oq, q, s, hb) in self._parts(nxt, T):
                    n = b - a
                    K._gnorm_k[(triton.cdiv(n, self.rows),)](o[a:b], proj[a:b, 6144:], ra[a:b] if dq else proj, cs if dq else proj, d['gn_w'], R2.sign, M['H32'], M['H64'], M['H128'],
                                                     q, s, hb, n, proj.stride(0), eps, DQ=dq, OUTQ=oq, HAD=2 if self.ohead else 1,
                                                     QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, ra, cs, lay, cache, dq, cos, sin)
                for (a, b, oq, q, s, hb) in self._parts(nxt, T):
                    n = b - a
                    K._agate_k[(triton.cdiv(n, self.rows),)](o[:, a:b], proj[a:b], ra[a:b] if dq else proj, cs if dq else proj, R2.sign, M['H32'], M['H64'], M['H16'],
                                                     q, s, hb, n, proj.stride(0), o.stride(0), o.stride(1), DQ=dq, OUTQ=oq, HAD=2 if self.ohead else 1,
                                                     QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
            cur = nxt
            D, ra, cs, dq = self._run6(cur, self.qw[i]['Wo'], T)
            nxt = self._nin(i, 'Wgu', T, r0)
            self._addq_parts(x, D, ra, cs, dq, nxt, T)
            cur = nxt
            eg = self.qw[i]['Wgu']; nxt = self._nin(i, 'Wd', T, r0)
            if eg.get('evt') and cur.get('full'):
                rq = cur['rq']
                GU = torch.empty(T, 6144, device=self.dev, dtype=torch.bfloat16)
                QG.swiglu_gemm(eg['kind'], cur['q'], eg['codes'], cur['s'], eg['cs_true'], cfg=(1 if (eg['kind'] == 's8' and T >= 2500) else 3), out=GU)
                gu = cur['hb'] @ eg['Wb'].t()
                g_ = gu[:, :6144].float(); u_ = gu[:, 6144:].float()
                mb = ((g_ * torch.sigmoid(g_)).to(torch.bfloat16).float() * u_).to(torch.bfloat16)
                GU[rq:] = torch.where(self._lowmask[rq:, None], GU[rq:], mb)
                dq = False; ra = cs = None; mi = True
            elif eg.get('evt'):
                rq = cur['rq']
                GU = torch.empty(T, 6144, device=self.dev, dtype=torch.bfloat16)
                def gpart():
                    if self.slot8 and rq > 0 and 'codes8' in eg:
                        hb = cur['hb'].float(); sa8 = hb.abs().amax(-1).clamp_min(1e-8) / 127.
                        a8 = torch.round(hb / sa8[:, None]).clamp_(-127, 127).to(torch.int8)
                        gu = QG.gemm('s8', a8, eg['codes8'], eg['alpha8'], 3).float() * sa8[:, None] * eg['csa8'][None, :]
                        gu = gu.to(torch.bfloat16)
                    else:
                        gu = cur['hb'] @ eg['Wb'].t()
                    g_ = gu[:, :6144].float(); u_ = gu[:, 6144:].float()
                    GU[rq:] = ((g_ * torch.sigmoid(g_)).to(torch.bfloat16).float() * u_).to(torch.bfloat16)
                ov = self.ovl and 0 < rq < T
                if ov:
                    self.s2.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.s2): gpart()
                if rq > 0:
                    QG.swiglu_gemm(eg['kind'], cur['q'], eg['codes'], cur['s'], eg['cs_true'], cfg=(1 if (eg['kind'] == 's8' and rq >= 2500) else 3), out=GU[:rq])
                if ov: torch.cuda.current_stream().wait_stream(self.s2)
                elif rq < T: gpart()
                dq = False; ra = cs = None; mi = True
            else:
                GU, ra, cs, dq = self._run6(cur, eg, T); mi = False
            for (a, b, oq, q, s, hb) in self._parts(nxt, T):
                n = b - a
                K._swiglu_k[(triton.cdiv(n, self.rows),)](GU[a:b], ra[a:b] if dq else GU, cs if dq else GU, self.R4.sign, M['P12T'], M['H16'], M['H32'], q, s, hb, n,
                                                  DQ=dq, OUTQ=oq, QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, MIN=mi, num_warps=8)
            cur = nxt
            D, ra, cs, dq = self._run6(cur, self.qw[i]['Wd'], T)
            nxt = self._nin(i + 1, 'Win', T, r0) if i + 1 < self.nL else self._nin(None, 'final', T, r0)
            self._addq_parts(x, D, ra, cs, dq, nxt, T)
            cur = nxt
        return cur['hb']

    _ba_rows = None

    def _gdn(self, i, d, proj, ra, cs, lay, cache, dq):
        if self._ba_rows is None: return super()._gdn(i, d, proj, ra, cs, lay, cache, dq)
        orig = K.conv; bar = self._ba_rows
        def conv2(*a_, **k_):
            out, ab = orig(*a_, **k_)
            ab[:bar.shape[0]] = bar.to(ab.dtype)
            return out, ab
        K.conv = conv2
        try:
            return super()._gdn(i, d, proj, ra, cs, lay, cache, dq)
        finally:
            K.conv = orig

    # ------------------------------------------------------------------ compiled schema (bundle optionally in bf16)
    @torch.no_grad()
    def compile_prefix(self, prefix_ids, Tmax, r0=None):
        P = prefix_ids.shape[0]
        lay = Lay('single', P, dev=self.dev)
        cache = dict(compile=True, S={}, tail={}, kb={}, vb={})
        hn = self.forward(prefix_ids, lay, cache, r0=r0)
        for i in list(cache['kb']):
            kb = torch.zeros(P + Tmax, 2, 256, device=self.dev, dtype=torch.bfloat16); kb[:P] = cache['kb'][i]; cache['kb'][i] = kb
            vb = torch.zeros(P + Tmax, 2, 256, device=self.dev, dtype=torch.bfloat16); vb[:P] = cache['vb'][i]; cache['vb'][i] = vb
        cache['compile'] = False; cache['P'] = P
        cache['hP'] = self.unrot(hn)
        return cache


def load_codes(spec):
    """'w8=path,w4=path' or a single path -> dict for QRT wcodes"""
    if not spec: return None
    if '=' in spec:
        return {kv.split('=')[0]: torch.load(os.path.expanduser(kv.split('=')[1])) for kv in spec.split(',')}
    return torch.load(os.path.expanduser(spec))


def load_lrot(path, dev='cuda'):
    if not path: return None
    sd = torch.load(os.path.expanduser(path), map_location=dev)
    out = {k: v for k, v in sd.items() if not k.startswith('_')}
    if sd.get('_lora_rows') == 'state': out['_lora_rows'] = 'state'
    return out
