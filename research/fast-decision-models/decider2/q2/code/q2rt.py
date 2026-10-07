"""Q2 runtime: H2's QRT (hobson-v19, rotated low-bit runtime, H1 FORMAT v1) with Q2's row-role GEMMs (FORMATS.md section 8).

fmt 'w4q8': every GEMM runs state rows (t < q0) as W4A4 per token and question rows (t >= q0) as W8A8 per token, in ONE q2gemm launch
            (two problems; the int8 tiles are scheduled first). The quantize step inside H2's glue kernels (qk2: H2's qk.py + RR)
            writes int4 codes for state rows and int8 codes for question rows. GEMM outputs are bf16 already dequantized
            (f = (acc * s_t) * s_w), so the consumers run with unit scales. gate_up uses q2gemm's SwiGLU epilogue.
fmt 'w8'  : all rows W8A8 through q2gemm (bf16-direct epilogue) - a same-kernel control for the int8 rows.
fmt 'w4'  : all rows W4A4 through q2gemm.
fmt may also be a dict {(layer, 'Win'|'Wo'|'Wgu'|'Wd'): 'rr'|'w4'|'w8'} (default 'rr'), or 'map:<json path>' with keys 'i.k'.
Set m.q0 (number of state rows) before each forward / graph capture. Weights: GPTQ codes from wcodes {'w4': .., 'w8': ..} or RTN.
"""
import os, sys, torch, triton, torch.nn.functional as F
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import qrt as Q
import qk2 as K
import q2k as QK

CFG_STATIC = int(os.environ.get('Q2CFG', '4'))


class QRT2(Q.QRT):
    def __init__(self, ln2, head=None, fmt='w4q8', wcodes=None, tune=True, **kw):
        import json
        nL = len(ln2.layers)
        if isinstance(fmt, str) and fmt.startswith('map:'):
            mp = json.load(open(os.path.expanduser(fmt[4:])))
            fmt = {(int(k.split('.')[0]), k.split('.')[1]): v for k, v in mp.items()}
        if isinstance(fmt, dict): self.gfmt = {(i, k): fmt.get((i, k), 'rr') for i in range(nL) for k in Q.GEMMS}
        else: self.gfmt = {(i, k): {'w4q8': 'rr', 'w8': 'w8', 'w4': 'w4'}[fmt] for i in range(nL) for k in Q.GEMMS}
        self.fmt = 'w4q8'
        super().__init__(ln2, head=head, prec='w8a8', wcodes=wcodes, **kw)
        self.q0 = 0
        self.rrb = 1.0
        self.pos = None; self.n4 = 0; self.rows4 = None; self.rows8 = None    # B8 arbitrary row partition (set_rowmask)
        self.q2tune = tune
        self.q2cfg = {}
        dev = self.dev
        self.ones = torch.ones(16384, device=dev)
        for i in range(self.nL):
            for k in Q.GEMMS:
                e = self.qw[i][k]
                e['sw8'] = (e['cs_true'] if e.get('evt') else e['csa'] * e['alpha']).float().contiguous()
                e['fmt'] = self.gfmt[(i, k)]
                if e['fmt'] in ('rr', 'w4'):
                    self._build4(i, k, e, wcodes)
        torch.cuda.empty_cache()

    def set_rowmask(self, is8):
        """B8: is8 bool [T] (int8 rows; must include the question rows). Fixes n4/n8 for the next forwards (graph-static sizes)."""
        is8 = is8.to(torch.bool)
        c8 = torch.cumsum(is8.to(torch.int32), 0); c4 = torch.cumsum((~is8).to(torch.int32), 0)
        self.pos = torch.where(is8, c8 - 1, -(c4 - 1) - 1).to(torch.int32).contiguous()
        idx = torch.arange(is8.shape[0], device=is8.device, dtype=torch.int32)
        self.rows8 = idx[is8].contiguous(); self.rows4 = idx[~is8].contiguous(); self.n4 = int(self.rows4.shape[0])

    def _upd_mask(self, is8):
        """in-graph update of the partition with the same number of int8 rows (buffers updated in place)"""
        is8 = is8.to(torch.bool)
        c8 = torch.cumsum(is8.to(torch.int32), 0); c4 = torch.cumsum((~is8).to(torch.int32), 0)
        self.pos.copy_(torch.where(is8, c8 - 1, -(c4 - 1) - 1).to(torch.int32))
        idx = torch.arange(is8.shape[0], device=is8.device, dtype=torch.int32)
        o8 = torch.argsort((~is8).to(torch.int32), stable=True).to(torch.int32)       # int8 rows first, in row order
        self.rows8.copy_(o8[:self.rows8.shape[0]]); self.rows4.copy_(o8[self.rows8.shape[0]:])

    def _build4(self, i, k, e, wcodes):
        src = wcodes.get('w4') if wcodes is not None else None
        if src is not None and (i, k) in src:
            q, s = src[(i, k)]; q = q.to(self.dev).float(); s = s.to(self.dev).float()
        else:
            W = self.wfold(i, k)
            s = Q.rtn_scales(W, 7.0, True)
            q = torch.round(W / s[:, None]).clamp(-7, 7)
            del W
        if k == 'Wgu':
            I = q.shape[0] // 2
            q = torch.stack([q[:I], q[I:]], 1).reshape(2 * I, -1); s = torch.stack([s[:I], s[I:]], 1).reshape(2 * I)
        e['c4'] = QK.pack4(q.to(torch.int8)); e['sw4'] = s.float().contiguous()

    # ------------------------------------------------------------------ buffers / GEMM
    def _next_in(self, i, k, T):
        if k == 'final':
            return super()._next_in(i, k, T)
        f = self.gfmt[(i, k)]; ncol = 6144 if k == 'Wd' else 2048
        d = dict(outq=True, s=torch.empty(T, device=self.dev, dtype=torch.float32), hb=None)
        if f == 'rr':
            d.update(q=torch.empty(T, ncol // 2, device=self.dev, dtype=torch.uint8), q8=torch.empty(T, ncol, device=self.dev, dtype=torch.int8),
                     qmax=7., clip=0.9, bits=4, rr=True)
        elif f == 'w4':
            d.update(q=torch.empty(T, ncol // 2, device=self.dev, dtype=torch.uint8), q8=None, qmax=7., clip=0.9, bits=4, rr=False)
        else:
            d.update(q=torch.empty(T, ncol, device=self.dev, dtype=torch.int8), q8=None, qmax=127., clip=1.0, bits=8, rr=False)
        return d

    def _q2(self, cur, e, swiglu):
        T = cur['s'].shape[0]; f = e['fmt']
        q0 = self.q0 if f == 'rr' else (T if f == 'w4' else 0)
        N = e['codes'].shape[0]
        out = torch.empty(T, N // 2 if swiglu else N, device=self.dev, dtype=torch.bfloat16)
        epi = 'swiglu' if swiglu else 'bf16'
        p0 = p1 = None
        if f == 'rr' and self.pos is not None:      # B8: arbitrary partition, rows scattered back by rowmap
            n4 = self.n4
            if n4 > 0: p0 = QK.prob(cur['q'][:n4], e['c4'], 's4', out, cur['s'][:n4], e['sw4'], epi=epi, N=N, rowmap=self.rows4)
            if n4 < T: p1 = QK.prob(cur['q8'][:T - n4], e['codes'], 's8', out, cur['s'][n4:], e['sw8'], epi=epi, N=N, rowmap=self.rows8)
            q0 = n4
        else:
            if q0 > 0:
                p0 = QK.prob(cur['q'][:q0], e['c4'], 's4', out, cur['s'][:q0], e['sw4'], epi=epi, N=N)
            if q0 < T:
                A8 = cur['q8'][:T - q0] if f == 'rr' else cur['q']
                p1 = QK.prob(A8, e['codes'], 's8', out, cur['s'][q0:], e['sw8'], epi=epi, N=N, row0=q0)
        if p0 is not None and p1 is not None: var, pa, pb = 's4', p0, p1
        elif p0 is not None: var, pa, pb = 's4', p0, None
        else: var, pa, pb = 's8', p1, None
        key = (var, q0, T, N, e['codes'].shape[1], swiglu)
        cfg = self.q2cfg.get(key)
        if cfg is None:
            cfg = self._tune(var, pa, pb) if self.q2tune else CFG_STATIC
            self.q2cfg[key] = cfg
        QK.run(var, cfg, pa, pb)
        return out

    def _tune(self, var, pa, pb):
        best = None
        for c in range(10):
            try:
                QK.run(var, c, pa, pb); torch.cuda.synchronize()
                ts = []
                for _ in range(5):
                    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                    a.record(); QK.run(var, c, pa, pb); b.record(); b.synchronize(); ts.append(a.elapsed_time(b))
                t = sorted(ts)[2]
                if best is None or t < best[0]: best = (t, c)
            except RuntimeError:
                continue
        return best[1]

    def _run(self, cur, e):
        if e['prec'] == 'bf16':
            return super()._run(cur, e)
        out = self._q2(cur, e, False)
        return out, self.ones[:out.shape[0]], self.ones, True

    # ------------------------------------------------------------------ forward (H2 QRT._fwd_q with qk2 glue and Q2 GEMMs)
    taps = (); taprows = None; tapped = None

    def forward(self, ids, lay, cache=None, keep_hn=False, x0=None, i0=0, i1=None):
        """as QRT.forward, plus J15-style layer ranges: start from residual buffer x0 at layer i0 (x0 is updated in place), stop after i1"""
        self._ztail = torch.zeros(1, 6144, device=self.dev)
        fr = lay.pos[:, None] * self.ln2.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
        return self._fwd_q(ids, lay, cache, cos, sin, x0, i0, i1)

    def unrot_raw(self, x):
        """residual rows -> unrotated basis (fp32)"""
        return self.R1.inv(x.float())

    def _fwd_q(self, ids, lay, cache, cos, sin, x0=None, i0=0, i1=None):
        T = lay.T; dev = self.dev; eps = self.eps; q0 = self.q0
        i1 = self.nL if i1 is None else i1
        qq = self.n4 if self.pos is not None else q0      # int4-row count (= q0 in the contiguous mode)
        x = F.embedding(ids, self.embed).contiguous() if x0 is None else x0
        nxt = self._next_in(i0, 'Win', T)
        K.addq(x, None, None, None, nxt['q'], nxt['s'], nxt['hb'], eps, dq=False, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'],
               q8=nxt['q8'] if nxt.get('rr') else None, q0=qq, rrb=self.rrb, pos=self.pos if nxt.get('rr') else None)
        cur = nxt
        for i in range(i0, i1):
            d = self.L[i]
            e = self.qw[i]['Win']
            proj, ra, cs, dq = self._run(cur, e)
            nxt = self._next_in(i, 'Wo', T)
            rr = nxt.get('rr', False); q8 = nxt['q8'] if rr else nxt['q']
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, ra, cs, lay, cache, dq)
                R2 = self.r2_for(i)
                K._gnorm_k[(triton.cdiv(T, self.rows),)](o, proj[:, 6144:], ra, cs, d['gn_w'], R2.sign, self.M['H32'], self.M['H64'], self.M['H128'],
                                                         nxt['q'], nxt['s'], nxt['q'], T, proj.stride(0), eps, DQ=True, OUTQ=True, HAD=2 if self.ohead else 1,
                                                         QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows,
                                                         Q8=q8, q0=qq, RR=rr, RRB=self.rrb, POS=self.pos if (rr and self.pos is not None) else q8, RRM=rr and self.pos is not None, num_warps=8)
            else:
                o = self._attn(i, d, proj, ra, cs, lay, cache, dq, cos, sin)
                R2 = self.r2_for(i)
                K._agate_k[(triton.cdiv(T, self.rows),)](o, proj, ra, cs, R2.sign, self.M['H32'], self.M['H64'], self.M['H16'],
                                                         nxt['q'], nxt['s'], nxt['q'], T, proj.stride(0), o.stride(0), o.stride(1), DQ=True, OUTQ=True,
                                                         HAD=2 if self.ohead else 1, QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec,
                                                         ROWS=self.rows, Q8=q8, q0=qq, RR=rr, RRB=self.rrb, POS=self.pos if (rr and self.pos is not None) else q8, RRM=rr and self.pos is not None, num_warps=8)
            cur = nxt
            D, ra, cs, dq = self._run(cur, self.qw[i]['Wo'])
            nxt = self._next_in(i, 'Wgu', T)
            K.addq(x, D, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'],
                   q8=nxt['q8'] if nxt.get('rr') else None, q0=qq, rrb=self.rrb, pos=self.pos if nxt.get('rr') else None)
            cur = nxt
            eg = self.qw[i]['Wgu']; nxt = self._next_in(i, 'Wd', T)
            rr = nxt.get('rr', False); q8 = nxt['q8'] if rr else nxt['q']
            GU = self._q2(cur, eg, True)
            K._swiglu_k[(triton.cdiv(T, self.rows),)](GU, GU, GU, self.R4.sign, self.M['P12T'], self.M['H16'], self.M['H32'], nxt['q'], nxt['s'], nxt['q'], T,
                                                      DQ=False, OUTQ=True, QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows,
                                                      MIN=True, Q8=q8, q0=qq, RR=rr, RRB=self.rrb, POS=self.pos if (rr and self.pos is not None) else q8, RRM=rr and self.pos is not None, num_warps=8)
            cur = nxt
            D, ra, cs, dq = self._run(cur, self.qw[i]['Wd'])
            nxt = self._next_in(i + 1, 'Win', T) if i + 1 < i1 else self._next_in(None, 'final', T)
            K.addq(x, D, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'],
                   q8=nxt['q8'] if nxt.get('rr') else None, q0=qq, rrb=self.rrb, pos=self.pos if nxt.get('rr') else None)
            cur = nxt
            if (i + 1) in self.taps and self.taprows is not None:
                self.tapped[i + 1] = x[self.taprows].clone()
        self.xres = x
        return cur['hb']


class QRT2C(QRT2):
    """row-role with H2's CUTLASS kernels for the state rows (W4A4, fp16 alpha*acc output, consumer dequant exactly as H2 W4A4) and
    q2gemm int8 for the question rows written in the same fp16 units: C16 = fp16(acc8 * g_n), g_n = sw8_n * alpha4 / (sw4_n * 32),
    with the question rows' activation scale stored as 32 * s8_t (exact power of two). gate_up: CUTLASS EVT SwiGLU for state rows,
    q2gemm SwiGLU for question rows with sb = sw8 / 32 (exact). FORMATS.md section 8, 'CUTLASS-backed' variant."""
    def __init__(self, ln2, head=None, fmt='w4q8', wcodes=None, tune=True, **kw):
        super().__init__(ln2, head=head, fmt=fmt, wcodes=wcodes, tune=tune, **kw)
        self.rrb = 32.0
        self.side = torch.cuda.Stream() if os.environ.get('Q2SIDE', '0') == '1' else None
        for i in range(self.nL):
            for k in Q.GEMMS:
                e = self.qw[i][k]
                if 'c4' not in e: continue
                Kd = e['c4'].shape[1] * 2
                e['a4'] = Q.alpha_for('s4', Kd)
                e['csa4'] = (e['sw4'] / e['a4']).float().contiguous()
                e['g'] = (e['sw8'] * e['a4'] / (e['sw4'] * 32.0)).float().contiguous()
                e['sw8q'] = (e['sw8'] / 32.0).float().contiguous()

    def _run(self, cur, e):
        if e['prec'] == 'bf16':
            return super()._run(cur, e)
        if e['fmt'] == 'w8':          # all rows W8A8 on H2's CUTLASS s8 (deployed arithmetic, consumer dequant with csa8)
            return Q.QRT._run(self, cur, e)
        T = cur['s'].shape[0]; N = e['c4'].shape[0]
        q0 = self.q0 if e['fmt'] == 'rr' else T
        C16 = torch.empty(T, N, device=self.dev, dtype=torch.float16)
        side = self.side if (self.side is not None and 0 < q0 < T) else None
        if side is not None:
            side.wait_stream(torch.cuda.current_stream())
        if q0 < T:
            p1 = QK.prob(cur['q8'][:T - q0], e['codes'], 's8', C16, None, e['g'], epi='fp16s', N=N, row0=q0)
            key = ('s8c', q0, T, N, e['codes'].shape[1])
            cfg = self.q2cfg.get(key)
            if cfg is None:
                cfg = self._tune('s8', p1, None) if self.q2tune else CFG_STATIC; self.q2cfg[key] = cfg
            if side is not None:
                with torch.cuda.stream(side): QK.run('s8', cfg, p1, stream=side.cuda_stream)
            else:
                QK.run('s8', cfg, p1)
        if q0 > 0:
            A = cur['q'][:q0]
            Q.QG.gemm('s4', A, e['c4'], e['a4'], self._cfg('s4', A, e['c4'], e['a4']), out=C16[:q0])
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
        return C16, cur['s'], e['csa4'], True

    def _q2(self, cur, e, swiglu):
        if swiglu and e['fmt'] == 'w8':   # H2's CUTLASS EVT int8 SwiGLU
            T = cur['s'].shape[0]
            return Q.QG.swiglu_gemm('s8', cur['q'], e['codes'], cur['s'], e['cs_true'], cfg=(1 if T >= 2500 else 3))
        if not swiglu:
            return super()._q2(cur, e, swiglu)
        T = cur['s'].shape[0]; N = e['c4'].shape[0]
        q0 = self.q0 if e['fmt'] == 'rr' else T
        out = torch.empty(T, N // 2, device=self.dev, dtype=torch.bfloat16)
        side = self.side if (self.side is not None and 0 < q0 < T) else None
        if side is not None:
            side.wait_stream(torch.cuda.current_stream())
        if q0 < T:
            p1 = QK.prob(cur['q8'][:T - q0], e['codes'], 's8', out, cur['s'][q0:], e['sw8q'], epi='swiglu', N=N, row0=q0)
            key = ('s8sw', q0, T, N, e['codes'].shape[1])
            cfg = self.q2cfg.get(key)
            if cfg is None:
                cfg = self._tune('s8', p1, None) if self.q2tune else CFG_STATIC; self.q2cfg[key] = cfg
            if side is not None:
                with torch.cuda.stream(side): QK.run('s8', cfg, p1, stream=side.cuda_stream)
            else:
                QK.run('s8', cfg, p1)
        if q0 > 0:
            Q.QG.swiglu_gemm('s4', cur['q'][:q0], e['c4'], cur['s'][:q0], e['sw4'], cfg=3, out=out[:q0])
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
        return out
