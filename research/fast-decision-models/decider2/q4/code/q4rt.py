"""Q4 B13: exact dead-computation elimination in H2's QRT (rotated low-bit runtime), hobson layout ('single' / 'packed').
  l0=True     layer 0's input projection becomes a per-token table: proj0[t] = TAB[id_t] (the fp16 GEMM output H2's kernels would compute,
              C16 = fp16(alpha * acc)) and ra0[t] = TABRA[id_t] (the per-token activation scale). The layer-0 RMSNorm+quantize and the
              layer-0 Win GEMM are not run. Exact: both depend only on the token id and integer GEMM rows are independent.
  l23='kv'    layer 23 (full attention): state rows (t < q0) compute only their K/V projection (Win rows 4096..5119, N = 1024), RoPE-free
              K norm + RoPE and V copy; their Q, attention, gate, Wo, Wgu, SwiGLU and Wd are skipped (nothing reads them). Question rows run the full layer.
  l23='read'  layer 23: as 'kv', and after attention only the rows the pointer head reads (answer row + option rows of each question) run
              the gate, Wo and the MLP (B14's 'compute the last layer only for the rows that are read').
  inv23       layer 23's bf16 GEMMs (Wo, Wd in b8) run on Q2's row-count-invariant bf16 kernel and the baseline issues layer 23's attention as two
              calls (state queries, question queries), so the baseline and the eliminations can be compared bit for bit (cuBLAS and SDPA pick
              kernels by row count, which changes the summation order).
python q4rt.py check [n]      exactness on real evalkit questions (+ packed multi-question requests)
"""
import os, sys, json, math, torch, triton, triton.language as tl
sys.path[:0] = [os.path.expanduser('~/work/q4'), os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/j15'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'),
                os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch.nn.functional as F
import qrt as Q
from qrt import K, QG
from torch.nn.attention.bias import causal_lower_right


# ------------------------------------------------------------------ attention prep variants (copies of qk._aprep_k restricted to K/V heads or Q heads)
@triton.jit
def _aprep_kv_k(P, RA, CS, KN, COS, SIN, KB, VB, koff, ps, eps, DQ: tl.constexpr, D: tl.constexpr, PCOL: tl.constexpr):
    # P holds only the K/V columns (PCOL = 0) or the full projection (PCOL = 4096); CS indexed in the full-projection column space
    r = tl.program_id(0).to(tl.int64); s = tl.program_id(1) + 8
    c = tl.arange(0, D)
    pc = tl.where(c < 32, c + 32, tl.where(c < 64, c - 32, c))
    if DQ:
        ra = tl.load(RA + r)
    if s < 10:
        col = 4096 + (s - 8) * 256
        w = tl.load(KN + c); wp = tl.load(KN + pc)
        out = KB + (koff + r) * 2 * D + (s - 8) * D
        x = tl.load(P + r * ps + (col - 4096 + PCOL) + c).to(tl.float32)
        xp = tl.load(P + r * ps + (col - 4096 + PCOL) + pc).to(tl.float32)
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
        tl.store(out + c, y.to(tl.bfloat16))
    else:
        col = 4608 + (s - 10) * 256
        x = tl.load(P + r * ps + (col - 4096 + PCOL) + c).to(tl.float32)
        if DQ:
            x = (x * ra) * tl.load(CS + col + c)
        tl.store(VB + (koff + r) * 2 * D + (s - 10) * D + c, x.to(tl.bfloat16))


@triton.jit
def _aprep_q_k(P, RA, CS, QN, COS, SIN, Qo, ps, eps, DQ: tl.constexpr, D: tl.constexpr):
    r = tl.program_id(0).to(tl.int64); s = tl.program_id(1)
    c = tl.arange(0, D)
    pc = tl.where(c < 32, c + 32, tl.where(c < 64, c - 32, c))
    col = s * 512
    w = tl.load(QN + c); wp = tl.load(QN + pc)
    x = tl.load(P + r * ps + col + c).to(tl.float32)
    xp = tl.load(P + r * ps + col + pc).to(tl.float32)
    if DQ:
        ra = tl.load(RA + r)
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
    tl.store(Qo + r * 8 * D + s * D + c, y.to(tl.bfloat16))


class QRT4(Q.QRT):
    l0 = False
    l23 = 'none'
    q0 = None            # number of state rows (rows < q0 are never read)
    read_rows = None     # LongTensor of rows the head reads (l23 = 'read')
    rmask = None         # bool [nR, T] attention mask of the read rows (built by set_read)

    # ---------------- setup
    def build_l0(self, chunk=8192):
        e = self.qw[0]['Win']; assert e['prec'] != 'bf16', 'layer-0 table implemented for the quantized Win'
        V = self.embed.shape[0]
        self.tab = torch.empty(V, e['codes'].shape[0], device=self.dev, dtype=torch.float16)
        self.tabra = torch.empty(V, device=self.dev, dtype=torch.float32)
        for a in range(0, V, chunk):
            ids = torch.arange(a, min(V, a + chunk), device=self.dev)
            x = F.embedding(ids, self.embed).contiguous()
            nxt = self._next_in(0, 'Win', ids.shape[0])
            K.addq(x, None, None, None, nxt['q'], nxt['s'], nxt['hb'], self.eps, dq=False, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            self.tab[a:a + ids.shape[0]] = self.qgemm(nxt['q'], e)
            self.tabra[a:a + ids.shape[0]] = nxt['s']
        torch.cuda.synchronize()
        return self.tab.numel() * 2 + self.tabra.numel() * 4

    # ---------------- B11: 2:4 sparse int8 weights for STATE rows (rows < q0), dense weights for question rows (two weight copies)
    sp_on = False

    def build_sparse(self, layers, gemms=('Win', 'Wo', 'Wgu', 'Wd'), masks=None):
        """compress the int8 codes of the chosen GEMMs to 2:4 (magnitude mask unless masks[(i, k)] is given: speed does not depend on
        which weights are kept; accuracy of a chosen mask is Q5's measurement)"""
        import q4sp as SP, test_sp as TS
        TS.set_layout()
        n = 0
        for i in layers:
            for k in gemms:
                e = self.qw[i][k]
                if e['prec'] != 'w8a8': continue
                q = e['codes']
                if e.get('evt'):     # H2 interleaves gate/up rows pairwise; the sparse SwiGLU epilogue wants blocks of 8 (gate 8 | up 8)
                    N = q.shape[0]; I = N // 2
                    idx = torch.arange(N, device=self.dev).reshape(I // 8, 8, 2).permute(0, 2, 1).reshape(-1)
                    qs = q[idx].contiguous(); cs = e['cs_true'][idx].contiguous()
                else:
                    qs = q; cs = None
                m = masks[(i, k)] if masks and (i, k) in masks else SP.mask24_mag(qs, 'sp8')
                Wc, E, _ = SP.compress(qs, m, 'sp8')
                e['sp'] = e['sp_built'] = dict(Wc=Wc, E=E, cs=cs, N=qs.shape[0], K=qs.shape[1])
                n += 1
        torch.cuda.empty_cache()
        self.sp_on = True
        return n

    def set_sparse_layers(self, layers):
        """activate the built sparse weights only for these layers (state rows); () = off"""
        L = set(layers)
        for i in range(self.nL):
            for k in ('Win', 'Wo', 'Wgu', 'Wd'):
                e = self.qw[i][k]
                if 'sp_built' in e: e['sp'] = e['sp_built'] if i in L else None
        self.sp_on = bool(L)

    _spcfg = {}

    def _sp_run(self, sp, A, out, sa, sb, epi, alpha):
        import q4sp as SP
        key = (sp['N'], sp['K'], A.shape[0], epi)
        if key not in self._spcfg:
            best = None
            for c in range(16):
                try:
                    SP.gemm('sp8', c, sp['Wc'], sp['E'], A, out, sp['N'], A.shape[0], sp['K'], sa=sa, sb=sb, epi=epi, alpha=alpha); torch.cuda.synchronize()
                    ts = []
                    for _ in range(5):
                        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                        e0.record(); SP.gemm('sp8', c, sp['Wc'], sp['E'], A, out, sp['N'], A.shape[0], sp['K'], sa=sa, sb=sb, epi=epi, alpha=alpha); e1.record(); e1.synchronize()
                        ts.append(e0.elapsed_time(e1))
                    t = sorted(ts)[2]
                except Exception:
                    continue
                if best is None or t < best[0]: best = (t, c)
            self._spcfg[key] = best[1]
        SP.gemm('sp8', self._spcfg[key], sp['Wc'], sp['E'], A, out, sp['N'], A.shape[0], sp['K'], sa=sa, sb=sb, epi=epi, alpha=alpha)

    sp_all_rows = False      # every row through the 2:4 weights (one launch, one weight copy): the short-decision variant

    def _q0_rows(self, T):
        if self.sp_all_rows: return T
        q0 = self.q0 if self.q0 is not None else 0
        return q0 if 0 < q0 < T else 0

    def qgemm(self, A, e):
        sp = e.get('sp')
        q0 = self._q0_rows(A.shape[0]) if (self.sp_on and sp is not None) else 0
        if not q0:
            return super().qgemm(A, e)
        out = torch.empty(A.shape[0], e['codes'].shape[0], device=self.dev, dtype=torch.float16)
        self._sp_run(sp, A[:q0], out[:q0], None, None, 'fp16a', e['alpha'])
        if q0 < A.shape[0]:
            Aq = A[q0:]
            QG.gemm(e['kind'], Aq, e['codes'], e['alpha'], self._cfg(e['kind'], Aq, e['codes'], e['alpha']), out=out[q0:])
        return out

    def _gateup(self, eg, cur, T):
        sp = eg.get('sp')
        q0 = self._q0_rows(T) if (self.sp_on and sp is not None) else 0
        cfg = (1 if (eg['kind'] == 's8' and T >= 2500) else 3)
        if not q0:
            return QG.swiglu_gemm(eg['kind'], cur['q'], eg['codes'], cur['s'], eg['cs_true'], cfg=cfg)
        GU = torch.empty(T, eg['codes'].shape[0] // 2, device=self.dev, dtype=torch.bfloat16)
        self._sp_run(sp, cur['q'][:q0], GU[:q0], cur['s'][:q0], sp['cs'], 'swiglu', 1.0)
        if q0 < T:
            QG.swiglu_gemm(eg['kind'], cur['q'][q0:], eg['codes'], cur['s'][q0:], eg['cs_true'], cfg=(1 if (eg['kind'] == 's8' and T - q0 >= 2500) else 3), out=GU[q0:])
        return GU

    def prep_l23(self):
        i = self.nL - 1; e = self.qw[i]['Win']
        assert self.L[i]['type'] != 'linear_attention' and e['prec'] != 'bf16'
        e['codes_kv'] = e['codes'][4096:5120].contiguous()

    def set_read(self, lay, rows):
        """read rows (absolute); the mask is kept for reference only (attention runs for every question row)"""
        rows = torch.as_tensor(sorted(set(int(r) for r in rows)), device=self.dev, dtype=torch.long)
        T = lay.T; kpos = torch.arange(T, device=self.dev)
        if lay.kind == 'single':
            m = kpos[None, :] <= rows[:, None]
        else:
            Ls = lay.Ls; seg = torch.full((T,), -1, device=self.dev, dtype=torch.long)
            for j, (s, e) in enumerate(lay.seg): seg[s:e] = j
            m = (kpos[None, :] < Ls) | ((seg[None, :] == seg[rows][:, None]) & (kpos[None, :] <= rows[:, None]))
            m &= ~((rows[:, None] < Ls) & (kpos[None, :] >= Ls))
        self.read_rows = rows; self.rmask = m[None, None]          # [1, 1, nR, T]

    # ---------------- forward
    def _fwd_q(self, ids, lay, cache, cos, sin):
        T = lay.T; dev = self.dev; eps = self.eps
        x = F.embedding(ids, self.embed).contiguous()
        if self.l0:
            proj0 = self.tab.index_select(0, ids); ra0 = self.tabra.index_select(0, ids)
            cur = None
        else:
            nxt = self._next_in(0, 'Win', T)
            K.addq(x, None, None, None, nxt['q'], nxt['s'], nxt['hb'], eps, dq=False, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
            cur = nxt
        for i, d in enumerate(self.L):
            if i == self.nL - 1 and self.l23 != 'none':
                return self._last(i, d, cur, x, lay, cache, cos, sin)
            e = self.qw[i]['Win']
            if i == 0 and self.l0:
                proj, ra, cs, dq = proj0, ra0, e['csa'], True
            else:
                proj, ra, cs, dq = self._run(cur, e)
            nxt = self._next_in(i, 'Wo', T)
            R2 = self.r2_for(i)
            if d['type'] == 'linear_attention':
                o = self._gdn(i, d, proj, ra, cs, lay, cache, dq)
                K._gnorm_k[(triton.cdiv(T, self.rows),)](o, proj[:, 6144:], ra if dq else proj, cs if dq else proj, d['gn_w'], R2.sign, self.M['H32'], self.M['H64'], self.M['H128'],
                                                 nxt['q'], nxt['s'], nxt['hb'], T, proj.stride(0), eps, DQ=dq, OUTQ=nxt['outq'], HAD=2 if self.ohead else 1,
                                                 QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
            else:
                o = self._attn(i, d, proj, ra, cs, lay, cache, dq, cos, sin)
                K._agate_k[(triton.cdiv(T, self.rows),)](o, proj, ra if dq else proj, cs if dq else proj, R2.sign, self.M['H32'], self.M['H64'], self.M['H16'],
                                                 nxt['q'], nxt['s'], nxt['hb'], T, proj.stride(0), o.stride(0), o.stride(1), DQ=dq, OUTQ=nxt['outq'], HAD=2 if self.ohead else 1,
                                                 QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
            cur = nxt
            x, cur = self._mlp_block(i, cur, x, T, last=(i + 1 == self.nL))
        return cur['hb']

    def _mlp_block(self, i, cur, x, T, last, hb_out=None):
        """Wo -> residual add + norm + quant -> Wgu (+SwiGLU) -> R4 + quant -> Wd -> residual add + norm (+ quant of the next Win input)"""
        eps = self.eps
        D, ra, cs, dq = self._run(cur, self.qw[i]['Wo'])
        nxt = self._next_in(i, 'Wgu', T)
        K.addq(x, D, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
        cur = nxt
        eg = self.qw[i]['Wgu']; nxt = self._next_in(i, 'Wd', T)
        if eg.get('evt'):
            GU = self._gateup(eg, cur, T); dq = False; ra = cs = None; mi = True
        else:
            GU, ra, cs, dq = self._run(cur, eg); mi = False
        K._swiglu_k[(triton.cdiv(T, self.rows),)](GU, ra if dq else GU, cs if dq else GU, self.R4.sign, self.M['P12T'], self.M['H16'], self.M['H32'], nxt['q'], nxt['s'], nxt['hb'], T,
                          DQ=dq, OUTQ=nxt['outq'], QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, MIN=mi, num_warps=8)
        cur = nxt
        D, ra, cs, dq = self._run(cur, self.qw[i]['Wd'])
        nxt = self._next_in(i + 1, 'Win', T) if not last else self._next_in(None, 'final', T)
        if last and hb_out is not None: nxt['hb'] = hb_out
        K.addq(x, D, ra, cs, nxt['q'], nxt['s'], nxt['hb'], eps, dq=dq, outq=nxt['outq'], qmax=nxt['qmax'], clip=nxt['clip'], bits=nxt['bits'])
        return x, nxt

    def _run(self, cur, e):
        if e['prec'] == 'bf16' and e.get('inv'):
            # row-count-invariant bf16 GEMM (Q2's q2gemm: fixed K order, no split-K), used at layer 23 when inv23 is set
            import q2k
            A = cur['hb']; out = torch.empty(A.shape[0], e['Wb'].shape[0], device=self.dev, dtype=torch.bfloat16)
            q2k.run('bf16', 4, q2k.prob(A, e['Wb'], 'bf16', out, epi='bf16'))
            return out, None, None, False
        return super()._run(cur, e)

    def set_inv23(self, on=True):
        i = self.nL - 1
        for k in ('Win', 'Wo', 'Wgu', 'Wd'): self.qw[i][k]['inv'] = on
        self.inv23 = on

    def _attn(self, i, d, proj, ra, cs, lay, cache, dq, cos, sin):
        if not (getattr(self, 'inv23', False) and i == self.nL - 1 and lay.kind == 'single'):
            return super()._attn(i, d, proj, ra, cs, lay, cache, dq, cos, sin)
        # baseline layer 23 with the attention issued as the elimination issues it: state queries and question queries in separate calls
        T = lay.T; q0 = self.q0; Tq = T - q0
        kb = torch.empty(T, 2, 256, device=self.dev, dtype=torch.bfloat16); vb = torch.empty_like(kb)
        q = K.aprep(proj, ra, cs, d['qn'], d['kn'], cos, sin, kb, vb, 0, self.eps, dq=dq)
        qh = q.transpose(0, 1)[None]
        o1 = F.scaled_dot_product_attention(qh[:, :, :q0], kb[:q0].transpose(0, 1)[None], vb[:q0].transpose(0, 1)[None], is_causal=True, enable_gqa=True)[0]
        qq = q[q0:].contiguous().transpose(0, 1)[None]
        o2 = F.scaled_dot_product_attention(qq, kb.transpose(0, 1)[None], vb.transpose(0, 1)[None], attn_mask=causal_lower_right(Tq, T), enable_gqa=True)[0]
        return torch.cat([o1, o2], 1)

    def _last(self, i, d, cur, x, lay, cache, cos, sin):
        T = lay.T; dev = self.dev; eps = self.eps
        e = self.qw[i]['Win']; csa = e['csa']
        kb = torch.empty(T, 2, 256, device=dev, dtype=torch.bfloat16); vb = torch.empty_like(kb)
        hb = torch.empty(T, 2048, device=dev, dtype=torch.bfloat16)
        R2 = self.r2_for(i)
        q0 = self.q0 if lay.kind == 'single' else lay.Ls
        Tq = T - q0
        # state rows: K/V projection only (Win rows 4096..5119)
        pkv = QG.gemm('s8', cur['q'][:q0], e['codes_kv'], e['alpha'], self._cfg('s8', cur['q'][:q0], e['codes_kv'], e['alpha']))
        _aprep_kv_k[(q0, 4)](pkv, cur['s'], csa, d['kn'], cos, sin, kb, vb, 0, pkv.stride(0), eps, DQ=True, D=256, PCOL=0, num_warps=4)
        # question rows: the full layer input projection, K/V into the same buffers, queries
        aq = cur['q'][q0:]
        pq = self.qgemm(aq, e)
        raq = cur['s'][q0:]
        qv = K.aprep(pq, raq, csa, d['qn'], d['kn'], cos[q0:], sin[q0:], kb, vb, q0, eps, dq=True)
        qh = qv.transpose(0, 1)[None]
        if lay.kind == 'single':
            o = F.scaled_dot_product_attention(qh, kb.transpose(0, 1)[None], vb.transpose(0, 1)[None], attn_mask=causal_lower_right(Tq, T), enable_gqa=True)[0]
        else:
            Ls = lay.Ls; ks = kb[:Ls]; vs = vb[:Ls]; outs = []
            for (s, e2) in lay.seg:
                kk = torch.cat([ks, kb[s:e2]], 0); vv = torch.cat([vs, vb[s:e2]], 0)
                outs.append(F.scaled_dot_product_attention(qh[:, :, s - Ls:e2 - Ls], kk.transpose(0, 1)[None], vv.transpose(0, 1)[None],
                                                           attn_mask=causal_lower_right(e2 - s, Ls + e2 - s), enable_gqa=True)[0])
            o = torch.cat(outs, 1)
        if self.l23 == 'read':     # only the rows the head reads continue past attention
            Rr = self.read_rows - q0
            o = o.index_select(1, Rr).contiguous(); pq = pq.index_select(0, Rr); raq = raq.index_select(0, Rr)
            n = Rr.shape[0]; xs = x.index_select(0, self.read_rows); hbs = torch.empty(n, 2048, device=dev, dtype=torch.bfloat16)
        else:
            n = Tq; xs = x[q0:]; hbs = hb[q0:]
        nxt = self._next_in(i, 'Wo', n)
        K._agate_k[(triton.cdiv(n, self.rows),)](o, pq, raq, csa, R2.sign, self.M['H32'], self.M['H64'], self.M['H16'],
                                          nxt['q'], nxt['s'], nxt['hb'], n, pq.stride(0), o.stride(0), o.stride(1), DQ=True, OUTQ=nxt['outq'], HAD=2 if self.ohead else 1,
                                          QMAX=nxt['qmax'], CLIP=nxt['clip'], BITS=nxt['bits'], PREC=self.hprec, ROWS=self.rows, num_warps=8)
        self._mlp_block(i, nxt, xs, n, last=True, hb_out=hbs)
        if self.l23 == 'read': hb.index_copy_(0, self.read_rows, hbs)
        return hb


def build(codes='w8=~/work/h2/codes_gptq_w8.pt', prec='map:~/work/h1/precmap_w8a8_b8.json:w8a8'):
    from kitrun import load_P
    from lean2 import Lean2
    from j15run import load_codes
    P = load_P()
    ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval()
    m = QRT4(ln, head=head, prec=prec, wcodes=load_codes(codes)); m.tune = False
    import gc
    for d in ln.layers:
        for k_ in ('Wgu_il', 'Win_f', 'Wgu_f'): d.pop(k_, None)
    gc.collect(); torch.cuda.empty_cache()
    m.prep_l23()
    return P, m, head


def check(n=300):
    """exactness on real evalkit questions (hobson single layout) and on packed multi-question requests (H2 bundles, random states):
    compare the head's probabilities and the read rows' final hidden between baselines and each elimination, bit for bit.
    h2: H2's runtime as deployed (cuBLAS bf16 GEMMs, one SDPA call); inv: layer 23 on row-count-invariant kernels (see module doc)."""
    import evalkit as EK, random
    from kitrun import prep_question
    P, m, head = build()
    nb = m.build_l0(); print('layer-0 table', round(nb / 1e9, 3), 'GB', flush=True)
    items = list(EK.all_question_items(None))
    random.Random(0).shuffle(items)
    items = items[:n]
    byid = {}
    for s in ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
        for it in EK.load_suite(s): byid[it['id']] = it
    modes = [('l0', True, 'none'), ('kv', False, 'kv'), ('read', False, 'read'), ('l0+read', True, 'read')]
    cmp = [('inv_vs_h2', None)] + [(f'{k}_{b}', k) for k, _, _ in modes for b in ('vs_h2', 'inv_vs_inv')]
    stat = {k: dict(n=0, hid_identical=0, prob_identical=0, maxdp=0.0, flips=0) for k, _ in cmp}
    def upd(k, ha, pa, hb_, pb):
        s = stat[k]; s['n'] += 1
        s['hid_identical'] += int(torch.equal(ha, hb_)); s['prob_identical'] += int(torch.equal(pa, pb))
        s['maxdp'] = max(s['maxdp'], (pa - pb).abs().max().item()); s['flips'] += int((pa.argmax(-1) != pb.argmax(-1)).any().item())
    MD = {k: (l0, l23) for k, l0, l23 in modes}
    def cfg(l0, l23, inv):
        m.l0, m.l23 = l0, l23; m.set_inv23(inv)
    with torch.inference_mode():
        for suite, iid, qn, st, spec in items:
            pr = prep_question(P, byid[iid], qn)
            ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
            rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
            lay = Q.Lay('single', T); m.q0 = pr['q0']; m.set_read(lay, rows.tolist())
            def run():
                hn = m.forward(ids, lay)
                h = m.unrot(hn[rows]).float()
                lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
                return hn[rows].clone(), torch.softmax(lg, -1)
            cfg(False, 'none', False); h0, p0 = run()
            cfg(False, 'none', True); hI, pI = run()
            upd('inv_vs_h2', h0, p0, hI, pI)
            for k, (l0, l23) in MD.items():
                cfg(l0, l23, False); h1, p1 = run(); upd(f'{k}_vs_h2', h0, p0, h1, p1)
                cfg(l0, l23, True); h2_, p2 = run(); upd(f'{k}_inv_vs_inv', hI, pI, h2_, p2)
        json.dump(stat, open(os.path.expanduser('~/work/q4/res_b13_exact.json'), 'w'), indent=1)
        # packed: state + 4 / 15 question branches (hobson's shared-state layout), random states
        bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt'))
        import bench as HB
        for bn in ('4q', '15q'):
            for Ts in (64, 1000):
                for rep in range(3):
                    req = HB.Req(m, Ts, bundles[bn], 'plain'); m.q0 = Ts
                    rows = req.pool_rows.tolist() + req.opt_rows.reshape(-1).tolist(); m.set_read(req.lay, rows)
                    def run2():
                        hn = m.forward(req.ids, req.lay)
                        return hn[torch.as_tensor(sorted(set(rows)), device='cuda')].clone(), req.run()
                    cfg(False, 'none', False); h0, p0 = run2()
                    for k, (l0, l23) in MD.items():
                        cfg(l0, l23, False); h1, p1 = run2(); upd(f'{k}_vs_h2', h0, p0, h1, p1)
    cfg(False, 'none', False)
    print(json.dumps(stat, indent=1), flush=True)
    json.dump(stat, open(os.path.expanduser('~/work/q4/res_b13_exact.json'), 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'check': check(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
