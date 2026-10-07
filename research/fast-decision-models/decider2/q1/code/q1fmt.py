"""Q1 format emulation: per-GEMM quantized GEMMs for one configuration, plugged into q1lib.Q1 via g.qfn.

Base arithmetic = H1 FORMAT v1 (h1lib.H1.lin): rotated inputs (R1 offline residual rotation for Win/Wgu, online R2/R4 for Wo/Wd), output-side R1 for
Wo/Wd, per-token symmetric absmax activations (int4 clip 0.9, int8 clip 1.0), per-channel symmetric weights (GPTQ act-order), integer GEMM
(torch._int_mm s8 x s8 -> s32, 4-bit codes in int8), epilogue acc * s_tok * s_chan in fp32 -> bf16.

spec (dict):
  a_s, a_q   : activation bits for state rows [0, q0) and question rows [q0, T): 4 | 8 | 16 (16 = bf16 GEMM, dense weights)
  wq         : weights for 4-bit rows: 'gptq' | 'rtn';  qw8: weights used by 8-bit rows: 'w8' (separate GPTQ8 copy) | 'w4' (W4A8, one weight copy)
  map        : {'i.k': 'w8a8' | 'bf16' | 'w4a4'}: per-GEMM override for ALL rows
  clip4      : int4 activation clip ratio (0.9)
  group      : 0 | 64: group-scaled int4 (activations per token x group of K, weights per channel x group; GPTQ with group scales)
  corr       : {'r': 64, 'src': 'all'|'s'|'q', 'w': 0|1, 'wr': 64}: B1 exact low-rank correction of the int4 activation rounding error in the
               top-r subspace of G_in (input side): y += ((X - deq Q X) U)(U^T W^T), bf16 operands, fp32 accumulation;
               w=1 also corrects the weight rounding error in the top-wr output subspace P of G_out: y += (X (W - deq Q W)^T P) P^T
  split      : {'r': 128, 'basis': 'pca'|'sens'|'mix', 'hi': 8}: inner-dimension split (ResQ-style): x_hi = x U (int8 per token, weights W U int8),
               x_lo = x - x_hi U^T rotated + int4 (GPTQ on the projected Hessian). basis pca = top-r of E[x^T x] (ResQ); sens = top-r of G_in (B1)
  dither     : None | 'sr' | 'tpdf': stochastic rounding / non-subtractive triangular dither on 4-bit STATE rows (B2), seeded per request
  proto      : {'Kc': 256}: B3 prototype subtraction (k-means codebook per GEMM input, nearest prototype, residual int4, + exact c W table)
  ba16       : GDN b/a gate rows of Win (32 rows) in bf16
"""
import os, math, torch
import h1lib as HL
import q1lib as QL

SPEC_DIR = os.path.expanduser('~/work/q1/spec')
PROTO_DIR = os.path.expanduser('~/work/q1/proto')
CODE_DIR = os.path.expanduser('~/work/q1/codes')
B5_DIR = os.path.expanduser('~/work/q1/b5')
HDD_DIR = os.path.expanduser('~/work/q1/hdd')
KIDX = {'Win': 0, 'Wo': 1, 'Wgu': 2, 'Wd': 3}


def gptq_group(W, H, qmax, gs=64, blocksize=128, percdamp=0.01, actorder=False):
    """GPTQ with per-(channel, group) symmetric scales (scale of a group set from the error-updated weights when the group starts; MSE clip search).
    No act-order (groups must stay contiguous for a group-scaled kernel). W [N,K] fp32, H [K,K]. Returns codes [N,K], scales [N, K/gs]."""
    W = W.clone().float(); N, K = W.shape; H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(H)); H[range(K), range(K)] += damp
    L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W); S = torch.zeros(N, K // gs, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); n = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(n):
            col = i1 + j
            if col % gs == 0:                       # blocksize % gs == 0, so the group lies inside this block
                S[:, col // gs] = HL.rtn_scales(W1[:, j:j + gs], qmax, True)
            s = S[:, col // gs]
            w = W1[:, j]; d = Hi[j, j]
            q = torch.round(w / s).clamp(-qmax, qmax)
            Q1[:, j] = q
            e = (w - q * s) / d
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, S


class Fmt:
    def __init__(self, g, spec):
        self.g = g; self.spec = dict(a_s=4, a_q=4, wq='gptq', qw8='w8', map={}, clip4=0.9, group=0, corr=None, split=None, dither=None, proto=None, ba16=False)
        self.spec.update(spec)
        self.c = {}          # per-GEMM caches
        self.seed = 0

    # ---------------------------------------------------------------- helpers
    def prec_rows(self, i, k):
        m = self.spec['map'].get(f'{i}.{k}')
        if m == 'bf16': return 16, 16
        if m == 'w8a8': return 8, 8
        if m == 'w4a4': return 4, 4
        return self.spec['a_s'], self.spec['a_q']

    def wout(self, i, k):
        """folded weight [N,K] (gain folded, NO input rotation), output side in the stored (R1-rotated) basis for Wo/Wd"""
        g = self.g; d = g.L[i]; W = d[k].float()
        if k == 'Win': W = W * d['in1'][None, :]
        if k == 'Wgu': W = W * d['post1'][None, :]
        if self.spec['ba16'] and k == 'Win' and d['type'] == 'linear_attention': W = W[:8192]
        if k in ('Wo', 'Wd') and g.opt['rout']: W = g.rots()['R1'](W.t().contiguous()).t().contiguous()
        return W

    def wcodes(self, i, k, bits):
        g = self.g; key = ('w', i, k, bits)
        if key not in self.c:
            fn = f"{CODE_DIR}/w{bits}_{self.spec['wq']}_ba{int(self.spec['ba16'])}_{i}_{k}.pt"
            if os.path.exists(fn):
                q, s = torch.load(fn, map_location=g.dev)
            else:
                g.opt['ba16'] = self.spec['ba16']
                wq = self.spec['wq']; hd0 = g.opt['hdir']
                if wq == 'gptqd':                          # GPTQ on the decision-weighted Hessian (q1hd.py), same scales/act-order
                    wq = 'gptq'; g.opt['hdir'] = HDD_DIR
                if bits == 4: g.opt['wq'] = wq
                else: g.opt['wq8'] = 'gptq' if wq == 'gptq' else 'rtn'; g.opt['wq'] = wq
                q, s = g.qweight(i, k, 'w4a4' if bits == 4 else 'w8a8')
                g.Qw = {}; g.opt['hdir'] = hd0
                os.makedirs(CODE_DIR, exist_ok=True); torch.save((q.cpu(), s.cpu()), fn)
            self.c[key] = (q, s)
        return self.c[key]

    def wrot(self, i, k):
        key = ('wrot', i, k)
        if key not in self.c:
            W = self.g.wfold(i, k)
            if self.spec['ba16'] and k == 'Win' and self.g.L[i]['type'] == 'linear_attention': W = W[:8192]
            self.c[key] = W.to(torch.bfloat16).contiguous()
        return self.c[key]

    def qact(self, xr, bits, dither=None, gen=None):
        qmax = HL.QMAX[bits]
        am = xr.abs().amax(-1).clamp_min(1e-8)       # Q2 FORMATS sec 0: s = fp32(fp32(amax / qmax) * clip), IEEE division by a tensor
        s = (am / torch.full_like(am, qmax)) * (self.spec['clip4'] if bits == 4 else 1.0)
        z = xr / s[:, None]
        if dither == 'sr':
            u = torch.rand(z.shape, device=z.device, generator=gen)
            q = torch.floor(z + u)
        elif dither == 'tpdf':
            u = torch.rand(z.shape, device=z.device, generator=gen) - torch.rand(z.shape, device=z.device, generator=gen)
            q = torch.round(z + u)
        else:
            q = torch.round(z)
        return q.clamp(-qmax, qmax).to(torch.int8), s

    def imm(self, qa, qw):
        return self.g.imm(qa, qw)

    def gen(self, i, k):
        G = torch.Generator(device=self.g.dev); G.manual_seed(int(self.seed) * 1009 + i * 4 + KIDX[k]); return G

    def U(self, i, k, side, src, r):
        key = ('U', i, k, side, src, r)
        if key not in self.c:
            sv = torch.load(f'{SPEC_DIR}/L{i}_{k}.pt', map_location=self.g.dev)
            self.c[key] = sv[(side, src)]['V'][:, :r].float().contiguous()
        return self.c[key]

    # ---------------------------------------------------------------- the GEMM
    def __call__(self, g, i, k, x, xn):
        d = g.L[i]; src = (xn if k in ('Win', 'Wgu') else x).float()
        T = x.shape[0]; q0 = g._q0 if g._q0 is not None else T
        q0 = min(max(q0, 0), T)
        bs, bq = self.prec_rows(i, k)
        Wd = d[k]
        ba = self.spec['ba16'] and k == 'Win' and d['type'] == 'linear_attention'
        Nq = 8192 if ba else Wd.shape[0]
        y = torch.empty(T, Wd.shape[0], device=x.device, dtype=x.dtype)
        segs = [(0, q0, bs, 's'), (q0, T, bq, 'q')]
        rm = getattr(g, '_row8', None)
        if rm is not None and bs == 4 and q0 > 0:      # B8 arbitrary row partition: selected state rows at int8 (W8A8)
            sel = torch.nonzero(rm[:q0]).flatten(); rest = torch.nonzero(~rm[:q0]).flatten()
            for idx, bits, role in ((sel, 8, 's8'), (rest, 4, 's')):
                if idx.numel() == 0: continue
                y[idx, :Nq] = self.qgemm(i, k, src[idx], bits, role).to(x.dtype)
                if ba: y[idx, Nq:] = x[idx] @ Wd[Nq:].t()
            segs = [(q0, T, bq, 'q')]
        hr = self.spec.get('hi_rows')
        if hr and bs == 4 and q0 > 0:                  # static row rule: first f and last l state rows at int8
            f_ = min(hr.get('first', 0), q0); l_ = min(hr.get('last', 0), q0 - f_)
            segs = [(0, f_, 8, 's8'), (f_, q0 - l_, 4, 's'), (q0 - l_, q0, 8, 's8'), (q0, T, bq, 'q')]
        for lo, hi, bits, role in segs:
            if hi <= lo: continue
            if bits == 16:
                y[lo:hi] = x[lo:hi] @ Wd.t(); continue
            y[lo:hi, :Nq] = self.qgemm(i, k, src[lo:hi], bits, role).to(x.dtype)
            if ba: y[lo:hi, Nq:] = x[lo:hi] @ Wd[Nq:].t()
        return y

    def qgemm(self, i, k, s_, bits, role):
        """s_: fp32 GEMM input rows (unrotated). Returns fp32 output rows in the unrotated output basis."""
        g = self.g; sp = self.spec
        R = g.rot_for(i, k)
        if bits == 8:
            wb = 4 if sp['qw8'] == 'w4' else 8
            qw, sw = self.wcodes(i, k, wb)
            xr = R(s_)
            qa, sa = self.qact(xr, 8)
            y = self.imm(qa, qw) * sa[:, None] * sw[None, :]
            return self.unrot_out(k, y)
        # ---- 4-bit rows
        if sp['split']: return self.qgemm_split(i, k, s_, role)
        if sp['proto']: return self.qgemm_proto(i, k, s_, role)
        if sp.get('tc'): return self.qgemm_tc(i, k, s_, role)
        xr = R(s_)
        dith = sp['dither'] if role == 's' else None
        if sp.get('wonly') or sp.get('aonly'):           # diagnostics: weight-only / activation-only 4-bit rounding (fp32 GEMM)
            if sp.get('wonly'):
                qw, sw = self.wcodes(i, k, 4); y = xr @ (qw.float() * sw[:, None]).t()
            else:
                qa, sa = self.qact(xr, 4, dith, self.gen(i, k) if dith else None); y = (qa.float() * sa[:, None]) @ self.wrot(i, k).float().t()
            return self.unrot_out(k, y)
        if sp.get('b9') and role == 's':                 # B9: keep a random fraction f of K tiles per output tile, rescale 1/f
            f = sp['b9']['f']; tk = sp['b9'].get('tile', 128); tn = sp['b9'].get('ntile', 128)
            qw, sw = self.wcodes(i, k, 4)
            qa, sa = self.qact(xr, 4)
            N, K = qw.shape; nkt = K // tk; nnt = (N + tn - 1) // tn
            G_ = self.gen(i, k)
            keep = torch.rand(nnt, nkt, device=xr.device, generator=G_).argsort(1) < int(round(f * nkt))   # exactly f of the K tiles per output tile
            mcol = keep.repeat_interleave(tn, 0)[:N].float() * (nkt / int(round(f * nkt)))                  # [N, nkt]
            y = torch.zeros(xr.shape[0], N, device=xr.device)
            for kt in range(nkt):
                y += self.imm(qa[:, kt * tk:(kt + 1) * tk].contiguous(), qw[:, kt * tk:(kt + 1) * tk].contiguous()) * mcol[None, :, kt]
            y = y * sa[:, None] * sw[None, :]
            return self.unrot_out(k, y)
        if sp['group']:
            y, qa_deq = self.gemm_group(i, k, xr, dith)
        else:
            qw, sw = self.wcodes(i, k, 4)
            qa, sa = self.qact(xr, 4, dith, self.gen(i, k) if dith else None)
            if sp.get('ns') and role == 's':           # noise shaping along rows (q1ns): same scales, error fed forward to the next row
                import q1ns
                qa = q1ns.ns_quant(xr, sa, 7.0, sp['ns'].get('c', 1.0), sp['ns'].get('chunk', 64))
            y = self.imm(qa, qw) * sa[:, None] * sw[None, :]
            qa_deq = None
            if sp['corr']: qa_deq = qa.float() * sa[:, None]
        if sp['corr']:
            y = y + self.correction(i, k, xr, qa_deq)
        return self.unrot_out(k, y)

    def unrot_out(self, k, y):
        if k in ('Wo', 'Wd') and self.g.opt['rout']: y = self.g.rots()['R1'].inv(y)
        return y

    # ---------------------------------------------------------------- B1 correction
    def correction(self, i, k, xr, xq):
        """input-side: ((xr - xq) Qr)(Qr^T W'^T) with Qr = R^T U (rotated basis); optional output-side weight-error term. bf16 operands, fp32 accum."""
        sp = self.spec['corr']; g = self.g
        key = ('corr', i, k)
        if key not in self.c:
            r = sp['r']; R = g.rot_for(i, k)
            U = self.U(i, k, 'in', sp.get('src', 'all'), r)                # [K, r] unrotated input basis
            Qr = R(U.t().contiguous()).t().contiguous()                      # [K, r] rotated basis
            Wf = self.wout(i, k)                                             # [N, K] unrotated input, stored output basis
            B = (U.t() @ Wf.t())                                             # [r, N]
            ent = dict(Qr=Qr.to(torch.bfloat16), B=B.to(torch.bfloat16))
            if sp.get('w'):
                wr = sp.get('wr', r)
                qw, sw = self.wcodes(i, k, 4)
                Wrot = R(Wf)                                                 # [N, K] rotated input basis (= wfold)
                dW = Wrot - qw.float() * sw[:, None]                          # weight rounding error, rotated input basis, stored output basis
                P = self.U(i, k, 'out', 'all', wr)                            # [N, wr] unrotated output basis
                if k in ('Wo', 'Wd') and g.opt['rout']: P = g.rots()['R1'](P.t().contiguous()).t().contiguous()
                if P.shape[0] != dW.shape[0]: P = P[:dW.shape[0]]
                ent['M'] = (dW.t() @ P).to(torch.bfloat16)                    # [K, wr]
                ent['P'] = P.t().contiguous().to(torch.bfloat16)              # [wr, N]
            self.c[key] = ent
        e = self.c[key]
        dx = (xr - xq).to(torch.bfloat16)
        z = torch.mm(dx, e['Qr']).float().to(torch.bfloat16)
        out = torch.mm(z, e['B']).float()
        if 'M' in e:
            z2 = torch.mm(xr.to(torch.bfloat16), e['M']).float().to(torch.bfloat16)
            out = out + torch.mm(z2, e['P']).float()
        return out

    # ---------------------------------------------------------------- group-scaled int4
    def gemm_group(self, i, k, xr, dith):
        gs = self.spec['group']; g = self.g; key = ('wg', i, k, gs)
        if key not in self.c:
            Wrot = g.wfold(i, k)
            if self.spec['ba16'] and k == 'Win' and g.L[i]['type'] == 'linear_attention': Wrot = Wrot[:8192]
            H = g.hess(i, k) if self.spec['wq'] == 'gptq' else None
            if H is not None: q, S = gptq_group(Wrot, H, 7.0, gs)
            else:
                N, K = Wrot.shape; Wg = Wrot.reshape(N, K // gs, gs)
                S = torch.stack([HL.rtn_scales(Wg[:, j], 7.0, True) for j in range(K // gs)], 1)
                q = torch.round(Wg / S[..., None]).clamp(-7, 7).reshape(N, K)
            self.c[key] = (q.to(torch.int8).contiguous(), S.float())
        qw, S = self.c[key]
        M, K = xr.shape; G_ = K // gs
        xg = xr.reshape(M, G_, gs)
        am = xg.abs().amax(-1).clamp_min(1e-8); s = (am / torch.full_like(am, 7.0)) * self.spec['clip4']        # [M, G]
        z = xg / s[..., None]
        if dith == 'sr': z = torch.floor(z + torch.rand(z.shape, device=z.device, generator=self.gen(i, k)))
        else: z = torch.round(z)
        qa = z.clamp(-7, 7).to(torch.int8)
        y = torch.zeros(M, qw.shape[0], device=xr.device)
        for j in range(G_):
            acc = self.imm(qa[:, j].contiguous(), qw[:, j * gs:(j + 1) * gs].contiguous())
            y += acc * s[:, j:j + 1] * S[None, :, j]
        return y, (qa.float() * s[..., None]).reshape(M, K)

    # ---------------------------------------------------------------- split (ResQ-style / B1-sens)
    def qgemm_split(self, i, k, s_, role):
        sp = self.spec['split']; g = self.g; key = ('split', i, k)
        R = g.rot_for(i, k)
        if key not in self.c:
            r = sp['r']; basis = sp.get('basis', 'pca')
            H0 = torch.load(f"{g.opt['hdir']}/H_{i}_{k}.pt", map_location=g.dev).float()    # unrotated input second moment
            if basis == 'pca':
                ev, V = torch.linalg.eigh(H0); U = V.flip(1)[:, :r].contiguous()
            elif basis == 'sens':
                U = self.U(i, k, 'in', sp.get('src', 'all'), r)
            else:   # mix: top-r of G_in/tr(G_in) + H/tr(H)
                sv = torch.load(f'{SPEC_DIR}/L{i}_{k}.pt', map_location=g.dev)
                ev_, V_ = sv[('in', 'all')]['ev'].float(), sv[('in', 'all')]['V'].float()
                Gl = (V_ * ev_[:V_.shape[1]][None, :]) @ V_.t()
                Mx = Gl / Gl.diagonal().sum() + H0 / H0.diagonal().sum()
                ev, V = torch.linalg.eigh(0.5 * (Mx + Mx.t())); U = V.flip(1)[:, :r].contiguous()
            Wf = self.wout(i, k)                                                   # [N, K], unrotated input
            Pi = torch.eye(U.shape[0], device=g.dev) - U @ U.t()
            Hlo = Pi @ H0 @ Pi
            Hlo_r = R(R(Hlo).t().contiguous())                                     # R^T Hlo R
            Hlo_r = 0.5 * (Hlo_r + Hlo_r.t())
            Wrot = R(Wf)
            if self.spec['wq'] == 'gptq':
                try: qlo, slo = HL.gptq(Wrot, Hlo_r, 7.0)
                except Exception: qlo, slo = HL.gptq(Wrot, Hlo_r, 7.0, percdamp=0.1)    # projected Hessian is rank-deficient by r
            else:
                slo = HL.rtn_scales(Wrot, 7.0, True); qlo = torch.round(Wrot / slo[:, None]).clamp(-7, 7)
            Whi = Wf @ U                                                           # [N, r]
            Hhi = U.t() @ H0 @ U; Hhi = 0.5 * (Hhi + Hhi.t())
            qhi, shi = HL.gptq(Whi, Hhi, 127.0, actorder=True) if self.spec['wq'] == 'gptq' else (None, None)
            if qhi is None:
                shi = Whi.abs().amax(1).clamp_min(1e-8) / 127.0; qhi = torch.round(Whi / shi[:, None]).clamp(-127, 127)
            self.c[key] = dict(U=U, qlo=qlo.to(torch.int8).contiguous(), slo=slo, qhi=qhi.to(torch.int8).contiguous(), shi=shi)
        e = self.c[key]
        xhi = s_ @ e['U']                                                          # [M, r]
        xlo = s_ - xhi @ e['U'].t()
        xr = R(xlo)
        qa, sa = self.qact(xr, 4, self.spec['dither'] if role == 's' else None, self.gen(i, k) if (self.spec['dither'] and role == 's') else None)
        y = self.imm(qa, e['qlo']) * sa[:, None] * e['slo'][None, :]
        hb = sp.get('hi', 8)
        if hb == 16:
            y = y + (xhi.to(torch.bfloat16) @ (e['qhi'].float() * e['shi'][:, None]).to(torch.bfloat16).t()).float()
        else:
            qh, sh = self.qact(xhi, 8)
            y = y + self.imm(qh, e['qhi']) * sh[:, None] * e['shi'][None, :]
        return self.unrot_out(k, y)

    # ---------------------------------------------------------------- B5 transform coding
    def qgemm_tc(self, i, k, s_, role):
        sp = self.spec['tc']; g = self.g; key = ('tc', i, k)
        if key not in self.c:
            fn = f"{B5_DIR}/{sp['tag']}/P_{i}_{k}.pt"
            if not os.path.exists(fn): self.c[key] = None
            else:
                P = torch.load(fn, map_location=g.dev)
                U = P['U'].float(); mu = P['mu'].float(); n8, n4 = P['n8'], P['n4']
                Wf = self.wout(i, k)
                if Wf.shape[1] != U.shape[0]: raise ValueError('dims')
                Wz = Wf @ U; Cz = U.t() @ P['C'].float() @ U
                ent = dict(U=U, mu=mu, n8=n8, n4=n4, bias=(mu @ Wf.t()))
                for nm, lo, hi, qmax, Rm in (('8', 0, n8, 127.0, P['R8']), ('4', n8, n8 + n4, 7.0, P['R4'])):
                    if hi <= lo: continue
                    Rm = Rm.float().to(g.dev); Ws = Wz[:, lo:hi] @ Rm; Hs = Rm.t() @ Cz[lo:hi, lo:hi] @ Rm
                    if self.spec['wq'] == 'gptq': q, s = HL.gptq(Ws, Hs, qmax)
                    else:
                        s = HL.rtn_scales(Ws, qmax, qmax < 100); q = torch.round(Ws / s[:, None]).clamp(-qmax, qmax)
                    ent['R' + nm] = Rm; ent['q' + nm] = q.to(torch.int8).contiguous(); ent['s' + nm] = s.float()
                    ent['UR' + nm] = (U[:, lo:hi] @ Rm).contiguous()          # basis and slice rotation folded: one K x n matmul
                for kk in ('U', 'R8', 'R4'): ent.pop(kk, None)                   # only the folded matrices are needed at run time
                self.c[key] = ent
        e = self.c[key]
        if e is None:                                   # no plan (zero sensitivity): plain rotated int4
            sp2 = dict(self.spec); xr = g.rot_for(i, k)(s_)
            qw, sw = self.wcodes(i, k, 4); qa, sa = self.qact(xr, 4)
            return self.unrot_out(k, self.imm(qa, qw) * sa[:, None] * sw[None, :])
        xc = s_ - e['mu'][None, :]
        y = e['bias'][None, :].expand(xc.shape[0], -1).clone()
        if e['n8']:
            z8 = xc @ e['UR8']; qa, sa = self.qact(z8, 8); y += self.imm(qa, e['q8']) * sa[:, None] * e['s8'][None, :]
        if e['n4']:
            z4 = xc @ e['UR4']
            dith = self.spec['dither'] if role == 's' else None
            qa, sa = self.qact(z4, 4, dith, self.gen(i, k) if dith else None); y += self.imm(qa, e['q4']) * sa[:, None] * e['s4'][None, :]
        return self.unrot_out(k, y)

    # ---------------------------------------------------------------- B3 prototypes
    def qgemm_proto(self, i, k, s_, role):
        sp = self.spec['proto']; g = self.g; key = ('proto', i, k)
        R = g.rot_for(i, k)
        if key not in self.c:
            C = torch.load(f"{PROTO_DIR}/C{sp['Kc']}_{i}_{k}.pt", map_location=g.dev).float()   # [Kc, K] unrotated input basis
            Wf = self.wout(i, k)
            Tb = C @ Wf.t()                                                        # [Kc, N] exact table (fp32)
            qw, sw = self.wcodes(i, k, 4)
            self.c[key] = dict(C=C, c2=(C * C).sum(1), T=Tb, qw=qw, sw=sw)
        e = self.c[key]
        idx = torch.argmax(s_ @ e['C'].t() - 0.5 * e['c2'][None, :], 1)
        res = s_ - e['C'][idx]
        xr = R(res)
        qa, sa = self.qact(xr, 4)
        y = self.imm(qa, e['qw']) * sa[:, None] * e['sw'][None, :]
        return self.unrot_out(k, y + e['T'][idx])            # T = C W^T is in the stored (R1-rotated for Wo/Wd) output basis

    # ---------------------------------------------------------------- install / cost
    def install(self, g):
        g.qfn = {(i, k): self for i in range(24) for k in QL.GEMMS}

    def tcdims(self):
        if not hasattr(self, '_tcd'):
            import json as _j
            f = f"{B5_DIR}/{self.spec['tc']['tag']}.json"; d = _j.load(open(f))['gemms'] if os.path.exists(f) else {}
            self._tcd = {(int(kk.split('.')[0]), kk.split('.')[1]): (v['n8'], v['n4']) for kk, v in d.items() if 'n8' in v}
        return self._tcd

    def work(self):
        """share of GEMM MACs (per row, over the request's row mix: returns per role) done in int4, int8, bf16, and extra correction MACs"""
        sp = self.spec; out = {}
        for role in ('s', 'q'):
            m4 = m8 = m16 = extra = 0.0; tot = 0.0
            for i in range(24):
                for k in QL.GEMMS:
                    N, K = self.g.L[i][k].shape; mac = N * K; tot += mac
                    bs, bq = self.prec_rows(i, k); b = bs if role == 's' else bq
                    if b == 16: m16 += mac; continue
                    if b == 8: m8 += mac; continue
                    if sp['split']:
                        r = sp['split']['r']; m4 += N * (K - r); m8 += N * r; extra += 2 * K * r
                    elif sp.get('tc') and (i, k) in self.tcdims():
                        n8, n4 = self.tcdims()[(i, k)]; m4 += N * n4; m8 += N * n8
                    else:
                        m4 += mac
                    if sp['corr']:
                        r = sp['corr']['r']; extra += K * r + r * N
                        if sp['corr'].get('w'): wr = sp['corr'].get('wr', r); extra += K * wr + wr * N
                    if sp['proto']: extra += K * sp['proto']['Kc']
            out[role] = dict(int4=m4 / tot, int8=m8 / tot, bf16=m16 / tot, extra_bf16_macs=extra / tot)
        return out
