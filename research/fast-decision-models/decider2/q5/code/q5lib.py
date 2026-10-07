"""Q5 library: structural savings (2:4 sparsity B11, MLP neuron removal B12) on top of Q1's emulation (q1lib.Q1 -> h1lib.H1 -> g2lib.G2),
with Q1's precision formats (q1fmt.Fmt) for every dense part. Q1's files are imported, never edited.

Structure config (dict), all keys optional:
  fmt : None (bf16 runtime) | Q1 Fmt spec dict (e.g. cfgs.json 'b8')        -> every GEMM / row that is not structured
  s24 : 2:4 sparsity along K (groups of 4 consecutive inputs in the KERNEL basis keep 2):
        layers [..], gemms ['Win','Wo','Wgu','Wd'], role 'all' | 's' (state rows sparse, question rows dense),
        prec 'bf16n' (natural basis, bf16) | 'bf16r' (rotated basis, bf16 operands) | 'int8' | 'int4' (rotated, H1 FORMAT v1 arithmetic),
        method 'mag0' (magnitude, no update) | 'wanda' (|w| sqrt(H_kk), no update) | 'sgpt' (SparseGPT, layer Hessian) |
               'sgptd' (SparseGPT, decision-weighted Hessian) | 'fisher' (decision Fisher-diagonal saliency x OBS compensability, H_d updates),
        cal 'gen' | 'bank' | 'ret', keep_bf16 True (GEMMs the base fmt keeps in bf16 stay dense)
  neur: MLP neuron removal: layers [..], frac f (per layer) or count {i: n}, method 'var' | 'dsal' | 'obs' | 'obsd' | 'dead',
        cal 'gen'|'bank'|'ret', comp True|False (Wd compensation for obs*), tol (for 'dead')
"""
import os, sys, json, math, random, time, hashlib
sys.path[:0] = [os.path.expanduser('~/work/q5'), os.path.expanduser('~/work/q1'), os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2'),
                os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
_mf = os.environ.get('Q5_MEMFRAC') or (open(os.path.expanduser('~/work/q5/memfrac')).read().strip() if os.path.exists(os.path.expanduser('~/work/q5/memfrac')) else None)
if _mf:                                              # BRIEF10: GPU memory < 22 GB in total; each concurrent job gets a fraction of the 23.7 GB A10G
    torch.cuda.set_per_process_memory_fraction(float(_mf))
import q1lib as QL, h1lib as HL, q1fmt as QF

W5 = os.path.expanduser('~/work/q5')
CAL = f'{W5}/cal'
CACHE = f'{W5}/cache'
GEMMS = HL.GEMMS
ATT = HL.ATT


# ------------------------------------------------------------------ request sets (train split only; eval tasks excluded)
def _pool():
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    rows = []
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV: continue
            rows.append(r)
    return rows


_POOL = None


def req_set(dom, part, n, maxT=2600, minS=150, per_task=None, seed=11, dev_frac=0.25):
    """dom: 'bank' | 'ret'. part: 'cal' (calibration tasks) | 'dev' (dev tasks, disjoint by tau task). One random question per request."""
    global _POOL
    if _POOL is None: _POOL = _pool()
    if per_task is None: per_task = 12 if part == 'dev' else 6
    pre = 'banking' if dom == 'bank' else 'retail'
    rows = [r for r in _POOL if r['task'].startswith(pre)]
    tasks = sorted({r['task'] for r in rows})
    rng = random.Random(seed); rng.shuffle(tasks)
    nd = max(1, int(round(dev_frac * len(tasks))))
    keep = set(tasks[:nd]) if part == 'dev' else set(tasks[nd:])
    rows = [r for r in rows if r['task'] in keep]
    rng2 = random.Random(seed + (1 if part == 'dev' else 2)); rng2.shuffle(rows)
    out, cnt = [], {}
    for r in rows:
        if r['n_state_tok'] < minS or cnt.get(r['task'], 0) >= per_task: continue
        ok = [q for q in sorted(r['questions']) if r['n_state_tok'] + r['q_tok'].get(q, 10 ** 9) <= maxT]
        if not ok: continue
        q = rng2.choice(ok)
        out.append(dict(rid=r['rid'], task=r['task'], dom=dom, state=r['state'], qname=q, q=r['questions'][q])); cnt[r['task']] = cnt.get(r['task'], 0) + 1
        if len(out) >= n: break
    return out


# ------------------------------------------------------------------ calibration statistics
def cal_load(cal, i, k, what='H'):
    """sum of per-domain raw accumulators. cal: 'bank' | 'ret' | 'gen' (= bank + ret)."""
    doms = {'gen': ['bank', 'ret'], 'gen2': ['bank', 'ret', 'bank2', 'ret2'], 'bank+': ['bank', 'bank2'], 'ret+': ['ret', 'ret2']}.get(cal, [cal])
    out = None
    for d in doms:
        fn = f'{CAL}/{d}/{what}_{i}_{k}.pt'
        x = torch.load(fn, map_location='cpu')
        if out is None: out = x
        elif isinstance(x, dict):
            for kk, v in x.items(): out[kk] = out[kk] + v
        else: out = out + x
    return out


def hessians(cal, i, k, dev):
    """returns dict of fp32 [K,K] on dev: H (all rows), Hs (state rows), Hd (decision-weighted, all rows), Hds; plus means."""
    A = cal_load(cal, i, k, 'H')
    o = {}
    o['H'] = ((A['Hs'] + A['Hq']) / (A['n_s'] + A['n_q'])).to(dev)
    o['Hs'] = (A['Hs'] / max(A['n_s'], 1)).to(dev)
    o['Hd'] = ((A['Hds'] + A['Hdq']) / (A['w_s'] + A['w_q'])).to(dev)
    o['Hds'] = (A['Hds'] / max(A['w_s'], 1e-30)).to(dev)
    o['mu'] = ((A['sum_s'] + A['sum_q']) / (A['n_s'] + A['n_q'])).float().to(dev)
    o['mu_s'] = (A['sum_s'] / max(A['n_s'], 1)).float().to(dev)
    o['mud'] = ((A['sumd_s'] + A['sumd_q']) / (A['w_s'] + A['w_q'])).float().to(dev)
    o['mud_s'] = (A['sumd_s'] / max(A['w_s'], 1e-30)).float().to(dev)
    return o


def rot_H(g, i, k, H):
    R = g.rot_for(i, k)
    return R(R(H).t().contiguous())                 # R^T H R


# ------------------------------------------------------------------ 2:4 pruning (SparseGPT family)
def prune24(W, H, method='sgpt', Fd=None, qmax=None, scales=None, mask=None, blocksize=128, percdamp=0.01, pairs=False):
    """W [N,K] fp32 in the kernel basis, H [K,K] (same basis). Returns (W_new [N,K] fp32 with exact zeros at pruned positions, keep mask bool [N,K], codes or None).
    method: mag0 | wanda | sgpt | sgptd (same code; H differs) | fisher (saliency Fd*w^2 / (H_kk d_k^2), OBS updates with H).
    mask: fixed keep-mask (then only the OBS/GPTQ update runs). qmax/scales: joint quantization of kept weights with per-channel scales [N]."""
    W = W.clone().float(); N, K = W.shape
    if method in ('mag0', 'wanda') and mask is None:
        sc = W.abs() if method == 'mag0' else W.abs() * torch.diag(H).clamp_min(0).sqrt()[None, :]
        if pairs:                                  # Ampere int4 sparsity: keep 2 of every 4 adjacent PAIRS (groups of 8)
            g8 = sc.pow(2).reshape(N, K // 8, 4, 2).sum(-1)
            idx = torch.topk(g8, 2, dim=2, largest=True).indices
            kp = torch.zeros_like(g8, dtype=torch.bool).scatter_(2, idx, True)
            keep = kp[..., None].expand(N, K // 8, 4, 2).reshape(N, K)
            return W * keep, keep, None
        g4 = sc.reshape(N, K // 4, 4)
        idx = torch.topk(g4, 2, dim=2, largest=True).indices
        keep = torch.zeros_like(g4, dtype=torch.bool).scatter_(2, idx, True).reshape(N, K)
        return W * keep, keep, None
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    Hkk = torch.diag(H).clone()
    damp = percdamp * torch.mean(torch.diag(H)); H[range(K), range(K)] += damp
    L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L); Hinv = torch.linalg.cholesky(Hinv, upper=True)
    keep = torch.ones(N, K, dtype=torch.bool, device=W.device) if mask is None else mask.clone()
    Q = torch.zeros_like(W); C = torch.zeros_like(W) if qmax is not None else None
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K); n = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        dg = torch.diag(Hi)
        for j in range(n):
            col = i1 + j
            gsz = 8 if pairs else 4
            if mask is None and j % gsz == 0:
                blk = W1[:, j:j + gsz]
                if method == 'fisher':
                    s = Fd[:, col:col + gsz] * blk ** 2 / (Hkk[col:col + gsz] * dg[j:j + gsz] ** 2)[None, :]
                else:
                    s = blk ** 2 / (dg[j:j + gsz] ** 2)[None, :]
                if pairs:                          # saliency of a pair = sum of its two elements; prune the 2 least salient pairs
                    sp = s.reshape(N, 4, 2).sum(-1)
                    ip = torch.topk(sp, 2, dim=1, largest=False).indices
                    idx = torch.cat([2 * ip, 2 * ip + 1], 1)
                else:
                    idx = torch.topk(s, 2, dim=1, largest=False).indices
                keep[:, col:col + gsz] = True
                keep[:, col:col + gsz].scatter_(1, idx, False)
            w = W1[:, j]; d = Hi[j, j]
            q = w * keep[:, col]
            if qmax is not None:
                cq = torch.round(q / scales).clamp(-qmax, qmax) * keep[:, col]
                C[:, col] = cq; q = cq * scales
            Q1[:, j] = q
            e = (w - q) / d
            W1[:, j:] -= e[:, None] * Hi[j, j:][None, :]
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, keep, C


def check24(keep, pairs=False):
    N, K = keep.shape
    if pairs:
        kp = keep.reshape(N, K // 8, 4, 2)
        return bool((kp[..., 0] == kp[..., 1]).all() and (kp[..., 0].sum(2) <= 2).all())
    return bool((keep.reshape(N, K // 4, 4).sum(2) <= 2).all())


# ------------------------------------------------------------------ the structured model
class S5:
    """Installs a structure config on a Q1 model g (g.qfn)."""
    def __init__(self, g, cfg, tag='x', verbose=True):
        self.g = g; self.cfg = cfg or {}; self.tag = tag; self.v = verbose
        fc = self.cfg.get('fcal')                         # B7: GPTQ calibration of the precision format: None = H1's pool sample | 'bank' | 'ret'
        g.opt['hdir'] = f'{W5}/hess_{fc}' if fc else os.path.expanduser('~/work/h1/hess')
        QF.CODE_DIR = f'{W5}/codes_{fc}' if fc else os.path.expanduser('~/work/q1/codes')
        g.Qw = {}
        fs = self.cfg.get('fmt')
        self.fm = QF.Fmt(g, fs) if fs else None
        self.sp = {}       # (i,k) -> dict(role, prec, W (bf16 natural) | Wr (bf16 rotated) | codes (q int8, s))
        self.nr = {}       # i -> dict(keep LongTensor, rowmask for Wgu outputs, Wd (bf16 [2048, 6144] with zeros), bias fp32 [2048], codes8 / codes4)
        self.stats = {}
        if self.cfg.get('s24'):
            for c in (self.cfg['s24'] if isinstance(self.cfg['s24'], list) else [self.cfg['s24']]): self.build_s24(c)
        if self.cfg.get('neur'): self.build_neur(self.cfg['neur'])

    # ---------------------------------------------------------- helpers
    def base_prec(self, i, k):
        if self.fm is None: return 16, 16
        return self.fm.prec_rows(i, k)

    def wnat(self, i, k):
        """folded weight [N,K] in the natural input basis (gain folded; Win/Wgu input = gain-free norm xn), unrotated output"""
        d = self.g.L[i]; W = d[k].float()
        if k == 'Win': W = W * d['in1'][None, :]
        if k == 'Wgu': W = W * d['post1'][None, :]
        return W

    # ---------------------------------------------------------- 2:4
    def build_s24(self, c):
        g = self.g; dev = g.dev
        layers = c['layers']; gemms = c.get('gemms', list(GEMMS)); role = c.get('role', 'all'); prec = c.get('prec', 'bf16n')
        method = c.get('method', 'sgptd'); cal = c.get('cal', 'gen')
        Hkey = {'sgpt': 'H', 'mag0': 'H', 'wanda': 'H', 'sgptd': 'Hd', 'fisher': 'Hd'}[method]
        if role == 's': Hkey = Hkey + 's'
        os.makedirs(CACHE, exist_ok=True)
        t0 = time.time(); nmac = 0
        for i in layers:
            for k in gemms:
                if c.get('keep_bf16', True) and self.fm is not None and self.fm.prec_rows(i, k) == (16, 16): continue
                key = f's24_{i}_{k}_{role}_{prec}_{method}_{cal}_{c.get("damp", 0.01)}' + (f'_sm{c.get("smooth", 0.0)}' if prec == 'int8n' else '')
                fn = f'{CACHE}/{key}.pt'
                if os.path.exists(fn):
                    ent = torch.load(fn, map_location=dev)
                else:
                    Hs = hessians(cal, i, k, dev); H = Hs[Hkey]
                    Wn = self.wnat(i, k)
                    Fd = None
                    if prec == 'int8n':
                        # natural basis + optional per-channel smoothing s_k (SmoothQuant-style, alpha = c['smooth']; applied in the quantize prologue);
                        # diagonal scaling keeps the 2:4 pattern and leaves the OBS mask unchanged
                        if method == 'fisher': Fd = cal_load(cal, i, k, 'Fn').to(dev)
                        al = c.get('smooth', 0.0)
                        Hp = hessians(cal, i, k, dev)['H']
                        rms = torch.diag(Hp).clamp_min(1e-12).sqrt(); wmax = Wn.abs().amax(0).clamp_min(1e-8)
                        sm = (rms ** al / wmax ** (1 - al)) if al > 0 else torch.ones_like(rms)
                        sm = sm / sm.mean() if al > 0 else sm
                        Ws = Wn * sm[None, :]; Hsm = H / torch.outer(sm, sm)
                        Fs = (Fd / (sm[None, :] ** 2)) if Fd is not None else None
                        W1, keep, _ = prune24(Ws, Hsm, method, Fd=Fs, percdamp=c.get('damp', 0.01))
                        sw = HL.rtn_scales(W1, 127.0, False)
                        src_W = W1 if method in ('mag0', 'wanda') else Ws
                        _, _, codes = prune24(src_W, Hsm, 'sgpt', mask=keep, qmax=127.0, scales=sw, percdamp=c.get('damp', 0.01))
                        ent = dict(q=codes.to(torch.int8).contiguous(), s=sw.float().contiguous(), sm=sm.float().contiguous())
                    elif prec == 'bf16n':
                        if method == 'fisher': Fd = cal_load(cal, i, k, 'Fn').to(dev)
                        Wq, keep, _ = prune24(Wn, H, method, Fd=Fd, percdamp=c.get('damp', 0.01))
                        gain = g.L[i]['in1'] if k == 'Win' else (g.L[i]['post1'] if k == 'Wgu' else None)
                        Wb = (Wq / gain[None, :]) if gain is not None else Wq
                        ent = dict(W=Wb.to(torch.bfloat16).contiguous())
                    else:
                        Wr = g.wfold(i, k)                                  # kernel basis (rotated input, rotated output for Wo/Wd)
                        Hr = rot_H(g, i, k, H)
                        if method == 'fisher': Fd = cal_load(cal, i, k, 'Frs' if role == 's' else 'Fr').to(dev)
                        if prec == 'bf16r':
                            Wq, keep, _ = prune24(Wr, Hr, method, Fd=Fd, percdamp=c.get('damp', 0.01))
                            ent = dict(Wr=Wq.to(torch.bfloat16).contiguous())
                        else:
                            qmax = 127.0 if prec == 'int8' else 7.0
                            pw = prec == 'int4p'
                            W1, keep, _ = prune24(Wr, Hr, method, Fd=Fd, percdamp=c.get('damp', 0.01), pairs=pw)
                            s = HL.rtn_scales(W1, qmax, qmax < 100)
                            if method in ('mag0', 'wanda'):      # no pruning compensation: GPTQ of the masked weights (quantization error only)
                                _, _, codes = prune24(W1, Hr, 'sgpt', mask=keep, qmax=qmax, scales=s, percdamp=c.get('damp', 0.01))
                            else:
                                _, _, codes = prune24(Wr, Hr, 'sgpt', mask=keep, qmax=qmax, scales=s, percdamp=c.get('damp', 0.01))
                            ent = dict(q=codes.to(torch.int8).contiguous(), s=s.float().contiguous())
                    assert check24(keep, pairs=(prec == 'int4p'))
                    torch.save(ent, fn)
                ent['role'] = role; ent['prec'] = prec
                self.sp[(i, k)] = ent
                N, K = g.L[i][k].shape; nmac += N * K
        self.stats['s24_macs'] = self.stats.get('s24_macs', 0) + nmac
        if self.v: print(f'[s24] {len(self.sp)} GEMMs {method}/{prec}/{role}/{cal} built in {time.time()-t0:.0f}s', flush=True)

    def sparse_gemm(self, i, k, x, xn, rows):
        g = self.g; e = self.sp[(i, k)]; prec = e['prec']
        src = xn if k in ('Win', 'Wgu') else x
        lo, hi = rows
        if prec == 'bf16n':
            return (x[lo:hi] @ e['W'].t())
        if prec == 'int8n':
            xs = src[lo:hi].float() / e['sm'][None, :]
            qa, sa = (self.fm or self._qa()).qact(xs, 8)
            return (g.imm(qa, e['q']) * sa[:, None] * e['s'][None, :]).to(x.dtype)
        R = g.rot_for(i, k)
        xr = R(src[lo:hi].float())
        if prec == 'bf16r':
            y = (xr.to(torch.bfloat16) @ e['Wr'].t()).float()
        else:
            bits = 8 if prec == 'int8' else 4          # int4 and int4p (pair-wise mask) share the arithmetic
            qa, sa = (self.fm or self._qa()).qact(xr, bits)
            y = g.imm(qa, e['q']) * sa[:, None] * e['s'][None, :]
        if k in ('Wo', 'Wd') and g.opt['rout']: y = g.rots()['R1'].inv(y)
        return y.to(x.dtype)

    def _qa(self):
        if not hasattr(self, '_qfm'): self._qfm = QF.Fmt(self.g, dict(a_s=8, a_q=8))
        return self._qfm

    # ---------------------------------------------------------- neuron removal
    def neuron_plan(self, c):
        """returns {layer: removed index LongTensor}"""
        g = self.g; method = c.get('method', 'dsal'); cal = c.get('cal', 'gen'); layers = c['layers']
        NE = neur_stats(cal)
        plan = {}
        if c.get('global'):
            # global ranking by decision saliency (mean replacement), total budget = frac * len(layers) * 6144 neurons
            sc = torch.cat([NE['S2m'][i] for i in layers])
            nrem = int(round(c['frac'] * len(layers) * 6144))
            order = torch.argsort(sc)[:nrem]
            for li, i in enumerate(layers):
                sel = order[(order >= li * 6144) & (order < (li + 1) * 6144)] - li * 6144
                if len(sel): plan[i] = sel
            return plan
        for i in layers:
            n = int(c['count'][str(i)]) if 'count' in c else int(round(c.get('frac', 0.1) * 6144))
            if method == 'dead':
                mx = NE['max'][i]; sel = torch.nonzero(mx < c.get('tol', 1e-2)).flatten()
            elif method == 'var':
                Wd = g.L[i]['Wd'].float().cpu(); sc = NE['var'][i] * Wd.pow(2).sum(0)
                sel = torch.argsort(sc)[:n]
            elif method == 'dsal':
                sel = torch.argsort(NE['S2m'][i])[:n]
            elif method in ('obs', 'obsd'):
                sel = None     # chosen inside build (greedy OBS)
            else: raise ValueError(method)
            plan[i] = sel if sel is not None else n
        return plan

    def build_neur(self, c):
        g = self.g; dev = g.dev; cal = c.get('cal', 'gen'); method = c.get('method', 'dsal')
        plan = self.neuron_plan(c)
        t0 = time.time(); nrem_tot = 0
        for i, sel in plan.items():
            d = g.L[i]; I = d['I']
            Wd = d['Wd'].float()                                   # [2048, 6144]
            need_H = isinstance(sel, int) or c.get('comp', False)
            if need_H:
                Hs = hessians(cal, i, 'Wd', dev)
                dec = method in ('obsd',) or c.get('dec_comp', False)
                Hm = Hs['Hd'] if dec else Hs['H']; mu = Hs['mud'] if dec else Hs['mu']
                Cv = Hm - torch.outer(mu, mu)                      # (weighted) covariance of m
            else:
                mu = neur_stats(cal)['mean'][i].float().to(dev)    # plain mean over calibration rows (mean replacement)
            if isinstance(sel, int):                               # greedy OBS selection with compensation
                Gm = None
                if method == 'obsd':
                    Go = cal_load(cal, i, 'Wd', 'Gout').to(dev).float(); Gm = Go / Go.diagonal().sum()
                sel, Wn = obs_remove(Wd, Cv, sel, Gm=Gm, percdamp=c.get('damp', 0.01))
            else:
                if c.get('comp', False) and len(sel):
                    Wn = obs_compensate(Wd, Cv, sel, percdamp=c.get('damp', 0.01))
                else:
                    Wn = Wd.clone()
            rem = torch.zeros(I, dtype=torch.bool, device=dev); rem[sel.to(dev)] = True
            # bias: removed neurons replaced by the (weighted) mean, kept neurons' change of weights applied to their mean as well
            dW = Wn - Wd                                           # kept columns' updates (removed columns ignored)
            bias = Wd[:, rem] @ mu[rem] - dW[:, ~rem] @ mu[~rem]
            Wn[:, rem] = 0
            ent = dict(rem=rem, Wd=Wn.to(torch.bfloat16).contiguous(), bias=bias.float(), nrem=int(rem.sum()))
            self.nr[i] = ent; nrem_tot += int(rem.sum())
        self.stats['neur_removed'] = nrem_tot
        self.stats['neur_per_layer'] = {int(i): e['nrem'] for i, e in self.nr.items()}
        if self.v: print(f'[neur] {method}/{cal}: removed {nrem_tot} neurons in {len(self.nr)} layers ({time.time()-t0:.0f}s) {self.stats["neur_per_layer"]}', flush=True)

    def neur_codes(self, i, bits):
        """GPTQ codes of the modified Wd (rotated basis; removed inputs dead in the Hessian), cached per config"""
        key = ('nc', i, bits)
        if key not in self.nr[i]:
            g = self.g; e = self.nr[i]
            Wf = e['Wd'].float()
            R = g.rot_for(i, 'Wd'); Wr = R(Wf)
            if g.opt['rout']: Wr = g.rots()['R1'](Wr.t().contiguous()).t().contiguous()
            H0 = torch.load(f"{g.opt['hdir']}/H_{i}_Wd.pt", map_location=g.dev).float()
            H0[e['rem'], :] = 0; H0[:, e['rem']] = 0
            Hr = R(R(H0).t().contiguous())
            q, s = HL.gptq(Wr, Hr, HL.QMAX[bits])
            self.nr[i][key] = (q.to(torch.int8).contiguous(), s.float().contiguous())
        return self.nr[i][key]

    # ---------------------------------------------------------- the GEMM hook
    def __call__(self, g, i, k, x, xn):
        T = x.shape[0]; q0 = g._q0 if g._q0 is not None else T
        q0 = min(max(q0, 0), T)
        if (i, k) in self.sp:
            e = self.sp[(i, k)]
            if e['role'] == 'all':
                return self.sparse_gemm(i, k, x, xn, (0, T))
            y = torch.empty(T, g.L[i][k].shape[0], device=x.device, dtype=x.dtype)
            if q0 > 0: y[:q0] = self.sparse_gemm(i, k, x, xn, (0, q0))
            if q0 < T: y[q0:] = self.dense_rows(i, k, x, xn, q0, T, role='q')
            return y
        if i in self.nr and k in ('Wgu', 'Wd'):
            return self.neur_gemm(i, k, x, xn)
        return self.dense(i, k, x, xn)

    def dense(self, i, k, x, xn):
        if self.fm is None: return x @ self.g.L[i][k].t()
        return self.fm(self.g, i, k, x, xn)

    def dense_rows(self, i, k, x, xn, lo, hi, role):
        g = self.g
        if self.fm is None: return x[lo:hi] @ g.L[i][k].t()
        q0 = g._q0; g._q0 = 0 if role == 'q' else hi - lo + 1
        y = self.fm(g, i, k, x[lo:hi], None if xn is None else xn[lo:hi]); g._q0 = q0
        return y

    def neur_gemm(self, i, k, x, xn):
        g = self.g; e = self.nr[i]; I = g.L[i]['I']
        if k == 'Wgu':
            y = self.dense(i, k, x, xn)
            y[:, :I][:, e['rem']] = 0                               # removed neurons: gate output 0 -> silu(0) * up = 0 exactly
            return y
        # Wd with modified weights (+ bias)
        if self.fm is None:
            y = x @ e['Wd'].t()
        else:
            bs, bq = self.fm.prec_rows(i, 'Wd')
            T = x.shape[0]; q0 = g._q0 if g._q0 is not None else T; q0 = min(max(q0, 0), T)
            y = torch.empty(T, 2048, device=x.device, dtype=x.dtype)
            for lo, hi, bits in ((0, q0, bs), (q0, T, bq)):
                if hi <= lo: continue
                if bits == 16: y[lo:hi] = x[lo:hi] @ e['Wd'].t(); continue
                R = g.rot_for(i, 'Wd'); xr = R(x[lo:hi].float())
                qw, sw = self.neur_codes(i, bits)
                qa, sa = self.fm.qact(xr, bits)
                yy = g.imm(qa, qw) * sa[:, None] * sw[None, :]
                if g.opt['rout']: yy = g.rots()['R1'].inv(yy)
                y[lo:hi] = yy.to(x.dtype)
        return (y.float() + e['bias'][None, :]).to(x.dtype)

    def install(self):
        self.g.qfn = {(i, k): self for i in range(24) for k in GEMMS}

    # ---------------------------------------------------------- work accounting
    def work(self, q_frac=0.1):
        """shares of dense GEMM MACs (per row): 2:4-sparse share, removed share; for state rows and question rows"""
        g = self.g; tot = 0.0; sp_s = sp_q = 0.0; rm = 0.0
        for i in range(24):
            for k in GEMMS:
                N, K = g.L[i][k].shape; mac = N * K; tot += mac
                if (i, k) in self.sp:
                    sp_s += mac
                    if self.sp[(i, k)]['role'] == 'all': sp_q += mac
            if i in self.nr: rm += self.nr[i]['nrem'] * 3 * 2048
        return dict(sparse_state=sp_s / tot, sparse_q=sp_q / tot, removed=rm / tot,
                    sparse_at_qfrac=(1 - q_frac) * sp_s / tot + q_frac * sp_q / tot)


def obs_compensate(W, C, sel, percdamp=0.01):
    """Remove columns sel of W [N,K] given input covariance C [K,K]; least-squares update of the kept columns: W_R += W_S C_SR C_RR^-1."""
    K = W.shape[1]; dev = W.device
    rem = torch.zeros(K, dtype=torch.bool, device=dev); rem[sel.to(dev)] = True
    C = C.clone(); damp = percdamp * torch.mean(torch.diag(C)); C[range(K), range(K)] += damp
    CRR = C[~rem][:, ~rem]; CSR = C[rem][:, ~rem]
    A = torch.linalg.solve(CRR, CSR.t()).t()                           # [S, R] = C_SR C_RR^-1
    Wn = W.clone()
    Wn[:, ~rem] = W[:, ~rem] + W[:, rem] @ A
    return Wn


def obs_remove(W, C, n, Gm=None, percdamp=0.01, chunk=32):
    """Greedy structured OBS: remove n input columns of W [N,K] (input covariance C) one chunk at a time, choosing the columns with the smallest
    error w_j^T G w_j / [C^-1]_jj (G = output metric, identity if None), updating W (compensation) and C^-1 (downdate). Returns (removed idx, W_new)."""
    W = W.clone().float(); K = W.shape[1]; dev = W.device
    C = C.clone().float(); damp = percdamp * torch.mean(torch.diag(C)); C[range(K), range(K)] += damp
    Ci = torch.linalg.inv(C.double()).float()
    alive = torch.ones(K, dtype=torch.bool, device=dev); removed = []
    while len(removed) < n:
        dg = torch.diag(Ci).clamp_min(1e-20)
        num = (W * (Gm @ W)).sum(0) if Gm is not None else W.pow(2).sum(0)
        err = num / dg; err[~alive] = float('inf')
        m = min(chunk, n - len(removed))
        js = torch.topk(err, m, largest=False).indices
        for j in js.tolist():                                           # sequential exact OBS steps within the chunk
            d = Ci[j, j]
            W -= torch.outer(W[:, j], Ci[j, :]) / d
            Ci -= torch.outer(Ci[:, j], Ci[j, :]) / d
            W[:, j] = 0; Ci[j, :] = 0; Ci[:, j] = 0; Ci[j, j] = 1e30     # removed: never chosen again (dg huge -> err 0 but alive masks it)
            alive[j] = False; removed.append(j)
    return torch.tensor(removed, device=dev), W


_NE = {}


def neur_stats(cal):
    """per-layer neuron statistics [24, 6144] (cpu, float64): mean, var, max, fire, S2 (zeroing), S2m (mean replacement) from the cal pass."""
    if cal in _NE: return _NE[cal]
    doms = {'gen': ['bank', 'ret'], 'gen2': ['bank', 'ret', 'bank2', 'ret2'], 'bank+': ['bank', 'bank2'], 'ret+': ['ret', 'ret2']}.get(cal, [cal])
    acc = None
    for d in doms:
        x = torch.load(f'{CAL}/{d}/neurons.pt', map_location='cpu')
        if acc is None: acc = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items() if not isinstance(v, dict)}
        else:
            for k, v in x.items():
                if isinstance(v, dict): continue
                if k == 'max': acc[k] = torch.maximum(acc[k], v)
                else: acc[k] = acc[k] + v
    n = acc['rows']
    mu = acc['sum'] / n; var = acc['sq'] / n - mu ** 2
    S2m = acc['A2'] - 2 * mu * acc['AB'] + mu ** 2 * acc['B2']
    out = dict(mean=mu, var=var.clamp_min(0), max=acc['max'], fire=acc['f2'] / n, fire3=acc['f3'] / n, fire1=acc['f1'] / n, S2=acc['A2'], S2m=S2m.clamp_min(0), absmean=acc['abs'] / n, n=n)
    _NE[cal] = out
    return out


# ------------------------------------------------------------------ cached prefix (layers < start identical to the base format)
from g2lib import rms_zc
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv


@torch.no_grad()
def fwd_from(self, ids, x0, start, q0=None, keep_x=(), stop=None):
    """H1.fwd copied verbatim (h1lib.py), except: the residual entering layer `start` is x0 (cached), layers < start are skipped."""
    dev = self.dev; T = len(ids); eps = self.eps; self._q0 = q0
    ids_t = torch.tensor(ids, device=dev)
    x = x0
    if getattr(self, 'fp32', False): x = x.float()
    pos = torch.arange(T, device=dev, dtype=torch.float32)
    fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
    caps = {}
    nL = 24 if stop is None else stop
    for i in range(start, nL):
        d = self.L[i]; gdn = d['type'] == 'linear_attention'
        xf = x.float(); xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        h = (xn * d['in1']).to(x.dtype)
        proj = self.lin(i, 'Win', h, xn)
        if gdn:
            qkv = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; a = proj[:, 8208:8224]
            beta = torch.sigmoid(b.float()); g = -d['A_log'].float().exp() * F.softplus(a.float() + d['dt_bias'])
            qkv = fla_conv(qkv[None].contiguous(), d['conv_w'], None, activation='silu')
            qkv = qkv[0] if isinstance(qkv, tuple) else qkv
            q, k, v = qkv.split(2048, dim=-1)
            o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), g[None],
                                          beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
            of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
            o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(x.dtype).reshape(T, 2048)
        else:
            qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
            kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
            qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
            def rope(t):
                xr_, xp = t[..., :64], t[..., 64:]; x1, x2 = xr_[..., :32], xr_[..., 32:]
                c = cos[:, None, :]; s_ = sin[:, None, :]
                return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
            qh, kk = rope(qh), rope(kk)
            o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
            o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
        x = x + self.lin(i, 'Wo', o)
        xf = x.float(); xn2 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        h2 = (xn2 * d['post1']).to(x.dtype)
        gu = self.lin(i, 'Wgu', h2, xn2); I = d['I']
        m = F.silu(gu[:, :I]) * gu[:, I:]
        x = x + self.lin(i, 'Wd', m)
        if i in keep_x: caps[i] = x
    if stop is not None: return None, caps
    return rms_zc(x, self.norm_w, eps), caps


# ------------------------------------------------------------------ dev evaluation (train split, held-out tau tasks)
def dev_items(nb=160, nr=96):
    return req_set('bank', 'dev', nb) + req_set('ret', 'dev', nr)


@torch.no_grad()
def run_items(g, items, prefix=None, start=0):
    """prefix: list (per item) of {layer: residual entering that layer (cpu bf16)} from the same base format; start: first layer to recompute"""
    out = []
    for j, it in enumerate(items):
        pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']
        if prefix is not None and start > 0 and start in prefix[j]:
            h, _ = fwd_from(g, ids, prefix[j][start].to(g.dev), start, q0=pr['q0'])
        else:
            h, _ = g.fwd(ids, q0=pr['q0'])
        out.append(g.logits(h, pr).float().cpu())
    return out


@torch.no_grad()
def build_prefix(g, items, layers):
    """residual entering each layer in `layers` (cpu bf16), with whatever g.qfn is installed (the base format)"""
    out = []
    for it in items:
        pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']
        _, caps = g.fwd(ids, keep_x=tuple(L - 1 for L in layers), stop=max(layers), q0=pr['q0'])
        out.append({L: caps[L - 1].cpu() for L in layers})
    return out


def first_layer(cfg):
    L = []
    if cfg.get('s24'):
        for c in (cfg['s24'] if isinstance(cfg['s24'], list) else [cfg['s24']]): L += c['layers']
    if cfg.get('neur'): L += cfg['neur']['layers']
    return min(L) if L else 0


def dev_metrics(ref, lg, items):
    """KL(ref||cfg), TV, argmax flips vs the reference logits, per domain"""
    res = {}
    for dom in ('bank', 'ret', 'all'):
        kl = tv = 0.0; fl = n = 0
        for r, l, it in zip(ref, lg, items):
            if dom != 'all' and it['dom'] != dom: continue
            p = torch.softmax(r.double(), -1); q = torch.softmax(l.double(), -1)
            kl += float((p * (p.clamp_min(1e-12).log() - q.clamp_min(1e-12).log())).sum()); tv += float(0.5 * (p - q).abs().sum())
            fl += int(p.argmax() != q.argmax()); n += 1
        res[dom] = dict(n=n, kl=kl / max(n, 1), tv=tv / max(n, 1), flips=fl, flip_rate=fl / max(n, 1))
    return res
