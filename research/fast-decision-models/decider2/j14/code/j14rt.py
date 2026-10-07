"""J14 runtime: H2's QRT (bf16 fold path and rotated W8A8-b8 path) on J5's short-M kernels, with the J14 layout:
rows computed per request = [state Ls rows] + [live question rows R] (option-end rows, '<answer>', suffix rows), every row at its
natural position. Compiled question rows contribute (per deployment, precomputed by j14lib in bf16):
  GDN : post-conv l2-normalised q, k, v and raw (b, a) of every compiled row (persistent span buffers; live rows are scattered in),
        raw pre-conv rows used as conv neighbours of live rows (TAIL table);
        'replay': varlen chunk GDN over each question's whole span from the state's final GDN state;
        'affine': the leading compiled run of each span (all rows before the first live row) is one precomputed affine transfer
                  S <- A_q S + B_q (exact: frozen compiled rows), then the remainder of the span is replayed.
  attn: post-norm PRE-RoPE K and V of every compiled row; K rotated to the natural positions inside the graph.
Requires the 'frozen' conv semantics of j14lib (compiled rows' post-conv values constant).
"""
import os, sys, json, time, math, statistics as st, collections
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/j14'), os.path.expanduser('~/work/j5'), os.path.expanduser('~/work/h2/code'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g')]
import torch, torch.nn.functional as F
import qrt as Q, qk as K
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

dev = 'cuda'


def rope(t, cos, sin):
    xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
    c = cos[:, None, :]; s_ = sin[:, None, :]
    return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)


def l2h(x):
    xf = x.float(); return (xf / torch.sqrt(xf.pow(2).sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)


class J14Lay:
    """qs: list of dict(q=ids, lp=live positions, comp=lib compile cache (per layer dicts), opt=option offsets, temp, n_slots)"""

    def __init__(self, Ls, qs, inv, mode='replay', layer_types=None):
        self.kind = 'j14'; self.Ls = Ls; self.mode = mode; n = len(qs); self.n = n
        live_ids = []; pos = list(range(Ls)); prev = [[t - 3 + j if t - 3 + j >= 0 else -1 for j in range(3)] for t in range(Ls)]
        sidx_full = []; sidx_rem = []; cu_rem = [0]; base_full = 0; base_rem = 0
        tail_rows = []                     # (question j, span position p) -> TAIL index
        tail_key = {}
        self.pool_rows = []; self.opt_rows = []; Kmax = max(len(x['opt']) for x in qs)
        r = Ls; l0s = []
        for j, x in enumerate(qs):
            q = x['q']; lp = x['lp']; Lq = len(q); lps = set(lp)
            l0 = lp[0] if mode == 'affine' else 0; l0s.append(l0)
            rowof = {}
            for p in lp:
                rowof[p] = r; live_ids.append(q[p]); pos.append(Ls + p)
                pv = []
                for jj in range(3):
                    pp = p - 3 + jj
                    if pp < 0:
                        sp = Ls + pp; pv.append(sp if sp >= 0 else -1)                  # state row (live proj row) or zero
                    elif pp in lps: pv.append(rowof[pp])
                    else:
                        key = (j, pp)
                        if key not in tail_key: tail_key[key] = len(tail_rows); tail_rows.append(key)
                        pv.append(-2 - tail_key[key])
                prev.append(pv)
                sidx_full.append(base_full + p); sidx_rem.append(base_rem + p - l0)
                r += 1
            self.pool_rows.append(rowof[Lq - 1])
            self.opt_rows.append([rowof[o] for o in x['opt']] + [rowof[x['opt'][0]]] * (Kmax - len(x['opt'])))
            base_full += Lq; base_rem += Lq - l0; cu_rem.append(base_rem)
        self.T = r; self.R = r - Ls; self.Nsp = base_full; self.Nrem = base_rem
        self.ids_live = torch.tensor(live_ids, device=dev)
        self.pos = torch.tensor(pos, device=dev, dtype=torch.float32)
        self.prev = torch.tensor(prev, device=dev, dtype=torch.int32).contiguous()
        self.sidx_full = torch.tensor(sidx_full, device=dev); self.sidx_rem = torch.tensor(sidx_rem, device=dev)
        self.cu = torch.tensor(cu_rem, device=dev, dtype=torch.long)
        self.pool_rows = torch.tensor(self.pool_rows, device=dev); self.opt_rows = torch.tensor(self.opt_rows, device=dev)
        self.temps = torch.tensor([x['temp'] for x in qs], device=dev); self.nsl = torch.tensor([x['n_slots'] for x in qs], device=dev)
        self.kmask = torch.arange(Kmax, device=dev)[None, :] < self.nsl[:, None]
        # natural positions of every span row (for K rotation)
        sp_pos = torch.cat([torch.arange(Ls, Ls + len(x['q']), device=dev, dtype=torch.float32) for x in qs])
        fr = sp_pos[:, None] * inv[None, :]; fr = torch.cat([fr, fr], -1)
        self.cos_sp, self.sin_sp = fr.cos().to(torch.bfloat16).contiguous(), fr.sin().to(torch.bfloat16).contiguous()
        # attention mask for live queries over [state Ls | spans Nsp]
        am = torch.zeros(self.R, Ls + self.Nsp, dtype=torch.bool, device=dev); am[:, :Ls] = True
        rr = 0; b = 0
        for x in qs:
            for p in x['lp']:
                am[rr, Ls + b:Ls + b + p + 1] = True; rr += 1
            b += len(x['q'])
        self.amask = am
        # compiled buffers per layer
        self.cc = dict(tail={}, sq={}, sk={}, sv={}, sab={}, kpre={}, vsp={}, A={}, B={})
        types = layer_types
        for i in range(24):
            if types[i] == 'linear_attention':
                rem = [(x, l0) for x, l0 in zip(qs, l0s)]
                cvs = torch.cat([x['comp'][i]['cv'][l0:] for x, l0 in rem], 0)                     # [Nrem, 6144] post-conv (pre-l2norm)
                qq, kk, vv = cvs.split(2048, -1)
                self.cc['sq'][i] = l2h(qq.reshape(-1, 16, 128)).contiguous(); self.cc['sk'][i] = l2h(kk.reshape(-1, 16, 128)).contiguous()
                self.cc['sv'][i] = vv.reshape(-1, 16, 128).to(torch.bfloat16).contiguous()
                self.cc['sab'][i] = torch.cat([x['comp'][i]['ab'][l0:] for x, l0 in rem], 0).to(torch.bfloat16).contiguous()   # [Nrem, 32] (b, a)
                tl_ = [qs[j]['comp'][i]['raw'][p].float() for j, p in tail_rows]
                self.cc['tail'][i] = (torch.stack(tl_, 0) if tl_ else torch.zeros(1, 6144, device=dev)).contiguous()
                if mode == 'affine':
                    self.cc['A'][i] = torch.stack([x['comp'][i]['A'][l0] for x, l0 in rem], 0).contiguous()   # [n, 16, 128, 128]
                    self.cc['B'][i] = torch.stack([x['comp'][i]['B'][l0] for x, l0 in rem], 0).contiguous()
            else:
                self.cc['kpre'][i] = torch.cat([x['comp'][i]['K'] for x in qs], 0).to(torch.bfloat16).contiguous()        # [Nsp, 2, 256]
                self.cc['vsp'][i] = torch.cat([x['comp'][i]['V'] for x in qs], 0).to(torch.bfloat16).contiguous()


class J14Mix:
    def _gdn(self, i, d, proj, ra, cs, lay, cache, dq):
        if lay.kind != 'j14': return super()._gdn(i, d, proj, ra, cs, lay, cache, dq)
        c = lay.cc; Ls = lay.Ls; n = lay.n
        qkv3, ab = K.conv(proj, ra, cs, d['conv_w'], lay.prev, c['tail'][i], dq=dq)
        kw = dict(use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'], use_beta_sigmoid_in_kernel=True)
        q, k, v = qkv3[0], qkv3[1], qkv3[2]
        o1, S = chunk_gated_delta_rule(q[None, :Ls], k[None, :Ls], v[None, :Ls], ab[:Ls, 16:32].reshape(1, Ls, 16), ab[:Ls, 0:16].reshape(1, Ls, 16),
                                       output_final_state=True, **kw)
        Qb, Kb, Vb, ABb = c['sq'][i], c['sk'][i], c['sv'][i], c['sab'][i]
        si = lay.sidx_rem
        Qb.index_copy_(0, si, q[Ls:]); Kb.index_copy_(0, si, k[Ls:]); Vb.index_copy_(0, si, v[Ls:]); ABb.index_copy_(0, si, ab[Ls:])
        if lay.mode == 'affine':
            S0 = (torch.matmul(c['A'][i], S) + c['B'][i]).contiguous()
        else:
            S0 = S.expand(n, -1, -1, -1).contiguous()
        o2, _ = chunk_gated_delta_rule(Qb[None], Kb[None], Vb[None], ABb[:, 16:32][None], ABb[:, 0:16][None], initial_state=S0, cu_seqlens=lay.cu, **kw)
        return torch.cat([o1[0], o2[0].index_select(0, si)], 0)

    def _attn(self, i, d, proj, ra, cs, lay, cache, dq, cos, sin):
        if lay.kind != 'j14': return super()._attn(i, d, proj, ra, cs, lay, cache, dq, cos, sin)
        c = lay.cc; Ls = lay.Ls; T = lay.T
        kb = torch.empty(T, 2, 256, device=dev, dtype=torch.bfloat16); vb = torch.empty_like(kb)
        q = K.aprep(proj, ra, cs, d['qn'], d['kn'], cos, sin, kb, vb, 0, self.eps, dq=dq)
        Ksp = rope(c['kpre'][i], lay.cos_sp, lay.sin_sp)
        Vsp = c['vsp'][i]
        Ksp.index_copy_(0, lay.sidx_full, kb[Ls:]); Vsp.index_copy_(0, lay.sidx_full, vb[Ls:])
        qh = q.transpose(0, 1)[None]
        o1 = F.scaled_dot_product_attention(qh[:, :, :Ls], kb[:Ls].transpose(0, 1)[None], vb[:Ls].transpose(0, 1)[None], is_causal=True, enable_gqa=True)
        Ka = torch.cat([kb[:Ls], Ksp], 0); Va = torch.cat([vb[:Ls], Vsp], 0)
        o2 = F.scaled_dot_product_attention(qh[:, :, Ls:], Ka.transpose(0, 1)[None], Va.transpose(0, 1)[None], attn_mask=lay.amask[None, None], enable_gqa=True)
        return torch.cat([o1[0], o2[0]], 1)


def make(ln, head, spec):
    import j5rt
    base = j5rt.make(ln, head, spec)
    cls = type('J14' + type(base).__name__, (J14Mix, type(base)), {})
    base.__class__ = cls
    return base


class J14Req:
    def __init__(self, m, Ls, qs, mode='replay'):
        self.m = m; self.Ts = Ls
        self.lay = J14Lay(Ls, qs, m.ln2.inv, mode=mode, layer_types=[d['type'] for d in m.L])
        self.ids = torch.cat([torch.randint(1000, 100000, (Ls,), device=dev), self.lay.ids_live])

    def run(self):
        m = self.m; lay = self.lay
        hn = m.forward(self.ids, lay, None)
        dec = m.unrot(hn[lay.pool_rows]).float()
        opts = m.unrot(hn[lay.opt_rows.reshape(-1)]).float().reshape(lay.opt_rows.shape[0], lay.opt_rows.shape[1], -1)
        lg = m.head(dec, opts) / lay.temps[:, None]
        lg = lg.masked_fill(~lay.kmask, float('-inf'))
        return torch.softmax(lg, -1)


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): out = fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    return g, out


def wall(req, g, out, reps=20, warm=3):
    Ts = req.Ts
    pool = [torch.randint(1000, 100000, (Ts,), dtype=torch.long).pin_memory() for _ in range(reps + warm)]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory()
    ts = []
    for x in pool:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        req.ids[:Ts].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[warm:])
    return dict(median=round(st.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3), n=len(ts))
