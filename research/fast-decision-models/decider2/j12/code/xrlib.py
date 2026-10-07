"""J12 XR: hobson-v19 (full 24 layers, h3lib differentiable lean forward, LoRA optional) + eXact-Relation readout modules.

XR instance at layer L (applied to the residual after layer L, QUESTION ROWS ONLY -> state rows, and so the shared state prefix, are untouched):
  slots   = typed literals of state + question (xlit.py), key = W_k RMSNorm(h[last token of the literal]) + E_cat[join/type category]
  per head h (H heads): two pointers p_a, p_b over slots (+ a learned null slot), causal (a row sees slots at positions <= itself)
  exact readout (all linear in p_b given p_a, O(Q*N) via scatter / cumsum / gather; no N^2):
     eq      = P(eqkey(a) == eqkey(b))                 same_blk = P(block(a) == block(b))
     lt / gt = P(val(a) < / > val(b), same order group (number | date))
     date    = P(val(b) - val(a) > t), P(val(a) - val(b) > t) for t in xlit.DATE_T days
     null_a, null_b
  -> W_o (zero init) -> added to the question rows' residual.  At init the model is exactly hobson.
"""
import os, sys, math, json
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import torch, torch.nn as nn, torch.nn.functional as F
from h3lib import H3, rms_zc, nrm
import xlit
from strands_decider.prompting import render_question, render_state

NT = len(xlit.DATE_T)
NF = 6 + 2 * NT


class XR(nn.Module):
    def __init__(self, d=2048, H=16, dk=64, ncat=xlit.NCAT, seed=0):
        super().__init__()
        self.H, self.dk = H, dk
        g = torch.Generator().manual_seed(seed)
        def lin(i, o):
            m = nn.Linear(i, o, bias=False)
            with torch.no_grad(): m.weight.copy_(torch.randn(o, i, generator=g) / math.sqrt(i))
            return m
        self.qa, self.qb, self.ka, self.kb = lin(d, H * dk), lin(d, H * dk), lin(d, H * dk), lin(d, H * dk)
        self.cat_a = nn.Embedding(ncat, H * dk); self.cat_b = nn.Embedding(ncat, H * dk)
        with torch.no_grad():
            self.cat_a.weight.normal_(0, 0.3, generator=g); self.cat_b.weight.normal_(0, 0.3, generator=g)
        self.null_a = nn.Parameter(torch.zeros(H, dk)); self.null_b = nn.Parameter(torch.zeros(H, dk))
        self.null_bias = nn.Parameter(torch.zeros(2, H))
        self.nw = nn.Parameter(torch.ones(d))
        self.wo = nn.Linear(H * NF, d, bias=False)
        nn.init.zeros_(self.wo.weight)
        self.last = None

    @staticmethod
    def readout(pa, pb, ix):
        """pa, pb [H, Q, N+1] pointer distributions (last = null) -> exact relation features [H, Q, NF]"""
        H, Q, N1 = pa.shape; N = N1 - 1; dev = pa.device
        pas, pbs = pa[..., :N], pb[..., :N]
        out = []
        for gk, G in (('eqg', ix['G']), ('blkg', ix['B'])):
            gid = ix[gk]
            M = torch.zeros(H, Q, G, device=dev, dtype=pbs.dtype).index_add_(2, gid, pbs)
            out.append((pas * M[..., gid]).sum(-1))
        NL = ix['NL']
        if NL > 0:
            O = ix['ord']; lvl = ix['lvl'][O]
            Ml = torch.zeros(H, Q, NL, device=dev, dtype=pbs.dtype).index_add_(2, lvl, pbs[..., O])
            Cp = F.pad(torch.cumsum(Ml, -1), (1, 0))
            lo, hi = ix['lo'][O], ix['hi'][O]
            pao = pas[..., O]
            m_lt = Cp[..., lvl] - Cp[..., lo]
            m_gt = Cp[..., hi + 1] - Cp[..., lvl + 1]
            out.append((pao * m_gt).sum(-1)); out.append((pao * m_lt).sum(-1))
            D = ix['dsl']
            if D.numel() > 0:
                pad_ = pao[..., D]; hiD = hi[D] + 1; loD = lo[D]
                for t in range(NT): out.append((pad_ * (Cp[..., hiD] - Cp[..., ix['dgt'][t]])).sum(-1))
                for t in range(NT): out.append((pad_ * (Cp[..., ix['dlt'][t] + 1] - Cp[..., loD])).sum(-1))
            else:
                out += [torch.zeros(H, Q, device=dev)] * (2 * NT)
        else:
            out += [torch.zeros(H, Q, device=dev)] * (2 + 2 * NT)
        out.append(pa[..., N]); out.append(pb[..., N])
        f = torch.stack(out, -1)
        return torch.cat([f[..., 4 + 2 * NT:], f[..., :4 + 2 * NT]], -1)   # na, nb, eq, blk, lt, gt, dgt.., dlt..

    def forward(self, h, q0, ix, keep_last=False):
        """h [T, d] (bf16) -> delta [T - q0, d] for the question rows"""
        T = h.shape[0]; Q = T - q0; H, dk = self.H, self.dk; dev = h.device
        hf = h.float()
        hn = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.nw
        rows = hn[q0:]
        Qa = self.qa(rows).view(Q, H, dk); Qb = self.qb(rows).view(Q, H, dk)
        sc = 1.0 / math.sqrt(dk)
        na = torch.einsum('qhd,hd->hq', Qa, self.null_a) * sc + self.null_bias[0][:, None]
        nb = torch.einsum('qhd,hd->hq', Qb, self.null_b) * sc + self.null_bias[1][:, None]
        N = ix['N']
        if N == 0:
            feats = torch.zeros(H, Q, NF, device=dev)
            feats[..., 0] = 1.0; feats[..., 1] = 1.0
            if keep_last: self.last = None
        else:
            S = hn[ix['pos']]
            Ka = self.ka(S).view(N, H, dk) + self.cat_a(ix['cat']).view(N, H, dk)
            Kb = self.kb(S).view(N, H, dk) + self.cat_b(ix['cat']).view(N, H, dk)
            la = torch.einsum('qhd,nhd->hqn', Qa, Ka) * sc
            lb = torch.einsum('qhd,nhd->hqn', Qb, Kb) * sc
            rowpos = torch.arange(q0, T, device=dev)
            bad = (ix['pos'][None, :] > rowpos[:, None])[None]           # [1, Q, N]
            la = la.masked_fill(bad, float('-inf')); lb = lb.masked_fill(bad, float('-inf'))
            pa = torch.softmax(torch.cat([la, na[..., None]], -1), -1)
            pb = torch.softmax(torch.cat([lb, nb[..., None]], -1), -1)
            if keep_last: self.last = (pa[:, -1], pb[:, -1])
            feats = self.readout(pa, pb, ix)
        y = self.wo(feats.permute(1, 0, 2).reshape(Q, H * NF))
        return y


def make_ix(s_lits, s_off, q_lits, q_off, q0, q_text, dev):
    sl = xlit.build_slots(s_lits, s_off, q_lits, q_off, q0, q_text)
    a = xlit.index_arrays(sl)
    t = lambda x: torch.tensor(x, dtype=torch.long, device=dev)
    ix = dict(N=a['N'], G=a['G'], B=a['B'], NL=a['NL'], pos=t(a['pos']), cat=t(a['cat']), eqg=t(a['eqg']), blkg=t(a['blkg']),
              lvl=t(a['lvl']), lo=t(a['lo']), hi=t(a['hi']))
    O = [i for i, l in enumerate(a['lvl']) if l >= 0]
    ix['ord'] = t(O)
    Dpos = [k for k, i in enumerate(O) if sl['grp'][i] == 2]
    ix['dsl'] = t(Dpos)
    Dslots = [O[k] for k in Dpos]
    ix['dgt'] = [t([a['dgt'][j][i] for i in Dslots]) for j in range(NT)]
    ix['dlt'] = [t([a['dlt'][j][i] for i in Dslots]) for j in range(NT)]
    ix['_sl'] = sl
    return ix


class Prep:
    """exact token ids as hobson's engine (_fit with max_length) + literal slots"""

    def __init__(self, eng, maxlen=16384):
        self.eng = eng; self.tok = eng.tok; self.maxlen = maxlen
        self._cache = {}

    def state(self, state):
        st = render_state(state)
        key = hash(st)
        if key in self._cache: return self._cache[key]
        enc = self.tok(st, add_special_tokens=True, return_offsets_mapping=True)
        r = (st, enc['input_ids'], enc['offset_mapping'], xlit.extract(st))
        if len(self._cache) > 64: self._cache.clear()
        self._cache[key] = r
        return r

    def __call__(self, state, qd, perm=None, max_tok=None):
        from pydantic import TypeAdapter
        import strands_decider.schema as SC
        q = TypeAdapter(SC.Question).validate_python(qd) if isinstance(qd, dict) else qd
        rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
        st, sids, soff, slits = self.state(state)
        enc = self.tok(rq.text, add_special_tokens=False, return_offsets_mapping=True)
        qids, qoff = enc['input_ids'], enc['offset_mapping']
        s, qs = self.eng._fit(st, [rq.text])
        opt = self.eng._option_idx([rq], 0)[0].tolist()
        assert qs[0] == qids, 'question truncated / mismatch'
        if s != sids[:len(s)]:
            raise AssertionError('state ids mismatch')
        if max_tok is not None and len(s) + len(qids) > max_tok:      # training only: keep head 1/4 + tail 3/4 of the state
            k = max_tok - len(qids)
            if k < 64: return None
            keep = list(range(k // 4)) + list(range(len(s) - (k - k // 4), len(s)))
            s = [s[i] for i in keep]; soff_ = [soff[i] for i in keep]
            mp = {old: new for new, old in enumerate(keep)}
            s_lits = slits; s_off = soff_
        else:
            s_off = soff[:len(s)]; s_lits = slits
        q_lits = xlit.extract(rq.text)
        q_lits = q_lits + xlit.qwords(rq.text, q_lits, s_lits)
        return dict(s=s, q=qids, opt=opt, rq=rq, q0=len(s), L=len(s) + len(qids), s_off=s_off, q_off=qoff, s_lits=s_lits, q_lits=q_lits,
                    st=st, qtext=rq.text)


class HX(H3):
    XR_LAYERS = (11, 17)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.xr = None; self.xr_on = True

    def add_xr(self, H=16, dk=64, layers=None):
        if layers is not None: self.XR_LAYERS = tuple(layers)
        self.xr = nn.ModuleList([XR(H=H, dk=dk, seed=17 + j) for j in range(len(self.XR_LAYERS))]).to(self.dev)
        return list(self.xr.parameters())

    def ix_for(self, pr):
        return make_ix(pr['s_lits'], pr['s_off'], pr['q_lits'], pr['q_off'], pr['q0'], pr['qtext'], self.dev)

    def fwd(self, pr, ix=None, ckpt=False, keep_last=False):
        ids = pr['s'] + pr['q']; q0 = pr['q0']
        dev = self.dev; T = len(ids)
        ids_t = torch.tensor(ids, device=dev)
        x = F.embedding(ids_t, self.embed)
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
        from h3lib import DENSE
        use_xr = self.xr is not None and self.xr_on
        if use_xr and ix is None: ix = self.ix_for(pr)
        self._q0 = None
        for i in range(24):
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(self.layer, i, x, cos, sin, DENSE, None, use_reentrant=False)
            else:
                x = self.layer(i, x, cos, sin, DENSE, None)
            if use_xr and i in self.XR_LAYERS:
                k = self.XR_LAYERS.index(i)
                y = self.xr[k](x, q0, ix, keep_last=keep_last)
                x = torch.cat([x[:q0], (x[q0:].float() + y).to(x.dtype)], 0)
        h = rms_zc(x, self.norm_w, self.eps)
        return h

    def logits_pr(self, h, pr, head=None):
        head = head or self.head or self.head0
        T = h.shape[0]
        opt_abs = torch.tensor([pr['q0'] + o for o in pr['opt']], device=self.dev)
        lg = head(h[T - 1].float()[None], h[opt_abs].float()[None])[0] / self.temp(pr['rq'].kind)
        return lg[:pr['rq'].n_slots]

    def trainable_state(self):
        sd = super().trainable_state()
        if self.xr is not None:
            for k, v in self.xr.state_dict().items(): sd[f'xr.{k}'] = v.detach().cpu()
            sd['_meta']['xr_layers'] = list(self.XR_LAYERS); sd['_meta']['xr_H'] = self.xr[0].H; sd['_meta']['xr_dk'] = self.xr[0].dk
        return sd

    def load_all(self, path):
        meta = self.load_trainable(path)
        sd = torch.load(path, map_location=self.dev)
        if any(k.startswith('xr.') for k in sd):
            self.add_xr(H=meta['xr_H'], dk=meta['xr_dk'], layers=meta['xr_layers'])
            self.xr.load_state_dict({k[3:]: v for k, v in sd.items() if k.startswith('xr.')})
            for p_ in self.xr.parameters(): p_.requires_grad_(False)
        return meta


def gold_slots(ix, it, pr):
    """slot indices (or N for null) of the gold pointers; None if not locatable"""
    sl = ix['_sl']; N = ix['N']
    st = it['state']; shift = len('<state>\n') - (len(st) - len(st.lstrip()))
    out = {}
    for k in ('a', 'b'):
        g = it['ptr'].get(k)
        if g is None: out[k] = None; continue
        if g[0] == 'null': out[k] = N; continue
        if g[0] == 's':
            a0 = g[1] + shift; a1 = a0 + len(g[2])
            js = [j for j, (l, q) in enumerate(zip(sl['lits'], sl['isq'])) if not q and l.start <= a0 and l.end >= a1]
        else:
            js = [j for j, (l, q) in enumerate(zip(sl['lits'], sl['isq'])) if q and (l.text == g[2] or l.text.strip('\'"') == g[2])]
        if not js: return None
        out[k] = js[0]
    return out


def order_gold(gs, ix, pr):
    """order (a, b) by first mention in the question text: a = operand mentioned first. State literals are located by their field name."""
    if gs is None or gs.get('b') is None or gs['a'] == ix['N'] or gs['b'] == ix['N']: return gs
    sl = ix['_sl']; qt = pr['qtext'].lower()
    def mpos(j):
        l = sl['lits'][j]
        if sl['isq'][j]: return l.start
        if l.field is None: return None
        cands = [qt.find(l.field), qt.find(l.field.replace(' ', '_'))]
        cands = [c for c in cands if c >= 0]
        return min(cands) if cands else None
    pa, pb = mpos(gs['a']), mpos(gs['b'])
    if pa is not None and pb is not None and pb < pa:
        return dict(a=gs['b'], b=gs['a'])
    return gs
