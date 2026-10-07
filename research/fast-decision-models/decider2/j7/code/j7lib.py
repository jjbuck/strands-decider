"""J7 library: hobson-v19 (H3/H7 differentiable lean forward) with a reader-native INPUT layer.
  * super-tokens (superbpe.Super): extended vocabulary V + k; embedding e(t) = mean(E[c]) + sum_j A[j] * E[c_j] + delta_t  (constituent
    composition, ZeTT-like shared part + per-token part); RoPE positions = ORIGINAL Qwen position of each token's last constituent.
  * side channels (chan.py), zero-initialised so the untrained model is exactly hobson:
      emb_rms * ( (R[val_a] + R[val_b]) @ P_val + (R[key_a]+R[key_b]) @ P_key + (R[rec_a]+R[rec_b]) @ P_rec ) + emb_rms * (E_place[place] + E_role[role] + E_turn[turn])
    R = fixed random +-1/8 codes [65536, 64] (two hash buckets per code: collisions of both ~ 2^-32).
Inputs are built by Prep, which reproduces the reference token ids exactly (engine _fit / _option_idx)."""
import os, sys, math, json
sys.path[:0] = [os.path.expanduser('~/work/j7'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/systems/g')]
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import h7lib
from h7lib import H7, Br
from h3lib import rms_zc, DENSE
import chan
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
NB = 65536; MC = 64
CH_KEYS = ('val', 'place', 'key', 'rec', 'role', 'turn')


class Prep:
    """(state, question spec, option order, vocab level) -> dict(ids, pos, opt, q0, feats, n_orig, rq).  Everything on host."""

    def __init__(self, eng, supers=None, protect=False):
        self.eng = eng; self.tok = eng.tok; self.supers = supers or {}
        self.protect = protect      # structure-aware granularity: user / assistant message lines stay at Qwen granularity

    def protect_mask(self, pr):
        enc = self.tok(pr['st_text'], add_special_tokens=True, return_offsets_mapping=True, truncation=True, max_length=len(pr['s']))
        assert enc['input_ids'] == pr['s']
        text = pr['st_text']; ann = chan.annotate(text); r = ann['role'] % 6
        return [bool(r[min(max(b - 1, 0), len(text) - 1)] in (1, 2)) if b > a else False for a, b in enc['offset_mapping']]

    def base(self, state_text, qd, perm=None):
        q = ta.validate_python(qd)
        rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
        s, qs = self.eng._fit(state_text, [rq.text])
        opt = self.eng._option_idx([rq], 0)[0].tolist()
        return dict(s=s, q=qs[0], opt=opt, rq=rq, q0=len(s), L=len(s) + len(qs[0]), st_text=state_text)

    def feats(self, pr):
        """channel features per original token (state then question)."""
        enc = self.tok(pr['st_text'], add_special_tokens=True, return_offsets_mapping=True, truncation=True, max_length=len(pr['s']))
        assert enc['input_ids'] == pr['s'], 'state ids differ from the engine'
        offs = enc['offset_mapping']
        fs = chan.token_feats(pr['st_text'], offs)
        qt = pr['rq'].text
        encq = self.tok(qt, add_special_tokens=False, return_offsets_mapping=True)
        qids = encq['input_ids']; qoffs = encq['offset_mapping']
        cut = len(qids) - len(pr['q'])
        assert qids[cut:] == pr['q']
        fq = chan.token_feats(qt, qoffs[cut:])
        return {k: np.concatenate([fs[k], fq[k]]) for k in CH_KEYS}

    def build(self, pr, level=None, with_feats=False):
        """-> dict ids (list), pos (list of original positions), opt (row indices into ids, question-relative -> absolute), ans (last row),
        q0 (first question row), feats (dict of np arrays aligned with ids) or None, ends (orig index per row)"""
        s, q = pr['s'], pr['q']
        F_ = self.feats(pr) if with_feats else None
        if level is None or level not in self.supers:
            ids = list(s) + list(q); ends = list(range(len(ids)))
            q0 = len(s); opt = [q0 + o for o in pr['opt']]
        else:
            S = self.supers[level]
            ms, es = S.merge_ids(s, self.protect_mask(pr) if self.protect else None); mq, eq = S.merge_ids(q)
            ids = ms + mq; ends = es + [len(s) + e for e in eq]; q0 = len(ms)
            e2r = {e: r for r, e in enumerate(eq)}
            opt = []
            for o in pr['opt']:
                if o in e2r: opt.append(q0 + e2r[o])
                else: opt.append(q0 + next(r for r, e in enumerate(eq) if e >= o))
        out = dict(ids=ids, pos=ends, opt=opt, ans=len(ids) - 1, q0=q0, ends=ends, n_orig=len(s) + len(q), rq=pr['rq'])
        if F_ is not None:
            if len(ends) == pr['L']:
                out['feats'] = F_
            else:
                fo = {}
                st = [0] + [e + 1 for e in ends[:-1]]
                for k in CH_KEYS:
                    a = F_[k]
                    if k in ('val', 'key', 'rec', 'place'):
                        # last non-zero over the constituents (digits are never merged, so place/val of digits stay per token)
                        nz = np.where(a != 0, np.arange(len(a)), -1)
                        last = np.maximum.accumulate(nz)
                        idx = last[np.array(ends)]
                        ok = idx >= np.array(st)
                        fo[k] = np.where(ok, a[np.clip(idx, 0, None)], 0)
                    else:
                        fo[k] = a[np.array(ends)]
                out['feats'] = fo
        return out


class Channels(nn.Module):
    def __init__(self, dev, emb_rms, n_role=48):
        super().__init__()
        g = torch.Generator(device='cpu'); g.manual_seed(7700)
        R = (torch.randint(0, 2, (NB, MC), generator=g).float() * 2 - 1) / math.sqrt(MC)
        R[0] = 0
        self.register_buffer('R', R.to(dev))
        self.P = nn.ParameterDict({k: nn.Parameter(torch.zeros(MC, 2048, device=dev)) for k in ('val', 'key', 'rec')})
        self.E = nn.ParameterDict(dict(place=nn.Parameter(torch.zeros(32, 2048, device=dev)), role=nn.Parameter(torch.zeros(n_role, 2048, device=dev)),
                                       turn=nn.Parameter(torch.zeros(32, 2048, device=dev))))
        self.scale = float(emb_rms)
        self.n_role = n_role

    def codes(self, h):
        """h: int64 numpy hashes (0 = none) -> [T, MC] (two buckets)"""
        a = torch.as_tensor((h % NB).astype(np.int64), device=self.R.device)
        b = torch.as_tensor(((h >> 20) % NB).astype(np.int64), device=self.R.device)
        z = torch.as_tensor(h == 0, device=self.R.device)
        b = torch.where(z, torch.zeros_like(b), b); a = torch.where(z, torch.zeros_like(a), a)
        return self.R[a] + self.R[b]

    def forward(self, feats):
        add = 0.0
        for k in ('val', 'key', 'rec'):
            add = add + self.codes(feats[k]) @ self.P[k]
        dev = self.R.device
        add = add + self.E['place'][torch.as_tensor(np.clip(feats['place'], 0, 31), device=dev)]
        add = add + self.E['role'][torch.as_tensor(np.clip(feats['role'], 0, self.n_role - 1), device=dev)]
        add = add + self.E['turn'][torch.as_tensor(np.clip(feats['turn'], 0, 31), device=dev)]
        return add * self.scale


class SuperEmb(nn.Module):
    """embeddings of super-tokens from their constituents: mean + per-position (from the end) gains + per-token delta (all trainable but E)."""

    def __init__(self, base_embed, toks, dev, maxc=24):
        super().__init__()
        N = len(toks); self.N = N
        C = torch.zeros(N, maxc, dtype=torch.long); M = torch.zeros(N, maxc)
        for i, t in enumerate(toks):
            t = t[-maxc:]; L = len(t)
            C[i, :L] = torch.tensor(t[::-1]); M[i, :L] = 1.0      # constituent j counted from the END (j = 0 is the last constituent)
        self.register_buffer('C', C.to(dev)); self.register_buffer('M', M.to(dev))
        self.A = nn.Parameter(torch.zeros(maxc, 2048, device=dev))
        self.delta = nn.Parameter(torch.zeros(N, 2048, device=dev))
        self.base = base_embed

    def freeze_table(self, chunk=2048):
        """inference: precompute every super-token row once -> plain bf16 table (a gather at serve time)."""
        with torch.no_grad():
            T = torch.empty(self.N, 2048, device=self.C.device, dtype=torch.bfloat16)
            for i in range(0, self.N, chunk):
                k = torch.arange(i, min(self.N, i + chunk), device=self.C.device)
                T[k] = self.rows(k).to(torch.bfloat16)
        self.table = T

    table = None

    def rows(self, k):
        if self.table is not None: return self.table[k].float()
        """k: LongTensor of super-token indices -> [n, 2048] fp32"""
        c = self.C[k]; m = self.M[k]
        e = self.base[c].float()                                   # [n, maxc, 2048]
        mean = (e * m[..., None]).sum(1) / m.sum(1, keepdim=True)
        comp = (e * m[..., None] * self.A[None]).sum(1)
        return mean + comp + self.delta[k]


class J7(H7):
    chans = None; sup = None; V = None

    def emb_rms(self):
        return float(self.embed.float().pow(2).mean().sqrt())

    def add_channels(self):
        self.chans = Channels(self.dev, self.emb_rms())
        return list(self.chans.parameters())

    def add_super(self, toks, V):
        self.V = V
        self.sup = SuperEmb(self.embed, toks, self.dev)
        return list(self.sup.parameters())

    def embed_in(self, b):
        ids = torch.as_tensor(b['ids'], device=self.dev)
        if self.sup is not None and (ids >= self.V).any():
            new = ids >= self.V
            x = F.embedding(torch.where(new, torch.zeros_like(ids), ids), self.embed).float()
            k = (ids[new] - self.V)
            x[new] = self.sup.rows(k)
        else:
            x = F.embedding(ids, self.embed).float()
        if self.chans is not None and b.get('feats') is not None:
            x = x + self.chans(b['feats'])
        return x.to(torch.bfloat16)

    def fwd_x(self, x, pos, br=None, ckpt=False, keep=(), keep_rows=None):
        """plain causal (br None) or H7 branch forward from input embeddings."""
        cos, sin = self.cos_sin(torch.as_tensor(pos, device=self.dev, dtype=torch.float32))
        self.kept = {}
        for i in range(24):
            if br is None:
                f = lambda i_, x_: self.layer(i_, x_, cos, sin, DENSE)
            else:
                f = lambda i_, x_: self.layer_br(i_, x_, cos, sin, br)
            if ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(f, i, x, use_reentrant=False)
            else:
                x = f(i, x)
            if i in keep: self.kept[i] = x if keep_rows is None else x[keep_rows]
        return rms_zc(x, self.norm_w, self.eps)

    def logits1(self, b, ckpt=False, keep=(), keep_rows=None, head=None):
        """one (state, question) sequence -> [n_slots] logits (temperature applied)"""
        head = head or self.head or self.head0
        x = self.embed_in(b)
        h = self.fwd_x(x, b['pos'], ckpt=ckpt, keep=keep, keep_rows=keep_rows)
        oi = torch.tensor(b['opt'], device=self.dev)
        lg = head(h[b['ans']].float()[None], h[oi].float()[None])[0] / self.temp(b['rq'].kind)
        return lg[:b['rq'].n_slots]

    @torch.no_grad()
    def logits_multi(self, bs, head=None):
        """several questions over one state (bs share the same state rows): state once, each question a branch (H7 'seqs')."""
        head = head or self.head or self.head0
        q0 = bs[0]['q0']
        ids = list(bs[0]['ids'][:q0]); pos = list(bs[0]['pos'][:q0]); seqs = []
        feats = None
        if bs[0].get('feats') is not None:
            feats = {k: [bs[0]['feats'][k][:q0]] for k in CH_KEYS}
        for b in bs:
            assert b['q0'] == q0
            seqs.append((len(ids), len(b['ids']) - q0)); ids += b['ids'][q0:]; pos += b['pos'][q0:]
            if feats is not None:
                for k in CH_KEYS: feats[k].append(b['feats'][k][q0:])
        bb = dict(ids=ids, feats={k: np.concatenate(v) for k, v in feats.items()} if feats is not None else None)
        x = self.embed_in(bb)
        br = Br('seqs', q0, seqs=seqs)
        h = self.fwd_x(x, pos, br=br)
        out = []
        for (s0, Lb), b in zip(seqs, bs):
            oi = torch.tensor([s0 + (o - q0) for o in b['opt']], device=self.dev)
            lg = head(h[s0 + Lb - 1].float()[None], h[oi].float()[None])[0] / self.temp(b['rq'].kind)
            out.append(lg[:b['rq'].n_slots])
        return out

    # ---------- state io ----------
    def j7_state(self):
        sd = self.trainable_state()
        if self.chans is not None:
            for k, v in self.chans.state_dict().items():
                if k != 'R': sd['chans.' + k] = v.detach().cpu()
        if self.sup is not None:
            sd['sup.A'] = self.sup.A.detach().cpu(); sd['sup.delta'] = self.sup.delta.detach().to(torch.bfloat16).cpu()
        return sd

    def load_j7(self, path, toks=None, V=None):
        sd = torch.load(path, map_location=self.dev)
        self.load_trainable(path)
        if any(k.startswith('chans.') for k in sd):
            self.add_channels()
            self.chans.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('chans.')}, strict=False)
            for p in self.chans.parameters(): p.requires_grad_(False)
        if 'sup.A' in sd:
            self.add_super(toks, V)
            with torch.no_grad():
                self.sup.A.copy_(sd['sup.A']); self.sup.delta.copy_(sd['sup.delta'].float())
            for p in self.sup.parameters(): p.requires_grad_(False)
        return sd
