"""Q1 library on top of H1's emulation (h1lib.H1 -> g2lib.G2 lean fused hobson-v19, merged LoRA).

Adds
  * a differentiable forward (dense bf16 GEMMs, autograd through fla GDN / SDPA / pointer head), with per-GEMM output hooks;
  * pluggable per-GEMM GEMM emulations: g.qfn[(i, k)] = callable(g, i, k, x, xn) -> y  (B1 corrections, B2 dither, B3 prototypes, baselines);
  * request sets from the train pool (eval tasks excluded), A and B disjoint by tau task;
  * the decision Fisher directions of the pointer-head distribution.
GEMM naming as H1: Win (GDN in_proj qkvzba [8224] / attn q+gate,k,v [5120]), Wo [2048], Wgu (gate+up) [12288], Wd [2048, K 6144].
GEMM inputs (what a quantized kernel sees): Win/Wgu: xn = gain-free RMSNorm of the residual (gain folded into W); Wo: mixer output; Wd: silu(g)*u.
"""
import os, sys, json, math, random
sys.path[:0] = [os.path.expanduser('~/work/q1'), os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
import h1lib as HL
from g2lib import rms_zc
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv

ATT = HL.ATT
GEMMS = HL.GEMMS
KEYS = [(i, k) for i in range(24) for k in GEMMS]


def kname(i, k):
    return f'{i}.{k}'


def req_sets(nA=256, nB=64, maxT=2600, minS=150, seed=7, per_task=4):
    """train-split (state, question) pairs, one random question per request; eval tasks excluded; A and B use disjoint tau tasks."""
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    rows = []
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV: continue
            rows.append(r)
    tasks = sorted({r['task'] for r in rows})
    rng = random.Random(seed); rng.shuffle(tasks)
    tB = set(tasks[:len(tasks) // 5])
    rng.shuffle(rows)
    A, B = [], []; cnt = {}
    for r in rows:
        if r['n_state_tok'] < minS: continue
        ok = [q for q in sorted(r['questions']) if r['n_state_tok'] + r['q_tok'].get(q, 10 ** 9) <= maxT]
        if not ok: continue
        if cnt.get(r['task'], 0) >= per_task: continue
        q = rng.choice(ok)
        it = dict(rid=r['rid'], task=r['task'], state=r['state'], qname=q, q=r['questions'][q])
        if r['task'] in tB:
            if len(B) < nB: B.append(it); cnt[r['task']] = cnt.get(r['task'], 0) + 1
        else:
            if len(A) < nA: A.append(it); cnt[r['task']] = cnt.get(r['task'], 0) + 1
        if len(A) >= nA and len(B) >= nB: break
    return A, B


def dev_set(n=400, maxT=2600, minS=150, seed=7, per_task=16, skip_rids=()):
    """train-split DEV requests from the held-out tau tasks (the B tasks), excluding given rids; one random question each (different rng)."""
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    rows = []
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV: continue
            rows.append(r)
    tasks = sorted({r['task'] for r in rows})
    rng = random.Random(seed); rng.shuffle(tasks)
    tB = set(tasks[:len(tasks) // 5])
    rng2 = random.Random(seed + 1000); rows = [r for r in rows if r['task'] in tB and r['rid'] not in set(skip_rids)]
    rng2.shuffle(rows)
    out = []; cnt = {}
    for r in rows:
        if r['n_state_tok'] < minS or cnt.get(r['task'], 0) >= per_task: continue
        ok = [q for q in sorted(r['questions']) if r['n_state_tok'] + r['q_tok'].get(q, 10 ** 9) <= maxT]
        if not ok: continue
        q = rng2.choice(ok)
        out.append(dict(rid=r['rid'], task=r['task'], state=r['state'], qname=q, q=r['questions'][q])); cnt[r['task']] = cnt.get(r['task'], 0) + 1
        if len(out) >= n: break
    return out


def fisher_dirs(lg, maxd=6, rel=1e-4):
    """Fisher of softmax(lg) wrt lg: F = diag(p) - p p^T. Returns [(lam, u)], captured share of tr F."""
    p = torch.softmax(lg.detach().double(), -1)
    Fm = torch.diag(p) - torch.outer(p, p)
    lam, U = torch.linalg.eigh(Fm)
    order = torch.argsort(lam, descending=True)
    tr = float(lam.clamp_min(0).sum())
    out = []
    for j in order[:maxd].tolist():
        if lam[j] <= rel * max(float(lam[order[0]]), 1e-30): break
        out.append((float(lam[j]), U[:, j].to(lg.dtype).to(lg.device)))
    cap = sum(l for l, _ in out) / tr if tr > 0 else 1.0
    return out, cap, p.float()


class Q1(HL.H1):
    def __init__(self, grad=True, **opt):
        super().__init__(**opt)
        if grad:
            self.detach_inference()
            for p_ in self.head.parameters(): p_.requires_grad_(False)
        self.qfn = {}          # (i, k) -> callable(g, i, k, x, xn) -> y
        self.hook_fn = None    # callable(i, k, grad[T, N]) on backward
        self.fwd_fn = None     # callable(i, k, x, xn, y) on forward
        self.track = None      # set of (i, k) for hooks (None = all)

    def gain(self, i, k):
        d = self.L[i]
        return d['in1'] if k == 'Win' else (d['post1'] if k == 'Wgu' else None)

    def lin(self, i, k, x, xn=None):
        f = self.qfn.get((i, k))
        y = f(self, i, k, x, xn) if f is not None else super().lin(i, k, x, xn)
        tr = self.track is None or (i, k) in self.track
        if tr and self.fwd_fn is not None: self.fwd_fn(i, k, x, xn, y)
        if tr and y.requires_grad:
            y.register_hook(lambda gr, i=i, k=k: (self.hook_fn(i, k, gr) if self.hook_fn is not None else None))
        return y

    def fwdg(self, ids, gfrom=0, q0=None):
        """H1.fwd without no_grad; layers < gfrom run without grad and the residual is a fresh leaf at layer gfrom."""
        dev = self.dev; T = len(ids); eps = self.eps; self._q0 = q0
        ids_t = torch.tensor(ids, device=dev)
        x = F.embedding(ids_t, self.embed)
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
        for i in range(24):
            if i == gfrom and torch.is_grad_enabled():
                x = x.detach().requires_grad_(True)
            ctx = torch.no_grad() if i < gfrom else torch.enable_grad()
            with ctx:
                x = self._layer(i, x, cos, sin, T, eps)
        return rms_zc(x, self.norm_w, eps)

    def _layer(self, i, x, cos, sin, T, eps):
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
            o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128).contiguous(), k.reshape(1, T, 16, 128).contiguous(), v.reshape(1, T, 16, 128).contiguous(),
                                          g[None].contiguous(), beta[None].to(q.dtype).contiguous(), use_qk_l2norm_in_kernel=True)
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
            ca = getattr(self, 'cap_att', None)
            if ca is not None and i in ca['layers'] and self._q0 is not None and 0 < self._q0 < T:
                with torch.no_grad():                     # mean attention of question rows onto state rows (for row routing)
                    q0 = self._q0; qq = qh[q0:].float(); kq = kk.repeat_interleave(4, dim=1).float()
                    lg_ = torch.einsum('qhd,khd->hqk', qq, kq) / 16.0
                    lg_ = lg_.masked_fill(torch.arange(T, device=x.device)[None, None, :] > (q0 + torch.arange(T - q0, device=x.device))[None, :, None], float('-inf'))
                    ca['scores'][i] = torch.softmax(lg_, -1)[..., :q0].mean(dim=(0, 1))
            o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
        x = x + self.lin(i, 'Wo', o)
        xf = x.float(); xn2 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        h2 = (xn2 * d['post1']).to(x.dtype)
        gu = self.lin(i, 'Wgu', h2, xn2); I = d['I']
        m = F.silu(gu[:, :I]) * gu[:, I:]
        x = x + self.lin(i, 'Wd', m)
        return x

    def logits_g(self, h, pr):
        T = h.shape[0]; opt_abs = [pr['q0'] + o for o in pr['opt']]
        lg = self.head(h[T - 1].float()[None], h[torch.tensor(opt_abs, device=self.dev)].float()[None])[0] / self.p.temp_for(pr['rq'].kind)
        return lg[:pr['rq'].n_slots].float()
