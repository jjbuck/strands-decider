"""G2 training: hobson-v19 + LoRA (r16, all 4 projection groups, 24 layers) under PER-TOKEN NESTED WIDTH at a random capacity each step
(MoNE / MatFormer style), self-distilled from frozen hobson (KL on the answer distribution) on train-split real states + gold CE on train_v5.
--arm nest : state rows outside the top-f by the router signal run at width w (MLP neurons and GDN/attention heads, leading sub-blocks),
             f ~ U{.10,.15,.20,.30}, w ~ U{1/8, 1/4, 1/2} per step. Router signal = frozen hobson's question->state attention at layer 7 from the
             teacher pass (exact) with prob .5, else that signal corrupted to the measured locator recall (log-attention + Gumbel noise).
--arm dense: identical data, steps, loss and teacher, no narrowing (the dense-FT control).
python g2train.py --arm nest --steps 1200 --tag a"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv
import g2lib as GL
from strands_decider.prompting import render_state

ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True); ap.add_argument('--steps', type=int, default=1200); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--r', type=int, default=16); ap.add_argument('--v5frac', type=float, default=0.3)
ap.add_argument('--maxtok', type=int, default=3072); ap.add_argument('--tag', default=''); ap.add_argument('--ckpts', type=int, default=3); ap.add_argument('--noise', type=float, default=1.0)
a = ap.parse_args(); random.seed(1); torch.manual_seed(1)
W = os.path.expanduser('~/work/g2/')
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
pool = []
with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV: continue
        if r['n_state_tok'] < 200: continue
        pool.append(dict(state=r['state'], questions=r['questions'], task=r['task'], hook=r['hook']))
v5 = []
with open(os.path.expanduser('~/work/training/data/train_v5.jsonl')) as f:
    for li, l in enumerate(f):
        if li % 25 == 0: v5.append(json.loads(l))
random.shuffle(v5); random.shuffle(pool)
print('pool', len(pool), 'v5', len(v5), flush=True)

g = GL.G2(); g.load_ranks(W + 'ranks.pt'); g.detach_inference()
eps = g.eps; dev = g.dev
for p_ in g.head.parameters(): p_.requires_grad_(False)


class LoRA(nn.Module):
    def __init__(self, out, inp, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, inp, device=dev) / math.sqrt(inp)); self.B = nn.Parameter(torch.zeros(out, r, device=dev))
    def forward(self, x, Wt):
        return x @ Wt.t() + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)


loras = nn.ModuleList()
for i in range(24):
    m = nn.ModuleDict({k: LoRA(g.L[i][k].shape[0], g.L[i][k].shape[1], a.r) for k in ('Win', 'Wo', 'Wgu', 'Wd')})
    loras.append(m)
params = list(loras.parameters())
print('lora params', sum(p.numel() for p in params) / 1e6, 'M', flush=True)


def layer(i, x, h, cos, sin, wm, wh):
    d = g.L[i]; lo = loras[i]; T = x.shape[0]
    proj = lo['Win'](h, d['Win'])
    if d['type'] == 'linear_attention':
        qkv = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; aa = proj[:, 8208:8224]
        beta = torch.sigmoid(b.float()); gg = -d['A_log'].float().exp() * F.softplus(aa.float() + d['dt_bias'])
        if wh is not None:
            qkv = (qkv.reshape(T, 3, 16, 128) * wh[:, None, :, None]).reshape(T, 6144); beta = beta * wh.float(); gg = gg * wh.float()
        qkv = fla_conv(qkv[None].contiguous(), d['conv_w'], None, activation='silu'); qkv = qkv[0] if isinstance(qkv, tuple) else qkv
        q, k, v = qkv.split(2048, dim=-1)
        o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), gg[None], beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
        of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
        o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(h.dtype).reshape(T, 2048)
        if wh is not None: o = (o.reshape(T, 16, 128) * wh[..., None]).reshape(T, 2048)
    else:
        qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
        kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
        qh = GL.rms_zc(qh, d['qn'], eps); kk = GL.rms_zc(kk, d['kn'], eps)
        def rope(t):
            xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
            c = cos[:, None, :]; s = sin[:, None, :]
            return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s[..., :32], x2 * c[..., 32:] + x1 * s[..., 32:]], -1), xp], -1)
        qh, kk = rope(qh), rope(kk)
        o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
        o = o[0].transpose(0, 1) * torch.sigmoid(gate)
        if wh is not None: o = o * wh[..., None]
        o = o.reshape(T, 2048)
    x = x + lo['Wo'](o, d['Wo'])
    h2 = GL.rms_zc(x, d['post_norm'], eps)
    gu = lo['Wgu'](h2, d['Wgu']); I = d['I']
    m = F.silu(gu[:, :I]) * gu[:, I:]
    if wm is not None: m = m * wm
    x = x + lo['Wd'](m, d['Wd'])
    nw = g.L[i + 1]['in_norm'] if i + 1 < 24 else g.norm_w
    return x, GL.rms_zc(x, nw, eps)


def student(ids, keep):   # keep: [T] width fraction per row (1 = full) or None
    T = len(ids); ids_t = torch.tensor(ids, device=dev)
    x = F.embedding(ids_t, g.embed)
    pos = torch.arange(T, device=dev, dtype=torch.float32); fr = pos[:, None] * g.inv[None, :]; fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
    h = GL.rms_zc(x, g.L[0]['in_norm'], eps)
    for i in range(24):
        wm = wh = None
        if keep is not None:
            d = g.L[i]; nh = 16 if d['type'] == 'linear_attention' else 8
            wm = (g.rank_mlp[i][None, :].float() < keep[:, None] * d['I']).to(x.dtype)
            rk = (g.rank_gdn if d['type'] == 'linear_attention' else g.rank_att)[g.tidx[i]]
            wh = (rk[None, :].float() < keep[:, None] * nh).to(x.dtype)
        x, h = checkpoint(layer, i, x, h, cos, sin, wm, wh, use_reentrant=False)
    return h


def logits_of(h, pr):
    T = h.shape[0]; opt_abs = [pr['q0'] + o for o in pr['opt']]
    lg = g.head(h[T - 1].float()[None], h[torch.tensor(opt_abs, device=dev)].float()[None])[0] / g.p.temp_for(pr['rq'].kind)
    return lg[:pr['rq'].n_slots]


def fit(state, qd):
    pr = g.prep(state, qd)
    if len(pr['s']) > a.maxtok:   # keep the head and the tail of a long state (the tail holds the newest turn)
        s = pr['s']; pr['s'] = s[:a.maxtok // 4] + s[-(a.maxtok - a.maxtok // 4):]; pr['q0'] = len(pr['s'])
    return pr


opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
nupd = max(1, a.steps // a.accum); warm = max(1, nupd // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * max(0.05, 1 - u / nupd))
save_at = set(int(a.steps * f) for f in np.linspace(0, 1, a.ckpts + 1)[1:])
name = f'{a.arm}{("_" + a.tag) if a.tag else ""}'
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time(); log = open(W + f'train_{name}.log.jsonl', 'w')
pi = vi = 0
for step in range(1, a.steps + 1):
    use_v5 = random.random() < a.v5frac
    if use_v5:
        r = v5[vi % len(v5)]; vi += 1
        qd = {'type': 'choice', 'instructions': random.choice([r['instructions']] + r.get('instruction_variants', [])[:2]), 'criteria': {n: d for n, d in r['options']}}
        pr = fit(r['state'], qd); gold = r['label']
    else:
        r = pool[pi % len(pool)]; pi += 1
        qn = random.choice(list(r['questions'])); pr = fit(r['state'], r['questions'][qn]); gold = None
    ids = pr['s'] + pr['q']; T = len(ids); q0 = pr['q0']
    with torch.no_grad():
        ht, caps = g.forward(ids, None, capture=(7,), q0=q0)
        tp = torch.softmax(logits_of(ht, pr).float(), -1)
        sig = caps[7]
    keep = None; f_full = w = 1.0
    if a.arm == 'nest' and q0 > 8:
        f_full = random.choice([0.10, 0.15, 0.20, 0.30]); w = random.choice([0.125, 0.25, 0.5])
        s = sig.clamp_min(1e-12).log()
        if random.random() < 0.5:   # emulate locator errors
            s = s + a.noise * (-torch.log(-torch.log(torch.rand_like(s).clamp(1e-9, 1 - 1e-9))))
        cls = GL.assign(s, q0, T, [(0, f_full), (1, 1 - f_full)])
        keep = torch.where(cls == 0, torch.ones(T, device=dev), torch.full((T,), w, device=dev))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hs = student(ids, keep)
        lg = logits_of(hs, pr).float()
    lp = F.log_softmax(lg, -1)
    kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
    if gold is not None:
        loss = 0.5 * F.nll_loss(lp[None], torch.tensor([gold], device=dev)) + 0.5 * kl; kind = 'v5'
    else:
        loss = kl; kind = 'kl'
    (loss / a.accum).backward(); lsum[kind] += float(kl); lcnt[kind] += 1
    if step % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0:
        rec = dict(step=step, t=round(time.time() - t0), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), **{k: lsum[k] / max(1, lcnt[k]) for k in lsum})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if step in save_at:
        torch.save({f'{i}.{k}.{ab}': getattr(loras[i][k], ab).detach().cpu() for i in range(24) for k in ('Win', 'Wo', 'Wgu', 'Wd') for ab in ('A', 'B')},
                   W + f'lora_{name}_s{step}.pt')
        print('saved', step, flush=True)
print('done', time.time() - t0, flush=True)
