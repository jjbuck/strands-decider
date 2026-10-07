"""G2 W4A4 quantization-aware distillation (rotated, QuaRot-style; KL to bf16 hobson; recipe after BitDistill 2510.13998, simplified).
Student: hobson + LoRA r16 on all 4 projection groups of 24 layers; EVERY linear runs as fake-quant W4A4 with straight-through rounding:
   y = Qa(R x_in) @ Qw(R-rotated (W + B A) [diag(1+w) folded for Win / Wgu])^T, per-token absmax activations (clip --aclip), per-channel weights
   (clip ratio fixed per channel from an MSE search at init), R = randomized Kronecker Hadamard (as in g2lib: 2048 = H32 x H64, 6144 = H12 x H16 x H32).
   Since (W + BA) R = WR + B (AR), WR is precomputed and only AR is rotated per step.
Teacher: frozen bf16 hobson (same weights, no LoRA). Loss: KL on the answer distribution + --hid * relative MSE of residuals at layers 5, 11, 17
   (all rows) ; train_v5 rows add 0.5 CE on gold.  Data: train-split real states (evalkit/train_pool, eval tasks excluded) + train_v5.
python g2qat.py --steps 1200 --tag a"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv
import g2lib as GL

ap = argparse.ArgumentParser(); ap.add_argument('--steps', type=int, default=1200); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--r', type=int, default=16); ap.add_argument('--v5frac', type=float, default=0.3)
ap.add_argument('--maxtok', type=int, default=2048); ap.add_argument('--tag', default=''); ap.add_argument('--ckpts', type=int, default=3)
ap.add_argument('--aclip', type=float, default=0.9); ap.add_argument('--hid', type=float, default=1.0); ap.add_argument('--wbits', type=int, default=4); ap.add_argument('--abits', type=int, default=4)
ap.add_argument('--gold', type=float, default=0.0, help='weight of gold CE on train_v5 rows (0 = pure self-distillation; train_v5 states still used)')
ap.add_argument('--noquant', action='store_true', help='dense-FT control: identical recipe without fake quantization')
a = ap.parse_args(); random.seed(2); torch.manual_seed(2)
W = os.path.expanduser('~/work/g2/')
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
pool = []
with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 200: continue
        pool.append(dict(state=r['state'], questions=r['questions']))
v5 = []
with open(os.path.expanduser('~/work/training/data/train_v5.jsonl')) as f:
    for li, l in enumerate(f):
        if li % 25 == 0: v5.append(json.loads(l))
random.shuffle(v5); random.shuffle(pool)
print('pool', len(pool), 'v5', len(v5), flush=True)

g = GL.G2(); g.detach_inference(); eps = g.eps; dev = g.dev
for p_ in g.head.parameters(): p_.requires_grad_(False)
qw = 7.0 if a.wbits == 4 else 127.0; qa = 7.0 if a.abits == 4 else 127.0
HID = (5, 11, 17)

# rotated (folded) weights + per-channel clip ratios
WR = []; CLIP = []
t0 = time.time()
for i in range(24):
    d = g.L[i]; wr = {}; cr = {}
    for k in ('Win', 'Wo', 'Wgu', 'Wd'):
        Wf = d[k].float()
        if k == 'Win': Wf = Wf * d['in1'][None, :]
        if k == 'Wgu': Wf = Wf * d['post1'][None, :]
        Wr = g.rot(Wf)
        s = Wr.abs().amax(1).clamp_min(1e-8) / qw; best_e = None; best_r = torch.ones_like(s)
        for rr in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
            e = ((torch.round(Wr / (s * rr)[:, None]).clamp(-qw, qw) * (s * rr)[:, None] - Wr) ** 2).sum(1)
            if best_e is None: best_e = e
            else:
                m = e < best_e; best_e = torch.where(m, e, best_e); best_r = torch.where(m, torch.full_like(best_r, rr), best_r)
        wr[k] = Wr.to(torch.bfloat16).contiguous(); cr[k] = best_r.contiguous()
        del Wf, Wr
    WR.append(wr); CLIP.append(cr)
print(f'rotated weights in {time.time()-t0:.0f}s, mem {torch.cuda.memory_allocated()/1e9:.1f}G', flush=True)


def ste(x): return (torch.round(x) - x).detach() + x


def fq_act(x):
    xf = x.float(); s = (xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qa * (a.aclip if a.abits == 4 else 1.0)).detach()
    return (ste(xf / s).clamp(-qa, qa) * s)


class LoRA(nn.Module):
    def __init__(self, out, inp, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, inp, device=dev) / math.sqrt(inp)); self.B = nn.Parameter(torch.zeros(out, r, device=dev))


loras = nn.ModuleList()
for i in range(24):
    loras.append(nn.ModuleDict({k: LoRA(g.L[i][k].shape[0], g.L[i][k].shape[1], a.r) for k in ('Win', 'Wo', 'Wgu', 'Wd')}))
params = list(loras.parameters())


def qlinear(xin, i, k):
    lo = loras[i][k]
    if a.noquant:
        Wt = g.L[i][k]
        if k == 'Win': xin = xin * g.L[i]['in1'].to(xin.dtype)
        if k == 'Wgu': xin = xin * g.L[i]['post1'].to(xin.dtype)
        return xin @ Wt.t() + (xin @ lo.A.t().to(xin.dtype)) @ lo.B.t().to(xin.dtype)
    fold = g.L[i]['in1'] if k == 'Win' else g.L[i]['post1'] if k == 'Wgu' else None
    Af = lo.A.float() * fold[None, :] if fold is not None else lo.A.float()     # (W + BA) diag(f) = W diag(f) + B (A diag(f))
    Wr = WR[i][k].float() + lo.B.float() @ g.rot(Af)                          # (W diag f + B A diag f) R
    s = (Wr.abs().amax(1, keepdim=True).clamp_min(1e-8) / qw * CLIP[i][k][:, None]).detach()
    Wq = ste(Wr / s).clamp(-qw, qw) * s
    xr = g.rot_act(xin)                                                        # fp16 online / offline rotation (differentiable)
    return (fq_act(xr).to(torch.bfloat16) @ Wq.to(torch.bfloat16).t())


def layer(i, x, cos, sin):
    d = g.L[i]; T = x.shape[0]
    xn = GL.nrm(x, eps)
    proj = qlinear(xn, i, 'Win')
    if d['type'] == 'linear_attention':
        qkv = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; aa = proj[:, 8208:8224]
        beta = torch.sigmoid(b.float()); gg = -d['A_log'].float().exp() * F.softplus(aa.float() + d['dt_bias'])
        qkv = fla_conv(qkv[None].contiguous(), d['conv_w'], None, activation='silu'); qkv = qkv[0] if isinstance(qkv, tuple) else qkv
        q, k, v = qkv.split(2048, dim=-1)
        o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), gg[None], beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
        of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
        o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(torch.bfloat16).reshape(T, 2048)
    else:
        qg = proj[:, :4096].reshape(T, 8, 512); qh, gate = qg[..., :256], qg[..., 256:]
        kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
        qh = GL.rms_zc(qh, d['qn'], eps); kk = GL.rms_zc(kk, d['kn'], eps)
        def rope(t):
            xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
            c = cos[:, None, :]; s_ = sin[:, None, :]
            return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
        qh, kk = rope(qh), rope(kk)
        o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
        o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
    x = x + qlinear(o, i, 'Wo')
    xn2 = GL.nrm(x, eps)
    gu = qlinear(xn2, i, 'Wgu'); I = d['I']
    m = F.silu(gu[:, :I]) * gu[:, I:]
    x = x + qlinear(m, i, 'Wd')
    return x


def student(ids):
    T = len(ids); ids_t = torch.tensor(ids, device=dev)
    x = F.embedding(ids_t, g.embed)
    pos = torch.arange(T, device=dev, dtype=torch.float32); fr = pos[:, None] * g.inv[None, :]; fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
    hx = {}
    for i in range(24):
        x = checkpoint(layer, i, x, cos, sin, use_reentrant=False)
        if i in HID: hx[i] = x
    return GL.rms_zc(x, g.norm_w, eps), hx


def logits_of(h, pr):
    T = h.shape[0]; opt_abs = [pr['q0'] + o for o in pr['opt']]
    lg = g.head(h[T - 1].float()[None], h[torch.tensor(opt_abs, device=dev)].float()[None])[0] / g.p.temp_for(pr['rq'].kind)
    return lg[:pr['rq'].n_slots]


def fit(state, qd):
    pr = g.prep(state, qd)
    if len(pr['s']) > a.maxtok:
        s = pr['s']; pr['s'] = s[:a.maxtok // 4] + s[-(a.maxtok - a.maxtok // 4):]; pr['q0'] = len(pr['s'])
    return pr


opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
nupd = max(1, a.steps // a.accum); warm = max(1, nupd // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * max(0.05, 1 - u / nupd))
save_at = set(int(a.steps * f) for f in np.linspace(0, 1, a.ckpts + 1)[1:])
name = ('dft' if a.noquant else f'qat{a.wbits}{a.abits}') + (('_' + a.tag) if a.tag else '')
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time(); log = open(W + f'train_{name}.log.jsonl', 'w')
pi = vi = 0
for step in range(1, a.steps + 1):
    if random.random() < a.v5frac:
        r = v5[vi % len(v5)]; vi += 1
        qd = {'type': 'choice', 'instructions': random.choice([r['instructions']] + r.get('instruction_variants', [])[:2]), 'criteria': {n: d_ for n, d_ in r['options']}}
        pr = fit(r['state'], qd); gold = r['label']
    else:
        r = pool[pi % len(pool)]; pi += 1
        qn = random.choice(list(r['questions'])); pr = fit(r['state'], r['questions'][qn]); gold = None
    ids = pr['s'] + pr['q']
    with torch.no_grad():
        ht, caps = g.forward(ids, None, keep_x=HID)
        tp = torch.softmax(logits_of(ht, pr).float(), -1)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hs, hx = student(ids)
        lg = logits_of(hs, pr).float()
    lp = F.log_softmax(lg, -1)
    kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
    hl = sum(((hx[i].float() - caps[('x', i)].float()) ** 2).sum() / caps[('x', i)].float().pow(2).sum() for i in HID) / len(HID)
    loss = kl + a.hid * hl
    if gold is not None and a.gold > 0: loss = loss + a.gold * F.nll_loss(lp[None], torch.tensor([gold], device=dev))
    (loss / a.accum).backward()
    k_ = 'v5' if gold is not None else 'kl'
    lsum[k_] += float(kl); lcnt[k_] += 1; lsum['hid'] += float(hl); lcnt['hid'] += 1
    if step % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 25 == 0 or step == 1:
        rec = dict(step=step, t=round(time.time() - t0), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), T=len(ids), **{k: lsum[k] / max(1, lcnt[k]) for k in lsum})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if step in save_at:
        torch.save({f'{i}.{k}.{ab}': getattr(loras[i][k], ab).detach().cpu() for i in range(24) for k in ('Win', 'Wo', 'Wgu', 'Wd') for ab in ('A', 'B')},
                   W + f'lora_{name}_s{step}.pt')
        print('saved', step, flush=True)
print('done', time.time() - t0, flush=True)
