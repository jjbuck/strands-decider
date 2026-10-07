"""H1 QAT: KL-to-bf16-hobson distillation of the UNIFORM low-bit model (h1lib numerics), LoRA in the ROTATED weight space + STE.
   W''_eff = W''_0 + B A    (W''_0 = R1^T-side/in-side rotated, gain-folded weight exactly as the int runtime stores it; for 4-bit GEMMs W''_0 is the
                              GPTQ-dequantized weight when --gptq, so step 0 == the GPTQ model)
   forward: y = Qa(R_in x) @ Qw(W''_eff)^T  (STE rounding, per-token act scales, FIXED per-channel weight scales from init), bf16 GEMMs: no quant.
   loss = KL(teacher || student) on the answer distribution + --hid * relative residual MSE at layers 5/11/17/23 + --gold * CE(gold) on train_v5 rows.
   Data: evalkit/train_pool.jsonl (eval tasks excluded), all questions of a request rotate; train_v5 rows (fraction --v5frac).
   Saves the rotated-space LoRA {i.k.A, i.k.B} -> evaluated with h1eval 'lorarot=PATH'.
python h1qat.py --base w4a4 --gptq --map precmap.json --steps 3000 --tag x"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv
import h1lib as HL
from g2lib import rms_zc

ap = argparse.ArgumentParser()
ap.add_argument('--base', default='w4a4'); ap.add_argument('--map', default=None, help='json {"i.k": prec} overrides'); ap.add_argument('--gptq', action='store_true'); ap.add_argument('--gptq8', action='store_true')
ap.add_argument('--ba16', action='store_true'); ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--r', type=int, default=32); ap.add_argument('--v5frac', type=float, default=0.25)
ap.add_argument('--maxtok', type=int, default=2048); ap.add_argument('--tag', default=''); ap.add_argument('--ckpts', type=int, default=6)
ap.add_argument('--hid', type=float, default=0.1); ap.add_argument('--gold', type=float, default=0.0); ap.add_argument('--resume', default=None)
ap.add_argument('--tkd', type=float, default=1.0, help='distillation temperature on top of the head temperature')
a = ap.parse_args(); random.seed(3); torch.manual_seed(3)
W = os.path.expanduser('~/work/h1/')
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
        if li % 20 == 0: v5.append(json.loads(l))
random.shuffle(v5); random.shuffle(pool)
print('pool', len(pool), 'v5', len(v5), flush=True)

g = HL.H1(); g.detach_inference(); eps = g.eps; dev = g.dev
for p_ in g.head.parameters(): p_.requires_grad_(False)
g.opt['ba16'] = a.ba16
if a.gptq or a.gptq8: g.opt['wq'] = 'gptq'
if a.gptq8: g.opt['wq8'] = 'gptq'
P = g.uniform(a.base)
if a.map:
    for key, v in json.load(open(a.map)).items(): P[(int(key.split('.')[0]), key.split('.')[1])] = v
g.set_prec(P)
HID = (5, 11, 17, 23)

# frozen rotated base weights W''_0 and fixed scales
W0 = {}; SC = {}; QM = {}
t0 = time.time()
for i in range(24):
    for k in HL.GEMMS:
        prec = P.get((i, k), 'bf16')
        if prec == 'bf16': continue
        wb, ab = HL.bits(prec)
        q, s = g.qweight(i, k, prec)                    # RTN-clip or GPTQ codes in the rotated space (same rule as h1lib.qweight)
        gq = g.opt['wq'] == 'gptq' and (wb == 4 or g.opt.get('wq8') == 'gptq')
        W0[(i, k)] = (q if gq else g.wfold(i, k).to(torch.bfloat16), s, gq); QM[(i, k)] = (HL.QMAX[wb], HL.QMAX[ab])
        g.Qw = {}
print(f'quantized base in {time.time()-t0:.0f}s, mem {torch.cuda.memory_allocated()/1e9:.1f}G', flush=True)


class LoRA(nn.Module):
    def __init__(self, out, inp, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, inp, device=dev) / math.sqrt(inp)); self.B = nn.Parameter(torch.zeros(out, r, device=dev))


def nout(i, k):
    if k == 'Win' and a.ba16 and g.L[i]['type'] == 'linear_attention': return 8192
    return g.L[i][k].shape[0]


loras = nn.ModuleList([nn.ModuleDict({k: LoRA(nout(i, k), g.L[i][k].shape[1], a.r) for k in HL.GEMMS}) for i in range(24)])
if a.resume:
    sd = torch.load(a.resume, map_location=dev)
    for i in range(24):
        for k in HL.GEMMS:
            loras[i][k].A.data.copy_(sd[f'{i}.{k}.A']); loras[i][k].B.data.copy_(sd[f'{i}.{k}.B'])
params = list(loras.parameters())


def ste(x): return (torch.round(x) - x).detach() + x


def qlinear(x, xn, i, k):
    """x: dense bf16 GEMM input; xn: fp32 unweighted normed residual (Win/Wgu)."""
    d = g.L[i]; lo = loras[i][k]; prec = P.get((i, k), 'bf16')
    src = xn if k in ('Win', 'Wgu') else x
    R = g.rot_for(i, k)
    if prec == 'bf16':                                   # bf16 GEMM + LoRA (LoRA in the rotated input basis)
        Wd_ = d[k][:nout(i, k)]
        yl = (R(src.float()) @ lo.A.float().t()) @ lo.B.float().t()
        if k == 'Win' and a.ba16 and d['type'] == 'linear_attention': yl = torch.cat([yl, torch.zeros(yl.shape[0], 32, device=dev)], 1)
        y = (x @ d[k].t()).float() + yl
        return y.to(torch.bfloat16)
    else:
        q, s, gq = W0[(i, k)]; qw, qa = QM[(i, k)]
        Wr = (q.float() * s[:, None] if gq else q.float()) + lo.B.float() @ lo.A.float()
        cw = ste(Wr / s[:, None]).clamp(-qw, qw)                       # integer-valued codes (exact in bf16), STE
        xr = R(src.float())
        sa = (xr.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qa * (g.opt['aclip4'] if qa < 100 else 1.0)).detach()
        cx = ste(xr / sa).clamp(-qa, qa)
        y = (cx.to(torch.bfloat16) @ cw.to(torch.bfloat16).t()).float() * sa * s[None, :]
        if k in ('Wo', 'Wd') and g.opt['rout']: y = g.rots()['R1'].inv(y)
    y = y.to(torch.bfloat16)
    if k == 'Win' and a.ba16 and d['type'] == 'linear_attention':
        y = torch.cat([y, x @ d['Win'][8192:].t()], 1)
    return y


def layer(i, x, cos, sin):
    d = g.L[i]; T = x.shape[0]
    xf = x.float(); xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps); h = (xn * d['in1']).to(x.dtype)
    proj = qlinear(h, xn, i, 'Win')
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
        qh = rms_zc(qh, d['qn'], eps); kk = rms_zc(kk, d['kn'], eps)
        def rope(t):
            xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
            c = cos[:, None, :]; s_ = sin[:, None, :]
            return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
        qh, kk = rope(qh), rope(kk)
        o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
        o = (o[0].transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
    x = x + qlinear(o, None, i, 'Wo')
    xf = x.float(); xn2 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps); h2 = (xn2 * d['post1']).to(x.dtype)
    gu = qlinear(h2, xn2, i, 'Wgu'); I = d['I']
    m = F.silu(gu[:, :I]) * gu[:, I:]
    x = x + qlinear(m, None, i, 'Wd')
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
    return rms_zc(x, g.norm_w, eps), hx


def logits_of(h, pr):
    T = h.shape[0]; opt_abs = [pr['q0'] + o for o in pr['opt']]
    lg = g.head(h[T - 1].float()[None], h[torch.tensor(opt_abs, device=dev)].float()[None])[0] / g.p.temp_for(pr['rq'].kind)
    return lg[:pr['rq'].n_slots]


def fit(state, qd):
    pr = g.prep(state, qd)
    if len(pr['s']) > a.maxtok:
        s = pr['s']; pr['s'] = s[:a.maxtok // 4] + s[-(a.maxtok - a.maxtok // 4):]; pr['q0'] = len(pr['s'])
    return pr


opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
nupd = max(1, a.steps // a.accum); warm = max(1, nupd // 30)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / nupd)))))
save_at = set(int(a.steps * f) for f in np.linspace(0, 1, a.ckpts + 1)[1:])
name = f'qat_{a.base}' + (('_' + a.tag) if a.tag else '')
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time(); log = open(W + f'res/train_{name}.log.jsonl', 'a')
pi = vi = 0
g.set_prec({})        # teacher = dense bf16 (h1lib fwd with empty map)
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
        ht, caps = g.fwd(ids, keep_x=HID)
        tl = logits_of(ht, pr).float()
        tp = torch.softmax(tl / a.tkd, -1)
    hs, hx = student(ids)                     # no autocast: bf16 where the dense model is bf16, fp32 rotations / scales
    lg = logits_of(hs, pr).float()
    lp = F.log_softmax(lg / a.tkd, -1)
    kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum() * a.tkd ** 2
    hl = sum(((hx[i].float() - caps[i].float()) ** 2).sum() / caps[i].float().pow(2).sum() for i in HID) / len(HID)
    loss = kl + a.hid * hl
    if gold is not None and a.gold > 0: loss = loss + a.gold * F.nll_loss(F.log_softmax(lg, -1)[None], torch.tensor([gold], device=dev))
    (loss / a.accum).backward()
    flip = int(lg.argmax() != tl.argmax())
    k_ = 'v5' if gold is not None else 'kl'
    lsum[k_] += float(kl); lcnt[k_] += 1; lsum['hid'] += float(hl); lcnt['hid'] += 1; lsum['flip'] += flip; lcnt['flip'] += 1
    if step % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0 or step == 1:
        rec = dict(step=step, t=round(time.time() - t0), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), lr=sched.get_last_lr()[0], **{k: lsum[k] / max(1, lcnt[k]) for k in lsum})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if step in save_at:
        torch.save({f'{i}.{k}.{ab}': getattr(loras[i][k], ab).detach().cpu() for i in range(24) for k in HL.GEMMS for ab in ('A', 'B')} | dict(_P={f'{i}.{k}': v for (i, k), v in P.items()}, _opt=dict(g.opt)),
                   W + f'lora_{name}_s{step}.pt')
        print('saved', step, flush=True)
print('done', time.time() - t0, flush=True)
