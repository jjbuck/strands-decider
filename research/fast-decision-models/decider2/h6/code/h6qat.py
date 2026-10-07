"""H6 QAT + bf16 control, schema-first hobson-v19, trained on the SAME examples in the SAME order (one process, two LoRA sets).
Layout per example: [q[:-3] bundle][state][q[-3:] '<answer>' slot]; decision = last row; options = bundle rows.
  ctl : every GEMM bf16 (+ LoRA)                                                         -> the bf16 schema-first control
  qat : precision map (default H1 k48: 48 GEMMs W8A8, 48 W4A4, GPTQ codes from ~/work/h2/codes_gptq_w{4,8}.pt = the deployed codes),
        mixed rows: bundle + slot rows bf16 (W'' + BA), STATE rows low-bit (codes = round((q s + B A)/s), H1's rule; scales fixed).
Quantized GEMM = exact H1/H2 quantizers (per-token absmax act codes, clip 0.9 for int4; per-channel weight codes) with STE; a custom autograd
function computes the LoRA gradients in low rank (dB = u^T (cx A^T), dA = (u B)^T cx, u = dy * s_act) and dx = dy @ (codes * s_w), so no [N,K]
weight gradient is materialized; codes are recomputed once per optimizer update.
Loss: KL(teacher || student) on the answer distribution (teacher = bf16 hobson STATE-FIRST, h6teach.py) + GOLD * CE on train_v5 gold rows.
python h6qat.py --steps 3600 --accum 4 --lr 1.5e-4 --tag a"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv
import h1lib as HL
from g2lib import rms_zc

ap = argparse.ArgumentParser()
ap.add_argument('--map', default='~/work/h2/precmap_w4a4_k48.json'); ap.add_argument('--dflt', default='w4a4')
ap.add_argument('--codes4', default='~/work/h2/codes_gptq_w4.pt'); ap.add_argument('--codes8', default='~/work/h2/codes_gptq_w8.pt')
ap.add_argument('--steps', type=int, default=3600); ap.add_argument('--accum', type=int, default=4); ap.add_argument('--lr', type=float, default=1.5e-4)
ap.add_argument('--r', type=int, default=32); ap.add_argument('--gold', type=float, default=0.1); ap.add_argument('--tag', default='a')
ap.add_argument('--every', type=int, default=300); ap.add_argument('--arms', default='ctl,qat'); ap.add_argument('--resume', default='')
ap.add_argument('--qstate_only', type=int, default=1, help='1: bundle+slot rows bf16 (mixed rows); 0: every row low-bit')
ap.add_argument('--lr_p', type=float, default=1e-4, help='LoRA lr of the qatp arm (starts at a near-optimal point)')
ap.add_argument('--hid', type=float, default=1.0, help='qatq: weight of the residual-stream relative MSE to the dense teacher')
ap.add_argument('--head_lr', type=float, default=1e-3, help='0: head frozen; else the pointer head is trained (one copy per arm)')
a = ap.parse_args(); random.seed(5); torch.manual_seed(5)
W = os.path.expanduser('~/work/h6/')
data = torch.load(W + 'teach.pt')
print('examples', len(data), flush=True)

g = HL.H1(); g.detach_inference(); eps = g.eps; dev = g.dev
for p_ in g.head.parameters(): p_.requires_grad_(False)
g.head.eval()
ARMS = a.arms.split(',')
PM = {'ctl': {}}
QARMS = [x for x in ARMS if x.startswith('qat')]
if QARMS:
    mp = {(i, k): a.dflt for i in range(24) for k in HL.GEMMS}
    if a.map:
        for key, v in json.load(open(os.path.expanduser(a.map))).items(): mp[(int(key.split('.')[0]), key.split('.')[1])] = v
    for x in QARMS: PM[x] = {key: v for key, v in mp.items() if v != 'bf16'}
    C4 = torch.load(os.path.expanduser(a.codes4)); C8 = torch.load(os.path.expanduser(a.codes8))
    Q0 = {}
    for key, prec in PM[QARMS[0]].items():
        wb, ab = HL.bits(prec)
        q, s = (C4 if wb == 4 else C8)[key]
        Q0[key] = (q.to(dev).to(torch.int8), s.to(dev).float(), HL.QMAX[wb], HL.QMAX[ab])
    del C4, C8
    print('codes loaded', len(Q0), f'mem {torch.cuda.memory_allocated()/1e9:.1f}G', flush=True)


class LoRA(nn.Module):
    def __init__(self, out, inp, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(r, inp, device=dev) / math.sqrt(inp)); self.B = nn.Parameter(torch.zeros(out, r, device=dev))


LO = {arm: nn.ModuleList([nn.ModuleDict({k: LoRA(g.L[i][k].shape[0], g.L[i][k].shape[1], a.r) for k in HL.GEMMS}) for i in range(24)]) for arm in ARMS}
if a.resume:
    for arm in ARMS:
        sd = torch.load(a.resume.replace('ARM', arm), map_location=dev)
        for i in range(24):
            for k in HL.GEMMS:
                LO[arm][i][k].A.data.copy_(sd[f'{i}.{k}.A']); LO[arm][i][k].B.data.copy_(sd[f'{i}.{k}.B'])
CW = {x: {} for x in QARMS}
import copy
HEADS = {}
HLR = {arm: (0.0 if arm in ('qatp', 'qatq') else a.head_lr) for arm in ARMS}     # plain layout: hobson's head is already right
for arm in ARMS:
    h_ = copy.deepcopy(g.head).float().eval()
    for p_ in h_.parameters(): p_.requires_grad_(HLR[arm] > 0)
    HEADS[arm] = h_
if a.resume:
    for arm in ARMS:
        sd = torch.load(a.resume.replace('ARM', arm), map_location=dev)
        if '_head' in sd: HEADS[arm].load_state_dict(sd['_head'])


@torch.no_grad()
def refresh_codes():
    for arm in QARMS:
        for key, (q, s, qw, qa) in Q0.items():
            i, k = key; lo = LO[arm][i][k]
            lat = torch.addmm(q.float() * s[:, None], lo.B.float(), lo.A.float())
            CW[arm][key] = torch.round(lat / s[:, None]).clamp_(-qw, qw).to(torch.int8)
            del lat


class QLinF(torch.autograd.Function):
    """y = (Q_a(xr) @ cw^T) * s_a * s_w ; STE on both roundings, act scale detached, LoRA grads in low rank."""
    @staticmethod
    def forward(ctx, xr, A, B, cw, s, qa, clip):
        cwb = cw.to(torch.bfloat16)
        sa = xr.abs().amax(-1).clamp_min(1e-8) / qa * clip
        cx = torch.round(xr / sa[:, None]).clamp_(-qa, qa).to(torch.bfloat16)
        y = (cx @ cwb.t()).float() * sa[:, None] * s[None, :]
        ctx.save_for_backward(cx, sa, A, B, cw, s)
        return y

    @staticmethod
    def backward(ctx, dy):
        cx, sa, A, B, cw, s = ctx.saved_tensors
        dyf = dy.float()
        cw = cw.to(torch.bfloat16)
        dxr = ((dyf * s[None, :]).to(torch.bfloat16) @ cw).float()
        u = (dyf * sa[:, None]).to(torch.bfloat16)
        dB = (u.t() @ (cx @ A.to(torch.bfloat16).t())).float()
        dA = ((u @ B.to(torch.bfloat16)).t() @ cx).float()
        return dxr, dA, dB, None, None, None, None


def qlinear(x, xn, i, k, arm, P0, P1):
    """x: dense bf16 GEMM input; xn: fp32 unweighted normed residual (Win/Wgu). Rows [P0,P1) low-bit if the GEMM is quantized in this arm."""
    d = g.L[i]; lo = LO[arm][i][k] if arm != 'teacher' else None; prec = PM[arm].get((i, k), 'bf16') if arm != 'teacher' else 'bf16'
    src = xn if k in ('Win', 'Wgu') else x
    R = g.rot_for(i, k)

    def bf(xs, ss):
        if arm in ('qatq', 'teacher'): return xs @ d[k].t()          # exact hobson weights on bf16 rows
        y = (xs @ d[k].t()).float() + (R(ss.float()) @ lo.A.t()) @ lo.B.t()
        return y.to(torch.bfloat16)
    if prec == 'bf16' or P1 <= P0: return bf(x, src)
    q, s, qw, qa = Q0[(i, k)]
    xr = R(src[P0:P1].float())
    yq = QLinF.apply(xr, lo.A, lo.B, CW[arm][(i, k)], s, qa, g.opt['aclip4'] if qa < 100 else 1.0)
    if k in ('Wo', 'Wd'): yq = g.rots()['R1'].inv(yq)
    yq = yq.to(torch.bfloat16)
    T = x.shape[0]; parts = []
    if P0 > 0: parts.append(bf(x[:P0], src[:P0]))
    parts.append(yq)
    if P1 < T: parts.append(bf(x[P1:], src[P1:]))
    return torch.cat(parts, 0) if len(parts) > 1 else parts[0]


def layer(i, x, cos, sin, arm, P0, P1):
    d = g.L[i]; T = x.shape[0]
    xf = x.float(); xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps); h = (xn * d['in1']).to(x.dtype)
    proj = qlinear(h, xn, i, 'Win', arm, P0, P1)
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
    x = x + qlinear(o, None, i, 'Wo', arm, P0, P1)
    xf = x.float(); xn2 = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps); h2 = (xn2 * d['post1']).to(x.dtype)
    gu = qlinear(h2, xn2, i, 'Wgu', arm, P0, P1); I = d['I']
    m = F.silu(gu[:, :I]) * gu[:, I:]
    x = x + qlinear(m, None, i, 'Wd', arm, P0, P1)
    return x


HIDL = (5, 11, 17, 23)


def student(ids, arm, P0, P1, keep=False):
    T = len(ids); ids_t = torch.tensor(ids, device=dev)
    x = F.embedding(ids_t, g.embed)
    pos = torch.arange(T, device=dev, dtype=torch.float32); fr = pos[:, None] * g.inv[None, :]; fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
    hx = {}
    for i in range(24):
        if arm == 'teacher': x = layer(i, x, cos, sin, arm, P0, P1)
        else: x = checkpoint(layer, i, x, cos, sin, arm, P0, P1, use_reentrant=False)
        if keep and i in HIDL: hx[i] = x
    return (rms_zc(x, g.norm_w, eps), hx) if keep else rms_zc(x, g.norm_w, eps)


params = {arm: list(LO[arm].parameters()) + (list(HEADS[arm].parameters()) if HLR[arm] > 0 else []) for arm in ARMS}
opts = {arm: torch.optim.AdamW([dict(params=list(LO[arm].parameters()), lr=(a.lr_p if arm in ('qatp', 'qatq') else a.lr))] + ([dict(params=list(HEADS[arm].parameters()), lr=HLR[arm])] if HLR[arm] > 0 else []),
                               weight_decay=0.0, betas=(0.9, 0.95)) for arm in ARMS}
nupd = max(1, a.steps // a.accum); warm = max(1, nupd // 30)
lam = lambda u: min(1.0, (u + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, u / nupd))))
scheds = {arm: torch.optim.lr_scheduler.LambdaLR(opts[arm], lam) for arm in ARMS}
log = open(W + f'train_{a.tag}.log.jsonl', 'a')
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time()
if QARMS: refresh_codes()
order = list(range(len(data))); random.shuffle(order)
for step in range(1, a.steps + 1):
    ex = data[order[(step - 1) % len(order)]]
    q, s = ex['q'], ex['s']
    pre = q[:-3]; ids = pre + s + q[-3:]; Pn = len(pre); Ls = len(s)
    opt_rows = torch.tensor(ex['opt'], device=dev)
    tp = ex['tp'].to(dev).float(); n = ex['n']
    for arm in ARMS:
        if arm in ('qatp', 'qatq'):          # hobson's own state-first layout, question rows bf16, state rows low-bit
            ids_a = s + q; P0, P1 = 0, Ls; orows = opt_rows + Ls
        else:
            ids_a = ids; orows = opt_rows
            P0, P1 = (Pn, Pn + Ls) if (arm == 'qat' and a.qstate_only) else (0, len(ids))
        hl = None
        if arm == 'qatq':
            with torch.no_grad():
                ht, hxt = student(ids_a, 'teacher', 0, 0, keep=True)
                tl_ = (HEADS[arm](ht[-1].float()[None], ht[orows].float()[None])[0] / ex['temp'])[:n].float()
                tpa = torch.softmax(tl_, -1)
            hs, hx = student(ids_a, arm, P0, P1, keep=True)
            hl = sum(((hx[i].float() - hxt[i].float()) ** 2).sum() / hxt[i].float().pow(2).sum() for i in HIDL) / len(HIDL)
        else:
            tpa = tp
            hs = student(ids_a, arm, P0, P1)
        lg = (HEADS[arm](hs[-1].float()[None], hs[orows].float()[None])[0] / ex['temp'])[:n].float()
        lp = F.log_softmax(lg, -1)
        kl = (tpa * (tpa.clamp_min(1e-8).log() - lp)).sum()
        loss = kl + (a.hid * hl if hl is not None else 0.0)
        if hl is not None: lsum[f'{arm}_hid'] += float(hl.detach()); lcnt[f'{arm}_hid'] += 1
        if ex['gold'] is not None and a.gold > 0: loss = loss + a.gold * F.nll_loss(lp[None], torch.tensor([ex['gold']], device=dev))
        (loss / a.accum).backward()
        lsum[f'{arm}_kl'] += float(kl.detach()); lcnt[f'{arm}_kl'] += 1
        lsum[f'{arm}_flip'] += int(lg.argmax() != tpa.argmax()); lcnt[f'{arm}_flip'] += 1
    if step % a.accum == 0:
        for arm in ARMS:
            torch.nn.utils.clip_grad_norm_(params[arm], 1.0); opts[arm].step(); scheds[arm].step(); opts[arm].zero_grad(set_to_none=True)
        if QARMS: refresh_codes()
    if step % 50 == 0 or step == 1:
        rec = dict(step=step, t=round(time.time() - t0), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), lr=scheds[ARMS[0]].get_last_lr()[0],
                   **{k: round(lsum[k] / max(1, lcnt[k]), 5) for k in lsum})
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if step % a.every == 0 or step == a.steps:
        for arm in ARMS:
            sd = {f'{i}.{k}.{ab}': getattr(LO[arm][i][k], ab).detach().cpu() for i in range(24) for k in HL.GEMMS for ab in ('A', 'B')}
            sd['_P'] = {f'{i}.{k}': v for (i, k), v in PM[arm].items()}
            sd['_lora_rows'] = 'state' if arm == 'qatq' else 'all'
            sd['_head'] = {k_: v.detach().cpu() for k_, v in HEADS[arm].state_dict().items()}
            torch.save(sd, W + f'lora_{a.tag}_{arm}_s{step}.pt')
        print('saved', step, flush=True)
print('done', time.time() - t0, flush=True)
