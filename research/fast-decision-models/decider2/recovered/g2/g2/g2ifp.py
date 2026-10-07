"""G2 per-question-spec structured sparsity (IFPruning-style, compiled per deployment question), training-free.
For each question spec (domain, hook, question) with >= 60 REAL-agree eval questions: decision-loss first-order Taylor scores of every MLP neuron in
layers 0-11 (L = -log p(own dense argmax); score_j = sum over calibration examples of |sum_t a_tj dL/da_tj|) on N train-split states of that spec.
Mask = keep the top --keep of neurons per layer (layers 0-11), per spec; GLOBAL mask = the same from the scores summed over all specs (equal weight).
Eval: the spec's REAL-agree questions under dense / own-spec mask / global mask / another spec's mask -> preds for evalkit.
python g2ifp.py --ncal 32 --keep 0.5 --out res/ifp.json"""
import os, sys, json, time, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.modules.convolution import causal_conv1d as fla_conv
import g2lib as GL
import evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('--ncal', type=int, default=32); ap.add_argument('--keep', type=float, default=0.5)
ap.add_argument('--layers', type=int, default=12); ap.add_argument('--out', required=True); ap.add_argument('--maxtok', type=int, default=3072)
a = ap.parse_args(); random.seed(3)
W = os.path.expanduser('~/work/g2/')
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
ra = EK.load_suite('REAL-agree')
cnt = collections.Counter((it['domain'], it['hook'], q) for it in ra for q in it['questions'])
SPECS = [k for k, v in cnt.most_common() if v >= 60]
print('specs', SPECS, flush=True)
cal = collections.defaultdict(list)
with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
    rows = [json.loads(l) for l in f]
random.shuffle(rows)
for r in rows:
    if r['task'] in EV or r['n_state_tok'] < 200: continue
    for q, qd in r['questions'].items():
        k = (r['domain'], r['hook'], q)
        if k in SPECS and len(cal[k]) < a.ncal: cal[k].append((r['state'], qd))
del rows
print({str(k): len(v) for k, v in cal.items()}, flush=True)

g = GL.G2(); g.detach_inference(); eps = g.eps; dev = g.dev
for p_ in g.head.parameters(): p_.requires_grad_(False)
NL = a.layers


def layer(i, x, cos, sin, gate):
    d = g.L[i]; T = x.shape[0]
    h = GL.rms_zc(x, d['in_norm'], eps)
    proj = h @ d['Win'].t()
    if d['type'] == 'linear_attention':
        qkv = proj[:, :6144]; z = proj[:, 6144:8192]; b = proj[:, 8192:8208]; aa = proj[:, 8208:8224]
        beta = torch.sigmoid(b.float()); gg = -d['A_log'].float().exp() * F.softplus(aa.float() + d['dt_bias'])
        qkv = fla_conv(qkv[None].contiguous(), d['conv_w'], None, activation='silu'); qkv = qkv[0] if isinstance(qkv, tuple) else qkv
        q, k, v = qkv.split(2048, dim=-1)
        o, _ = chunk_gated_delta_rule(q.reshape(1, T, 16, 128), k.reshape(1, T, 16, 128), v.reshape(1, T, 16, 128), gg[None], beta[None].to(q.dtype), use_qk_l2norm_in_kernel=True)
        of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
        o = ((d['gn_w'] * of.to(o.dtype)).float() * F.silu(z.reshape(-1, 128).float())).to(torch.bfloat16).reshape(T, 2048)
    else:
        qg = proj[:, :4096].reshape(T, 8, 512); qh, gt = qg[..., :256], qg[..., 256:]
        kk = proj[:, 4096:4608].reshape(T, 2, 256); v = proj[:, 4608:5120].reshape(T, 2, 256)
        qh = GL.rms_zc(qh, d['qn'], eps); kk = GL.rms_zc(kk, d['kn'], eps)
        def rope(t):
            xr, xp = t[..., :64], t[..., 64:]; x1, x2 = xr[..., :32], xr[..., 32:]
            c = cos[:, None, :]; s_ = sin[:, None, :]
            return torch.cat([torch.cat([x1 * c[..., :32] - x2 * s_[..., :32], x2 * c[..., 32:] + x1 * s_[..., 32:]], -1), xp], -1)
        qh, kk = rope(qh), rope(kk)
        o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kk.transpose(0, 1)[None], v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
        o = (o[0].transpose(0, 1) * torch.sigmoid(gt)).reshape(T, 2048)
    x = x + o @ d['Wo'].t()
    h2 = GL.rms_zc(x, d['post_norm'], eps)
    gu = h2 @ d['Wgu'].t(); I = d['I']
    m = F.silu(gu[:, :I]) * gu[:, I:]
    if gate is not None: m = m * gate.to(m.dtype)[None, :]
    return x + m @ d['Wd'].t()


def taylor(pr):
    ids = pr['s'] + pr['q']; T = len(ids)
    gates = [torch.ones(g.L[i]['I'], device=dev, requires_grad=True) for i in range(NL)]
    ids_t = torch.tensor(ids, device=dev)
    x = F.embedding(ids_t, g.embed)
    pos = torch.arange(T, device=dev, dtype=torch.float32); fr = pos[:, None] * g.inv[None, :]; fr = torch.cat([fr, fr], -1)
    cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
    for i in range(24):
        x = checkpoint(layer, i, x, cos, sin, gates[i] if i < NL else None, use_reentrant=False)
    h = GL.rms_zc(x, g.norm_w, eps)
    opt_abs = [pr['q0'] + o for o in pr['opt']]
    lg = g.head(h[T - 1].float()[None], h[torch.tensor(opt_abs, device=dev)].float()[None])[0][:pr['rq'].n_slots] / g.p.temp_for(pr['rq'].kind)
    lp = F.log_softmax(lg, -1)
    loss = -lp[lp.argmax()]
    loss.backward()
    return [gt.grad.abs() for gt in gates]


def fit(state, qd):
    pr = g.prep(state, qd)
    if len(pr['s']) > a.maxtok:
        s = pr['s']; pr['s'] = s[:a.maxtok // 4] + s[-(a.maxtok - a.maxtok // 4):]; pr['q0'] = len(pr['s'])
    return pr


t0 = time.time()
scores = {}
for k in SPECS:
    acc = [torch.zeros(g.L[i]['I'], device=dev) for i in range(NL)]
    for st, qd in cal[k]:
        sc = taylor(fit(st, qd))
        for i in range(NL): acc[i] += sc[i]
    scores[k] = [s_ / max(1, len(cal[k])) for s_ in acc]
    print('scored', k, f'{time.time()-t0:.0f}s', flush=True)


def mask_of(sc):
    out = {}
    for i in range(NL):
        n = int(round(a.keep * sc[i].numel())); m = torch.zeros_like(sc[i]); m[torch.topk(sc[i], n).indices] = 1; out[i] = m.to(torch.bfloat16)
    return out


masks = {k: mask_of(scores[k]) for k in SPECS}
glob = [sum(scores[k][i] / scores[k][i].sum() for k in SPECS) for i in range(NL)]
gmask = mask_of(glob)
ov = {str(k): float(sum((masks[k][i] * gmask[i]).sum() for i in range(NL)) / sum(gmask[i].sum() for i in range(NL))) for k in SPECS}
print('overlap of each spec mask with the global mask', ov, flush=True)
torch.save(dict(scores={str(k): v for k, v in scores.items()}), W + 'ifp_scores.pt')

preds = collections.defaultdict(dict)
for j, k in enumerate(SPECS):
    other = SPECS[(j + 1) % len(SPECS)]
    for it in ra:
        if (it['domain'], it['hook']) != k[:2] or k[2] not in it['questions']: continue
        pr = g.prep(it['state'], it['questions'][k[2]]); ids = pr['s'] + pr['q']
        for cfg, mm in (('dense', None), ('own', masks[k]), ('global', gmask), ('other', masks[other])):
            h, _ = g.forward(ids, None, mlp_mask=mm)
            p = g.probs(h, pr)
            preds[cfg].setdefault(it['id'], {})[k[2]] = {lab: p[jj] for jj, lab in enumerate(pr['rq'].slot_labels)}
    print('evaluated', k, f'{time.time()-t0:.0f}s', flush=True)
    json.dump(dict(preds=preds, specs=[list(s) for s in SPECS], overlap=ov, keep=a.keep, layers=NL), open(a.out, 'w'))
print('done', flush=True)
