"""Step 3b: short heal of the thin path (train split only).  Trainable = the thin factors A_r [N, r], B_r [r, K] of every GEMM in
layers >= k (used ONLY by thin state rows; question rows and the dense weights are untouched).  All state rows (except 4 sinks)
thin in layers >= k.  Loss = KL(p_dense || p_thin) on the answer + rel-MSE of the final hidden states of the question rows.
python heal.py K R NITEMS OUT.pt"""
import os, sys, json, random, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from tw import TW, GEMMS, SINK

K, R, NIT, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
LR = float(os.environ.get('LR', '1e-4')); ACC = int(os.environ.get('ACC', '4')); LAM = float(os.environ.get('LAM', '1.0'))
KIT = os.path.expanduser('~/work/evalkit')
ev = set(json.load(open(f'{KIT}/split.json'))['eval_tasks'])
pool = [json.loads(l) for l in open(f'{KIT}/train_pool.jsonl')]
pool = [r for r in pool if r['task'] not in ev and 300 <= r['n_state_tok'] <= 3500]
rng = random.Random(77); rng.shuffle(pool)
fitset = set()   # (the 128 basis-fit states were drawn with another seed; overlap is harmless: train split either way)
tw = TW(bases=os.path.expanduser('~/work/j13/bases.pt'), modes=('out',))
fac = {}; params = []
for l in range(K, tw.NL):
    fac[l] = {}
    for g in GEMMS:
        A = torch.nn.Parameter(tw.A['out'][l][g][:, :R].float().clone()); B = torch.nn.Parameter(tw.B['out'][l][g][:R].float().clone())
        fac[l][g] = (A, B); params += [A, B]
opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
nstep = NIT // ACC
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, nstep) / nstep)))
log = open(OUT + '.log.jsonl', 'w')
t0 = time.time(); it = 0; ema = None
for r in pool[:NIT]:
    qn = rng.choice(list(r['questions']))
    try: pr = tw.prep(r, qn)
    except Exception: continue
    q0 = pr['q0']; T = len(pr['ids'])
    with torch.no_grad():
        x, cos, sin = tw.embed_rope(pr['ids'])
        hk, _ = tw.run(x, cos, sin, 0, K)
        xd, _ = tw.run(hk, cos, sin, K)
        lpd = tw.head(xd, pr)
        hq_d = xd[q0:].float()
    thin = list(range(SINK, q0))
    base = tw.thin_plan('out', R, thin, T)
    def lay(l, xx, *ps):
        pl = dict(base); pl['fac'] = {g: (fac[l][g][0].to(torch.bfloat16), fac[l][g][1].to(torch.bfloat16)) for g in GEMMS}
        return tw.layer(l, xx, cos, sin, pl)
    xx = hk.detach()
    for l in range(K, tw.NL):
        ps = [p for g in GEMMS for p in fac[l][g]]
        xx = torch.utils.checkpoint.checkpoint(lay, l, xx, *ps, use_reentrant=False) if T > 1500 else lay(l, xx, *ps)
    lp = tw.head(xx, pr)
    kl = (lpd.exp() * (lpd - lp)).sum()
    hq = xx[q0:].float()
    mse = ((hq - hq_d).pow(2).sum(-1) / hq_d.pow(2).sum(-1).clamp_min(1e-6)).mean()
    loss = (kl + LAM * mse) / ACC
    loss.backward()
    it += 1
    if it % ACC == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()
    v = float(kl); ema = v if ema is None else 0.95 * ema + 0.05 * v
    log.write(json.dumps({'it': it, 'T': T, 'kl': round(v, 5), 'mse': round(float(mse), 5), 'ema_kl': round(ema, 5)}) + '\n'); log.flush()
    if it % 25 == 0:
        print(it, 'T', T, 'kl %.4f ema %.4f mse %.4f' % (v, ema, float(mse)), '%.0fs' % (time.time() - t0), 'mem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
torch.save({'k': K, 'r': R, 'fac': {(l, g): (fac[l][g][0].detach().to(torch.bfloat16).cpu(), fac[l][g][1].detach().to(torch.bfloat16).cpu()) for l in fac for g in GEMMS}}, OUT)
print('done %.0fs' % (time.time() - t0))
