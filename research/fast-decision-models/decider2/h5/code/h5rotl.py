"""H5: learn the residual rotation R1 against a LOCAL objective (fast, deterministic, full-batch): the output-space error of 4-bit
per-token activation quantization at all 48 R1-rotated GEMM inputs (Win, Wgu of 24 layers), weights 16-bit (SpinQuant's W16A4 setting):
   L(R) = sum_sites  sum_t || (Q(x_t R) R^T - x_t) diag(f) W^T ||^2 / sum_t || x_t diag(f) W^T ||^2
computed through the Gram matrix G = diag(f) W^T W diag(f) (2048 x 2048 per site). Q: per-token symmetric absmax (clip), STE rounding,
scale NOT detached. Activations: normed residual rows from the bf16 model on train-split states (dev rids excluded), token 0 always kept.
R1 = R0 expm(A - A^T), Adam on A. Then a dev check in 'learn' mode (W16A4, end to end).
python h5rotl.py --steps 300 --lr 1e-3 --tag l1 [--plain]   (--plain: unweighted activation MSE instead of the Gram-weighted error)"""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.expanduser('~/work/h5'))
import torch, torch.nn.functional as F
import h5lib as H
import g2lib as GL

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=300); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--tag', default='l1')
ap.add_argument('--nseq', type=int, default=64); ap.add_argument('--tok', type=int, default=128); ap.add_argument('--aclip', type=float, default=0.9)
ap.add_argument('--plain', action='store_true'); ap.add_argument('--ndev', type=int, default=120); ap.add_argument('--abits', type=int, default=4)
a = ap.parse_args(); random.seed(3); torch.manual_seed(3)
W = os.path.expanduser('~/work/h5/'); os.makedirs(W + 'rot', exist_ok=True)
m = H.Q5(); m.aclip = a.aclip
CL = ('Win_g', 'Win_a', 'Wo_g', 'Wo_a', 'Wgu', 'Wd')
m.cfg = {c: (16, a.abits) for c in CL}

pool = H.load_pool(150, 3000)
rng = random.Random(11); dev_pool = list(pool); rng.shuffle(dev_pool)
dev_rids = {r['rid'] for r in dev_pool[:240]}
DEV = []
for r in dev_pool[:a.ndev]:
    qn = rng.choice(sorted(r['questions'])); pr = m.prep(r['state'], r['questions'][qn]); DEV.append((pr, pr['s'] + pr['q']))
cal = [r for r in H.load_pool(300, 3000) if r['rid'] not in dev_rids]; random.shuffle(cal)

# ---- collect normed residual rows at the 48 R1 sites
X = {(i, s): [] for i in range(24) for s in ('Win', 'Wgu')}
caps = {}
orig_in = m.site_in


def cap_in(x, i, site, mode, R):
    xn = GL.nrm(x, m.eps); T = xn.shape[0]
    idx = torch.cat([torch.zeros(1, dtype=torch.long, device=xn.device), torch.randperm(T - 1, device=xn.device)[:a.tok - 1] + 1])
    X[(i, site)].append(xn[idx].clone())
    return orig_in(x, i, site, mode, R)


m.site_in = cap_in
with torch.no_grad():
    for r in cal[:a.nseq]:
        qn = sorted(r['questions'])[0]; pr = m.prep(r['state'], r['questions'][qn]); m.forward(pr['s'] + pr['q'], 'ref')
m.site_in = orig_in
X = {k: torch.cat(v).to(torch.bfloat16) for k, v in X.items()}
print('collected', {k: tuple(v.shape) for k, v in list(X.items())[:2]}, flush=True)
Gm = {}
for i in range(24):
    for s in ('Win', 'Wgu'):
        Wf = (m.L[i]['Wf_in'] if s == 'Win' else m.L[i]['Wf_gu']).float()
        if s == 'Win' and m.L[i]['type'] == 'linear_attention': Wf = Wf[:8192]   # b/a rows run in bf16
        Gm[(i, s)] = (Wf.t() @ Wf).to(torch.bfloat16)
den = {}
with torch.no_grad():
    for k, x in X.items():
        xf = x.float()
        den[k] = float((xf * (xf @ Gm[k].float())).sum()) if not a.plain else float(xf.pow(2).sum())

R0 = m.R1.clone().float()
A = torch.zeros(2048, 2048, device=m.dev, requires_grad=True)
opt = torch.optim.Adam([A], lr=a.lr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / 10) * max(0.0, 1 - u / a.steps))


def loss_fn(R, per=False):
    Rb = R.to(torch.bfloat16); tot = 0.0; parts = {}
    for k, x in X.items():
        xr = x @ Rb
        e = (H.fq_act(xr, a.abits, a.aclip, detach_scale=False) - xr) @ Rb.t()      # error back in the original basis
        if a.plain: l = e.float().pow(2).sum() / den[k]
        else: l = (e.float() * (e @ Gm[k]).float()).sum() / den[k]
        tot = tot + l
        if per: parts[k] = float(l)
    return (tot / len(X), parts) if per else tot / len(X)


@torch.no_grad()
def dev_eval(R):
    fl = 0; tv = 0.0; kl = 0.0
    for pr, ids in DEV:
        ht, _ = m.forward(ids, 'ref'); pt = torch.softmax(m.logits(ht, pr).float(), -1)
        hs, _ = m.forward(ids, 'learn', R=R); ps = torch.softmax(m.logits(hs, pr).float(), -1)
        fl += int(pt.argmax() != ps.argmax()); tv += H.tv(pt, ps); kl += H.kl(pt, ps)
    n = len(DEV); return dict(flips=fl, n=n, flip_rate=round(fl / n, 4), tv=round(tv / n, 4), kl=round(kl / n, 4))


log = open(W + f'rot/rotl_{a.tag}.log.jsonl', 'a')
with torch.no_grad():
    l0 = float(loss_fn(R0)); lid = float(loss_fn(torch.eye(2048, device=m.dev)))
print(f'local loss: identity {lid:.5f}  hadamard R0 {l0:.5f}', flush=True); log.write(json.dumps(dict(identity=lid, hadamard=l0)) + '\n')
t0 = time.time()
for step in range(1, a.steps + 1):
    R = R0 @ torch.linalg.matrix_exp(A - A.t())
    Rl = R.detach().requires_grad_(True); Rb = Rl.to(torch.bfloat16); loss = 0.0
    for k, x in X.items():                       # per-site backward keeps memory at one site's graph
        xr = x @ Rb
        e = (H.fq_act(xr, a.abits, a.aclip, detach_scale=False) - xr) @ Rb.t()
        l = (e.float().pow(2).sum() if a.plain else (e.float() * (e @ Gm[k]).float()).sum()) / den[k] / len(X)
        l.backward(retain_graph=True); loss += float(l)
        Rb = Rl.to(torch.bfloat16)
    R.backward(Rl.grad); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 20 == 0 or step == 1:
        with torch.no_grad(): dR = float((R - R0).norm() / R0.norm())
        e = dict(step=step, loss=float(loss), dR=round(dR, 4), t=round(time.time() - t0)); print(json.dumps(e), flush=True); log.write(json.dumps(e) + '\n'); log.flush()
with torch.no_grad():
    R = R0 @ torch.linalg.matrix_exp(A - A.t())
    lf, parts = loss_fn(R, per=True)
torch.save(R.cpu(), W + f'rot/R1_{a.tag}.pt')
print('final local loss', float(lf), flush=True)
e1 = dev_eval(R); e0 = dev_eval(R0)
print('DEV W16A4 hadamard', json.dumps(e0), flush=True); print('DEV W16A4 learned', json.dumps(e1), flush=True)
log.write(json.dumps(dict(final=float(lf), dev_hadamard=e0, dev_learned=e1)) + '\n'); log.flush()
