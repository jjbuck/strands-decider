"""H5: per-GEMM learned Kronecker rotations (FlatQuant-style structure, but orthogonal: P = P1 (x) P2, P_k = expm(A_k - A_k^T)),
plus a learned per-GEMM activation clip and weight clip, on top of the residual rotation R1 and the online Hadamards.
Each GEMM is trained independently (local, full batch) on bf16-model inputs from train-split states:
   loss = || Q4(x R P) Q4(W R P)^T - x W^T ||^2 / || x W^T ||^2       (A: per-token, W: per-channel; STE; scales not detached)
P is init at identity (so step 0 = the current format). Runtime cost: the online kernels that already do H32 (x) H64 (H2's Hadamard kernel)
do (H32 P1) (x) (H64 P2) instead; the Win/Wgu inputs get one such kernel fused into the RMSNorm+quant kernel.
python h5kron.py --R rot/R1_l1.pt --steps 150 --tag k1"""
import os, sys, json, time, random, argparse, math
sys.path.insert(0, os.path.expanduser('~/work/h5'))
import torch, torch.nn.functional as F
import h5lib as H
import g2lib as GL

ap = argparse.ArgumentParser()
ap.add_argument('--R', default='rot/R1_l1.pt'); ap.add_argument('--steps', type=int, default=150); ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--tag', default='k1'); ap.add_argument('--nseq', type=int, default=64); ap.add_argument('--tok', type=int, default=64)
ap.add_argument('--abits', type=int, default=4); ap.add_argument('--wbits', type=int, default=4); ap.add_argument('--sites', default='Win,Wo,Wgu,Wd')
a = ap.parse_args(); random.seed(4); torch.manual_seed(4)
W = os.path.expanduser('~/work/h5/')
m = H.Q5(lean=True)
if a.R != 'had': m.R1 = torch.load(W + a.R).to(m.dev).float()
SITES = a.sites.split(',')

pool = H.load_pool(150, 3000)
rng = random.Random(11); dev_pool = list(pool); rng.shuffle(dev_pool)
dev_rids = {r['rid'] for r in dev_pool[:240]}
cal = [r for r in H.load_pool(300, 3000) if r['rid'] not in dev_rids]; random.Random(31).shuffle(cal)

X = {(i, s): [] for i in range(24) for s in SITES}
orig_in, orig_out = m.site_in, m.site_out


def pick(v):
    T = v.shape[0]
    idx = torch.cat([torch.zeros(1, dtype=torch.long, device=v.device), torch.randperm(T - 1, device=v.device)[:a.tok - 1] + 1])
    return v[idx].clone()


def cin(x, i, site, mode, R):
    if site in SITES: X[(i, site)].append(pick(GL.nrm(x, m.eps)))
    return orig_in(x, i, site, mode, R)


def cout(o, i, site, mode, R):
    if site in SITES: X[(i, site)].append(pick(o))
    return orig_out(o, i, site, mode, R)


m.site_in, m.site_out = cin, cout
with torch.no_grad():
    for r in cal[:a.nseq]:
        qn = sorted(r['questions'])[0]; pr = m.prep(r['state'], r['questions'][qn]); m.forward(pr['s'] + pr['q'], 'ref')
m.site_in, m.site_out = orig_in, orig_out
X = {k: torch.cat(v).to(torch.bfloat16) for k, v in X.items()}
print('collected', len(X), 'GEMMs x', next(iter(X.values())).shape[0], 'tokens', flush=True)

qa = 2 ** (a.abits - 1) - 1; qw = 2 ** (a.wbits - 1) - 1
inv_sig = lambda p: math.log(p / (1 - p))


def wq(Wr, cw):
    s = Wr.abs().amax(1, keepdim=True).clamp_min(1e-8) * cw / qw
    return H.ste(Wr / s).clamp(-qw, qw) * s


def aq(xr, ca):
    s = xr.abs().amax(-1, keepdim=True).clamp_min(1e-8) * ca / qa
    return H.ste(xr / s).clamp(-qa, qa) * s


Kout = {}; clipA = {}; stats = []
t0 = time.time()
for i in range(24):
    for s in SITES:
        x = X[(i, s)]
        xr0 = m.gemm_input(i, s, x, use_k=False).float()                        # [T, K] rotated by R1 / Ho / Hd
        W0 = m.rotated_weight(i, s, use_k=False).float()
        if s == 'Win' and m.L[i]['type'] == 'linear_attention': W0 = W0[:8192]  # b/a rows run in bf16
        yref = xr0 @ W0.t(); den = yref.pow(2).sum()
        K = W0.shape[1]; k1, k2 = H.KFAC[K]
        A1 = torch.zeros(k1, k1, device=m.dev, requires_grad=True); A2 = torch.zeros(k2, k2, device=m.dev, requires_grad=True)
        ga = torch.tensor(inv_sig(0.9), device=m.dev, requires_grad=True); gw = torch.tensor(inv_sig(0.95), device=m.dev, requires_grad=True)
        opt = torch.optim.Adam([A1, A2, ga, gw], lr=a.lr)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / 10) * max(0.0, 1 - u / a.steps))

        def lossf():
            P1 = torch.linalg.matrix_exp(A1 - A1.t()); P2 = torch.linalg.matrix_exp(A2 - A2.t())
            xr = H.kron_apply(xr0, P1, P2); Wr = H.kron_apply(W0, P1, P2)
            y = aq(xr, torch.sigmoid(ga)).to(torch.bfloat16) @ wq(Wr, torch.sigmoid(gw)).to(torch.bfloat16).t()
            return (y.float() - yref).pow(2).sum() / den
        with torch.no_grad(): l0 = float(lossf())
        for step in range(a.steps):
            l = lossf(); l.backward(); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            l1 = float(lossf())
            P1 = torch.linalg.matrix_exp(A1 - A1.t()).detach(); P2 = torch.linalg.matrix_exp(A2 - A2.t()).detach()
        Kout[(i, s)] = (P1.cpu(), P2.cpu()); clipA[(i, s)] = float(torch.sigmoid(ga))
        stats.append(dict(i=i, s=s, l0=round(l0, 5), l1=round(l1, 5), ca=round(clipA[(i, s)], 3), cw=round(float(torch.sigmoid(gw)), 3)))
        print(json.dumps(stats[-1]), round(time.time() - t0), 's', flush=True)
        del xr0, W0, yref
torch.save(dict(K=Kout, clipA=clipA, stats=stats, R=a.R), W + f'rot/kron_{a.tag}.pt')
r0 = sum(x['l0'] for x in stats); r1 = sum(x['l1'] for x in stats)
print('total local loss', round(r0, 4), '->', round(r1, 4), f'({100 * (1 - r1 / r0):.1f}% lower)', flush=True)
