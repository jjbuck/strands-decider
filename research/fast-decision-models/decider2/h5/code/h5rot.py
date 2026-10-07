"""H5: SpinQuant-style learned residual rotation R1 for hobson-v19, weights frozen (16-bit during learning, as SpinQuant), activations
fake-quantized (per-token symmetric, STE) at every GEMM input; Ho/Hd fixed Hadamard online rotations.
R1 = R0 expm(A - A^T) (R0 = randomized Hadamard; Riemannian/Stiefel parametrization via the exponential map), Adam on A.
Loss: KL(bf16 hobson || quantized) on the calibrated answer distribution + hid * mean per-token relative residual error at layers HID.
Data: evalkit/train_pool.jsonl (train-split tasks only), one random question per request; dev = the h5sens dev set (seed 11) is excluded.
python h5rot.py --steps 200 --accum 4 --lr 2e-3 --tag a"""
import os, sys, json, time, random, argparse, math
sys.path.insert(0, os.path.expanduser('~/work/h5'))
import torch, torch.nn.functional as F
import h5lib as H

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=200); ap.add_argument('--accum', type=int, default=4); ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--maxtok', type=int, default=2500); ap.add_argument('--hid', type=float, default=1.0); ap.add_argument('--abits', type=int, default=4)
ap.add_argument('--aclip', type=float, default=1.0); ap.add_argument('--tag', default='a'); ap.add_argument('--ndev', type=int, default=120)
ap.add_argument('--evalevery', type=int, default=25); ap.add_argument('--wq', type=int, default=16, help='16 = SpinQuant W16 learning; 4 = RTN W4 in the loop (q mode)')
a = ap.parse_args(); random.seed(5); torch.manual_seed(5)
W = os.path.expanduser('~/work/h5/')
os.makedirs(W + 'rot', exist_ok=True)
m = H.Q5(); m.aclip = a.aclip
CL = ('Win_g', 'Win_a', 'Wo_g', 'Wo_a', 'Wgu', 'Wd')
m.cfg = {c: (16, a.abits) for c in CL}
HID = (5, 11, 17, 23)

pool = H.load_pool(150, 3000)
rng = random.Random(11); dev_pool = list(pool); rng.shuffle(dev_pool)
dev_rids = {r['rid'] for r in dev_pool[:240]}            # the h5sens dev set; never trained on
DEV = []
for r in dev_pool[:a.ndev]:
    qn = rng.choice(sorted(r['questions'])); pr = m.prep(r['state'], r['questions'][qn]); DEV.append((pr, pr['s'] + pr['q']))
train = [r for r in H.load_pool(150, a.maxtok) if r['rid'] not in dev_rids]
random.shuffle(train)
print('train requests', len(train), 'dev', len(DEV), flush=True)

R0 = m.R1.clone().float()
A = torch.zeros(2048, 2048, device=m.dev, requires_grad=True)
opt = torch.optim.Adam([A], lr=a.lr)
nupd = a.steps; warm = max(1, nupd // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * max(0.0, 1 - u / nupd))


def Rcur():
    return R0 @ torch.linalg.matrix_exp(A - A.t())


@torch.no_grad()
def dev_eval(R):
    fl = 0; tv = 0.0; kl = 0.0
    for pr, ids in DEV:
        ht, _ = m.forward(ids, 'ref'); pt = torch.softmax(m.logits(ht, pr).float(), -1)
        hs, _ = m.forward(ids, 'learn', R=R); ps = torch.softmax(m.logits(hs, pr).float(), -1)
        fl += int(pt.argmax() != ps.argmax()); tv += H.tv(pt, ps); kl += H.kl(pt, ps)
    n = len(DEV)
    return dict(flips=fl, n=n, flip_rate=round(fl / n, 4), tv=round(tv / n, 4), kl=round(kl / n, 4))


log = open(W + f'rot/rot_{a.tag}.log.jsonl', 'a')
with torch.no_grad():
    R = Rcur(); e0 = dev_eval(R)
print('step 0 dev', json.dumps(e0), flush=True); log.write(json.dumps(dict(step=0, dev=e0)) + '\n'); log.flush()
t0 = time.time(); k = 0; acc = dict(kl=0.0, hid=0.0, n=0)
for step in range(1, a.steps + 1):
    for _ in range(a.accum):
        r = train[k % len(train)]; k += 1
        qn = random.choice(sorted(r['questions'])); pr = m.prep(r['state'], r['questions'][qn]); ids = pr['s'] + pr['q']
        with torch.no_grad():
            ht, ct = m.forward(ids, 'ref', keep=HID); pt = torch.softmax(m.logits(ht, pr).float(), -1)
        R = Rcur()
        hs, cs = m.forward(ids, 'learn', R=R, keep=HID, ckpt=True)
        lp = F.log_softmax(m.logits(hs, pr).float(), -1)
        kl = (pt * (pt.clamp_min(1e-9).log() - lp)).sum()
        hl = sum((((cs[i].float() - ct[i].float()) ** 2).sum(-1) / ct[i].float().pow(2).sum(-1).clamp_min(1e-6)).mean() for i in HID) / len(HID)
        loss = (kl + a.hid * hl) / a.accum
        loss.backward()
        acc['kl'] += float(kl); acc['hid'] += float(hl); acc['n'] += 1
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 5 == 0 or step == 1:
        with torch.no_grad():
            Rn = Rcur(); orth = float((Rn.t() @ Rn - torch.eye(2048, device=m.dev)).abs().max()); dR = float((Rn - R0).norm() / R0.norm())
        e = dict(step=step, kl=acc['kl'] / acc['n'], hid=acc['hid'] / acc['n'], t=round(time.time() - t0), lr=sched.get_last_lr()[0],
                 orth_err=orth, dR=round(dR, 4), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        print(json.dumps(e), flush=True); log.write(json.dumps(e) + '\n'); log.flush(); acc = dict(kl=0.0, hid=0.0, n=0)
    if step % a.evalevery == 0 or step == a.steps:
        with torch.no_grad():
            Rn = Rcur(); ev = dev_eval(Rn)
        torch.save(Rn.cpu(), W + f'rot/R1_{a.tag}_s{step}.pt')
        print('dev', step, json.dumps(ev), flush=True); log.write(json.dumps(dict(step=step, dev=ev)) + '\n'); log.flush()
print('done', round(time.time() - t0), flush=True)
