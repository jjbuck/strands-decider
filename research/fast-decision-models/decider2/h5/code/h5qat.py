"""H5: short LoRA QAT on top of a saved rotated+quantized hobson (h5build --save). LoRA lives in the ROTATED basis of every GEMM
(W_eff = Wr + B A, Wr = the GPTQ/RTN dequantized weight, on its grid), re-quantized each step with the FIXED per-channel scales and
straight-through rounding (so step 0 reproduces the GPTQ solution exactly). Activations fake-quantized as at inference.
Teacher: bf16 hobson (same runtime, 'ref' path). Loss: KL on the calibrated answer distribution + hid * per-token relative residual error.
Data: evalkit/train_pool (train split), dev rids excluded.  python h5qat.py --q q/NAME.pt --steps 150 --accum 4 --lr 1e-4 --tag a"""
import os, sys, json, time, random, argparse, math
sys.path.insert(0, os.path.expanduser('~/work/h5'))
import torch, torch.nn as nn, torch.nn.functional as F
import h5lib as H

ap = argparse.ArgumentParser()
ap.add_argument('--q', required=True); ap.add_argument('--steps', type=int, default=150); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--r', type=int, default=16); ap.add_argument('--maxtok', type=int, default=3000)
ap.add_argument('--hid', type=float, default=1.0); ap.add_argument('--tag', default='a'); ap.add_argument('--ndev', type=int, default=120)
ap.add_argument('--evalevery', type=int, default=50)
a = ap.parse_args(); random.seed(7); torch.manual_seed(7)
W = os.path.expanduser('~/work/h5/'); os.makedirs(W + 'qat', exist_ok=True)
m = H.Q5()
D = torch.load(W + a.q)
m.Wr = [{k: v.to(m.dev) for k, v in d.items()} for d in D['Wr']]; m.Ws = [{k: v.to(m.dev) for k, v in d.items()} for d in D['Ws']]
m.R1 = D['R1'].to(m.dev).float(); m.cfg = D['cfg']; m.aclip = D['aclip']; m.ba_hp = D['ba_hp']; del D
HID = (5, 11, 17, 23)


class LoRA:
    def __init__(self, m, r):
        self.A = {}; self.B = {}; self.params = []
        for i in range(24):
            for s in H.SITES:
                N, K = m.Wr[i][s].shape
                A_ = torch.randn(r, K, device=m.dev) / math.sqrt(K); B_ = torch.zeros(N, r, device=m.dev)
                A_.requires_grad_(True); B_.requires_grad_(True)
                self.A[(i, s)] = A_; self.B[(i, s)] = B_; self.params += [A_, B_]

    def apply(self, i, s, Wr, m):
        We = Wr.float() + self.B[(i, s)] @ self.A[(i, s)]
        wb, _ = m.bits(i, s)
        if wb >= 16: return We.to(torch.bfloat16)
        qm = 2 ** (wb - 1) - 1; sc = m.Ws[i][s][:, None]
        return (H.ste(We / sc).clamp(-qm, qm) * sc).to(torch.bfloat16)


lora = LoRA(m, a.r); m.lora = lora
pool = H.load_pool(150, 3000)
rng = random.Random(11); dev_pool = list(pool); rng.shuffle(dev_pool)
dev_rids = {r['rid'] for r in dev_pool[:240]}
DEV = []
for r in dev_pool[:a.ndev]:
    qn = rng.choice(sorted(r['questions'])); pr = m.prep(r['state'], r['questions'][qn]); DEV.append((pr, pr['s'] + pr['q']))
train = [r for r in H.load_pool(150, a.maxtok) if r['rid'] not in dev_rids]
random.shuffle(train)
print('train', len(train), 'lora params', sum(p.numel() for p in lora.params), flush=True)
opt = torch.optim.AdamW(lora.params, lr=a.lr, weight_decay=0.0)
warm = max(1, a.steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * u / a.steps))))
REFD = None


@torch.no_grad()
def dev_eval():
    global REFD
    if REFD is None:
        REFD = []
        for pr, ids in DEV:
            h, _ = m.forward(ids, 'ref'); REFD.append(torch.softmax(m.logits(h, pr).float(), -1))
    fl = 0; tv = 0.0; kl = 0.0
    for (pr, ids), pt in zip(DEV, REFD):
        h, _ = m.forward(ids, 'q'); ps = torch.softmax(m.logits(h, pr).float(), -1)
        fl += int(pt.argmax() != ps.argmax()); tv += H.tv(pt, ps); kl += H.kl(pt, ps)
    n = len(DEV); return dict(flips=fl, n=n, flip_rate=round(fl / n, 4), tv=round(tv / n, 4), kl=round(kl / n, 4))


def save(tag):
    """merge LoRA, re-quantize with the fixed scales, save in h5build format"""
    with torch.no_grad():
        Wr = [{s: lora.apply(i, s, m.Wr[i][s], m).cpu() for s in H.SITES} for i in range(24)]
    torch.save(dict(Wr=Wr, Ws=[{k: v.cpu() for k, v in d.items()} for d in m.Ws], R1=m.R1.cpu(), cfg=m.cfg, aclip=m.aclip, ba_hp=m.ba_hp), W + f'qat/{tag}.pt')


log = open(W + f'qat/qat_{a.tag}.log.jsonl', 'a')
e0 = dev_eval(); print('step 0 dev', json.dumps(e0), flush=True); log.write(json.dumps(dict(step=0, dev=e0)) + '\n'); log.flush()
t0 = time.time(); k = 0; acc = dict(kl=0.0, hid=0.0, n=0)
for step in range(1, a.steps + 1):
    for _ in range(a.accum):
        r = train[k % len(train)]; k += 1
        qn = random.choice(sorted(r['questions'])); pr = m.prep(r['state'], r['questions'][qn]); ids = pr['s'] + pr['q']
        with torch.no_grad():
            ht, ct = m.forward(ids, 'ref', keep=HID); pt = torch.softmax(m.logits(ht, pr).float(), -1)
        hs, cs = m.forward(ids, 'q', keep=HID, ckpt=True)
        lp = F.log_softmax(m.logits(hs, pr).float(), -1)
        kl = (pt * (pt.clamp_min(1e-9).log() - lp)).sum()
        hl = sum((((cs[i].float() - ct[i].float()) ** 2).sum(-1) / ct[i].float().pow(2).sum(-1).clamp_min(1e-6)).mean() for i in HID) / len(HID)
        ((kl + a.hid * hl) / a.accum).backward()
        acc['kl'] += float(kl.detach()); acc['hid'] += float(hl.detach()); acc['n'] += 1
    torch.nn.utils.clip_grad_norm_(lora.params, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 5 == 0 or step == 1:
        e = dict(step=step, kl=acc['kl'] / acc['n'], hid=acc['hid'] / acc['n'], t=round(time.time() - t0), lr=sched.get_last_lr()[0],
                 mem=round(torch.cuda.max_memory_allocated() / 1e9, 1))
        print(json.dumps(e), flush=True); log.write(json.dumps(e) + '\n'); log.flush(); acc = dict(kl=0.0, hid=0.0, n=0)
    if step % a.evalevery == 0 or step == a.steps:
        ev = dev_eval(); print('dev', step, json.dumps(ev), flush=True); log.write(json.dumps(dict(step=step, dev=ev)) + '\n'); log.flush()
        save(f'{a.tag}_s{step}')
print('done', round(time.time() - t0), flush=True)
