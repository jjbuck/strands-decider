"""H3 fine-tune arms (identical budget, data order and loss; only the architecture / QAT switch differs):
  --head std|hyp  (hyp = v1 hyperspherical pointer head)     --norm rms|l2|l2s  (l2 = v2 literal, l2s = v2 with scalar gain)
  --qat none|int4|nvfp4  (STE fake quant W4A4 on every torso GEMM, per-channel W with clip ratios fixed at init, per-token A)
Student: hobson-v19 (merged) + LoRA r16/a32 on every projection group (Win incl. GDN in_proj qkv/z/a/b, Wo, Wgu, Wd) + trainable head (+ norm scalars).
Teacher: frozen bf16 hobson in the same process (calibrated: logits / T_kind).
Loss: KL(teacher || student) on the answer distribution + --hid * relative MSE of the residual stream at layers 5, 11, 17 (dense signal).
No gold CE: the target is fidelity to hobson (flips vs bf16 hobson), not accuracy.
Data: train-split real states (evalkit/train_pool.jsonl, eval tasks excluded) and train_v5 rows (fraction --v5frac), same order for every arm.
Optimiser (F7): AdamW(0.9, 0.95), lr 1e-4 (LoRA), head / norm lr 1e-3, 3% warmup, cosine, clip 1.0, --accum micro-steps per update.
python train_h3.py --arm v1 --head hyp --steps 1200"""
import os, sys, json, time, random, argparse, collections, math
sys.path.insert(0, os.path.expanduser('~/work/h3'))
import numpy as np, torch, torch.nn.functional as F
import h3lib as H

ap = argparse.ArgumentParser(); ap.add_argument('--arm', required=True); ap.add_argument('--head', default='std'); ap.add_argument('--norm', default='rms')
ap.add_argument('--qat', default='none'); ap.add_argument('--steps', type=int, default=1200); ap.add_argument('--accum', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--hlr', type=float, default=1e-3); ap.add_argument('--r', type=int, default=16)
ap.add_argument('--alpha', type=float, default=32); ap.add_argument('--v5frac', type=float, default=0.3); ap.add_argument('--maxtok', type=int, default=2048)
ap.add_argument('--hid', type=float, default=1.0); ap.add_argument('--ckpts', type=int, default=3); ap.add_argument('--qrot', action='store_true')
a = ap.parse_args(); random.seed(2); torch.manual_seed(2)
W = os.path.expanduser('~/work/h3/'); CK = W + f'ck/{a.arm}/'; os.makedirs(CK, exist_ok=True)
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

m = H.H3(); m.detach_inference(); dev = m.dev
HID = (5, 11, 17)


def fit(state, qd):
    pr = m.prep(state, qd)
    if len(pr['s']) > a.maxtok:
        s = pr['s']; pr['s'] = s[:a.maxtok // 4] + s[-(a.maxtok - a.maxtok // 4):]; pr['q0'] = len(pr['s'])
    return pr


def sample(step_rng):
    global pi, vi
    if step_rng.random() < a.v5frac:
        r = v5[vi % len(v5)]; vi += 1
        qd = {'type': 'choice', 'instructions': step_rng.choice([r['instructions']] + r.get('instruction_variants', [])[:2]),
              'criteria': {n: d_ for n, d_ in r['options']}}
        return fit(r['state'], qd), 'v5'
    r = pool[pi % len(pool)]; pi += 1
    qn = step_rng.choice(sorted(r['questions'])); return fit(r['state'], r['questions'][qn]), 'real'


class Teacher:
    """bf16 hobson: no LoRA, rms norms, hobson's head, no fake quant"""
    def __enter__(self):
        self.s = (m.lora, m.norm_mode, m.qat, m.head); m.lora = None; m.norm_mode = 'rms'; m.qat = None; m.head = m.head0
    def __exit__(self, *e):
        m.lora, m.norm_mode, m.qat, m.head = self.s


# ---- student setup
p_lora = m.add_lora(r=a.r, alpha=a.alpha, seed=2)
p_head = m.set_head(a.head)
p_norm = m.set_norm(a.norm)
if a.qat != 'none':
    m.init_qat(a.qat)
    if a.qrot: raise NotImplementedError('rotated QAT: use G2 g2qat.py')
# hyperspherical head: fit tau by least squares to hobson's (pre-temperature) logits on 32 samples (centered per question)
if a.head == 'hyp':
    prng = random.Random(99); num = den = 0.0
    pi = vi = 10 ** 6  # use rows far from the training order start
    with torch.no_grad():
        for _ in range(32):
            pr, _k = sample(prng); ids = pr['s'] + pr['q']
            with Teacher():
                h = m.forward(ids); lt = m.logits(h, pr) * m.temp(pr['rq'].kind)
            m.head.log_tau.fill_(0.0); lc = m.logits(m.forward(ids), pr) * m.temp(pr['rq'].kind)
            lt = lt - lt.mean(); lc = lc - lc.mean(); num += float((lt * lc).sum()); den += float((lc * lc).sum())
        tau = max(num / max(den, 1e-9), 1.0); m.head.log_tau.fill_(math.log(tau))
    print('hyp head tau', tau, flush=True)
pi = vi = 0
srng = random.Random(5)
groups = [dict(params=p_lora, lr=a.lr), dict(params=p_head + p_norm, lr=a.hlr)]
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
nupd = max(1, a.steps // a.accum); warm = max(1, int(0.03 * nupd))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, u / nupd))))
save_at = set(int(a.steps * f) for f in np.linspace(0, 1, a.ckpts + 1)[1:])
lsum = collections.defaultdict(float); lcnt = collections.Counter(); t0 = time.time(); log = open(CK + 'train.log.jsonl', 'w')
json.dump(vars(a), open(CK + 'args.json', 'w'))
params = p_lora + p_head + p_norm
for step in range(1, a.steps + 1):
    pr, kind = sample(srng)
    ids = pr['s'] + pr['q']
    with torch.no_grad(), Teacher():
        ht, kt = m.forward(ids, keep=HID)
        tp = torch.softmax(m.logits(ht, pr).float(), -1)
    hs, ks = m.forward(ids, ckpt=True, keep=HID)
    lp = F.log_softmax(m.logits(hs, pr).float(), -1)
    kl = (tp * (tp.clamp_min(1e-8).log() - lp)).sum()
    hl = sum(((ks[i].float() - kt[i].float()) ** 2).sum() / kt[i].float().pow(2).sum() for i in HID) / len(HID)
    loss = kl + a.hid * hl
    (loss / a.accum).backward()
    lsum['kl_' + kind] += float(kl); lcnt['kl_' + kind] += 1; lsum['hid'] += float(hl); lcnt['hid'] += 1
    lsum['flip'] += float(lp.argmax() != tp.argmax()); lcnt['flip'] += 1
    if step % a.accum == 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step % 25 == 0 or step == 1:
        rec = dict(step=step, t=round(time.time() - t0), mem=round(torch.cuda.max_memory_allocated() / 1e9, 1), T=len(ids), lr=sched.get_last_lr()[0],
                   **{k: round(lsum[k] / max(1, lcnt[k]), 5) for k in lsum})
        if a.head == 'hyp': rec['tau'] = round(float(m.head.log_tau.exp()), 3)
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush(); lsum.clear(); lcnt.clear()
    if step in save_at:
        torch.save(m.trainable_state(), CK + f's{step}.pt'); print('saved', step, flush=True)
torch.save(m.trainable_state(), CK + 'final.pt')
print('done', time.time() - t0, flush=True)
