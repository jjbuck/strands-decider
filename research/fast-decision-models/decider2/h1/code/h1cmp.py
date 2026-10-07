"""H1: compare whole configs on train-split calibration questions (KL(dense||cfg), head-row hidden rel err, flips). Same config grammar as h1eval.
python h1cmp.py --cfg 'w4a4,gptq;w4a8,gptq' --n 60 --out res/cmp_x.json"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2')]
import torch, h1lib as HL
ap = argparse.ArgumentParser(); ap.add_argument('--cfg', required=True); ap.add_argument('--n', type=int, default=60); ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=1); ap.add_argument('--maxtok', type=int, default=3000); a = ap.parse_args()
g = HL.H1(); base_opt = dict(g.opt)


def parse(c):
    parts = c.split(','); opt = dict(base_opt); P = g.uniform(parts[0]); lr = None
    for p in parts[1:]:
        if p == 'gptq': opt['wq'] = 'gptq'
        elif p == 'gptq8': opt['wq8'] = 'gptq'; opt['wq'] = 'gptq'
        elif p == 'ba16': opt['ba16'] = True
        elif p == 'ohead': opt['ohead'] = True
        elif p == 'norout': opt['rout'] = False
        elif p == 'qb16': opt['qb16'] = True
        elif p.startswith('clip='): opt['aclip4'] = float(p[5:])
        elif p.startswith('seed='): opt['rseed'] = int(p[5:])
        elif p.startswith('k8=') or p.startswith('k16='):
            f, n = p.split('=')[1].rsplit(':', 1); rk = json.load(open(f))['rank']
            for key in rk[:int(n)]: P[(int(key.split('.')[0]), key.split('.')[1])] = 'w8a8' if p.startswith('k8=') else 'bf16'
        elif p.startswith('ex=') or p.startswith('e8='):
            for key in p[3:].split('+'): P[(int(key.split('.')[0]), key.split('.')[1])] = 'bf16' if p.startswith('ex=') else 'w8a8'
        elif p.startswith('map='):
            for key, v in json.load(open(p[4:])).items(): P[(int(key.split('.')[0]), key.split('.')[1])] = v
        elif p.startswith('lrot='): lr = p[5:]
        else: raise ValueError(p)
    return opt, P, lr


its = HL.cal_items(a.n, seed=a.seed); data = []
for st, qd in its:
    pr = g.prep(st, qd); ids = pr['s'] + pr['q']
    if len(ids) <= a.maxtok: data.append((pr, ids))
print('items', len(data), flush=True)
ref = []
g.set_prec({})
for pr, ids in data:
    h, _ = g.fwd(ids); rows = torch.tensor([len(ids) - 1] + [pr['q0'] + o for o in pr['opt']], device=g.dev)
    ref.append((torch.log_softmax(g.logits(h, pr), -1), h[rows].float(), rows))
res = {}
if os.path.exists(a.out): res = json.load(open(a.out))
for c in a.cfg.split(';'):
    if c in res: continue
    t0 = time.time(); opt, P, lr = parse(c); g.opt = opt; g.load_lrot(lr); g.set_prec(P); kl = hid = fl = 0.0
    for (pr, ids), (p0, hr0, rows) in zip(data, ref):
        h, _ = g.fwd(ids, q0=pr['q0']); p1 = torch.log_softmax(g.logits(h, pr), -1)
        kl += float((p0.exp() * (p0 - p1)).sum()); hid += float((h[rows].float() - hr0).norm() / hr0.norm()); fl += int(p1.argmax() != p0.argmax())
    n = len(data); res[c] = dict(kl=kl / n, hid=hid / n, flips=fl, n=n, w8=sum(v == 'w8a8' for v in P.values()), bf16=96 - len(P))
    print(c, res[c], f'{time.time()-t0:.0f}s', flush=True); json.dump(res, open(a.out, 'w'), indent=1)
    g.drop_cache(); g.opt = dict(base_opt)
