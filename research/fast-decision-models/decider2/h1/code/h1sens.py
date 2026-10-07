"""H1 per-GEMM sensitivity (error propagation to the decision): quantize ONE GEMM (leave-one-in) or all-but-one (leave-one-out) at --prec,
measure on train-split calibration questions: KL(dense || quant) of the answer distribution, centered-logit MSE, argmax flips.
python h1sens.py --prec w4a4 --opts gptq --n 40 --out res/sens_w4.json [--mode lout --base w4a4 --hi w8a8]"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2')]
import torch, h1lib as HL
ap = argparse.ArgumentParser(); ap.add_argument('--prec', default='w8a8'); ap.add_argument('--opts', default=''); ap.add_argument('--n', type=int, default=40)
ap.add_argument('--out', required=True); ap.add_argument('--mode', default='lin'); ap.add_argument('--hi', default='bf16'); ap.add_argument('--maxtok', type=int, default=3000)
ap.add_argument('--seed', type=int, default=1); ap.add_argument('--fp32', action='store_true')
a = ap.parse_args()
g = HL.H1()
if a.fp32: g.to_fp32()
for o in filter(None, a.opts.split(',')):
    if o == 'gptq': g.opt['wq'] = 'gptq'
    elif o == 'gptq8': g.opt['wq'] = 'gptq'; g.opt['wq8'] = 'gptq'
    elif o == 'ba16': g.opt['ba16'] = True
    elif o == 'ohead': g.opt['ohead'] = True
    elif o.startswith('clip='): g.opt['aclip4'] = float(o[5:])
    elif o.startswith('seed='): g.opt['rseed'] = int(o[5:])
its = HL.cal_items(a.n, seed=a.seed)
data = []
for st, qd in its:
    pr = g.prep(st, qd); ids = pr['s'] + pr['q']
    if len(ids) > a.maxtok: continue
    data.append((pr, ids))
print('sens items', len(data), 'tokens', sum(len(x[1]) for x in data), flush=True)
keys = [(i, k) for i in range(24) for k in HL.GEMMS]
cfgs = {'ALL': g.uniform(a.prec)}
for key in keys:
    nm = f'{key[0]}.{key[1]}'
    cfgs[nm] = {key: a.prec} if a.mode == 'lin' else g.uniform(a.prec, {key: a.hi})
acc = {c: dict(kl=0.0, mse=0.0, flip=0, hid=0.0) for c in cfgs}
t0 = time.time()
for di, (pr, ids) in enumerate(data):
    g.set_prec({}); h0, _ = g.fwd(ids); l0 = g.logits(h0, pr); p0 = torch.log_softmax(l0, -1)
    rows = torch.tensor([len(ids) - 1] + [pr['q0'] + o for o in pr['opt']], device=g.dev); hr0 = h0[rows].float()
    for c, P in cfgs.items():
        g.set_prec(P); h, _ = g.fwd(ids); l1 = g.logits(h, pr); p1 = torch.log_softmax(l1, -1)
        acc[c]['hid'] += float((h[rows].float() - hr0).pow(2).sum() / hr0.pow(2).sum())
        acc[c]['kl'] += float((p0.exp() * (p0 - p1)).sum()); acc[c]['mse'] += float(((l1 - l1.mean()) - (l0 - l0.mean())).pow(2).mean())
        acc[c]['flip'] += int(l1.argmax() != l0.argmax())
    if di % 5 == 0: print(di, len(ids), f'{time.time()-t0:.0f}s', 'ALL kl', acc['ALL']['kl'] / (di + 1), flush=True)
n = len(data)
for c in acc: acc[c] = {k: v / n for k, v in acc[c].items()}
rank = sorted([c for c in cfgs if c != 'ALL'], key=lambda c: -acc[c]['kl'] if a.mode == 'lin' else acc[c]['kl'])
json.dump(dict(args=vars(a), n=n, acc=acc, rank=rank, sum_lin=sum(acc[c]['kl'] for c in rank)), open(a.out, 'w'), indent=1)
print('ALL', acc['ALL'], 'sum of singles', sum(acc[c]['kl'] for c in rank), flush=True)
print('top', [(c, round(acc[c]['kl'], 5)) for c in rank[:16]], flush=True)
