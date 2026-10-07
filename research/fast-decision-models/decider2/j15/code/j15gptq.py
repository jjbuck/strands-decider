"""GPTQ code-draw variance: H1 recipe with a different calibration draw (cal_items seed S), W8 codes only. -> ~/work/j15/codes_w8_s{S}.pt"""
import os, sys, time, torch
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2')]
import h1lib as HL
S = int(sys.argv[1]); hdir = os.path.expanduser(f'~/work/j15/hess_s{S}')
os.makedirs(hdir, exist_ok=True)
g = HL.H1(hdir=hdir)
its = HL.cal_items(64, seed=S)
seqs = []
for st, qd in its:
    pr = g.prep(st, qd); ids = pr['s'] + pr['q']
    if len(ids) > 3000: ids = ids[:750] + ids[-2250:]
    seqs.append(ids)
t0 = time.time()
for grp in (range(0, 6), range(6, 12), range(12, 18), range(18, 24)):
    g.cap = {(i, k) for i in grp for k in HL.GEMMS}; g.Hacc = {}
    for ids in seqs: g.fwd(ids, stop=max(grp) + 1)
    for (i, k), H in g.Hacc.items(): torch.save((H / sum(map(len, seqs))).cpu(), f"{hdir}/H_{i}_{k}.pt")
    g.Hacc = {}; torch.cuda.empty_cache()
print('calib', f'{time.time()-t0:.0f}s', flush=True)
del g; torch.cuda.empty_cache()
g = HL.H1(wq='gptq', wq8='gptq', hdir=hdir)
out = {}
for i in range(24):
    for k in HL.GEMMS:
        q, s = g.qweight(i, k, 'w8a8'); out[(i, k)] = (q.cpu(), s.cpu()); g.Qw = {}
torch.save(out, os.path.expanduser(f'~/work/j15/codes_w8_s{S}.pt'))
print('saved', f'{time.time()-t0:.0f}s', flush=True)
