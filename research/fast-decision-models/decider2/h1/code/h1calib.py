"""H1 calibration: unrotated input Hessians H = sum x^T x of every GEMM input (96) over train-split real states. -> ~/work/h1/hess/H_{i}_{k}.pt
python h1calib.py --n 64"""
import os, sys, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2')]
import torch, h1lib as HL
ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=64); ap.add_argument('--maxtok', type=int, default=3000); a = ap.parse_args()
g = HL.H1(); os.makedirs(g.opt['hdir'], exist_ok=True)
its = HL.cal_items(a.n, seed=0)
seqs = []
for st, qd in its:
    pr = g.prep(st, qd); ids = pr['s'] + pr['q']
    if len(ids) > a.maxtok: ids = ids[:a.maxtok // 4] + ids[-(a.maxtok - a.maxtok // 4):]
    seqs.append(ids)
print('calib seqs', len(seqs), 'tokens', sum(map(len, seqs)), flush=True)
t0 = time.time()
for grp in (range(0, 6), range(6, 12), range(12, 18), range(18, 24)):
    g.cap = {(i, k) for i in grp for k in HL.GEMMS}; g.Hacc = {}
    for ids in seqs: g.fwd(ids, stop=max(grp) + 1)
    for (i, k), H in g.Hacc.items(): torch.save((H / sum(map(len, seqs))).cpu(), f"{g.opt['hdir']}/H_{i}_{k}.pt")
    print('group', list(grp), f'{time.time()-t0:.0f}s', flush=True)
    g.Hacc = {}; torch.cuda.empty_cache()
print('done', flush=True)
