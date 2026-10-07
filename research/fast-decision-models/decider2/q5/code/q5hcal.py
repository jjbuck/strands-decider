"""B7 for precision formats: GPTQ input Hessians (H1 recipe: unrotated H = sum x^T x / tokens of every GEMM input, dense bf16 forward) from ONE
deployment domain's train-split calibration requests (Q5 cal tasks), instead of H1's pool sample (64 states, ~77% banking by pool share).
python q5hcal.py --dom bank --n 64   -> ~/work/q5/hess_bank/H_{i}_{k}.pt   (used by S5 configs with fcal='bank')"""
import os, sys, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/q5'), os.path.expanduser('~/work/q1')]
import q5lib as Q5
import torch, h1lib as HL
ap = argparse.ArgumentParser(); ap.add_argument('--dom', required=True); ap.add_argument('--n', type=int, default=64); ap.add_argument('--maxtok', type=int, default=3000); a = ap.parse_args()
g = HL.H1(); out = f'{Q5.W5}/hess_{a.dom}'; os.makedirs(out, exist_ok=True)
its = Q5.req_set(a.dom, 'cal', a.n, maxT=10 ** 9, minS=300, per_task=2, seed=23)
seqs = []
for it in its:
    pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']
    if len(ids) > a.maxtok: ids = ids[:a.maxtok // 4] + ids[-(a.maxtok - a.maxtok // 4):]
    seqs.append(ids)
print('calib seqs', len(seqs), 'tokens', sum(map(len, seqs)), flush=True)
t0 = time.time()
for grp in (range(0, 6), range(6, 12), range(12, 18), range(18, 24)):
    g.cap = {(i, k) for i in grp for k in HL.GEMMS}; g.Hacc = {}
    for ids in seqs: g.fwd(ids, stop=max(grp) + 1)
    for (i, k), H in g.Hacc.items(): torch.save((H / sum(map(len, seqs))).cpu(), f"{out}/H_{i}_{k}.pt")
    print('group', list(grp), f'{time.time()-t0:.0f}s', flush=True)
    g.Hacc = {}; torch.cuda.empty_cache()
print('done', flush=True)
