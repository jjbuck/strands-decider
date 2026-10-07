"""B3 first measurement: prototype subtraction. Per GEMM input (row samples from q1cal, 128 train-split requests x 48 rows):
k-means codebooks (Kc = 1 (mean), 64, 256) on 80% of rows; on the other 20%: crest (max/rms after the GEMM's rotation) of x and of the
residual x - c, and the per-token int4 (clip .9) relative error ||x - xhat|| / ||x|| for plain rotated int4 and for residual coding
(c added back exactly). Also the cost of the nearest-prototype search relative to the GEMM (K*Kc vs K*N MACs).
python q1b3.py -> ~/work/q1/res_b3.json"""
import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL

g = QL.Q1(grad=False); dev = g.dev
CAL = os.path.expanduser('~/work/q1/cal')
out = {}; t0 = time.time()


def q4err(xr):
    am = xr.abs().amax(1).clamp_min(1e-8); s = (am / torch.full_like(am, 7.0)) * 0.9
    q = torch.round(xr / s[:, None]).clamp(-7, 7)
    return xr - q * s[:, None]


def kmeans(X, k, it=20, seed=0):
    gen = torch.Generator(device=X.device); gen.manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=gen, device=X.device)[:k]].clone()
    for _ in range(it):
        d = (X * X).sum(1, keepdim=True) - 2 * X @ C.t() + (C * C).sum(1)[None, :]
        a = d.argmin(1)
        Cn = torch.zeros_like(C).index_add_(0, a, X); cnt = torch.bincount(a, minlength=k).float()
        m = cnt > 0; C[m] = Cn[m] / cnt[m, None]
    return C


for i in range(24):
    for k in QL.GEMMS:
        Xs = torch.load(f'{CAL}/X_{i}_{k}.pt').float().to(dev)
        X = Xs[:, :-1]; isq = Xs[:, -1] > 0.5
        X = X[~isq]                                      # state rows
        n = X.shape[0]; perm = torch.randperm(n, generator=torch.Generator().manual_seed(1)).to(dev)
        tr, te = X[perm[: int(.8 * n)]], X[perm[int(.8 * n):]]
        R = g.rot_for(i, k)
        xr = R(te); nx = te.pow(2).sum(1).sqrt()
        e0 = q4err(xr).pow(2).sum(1).sqrt() / nx
        crest = lambda Z: float((Z.abs().amax(1) / Z.pow(2).mean(1).sqrt().clamp_min(1e-12)).mean())
        o = dict(crest_raw=crest(te), crest_rot=crest(xr), relerr_int4=float(e0.mean()), N=g.L[i][k].shape[0], K=X.shape[1])
        for kc in (1, 64, 256):
            C = tr.mean(0, keepdim=True) if kc == 1 else kmeans(tr, kc)
            d = (te * te).sum(1, keepdim=True) - 2 * te @ C.t() + (C * C).sum(1)[None, :]
            r = te - C[d.argmin(1)]
            rr = R(r)
            e = q4err(rr).pow(2).sum(1).sqrt() / nx
            o[f'kc{kc}'] = dict(crest_res_rot=crest(rr), res_norm=float((r.pow(2).sum(1).sqrt() / nx).mean()), relerr_int4=float(e.mean()),
                               err_ratio_vs_plain=float(e.mean() / e0.mean()), search_cost_vs_gemm=kc / o['N'])
        out[QL.kname(i, k)] = o
    print(i, f'{time.time()-t0:.0f}s', QL.kname(i, 'Wo'), {kk: (round(v['err_ratio_vs_plain'], 3) if isinstance(v, dict) else v) for kk, v in out[QL.kname(i, 'Wo')].items() if kk.startswith('kc')}, flush=True)
json.dump(out, open(os.path.expanduser('~/work/q1/res_b3.json'), 'w'), indent=1)
print('done', flush=True)
