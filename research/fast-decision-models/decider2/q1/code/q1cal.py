"""Q1 calibration statistics of every GEMM input (what a quantized kernel sees: xn for Win/Wgu, mixer output for Wo, silu(g)*u for Wd),
on set A requests (train split), dense bf16 forward:
  mean and second moment per row role (s/q), per-token crest (max/rms) before and after the H1 rotation, and a row sample (for B3 k-means).
python q1cal.py --n 128 -> ~/work/q1/cal/S_{i}_{k}.pt  (dict: n_s, n_q, sum_s, sum_q, M2_s, M2_q, crest stats), ~/work/q1/cal/X_{i}_{k}.pt (row sample, bf16)"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=128); ap.add_argument('--rows', type=int, default=48)
ap.add_argument('--out', default=os.path.expanduser('~/work/q1/cal')); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
g = QL.Q1(grad=False); dev = g.dev
A, _ = QL.req_sets()
items = A[:a.n]
acc = {}; samp = {}
gen = torch.Generator(device=dev); gen.manual_seed(0)
st = dict(q0=0)


def fwd_fn(i, k, x, xn, y):
    src = (xn if k in ('Win', 'Wgu') else x).float(); q0 = st['q0']; T = src.shape[0]
    K = src.shape[1]
    d = acc.get((i, k))
    if d is None:
        d = acc[(i, k)] = dict(n_s=0, n_q=0, sum_s=torch.zeros(K, device=dev, dtype=torch.float64), sum_q=torch.zeros(K, device=dev, dtype=torch.float64),
                               M2_s=torch.zeros(K, K, device=dev), M2_q=torch.zeros(K, K, device=dev), crest_raw=[0.0, 0.0], crest_rot=[0.0, 0.0], ncr=[0, 0])
    xr = g.rot_for(i, k)(src)
    for j, (r, sl) in enumerate((('s', slice(0, q0)), ('q', slice(q0, T)))):
        X = src[sl]
        if X.shape[0] == 0: continue
        d[f'n_{r}'] += X.shape[0]; d[f'sum_{r}'] += X.double().sum(0); d[f'M2_{r}'].addmm_(X.t(), X)
        rms = X.pow(2).mean(1).sqrt().clamp_min(1e-12)
        d['crest_raw'][j] += float((X.abs().amax(1) / rms).sum())
        Xr = xr[sl]; d['crest_rot'][j] += float((Xr.abs().amax(1) / Xr.pow(2).mean(1).sqrt().clamp_min(1e-12)).sum()); d['ncr'][j] += X.shape[0]
    idx = torch.randperm(T, device=dev, generator=gen)[:a.rows]
    samp.setdefault((i, k), []).append(torch.cat([src[idx].to(torch.bfloat16), (idx >= q0).to(torch.bfloat16)[:, None]], 1).cpu())


g.fwd_fn = fwd_fn
t0 = time.time()
for di, it in enumerate(items):
    pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']; st['q0'] = pr['q0']
    with torch.no_grad(): g.fwd(ids, q0=pr['q0'])
    if di % 20 == 0: print(di, len(ids), f'{time.time()-t0:.0f}s', flush=True)
for (i, k), d in acc.items():
    out = {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in d.items()}
    torch.save(out, f'{a.out}/S_{i}_{k}.pt')
    torch.save(torch.cat(samp[(i, k)], 0), f'{a.out}/X_{i}_{k}.pt')
summary = {QL.kname(i, k): dict(crest_raw_s=d['crest_raw'][0] / max(1, d['ncr'][0]), crest_raw_q=d['crest_raw'][1] / max(1, d['ncr'][1]),
                                crest_rot_s=d['crest_rot'][0] / max(1, d['ncr'][0]), crest_rot_q=d['crest_rot'][1] / max(1, d['ncr'][1]), n_s=d['n_s'], n_q=d['n_q'],
                                m2tr_s=float(d['M2_s'].diagonal().sum()) / max(1, d['n_s']), m2tr_q=float(d['M2_q'].diagonal().sum()) / max(1, d['n_q']),
                                ctr_s=(float(d['M2_s'].diagonal().sum()) / max(1, d['n_s']) - float((d['sum_s'] / max(1, d['n_s'])).pow(2).sum())))
           for (i, k), d in acc.items()}
json.dump(summary, open(f'{a.out}/summary.json', 'w'), indent=1)
print('done', f'{time.time()-t0:.0f}s', flush=True)
