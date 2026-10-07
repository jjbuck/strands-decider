"""Decision-weighted statistics, one gradient pass over set A (train split):
 (1) decision-weighted input Hessian per GEMM: Hd = sum_req sum_dirs lam sum_t ||g_out,t||^2 x_t^T x_t / sum(weights)  (B11 2:4 masks, GPTQ-d);
     with E[(g_t dW x_t)^2] ~ ||g_t||^2 ||dW x_t||^2 / N for an isotropic per-row output sensitivity, Hd is the decision version of GPTQ's H;
 (2) per MLP neuron j of every layer (input of Wd: m_j = silu(gate_j) * up_j), by domain (banking / retail):
     firing: mean |m_j|, fraction of rows with |m_j| > tol, E[m_j^2];
     first-order decision effect of removing neuron j: S_j = sum_t g_tj m_tj (zeroing) and S'_j = sum_t g_tj (m_tj - mu_j) (mean replacement),
     accumulated as Fisher-weighted E[S_j^2] over requests.  (B12)
python q1hd.py --n 160 -> ~/work/q1/hd/Hd_{i}_{k}.pt, ~/work/q1/hd/neurons.pt"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL

ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=160); ap.add_argument('--tol', type=float, default=1e-3)
ap.add_argument('--out', default=os.path.expanduser('~/work/q1/hd')); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
g = QL.Q1(); dev = g.dev
A, _ = QL.req_sets()
items = A[:a.n]
# neuron means (for mean replacement) from the calibration stats if present
MU = {}
for i in range(24):
    f = os.path.expanduser(f'~/work/q1/cal/S_{i}_Wd.pt')
    if os.path.exists(f):
        S = torch.load(f); MU[i] = ((S['sum_s'] + S['sum_q']) / (S['n_s'] + S['n_q'])).float().to(dev)
Hd = {}; wsum = [0.0]
NE = {dom: dict(abs=torch.zeros(24, 6144, device=dev, dtype=torch.float64), fire=torch.zeros(24, 6144, device=dev, dtype=torch.float64),
                m2=torch.zeros(24, 6144, device=dev, dtype=torch.float64), S2=torch.zeros(24, 6144, device=dev, dtype=torch.float64),
                S2m=torch.zeros(24, 6144, device=dev, dtype=torch.float64), rows=0, w=0.0) for dom in ('banking', 'retail')}
st = dict(w=1.0, X={}, S={}, Sm={})


def fwd_fn(i, k, x, xn, y):
    src = (xn if k in ('Win', 'Wgu') else x).detach()
    st['X'][(i, k)] = src.to(torch.bfloat16)


def hook(i, k, gr):
    X = st['X'].pop((i, k)).float(); w = st['w']
    rn = gr.float().pow(2).sum(1)                         # per-row output-gradient energy
    Xw = X * rn.sqrt()[:, None]
    M = Hd.get((i, k))
    torch.backends.cuda.matmul.allow_tf32 = True
    if M is None: M = Hd[(i, k)] = torch.zeros(X.shape[1], X.shape[1], device=dev)
    M.addmm_(Xw.t(), Xw, alpha=w)
    if k == 'Wd':
        gi = gr.float() @ g.L[i]['Wd'].float()           # gradient wrt m [T, 6144]
        s = (gi * X).sum(0)
        st['S'].setdefault(i, []).append(s)
        if i in MU: st['Sm'].setdefault(i, []).append((gi * (X - MU[i][None, :])).sum(0))
    torch.backends.cuda.matmul.allow_tf32 = False


g.fwd_fn = fwd_fn; g.hook_fn = hook
t0 = time.time()
for di, it in enumerate(items):
    dom = 'banking' if it['task'].startswith('banking') else 'retail'
    pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']
    st['X'] = {}; st['S'] = {}; st['Sm'] = {}
    h = g.fwdg(ids, gfrom=0, q0=pr['q0']); lg = g.logits_g(h, pr)
    # firing stats from the captured Wd inputs (dense trajectory)
    ne = NE[dom]
    for i in range(24):
        m = st['X'][(i, 'Wd')].float()
        ne['abs'][i] += m.abs().sum(0).double(); ne['fire'][i] += (m.abs() > a.tol).sum(0).double(); ne['m2'][i] += m.pow(2).sum(0).double()
    ne['rows'] += len(ids)
    dirs, cap, p = QL.fisher_dirs(lg)
    Xkeep = dict(st['X'])
    for j, (lam, u) in enumerate(dirs):
        st['w'] = lam; st['X'] = dict(Xkeep)
        (lg * u).sum().backward(retain_graph=(j < len(dirs) - 1))
        for i, lst in st['S'].items(): ne['S2'][i] += lam * lst[-1].double().pow(2)
        for i, lst in st['Sm'].items(): ne['S2m'][i] += lam * lst[-1].double().pow(2)
        st['S'] = {}; st['Sm'] = {}
        wsum[0] += lam
    ne['w'] += sum(l for l, _ in dirs)
    del h, lg, Xkeep; st['X'] = {}
    if di % 20 == 0: print(di, len(ids), f'{time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
for (i, k), M in Hd.items():
    torch.save((M / wsum[0]).cpu(), f'{a.out}/Hd_{i}_{k}.pt')
torch.save({dom: {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in ne.items()} for dom, ne in NE.items()}, f'{a.out}/neurons.pt')
print('done', f'{time.time()-t0:.0f}s', flush=True)
