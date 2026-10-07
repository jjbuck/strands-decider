"""Q5 calibration pass: one decision-gradient pass over a train-split request set of one deployment domain.

For every GEMM (i, k) in --layers, per row role (s = state rows [0, q0), q = question rows):
  Hs/Hq   = sum_t x_t x_t^T                       (plain input second moment; x = what the kernel sees, unrotated: xn for Win/Wgu, mixer out for Wo, m for Wd)
  Hds/Hdq = sum_t w_t x_t x_t^T, w_t = sum_j lam_j ||g_jt||^2   (decision-weighted: g_jt = d(u_j . logits)/d y_t, (lam_j, u_j) = softmax-Fisher eigenpairs)
  sum_*, sumd_*, n_*, w_* (first moments, counts, weight sums)
  Fr      = sum_req sum_j lam_j (Gamma_j)^2 , Gamma_j = R_out(g_j)^T R_in(X)  : the decision Fisher diagonal of the weights in the KERNEL basis
  Fn      = same in the natural basis (optional --fn)
  Gout    = sum_j lam_j g_j^T g_j (Wo, Wd only: output-side [2048, 2048])
Neurons (--neur, all 24 layers, input of Wd m = silu(gate) * up): sum, sq, abs, max |m|, fire counts (|m| > 1e-3 / 1e-2 / 1e-1), per request and
  direction a = sum_t gm_t m_t, b = sum_t gm_t (gm = d/dm): A2 = sum lam a^2, AB = sum lam a b, B2 = sum lam b^2 (so the mean-replacement saliency
  sum lam (a - mu b)^2 can be formed for any mu afterwards); per question name A2 too (B7 per-question).
python q5cal.py --dom bank --n 160 --layers 12-15 [--neur] [--fn]   -> ~/work/q5/cal/<dom>/{H,Fr,Fn,Gout}_{i}_{k}.pt, neurons.pt
"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/q5'), os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q5lib as Q5

ap = argparse.ArgumentParser(); ap.add_argument('--dom', required=True); ap.add_argument('--n', type=int, default=160)
ap.add_argument('--layers', default=''); ap.add_argument('--neur', action='store_true'); ap.add_argument('--fn', action='store_true')
ap.add_argument('--nofisher', action='store_true'); ap.add_argument('--frs', action='store_true'); ap.add_argument('--maxT', type=int, default=2600)
a = ap.parse_args()
out = f'{Q5.CAL}/{a.dom}'; os.makedirs(out, exist_ok=True)
LAY = []
for part in a.layers.split(','):
    if not part: continue
    if '-' in part: lo, hi = map(int, part.split('-')); LAY += list(range(lo, hi + 1))
    else: LAY.append(int(part))
LAY = sorted(set(LAY))
torch.backends.cuda.matmul.allow_tf32 = False
g = QL.Q1(); dev = g.dev
if a.dom.endswith('2'):                     # a second, disjoint calibration set from the same calibration tasks (rids of set 1 excluded)
    base = a.dom[:-1]; first = {it['rid'] for it in Q5.req_set(base, 'cal', a.n, maxT=a.maxT)}
    items = [it for it in Q5.req_set(base, 'cal', 4 * a.n, maxT=a.maxT, per_task=24) if it['rid'] not in first][:a.n]
else:
    items = Q5.req_set(a.dom, 'cal', a.n, maxT=a.maxT)
print('requests', len(items), 'layers', LAY, 'neur', a.neur, flush=True)
KEYS = [(i, k) for i in LAY for k in Q5.GEMMS]
acc = {}
for (i, k) in KEYS:
    N, K = g.L[i][k].shape
    acc[(i, k)] = dict(Hs=torch.zeros(K, K, device=dev), Hq=torch.zeros(K, K, device=dev), Hds=torch.zeros(K, K, device=dev), Hdq=torch.zeros(K, K, device=dev),
                       sum_s=torch.zeros(K, device=dev, dtype=torch.float64), sum_q=torch.zeros(K, device=dev, dtype=torch.float64),
                       sumd_s=torch.zeros(K, device=dev, dtype=torch.float64), sumd_q=torch.zeros(K, device=dev, dtype=torch.float64),
                       n_s=0, n_q=0, w_s=0.0, w_q=0.0)
FR = {key: torch.zeros(g.L[key[0]][key[1]].shape, device=dev) for key in KEYS} if not a.nofisher else {}
FN = {key: torch.zeros(g.L[key[0]][key[1]].shape, device=dev) for key in KEYS} if a.fn else {}
FRS = {key: torch.zeros(g.L[key[0]][key[1]].shape, device=dev) for key in KEYS} if (a.frs and not a.nofisher) else {}
GO = {key: torch.zeros(2048, 2048, device=dev) for key in KEYS if key[1] in ('Wo', 'Wd')}
NE = None
if a.neur:
    z = lambda: torch.zeros(24, 6144, device=dev, dtype=torch.float64)
    NE = dict(sum=z(), sq=z(), abs=z(), max=torch.zeros(24, 6144, device=dev), f3=z(), f2=z(), f1=z(), A2=z(), AB=z(), B2=z(),
              sum_s=z(), sq_s=z(), A2s=z(), ABs=z(), B2s=z(), rows=0, rows_s=0, wsum=0.0, perq={})
st = dict(X={}, w={}, lam=1.0, q0=0, ab={})
TRACK = set(KEYS) | ({(i, 'Wd') for i in range(24)} if a.neur else set())
g.track = TRACK


def fwd_fn(i, k, x, xn, y):
    st['X'][(i, k)] = (xn if k in ('Win', 'Wgu') else x).detach().to(torch.bfloat16)


def hook(i, k, gr):
    lam = st['lam']; key = (i, k); q0 = st['q0']
    grf = gr.float()
    if key in acc:
        rn = grf.pow(2).sum(1)
        st['w'][key] = st['w'].get(key, 0) + lam * rn
        if FR or FN:
            X = st['X'][key].float()
            torch.backends.cuda.matmul.allow_tf32 = True
            if FR:
                Xr = g.rot_for(i, k)(X)
                gr_ = g.rots()['R1'](grf) if (k in ('Wo', 'Wd') and g.opt['rout']) else grf
                Gm = gr_.t() @ Xr
                FR[key].add_(Gm.pow(2), alpha=lam); del Gm
                if FRS and q0 > 0:
                    Gm = gr_[:q0].t() @ Xr[:q0]; FRS[key].add_(Gm.pow(2), alpha=lam); del Gm
                del Xr, gr_
            if FN:
                Gm = grf.t() @ X; FN[key].add_(Gm.pow(2), alpha=lam); del Gm
            torch.backends.cuda.matmul.allow_tf32 = False
        if key in GO:
            GO[key].addmm_(grf.t(), grf, alpha=lam)
    if NE is not None and k == 'Wd':
        X = st['X'][key].float()
        gi = grf @ g.L[i]['Wd'].float()                       # d/dm [T, 6144]
        a_ = (gi * X).sum(0).double(); b_ = gi.sum(0).double()
        a2 = (gi[:q0] * X[:q0]).sum(0).double(); b2 = gi[:q0].sum(0).double()
        NE['A2'][i] += lam * a_ * a_; NE['AB'][i] += lam * a_ * b_; NE['B2'][i] += lam * b_ * b_
        NE['A2s'][i] += lam * a2 * a2; NE['ABs'][i] += lam * a2 * b2; NE['B2s'][i] += lam * b2 * b2
        pq = st['ab'].setdefault(i, [])
        pq.append((lam, a_, b_))


g.fwd_fn = fwd_fn; g.hook_fn = hook
t0 = time.time(); ntok = 0
for di, it in enumerate(items):
    pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']; T = len(ids); q0 = pr['q0']; st['q0'] = q0; ntok += T
    st['X'] = {}; st['w'] = {}; st['ab'] = {}
    h = g.fwdg(ids, gfrom=0, q0=q0); lg = g.logits_g(h, pr)
    dirs, cap, p = QL.fisher_dirs(lg)
    for j, (lam, u) in enumerate(dirs):
        st['lam'] = lam
        (lg * u).sum().backward(retain_graph=(j < len(dirs) - 1))
    # accumulate per request
    for key in KEYS:
        A = acc[key]; X = st['X'][key].float(); w = st['w'].get(key)
        if w is None: w = torch.zeros(T, device=dev)
        torch.backends.cuda.matmul.allow_tf32 = True
        Xs, Xq = X[:q0], X[q0:]
        A['Hs'].addmm_(Xs.t(), Xs); A['Hq'].addmm_(Xq.t(), Xq)
        sw = w.clamp_min(0).sqrt()[:, None]
        Xw = X * sw
        A['Hds'].addmm_(Xw[:q0].t(), Xw[:q0]); A['Hdq'].addmm_(Xw[q0:].t(), Xw[q0:])
        torch.backends.cuda.matmul.allow_tf32 = False
        A['sum_s'] += Xs.double().sum(0); A['sum_q'] += Xq.double().sum(0)
        A['sumd_s'] += (Xs * w[:q0, None]).double().sum(0); A['sumd_q'] += (Xq * w[q0:, None]).double().sum(0)
        A['n_s'] += q0; A['n_q'] += T - q0; A['w_s'] += float(w[:q0].sum()); A['w_q'] += float(w[q0:].sum())
    if NE is not None:
        for i in range(24):
            m = st['X'][(i, 'Wd')].float(); am = m.abs()
            NE['sum'][i] += m.sum(0).double(); NE['sq'][i] += m.pow(2).sum(0).double(); NE['abs'][i] += am.sum(0).double()
            NE['max'][i] = torch.maximum(NE['max'][i], am.amax(0))
            NE['f3'][i] += (am > 1e-3).sum(0).double(); NE['f2'][i] += (am > 1e-2).sum(0).double(); NE['f1'][i] += (am > 1e-1).sum(0).double()
            NE['sum_s'][i] += m[:q0].sum(0).double(); NE['sq_s'][i] += m[:q0].pow(2).sum(0).double()
        NE['rows'] += T; NE['rows_s'] += q0; NE['wsum'] += sum(l for l, _ in dirs)
        pq = NE['perq'].setdefault(it['qname'], dict(A2=torch.zeros(24, 6144, dtype=torch.float64, device=dev), n=0))
        for i, lst in st['ab'].items():
            for lam, a_, b_ in lst: pq['A2'][i] += lam * a_ * a_
        pq['n'] += 1
    del h, lg; st['X'] = {}; st['w'] = {}; st['ab'] = {}
    if di % 10 == 0:
        print(di, T, f'{time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
for key in KEYS:
    i, k = key; A = acc[key]
    torch.save({kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in A.items()}, f'{out}/H_{i}_{k}.pt')
    if FR: torch.save(FR[key].cpu(), f'{out}/Fr_{i}_{k}.pt')
    if FN: torch.save(FN[key].cpu(), f'{out}/Fn_{i}_{k}.pt')
    if FRS: torch.save(FRS[key].cpu(), f'{out}/Frs_{i}_{k}.pt')
    if key in GO: torch.save(GO[key].cpu(), f'{out}/Gout_{i}_{k}.pt')
if NE is not None:
    sv = {kk: (v.cpu() if torch.is_tensor(v) else v) for kk, v in NE.items() if kk != 'perq'}
    sv['perq'] = {q: dict(A2=v['A2'].cpu(), n=v['n']) for q, v in NE['perq'].items()}
    torch.save(sv, f'{out}/neurons.pt')
json.dump(dict(dom=a.dom, n=len(items), tokens=ntok, layers=LAY, neur=a.neur, secs=time.time() - t0,
               rids=[it['rid'] for it in items]), open(f'{out}/meta_{a.layers or "none"}.json', 'w'))
print('done', f'{time.time()-t0:.0f}s tokens {ntok}', flush=True)
