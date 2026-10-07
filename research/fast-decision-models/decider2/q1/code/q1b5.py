"""B5 planner: decision-weighted transform coding of each GEMM input, hardware bit groups {8, 4, 0}.

For GEMM input x (role r: state rows by default) with centered covariance C (q1cal) and per-row decision sensitivity Lam = G_in(r) / rows per request (q1spec):
  basis U (orthogonal, K x K) from candidates; coordinate variance c_i = (U^T C U)_ii, sensitivity l_i = (U^T Lam U)_ii, priority p_i = c_i l_i;
  sort by p_i; slices: top n8 at int8, next n4 at int4, last n0 dropped (replaced by the mean -> exact bias mu W^T). Budget: 8 n8 + 4 n4 = B K.
  Inside each coded slice a Haar rotation spreads the variance; per-token absmax quantization. Distortion model (first-order decision variance):
    D = sum_{coded S} eps_S tr(C_SS) tr(Lam_SS) / |S|  +  tr(C_00 Lam_00)            (exact for the dropped slice)
  eps4 = 0.0225 (15% rel. rms, measured for rotated per-token int4), eps8 = eps4 / 256. Baseline: rotated int4 of all K: eps4 tr(M2) tr(Lam) / K
  (uncentered, as H1/H2), and centered: eps4 tr(C) tr(Lam) / K.
Candidates: pca (eig C), sens (eig Lam), joint (eig Lam^1/2 C Lam^1/2), jointc (eig C^1/2 Lam C^1/2), mix (eig C/trC + Lam/trLam).
python q1b5.py --budget 4 --role s --tag b4s   -> ~/work/q1/b5/{tag}/P_{i}_{k}.pt + ~/work/q1/b5/{tag}.json
"""
import os, sys, json, time, argparse, math
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL

ap = argparse.ArgumentParser(); ap.add_argument('--budget', type=float, default=4.0); ap.add_argument('--role', default='s'); ap.add_argument('--tag', required=True)
ap.add_argument('--eps4', type=float, default=0.0225); ap.add_argument('--save', type=int, default=1); ap.add_argument('--cands', default='pca,sens,joint,jointc,mix'); ap.add_argument('--shrink', type=float, default=0.0); ap.add_argument('--rot', default='haar'); ap.add_argument('--only', default='')
a = ap.parse_args()
dev = 'cuda'
SPEC = os.path.expanduser('~/work/q1/spec'); CAL = os.path.expanduser('~/work/q1/cal'); OUT = os.path.expanduser(f'~/work/q1/b5/{a.tag}')
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f'{SPEC}/grp7.json'))['meta']
rows_s = sum(m['q0'] for m in meta) / len(meta); rows_q = sum(m['T'] - m['q0'] for m in meta) / len(meta)
eps4 = a.eps4; eps8 = eps4 / 256.0


def sqrtm(M):
    ev, V = torch.linalg.eigh(M); return (V * ev.clamp_min(0).sqrt()[None, :]) @ V.t()


def lam_of(i, k, role):
    sv = torch.load(f'{SPEC}/L{i}_{k}.pt', map_location=dev)
    d = sv[('in', role)]
    if 'G' in d: G = d['G'].float()
    else:                                   # top-1024 eigenpairs + the remaining trace spread isotropically over the complement
        V = d['V'].float(); evall = d['ev'].float().clamp_min(0); ev = evall[:V.shape[1]]; K_ = V.shape[0]
        rest = float(evall[V.shape[1]:].sum()) / max(1, K_ - V.shape[1])
        G = (V * (ev - rest)[None, :]) @ V.t() + rest * torch.eye(K_, device=V.device)
    G = 0.5 * (G + G.t()) / (rows_s if role == 's' else rows_q)
    if a.shrink > 0:                        # shrink toward isotropic: held-out tasks put 5-20% of their trace outside A's top directions
        G = (1 - a.shrink) * G + a.shrink * float(G.diagonal().sum()) / G.shape[0] * torch.eye(G.shape[0], device=G.device)
    return G


def plan(C, Lam, M2, budget):
    K = C.shape[0]; trC = float(C.diagonal().sum()); trL = float(Lam.diagonal().sum())
    base_unc = eps4 * float(M2.diagonal().sum()) * trL / K; base_c = eps4 * trC * trL / K
    cands = {}; want = a.cands.split(',')
    if 'shared' in want:
        if K == 2048 and CUR[1] in ('Win', 'Wgu'): cands['shared'] = USH
        else: cands['ident'] = torch.eye(K, device=C.device)
    if 'ident' in want: cands['ident'] = torch.eye(K, device=C.device)
    if 'pca' in want: ev, V = torch.linalg.eigh(C); cands['pca'] = V
    if 'sens' in want: ev, V = torch.linalg.eigh(Lam); cands['sens'] = V
    if 'joint' in want: Ls = sqrtm(Lam); ev, V = torch.linalg.eigh(Ls @ C @ Ls); cands['joint'] = V
    if 'jointc' in want: Cs = sqrtm(C); ev, V = torch.linalg.eigh(Cs @ Lam @ Cs); cands['jointc'] = V
    if 'mix' in want: Mx = C / max(trC, 1e-30) + Lam / max(trL, 1e-30); ev, V = torch.linalg.eigh(0.5 * (Mx + Mx.t())); cands['mix'] = V
    best = None; res = {}
    for nm, U in cands.items():
        Cu = U.t() @ C @ U; Lu = U.t() @ Lam @ U
        p = Cu.diagonal() * Lu.diagonal()
        order = torch.argsort(p, descending=True); U = U[:, order]; Cu = Cu[order][:, order]; Lu = Lu[order][:, order]
        cdiag = Cu.diagonal().double().cpu(); ldiag = Lu.diagonal().double().cpu()
        cc = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(cdiag, 0)]); lc = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(ldiag, 0)])
        # tr(C_00 Lam_00) for the trailing block of size n0: computed for a grid
        grid = list(range(0, K // 2 + 1, 64))          # K slices in multiples of 64 (kernel tiles)
        bestD = None
        def hadok(n):
            if n == 0: return True
            m = n // 12 if n % 12 == 0 else n
            return m & (m - 1) == 0
        for n8 in grid:
            # budget: 8 n8 + 4 n4 = budget K, n0 = K - n8 - n4
            n4 = int(round((budget * K - 8 * n8) / 4 / 64)) * 64
            if n4 < 0 or n8 + n4 > K: continue
            if a.rot == 'hadfull' and not (hadok(n8) and hadok(n4)):
                # largest Hadamard-able n4 not above the budget
                n4s = [m for m in range(n4, -1, -64) if hadok(m)]
                if not n4s: continue
                n4 = n4s[0]
                if not hadok(n8): continue
            n0 = K - n8 - n4
            D = 0.0
            if n8: D += eps8 * float(cc[n8] - cc[0]) * float(lc[n8] - lc[0]) / n8
            if n4: D += eps4 * float(cc[n8 + n4] - cc[n8]) * float(lc[n8 + n4] - lc[n8]) / n4
            if n0: D += float((Cu[K - n0:, K - n0:] * Lu[K - n0:, K - n0:]).sum())
            if bestD is None or D < bestD[0]: bestD = (D, n8, n4, n0)
        res[nm] = dict(D=bestD[0], n8=bestD[1], n4=bestD[2], n0=bestD[3], rel_unc=bestD[0] / base_unc, rel_c=bestD[0] / base_c)
        if best is None or bestD[0] < best[0]: best = (bestD[0], nm, U, bestD[1], bestD[2], bestD[3])
    return dict(base_unc=base_unc, base_c=base_c, cands=res, best=best[1], U=best[2], n8=best[3], n4=best[4], n0=best[5], D=best[0])


out = {}; t0 = time.time()
keys = [(i, k) for i in range(24) for k in QL.GEMMS]
USH = None
if 'shared' in a.cands:   # one basis for all residual readers (Win, Wgu): eig of sum_l (C_l / tr C_l + Lam_l / tr Lam_l); deployable as the offline residual rotation
    Msum = torch.zeros(2048, 2048, device=dev)
    for (i, k) in keys:
        if k not in ('Win', 'Wgu'): continue
        S = torch.load(f'{CAL}/S_{i}_{k}.pt', map_location=dev); r = a.role
        n = S[f'n_{r}']; mu = (S[f'sum_{r}'] / n).float(); C = S[f'M2_{r}'].float() / n - torch.outer(mu, mu)
        Lm = lam_of(i, k, r); tL = float(Lm.diagonal().sum())
        if tL <= 0: continue
        Msum += C / float(C.diagonal().sum()) + Lm / tL
    ev, USH = torch.linalg.eigh(0.5 * (Msum + Msum.t()))
    print('shared basis built', f'{time.time()-t0:.0f}s', flush=True)
if a.only: keys = [tuple([int(s.split('.')[0]), s.split('.')[1]]) for s in a.only.split(',')]
for (i, k) in keys:
    S = torch.load(f'{CAL}/S_{i}_{k}.pt', map_location=dev)
    r = a.role
    n = S[f'n_{r}']; mu = (S[f'sum_{r}'] / n).float(); M2 = S[f'M2_{r}'].float() / n
    C = M2 - torch.outer(mu, mu); C = 0.5 * (C + C.t())
    Lam = lam_of(i, k, r)
    if float(Lam.diagonal().sum()) <= 0:
        out[QL.kname(i, k)] = dict(skip='zero sensitivity'); continue
    CUR = (i, k)
    P = plan(C.double().float(), Lam, M2, a.budget)
    K = C.shape[0]
    out[QL.kname(i, k)] = dict(K=K, n8=P['n8'], n4=P['n4'], n0=P['n0'], best=P['best'], D=P['D'], base_unc=P['base_unc'], base_c=P['base_c'],
                               gain_vs_unc=P['base_unc'] / P['D'] if P['D'] > 0 else float('inf'), cands=P['cands'])
    if a.save:
        g_ = torch.Generator(device='cpu'); g_.manual_seed(1000 + i * 4 + ['Win', 'Wo', 'Wgu', 'Wd'].index(k))
        def haar(n):
            if n == 0: return torch.zeros(0, 0)
            if a.rot == 'hadfull':        # deployable fast transform: random signs + full-slice Kronecker Hadamard (n = 2^a or 12 * 2^a)
                from g2lib import hadamard, paley12
                m = 12 if n % 12 == 0 and ((n // 12) & (n // 12 - 1)) == 0 else 1
                p2 = n // m; assert p2 & (p2 - 1) == 0, n
                H = hadamard(p2, 'cpu') if m == 1 else torch.kron(paley12('cpu'), hadamard(p2, 'cpu'))
                sg = (torch.randint(0, 2, (n,), generator=g_) * 2 - 1).float()
                return (sg[:, None] * H).float()
            if a.rot == 'had64':          # deployable: random signs + block-diagonal Hadamard of order 64 (slices are multiples of 64)
                from g2lib import hadamard
                H = hadamard(64, 'cpu'); sg = (torch.randint(0, 2, (n,), generator=g_) * 2 - 1).float()
                return (sg[:, None] * torch.block_diag(*([H] * (n // 64)))).float()
            q, r_ = torch.linalg.qr(torch.randn(n, n, generator=g_, dtype=torch.float64)); return (q * torch.sign(torch.diagonal(r_))[None, :]).float()
        torch.save(dict(U=P['U'].cpu(), mu=mu.cpu(), n8=P['n8'], n4=P['n4'], n0=P['n0'], R8=haar(P['n8']), R4=haar(P['n4']), C=C.cpu()), f'{OUT}/P_{i}_{k}.pt')
    print(QL.kname(i, k), 'K', K, 'n8/n4/n0', P['n8'], P['n4'], P['n0'], P['best'], f"D/base_unc {P['D']/P['base_unc']:.4f} D/base_c {P['D']/P['base_c']:.4f}",
          {nm: round(c['rel_unc'], 4) for nm, c in P['cands'].items()}, f'{time.time()-t0:.0f}s', flush=True)
json.dump(dict(budget=a.budget, role=a.role, eps4=eps4, rows_s=rows_s, rows_q=rows_q, gemms=out), open(f'{OUT}.json', 'w'), indent=1)
print('done', flush=True)
