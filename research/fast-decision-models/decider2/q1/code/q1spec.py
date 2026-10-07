"""Q1 B1.1: eigen-spectrum of the decision-loss gradient second moment at every GEMM, output and input side, state vs question rows.

Loss: KL(p || p_perturbed) of the pointer-head decision distribution; to second order KL = 1/2 d^T F d with F the softmax Fisher, so
G = sum_requests sum_dirs lam_j  g_j^T g_j  where g_j = d(u_j . logits)/d(GEMM output) for the Fisher eigenpairs (lam_j, u_j) (exact for binary).
G_out [N, N] wrt the GEMM output; G_in [K, K] wrt the GEMM input as a quantized kernel sees it (xn for Win/Wgu, so g_in = (g_out W) * gain).
Rows split: state rows [0, q0) vs question rows [q0, T) (question text, options, readout row).
Runs in layer groups (accumulators for 3 layers at a time fit in 22 GB); bf16 dense forward + backward; products exact (bf16 values in TF32), fp32 sums.

python q1spec.py --mode A --group 0      (layers 0-2)  ... --group 7     -> spec/L{i}_{k}.pt + spec/grp{g}.json
python q1spec.py --mode B                 held-out captured fraction on set B with A's eigenvectors  -> spec/heldout.json
"""
import os, sys, json, time, argparse, math
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL

ap = argparse.ArgumentParser()
ap.add_argument('--mode', default='A'); ap.add_argument('--group', type=int, default=0); ap.add_argument('--gsize', type=int, default=3)
ap.add_argument('--nA', type=int, default=256); ap.add_argument('--nB', type=int, default=64); ap.add_argument('--maxT', type=int, default=2600)
ap.add_argument('--out', default=os.path.expanduser('~/work/q1/spec')); ap.add_argument('--topv', type=int, default=1024)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
g = QL.Q1()
dev = g.dev
A, B = QL.req_sets(a.nA, a.nB, a.maxT)
items = A if a.mode == 'A' else B
if a.limit: items = items[:a.limit]
print('requests', len(items), flush=True)
DIMS = {(i, k): tuple(g.L[i][k].shape) for i in range(24) for k in QL.GEMMS}   # (N, K)


def rank_at(ev, f):
    c = torch.cumsum(ev, 0); t = float(c[-1])
    if t <= 0: return 0
    return int(torch.searchsorted(c, torch.tensor(f * t, dtype=c.dtype)).item()) + 1


def summ(ev):
    ev = ev.detach().double().cpu().clamp_min(0)
    t = float(ev.sum())
    return dict(trace=t, r50=rank_at(ev, .5), r90=rank_at(ev, .9), r99=rank_at(ev, .99), r999=rank_at(ev, .999),
                pr=(t * t / float((ev * ev).sum())) if t > 0 else 0.0, top1=float(ev[0]) / t if t > 0 else 0.0,
                top_share={str(r): float(ev[:r].sum()) / t if t > 0 else 0.0 for r in (8, 16, 32, 64, 128, 256, 512, 1024)})


class RowStats:
    """per-request row-energy statistics per GEMM (output side): readout row, option rows, other question rows, state rows; state concentration and position."""
    def __init__(self):
        self.agg = {}

    def add(self, key, rowE, q0, opt_abs):
        T = rowE.shape[0]; tot = float(rowE.sum())
        if tot <= 0: return
        st = rowE[:q0]; qq = rowE[q0:]
        last = float(rowE[T - 1]); opt = float(rowE[torch.tensor(opt_abs, device=rowE.device)].sum())
        s = float(st.sum()); qrest = float(qq.sum()) - last - opt
        conc = {}
        if q0 > 0 and s > 0:
            ss = torch.sort(st, descending=True).values; c = torch.cumsum(ss, 0) / s
            for f in (0.01, 0.1, 0.5):
                conc[str(f)] = float(c[max(0, int(math.ceil(f * q0)) - 1)])
            bins = torch.tensor_split(st, 10)
            pos = [float(b.sum()) / s for b in bins]
        else: pos = [0.0] * 10
        A_ = self.agg.setdefault(key, dict(n=0, tot=0.0, state=0.0, readout=0.0, options=0.0, qrest=0.0, conc={'0.01': 0.0, '0.1': 0.0, '0.5': 0.0}, pos=[0.0] * 10, wconc=0.0))
        A_['n'] += 1; A_['tot'] += tot; A_['state'] += s; A_['readout'] += last; A_['options'] += opt; A_['qrest'] += qrest
        if conc:
            for f in conc: A_['conc'][f] += conc[f] * s
            for j in range(10): A_['pos'][j] += pos[j] * s
            A_['wconc'] += s

    def out(self):
        o = {}
        for key, A_ in self.agg.items():
            t = A_['tot'] or 1.0; w = A_['wconc'] or 1.0
            o[key] = dict(share_state=A_['state'] / t, share_readout=A_['readout'] / t, share_options=A_['options'] / t, share_qrest=A_['qrest'] / t,
                          state_top1pct=A_['conc']['0.01'] / w, state_top10pct=A_['conc']['0.1'] / w, state_top50pct=A_['conc']['0.5'] / w,
                          state_pos_deciles=[p / w for p in A_['pos']])
        return o


def prep_all(items):
    out = []
    for it in items:
        pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']
        out.append((it, pr, ids))
    return out


data = prep_all(items)
print('tokens', sum(len(x[2]) for x in data), 'max T', max(len(x[2]) for x in data), flush=True)

if a.mode == 'A':
    gs = a.gsize; layers = list(range(a.group * gs, min(24, (a.group + 1) * gs)))
    if all(os.path.exists(f'{a.out}/L{i}_{k}.pt') for i in layers for k in QL.GEMMS) and os.path.exists(f'{a.out}/grp{a.group}.json'):
        print('group done', a.group); sys.exit(0)
    track = {(i, k) for i in layers for k in QL.GEMMS}
    G = {}
    for key in track:
        N, K = DIMS[key]
        for r in ('s', 'q'):
            G[(key, 'out', r)] = torch.zeros(N, N, device=dev)
            G[(key, 'in', r)] = torch.zeros(K, K, device=dev)
    print('accumulators GB', sum(v.numel() for v in G.values()) * 4 / 1e9, flush=True)
    st = dict(w=1.0, q0=0, rowE={})
    rs = RowStats()

    def hook(i, k, gr):
        key = (i, k); q0 = st['q0']; w = st['w']
        torch.backends.cuda.matmul.allow_tf32 = True
        go = gr.float()
        W = g.L[i][k]
        gi = go @ W.float()
        gn = g.gain(i, k)
        if gn is not None: gi = gi * gn[None, :]
        for r, sl in (('s', slice(0, q0)), ('q', slice(q0, None))):
            a_ = go[sl]; b_ = gi[sl]
            if a_.shape[0] == 0: continue
            G[(key, 'out', r)].addmm_(a_.t(), a_, alpha=w)
            G[(key, 'in', r)].addmm_(b_.t(), b_, alpha=w)
        torch.backends.cuda.matmul.allow_tf32 = False
        re = st['rowE'].get(key)
        e = w * go.pow(2).sum(1)
        st['rowE'][key] = e if re is None else re + e

    g.hook_fn = hook; g.track = track
    meta = []
    t0 = time.time()
    for di, (it, pr, ids) in enumerate(data):
        q0 = pr['q0']; st['q0'] = q0; st['rowE'] = {}
        h = g.fwdg(ids, gfrom=layers[0])
        lg = g.logits_g(h, pr)
        dirs, cap, p = QL.fisher_dirs(lg)
        for j, (lam, u) in enumerate(dirs):
            st['w'] = lam
            (lg * u).sum().backward(retain_graph=(j < len(dirs) - 1))
        opt_abs = [q0 + o for o in pr['opt']]
        for key, e in st['rowE'].items(): rs.add(QL.kname(*key), e, q0, opt_abs)
        meta.append(dict(rid=it['rid'], task=it['task'], q=it['qname'], T=len(ids), q0=q0, n=len(p), p=p.tolist(), fisher_tr=sum(l for l, _ in dirs), cap=cap, ndirs=len(dirs)))
        del h, lg
        if di % 20 == 0:
            print(f'grp {a.group} {di}/{len(data)} T={len(ids)} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
    g.hook_fn = None
    print('accumulated', f'{time.time()-t0:.0f}s', flush=True)
    res = {}
    nreq = len(data)
    for key in sorted(track):
        i, k = key; N, K = DIMS[key]; sv = {}
        o = dict(N=N, K=K, macs=N * K)
        for side in ('out', 'in'):
            Ms = G.pop((key, side, 's')) / nreq; Mq = G.pop((key, side, 'q')) / nreq
            D = Ms.shape[0]
            for r, M in (('s', Ms), ('q', Mq), ('all', None)):
                if M is None: M = Ms + Mq
                M = 0.5 * (M + M.t())
                vecs = side == 'in' or D <= 2048 or r == 'all'
                if vecs:
                    ev, V = torch.linalg.eigh(M)
                    ev = ev.flip(0); V = V.flip(1)
                    sv[(side, r)] = dict(ev=ev.cpu(), V=V[:, :a.topv].contiguous().cpu())
                else:
                    ev = torch.linalg.eigvalsh(M).flip(0)
                    sv[(side, r)] = dict(ev=ev.cpu())
                o[f'{side}_{r}'] = summ(ev)
                if side == 'in' and r != 'all' and D <= 2048: sv[(side, r)]['G'] = M.cpu()
                del M
            del Ms, Mq
            torch.cuda.empty_cache()
        torch.save(sv, f'{a.out}/L{i}_{k}.pt')
        res[QL.kname(i, k)] = o
        print(QL.kname(i, k), 'out s/q/all r90', o['out_s']['r90'], o['out_q']['r90'], o['out_all']['r90'], 'in r90', o['in_s']['r90'], o['in_q']['r90'], o['in_all']['r90'],
              f'{time.time()-t0:.0f}s', flush=True)
    json.dump(dict(layers=layers, n=nreq, gemms=res, rows=rs.out(), meta=meta), open(f'{a.out}/grp{a.group}.json', 'w'))
    print('done group', a.group, f'{time.time()-t0:.0f}s', flush=True)

else:   # held-out: captured fraction of B's trace by A's top-r eigenvectors
    RG = [8, 16, 32, 64, 128, 256, 512, 1024]
    V = {}
    for i in range(24):
        for k in QL.GEMMS:
            sv = torch.load(f'{a.out}/L{i}_{k}.pt')
            for (side, r), d in sv.items():
                if 'V' in d: V[((i, k), side, r)] = d['V'].to(dev).to(torch.bfloat16)
    print('V loaded GB', sum(v.numel() for v in V.values()) * 2 / 1e9, flush=True)
    acc = {}
    st = dict(w=1.0, q0=0)

    def hook(i, k, gr):
        key = (i, k); q0 = st['q0']; w = st['w']
        go = gr.float(); gi = go @ g.L[i][k].float(); gn = g.gain(i, k)
        if gn is not None: gi = gi * gn[None, :]
        for side, M in (('out', go), ('in', gi)):
            for r, sl in (('s', slice(0, q0)), ('q', slice(q0, None))):
                X = M[sl]
                if X.shape[0] == 0: continue
                tot = float(X.pow(2).sum()) * w
                for src in (r, 'all'):
                    Vv = V.get((key, side, src))
                    if Vv is None: continue
                    pj = (X.to(torch.bfloat16) @ Vv).float().pow(2).sum(0)
                    cs = torch.cumsum(pj, 0)
                    caps = [float(cs[min(rr, cs.shape[0]) - 1]) * w for rr in RG]
                    A_ = acc.setdefault((QL.kname(*key), side, r, src), dict(tot=0.0, cap=[0.0] * len(RG)))
                    A_['tot'] += tot
                    for j in range(len(RG)): A_['cap'][j] += caps[j]

    g.hook_fn = hook; g.track = None
    t0 = time.time()
    for di, (it, pr, ids) in enumerate(data):
        st['q0'] = pr['q0']
        h = g.fwdg(ids, gfrom=0)
        lg = g.logits_g(h, pr)
        dirs, cap, p = QL.fisher_dirs(lg)
        for j, (lam, u) in enumerate(dirs):
            st['w'] = lam
            (lg * u).sum().backward(retain_graph=(j < len(dirs) - 1))
        if di % 10 == 0: print(f'B {di}/{len(data)} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
    out = {}
    for (kn, side, r, src), A_ in acc.items():
        out.setdefault(kn, {})[f'{side}_{r}_by_{src}'] = dict(tot=A_['tot'] / len(data), cap={str(rr): A_['cap'][j] / A_['tot'] if A_['tot'] > 0 else 0.0 for j, rr in enumerate(RG)})
    json.dump(dict(n=len(data), r_grid=RG, gemms=out), open(f'{a.out}/heldout.json', 'w'))
    print('done heldout', f'{time.time()-t0:.0f}s', flush=True)
