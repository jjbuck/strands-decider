"""Step 1a: second moments of every GEMM input over state tokens of N train-split states; spectra and thin-path bases.
python fit.py [N=128] [maxtok=4000]  ->  ~/work/j13/bases.pt  ({'in': [l][g] P [K,512], 'out': [l][g] U [N,512]}),  ~/work/j13/spectra.json"""
import os, sys, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from tw import TW, GEMMS, RMAX, SINK

N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
OUT = os.path.expanduser('~/work/j13')
os.makedirs(OUT, exist_ok=True)
KIT = os.path.expanduser('~/work/evalkit')
ev = set(json.load(open(f'{KIT}/split.json'))['eval_tasks'])
pool = [json.loads(l) for l in open(f'{KIT}/train_pool.jsonl')]
pool = [r for r in pool if r['task'] not in ev and 200 <= r['n_state_tok'] <= MAXT]
rng = random.Random(1313); rng.shuffle(pool)
pool = pool[:N]
print('states', len(pool), 'tokens', sum(r['n_state_tok'] for r in pool), flush=True)

tw = TW()
S = {}; cnt = {}

def collect(l, g, a):
    xs = a[SINK:tw._q0].float()
    if (l, g) not in S:
        S[(l, g)] = torch.zeros(xs.shape[1], xs.shape[1], device='cuda', dtype=torch.float32); cnt[(l, g)] = 0
    S[(l, g)].addmm_(xs.t(), xs); cnt[(l, g)] += xs.shape[0]

t0 = time.time()
with torch.no_grad():
    for i, r in enumerate(pool):
        qn = next(iter(r['questions']))
        pr = tw.prep(r, qn)
        tw._q0 = pr['q0']
        x, cos, sin = tw.embed_rope(pr['s'])          # state rows only (causal: identical to the state part of any prompt)
        tw.run(x, cos, sin, collect=collect)
        if i % 16 == 0: print(i, pr['q0'], '%.0fs' % (time.time() - t0), 'mem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)

spec = {'n_states': len(pool), 'n_tok': cnt[(0, 'in')], 'layers': []}
bases = {'in': [], 'out': []}

def ranks(e):
    c = torch.cumsum(e, 0) / e.sum()
    return {f'r{int(p*100)}': int((c < p).sum().item()) + 1 for p in (0.9, 0.95, 0.99)}, {f'e@{r}': round(float(c[min(r, len(c)) - 1]), 4) for r in (64, 128, 256, 512, 1024)}

for l in range(tw.NL):
    rec = {'layer': l, 'type': tw.types[l]}
    bi = {}; bo = {}
    for g in GEMMS:
        Sg = (S.pop((l, g)).double() / cnt[(l, g)])
        ev_, Q = torch.linalg.eigh(Sg)                     # ascending
        ev_ = ev_.flip(0).clamp_min(0); Q = Q.flip(1)
        ri, ei = ranks(ev_)
        bi[g] = Q[:, :RMAX].float().cpu()
        W = tw.L[l]['W'][g].double()                       # [N, K]
        M = W @ (Q * ev_.sqrt()[None, :])                  # same left singular vectors as W Sigma^1/2
        Nn, K = M.shape
        if Nn <= K:
            e2, V = torch.linalg.eigh(M @ M.t()); e2 = e2.flip(0).clamp_min(0); U = V.flip(1)[:, :RMAX]
        else:
            e2, V = torch.linalg.eigh(M.t() @ M); e2 = e2.flip(0).clamp_min(0); V = V.flip(1)[:, :RMAX]
            U = (M @ V) / e2[:RMAX].sqrt()[None, :]
            U, _ = torch.linalg.qr(U)                      # re-orthonormalize
        ro, eo = ranks(e2)
        bo[g] = U.float().cpu()
        rec[g] = {'K': K, 'N': Nn, 'in_rank': ri, 'in_energy': ei, 'out_rank': ro, 'out_energy': eo,
                  'in_top1': round(float(ev_[0] / ev_.sum()), 4)}
        del Sg, Q, W, M, V, U
        torch.cuda.empty_cache()
    bases['in'].append(bi); bases['out'].append(bo)
    spec['layers'].append(rec)
    print(json.dumps(rec), flush=True)
torch.save(bases, f'{OUT}/bases.pt')
json.dump(spec, open(f'{OUT}/spectra.json', 'w'), indent=1)
print('done %.0fs' % (time.time() - t0))
