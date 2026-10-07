"""B3 codebooks: k-means (Kc = 64, 256) on state-row samples of every GEMM input (q1cal) -> ~/work/q1/proto/C{Kc}_{i}_{k}.pt (unrotated basis)"""
import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import q1lib as QL
from q1b3_km import kmeans
CAL = os.path.expanduser('~/work/q1/cal'); OUT = os.path.expanduser('~/work/q1/proto'); os.makedirs(OUT, exist_ok=True)
for i in range(24):
    for k in QL.GEMMS:
        Xs = torch.load(f'{CAL}/X_{i}_{k}.pt').float().cuda(); X = Xs[Xs[:, -1] < 0.5, :-1]
        for kc in (64, 256): torch.save(kmeans(X, kc).cpu(), f'{OUT}/C{kc}_{i}_{k}.pt')
    print(i, flush=True)
