import os, sys, time, torch
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import q1lib as QL
g = QL.Q1(); A, B = QL.req_sets(256, 64, 2600)
def run(it, gfrom, hook):
    pr = g.prep(it['state'], it['q']); ids = pr['s'] + pr['q']
    torch.cuda.synchronize(); t0 = time.time()
    h = g.fwdg(ids, gfrom=gfrom); lg = g.logits_g(h, pr); torch.cuda.synchronize(); t1 = time.time()
    dirs, cap, p = QL.fisher_dirs(lg)
    for j, (lam, u) in enumerate(dirs): (lg * u).sum().backward(retain_graph=(j < len(dirs) - 1))
    torch.cuda.synchronize(); t2 = time.time()
    print(f'T={len(ids)} gfrom={gfrom} hook={hook} fwd {t1-t0:.2f}s bwd {t2-t1:.2f}s ndirs {len(dirs)}', flush=True)
for it in [A[0], A[0], A[1], A[1], A[2]]: run(it, 0, False)
g.hook_fn = lambda i, k, gr: None
for it in [A[0], A[3]]: run(it, 0, True)
