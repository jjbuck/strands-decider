import sys, os, time, torch
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import bench as B, qrt as Q
from lean2 import Lean2
bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt'))
torso, head = B.load_model(); ln = Lean2(torso, fuse='fold'); del torso
for prec in ('bf16', 'w4a4'):
    m = Q.QRT(ln, head=head, prec=prec)
    pre = torch.tensor([t for q in bundles['15q'] for t in q['ids']], device='cuda')
    for rep in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        c = m.compile_prefix(pre, 4015); torch.cuda.synchronize()
        print(prec, 'compile 15q prefix (3720 tok), eager, rep', rep, f'{(time.perf_counter() - t0) * 1000:.1f} ms', 'cache MB', round(sum(v.numel() * v.element_size() for d in ('S', 'tail', 'kb', 'vb') for v in c[d].values()) / 1e6, 1), flush=True)
    del m; torch.cuda.empty_cache()
