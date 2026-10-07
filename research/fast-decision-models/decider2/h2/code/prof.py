"""kernel-time breakdown of one graph replay (CUPTI via torch.profiler). usage: python prof.py T prec[,prec] [layout]"""
import sys, os, time, json, torch, collections, statistics as st
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
from prof_d1 import load_torso, capture
from lean2 import Lean2
import qrt as Q
from torch.profiler import profile, ProfilerActivity
T = int(sys.argv[1]); precs = sys.argv[2].split(',')
torso = load_torso(); ln = Lean2(torso, fuse='fold'); del torso; torch.cuda.empty_cache()
def cat(n):
    n = n.lower()
    if 'cutlass' in n or 'gemm_k' in n or 'gemm' in n and 'triton' not in n: return 'GEMM'
    for k in ('swiglu', 'gnorm', 'agate', 'addq', 'conv_k', 'aprep', 'flash', 'attention', 'chunk', 'fwd_kernel', 'kkt', 'solve', 'recompute', 'h_kernel', 'o_kernel', 'gate'):
        if k in n: return k
    return n[:60]
res = {}
for prec in precs:
    m = Q.QRT(ln, prec=prec)
    ids = torch.randint(1000, 100000, (T,), device='cuda'); lay = Q.Lay('single', T)
    g, out = capture(lambda: m.forward(ids, lay))
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    agg = collections.defaultdict(float); cnt = collections.Counter(); names = collections.defaultdict(float)
    for e in p.events():
        if e.device_type.name != 'CUDA': continue
        d = e.device_time if hasattr(e, 'device_time') else e.cuda_time
        c = cat(e.name); agg[c] += d / 1000; cnt[c] += 1; names[e.name[:90]] += d / 1000
    tot = sum(agg.values())
    print(f'== {prec} T={T}: kernel total {tot:.2f} ms', flush=True)
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]): print(f'   {k:40s} {v:7.2f} ms  n={cnt[k]}')
    print('  top names:'); [print(f'     {v:7.2f} {k}') for k, v in sorted(names.items(), key=lambda kv: -kv[1])[:12]]
    res[prec] = dict(total=tot, cats=dict(agg))
    del g, out, m; torch.cuda.empty_cache()
json.dump(res, open(os.path.expanduser(f'~/work/h2/prof_{T}_{"_".join(precs)}.json'), 'w'), indent=1)
