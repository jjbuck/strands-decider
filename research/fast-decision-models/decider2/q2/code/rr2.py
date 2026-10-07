"""row-role schedule test: int8-first (0) vs interleaved (1); correctness + timing at T=1000 + 125 question rows"""
import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/q2')]
import q2k as Q
from bench_gemm import tm, best, SH
dev = 'cuda'
T, nq = (int(x) for x in sys.argv[1:3]) if len(sys.argv) > 2 else (1000, 125)
M = T + nq; q0 = T
tot = {0: 0.0, 1: 0.0, 's4': 0.0}
CNT = {'gdn_in': 18, 'attn_in': 6, 'out': 24, 'gate_up': 24, 'down': 24}
for name, (N, K) in SH.items():
    qa = torch.randint(-7, 8, (M, K), device=dev, dtype=torch.int8); a8 = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8)
    qw = torch.randint(-7, 8, (N, K), device=dev, dtype=torch.int8); w8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
    a4 = Q.pack4(qa); w4 = Q.pack4(qw); sa = torch.rand(M, device=dev); sb = torch.rand(N, device=dev)
    o = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = torch.cat([Q.ref_int(qa[:q0], sa[:q0], qw, sb), Q.ref_int(a8[q0:], sa[q0:], w8, sb)]).to(torch.bfloat16)
    p0 = Q.prob(a4[:q0], w4, 's4', o, sa[:q0], sb); p1 = Q.prob(a8[q0:], w8, 's8', o, sa[q0:], sb, row0=q0)
    for sch in (0, 1):
        o.zero_(); Q.run('s4', 4, p0, p1, sched=sch); torch.cuda.synchronize()
        assert torch.equal(o, ref), (name, sch)
    r = {sch: best([(c, (lambda i, c=c, sch=sch: Q.run('s4', c, p0, p1, sched=sch))) for c in range(10)]) for sch in (0, 1)}
    r['s4'] = best([(c, (lambda i, c=c: Q.run('s4', c, Q.prob(a4, w4, 's4', o, sa, sb)))) for c in range(10)])
    for k in tot: tot[k] += CNT[name] * r[k][1]
    print(name, {k: (v[0], v[1]) for k, v in r.items()}, flush=True)
print('totals ms (96 GEMMs, bf16 epilogue everywhere):', {k: round(v / 1000, 2) for k, v in tot.items()})
