"""cost of an online dense K x K basis change per GEMM input (B5 as emulated by Q1), cuBLAS bf16 and int8 (torch._int_mm), CUDA graph, M rows"""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q2')]
import q2k as Q
sys.path[:0] = [os.path.expanduser('~/work/q2')]
from bench_pro import gtm
dev = 'cuda'
out = {}
for M in (1125, 4125):
    for K in (2048, 6144):
        x = [torch.randn(M, K, device=dev).to(torch.bfloat16) for _ in range(3)]; U = torch.randn(K, K, device=dev).to(torch.bfloat16)
        x8 = [torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8) for _ in range(3)]; U8 = torch.randint(-127, 128, (K, K), device=dev, dtype=torch.int8)
        o = torch.empty(M, K, device=dev, dtype=torch.bfloat16); sa = torch.ones(M, device=dev); sb = torch.ones(K, device=dev)
        b8 = None
        for c in range(10):
            try: t = gtm(lambda i, c=c: Q.run('s8', c, Q.prob(x8[i % 3], U8, 's8', o, sa, sb)))
            except Exception: continue
            if b8 is None or t[0] < b8[0]: b8 = t
        r = dict(bf16=gtm(lambda i: torch.mm(x[i % 3], U)), int8_q2=b8)
        out[f'{M}|{K}'] = r; print(M, K, r, flush=True)
json.dump(out, open(os.path.expanduser('~/work/q2/res_basis_cost.json'), 'w'))
