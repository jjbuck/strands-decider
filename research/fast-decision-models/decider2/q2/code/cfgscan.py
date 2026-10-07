"""time every q2gemm config for one shape: python cfgscan.py M N K [variant]"""
import sys, os, torch
sys.path[:0] = [os.path.expanduser('~/work/q2')]
import q2k as Q
from bench_gemm import tm
M, N, K = (int(x) for x in sys.argv[1:4]); var = sys.argv[4] if len(sys.argv) > 4 else 's4'
dev = 'cuda'
if var == 's4':
    A = [Q.pack4(torch.randint(-7, 8, (M, K), device=dev, dtype=torch.int8)) for _ in range(3)]; B = Q.pack4(torch.randint(-7, 8, (N, K), device=dev, dtype=torch.int8))
elif var == 's8':
    A = [torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8) for _ in range(3)]; B = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
else:
    A = [torch.randn(M, K, device=dev, dtype=torch.bfloat16) for _ in range(3)]; B = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
sa = torch.rand(M, device=dev); sw = torch.rand(N, device=dev)
o = torch.empty(M, N, device=dev, dtype=torch.bfloat16); o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
ops = 2 * M * N * K
for c in range(10):
    for epi, out in (('bf16', o), ('fp16a', o16), ('none', o16)):
        if var == 'bf16' and epi != 'bf16': continue
        try:
            m, p = tm(lambda i: Q.run(var, c, Q.prob(A[i % 3], B, var, out, sa, sw, epi=epi)))
            print(f'cfg{c} {epi:6s} {m:8.1f} us  {ops / m / 1e6:6.1f} TOPS', flush=True)
        except Exception as e:
            print(f'cfg{c} {epi} fail {str(e)[:80]}')
