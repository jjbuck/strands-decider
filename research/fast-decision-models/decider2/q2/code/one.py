import sys, os, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
import q2k as Q, qgemm as QG
M, N, K, c = 1125, 8224, 2048, int(sys.argv[1])
A = Q.pack4(torch.randint(-7, 8, (M, K), device='cuda', dtype=torch.int8)); B = Q.pack4(torch.randint(-7, 8, (N, K), device='cuda', dtype=torch.int8))
o16 = torch.empty(M, N, device='cuda', dtype=torch.float16)
for _ in range(3):
    if c >= 0: Q.run('s4', c, Q.prob(A, B, 's4', o16, epi='fp16a', alpha=0.5))
    else: QG.gemm('s4', A, B, 0.5, -c - 1, out=o16)
torch.cuda.synchronize()
