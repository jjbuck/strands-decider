"""one sparse int8 gate_up GEMM (M=1125, best config 7) and one dense int4 CUTLASS GEMM, for ncu"""
import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/q4'), os.path.expanduser('~/work/h2')]
import q4sp as S, test_sp as TS, qgemm as QG
TS.set_layout()
M, N, K = 1125, 12288, 2048
w = torch.randint(-127, 128, (N, K), device='cuda', dtype=torch.int8); x = torch.randint(-127, 128, (M, K), device='cuda', dtype=torch.int8)
Wc, E, _ = S.compress(w, S.mask24_mag(w, 'sp8'), 'sp8')
o = torch.empty(M, N, device='cuda', dtype=torch.float16)
cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 7
for _ in range(3): S.gemm('sp8', cfg, Wc, E, x, o, N, M, K, epi='fp16a', alpha=2 ** -10)
x4 = S.pack4(torch.randint(-7, 8, (M, K), device='cuda', dtype=torch.int8)); w4 = S.pack4(torch.randint(-7, 8, (N, K), device='cuda', dtype=torch.int8))
for _ in range(3): QG.gemm('s4', x4, w4, 0.5, 3, out=o)
torch.cuda.synchronize()
