import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/j5')]
import cutsk
dev = 'cuda'; torch.manual_seed(0)
for (M, N, K) in [(140, 2048, 6144), (16, 2048, 2048), (364, 12288, 2048)]:
    A = torch.randint(-100, 100, (M, K), device=dev, dtype=torch.int8); B = torch.randint(-100, 100, (N, K), device=dev, dtype=torch.int8)
    ref = (A.double() @ B.double().t()) / 65536
    for c, sl in [(0, 2), (1, 4), (2, 8), (3, 3), (4, 4)]:
        try:
            errs = []
            for r in range(3):
                C = cutsk.gemm(A, B, 1 / 65536, c, sl); torch.cuda.synchronize()
                errs.append(float((C.double() - ref).norm() / ref.norm()))
            print(M, N, K, c, sl, ['%.1e' % e for e in errs], flush=True)
        except Exception as ex: print(M, N, K, c, sl, 'ERR', ex)
