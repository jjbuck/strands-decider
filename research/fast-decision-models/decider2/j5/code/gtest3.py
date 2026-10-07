import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/j5')]
os.environ['SKTUNE'] = '0'
import sk
torch.manual_seed(0); dev = 'cuda'
def rel(a, b): return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))
for (M, N, K, cfg) in [(16, 64, 2048, (16, 64, 64, 4, 4, 3)), (32, 128, 2048, (32, 128, 64, 4, 4, 3)), (64, 64, 2048, (64, 64, 64, 4, 4, 3)),
                       (64, 128, 2048, (32, 128, 64, 4, 4, 3)), (64, 640, 2048, (32, 128, 64, 4, 4, 3)), (172, 8224, 2048, (32, 128, 64, 4, 4, 3)),
                       (172, 8224, 2048, (32, 128, 64, 4, 8, 3)), (172, 8224, 2048, (32, 128, 64, 4, 4, 1))]:
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev) * 0.02
    w = sk.QW('bf16', W.to(torch.bfloat16).contiguous()); ref = A.float() @ W.to(torch.bfloat16).float().t()
    errs = []
    for r in range(6):
        c = sk.launch(A, w, 0, cfg=cfg); torch.cuda.synchronize(); errs.append(round(rel(c, ref), 4))
    bad = ((c.float() - ref).abs() > 0.05 * ref.abs().max()).nonzero()
    print(M, N, K, cfg, errs, 'bad elems', bad.shape[0], 'rows', bad[:, 0].unique().tolist()[:8], 'cols', bad[:, 1].unique().tolist()[:8], flush=True)
