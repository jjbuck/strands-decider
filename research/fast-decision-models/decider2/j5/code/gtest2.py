import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/j5')]
os.environ['SKTUNE'] = '0'
import sk
torch.manual_seed(0); dev = 'cuda'
def rel(a, b): return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))
M, N, K = 172, 8224, 2048
A = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev) * 0.02
w = sk.QW('bf16', W.to(torch.bfloat16).contiguous()); ref = A.float() @ W.to(torch.bfloat16).float().t()
for cfg in [(32, 128, 64, 4, 4, 3), (64, 64, 64, 4, 4, 3), (32, 128, 64, 2, 4, 3), (16, 256, 64, 4, 4, 3)]:
    errs = []
    for r in range(5):
        c = sk.launch(A, w, 0, cfg=cfg); torch.cuda.synchronize(); errs.append(round(rel(c, ref), 4))
    ws = cnt = torch.zeros(1)
    print(cfg, errs, 'ws absmax', float(ws.abs().max()), 'cnt max', int(cnt.max()), 'tiles', (-(-M // cfg[0])) * (-(-N // cfg[1])), flush=True)
