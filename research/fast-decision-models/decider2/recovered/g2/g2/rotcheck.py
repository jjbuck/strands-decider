import sys, os, torch; sys.path.insert(0, os.path.expanduser('~/work/g2'))
import g2lib as GL
class D: pass
g = GL.G2.__new__(GL.G2); g.dev = 'cuda'; g._rot = {}
for K in (2048, 6144):
    x = torch.randn(64, K, device='cuda').bfloat16()
    a = g.rot(x); b = g.rot_act(x).float()
    print(K, 'max|exact-fp16| / max', ((a - b).abs().max() / a.abs().max()).item(), 'norm ratio', (a.norm() / x.float().norm()).item())
    W = torch.randn(32, K, device='cuda')
    print('  orthogonal check: (xR)(WR)^T vs xW^T rel err', (((g.rot(x) @ g.rot(W).t()) - x.float() @ W.t()).norm() / (x.float() @ W.t()).norm()).item())
