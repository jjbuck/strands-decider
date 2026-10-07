"""QLinF gradient check against autograd on the straight-through reference (small random matrices; seconds, ~50 MB GPU)"""
import os, sys
sys.path[:0] = [os.path.expanduser('~/work/q3')]
import torch
import q3lib as QL
torch.manual_seed(0); dev = 'cuda'
M, K, N = 64, 256, 128
x = torch.randn(M, K, device=dev)
W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
s = W.float().abs().amax(1) / 7 * 0.9
gy = torch.randn(M, N, device=dev).to(torch.bfloat16)
qa, sa = QL._qact_e(x, 7.0, 0.9); Xd = (qa.float() * sa[:, None])
# lat
A = W.clone().requires_grad_(True); xx = x.clone().requires_grad_(True)
qc = QL._wcode_e(A.detach(), s, 7.0)
y = QL.QLinF.apply(xx, A, s, qc, 'lat', 7.0, 7.0, 0.9, 0.0); (y.float() * gy.float()).sum().backward()
Wd = torch.round(W.float() / s[:, None]).clamp(-7, 7) * s[:, None]
print('lat fwd max|y - Xd Wd^T| / max|y|', float((y.float() - Xd @ Wd.t()).abs().max() / y.float().abs().max()))
print('lat dA rel err', float((A.grad.float() - gy.float().t() @ Xd).norm() / (gy.float().t() @ Xd).norm()))
print('lat dx rel err', float((xx.grad.float() - gy.float() @ Wd).norm() / (gy.float() @ Wd).norm()))
# soft
u = W.float() / s[:, None]; cf = torch.floor(u).clamp(-7, 6); rest = (u - cf).clamp(0, 1)
p_ = ((rest - QL.GAMMA) / (QL.ZETA - QL.GAMMA)).clamp(1e-4, 1 - 1e-4); V0 = torch.log(p_ / (1 - p_))
V = V0.to(torch.bfloat16).requires_grad_(True)
y = QL.QLinF.apply(x, V, s, cf.to(torch.int8), 'soft', 7.0, 7.0, 0.9, 0.0); (y.float() * gy.float()).sum().backward()
Vr = V0.to(torch.bfloat16).float().requires_grad_(True)
U = (cf + QL.hsoft(Vr)).clamp(-7, 7); yr = Xd @ (U * s[:, None]).t(); (yr * gy.float()).sum().backward()
print('soft fwd rel', float((y.float() - yr.detach()).norm() / yr.detach().norm()))
print('soft dV rel err', float((V.grad.float() - Vr.grad).norm() / Vr.grad.norm()), 'cos', float(torch.nn.functional.cosine_similarity(V.grad.float().flatten(), Vr.grad.flatten(), 0)))
# code (scale grad)
rho = torch.zeros(N, device=dev, requires_grad=True)
q = torch.round(W.float() / s[:, None]).clamp(-7, 7).to(torch.int8)
y = QL.QLinF.apply(x, q, s * rho.exp(), None, 'code', 7.0, 7.0, 0.9, 0.0); (y.float() * gy.float()).sum().backward()
rho2 = torch.zeros(N, device=dev, requires_grad=True)
yr = Xd @ (q.float() * (s * rho2.exp())[:, None]).t(); (yr * gy.float()).sum().backward()
print('code drho rel err', float((rho.grad - rho2.grad).norm() / rho2.grad.norm()))
# Had vs Rot
for K_, seed in ((2048, 1235), (6144, 1236)):
    R = QL.Rot(K_, seed, dev); H = QL.Had(R)
    z = torch.randn(32, K_, device=dev).to(torch.bfloat16)
    print('Had vs Rot', K_, float((H(z) - R(z.float())).abs().max()))
    zz = z.clone().float().requires_grad_(True); g = torch.randn(32, K_, device=dev)
    (H(zz) * g).sum().backward(); ref = R.inv(g)       # d/dx <x R, g> = g R^T
    print('Had grad vs R^T', K_, float((zz.grad - ref).abs().max() / ref.abs().max()))
