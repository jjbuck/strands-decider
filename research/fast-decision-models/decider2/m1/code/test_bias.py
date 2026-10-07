"""unit test: the 136-dim augmented SDPA (gate biases in extra q/k dims) == explicit fp32 softmax with the bias matrix"""
import os, sys, torch, torch.nn.functional as F
sys.path[:0] = [os.path.expanduser('~/work/m1')]
import m1lib as ML
torch.manual_seed(0); dev = 'cuda'
T, H = 700, 16
q = torch.randn(T, H, 128, device=dev).to(torch.bfloat16); k = torch.randn(T, H, 128, device=dev).to(torch.bfloat16); v = torch.randn(T, H, 128, device=dev).to(torch.bfloat16)
g = -torch.rand(T, H, device=dev) * 3.0; G = g.cumsum(0) - 5000.0       # large magnitudes on purpose
lb = torch.nn.functional.logsigmoid(torch.randn(T, H, device=dev) * 2)
sc = 128 ** -0.5
# reference
lg = torch.einsum('thd,shd->hts', q.float(), k.float()) * sc + (G.t()[:, :, None] - G.t()[:, None, :]) + lb.t()[:, None, :]
lg = lg.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))[None], float('-inf'))
ref = torch.einsum('hts,shd->thd', torch.softmax(lg, -1), v.float())
a1, a2, a3 = ML.split3(G); b1, b2, b3 = ML.split3(lb - G)
one = torch.ones(T, H, 3, device=dev, dtype=torch.bfloat16); zer = torch.zeros(T, H, 2, device=dev, dtype=torch.bfloat16)
qa = torch.cat([q * sc, a1[..., None], a2[..., None], a3[..., None], one, zer], -1)
ka = torch.cat([k, one, b1[..., None], b2[..., None], b3[..., None], zer], -1)
va = torch.cat([v, torch.zeros(T, H, 8, device=dev, dtype=torch.bfloat16)], -1)
lay = dict(Ls=T)
out = ML.attn_core(qa, ka, va, lay, 1.0)[..., :128].float()
err = (out - ref).abs().max().item(); rel = ((out - ref).norm() / ref.norm()).item()
# plain bf16 reference error scale: SDPA without biases vs fp32 math
lg0 = torch.einsum('thd,shd->hts', q.float(), k.float()) * sc
lg0 = lg0.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))[None], float('-inf'))
ref0 = torch.einsum('hts,shd->thd', torch.softmax(lg0, -1), v.float())
out0 = ML.attn_core(q, k, v, lay, sc).float()
print('aug vs fp32 ref: max abs', err, 'rel', rel, '| plain bf16 sdpa vs fp32 ref: rel', ((out0 - ref0).norm() / ref0.norm()).item())
