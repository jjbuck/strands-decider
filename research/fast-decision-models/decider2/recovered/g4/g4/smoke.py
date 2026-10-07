"""smoke test: causality check, train throughput (compiled), decision forward.  usage: python smoke.py SCALE ARM [steps]"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
sc, arm = sys.argv[1], sys.argv[2]; nsteps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
torch.backends.cuda.matmul.allow_tf32 = True
cfg = C.Cfg(sc, arm)
m = C.build(cfg).cuda()
print(cfg, 'params', sum(p.numel() for p in m.parameters()) / 1e6, 'nonemb', C.nonemb_params(cfg) / 1e6, flush=True)
if cfg.kind == 'slot': m._pt_setup(1024, 'cuda')
# causality: perturb token 700 -> losses at positions < 699 must not change
x = torch.randint(0, C.V, (2, 1025), device='cuda')
f = torch.compile(lambda a, b: m.lm_loss(a, b))
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    l1 = f(x[:, :-1], x[:, 1:]).float()
    x2 = x.clone(); x2[:, 700] = (x2[:, 700] + 1) % C.V
    l2 = f(x2[:, :-1], x2[:, 1:]).float()
d = (l1 - l2).abs()
print('causality: max |dloss| before 699:', d[:, :699].max().item(), ' at/after 699:', d[:, 699:].max().item(), flush=True)
# throughput
opt = torch.optim.AdamW(C.param_groups(m, 0.1), lr=1e-4, fused=True)
def step(xb):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        l = m.lm_loss(xb[:, :-1], xb[:, 1:]).mean()
    return l
cs = torch.compile(step)
bs = 16
for i in range(nsteps):
    if i == 5: torch.cuda.synchronize(); t0 = time.time()
    xb = torch.randint(0, C.V, (bs, 1025), device='cuda')
    l = cs(xb); l.backward(); opt.step(); opt.zero_grad(set_to_none=True)
torch.cuda.synchronize(); dt = (time.time() - t0) / (nsteps - 5)
ft = 6 * C.train_macs_per_token(cfg)
print(f'train: {bs*1024/dt:.0f} tok/s  {bs*1024/dt*ft/1e12:.1f} TFLOPS(analytic)  loss {l.item():.3f}  mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB', flush=True)
# decision forward
ids = torch.randint(0, C.V, (16, 1024), device='cuda'); last = torch.full((16,), 1000, device='cuda')
nctx = torch.full((16,), 900, device='cuda'); nq = torch.full((16,), 50, device='cuda')
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    h = m.dec_hidden(ids, nctx, nq) if cfg.kind == 'slot' else m.dec_hidden(ids, last)
print('dec ok', h.shape, h.float().abs().mean().item(), flush=True)
