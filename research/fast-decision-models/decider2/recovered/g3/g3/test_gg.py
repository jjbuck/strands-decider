import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe as M
g = M.load_cfg("st4b"); W = M.load_weights(g, layers=4)
t = M.MoETorso(g, W).cuda(); t.train()
torch.manual_seed(0)
for p in t.lora.parameters():
    if p.ndim >= 2 and p.abs().sum() == 0: p.data.normal_(0, 0.01)
ids = torch.randint(0, 150000, (3, 700), device="cuda"); am = torch.ones_like(ids); am[1, 500:] = 0; am[2, 300:] = 0
res = {}
for tg in (False, True):
    t.tg = tg
    for p in t.lora.parameters(): p.grad = None
    out = t(ids, am).last_hidden_state
    loss = (out.float() * am[..., None]).pow(2).mean(); loss.backward()
    res[tg] = (out.detach().float(), {n: p.grad.detach().clone() for n, p in t.lora.named_parameters()})
o0, g0 = res[False]; o1, g1 = res[True]
m = am.bool()
print("out rel", ((o0 - o1)[m].norm() / o0[m].norm()).item())
worst = sorted(((((g0[n] - g1[n]).norm() / (g0[n].norm() + 1e-12)).item(), n) for n in g0), reverse=True)[:6]
print("grad rel worst", [(round(a, 4), b) for a, b in worst])
for tg in (False, True):
    t.tg = tg; torch.cuda.synchronize(); t0 = time.time()
    for _ in range(3):
        out = t(ids, am).last_hidden_state; (out.float().pow(2).mean()).backward()
    torch.cuda.synchronize(); print("tg", tg, "fwd+bwd s/iter", round((time.time() - t0) / 3, 3))
