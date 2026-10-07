"""J2 structural + throughput checks with autograd (box only).
 1. masked state cache, exactly: d(state-row outputs)/d(question input embeddings) == 0 in 'qag', != 0 in 'qa' (gates on);
 2. train-step throughput (LoRA r16 + gates + head, checkpointing) for causal / qag."""
import os, sys, json, time, random
sys.path[:0] = [os.path.expanduser('~/work/j2'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from j2lib import J2, Pack, GDN_LAYERS
from h3lib import StdHead

m = J2(); m.detach_inference()
for d in m.L:
    for k, v in d.items():
        if torch.is_tensor(v): v.requires_grad_(False)
rng = random.Random(0)
S = [rng.randrange(1000, 50000) for _ in range(300)]; Q = [rng.randrange(1000, 50000) for _ in range(40)]
for mode in ('qag', 'qa'):
    m.setup_bidir(mode)
    with torch.no_grad():
        for p_ in m.gam.values(): p_.fill_(0.7)
        for p_ in m.lam.values(): p_.fill_(0.6)
    pk = Pack([(S + Q, len(S))], m.dev, mode)
    x0 = F.embedding(pk.ids, m.embed).detach().clone().requires_grad_(True)
    h = m.fwd_pk(pk, x0=x0)
    h[:len(S)].float().sum().backward()
    gq = x0.grad[len(S):].float().abs().max().item(); gs = x0.grad[:len(S)].float().abs().max().item()
    # and the reverse: question rows must see the state's early tokens
    x1 = F.embedding(pk.ids, m.embed).detach().clone().requires_grad_(True)
    h1 = m.fwd_pk(pk, x0=x1); h1[-1].float().sum().backward()
    g0 = x1.grad[:5].float().abs().max().item()
    print(f'{mode}: max|d state rows / d question inputs| = {gq:.3e} (vs d/d state inputs {gs:.3e}); d answer row / d first 5 state tokens {g0:.3e}', flush=True)

m.add_lora(r=16, alpha=32, seed=0)
m.head = StdHead(m.head0).to(m.dev)
for mode in ('causal', 'qag'):
    gp = m.setup_bidir(mode)
    params = list(m.lora.parameters()) + list(m.head.parameters()) + gp
    for T_pack, L in ((8192, 160), (8192, 1024), (8192, 4096), (6144, 6144)):
        n = T_pack // L
        rows = []
        for r in range(n):
            ids = [rng.randrange(1000, 50000) for _ in range(L)]
            rows.append(dict(s=ids[:L - 40], q=ids[L - 40:], opt=[5, 10], kind='noul', n_slots=2))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for rep in range(3):
            t0 = time.time()
            out = m.decide(rows, ckpt=True)
            loss = sum(o.float().logsumexp(-1) for o in out)
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
        for p_ in params: p_.grad = None
        print(f'train {mode} rows {n}x{L}: {dt:.2f}s/pack {n * L / dt:.0f} tok/s peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB', flush=True)
