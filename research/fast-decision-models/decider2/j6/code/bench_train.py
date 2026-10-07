"""warm timings of the training step parts; TTL (fused) state pass vs h3lib state pass, and equality of their caches."""
import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/h4')]
import torch, torch.nn.functional as F, numpy as np
from j6lib import J6, Seg, QAdapters, KEEP
from h3lib import StdHead
import qtab, evalkit as EK
from tt_lean import TTL
m = J6(); dev = m.dev; eng = m.p.eng
rt = TTL(m.p.tm, list(range(10)))
m.detach_inference()
import gc; gc.collect(); torch.cuda.empty_cache()
print('mem after TTL+detach %.1fG' % (torch.cuda.memory_allocated() / 1e9), flush=True)
pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}
qn = list(specs)
ad = QAdapters(m, qn, {q: qtab.slot_init(m, eng, PQ[q]) for q in qn})
it = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) == 4][0]
names = list(it['questions'])
s0 = qtab.state_ids(eng, it['state'])
def ttl_cache(s):
    rt.set_fuse(len(s))
    with torch.no_grad():
        _, c = rt.fwd(torch.tensor([s], device=dev), want_cache=True)
    out = []
    for i in range(24):
        if 'S' in c[i]: out.append(dict(S=c[i]['S'], tail=c[i]['tail'][:, :6144].contiguous()))
        else: out.append(dict(k=c[i]['k'], v=c[i]['v']))
    return out
def tm(f, n=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): r = f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000, r
for T in (1000, 2000, 4000):
    s = (s0 * 20)[:T]
    t_h, c1 = tm(lambda: m.state_cache(s))
    t_t, c2 = tm(lambda: ttl_cache(s))
    # compare teacher answers with each cache
    qt = [PQ[q] for q in names]
    xt = F.embedding(torch.tensor([t for p in qt for t in p['q']], device=dev), m.embed)
    sg = Seg(T, [len(p['q']) for p in qt], dev)
    def teach(c):
        with torch.no_grad():
            ht, _ = m.branch(xt, sg, c, None)
        o = []
        for j, p in enumerate(qt):
            r0 = sg.r0[j]; oi = torch.tensor([r0 + x for x in p['opt']], device=dev)
            o.append(torch.softmax(m.head_logits(m.head0, ht[r0 + len(p['q']) - 1], ht[oi], p['kind']).float(), -1))
        return o
    t_b, p1 = tm(lambda: teach(c1)); p2 = teach(c2)
    dp = max(float((a - b).abs().max()) for a, b in zip(p1, p2))
    sg2 = Seg(T, [PQ[q]['K'] + 1 for q in names], dev); ad.seg = sg2; ad.cur = [ad.qi[q] for q in names]
    def stud():
        hs, ks = m.branch(ad.slot_inputs(names).to(torch.bfloat16), sg2, c1, ad, keep=KEEP, krows=torch.arange(sg2.R, device=dev))
        (hs.float().pow(2).mean() + sum(v.float().pow(2).mean() for v in ks.values())).backward()
    t_s, _ = tm(stud)
    print(f'T {T}: state h3lib {t_h:.0f} ms, TTL {t_t:.0f} ms; teacher branches ({sg.R} rows) {t_b:.0f} ms; student fwd+bwd ({sg2.R} rows) {t_s:.0f} ms; teacher dp(h3lib vs TTL cache) {dp:.4f}; mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
ad.cur = [0]
def fa():
    h = m.full((s0 * 20)[:1000], ad.slot_inputs([qn[0]]), ad); h.float().pow(2).mean().backward()
t_a, _ = tm(fa)
print(f'variant A full fwd+bwd T=1000: {t_a:.0f} ms', flush=True)
