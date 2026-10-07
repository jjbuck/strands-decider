"""sanity: (1) teacher layout (state_cache + one branch per question) vs hobson references; (2) student fwd/bwd runs; (3) speed."""
import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F, numpy as np
from j6lib import J6, Seg, QAdapters, KEEP
from h3lib import StdHead
import qtab, evalkit as EK

m = J6(); m.detach_inference(); dev = m.dev; eng = m.p.eng
pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}
refs = EK.load_refs('REAL-agree')
its = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) >= 2][:10] + EK.load_suite('LONG')[:3]
lrefs = EK.load_refs('LONG')
res = []
t0 = time.time()
with torch.no_grad():
    for it in its:
        R = refs.get(it['id']) or lrefs.get(it['id'])
        names = list(it['questions'])
        s = qtab.state_ids(eng, it['state']); Ls = len(s)
        cache = m.state_cache(s)
        qt = [PQ[q] for q in names]
        xt = F.embedding(torch.tensor([t for p in qt for t in p['q']], device=dev), m.embed)
        sg = Seg(Ls, [len(p['q']) for p in qt], dev)
        ht, _ = m.branch(xt, sg, cache, None)
        for j, q in enumerate(names):
            p = qt[j]; r0 = sg.r0[j]
            oi = torch.tensor([r0 + o for o in p['opt']], device=dev)
            pr = torch.softmax(m.head_logits(m.head0, ht[r0 + len(p['q']) - 1], ht[oi], p['kind']).float(), -1).tolist()
            ref = R['hobson'][q]; b = [ref[lab] for lab in p['rq'].slot_labels]
            res.append((int(np.argmax(pr)) == int(np.argmax(b)), max(abs(x - y) for x, y in zip(pr, b)), Ls))
torch.cuda.synchronize()
print('teacher vs hobson refs: argmax %d/%d, max|dp| med %.4f max %.4f (%.1fs)' % (sum(r[0] for r in res), len(res), np.median([r[1] for r in res]), max(r[1] for r in res), time.time() - t0), flush=True)

# student fwd/bwd
qn = list(specs)
ad = QAdapters(m, qn, {q: qtab.slot_init(m, eng, PQ[q]) for q in qn})
head = StdHead(m.head0).to(dev)
it = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) == 4][0]
names = list(it['questions'])
s = qtab.state_ids(eng, it['state'])
for T in (1000, 4000):
    ss_ = (s * 10)[:T]
    torch.cuda.synchronize(); t1 = time.time()
    cache = m.state_cache(ss_); torch.cuda.synchronize(); t2 = time.time()
    sg = Seg(T, [PQ[q]['K'] + 1 for q in names], dev); ad.seg = sg; ad.cur = [ad.qi[q] for q in names]
    hs, ks = m.branch(ad.slot_inputs(names).to(torch.bfloat16), sg, cache, ad, keep=KEEP, krows=torch.arange(sg.R, device=dev))
    loss = sum(hs.float().pow(2).mean() for _ in range(1)) + sum(v.float().pow(2).mean() for v in ks.values())
    loss.backward(); torch.cuda.synchronize(); t3 = time.time()
    ng = sum(1 for p_ in ad.parameters() if p_.grad is not None)
    print('T', T, 'state_cache %.1f ms' % ((t2 - t1) * 1000), 'student fwd+bwd %.1f ms' % ((t3 - t2) * 1000), 'R', sg.R, 'params with grad', ng,
          'mem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    ad.zero_grad(set_to_none=True)
# variant A full fwd/bwd
ad.cur = [0]
torch.cuda.synchronize(); t1 = time.time()
h = m.full((s * 10)[:1000], ad.slot_inputs([qn[0]]), ad)
h.float().pow(2).mean().backward(); torch.cuda.synchronize()
print('variant A full fwd+bwd T=1000: %.1f ms, mem %.1fG' % ((time.time() - t1) * 1000, torch.cuda.max_memory_allocated() / 1e9), flush=True)
