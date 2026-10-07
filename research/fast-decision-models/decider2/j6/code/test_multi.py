import os, sys, time
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from j6lib import J6, Seg, QAdapters, KEEP, FastState, merge_caches
import qtab, evalkit as EK
m = J6(); fs = FastState(m); m.detach_inference(); dev = m.dev; eng = m.p.eng
pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}; qn = list(specs)
ad = QAdapters(m, qn, {q: qtab.slot_init(m, eng, PQ[q]) for q in qn})
with torch.no_grad():
    for k in ad.sB: ad.sB[k].normal_(0, 0.002)
    for j in range(len(qn)):
        for k in ad.qB[j]: ad.qB[j][k].normal_(0, 0.002)
reqs = pool[:4]
batch = [(qtab.state_ids(eng, r['state'])[:3000], list(r['questions'])) for r in reqs]
caches = [fs(s) for s, _ in batch]
def run(bt, cs, idx):
    cache = merge_caches(cs) if len(cs) > 1 else cs[0]
    allq = [(r, q) for r, (_, names) in enumerate(bt) for q in names]
    ss = Seg([len(s) for s, _ in bt], [PQ[q]['K'] + 1 for _, q in allq], dev, req=[r for r, _ in allq])
    ad.seg = ss; ad.cur = [ad.qi[q] for _, q in allq]
    hs, _ = m.branch(ad.slot_inputs([q for _, q in allq]).to(torch.bfloat16), ss, cache, ad)
    out = []
    for j, (_, q) in enumerate(allq):
        K = PQ[q]['K']; r0 = ss.r0[j]
        out.append(torch.softmax(m.head_logits(m.head0, hs[r0 + K], hs[r0:r0 + K], PQ[q]['kind']).float(), -1))
    return out
with torch.no_grad():
    merged = run(batch, caches, None)
    single = [p for i in range(4) for p in run([batch[i]], [caches[i]], None)]
print('nq', len(merged), 'max |dp| merged vs single', max(float((a - b).abs().max()) for a, b in zip(merged, single)), flush=True)
def tm(f, n=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1000
def stud():
    cache = merge_caches(caches)
    allq = [(r, q) for r, (_, names) in enumerate(batch) for q in names]
    ss = Seg([len(s) for s, _ in batch], [PQ[q]['K'] + 1 for _, q in allq], dev, req=[r for r, _ in allq])
    ad.seg = ss; ad.cur = [ad.qi[q] for _, q in allq]
    hs, ks = m.branch(ad.slot_inputs([q for _, q in allq]).to(torch.bfloat16), ss, cache, ad, keep=KEEP, krows=torch.arange(ss.R, device=dev))
    (hs.float().pow(2).mean() + sum(v.float().pow(2).mean() for v in ks.values())).backward()
    return ss.R
print('4-request student fwd+bwd %.0f ms, rows %d, questions %d' % (tm(stud), stud(), sum(len(n) for _, n in batch)), flush=True)
print('state pass x4 %.0f ms (tokens %d)' % (tm(lambda: [fs(s) for s, _ in batch]), sum(len(s) for s, _ in batch)), flush=True)
