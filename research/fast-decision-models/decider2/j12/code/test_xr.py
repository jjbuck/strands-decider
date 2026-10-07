"""J12 smoke tests on the box: (1) base fwd == evalkit merged_full refs; (2) XR at init leaves outputs unchanged;
(3) exact readout with one-hot gold pointers == ground-truth relation on synthetic dev items; (4) gradient flows; (5) step time."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
import evalkit as EK
from xrlib import HX, Prep, XR, NT
import xlit

m = HX(); m.detach_inference(); dev = m.dev
prep = Prep(m.p.eng)
# (1) base vs refs
refs = EK.load_refs('CF-probe')
its = EK.load_suite('CF-probe')[:6] + EK.load_suite('REAL-agree')[:6]
refs2 = EK.load_refs('REAL-agree')
mx = 0; agree = 0; n = 0
with torch.no_grad():
    for it in its:
        r = refs if it['id'] in refs else refs2
        for q, spec in it['questions'].items():
            pr = prep(it['state'], spec)
            p = torch.softmax(m.logits_pr(m.fwd(pr), pr).float(), -1).tolist()
            ref = r[it['id']]['cfg']['merged_full'][q]
            d = max(abs(p[i] - ref[l]) for i, l in enumerate(pr['rq'].slot_labels)); mx = max(mx, d)
            agree += (pr['rq'].slot_labels[max(range(len(p)), key=p.__getitem__)] == max(ref, key=ref.get)); n += 1
print('(1) base vs merged_full refs: max|dp| %.4f  argmax %d/%d' % (mx, agree, n), flush=True)
# (2) XR at init
m.add_xr()
with torch.no_grad():
    it = its[0]; q, spec = next(iter(it['questions'].items()))
    pr = prep(it['state'], spec)
    m.xr_on = False; a = m.logits_pr(m.fwd(pr), pr)
    m.xr_on = True; ix = m.ix_for(pr); b = m.logits_pr(m.fwd(pr, ix=ix), pr)
print('(2) XR init: N slots', ix['N'], 'G', ix['G'], 'NL', ix['NL'], 'max|dlogit|', float((a - b).abs().max()), flush=True)
# (3) exact readout with one-hot gold pointers on synthetic dev items
sys.argv = ['x']
from xrlib import gold_slots
dev_items = [json.loads(l) for l in open(os.path.expanduser('~/work/j12/data/xr_dev.jsonl'))]
ok = 0; tot = 0; miss = 0; bykind = {}
for it in dev_items[:300]:
    if it['kind'] in ('exists', 'ident2'): continue
    pr = prep(it['state'], it['questions']['x'])
    ix = m.ix_for(pr)
    gs = gold_slots(ix, it, pr)
    if gs is None or gs['b'] is None: miss += 1; continue
    N = ix['N']
    pa = torch.zeros(1, 1, N + 1, device=dev); pa[0, 0, gs['a']] = 1
    pb = torch.zeros(1, 1, N + 1, device=dev); pb[0, 0, gs['b']] = 1
    f = XR.readout(pa, pb, ix)[0, 0].tolist()
    na, nb, eq, blk, lt, gt = f[:6]; dg = f[6:6 + NT]; dl = f[6 + NT:]
    sl = ix['_sl']; la, lb = sl['lits'][gs['a']], sl['lits'][gs['b']]
    if la.grp is not None and la.grp == lb.grp:
        good = (lt == float(la.val < lb.val)) and (gt == float(la.val > lb.val))
        if la.grp == 'date':
            good = good and all(dg[k] == float(lb.val - la.val > t) for k, t in enumerate(xlit.DATE_T))
    else:
        good = eq == float(la.eq == lb.eq)
    ok += good; tot += 1
    bykind.setdefault(it['kind'], [0, 0]); bykind[it['kind']][0] += good; bykind[it['kind']][1] += 1
    if not good and tot - ok <= 5: print('  BAD', it['kind'], la, lb, f[:6], dg)
print('(3) readout exact on gold pointers: %d/%d (gold not located: %d)' % (ok, tot, miss), bykind, flush=True)
# (4) grads + (5) step time
p_lora = m.add_lora(); p_head = m.set_head('std'); p_xr = list(m.xr.parameters())
with torch.no_grad():
    for x in m.xr: x.wo.weight.normal_(0, 1e-3)
it = dev_items[0]; pr = prep(it['state'], it['questions']['x']); ix = m.ix_for(pr)
for rep in range(5):
    if rep == 2: torch.cuda.synchronize(); t0 = time.time()
    h = m.fwd(pr, ix=ix, ckpt=True, keep_last=True)
    lg = m.logits_pr(h, pr); loss = -F.log_softmax(lg.float(), -1)[0]
    pa, pb = m.xr[0].last; loss = loss - torch.log(pa[:4, 0].mean() + 1e-6)
    loss.backward()
torch.cuda.synchronize()
print('(4) grad norms: xr.qa %.3e xr.wo %.3e lora %.3e head %.3e' % (m.xr[0].qa.weight.grad.norm(), m.xr[0].wo.weight.grad.norm(),
      sum(float(p.grad.norm()) for p in p_lora if p.grad is not None), sum(float(p.grad.norm()) for p in p_head if p.grad is not None)))
print('(5) fwd+bwd T=%d: %.3f s/step, mem %.1f GB' % (pr['L'], (time.time() - t0) / 3, torch.cuda.max_memory_allocated() / 1e9), flush=True)
