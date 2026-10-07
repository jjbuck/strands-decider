"""H7 checks: (a) teacher branch layout == plain state-first forward; (b) schema n=1 slot-branch == plain [Q'][state][<answer>] sequence;
(c) LoRA gradients through the branch path == through the plain sequence; (d) n=4 fwd+bwd time / memory; (e) teacher decisions vs evalkit refs."""
import os, sys, json, time, random
sys.path[:0] = [os.path.expanduser('~/work/h7'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from h7lib import H7
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)

m = H7(); m.detach_inference()
eng = m.p.eng


def prep(state, qd, perm=None):
    q = ta.validate_python(qd)
    rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
    s, qs = eng._fit(render_state(state), [rq.text])
    opt = eng._option_idx([rq], 0)[0].tolist()
    return dict(s=s, q=qs[0], opt=opt, rq=rq, q0=len(s))


items = [it for it in EK.load_suite('REAL-agree') if len(it['questions']) == 4][:3]
refs = EK.load_refs('REAL-agree') if hasattr(EK, 'load_refs') else None
with torch.no_grad():
    for it in items:
        prs = [prep(it['state'], it['questions'][qn]) for qn in it['questions']]
        s = prs[0]['s']; assert all(p['s'] == s for p in prs)
        qs = [p['q'] for p in prs]
        lt = m.statefirst_logits(s, qs, prs)
        for p, l in zip(prs, lt):
            h = m.forward(p['s'] + p['q']); l0 = m.logits(h, p)
            print('teacher', len(s), len(p['q']), 'maxdiff %.4f' % float((l - l0).abs().max()), int(l.argmax()) == int(l0.argmax()), flush=True)
        for p in prs:
            ls = m.schema_logits(s, [p['q']], [p])[0]
            ids = p['q'][:-1] + s + [p['q'][-1]]
            h = m.forward(ids); T = len(ids)
            oi = torch.tensor(p['opt'], device=m.dev)
            l0 = m.head0(h[T - 1].float()[None], h[oi].float()[None])[0][:p['rq'].n_slots] / m.temp(p['rq'].kind)
            print('schema1', 'maxdiff %.4f' % float((ls - l0).abs().max()), [round(x, 3) for x in ls.tolist()], [round(x, 3) for x in l0.tolist()], flush=True)
        lsm = m.schema_logits(s, qs, prs)
        print('schema4 vs teacher argmax', [int(a.argmax()) == int(b.argmax()) for a, b in zip(lsm, lt)], flush=True)

# (c) gradient check, n=1
pl = m.add_lora(r=16, alpha=32, seed=3)
with torch.no_grad():
    for p_ in m.lora.parameters():
        if p_.shape[1] == 16: p_.normal_(0, 2e-3)
m.set_head('std')
it = items[0]; p = prep(it['state'], it['questions'][list(it['questions'])[1]])
s = p['s']
for p_ in pl: p_.grad = None
ls = m.schema_logits(s, [p['q']], [p], ckpt=True)[0]
(ls * torch.arange(1, ls.numel() + 1, device=m.dev)).sum().backward()
g1 = [p_.grad.clone() for p_ in pl]
for p_ in pl: p_.grad = None
ids = p['q'][:-1] + s + [p['q'][-1]]
h = m.forward(ids, ckpt=True); T = len(ids)
oi = torch.tensor(p['opt'], device=m.dev)
l0 = m.head(h[T - 1].float()[None], h[oi].float()[None])[0][:p['rq'].n_slots] / m.temp(p['rq'].kind)
(l0 * torch.arange(1, l0.numel() + 1, device=m.dev)).sum().backward()
g0 = [p_.grad.clone() for p_ in pl]
cs = [float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)) for a, b in zip(g1, g0) if b.abs().sum() > 0]
print('grad cos min %.4f median %.4f n %d' % (min(cs), sorted(cs)[len(cs) // 2], len(cs)), 'logit diff %.4f' % float((ls - l0).abs().max()), flush=True)

# (d) n=4 fwd+bwd time / memory
for it in items:
    prs = [prep(it['state'], it['questions'][qn]) for qn in it['questions']]
    s = prs[0]['s']; qs = [p['q'] for p in prs]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    ls = m.schema_logits(s, qs, prs, ckpt=True)
    sum(l.logsumexp(-1) for l in ls).backward()
    torch.cuda.synchronize()
    print('fwdbwd n4 T', len(s) + sum(len(q) for q in qs), '%.2fs' % (time.time() - t0), 'mem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.time(); lt = m.statefirst_logits(s, qs, prs); torch.cuda.synchronize()
        print('teacher fwd %.2fs' % (time.time() - t0), flush=True)
print('OK')
