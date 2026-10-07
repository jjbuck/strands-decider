"""sets layout checks: n=1 == plain [Q'][state][opt tokens][<answer>] sequence (forward + LoRA grads); multi-question run; timing."""
import os, sys, time
sys.path[:0] = [os.path.expanduser('~/work/h7'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from h7lib import H7
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
m = H7(); m.detach_inference(); eng = m.p.eng
def prep(state, qd):
    q = ta.validate_python(qd); rq = render_question(q)
    s, qs = eng._fit(render_state(state), [rq.text]); return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq)
items = [it for it in EK.load_suite('REAL-agree') if len(it['questions']) == 4][:2] + [it for it in EK.load_suite('JB-all')][:2]
def plain(p, s, head):
    q = p['q']; ids = q[:-1] + s + [q[o] for o in p['opt']] + [q[-1]]; T = len(ids); K = len(p['opt'])
    h = m.forward(ids, ckpt=torch.is_grad_enabled())
    return head(h[T - 1].float()[None], h[T - 1 - K:T - 1].float()[None])[0][:p['rq'].n_slots] / m.temp(p['rq'].kind)
with torch.no_grad():
    for it in items:
        prs = [prep(it['state'], it['questions'][qn]) for qn in it['questions']]; s = prs[0]['s']
        for p in prs:
            a = m.sets_logits(s, [p['q']], [p])[0]; b = plain(p, s, m.head0)
            print('sets1 vs plain maxdiff %.4f' % float((a - b).abs().max()), len(p['opt']), flush=True)
        lt = m.statefirst_logits(s, [p['q'] for p in prs], prs); ls = m.sets_logits(s, [p['q'] for p in prs], prs)
        print('untrained sets(n=%d) vs teacher argmax' % len(prs), [int(x.argmax()) == int(y.argmax()) for x, y in zip(ls, lt)],
              [round(float(torch.softmax(x, -1).max()), 2) for x in lt], flush=True)
pl = m.add_lora(r=16, alpha=32, seed=3)
with torch.no_grad():
    for p_ in m.lora.parameters():
        if p_.shape[1] == 16: p_.normal_(0, 2e-3)
m.set_head('std')
p = prep(items[0]['state'], items[0]['questions'][list(items[0]['questions'])[0]]); s = p['s']
for p_ in pl: p_.grad = None
l1 = m.sets_logits(s, [p['q']], [p], ckpt=True)[0]; (l1 * torch.arange(1, l1.numel() + 1, device=m.dev)).sum().backward(); g1 = [x.grad.clone() for x in pl]
for p_ in pl: p_.grad = None
l0 = plain(p, s, m.head); (l0 * torch.arange(1, l0.numel() + 1, device=m.dev)).sum().backward(); g0 = [x.grad.clone() for x in pl]
cs = [float(torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)) for a, b in zip(g1, g0) if b.abs().sum() > 0]
print('grad cos min %.4f median %.4f' % (min(cs), sorted(cs)[len(cs) // 2]), flush=True)
prs = [prep(items[0]['state'], items[0]['questions'][qn]) for qn in items[0]['questions']]; s = prs[0]['s']
for r in range(2):
    torch.cuda.synchronize(); t0 = time.time()
    ls = m.sets_logits(s, [p['q'] for p in prs], prs, ckpt=True); sum(l.logsumexp(-1) for l in ls).backward(); torch.cuda.synchronize()
    print('fwdbwd sets n4 %.2fs' % (time.time() - t0), flush=True)
print('OK')
