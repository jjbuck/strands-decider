"""J3 smoke test for dtset: invariance (rotated options -> same distribution), untrained agreement with hobson, gradient flow."""
import os, sys, json
sys.path[:0] = [os.path.expanduser('~/work/j3'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from dtlib import DT
import dtset
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
m = DT(); m.detach_inference(); eng = m.p.eng; m.head = m.head0


def prep(state_text, qd, order=None):
    q = ta.validate_python(qd); rq = render_question(q, option_order=order) if order is not None else render_question(q)
    s, qs = eng._fit(state_text, [rq.text])
    return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd)


its = [x for x in EK.load_suite('REAL-agree') if len(x['questions']) >= 2][:6] + EK.load_suite('JB-hard')[:6]
refs = {}
for s_ in ('REAL-agree', 'JB-all'):
    for iid, r in EK.load_refs(s_).items(): refs[iid] = r.get('hobson', {})
inv = []; agree = {8: 0, 24: 0}; n = 0
with torch.no_grad():
    for it in its:
        st = render_state(it['state']); names = list(it['questions'])
        prs = [prep(st, it['questions'][q]) for q in names]
        rot = []
        for q in names:
            qd = it['questions'][q]; K = 2 if qd['type'] == 'noul' else len(qd['criteria'])
            rot.append(prep(st, qd, [(i + 1) % K for i in range(K)] if qd['type'] != 'score' else list(reversed(range(K)))))
        for Ls in (8, 24):
            a = dtset.set_logits(m, prs[0]['s'], prs, Ls, 'A'); b = dtset.set_logits(m, prs[0]['s'], rot, Ls, 'A')
            for qn, p, r, la, lb in zip(names, prs, rot, a, b):
                pa = dict(zip(p['rq'].slot_labels, torch.softmax(la.float(), -1).tolist())); pb = dict(zip(r['rq'].slot_labels, torch.softmax(lb.float(), -1).tolist()))
                inv.append(max(abs(pa[k] - pb[k]) for k in pa))
                rd = refs.get(it['id'], {}).get(qn)
                if rd: agree[Ls] += int(max(pa, key=pa.get) == max(rd, key=rd.get)); n += (Ls == 8)
print('invariance max|dp| under rotation: max %.4f median %.5f (n %d)' % (max(inv), sorted(inv)[len(inv) // 2], len(inv)), flush=True)
print('untrained set layout argmax agreement with hobson refs:', {k: f'{v}/{n}' for k, v in agree.items()}, flush=True)
# gradient flow
pl = m.add_lora(r=4, alpha=8); pm = m.add_mem([8], 'A', r=4, alpha=4)
it = its[0]; st = render_state(it['state']); prs = [prep(st, it['questions'][q]) for q in it['questions']]
out = dtset.set_logits(m, prs[0]['s'], prs, 8, 'A', ckpt=True, keep=(11, 23))
loss = sum(o.float().logsumexp(-1) for o in out) + sum(v.float().pow(2).mean() for v in m.kept.values()); loss.backward()
gB = [float(m.lora[i]['Win'].B.grad.abs().sum()) for i in (0, 7, 8, 23)]; gM = [float(p.grad.abs().sum()) for p in pm if p.grad is not None]
print('grad |B| Win layers 0,7,8,23:', gB, 'mem grads nonzero:', sum(g > 0 for g in gM), '/', len(pm), flush=True)
print('peak mem GB', torch.cuda.max_memory_allocated() / 1e9)
