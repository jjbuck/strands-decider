"""ba16 smoke: k48 uniform, k48 + ba16 (r0=T path), plainqb k48 + ba16, vs fold bf16 on 8 REAL questions"""
import sys, os, json, torch
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import evalkit as EK
from kitrun import load_P, prep_question
from lean2 import Lean2
import qrt6 as Q6
from qrt import Lay
P = load_P(); ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval()
for d in ln.layers:
    for k_ in ('Wgu_il', 'Win_f', 'Wgu_f'): d.pop(k_, None)
items = list(EK.all_question_items(['REAL-agree']))[:8]
byid = {it['id']: it for it in EK.load_suite('REAL-agree')}
codes = Q6.load_codes('w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt')
ref = {json.loads(l)['id'] + json.loads(l)['q']: json.loads(l)['probs'] for l in open(os.path.expanduser('~/work/h2/preds_bf16.jsonl'))}
os.environ['BA16'] = '1'
mq = Q6.QRT6(ln, head=head, prec='map:~/work/h2/precmap_w4a4_k48.json', wcodes=codes, split=True, slim=True); mq.tune = False
def dec(m, pr, r0):
    ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
    hn = m.forward(ids, Lay('single', T), r0=r0)
    rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
    h = m.unrot(hn[rows]).float()
    lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots]
    return [round(x, 3) for x in torch.softmax(lg, -1).tolist()]
with torch.inference_mode():
    for suite, iid, qn, st, spec in items:
        pr = prep_question(P, byid[iid], qn)
        mq.ba16 = False; u = dec(mq, pr, None); qb = dec(mq, pr, pr['q0'])
        mq.ba16 = True; ub = dec(mq, pr, None); qbb = dec(mq, pr, pr['q0'])
        print(qn[:18], 'bf16', [round(x, 3) for x in ref[iid + qn].values()], 'k48', u, 'k48ba', ub, 'qb', qb, 'qb+ba', qbb, flush=True)
