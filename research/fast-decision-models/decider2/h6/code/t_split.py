"""smoke: QRT6 row split. (a) r0=0 (all rows via bf16 split path) vs fold bf16; (b) r0=None uniform k48 vs H2 preds; (c) r0=q0."""
import sys, os, json, torch
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import evalkit as EK
from kitrun import load_P, prep_question
from lean2 import Lean2
import qrt6 as Q6
from qrt import Lay
P = load_P(); ln = Lean2(P.tm, fuse='fold'); head = P.model.head.float().eval()
items = list(EK.all_question_items(['REAL-agree']))[:12]
byid = {it['id']: it for it in EK.load_suite('REAL-agree')}
codes = Q6.load_codes('w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt')
mb = Q6.QRT6(ln, head=head, prec='bf16'); mb.tune = False
mq = Q6.QRT6(ln, head=head, prec='map:~/work/h2/precmap_w4a4_k48.json', wcodes=codes, split=True); mq.tune = False
print('mem', torch.cuda.memory_allocated() / 1e9, flush=True)
def dec(m, pr, r0):
    ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
    hn = m.forward(ids, Lay('single', T), r0=r0)
    rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
    h = m.unrot(hn[rows]).float()
    lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots]
    return torch.softmax(lg, -1), h
with torch.inference_mode():
    for suite, iid, qn, st, spec in items:
        pr = prep_question(P, byid[iid], qn)
        pb, hb = dec(mb, pr, None); p0, h0 = dec(mq, pr, 0); pu, hu = dec(mq, pr, None); pq, hq = dec(mq, pr, pr['q0'])
        T = pr['q0'] + len(pr['q'])
        pt, ht = dec(mq, pr, T - 1)
        cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
        print(qn[:18], T, 'bf16', [round(x, 3) for x in pb.tolist()], 'split0', [round(x, 3) for x in p0.tolist()], f'cos {cos(hb, h0):.5f}',
              '| k48', [round(x, 3) for x in pu.tolist()], f'cos {cos(hb, hu):.4f}', '| qb16', [round(x, 3) for x in pq.tolist()], f'cos {cos(hb, hq):.4f}',
              '| lastrow-bf16', f'cos {cos(hb, ht):.4f}', flush=True)
