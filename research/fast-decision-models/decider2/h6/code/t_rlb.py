"""plainqbr smoke: lowrows = [] must equal plainqb; lowrows = readout rows runs; k64 + ba16"""
import sys, os, json, torch
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ['BA16'] = '1'
import evalkit as EK
from kitrun import load_P, prep_question
from lean2 import Lean2
import qrt6 as Q6
from qrt import Lay
P = load_P(); ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval()
for d in ln.layers:
    for k_ in ('Wgu_il', 'Win_f', 'Wgu_f'): d.pop(k_, None)
P.model.torso = None; P.tm = None; torch.cuda.empty_cache()
items = list(EK.all_question_items(['REAL-agree']))[:6]
byid = {it['id']: it for it in EK.load_suite('REAL-agree')}
m = Q6.QRT6(ln, head=head, prec='map:~/work/h6/precmap_w4a4_k64.json', wcodes=Q6.load_codes('w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt'), split=True, slim=True); m.tune = False
def dec(pr, r0, lr):
    ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
    hn = m.forward(ids, Lay('single', T), r0=r0, lowrows=lr)
    rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
    h = m.unrot(hn[rows]).float()
    lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots]
    return [round(x, 4) for x in torch.softmax(lg, -1).tolist()]
with torch.inference_mode():
    for suite, iid, qn, st, spec in items:
        pr = prep_question(P, byid[iid], qn); T = pr['q0'] + len(pr['q'])
        a = dec(pr, pr['q0'], None); b = dec(pr, pr['q0'], torch.tensor([], dtype=torch.long, device='cuda'))
        lr = torch.tensor(sorted(set([pr['q0'] + o for o in pr['opt']] + [T - 3, T - 2, T - 1])), device='cuda')
        c = dec(pr, pr['q0'], lr); u = dec(pr, None, None)
        print(qn[:18], 'qb', a, 'qb(full,empty)', b, 'qb-readout-lowbit', c, 'uniform', u, flush=True)
