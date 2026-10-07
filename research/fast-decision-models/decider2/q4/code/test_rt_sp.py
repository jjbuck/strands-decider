"""smoke test of QRT4's sparse state-row GEMMs: finite outputs, sparse rows really used (outputs differ from dense), question rows untouched
when q0 = T (no state rows)."""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q4')]
import q4rt as R
import evalkit as EK
from kitrun import prep_question
P, m, head = R.build()
print('sparse built', m.build_sparse(range(12, 23)), flush=True)
items = list(EK.all_question_items(['REAL-agree']))[:6]
byid = {it['id']: it for it in EK.load_suite('REAL-agree')}
with torch.inference_mode():
    for suite, iid, qn, st, spec in items:
        pr = prep_question(P, byid[iid], qn)
        ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
        rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
        lay = R.Q.Lay('single', T); m.q0 = pr['q0']
        out = []
        for spl in ((), range(12, 23)):
            m.set_sparse_layers(spl)
            hn = m.forward(ids, lay); h = m.unrot(hn[rows]).float()
            lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
            out.append(torch.softmax(lg, -1))
        print(T, pr['q0'], 'dense', [round(x, 3) for x in out[0].tolist()], 'sparse12-22', [round(x, 3) for x in out[1].tolist()], 'finite', bool(torch.isfinite(out[1]).all()), flush=True)
