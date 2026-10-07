import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/j14'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from j14lib import J14, live_positions
m = J14(); m.setup(); m.free_hf()
print('U', m.U, m.tok.convert_ids_to_tokens(m.U))
its = [x for x in EK.all_question_items(['REAL-agree'])][:40:4]
with torch.inference_mode():
    for s, iid, q, st, spec in its[:6]:
        pr = m.prep(st, spec)
        h = m.forward(list(pr['s']) + list(pr['q']))
        p0 = m.probs(h, pr)
        cache, T = m.prefix(pr['s'])
        Lq = len(pr['q'])
        hl = m.suffix(cache, T, pr['q'], list(range(Lq)))
        p1 = torch.softmax(m.readout(hl, list(range(Lq)), pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots).float(), -1)
        out = [f'{iid[:20]} {q[:18]} T={T} Lq={Lq} nopt={len(pr["opt"])} plain={p0.tolist()[:3]} allive={p1.tolist()[:3]}']
        for lv in ('oa', 'oa+sfx', 'oa+sfx+ok4', 'ob'):
            lp = live_positions(m.tok, pr, lv)
            hc = m.suffix(cache, T, pr['q'], lp)
            comp = m.comp_out
            pc = torch.softmax(m.readout(hc, lp, pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots).float(), -1)
            hc2 = m.suffix(cache, T, pr['q'], lp, comp=comp)
            pc2 = torch.softmax(m.readout(hc2, lp, pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots).float(), -1)
            out.append(f'  {lv} nl={len(lp)} p={[round(x,4) for x in pc.tolist()[:3]]} cached_diff={float((pc-pc2).abs().max()):.2e}')
        print('\n'.join(out), flush=True)
    # empty-state identity: compile context == state -> compiled layout must equal hobson
    for s, iid, q, st, spec in its[:4]:
        pr = m.prep('', spec)
        cache, T = m.prefix(pr['s']); Lq = len(pr['q'])
        assert list(pr['s']) == list(m.U), (pr['s'], m.U)
        hl = m.suffix(cache, T, pr['q'], list(range(Lq)))
        p1 = torch.softmax(m.readout(hl, list(range(Lq)), pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots).float(), -1)
        lp = live_positions(m.tok, pr, 'oa')
        hc = m.suffix(cache, T, pr['q'], lp)
        pc = torch.softmax(m.readout(hc, lp, pr['opt'], Lq, pr['rq'].kind, pr['rq'].n_slots).float(), -1)
        print('EMPTY-STATE identity', q, 'maxdiff', float((p1 - pc).abs().max()), flush=True)
    # timing
    st = its[0][3]; pr = m.prep(st, its[0][4])
    for _ in range(2): m.prefix(pr['s'])
    torch.cuda.synchronize(); t0 = time.time(); m.prefix(pr['s']); torch.cuda.synchronize()
    print('prefix', len(pr['s']), 'tok', time.time() - t0)
