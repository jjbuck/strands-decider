import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import torch, h1lib as HL, evalkit as EK
g = HL.H1(); print('intmm', g._intmm, 'mem', torch.cuda.memory_allocated() / 1e9, flush=True)
q = json.load(open(os.path.expanduser('~/work/g2/res/qmain.json')))['preds']
its = {it['id']: it for it in EK.load_suite('REAL-agree')}
ids_ = list(q['dense'].keys())[:6]
for c in ('dense', 'w8a8', 'w4a4'):
    g.set_prec({} if c == 'dense' else g.uniform(c)); t0 = time.time(); out = []
    for iid in ids_:
        if iid not in its: continue
        for qn in list(its[iid]['questions'])[:1]:
            pr = g.prep(its[iid]['state'], its[iid]['questions'][qn]); ids = pr['s'] + pr['q']
            torch.cuda.synchronize(); t1 = time.time(); h, _ = g.fwd(ids); torch.cuda.synchronize(); dt = time.time() - t1
            p = g.pdict(g.logits(h, pr), pr); ref = q['dense' if c == 'dense' else ('Q:r8' if c == 'w8a8' else 'Q:r4')][iid][qn]
            out.append((len(ids), round(dt, 3), round(max(abs(p[k] - ref[k]) for k in p), 4)))
    print(c, out, 'mem', torch.cuda.max_memory_allocated() / 1e9, flush=True)
