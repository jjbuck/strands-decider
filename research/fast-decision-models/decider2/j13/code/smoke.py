import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from tw import TW
import evalkit as EK
tw = TW()
print('mem after load %.1fG' % (torch.cuda.memory_allocated() / 1e9), flush=True)
refs = EK.load_refs('REAL-agree')
its = EK.load_suite('REAL-agree')[:12]
tv = []; fl = 0; n = 0; t0 = time.time(); ntok = 0
with torch.no_grad():
    for it in its:
        for qn in list(it['questions'])[:2]:
            pr = tw.prep(it, qn)
            x, cos, sin = tw.embed_rope(pr['ids'])
            xo, _ = tw.run(x, cos, sin)
            pd = tw.probdict(pr, tw.head(xo, pr)); h = refs[it['id']]['hobson'][qn]
            tv.append(EK._tv(pd, h)); fl += EK._arg(pd) != EK._arg(h); n += 1; ntok += len(pr['ids'])
torch.cuda.synchronize()
print('n', n, 'mean tv %.4f max %.4f flips %d' % (sum(tv) / n, max(tv), fl), '%.0f tok/s' % (ntok / (time.time() - t0)), flush=True)
# timing on a long input
it = max(EK.load_suite('REAL-agree'), key=lambda i: i['n_state_tok'])
pr = tw.prep(it, next(iter(it['questions'])))
with torch.no_grad():
    x, cos, sin = tw.embed_rope(pr['ids'])
    for _ in range(2): tw.run(x, cos, sin)
    torch.cuda.synchronize(); t0 = time.time(); tw.run(x, cos, sin); torch.cuda.synchronize()
print('T', len(pr['ids']), 'dense fwd %.1f ms' % ((time.time() - t0) * 1000), flush=True)
