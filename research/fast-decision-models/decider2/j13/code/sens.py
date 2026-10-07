"""Step 1b: decision sensitivity of projecting ONE layer's state-token GEMM inputs (all 4 GEMMs of that layer) onto the top-r
subspace, and of the cumulative all-thin path (every layer >= k).  Sinks (first 4 rows) and question rows stay full.
python sens.py LIST OUT   (resumable; one json line per question)"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from tw import TW, SINK
import evalkit as EK

lst, out = sys.argv[1], sys.argv[2]
what = sys.argv[3] if len(sys.argv) > 3 else 'both'
L = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lists.json')))[lst]
PER = [('in', r) for r in (64, 128, 256, 512)] + [('out', r) for r in (128, 512)]
CUM = [(m, k, r) for m in ('in', 'out') for k in (4, 8) for r in (128, 256, 512)]
if what == 'cum': PER = []
if what == 'per': CUM = []
done = set()
if os.path.exists(out):
    for l in open(out):
        try: j = json.loads(l); done.add((j['id'], j['q']))
        except Exception: pass
tw = TW(bases=os.path.expanduser('~/work/j13/bases.pt'))
items = {}
for s in {x[0] for x in L}:
    for it in EK.load_suite(s): items[it['id']] = it
t0 = time.time()
with open(out, 'a') as fo, torch.no_grad():
    for n, (suite, iid, qn) in enumerate(L):
        if (iid, qn) in done: continue
        pr = tw.prep(items[iid], qn); q0 = pr['q0']; T = len(pr['ids'])
        x, cos, sin = tw.embed_rope(pr['ids'])
        xf, kept = tw.run(x, cos, sin, keep=set(range(tw.NL)) if PER else {4, 8})
        rec = {'suite': suite, 'id': iid, 'q': qn, 'n_s': q0, 'T': T, 'dense': tw.probdict(pr, tw.head(xf, pr)), 'cfg': {}}
        thin = list(range(SINK, q0))
        for mode, r in PER:
            pl = tw.thin_plan(mode, r, thin, T)
            for l in range(tw.NL):
                xo, _ = tw.run(kept[l], cos, sin, l, plans={l: pl})
                rec['cfg'][f'L{l}{mode[0]}{r}'] = tw.probdict(pr, tw.head(xo, pr))
        for mode, k, r in CUM:
            pl = tw.thin_plan(mode, r, thin, T)
            xo, _ = tw.run(kept[k], cos, sin, k, plans={l: pl for l in range(k, tw.NL)})
            rec['cfg'][f'thin@{k}r{r}{mode[0]}'] = tw.probdict(pr, tw.head(xo, pr))
            rec.setdefault('fl', {})[f'thin@{k}r{r}{mode[0]}'] = round(tw.flops_ratio(T, len(thin), k, r), 4)
        del kept
        fo.write(json.dumps(rec) + '\n'); fo.flush()
        if n % 10 == 0: print(n, len(L), T, '%.0fs' % (time.time() - t0), 'mem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
print('done')
