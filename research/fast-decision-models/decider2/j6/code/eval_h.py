"""J6 (b) evaluation: the hypernetwork compiles every question spec it is given (deployed, held-out, JevBench, CF-probe) into a weight delta once,
then answers each item with the question in the weights. python eval_h.py CKPT OUT.json [--suites ...] [--limit N]"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from j6lib import J6, Seg, HyperAdapters, FastState
from h3lib import StdHead
import qtab, evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--suites', default='JB-all,REAL-agree,LONG,CF,CF-probe')
a = ap.parse_args()
m = J6(); fs = FastState(m); m.free_hf(); dev = m.dev; eng = m.p.eng
S0 = qtab.state_ids(eng, '')
ck = torch.load(os.path.expanduser(a.ckpt), map_location=dev, weights_only=False)
ha = HyperAdapters(m, r_h=ck['args']['r_h']); ha.load_state_dict(ck['ha']); ha.eval()
head = StdHead(m.head0).to(dev); head.load_state_dict(ck['head']); head.eval()
preds = json.load(open(a.out)) if os.path.exists(a.out) else {}
CODE = {}


def compile_q(spec):
    key = json.dumps(spec, sort_keys=True)
    if key not in CODE:
        pq = qtab.prep_q(eng, spec)
        code = ha.generate(ha.features(m, pq, S0), qtab.slot_init(m, eng, pq))
        CODE[key] = (pq, code)
    return CODE[key]


items = []; seen = set()
for sname in a.suites.split(','):
    for it in EK.load_suite(sname):
        if it['id'] in seen: continue
        seen.add(it['id']); items.append(it)
if a.limit: items = items[:a.limit]
t0 = time.time(); n = 0; tc = 0.0
with torch.no_grad():
    for it in items:
        if it['id'] in preds: continue
        names = list(it['questions'])
        t1 = time.time(); cq = [compile_q(it['questions'][q]) for q in names]; tc += time.time() - t1
        s = qtab.state_ids(eng, it['state']); Ls = len(s)
        cache = fs(s)
        ss = Seg(Ls, [pq['K'] + 1 for pq, _ in cq], dev); ha.seg = ss; ha.cur = [c for _, c in cq]
        hs, _ = m.branch(torch.cat([c['slots'] for _, c in cq], 0).to(torch.bfloat16), ss, cache, ha)
        out = {}
        for j, q in enumerate(names):
            pq = cq[j][0]; K = pq['K']; r0 = ss.r0[j]
            pr = torch.softmax(m.head_logits(head, hs[r0 + K], hs[r0:r0 + K], pq['kind']).float(), -1).tolist()
            out[q] = {lab: pr[k] for k, lab in enumerate(pq['rq'].slot_labels)}
        preds[it['id']] = out
        n += 1
        if n % 200 == 0:
            print(n, len(items), '%.0fs' % (time.time() - t0), 'compile %.0fs for %d specs' % (tc, len(CODE)), flush=True)
            json.dump(preds, open(a.out, 'w'))
json.dump(preds, open(a.out, 'w'))
print('done', n, time.time() - t0, 'specs compiled', len(CODE), flush=True)
