"""J6 (a) evaluation: question-in-weights student on every evalkit item whose questions are deployed questions (REAL-agree, LONG, CF; SHUF and
REAL-label are views of CF / REAL-agree). Also (--teacher) hobson in context in the same runtime, as the runtime-noise reference.
python eval_a.py CKPT OUT.json [--teacher] [--limit N] [--suites REAL-agree,LONG,CF]"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/evalkit')]
import torch, torch.nn.functional as F
from j6lib import J6, Seg, QAdapters, FastState
from h3lib import StdHead
import qtab, evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('--teacher', action='store_true')
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--suites', default='REAL-agree,LONG,CF')
a = ap.parse_args()
m = J6(); fs = FastState(m); m.free_hf(); dev = m.dev; eng = m.p.eng
pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}
ad = None
if a.ckpt != 'none':
    ck = torch.load(os.path.expanduser(a.ckpt), map_location=dev, weights_only=False)
    qn = ck['qnames']; ar = ck['args']
    ad = QAdapters(m, qn, {q: torch.zeros(PQ[q]['K'] + 1, 2048) for q in qn}, r_s=ar['r_s'], r_q=ar['r_q'])
    ad.load_state_dict(ck['ad']); ad.eval()
    head = StdHead(m.head0).to(dev); head.load_state_dict(ck['head']); head.eval()

preds = {}; tpreds = {}
if os.path.exists(a.out):
    d = json.load(open(a.out)); preds = d.get('student', {}); tpreds = d.get('teacher', {})
items = []
seen = set()
for sname in a.suites.split(','):
    for it in EK.load_suite(sname):
        if it['id'] in seen: continue
        seen.add(it['id']); items.append(it)
if a.limit: items = items[:a.limit]
t0 = time.time(); n = 0
with torch.no_grad():
    for it in items:
        if it['id'] in preds and (not a.teacher or it['id'] in tpreds): continue
        names = [q for q in it['questions'] if q in PQ and json.dumps(it['questions'][q], sort_keys=True) == json.dumps(specs[q], sort_keys=True)]
        if not names: continue
        s = qtab.state_ids(eng, it['state']); Ls = len(s)
        cache = fs(s)
        if ad is not None:
            nm2 = [q for q in names if q in ad.qi]
        if ad is not None and nm2:
            ss = Seg(Ls, [PQ[q]['K'] + 1 for q in nm2], dev); ad.seg = ss; ad.cur = [ad.qi[q] for q in nm2]
            hs, _ = m.branch(ad.slot_inputs(nm2).to(torch.bfloat16), ss, cache, ad)
            out = {}
            for j, q in enumerate(nm2):
                p = PQ[q]; K = p['K']; r0 = ss.r0[j]
                pr = torch.softmax(m.head_logits(head, hs[r0 + K], hs[r0:r0 + K], p['kind']).float(), -1).tolist()
                out[q] = {lab: pr[k] for k, lab in enumerate(p['rq'].slot_labels)}
            preds[it['id']] = out
        if a.teacher:
            qt = [PQ[q] for q in names]
            xt = F.embedding(torch.tensor([t for p in qt for t in p['q']], device=dev), m.embed)
            sg = Seg(Ls, [len(p['q']) for p in qt], dev)
            ht, _ = m.branch(xt, sg, cache, None)
            out = {}
            for j, q in enumerate(names):
                p = qt[j]; r0 = sg.r0[j]; K = p['K']
                oi = torch.tensor([r0 + o for o in p['opt']], device=dev)
                pr = torch.softmax(m.head_logits(m.head0, ht[r0 + len(p['q']) - 1], ht[oi], p['kind']).float(), -1).tolist()
                out[q] = {lab: pr[k] for k, lab in enumerate(p['rq'].slot_labels)}
            tpreds[it['id']] = out
        n += 1
        if n % 100 == 0:
            print(n, len(items), '%.0fs' % (time.time() - t0), flush=True)
            json.dump(dict(student=preds, teacher=tpreds), open(a.out, 'w'))
json.dump(dict(student=preds, teacher=tpreds), open(a.out, 'w'))
print('done', n, time.time() - t0, flush=True)
