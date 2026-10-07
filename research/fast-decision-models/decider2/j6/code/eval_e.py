"""J6 Brooker test, evaluation of the 'early' arm (question in the weights from layer 0: per-question adapter on the state rows too).
One full pass per (item, question). python eval_e.py CKPT_EARLY OUT.json [--suites REAL-agree,LONG,CF] [--limit N]"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/evalkit')]
import torch
from j6lib import J6, QAdapters, QStateAd
from h3lib import StdHead
import qtab, evalkit as EK
ap = argparse.ArgumentParser(); ap.add_argument('ckpt'); ap.add_argument('out'); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--suites', default='REAL-agree,LONG,CF')
a = ap.parse_args()
m = J6(); m.free_hf(); dev = m.dev; eng = m.p.eng
pool = qtab.load_pool(); specs = qtab.deployed_specs(pool)
PQ = {q: qtab.prep_q(eng, specs[q]) for q in specs}
ck = torch.load(os.path.expanduser(a.ckpt), map_location=dev, weights_only=False)
qn = ck['qnames']; ar = ck['args']
ad = QAdapters(m, qn, {q: torch.zeros(PQ[q]['K'] + 1, 2048) for q in qn}, r_s=ar['r_s'], r_q=ar['r_q'])
ad.load_state_dict(ck['ad'])
sa = QStateAd(m, len(qn), r=ck.get('r_e', 8)); sa.load_state_dict(ck['state_ad']); ad.state_ad = sa
head = StdHead(m.head0).to(dev); head.load_state_dict(ck['head']); head.eval()
preds = json.load(open(a.out)) if os.path.exists(a.out) else {}
items = []; seen = set()
for sname in a.suites.split(','):
    for it in EK.load_suite(sname):
        if it['id'] in seen: continue
        seen.add(it['id']); items.append(it)
if a.limit: items = items[:a.limit]
t0 = time.time(); n = 0
with torch.no_grad():
    for it in items:
        if it['id'] in preds: continue
        names = [q for q in it['questions'] if q in ad.qi and json.dumps(it['questions'][q], sort_keys=True) == json.dumps(specs[q], sort_keys=True)]
        if not names: continue
        s = qtab.state_ids(eng, it['state']); out = {}
        for q in names:
            p = PQ[q]; K = p['K']
            ad.cur = [ad.qi[q]]; sa.cur = ad.qi[q]; ad.Lslot = len(s)
            hs = m.full(s, ad.slot_inputs([q]).to(torch.bfloat16), ad, ckpt=False)
            pr = torch.softmax(m.head_logits(head, hs[K], hs[:K], p['kind']).float(), -1).tolist()
            out[q] = {lab: pr[k] for k, lab in enumerate(p['rq'].slot_labels)}
        preds[it['id']] = out; n += 1
        if n % 100 == 0:
            print(n, len(items), '%.0fs' % (time.time() - t0), flush=True); json.dump(preds, open(a.out, 'w'))
json.dump(preds, open(a.out, 'w'))
print('done', n, time.time() - t0, flush=True)
