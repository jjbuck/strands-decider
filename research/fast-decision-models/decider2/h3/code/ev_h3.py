"""H3 eval: hobson (or a fine-tuned H3 arm) under quant configs over evalkit rows -> preds JSON (one per config), resumable.
python ev_h3.py --ckpt ck/v1/final.pt --cfgs bf16,w4a4,w4a4r,nvfp4 --sub dev --tag v1
Rows: subsets.json (G2's laptop-built lists): dev = REAL-ALL (1083) + CF-T (218) + CF-probe-T (210) + JB-hard (130); long = LONG-SD (165);
cf = CF-ALL + CF-probe-ALL (full CF suites)."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.expanduser('~/work/h3'))
import torch
import h3lib as H
import evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default=None); ap.add_argument('--cfgs', default='bf16,w4a4')
ap.add_argument('--sub', default='dev'); ap.add_argument('--tag', default='hobson'); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--merge', action='store_true', help='merge LoRA into bf16 weights (old behaviour; rounds small updates)')
a = ap.parse_args()
W = os.path.expanduser('~/work/h3/')
os.makedirs(W + 'preds', exist_ok=True)
SUBS = json.load(open(W + 'subsets.json'))
PLANS = dict(dev=['REAL-ALL', 'CF-T', 'CF-probe-T', 'JB-hard'], long=['LONG-SD'], cf=['CF-ALL', 'CF-probe-ALL'], real=['REAL-ALL'],
             quick=['REAL-agree-SD', 'CF-T', 'CF-probe-T'], longall=['LONG-ALL'])
rows = []
seen = set()
for nm in PLANS[a.sub]:
    for r in SUBS[nm]:
        k = (r[1], r[2])
        if k not in seen: seen.add(k); rows.append(r)
if a.limit: rows = rows[:a.limit]
items = {}
for s in ('REAL-agree', 'LONG', 'CF', 'CF-probe', 'JB-hard'):
    for it in EK.load_suite(s): items[it['id']] = it


def ab16(fmt, **kw):
    return H.QCfg([dict(w=fmt, a=fmt), dict(w='bf16', a='bf16', names=['Win.ab'])], **kw)


CFG = {
    'bf16': H.DENSE,
    'w8a8': H.Q('int8'),
    'w4a4': H.Q('int4'), 'w4a4c9': H.Q('int4', clip=0.9), 'w4a4f': H.Q('int4', fold=True), 'w4a4r': H.Q('int4', rot=True),
    'w4a16': H.Q('int4', 'bf16'), 'w16a4': H.Q('bf16', 'int4'), 'w4a8r': H.Q('int4', 'int8', rot=True),
    'nvfp4': H.Q('nvfp4'), 'nvfp4r': H.Q('nvfp4', rot=True), 'mxfp4': H.Q('mxfp4'), 'mxfp4r': H.Q('mxfp4', rot=True),
    'g128': H.Q('g128'), 'g128r': H.Q('g128', rot=True),
    'nvfp4-qb16': H.QCfg([dict(w='nvfp4', a='nvfp4')], qb16=True), 'w4a4r-qb16': H.QCfg([dict(w='int4', a='int4')], rot=True, qb16=True),
    'nvfp4r-qb16': H.QCfg([dict(w='nvfp4', a='nvfp4')], rot=True, qb16=True),
    'w4a4-ab16': ab16('int4'), 'w4a4r-ab16': ab16('int4', rot=True), 'nvfp4-ab16': ab16('nvfp4'),
}
# mixed precision from the step-1 sensitivity map: W8A8 on the layers named in MIX8 (env), W4A4 elsewhere (rotated)
if os.environ.get('MIX8'):
    L8 = [int(x) for x in os.environ['MIX8'].split(',')]
    CFG['mix8r'] = H.QCfg([dict(w='int4', a='int4'), dict(w='int8', a='int8', layers=L8)], rot=True)
    CFG['mix8'] = H.QCfg([dict(w='int4', a='int4'), dict(w='int8', a='int8', layers=L8)])

m = H.H3(); m.free_hf()
if a.ckpt:
    meta = m.load_trainable(a.ckpt, merge=a.merge); print('loaded', a.ckpt, meta, 'merged' if a.merge else 'unmerged LoRA', flush=True)
t0 = time.time()
for cname in a.cfgs.split(','):
    qc = CFG[cname]; m.wcache = {}
    out = W + f'preds/{a.tag}__{cname}__{a.sub}.json'
    preds = json.load(open(out)) if os.path.exists(out) else {}
    done = {(i, q) for i, d in preds.items() for q in d}
    n0 = len(done); t1 = time.time()
    with torch.inference_mode():
        for ri, (suite, iid, qn) in enumerate(rows):
            if (iid, qn) in done: continue
            it = items[iid]
            pr = m.prep(it['state'], it['questions'][qn])
            p = m.probs(m.forward(pr['s'] + pr['q'], qc, q0=pr['q0']), pr).tolist()
            preds.setdefault(iid, {})[qn] = {lab: p[j] for j, lab in enumerate(pr['rq'].slot_labels)}
            if ri % 100 == 0:
                json.dump(preds, open(out, 'w'))
                print(f'{a.tag} {cname} {ri}/{len(rows)} {time.time() - t1:.0f}s', flush=True)
    json.dump(preds, open(out, 'w'))
    print(f'DONE {a.tag} {cname} {len(rows)} rows ({len(rows) - n0} new) {time.time() - t1:.0f}s', flush=True)
print('all done', time.time() - t0, flush=True)
