"""H5: run every evalkit question item (JB-all, REAL-agree, LONG, CF, CF-probe; 3227 deduplicated) through a saved quantized hobson
(and/or the dense bf16 reference of the same runtime). Token ids = kitrun/plib prep (max_length 16384, identical to the references).
python h5suite.py --q q/NAME.pt --out res/preds_NAME.jsonl    |   python h5suite.py --ref --out res/preds_dense.jsonl
Resumable (one JSON line per (item, question)); --shard i/n."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.expanduser('~/work/h5'))
import torch
import h5lib as H
import evalkit as EK
from kitrun import probdict

ap = argparse.ArgumentParser(); ap.add_argument('--q', default=''); ap.add_argument('--ref', action='store_true'); ap.add_argument('--out', required=True)
ap.add_argument('--shard', default='0/1'); ap.add_argument('--suites', default='')
a = ap.parse_args()
W = os.path.expanduser('~/work/h5/')
m = H.Q5(lean=True)
mode = 'ref'
if a.q:
    D = torch.load(W + a.q)
    m.Wr = [{k: v.to(m.dev) for k, v in d.items()} for d in D['Wr']]; m.R1 = D['R1'].to(m.dev).float(); m.cfg = D['cfg']; m.aclip = D['aclip']; m.ba_hp = D['ba_hp']; m.layer_cfg = D.get('layer_cfg', {})
    m.K = {k: (v[0].to(m.dev), v[1].to(m.dev)) for k, v in D.get('K', {}).items()}; m.clipA = D.get('clipA', {})
    H.FMT.update(D.get('fmt', {}))
    if D.get('norot'): m.Ho = H.Ident(); m.Hd = H.Ident()
    del D; mode = 'q'
out = W + a.out
done = set()
if os.path.exists(out):
    good = []
    for l in open(out):
        try: r = json.loads(l); done.add((r['id'], r['q'])); good.append(l)
        except Exception: pass
    open(out, 'w').writelines(good)
si, sn = map(int, a.shard.split('/'))
items = list(EK.all_question_items(a.suites.split(',') if a.suites else None))[si::sn]
print('items', len(items), 'done', len(done), 'mode', mode, flush=True)
t0 = time.time(); n = 0
with open(out, 'a') as f, torch.no_grad():
    for s, iid, q, st, spec in items:
        if (iid, q) in done: continue
        pr = m.prep(st, spec); ids = pr['s'] + pr['q']
        h, _ = m.forward(ids, mode)
        p = torch.softmax(m.logits(h, pr).float(), -1).tolist()
        f.write(json.dumps(dict(id=iid, q=q, suite=s, T=len(ids), p=probdict(pr['rq'], p))) + '\n'); f.flush()
        n += 1
        if n % 100 == 0: print(n, len(items), s, f'{time.time() - t0:.0f}s', flush=True)
print('done', n, round(time.time() - t0), flush=True)
