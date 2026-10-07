"""Q1 eval: run format configs on all 3227 evalkit questions (exact emulation), write preds JSON (resumable, config-outer).
python q1eval.py --cfgs cfgs.json --tags a,b --out ~/work/q1/preds/x.json [--shard 0/1] [--sub all|real]
cfgs.json: {tag: spec} (spec as in q1fmt.Fmt; tag 'dense' = bf16 runtime, no quantization)."""
import os, sys, json, time, argparse, zlib
sys.path[:0] = [os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q1fmt as QF
import evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('--cfgs', required=True); ap.add_argument('--tags', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--shard', default='0/1'); ap.add_argument('--sub', default='all'); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
CF = json.load(open(a.cfgs)); tags = a.tags.split(',')
rows = list(EK.all_question_items())
if a.sub == 'real': rows = [r for r in rows if r[0] in ('REAL-agree',)]
si, sn = map(int, a.shard.split('/')); rows = rows[si::sn]
if a.limit: rows = rows[:a.limit]
print('questions', len(rows), flush=True)
g = QL.Q1(grad=False)
res = {t: {} for t in tags}; meta = {}
if os.path.exists(a.out):
    old = json.load(open(a.out)); meta = old.get('meta', {})
    for t in tags: res[t] = old['preds'].get(t, {})
t0 = time.time()
for t in tags:
    spec = CF[t]
    if t == 'dense' or spec is None: fm = None; g.qfn = {}
    else: fm = QF.Fmt(g, spec); fm.install(g); meta[t] = dict(spec=spec, work=fm.work())
    n_new = 0
    for ri, (suite, iid, qn, state, qspec) in enumerate(rows):
        if qn in res[t].get(iid, {}): continue
        pr = g.prep(state, qspec); ids = pr['s'] + pr['q']
        if fm is not None: fm.seed = zlib.crc32(f'{iid}|{qn}'.encode())
        with torch.no_grad():
            h, _ = g.fwd(ids, q0=pr['q0'])
            res[t].setdefault(iid, {})[qn] = g.pdict(g.logits(h, pr), pr)
        n_new += 1
        if n_new % 100 == 0:
            print(f'{t} {ri}/{len(rows)} {suite} T={len(ids)} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
            json.dump(dict(preds=res, meta=meta), open(a.out + '.tmp', 'w')); os.replace(a.out + '.tmp', a.out)
    json.dump(dict(preds=res, meta=meta), open(a.out + '.tmp', 'w')); os.replace(a.out + '.tmp', a.out)
    print('config done', t, f'{time.time()-t0:.0f}s', flush=True)
    g.qfn = {}; del fm; torch.cuda.empty_cache()
print('done', f'{time.time()-t0:.0f}s', flush=True)
