"""Q5 kit eval: structure configs on all 3227 evalkit questions (exact emulation), preds JSON (resumable), like q1eval.
python q5kit.py --cfgs cfgs.json --tags a,b --out ~/work/q5/preds/x.json [--sub all|bank] [--shard 0/1]
tag 'dense' with cfg null = the bf16 runtime."""
import os, sys, json, time, argparse, hashlib
sys.path[:0] = [os.path.expanduser('~/work/q5'), os.path.expanduser('~/work/q1')]
import torch
import q1lib as QL, q5lib as Q5
import evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('--cfgs', required=True); ap.add_argument('--tags', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--shard', default='0/1'); ap.add_argument('--sub', default='all'); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--start', type=int, default=0, help='reuse the cached residual entering this layer (layers below must equal the base format)')
ap.add_argument('--check', type=int, default=0, help='compare cached-prefix logits with a full forward on N questions')
a = ap.parse_args()
CF = json.load(open(a.cfgs)); tags = a.tags.split(',')
rows = list(EK.all_question_items())
if a.sub == 'bank':
    dom = {}
    for s in ('REAL-agree', 'LONG', 'CF', 'CF-probe'):
        for it in EK.load_suite(s): dom[it['id']] = it.get('domain', '')
    rows = [r for r in rows if dom.get(r[1], '').startswith('banking')]
si, sn = map(int, a.shard.split('/')); rows = rows[si::sn]
if a.limit: rows = rows[:a.limit]
print('questions', len(rows), flush=True)
g = QL.Q1(grad=False)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
res = {t: {} for t in tags}; meta = {}
if os.path.exists(a.out):
    old = json.load(open(a.out)); meta = old.get('meta', {})
    for t in tags: res[t] = old['preds'].get(t, {})
    for t, v in old['preds'].items():
        if t not in res: res[t] = v
t0 = time.time()
for t in tags:
    cfg = CF.get(t)
    if cfg is None: g.qfn = {}; S = None
    else:
        S = Q5.S5(g, cfg, tag=t); S.install(); meta[t] = dict(cfg=cfg, work=S.work(), stats=S.stats)
    n_new = 0; st = a.start
    if st:
        if S is not None:
            assert all(i >= st for (i, k) in S.sp) and all(i >= st for i in S.nr), 'structure below --start'
        cname = ((cfg or {}).get('fmt_name') or ('dense' if not (cfg or {}).get('fmt') else None))
        assert cname, 'fmt_name needed for the prefix cache'
        if (cfg or {}).get('fcal'): cname += '.f' + cfg['fcal']
        cdir = f'{Q5.W5}/xcache/{cname}_L{st}'; os.makedirs(cdir, exist_ok=True)
    nchk = 0; maxd = 0.0
    for ri, (suite, iid, qn, state, qspec) in enumerate(rows):
        if qn in res[t].get(iid, {}) and not (a.check and nchk < a.check): continue
        pr = g.prep(state, qspec); ids = pr['s'] + pr['q']
        with torch.no_grad():
            if st:
                fn = f"{cdir}/{hashlib.md5(f'{iid}|{qn}'.encode()).hexdigest()}.pt"
                if os.path.exists(fn): x0 = torch.load(fn, map_location=g.dev)
                else:
                    _, caps = g.fwd(ids, keep_x=(st - 1,), stop=st, q0=pr['q0']); x0 = caps[st - 1]; torch.save(x0.cpu(), fn)
                h, _ = Q5.fwd_from(g, ids, x0, st, q0=pr['q0'])
                if a.check and nchk < a.check:
                    h2, _ = g.fwd(ids, q0=pr['q0']); d_ = float((g.logits(h, pr) - g.logits(h2, pr)).abs().max()); maxd = max(maxd, d_); nchk += 1
                    if nchk == a.check: print(f'CHECK cached prefix vs full forward: {nchk} questions, max |d logit| = {maxd}', flush=True)
            else:
                h, _ = g.fwd(ids, q0=pr['q0'])
            res[t].setdefault(iid, {})[qn] = g.pdict(g.logits(h, pr), pr)
        n_new += 1
        if n_new % 200 == 0:
            print(f'{t} {ri}/{len(rows)} {suite} T={len(ids)} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
            json.dump(dict(preds=res, meta=meta), open(a.out + '.tmp', 'w')); os.replace(a.out + '.tmp', a.out)
    json.dump(dict(preds=res, meta=meta), open(a.out + '.tmp', 'w')); os.replace(a.out + '.tmp', a.out)
    print('config done', t, f'{time.time()-t0:.0f}s', flush=True)
    g.qfn = {}; del S; torch.cuda.empty_cache()
print('done', f'{time.time()-t0:.0f}s', flush=True)
