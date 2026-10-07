"""M2 training-free sweep: hobson-v19 weights + hobson head through m2lib layouts.
python m2tf.py OUTDIR --configs 'name=spec|name=spec|...' [--subset sweep|full] [--suites ...] [--limit N] [--ckpt CK]
  subset 'sweep': every REAL-label item (316 REAL-agree requests, all their questions), 40 LONG, all 130 JB-hard, 120 CF pairs,
                  120 CF-probe pairs (hash-selected as J3)
Writes OUTDIR/<name>.json preds {iid: {q: {label: p}}} (+ .part.jsonl, resumable) and OUTDIR/<name>.stats.json (segments, rows)."""
import os, sys, json, time, argparse, collections, hashlib
sys.path[:0] = [os.path.expanduser('~/work/m2'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from m2lib import M2, Cfg

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--configs', required=True)
ap.add_argument('--subset', default='sweep'); ap.add_argument('--suites', default=''); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--ckpt', default=''); ap.add_argument('--shard', default='0/1')
a = ap.parse_args()
OUT = os.path.expanduser(a.out); os.makedirs(OUT, exist_ok=True)
CFGS = []
for c in a.configs.split('|'):
    nm, sp = c.split('=', 1)
    CFGS.append((nm, Cfg(sp)))
m = M2(); m.free_hf(); m.setup()
if a.ckpt:
    import m2train_util as TU
    TU.load_student(m, a.ckpt)
else:
    m.head = m.head0
from strands_decider.prompting import render_state


def h(x): return hashlib.sha1(x.encode()).hexdigest()


def items_all(suites=None):
    byitem = collections.OrderedDict()
    for suite, iid, qn, st, spec in EK.all_question_items(suites):
        e = byitem.setdefault(iid, dict(state=st, qs=collections.OrderedDict(), suite=suite))
        e['qs'][qn] = spec
    return list(byitem.items())


def items_sweep():
    want = set()
    want |= {json.loads(l)['id'] for l in open(os.path.expanduser('~/work/evalkit/suites/REAL-label.jsonl'))}
    lo = sorted(EK.load_suite('LONG'), key=lambda it: h(it['id']))[:40]; want |= {it['id'] for it in lo}
    want |= {it['id'] for it in EK.load_suite('JB-hard')}
    for s in ('CF', 'CF-probe'):
        ps = sorted(EK._pairs(s), key=lambda p: h(p['pair']))[:120]; want |= {x for p in ps for x in (p['a'], p['b'])}
    return [(iid, e) for iid, e in items_all() if iid in want]


items = items_sweep() if a.subset == 'sweep' else items_all(a.suites.split(',') if a.suites else None)
import random
random.Random(5).shuffle(items)          # interleave suites, so a partial run is a representative sample
if a.limit: items = items[:a.limit]
si, sn = map(int, a.shard.split('/')); items = items[si::sn]
done = {nm: {} for nm, _ in CFGS}; side = {}
for nm, _ in CFGS:
    p = f'{OUT}/{nm}.json.part.jsonl'
    if os.path.exists(p):
        for l in open(p):
            try: r = json.loads(l); done[nm][r['id']] = r['p']
            except Exception: pass
    side[nm] = open(p, 'a')
stats = {nm: collections.Counter() for nm, _ in CFGS}
print('items', len(items), 'configs', [nm for nm, _ in CFGS], 'done', {k: len(v) for k, v in done.items()}, flush=True)
t0 = time.time(); n = 0; mism = 0
with torch.inference_mode():
    for iid, e in items:
        todo = [(nm, c) for nm, c in CFGS if iid not in done[nm]]
        if not todo: continue
        st = render_state(e['state']); names = list(e['qs'])
        prs = [m.prep_q(st, e['qs'][qn]) for qn in names]
        s = prs[0]['s']
        segc = {}
        for nm, cfg in todo:
            if cfg.gran not in segc: segc[cfg.gran] = m.segs_for(st, s, cfg.gran)
            ts = segc[cfg.gran]
            if ts is None: mism += 1
            it = m.build(s, ts, prs, cfg)
            hh = m.fwd_m2(it)
            lg = m.logits_m2(it, hh)
            res = {}
            for qn, p, l in zip(names, prs, lg):
                pr_ = torch.softmax(l.float(), -1).tolist()
                res[qn] = {lab: pr_[k] for k, lab in enumerate(p['rq'].slot_labels)}
            done[nm][iid] = res
            side[nm].write(json.dumps(dict(id=iid, p=res)) + '\n')
            c = stats[nm]; c['items'] += 1; c['state_rows'] += it.Tm; c['segments'] += it.nseg; c['iso_rows'] += len(it.iso_idx)
        n += 1
        if n % 50 == 0:
            for f in side.values(): f.flush()
            print(n, len(items), f'{time.time() - t0:.0f}s', 'mism', mism, flush=True)
for nm, _ in CFGS:
    side[nm].close()
    allp = {}
    for l in open(f'{OUT}/{nm}.json.part.jsonl'):        # union over shards (they share the side file)
        try: r = json.loads(l); allp[r['id']] = r['p']
        except Exception: pass
    json.dump(allp, open(f'{OUT}/{nm}.json', 'w'))
    json.dump(dict(stats[nm], spec=dict(CFGS)[nm].spec, seg_mismatch=mism), open(f'{OUT}/{nm}.stats.json', 'w'))
print('done', n, f'{time.time() - t0:.0f}s', 'mism', mism, flush=True)
