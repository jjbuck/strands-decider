"""J3 eval: evalkit questions through the DT forward (dtlib). Writes preds JSON {iid: {q: {label: p}}} per config.
  python dt_eval.py check                         : Ls=24 == hobson (teacher path) on 20 items; argmax vs evalkit hobson refs
  python dt_eval.py sweep OUTDIR --configs 4A,8A,12A,16A,4G,8G,12G,24A [--per N]
        training-free (hobson weights + hobson head): ONE shallow pass per item, the deep stack run from each split. Subset: N items per suite
        (REAL-agree, LONG/3, JB-hard all), N pairs of CF and CF-probe.
  python dt_eval.py eval OUT.json --ckpt CK --Ls 8 --bridge A [--suites ...]   : every evalkit question (resumable side file)"""
import os, sys, json, time, argparse, collections, hashlib
sys.path[:0] = [os.path.expanduser('~/work/j3'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from dtlib import DT
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('out', nargs='?', default='')
ap.add_argument('--ckpt', default=''); ap.add_argument('--Ls', type=int, default=8); ap.add_argument('--bridge', default='A')
ap.add_argument('--configs', default='4A,8A,12A,16A,4G,8G,12G,24A'); ap.add_argument('--per', type=int, default=120)
ap.add_argument('--suites', default=''); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--layout', default='seqs'); ap.add_argument('--subset', type=int, default=0)
a = ap.parse_args()
m = DT(); m.free_hf(); eng = m.p.eng


def prep(state_text, qd):
    q = ta.validate_python(qd); rq = render_question(q)
    s, qs = eng._fit(state_text, [rq.text])
    return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd)


def h(x): return hashlib.sha1(x.encode()).hexdigest()


def items_all(suites=None):
    byitem = collections.OrderedDict()
    for suite, iid, qn, st, spec in EK.all_question_items(suites):
        e = byitem.setdefault(iid, dict(state=st, qs=collections.OrderedDict()))
        e['qs'][qn] = spec
    return list(byitem.items())


def items_subset(per):
    want = set()
    ra = sorted(EK.load_suite('REAL-agree'), key=lambda it: h(it['id']))[:per]; want |= {it['id'] for it in ra}
    lo = sorted(EK.load_suite('LONG'), key=lambda it: h(it['id']))[:max(10, per // 3)]; want |= {it['id'] for it in lo}
    want |= {it['id'] for it in EK.load_suite('JB-hard')}
    for s in ('CF', 'CF-probe'):
        ps = sorted(EK._pairs(s), key=lambda p: h(p['pair']))[:per]; want |= {x for p in ps for x in (p['a'], p['b'])}
    return [(iid, e) for iid, e in items_all() if iid in want]


def dist(p, lg):
    pr_ = torch.softmax(lg.float(), -1).tolist()
    return {lab: pr_[i] for i, lab in enumerate(p['rq'].slot_labels)}


if a.mode == 'check':
    m.head = m.head0
    its = items_all(['REAL-agree'])[:12] + items_all(['JB-hard'])[:8]
    refs = {}
    for s in ('REAL-agree', 'JB-all'):
        for iid, r in EK.load_refs(s).items(): refs[iid] = r.get('hobson', {})
    agree = n = 0; maxd = 0.0; arg_ref = 0
    with torch.inference_mode():
        for iid, e in its:
            st = render_state(e['state']); names = list(e['qs']); prs = [prep(st, e['qs'][q]) for q in names]; s = prs[0]['s']
            lt = m.teacher_logits(s, [p['q'] for p in prs], prs)
            ld = m.dt_logits(s, [p['q'] for p in prs], prs, 24)
            ls8 = m.dt_logits(s, [p['q'] for p in prs], prs, 8, 'A')
            for qn, p, x, y, z in zip(names, prs, lt, ld, ls8):
                px, py = torch.softmax(x.float(), -1), torch.softmax(y.float(), -1)
                maxd = max(maxd, float((px - py).abs().max())); n += 1; agree += int(px.argmax() == py.argmax())
                rd = refs.get(iid, {}).get(qn)
                if rd: arg_ref += int(p['rq'].slot_labels[int(px.argmax())] == max(rd, key=rd.get))
    print(f'check: Ls=24 vs teacher argmax {agree}/{n} max|dp| {maxd:.2e}; teacher vs evalkit hobson refs argmax {arg_ref}/{n}', flush=True)
    sys.exit(0)

if a.mode == 'sweep':
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    cfgs = [(int(c[:-1]), c[-1]) for c in a.configs.split(',')]
    m.head = m.head0
    its = items_subset(a.per)
    print('sweep items', len(its), 'configs', cfgs, flush=True)
    preds = {c: {} for c in cfgs}; t0 = time.time()
    splits = sorted({L for L, _ in cfgs})
    with torch.inference_mode():
        for k, (iid, e) in enumerate(its):
            st = render_state(e['state']); names = list(e['qs']); prs = [prep(st, e['qs'][q]) for q in names]; s = prs[0]['s']
            qs = [p['q'] for p in prs]
            ids, pos, br = m.seqs_inputs(s, qs)
            x = torch.nn.functional.embedding(torch.as_tensor(ids, device=m.dev), m.embed)
            cos, sin = m.cos_sin(torch.as_tensor(pos, device=m.dev, dtype=torch.float32))
            snap = {}
            for i in range(max(splits)):
                x = m.layer_br(i, x, cos, sin, br)
                if i + 1 in splits: snap[i + 1] = x
            if 0 in splits: snap[0] = torch.nn.functional.embedding(torch.as_tensor(ids, device=m.dev), m.embed)
            for (L, b) in cfgs:
                lg = m.dt_logits(s, qs, prs, L, b, x_shallow=(snap[L], cos, sin))
                preds[(L, b)][iid] = {qn: dist(p, l) for qn, p, l in zip(names, prs, lg)}
            if k % 50 == 0: print(k, len(its), f'{time.time() - t0:.0f}s', flush=True)
    for (L, b), pd in preds.items():
        json.dump(pd, open(os.path.expanduser(f'{a.out}/tf_{L}{b}.json'), 'w'))
    print('done', f'{time.time() - t0:.0f}s', flush=True)
    sys.exit(0)

# ---- eval (trained checkpoint or untrained if --ckpt '')
if a.ckpt:
    meta = m.dt_load(a.ckpt); print('loaded', a.ckpt, {k: v for k, v in meta.items()}, flush=True)
else:
    m.head = m.head0
items = items_subset(a.subset) if a.subset else items_all(a.suites.split(',') if a.suites else None)
if a.limit: items = items[:a.limit]
side = os.path.expanduser(a.out) + '.part.jsonl'
done = {}
if os.path.exists(side):
    for l in open(side):
        try: r = json.loads(l); done[r['id']] = r['p']
        except Exception: pass
t0 = time.time(); n = 0
with open(side, 'a') as f, torch.inference_mode():
    for iid, e in items:
        if iid in done: continue
        st = render_state(e['state']); names = list(e['qs']); prs = [prep(st, e['qs'][qn]) for qn in names]
        if a.layout == 'set':
            import dtset
            lgs = dtset.set_logits(m, prs[0]['s'], prs, a.Ls, a.bridge)
        else:
            lgs = m.dt_logits(prs[0]['s'], [p['q'] for p in prs], prs, a.Ls, a.bridge)
        res = {qn: dist(p, lg) for qn, p, lg in zip(names, prs, lgs)}
        done[iid] = res
        f.write(json.dumps(dict(id=iid, p=res)) + '\n'); n += 1
        if n % 100 == 0: f.flush(); print(n, len(items), f'{time.time() - t0:.0f}s', flush=True)
json.dump(done, open(os.path.expanduser(a.out), 'w'))
print('done', len(done), f'{time.time() - t0:.0f}s', flush=True)
