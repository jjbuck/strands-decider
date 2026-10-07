"""J9 eval: every evalkit question (3227) through a compiled-constant layout.
  --layout native : hobson's own layout through the same library (sanity / noise floor)
  --layout Rx     : constants-first ORDER, but computed exactly (full recompute in the new order; isolates the reorder)
  --layout R      : constants-first, blocks compiled independently (U context), composed at runtime (--comp affine|last|skip)
  --layout S      : native order, blocks compiled independently and spliced in place (--comp ...)
--ckpt '' = untrained hobson. Writes preds JSON {iid: {q: {label: p}}} (+ .part.jsonl, resumable) and stats (live tokens etc.).
python j9eval.py OUT.json --layout R --comp affine [--ckpt ck/s600.pt] [--suites REAL-agree,LONG]"""
import os, sys, json, time, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/j9'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
import j9lib as J
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ckpt', default=''); ap.add_argument('--layout', default='R')
ap.add_argument('--comp', default='affine'); ap.add_argument('--suites', default=''); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--minb', type=int, default=32); ap.add_argument('--shard', default='0/1'); ap.add_argument('--stride', type=int, default=1)
ap.add_argument('--adapter', default='all')
a = ap.parse_args()
m = J.J9(); m.free_hf(); eng = m.p.eng; tok = eng.tok
m.setup_u(tok)
if a.ckpt: m.load_trainable(os.path.expanduser(a.ckpt))
else: m.head = m.head0
m.compile_only = (a.adapter == 'compile')
LL = J.LineLib(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))
suites = a.suites.split(',') if a.suites else None
byitem = collections.OrderedDict()
for suite, iid, qn, st, spec in EK.all_question_items(suites):
    e = byitem.setdefault(iid, dict(state=st, qs=collections.OrderedDict(), suite=suite))
    e['qs'][qn] = spec
items = list(byitem.items())
if a.limit: items = items[:a.limit]
items = items[::a.stride]
si, sn = map(int, a.shard.split('/')); items = items[si::sn]
side = os.path.expanduser(a.out) + '.part.jsonl'
done = {}
if os.path.exists(side):
    for l in open(side):
        try: r = json.loads(l); done[r['id']] = r
        except Exception: pass


def prep(state_text, qd):
    q = ta.validate_python(qd); rq = render_question(q)
    s, qs = eng._fit(state_text, [rq.text])
    return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq)


t0 = time.time(); n = 0; mism = 0
with open(side, 'a') as f, torch.inference_mode():
    for iid, e in items:
        if iid in done: continue
        st = render_state(e['state']); names = list(e['qs'])
        req = J.tokenize_pieces(tok, J.pieces(st, LL), minb=a.minb)
        cat = [t for _, ids, _ in req for t in ids]
        res = {}; stats = {}
        for qn in names:
            p = prep(st, e['qs'][qn])
            if cat != p['s']:
                mism += 1; lay = 'native'
            else:
                lay = a.layout
            if lay == 'native':
                lg = m.native_logits(p['s'], p['q'], p['opt'], p['rq'].kind)
                live = len(p['s']) + len(p['q'])
            elif lay == 'Rx':
                fr = [t for k, ids, _ in req if k == 'frame' for t in ids]
                seen = set(); bl = []
                for k, ids, key in req:
                    if k == 'blk' and key not in seen: seen.add(key); bl += ids
                dy = [t for k, ids, _ in req if k == 'dyn' for t in ids]
                lg = m.native_logits(fr + bl + dy, p['q'], p['opt'], p['rq'].kind)
                live = len(p['s']) + len(p['q'])
            else:
                P = J.build_plan(m, req, p['q'], lay, m.dev)
                h = m.fwd_c(P, comp=a.comp)
                lg = m.logits_c(P, h, p['opt'], p['rq'].kind)
                live = P.n_live
            pr_ = torch.softmax(lg[:p['rq'].n_slots].float(), -1).tolist()
            res[qn] = {lab: pr_[i] for i, lab in enumerate(p['rq'].slot_labels)}
            stats[qn] = dict(live=live, total=len(p['s']) + len(p['q']), state=len(p['s']), nq=len(p['q']))
        rec = dict(id=iid, p=res, st=stats)
        done[iid] = rec
        f.write(json.dumps(rec) + '\n'); n += 1
        if n % 100 == 0: f.flush(); print(n, len(items), f'{time.time() - t0:.0f}s', 'mism', mism, flush=True)
json.dump({k: v['p'] for k, v in done.items()}, open(os.path.expanduser(a.out), 'w'))
print('done', len(done), f'{time.time() - t0:.0f}s', 'mism', mism, flush=True)
