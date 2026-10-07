"""H7 eval: every evalkit question (all suites, 3227) through the h7lib forward.
  --layout bundle : schema-first, bundle = the item's questions in item order, one '<answer>' slot per question (per-question mask)
  --layout q1     : schema-first, one question per sequence (H6's layout)
  --layout sets   : schema-first, bundle = item's questions; per-question slot SET = [option-end tokens, '<answer>'] after the state (h7lib.sets_*)
  --layout sets1  : sets layout, one question per sequence
  --layout sf     : hobson state-first (student weights), one sequence per question
--ckpt '' = untrained hobson (merged LoRA, hobson's head). Writes preds JSON {iid: {q: {label: p}}} (resumable via a .jsonl side file).
python h7eval.py OUT.json --ckpt ck/s300.pt --layout bundle"""
import os, sys, json, time, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/h7'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from h7lib import H7
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ckpt', default=''); ap.add_argument('--layout', default='bundle')
ap.add_argument('--suites', default=''); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
m = H7(); m.free_hf(); eng = m.p.eng
if a.ckpt:
    m.load_trainable(os.path.expanduser(a.ckpt))
else:
    m.head = m.head0
suites = a.suites.split(',') if a.suites else None
byitem = collections.OrderedDict()
for suite, iid, qn, st, spec in EK.all_question_items(suites):
    e = byitem.setdefault(iid, dict(state=st, qs=collections.OrderedDict()))
    e['qs'][qn] = spec
items = list(byitem.items())
if a.limit: items = items[:a.limit]
side = os.path.expanduser(a.out) + '.part.jsonl'
done = {}
if os.path.exists(side):
    for l in open(side):
        try: r = json.loads(l); done[r['id']] = r['p']
        except Exception: pass


def prep(state_text, qd):
    q = ta.validate_python(qd); rq = render_question(q)
    s, qs = eng._fit(state_text, [rq.text])
    return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq)


t0 = time.time(); n = 0
with open(side, 'a') as f, torch.inference_mode():
    for iid, e in items:
        if iid in done: continue
        st = render_state(e['state']); names = list(e['qs'])
        prs = [prep(st, e['qs'][qn]) for qn in names]
        s = prs[0]['s']
        if a.layout == 'bundle':
            lgs = m.schema_logits(s, [p['q'] for p in prs], prs)
        elif a.layout == 'sets':
            lgs = m.sets_logits(s, [p['q'] for p in prs], prs)
        elif a.layout == 'sets1':
            lgs = [m.sets_logits(p['s'], [p['q']], [p])[0] for p in prs]
        elif a.layout == 'q1':
            lgs = [m.schema_logits(p['s'], [p['q']], [p])[0] for p in prs]
        else:
            lgs = m.statefirst_logits(s, [p['q'] for p in prs], prs)
        res = {}
        for qn, p, lg in zip(names, prs, lgs):
            pr_ = torch.softmax(lg.float(), -1).tolist()
            res[qn] = {lab: pr_[i] for i, lab in enumerate(p['rq'].slot_labels)}
        done[iid] = res
        f.write(json.dumps(dict(id=iid, p=res)) + '\n'); n += 1
        if n % 100 == 0: f.flush(); print(n, len(items), f'{time.time() - t0:.0f}s', flush=True)
json.dump(done, open(os.path.expanduser(a.out), 'w'))
print('done', len(done), f'{time.time() - t0:.0f}s', flush=True)
