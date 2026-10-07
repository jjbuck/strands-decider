"""every evalkit question through a J4 model: root = the item's state, children = its questions (each sees state + itself, = hobson's shared
prefix), exact evalkit token ids (render_state + render_question, eng._fit at max_length 16384).  Raw student logits (T = 1).
python evalj4.py OUT.json --stack ck1.pt,ck2.pt [--suites CF,CF-probe] [--hobson]"""
import os, sys, json, time, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/j4'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from j4lib import J4
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--stack', default=''); ap.add_argument('--suites', default='')
ap.add_argument('--hobson', type=int, default=0); ap.add_argument('--mtok', type=int, default=16384)
a = ap.parse_args()
m = J4('hobson' if a.hobson else 'base')
if a.hobson: m.head = m.hob_head
else:
    m.load_stack([s for s in a.stack.split(',') if s])
    if m.lora is not None: m.merge_lora()   # eval with every LoRA folded into bf16 weights (as deployed / as merged_full)
m.head.eval()
m.eng.model.config.max_length = 16384
from strands_decider.prompting import render_question, render_state
suites = a.suites.split(',') if a.suites else None
byitem = collections.OrderedDict()
for suite, iid, qn, st, spec in EK.all_question_items(suites):
    e = byitem.setdefault(iid, dict(state=st, qs=collections.OrderedDict()))
    e['qs'][qn] = spec
items = list(byitem.items())
print('items', len(items), 'questions', sum(len(e['qs']) for _, e in items), flush=True)
os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.out))), exist_ok=True)
side = os.path.expanduser(a.out) + '.part.jsonl'
done = {}
if os.path.exists(side):
    for l in open(side):
        try: r = json.loads(l); done[r['id']] = r['p']
        except Exception: pass
todo = [(iid, e) for iid, e in items if iid not in done]
t0 = time.time(); n = 0
with open(side, 'a') as f, torch.no_grad():
    j = 0
    while j < len(todo):
        mb = []; cur = 0
        while j < len(todo):
            iid, e = todo[j]
            names = list(e['qs'])
            prs = [m.prep(e['state'], e['qs'][qn]) for qn in names]
            L = len(prs[0]['s']) + sum(len(p['q']) for p in prs)
            if mb and cur + L > a.mtok: break
            mb.append((iid, names, prs)); cur += L; j += 1
        roots = [prs[0]['s'] for _, _, prs in mb]
        ch = [(k, p['q']) for k, (_, _, prs) in enumerate(mb) for p in prs]
        h, rsp, csp = m.forward(roots, ch)
        ci = 0
        for iid, names, prs in mb:
            res = {}
            for qn, p in zip(names, prs):
                c0, c1 = csp[ci]; ci += 1
                lg = m.pointer_logits(h, c1 - 1, [c0 + o for o in p['opt']])[:p['rq'].n_slots]
                if a.hobson: lg = lg / m.temp(p['rq'].kind)
                pr = torch.softmax(lg.float(), -1).tolist()
                res[qn] = {lab: pr[i] for i, lab in enumerate(p['rq'].slot_labels)}
            done[iid] = res
            f.write(json.dumps(dict(id=iid, p=res)) + '\n')
        n += len(mb)
        if n % 200 < len(mb): f.flush(); print(n, len(todo), f'{time.time() - t0:.0f}s', flush=True)
json.dump(done, open(os.path.expanduser(a.out), 'w'))
print('done', len(done), f'{time.time() - t0:.0f}s', flush=True)
