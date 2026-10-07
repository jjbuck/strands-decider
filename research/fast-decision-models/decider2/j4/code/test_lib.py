"""correctness: forest forward (roots varlen + children branches) == plain per-sequence forward; hobson weights reproduce evalkit refs"""
import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/j4')]
import torch
from j4lib import J4
import evalkit as EK
m = J4('hobson'); m.head = m.hob_head
items = [x for x in EK.all_question_items(['CF'])][:6]
refs = {}
for l in open(os.path.expanduser('~/work/evalkit/refs/CF.hobson.jsonl')):
    r = json.loads(l); refs[r['id']] = r
print('ref keys', list(refs[next(iter(refs))].keys())[:5])
with torch.no_grad():
    prs = [m.prep(st, spec) for (_, iid, q, st, spec) in items]
    # plain: one root per [state + question]
    roots = [p['s'] + p['q'] for p in prs]
    h, rsp, _ = m.forward(roots)
    plain = []
    for p, (r0, r1) in zip(prs, rsp):
        lg = m.pointer_logits(h, r1 - 1, [r0 + len(p['s']) + o for o in p['opt']]) / m.temp(p['rq'].kind)
        plain.append(torch.softmax(lg[:p['rq'].n_slots], -1))
    # forest: root = state, child = question
    roots2 = [p['s'] for p in prs]; ch = [(j, p['q']) for j, p in enumerate(prs)]
    h2, rsp2, csp = m.forward(roots2, ch)
    for j, (p, (c0, c1)) in enumerate(zip(prs, csp)):
        lg = m.pointer_logits(h2, c1 - 1, [c0 + o for o in p['opt']]) / m.temp(p['rq'].kind)
        pf = torch.softmax(lg[:p['rq'].n_slots], -1)
        iid, q = items[j][1], items[j][2]
        ref = refs.get(iid, {})
        print(iid, q, 'T', len(p['s']) + len(p['q']), 'plain', [round(x, 4) for x in plain[j].tolist()], 'forest', [round(x, 4) for x in pf.tolist()],
              'labels', p['rq'].slot_labels, 'ref', json.dumps(ref)[:200])
# timing: training-style fwd+bwd with LoRA on a packed 8k-token root batch
m.add_lora(16, 32); m.set_head()
ids = [list(range(1000, 1000 + 400))] * 20
torch.cuda.synchronize(); 
for rep in range(3):
    t0 = time.time()
    h, rsp, _ = m.forward(ids, ckpt=True)
    loss = h.float().pow(2).mean()
    loss.backward()
    torch.cuda.synchronize(); print('fwd+bwd 8000 tok', round(time.time() - t0, 2), 's', 'mem', round(torch.cuda.max_memory_allocated() / 1e9, 1))
