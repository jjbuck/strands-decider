"""J2 eval: every evalkit question (3227) through the J2 packed forward, one row per (state, question) exactly as hobson renders it
(render_state + render_question, engine _fit at max_length 16384, no truncation). Writes preds JSON {iid: {q: {label: p}}}.
--ckpt '' -> untrained (hobson head, no LoRA, gates 0).
python j2eval.py OUT.json --mode qag --ckpt ck/qag_all/final.pt"""
import os, sys, json, time, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/j2'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import evalkit as EK
from j2lib import J2, GDN_LAYERS
from j2data import prep_item
from strands_decider.prompting import render_state

ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ckpt', default=''); ap.add_argument('--mode', default='')
ap.add_argument('--suites', default=''); ap.add_argument('--zero_gates', type=int, default=0); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--maxpack', type=int, default=12288)
a = ap.parse_args()
m = J2(); m.free_hf(); eng = m.p.eng
if a.ckpt:
    sd = torch.load(os.path.expanduser(a.ckpt), map_location=m.dev)
    j = sd.get('_j2', {})
    mode = a.mode or j.get('mode', 'causal')
    m.load_trainable(os.path.expanduser(a.ckpt))
    m.setup_bidir(mode, rev_layers=j.get('rev', GDN_LAYERS), attn_nc=j.get('attn_nc', True))
    m.load_gates(sd)
    if a.zero_gates:
        with torch.no_grad():
            for p_ in list(m.gam.values()) + list(m.lam.values()): p_.zero_()
else:
    m.head = m.head0
    m.setup_bidir(a.mode or 'causal')
print('mode', m.mode, 'rev', sorted(m.rev_layers), 'attn_nc', m.attn_nc, flush=True)
suites = a.suites.split(',') if a.suites else None
qs = list(EK.all_question_items(suites))
if a.limit: qs = qs[:a.limit]
side = os.path.expanduser(a.out) + '.part.jsonl'
done = collections.defaultdict(dict)
if os.path.exists(side):
    for l in open(side):
        try:
            r = json.loads(l); done[r['id']][r['q']] = r['p']
        except Exception: pass
todo = [(iid, qn, st, spec) for _, iid, qn, st, spec in qs if qn not in done.get(iid, {})]
print('questions', len(qs), 'todo', len(todo), flush=True)
t0 = time.time()
# prep (CPU) then sort by length and pack
cache = {}
items = []
for iid, qn, st, spec in todo:
    if iid not in cache: cache = {iid: render_state(st)}
    it = prep_item(eng, None, spec, rendered_state=cache[iid]); it['id'] = iid; it['qn'] = qn
    items.append(it)
print('prepped', len(items), f'{time.time() - t0:.0f}s', flush=True)
items.sort(key=lambda it: len(it['s']) + len(it['q']))
n = 0
with open(side, 'a') as f, torch.inference_mode():
    i = 0
    while i < len(items):
        pk = []; tot = 0
        while i < len(items) and (not pk or tot + len(items[i]['s']) + len(items[i]['q']) <= a.maxpack):
            pk.append(items[i]); tot += len(items[i]['s']) + len(items[i]['q']); i += 1
        lgs = m.decide(pk)
        for it, lg in zip(pk, lgs):
            p = torch.softmax(lg.float(), -1).tolist()
            res = {lab: p[j] for j, lab in enumerate(it['labels'])}
            done[it['id']][it['qn']] = res
            f.write(json.dumps(dict(id=it['id'], q=it['qn'], p=res)) + '\n'); n += 1
        f.flush()
        if n % 500 < len(pk): print(n, len(items), f'{time.time() - t0:.0f}s', flush=True)
json.dump(done, open(os.path.expanduser(a.out), 'w'))
print('done', sum(len(v) for v in done.values()), f'{time.time() - t0:.0f}s', flush=True)
