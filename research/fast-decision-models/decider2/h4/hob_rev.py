"""H4: hobson-v19 (deployed engine code path, as kitrun refs) with every question's option order reversed: the same
position/label-bias probe as this-that's 'rev' variant. -> ~/work/h4/preds/hob_rev.jsonl"""
import os, sys, json, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.expanduser('~/work/evalkit')); sys.path.insert(0, os.path.expanduser('~/work/tokens'))
import torch
import evalkit as EK
from kitrun import load_engine, probdict, ta
from strands_decider.prompting import render_question, render_state

out = os.path.expanduser('~/work/h4/preds/hob_rev.jsonl')
eng = load_engine()
done = set()
if os.path.exists(out):
    for l in open(out):
        try: done.add(json.loads(l)['id'])
        except Exception: pass


def nopt(spec):
    if spec['type'] == 'noul': return 2
    return len(spec['criteria'])


@torch.inference_mode()
def ev(state, questions):
    names = list(questions)
    rendered = [render_question(ta.validate_python(questions[n]), option_order=list(range(nopt(questions[n])))[::-1]) for n in names]
    st = render_state(state); res = {}
    for start in range(0, len(names), eng.cfg.max_batch):
        ch = rendered[start:start + eng.cfg.max_batch]
        slots = [rq.n_slots for rq in ch]; kinds = [rq.kind for rq in ch]
        if eng.cfg.use_prefix_cache and len(ch) > 1:
            probs, _ = eng._slot_probs_shared_prefix(st, [rq.text for rq in ch], slots, kinds, rendered=ch)
        else:
            probs, _ = eng._slot_probs_batched(st, [rq.text for rq in ch], slots, kinds, rendered=ch)
        for i, n in enumerate(names[start:start + eng.cfg.max_batch]):
            res[n] = probdict(ch[i], probs[i, :ch[i].n_slots].tolist())
    return res


items = [it for s in ['JB-all', 'REAL-agree', 'CF', 'CF-probe'] for it in EK.load_suite(s)]
t0 = time.time(); n = 0
with open(out, 'a') as f:
    for it in items:
        if it['id'] in done: continue
        f.write(json.dumps({'id': it['id'], 'q': None, 'p': ev(it['state'], it['questions'])}) + '\n'); f.flush(); n += 1
        if n % 100 == 0: print('hob_rev', n, len(items), '%.0fs' % (time.time() - t0), flush=True)
print('done hob_rev', n, flush=True)
