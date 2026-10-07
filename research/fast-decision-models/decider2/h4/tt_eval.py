"""H4: run this-that-model-1.0 on every evalkit question with its own prompt layout and its own JevBench adapter.

  python tt_eval.py VARIANT [--model M] [--limit N] [--shard i/n]
VARIANT:
  sf      state_first, one question per sequence, state untruncated            (primary; matches hobson's one-question-per-sequence refs)
  schema  schema_first, one question per sequence, state untruncated           (the cacheable layout)
  ship    state_first, one question per sequence, the shipped default max_state_tokens=1536 (keeps the HEAD of the state)
  multi   state_first, all of an item's questions in ONE pass (this-that's native multi-question request)
  mschema schema_first, all of an item's questions in ONE pass (the cacheable multi-question layout timed in tt_lat.py)
Output: ~/work/h4/preds/<model-tag>_<VARIANT>.jsonl, one line per (id, q) [multi: per item]; resumable.
The question adapter is this-that's own thisthat.systemone_protocol (noul -> no/yes with a legend, choice -> criteria keys,
score -> '0'..'n-1'). Labels are mapped back to evalkit's: noul 'true'=P(yes), 'false'=P(no).
"""
import os, sys, json, time, argparse
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
import numpy as np, torch
import evalkit as EK
from thisthat import TypedDecider
from thisthat.prompt import build
from thisthat.systemone_protocol import _typed
from thisthat.model import _softmax
from thisthat.systemone_protocol import TypedQuestion, _legend
from thisthat import Question


def rev_typed(k, s):
    """the same question with the option order (labels AND legend lines) reversed: a pure position/letter-bias probe"""
    t = _typed(k, s)
    crit = s.get('criteria')
    if t.type == 'noul':
        crit = crit if isinstance(crit, dict) else {}
        lines = [('yes', crit.get('true')), ('no', crit.get('false'))]
        text = _legend(lines, s['instructions'] + '\n\nOptions:') if any(d for _, d in lines) else s['instructions']
    elif t.type == 'choice':
        lines = [(str(k2), v) for k2, v in crit.items()][::-1] if isinstance(crit, dict) else [(str(x), None) for x in crit][::-1]
        text = _legend(lines, s['instructions'] + '\n\nOptions:')
    else:
        lines = [(str(i), str(d)) for i, d in enumerate(crit)][::-1]
        text = _legend(lines, s['instructions'] + '\n\nLevels:')
    labels = tuple(l for l, _ in lines)
    return TypedQuestion(k, t.type, labels, Question(text, labels))

ap = argparse.ArgumentParser()
ap.add_argument('variant'); ap.add_argument('--model', default='flock-io/this-that-model-1.0')
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--shard', default='0/1')
ap.add_argument('--suites', default='JB-all,REAL-agree,LONG,CF,CF-probe')
ap.add_argument('--temperature', type=float, default=1.0)
a = ap.parse_args()
V = a.variant
layout = 'schema_first' if V in ('schema', 'mschema') else 'state_first'
maxst = 1536 if V == 'ship' else 10 ** 7
tag = a.model.split('/')[-1]
os.makedirs(os.path.expanduser('~/work/h4/preds'), exist_ok=True)
out = os.path.expanduser(f'~/work/h4/preds/{tag}_{V}.jsonl')

dec = TypedDecider.from_pretrained(a.model, device='cuda')
tok = dec.tokenizer; be = dec.backend


def to_kit(tq, probs):
    p = [float(x) for x in probs]
    if tq.type == 'noul':
        d = dict(zip(tq.labels, p)); return {'true': d['yes'], 'false': d['no']}
    return dict(zip(tq.labels, p))


@torch.inference_mode()
def run(state, specs):
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)   # what thisthat.systemone_protocol.parse_request does
    specs = {k: (dict(s, instructions=s['instructions'] if isinstance(s['instructions'], str) else json.dumps(s['instructions'], ensure_ascii=False))) for k, s in specs.items()}
    tqs = [(rev_typed if V == 'rev' else _typed)(k, s) for k, s in specs.items()]
    b = build(tok, state, [t.question for t in tqs], layout=layout, max_state_tokens=maxst)
    ids = np.array([b['ids']], dtype=np.int64); attn = np.ones_like(ids)
    lg = be.slot_logits(ids, attn, np.array(b['slots']), np.zeros(len(tqs), dtype=np.int64), np.array(b['n_options']))
    pr = _softmax(lg, a.temperature)
    res = {t.key: to_kit(t, pr[k][:len(t.labels)]) for k, t in enumerate(tqs)}
    return res, len(b['ids'])


done = set()
if os.path.exists(out):
    good = []
    for l in open(out):
        try: r = json.loads(l); done.add((r['id'], r.get('q'))); good.append(l)
        except Exception: pass
    open(out, 'w').writelines(good)
si, sn = map(int, a.shard.split('/'))
work = list(EK.all_question_items(a.suites.split(',')))
if V in ('multi', 'mschema'):  # one pass per item with all its questions (dedup by item)
    items = {}
    for s, iid, q, st, spec in work:
        items.setdefault(iid, (s, st, {}))[2][q] = spec
    work = [(s, iid, None, st, sp) for iid, (s, st, sp) in items.items()]
work = work[si::sn]
if a.limit: work = work[:a.limit]
t0 = time.time(); n = 0
with open(out, 'a') as f:
    for s, iid, q, st, spec in work:
        if (iid, q) in done: continue
        t1 = time.time()
        specs = spec if V in ('multi', 'mschema') else {q: spec}
        try:
            res, L = run(st, specs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print('OOM', iid, q, flush=True); continue
        f.write(json.dumps({'suite': s, 'id': iid, 'q': q, 'p': res, 'L': L, 'ms': round((time.time() - t1) * 1000, 1)}) + '\n'); f.flush()
        n += 1
        if n % 100 == 0: print(V, n, len(work), '%.0fs' % (time.time() - t0), 'mem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
print('done', V, n, '%.0fs' % (time.time() - t0), flush=True)
