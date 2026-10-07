"""Deployed question table: every question spec of the train pool (38 names, one spec text each), rendered exactly as hobson renders it."""
import os, sys, json
sys.path[:0] = [os.path.expanduser('~/work/j6'), os.path.expanduser('~/work/evalkit')]
import torch
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
KIT = os.path.expanduser('~/work/evalkit')


def load_pool(min_tok=16):
    EV = set(json.load(open(f'{KIT}/split.json'))['eval_tasks'])
    pool = []
    with open(f'{KIT}/train_pool.jsonl') as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV or r['n_state_tok'] < min_tok: continue
            pool.append(r)
    return pool


def deployed_specs(pool):
    spec = {}
    for r in pool:
        for q, s in r['questions'].items():
            js = json.dumps(s, sort_keys=True)
            if q in spec: assert spec[q][1] == js, q
            else: spec[q] = (s, js)
    return {q: v[0] for q, v in sorted(spec.items())}


def prep_q(eng, spec, perm=None):
    """-> dict(q=token ids (ends with '<answer>'), opt=option-end indices into q, rq=RenderedQuestion, kind)"""
    q = ta.validate_python(spec)
    rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
    s, qs = eng._fit('S', [rq.text])
    opt = eng._option_idx([rq], 0)[0].tolist()
    return dict(q=qs[0], opt=opt, rq=rq, kind=rq.kind, K=len(opt))


def state_ids(eng, state):
    s, _ = eng._fit(render_state(state), ['x'])
    return s


def slot_init(m, eng, pq):
    """initial slot-input vectors for a question: option k = mean embedding of its option line (rescaled to a token-embedding norm),
    answer row = the '<answer>' token embedding. -> [K+1, 2048] fp32"""
    E = m.embed.float()
    rows = []
    starts = [0] + [o + 1 for o in pq['opt'][:-1]]
    # option k's line = tokens (prev option end, this option end]; the first option starts after '<options>\n' -> take at most 24 tokens back
    for k, o in enumerate(pq['opt']):
        a = max(starts[k], o - 23) if k > 0 else max(0, o - 23)
        e = E[torch.tensor(pq['q'][a:o + 1], device=E.device)]
        v = e.mean(0); v = v / v.norm() * e.norm(dim=-1).mean()
        rows.append(v)
    rows.append(E[pq['q'][-1]])
    return torch.stack(rows, 0)
