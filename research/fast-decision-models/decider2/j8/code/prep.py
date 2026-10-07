"""prep.py (CPU box): exact token ids for every evalkit question (same _fit/_option_idx path as kitrun.prep_question, max_length 16384)
plus latency inputs (exact-T real train-split states + real question specs). Writes ~/work/j8/ids.jsonl, ~/work/j8/lat_inputs.json"""
import os, sys, json, glob, random
sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
from transformers import AutoTokenizer
from strands_decider.infer import SystemOneEngine, EngineConfig
import evalkit as EK

ta = TypeAdapter(SC.Question)
OUT = os.path.expanduser('~/work/j8'); os.makedirs(OUT, exist_ok=True)
hc = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]


class _Cfg:  # the only model attributes _fit/_option_idx read
    max_length = 16384
    head_type = 'pointer'


class _M:
    config = _Cfg()


eng = SystemOneEngine.__new__(SystemOneEngine)
eng.cfg = EngineConfig(device='cpu'); eng.model = _M(); eng.tok = AutoTokenizer.from_pretrained(hc); eng.device = 'cpu'
if eng.tok.pad_token is None: eng.tok.pad_token = eng.tok.eos_token


def prep(state, spec):
    rq = render_question(ta.validate_python(spec))
    s, qs = eng._fit(render_state(state), [rq.text])
    opt = eng._option_idx([rq], 0)[0].tolist()
    return dict(s=s, q=qs[0], opt=opt, kind=rq.kind, labels=list(rq.slot_labels))


if __name__ == '__main__':
    n = 0
    with open(f'{OUT}/ids.jsonl', 'w') as f:
        for suite, iid, q, st, spec in EK.all_question_items():
            r = prep(st, spec); r.update(suite=suite, id=iid, qn=q)
            f.write(json.dumps(r) + '\n'); n += 1
    print('items', n, flush=True)
    # latency inputs: real train-split states cut to exactly T tokens; real questions from one 4-question REAL-agree request
    pool = [json.loads(l) for _, l in zip(range(400), open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')))]
    rng = random.Random(0)
    longs = []
    for r in pool:
        ids = eng.tok(render_state(r['state']), add_special_tokens=True)['input_ids']
        longs.append(ids)
    longs = [x for x in longs if len(x) >= 600]
    real = [json.loads(l) for l in open(os.path.expanduser('~/work/evalkit/suites/REAL-agree.jsonl'))]
    four = [r for r in real if len(r['questions']) == 4][0]
    qs = []
    for qn, spec in four['questions'].items():
        p = prep('', spec); qs.append(dict(qn=qn, q=p['q'], opt=p['opt'], kind=p['kind']))
    states = {}
    for T in (64, 128, 256, 400, 1000, 4000):
        out = []
        for rep in range(24):
            # concatenate random real states until >= T, then cut exactly T (fresh content each rep)
            buf = []
            while len(buf) < T + rng.randint(0, 500):
                buf += longs[rng.randrange(len(longs))]
            o = rng.randint(0, len(buf) - T)
            out.append(buf[o:o + T])
        states[T] = out
    json.dump(dict(questions=qs, states=states, req=four['id']), open(f'{OUT}/lat_inputs.json', 'w'))
    print('lat inputs ok', [len(q['q']) for q in qs], flush=True)
