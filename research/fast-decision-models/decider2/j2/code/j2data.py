"""J2 data: rendering/tokenisation exactly as hobson (strands_decider prompting + engine _fit), the F7-recipe training mix, and the
hobson teacher pass (calibrated distributions on the SAME token ids the student sees).

python j2data.py build  -> ~/work/j2/data/ft.jsonl (+ mntp.jsonl)   [needs the GPU for the teacher pass]
"""
import os, sys, json, random, collections, glob, time
sys.path[:0] = [os.path.expanduser('~/work/j2'), os.path.expanduser('~/work/evalkit')]
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
W = os.path.expanduser('~/work')


def prep_item(eng, state, qd, perm=None, maxlen=None, rendered_state=None):
    """-> dict(s, q, opt, rq, kind, n_slots, labels). maxlen: temporarily override the engine window (None = current config)."""
    q = ta.validate_python(qd)
    rq = render_question(q, option_order=perm) if perm is not None else render_question(q)
    st = rendered_state if rendered_state is not None else render_state(state)
    cfg = eng.model.config; old = cfg.max_length
    if maxlen: cfg.max_length = maxlen
    try:
        s, qs = eng._fit(st, [rq.text])
    finally:
        cfg.max_length = old
    opt = eng._option_idx([rq], 0)[0].tolist()
    return dict(s=list(s), q=list(qs[0]), opt=opt, rq=rq, kind=rq.kind, n_slots=rq.n_slots, labels=list(rq.slot_labels))


def v5_q(r):
    ins = r['instructions']
    if r['kind'] == 'choice': return {'type': 'choice', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    if r['kind'] == 'noul': return {'type': 'noul', 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


def nopt(qd):
    return 2 if qd['type'] == 'noul' else len(qd['criteria'])


def build(n_v5=24000, n_real=4500, n_syn=2500, n_mntp=3000, seed=20261006):
    import torch
    from j2lib import J2
    rng = random.Random(seed)
    EV = set(json.load(open(f'{W}/evalkit/split.json'))['eval_tasks'])
    m = J2(); m.free_hf(); m.head = m.head0; eng = m.p.eng
    m.setup_bidir('causal')
    src = []
    v5 = [json.loads(l) for l in open(f'{W}/training/data/train_v5.jsonl')]
    rng.shuffle(v5)
    for r in v5[:n_v5]:
        qd, gold = v5_q(r)
        if r.get('instruction_variants') and rng.random() < 0.3: qd['instructions'] = rng.choice(r['instruction_variants'])
        src.append(('v5', r['state'], qd, gold, r.get('task')))
    syn = []
    for name in ('generated_v16', 'generated_v18', 'adequacy_gen'):
        for p in glob.glob(f'{W}/sd/data/synthetic/{name}.jsonl') + glob.glob(f'{W}/training/data/synthetic/{name}.jsonl'):
            rows = [json.loads(l) for l in open(p)]; rng.shuffle(rows); syn += [(name, r) for r in rows]; break
    rng.shuffle(syn)
    for name, r in syn[:n_syn]:
        try:
            qd, gold = v5_q(r)
        except Exception:
            continue
        src.append(('syn', r['state'], qd, gold, name))
    pool = []
    for l in open(f'{W}/evalkit/train_pool.jsonl'):
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 32: continue
        pool.append(r)
    rng.shuffle(pool)
    for r in pool[:n_real]:
        qn = rng.choice(list(r['questions']))
        src.append(('real', r['state'], r['questions'][qn], None, r['task']))
    print('sources', collections.Counter(s[0] for s in src), 'syn files', len(syn), flush=True)
    rng.shuffle(src)
    os.makedirs(f'{W}/j2/data', exist_ok=True)
    out = open(f'{W}/j2/data/ft.jsonl', 'w')
    t0 = time.time(); buf = []; ntok = 0; nrow = 0; agree = 0; nlab = 0

    def flush():
        nonlocal agree, nlab
        with torch.inference_mode():
            lgs = m.decide(buf)
        for it, lg in zip(buf, lgs):
            tp = torch.softmax(lg.float(), -1).tolist()
            rec = dict(src=it['src'], task=it['task'], s=it['s'], q=it['q'], opt=it['opt'], kind=it['kind'], n_slots=it['n_slots'],
                       labels=it['labels'], gold=it['gold'], tp=[round(x, 6) for x in tp])
            if it['gold'] >= 0:
                nlab += 1; agree += int(max(range(len(tp)), key=lambda j: tp[j]) == it['gold'])
            out.write(json.dumps(rec) + '\n')
        buf.clear()

    for kind, state, qd, gold, task in src:
        try:
            perm = None
            if qd['type'] in ('choice', 'noul') and rng.random() < 0.5:
                perm = list(range(nopt(qd))); rng.shuffle(perm)
            it = prep_item(eng, state, qd, perm=perm, maxlen=4096 if kind != 'real' else 6144)
        except Exception as e:
            continue
        it['src'] = kind; it['task'] = task
        it['gold'] = it['labels'].index(gold) if gold is not None and gold in it['labels'] else -1
        if kind != 'real' and it['gold'] < 0: continue
        L = len(it['s']) + len(it['q'])
        if buf and (ntok + L > 16384): flush(); ntok = 0
        buf.append(it); ntok += L; nrow += 1
        if nrow % 2000 == 0: print(nrow, f'{time.time() - t0:.0f}s', flush=True)
    if buf: flush()
    out.close()
    print('ft rows', nrow, 'teacher argmax==gold', agree / max(1, nlab), f'{time.time() - t0:.0f}s', flush=True)
    # MNTP states: real train-split states (unused by the FT real rows where possible), rendered, cut to 2048 tokens
    tok = eng.tok
    with open(f'{W}/j2/data/mntp.jsonl', 'w') as f:
        k = 0
        for r in pool[n_real:] + pool[:n_real]:
            ids = tok(render_state(r['state']), add_special_tokens=False)['input_ids']
            if len(ids) < 128: continue
            if len(ids) > 2048: ids = ids[-2048:]
            f.write(json.dumps(dict(ids=ids, task=r['task'])) + '\n'); k += 1
            if k >= n_mntp: break
    print('mntp rows', k, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        kw = {k: int(v) for k, v in (a.split('=') for a in sys.argv[2:])}
        build(**kw)
