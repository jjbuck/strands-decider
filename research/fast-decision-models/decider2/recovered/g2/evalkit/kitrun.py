"""evalkit box-side runner (GPU box only; never on the laptop).

  python kitrun.py refs   SUITE [--limit N]      -> refs/SUITE.hobson.jsonl   (hobson-v19 exactly as deployed: unmerged PEFT LoRA,
                                                    SystemOneEngine code paths, max_length raised to 16384 so nothing is truncated)
  python kitrun.py base   SUITE [--limit N]      -> refs/SUITE.base.jsonl     (merged-LoRA layer-by-layer runtime (tokens/plib.py):
                                                    its own full pass + nostate + drop-all / random / q-attention selection baselines)
  python kitrun.py fidelity N                     -> refs/fidelity.jsonl       (engine at the DEPLOYED max_length 4096 vs answers stored
                                                    in the gate logs, hobson-local records)

Library use by other agents:  from kitrun import load_items, prep_question, probdict
  prep_question(P, item, qname) gives the exact token ids every reference used (plib.P.prep with max_length 16384).
"""
import os, sys, json, time, hashlib, argparse, glob
sys.path.insert(0, os.path.expanduser('~/work/tokens'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state

KIT = os.path.dirname(os.path.abspath(__file__))
MAXLEN = 16384
ta = TypeAdapter(SC.Question)
CKPT = None


def ckpt():
    return glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]


def load_items(suite):
    return [json.loads(l) for l in open(f'{KIT}/suites/{suite}.jsonl')]


def probdict(rq, probs):
    """canonical per-question distribution: {label: p}. noul labels 'true'/'false'; choice option names; score level indices '0'.."""
    l = [float(x) for x in probs]
    return {lab: l[i] for i, lab in enumerate(rq.slot_labels)}


# ------------------------------------------------------------------ deployed engine, raw probabilities (same code paths as SystemOneEngine.evaluate)
def load_engine(max_length=MAXLEN):
    from strands_decider.infer import load_engine as LE
    eng = LE(ckpt(), device='cuda')
    eng.model.config.max_length = max_length
    return eng


@torch.inference_mode()
def engine_eval(eng, state, questions):
    names = list(questions)
    rendered = [render_question(ta.validate_python(questions[n])) for n in names]
    st = render_state(state)
    out = {}
    for start in range(0, len(names), eng.cfg.max_batch):
        ch = rendered[start:start + eng.cfg.max_batch]
        slots = [rq.n_slots for rq in ch]; kinds = [rq.kind for rq in ch]
        if eng.cfg.use_prefix_cache and len(ch) > 1:
            probs, _ = eng._slot_probs_shared_prefix(st, [rq.text for rq in ch], slots, kinds, rendered=ch)
        else:
            probs, _ = eng._slot_probs_batched(st, [rq.text for rq in ch], slots, kinds, rendered=ch)
        for i, n in enumerate(names[start:start + eng.cfg.max_batch]):
            out[n] = probdict(ch[i], probs[i, :ch[i].n_slots].tolist())
    return out


# ------------------------------------------------------------------ merged layer-by-layer runtime + baselines
def load_P(max_length=MAXLEN):
    from plib import P
    p = P()
    p.model.config.max_length = max_length
    return p


def prep_question(p, item, qname):
    return p.prep(item['state'], item['questions'][qname])


BASE_CFGS = [('drop', k, 0.0) for k in (0, 3, 7)] + [('rand', k, f) for k in (0, 3, 7) for f in (0.10, 0.25, 0.50)] + [('qattn', k, 0.10) for k in (3, 7)]
_FULL = list(BASE_CFGS)
if os.environ.get('BASE_GRID') == 'rest':  # the configs 'lite' leaves out (merged into the same base file afterwards)
    BASE_CFGS = [c for c in _FULL if c not in [('drop', 7, 0.0), ('rand', 7, 0.10), ('rand', 7, 0.25), ('rand', 7, 0.50), ('rand', 3, 0.25), ('rand', 0, 0.25), ('qattn', 7, 0.10)]]
if os.environ.get('BASE_GRID') == 'lite':  # LONG / CF-probe: the essential grid (time budget on box f0)
    BASE_CFGS = [('drop', 7, 0.0), ('rand', 7, 0.10), ('rand', 7, 0.25), ('rand', 7, 0.50), ('rand', 3, 0.25), ('rand', 0, 0.25), ('qattn', 7, 0.10)]


def cfg_name(kind, k, f):
    return f'drop@{k}' if kind == 'drop' else f'{kind}{int(round(f * 100))}@{k}'


def seed_of(*a):
    return int(hashlib.sha1('|'.join(map(str, a)).encode()).hexdigest()[:8], 16)


@torch.inference_mode()
def baselines(p, item, qname):
    pr = prep_question(p, item, qname)
    full = p.base(pr)
    res = {'merged_full': probdict(pr['rq'], full.tolist()), 'n_state': pr['q0'], 'n_q': pr['L'] - pr['q0']}
    cf = {}
    for kind, k, f in BASE_CFGS:
        nm = cfg_name(kind, k, f)
        g = torch.Generator(device='cuda'); g.manual_seed(seed_of(item['id'], qname, nm))
        scorer = 'random' if kind in ('rand', 'drop') else 'qattn'
        pp = p.pruned(pr, [(k, f)], scorer=scorer, sink=4, rng=g)
        cf[nm] = probdict(pr['rq'], pp.tolist())
    del pr
    if os.environ.get('BASE_GRID') != 'rest':
        pr0 = p.prep('', item['questions'][qname])
        cf['nostate'] = probdict(pr0['rq'], p.base(pr0).tolist())
    res['cfg'] = cf
    return res


def _done_ids(out):
    ids = set()
    if os.path.exists(out):
        good = []
        for l in open(out):
            try: ids.add(json.loads(l)['id']); good.append(l)
            except Exception: pass  # a line cut by a kill
        open(out, 'w').writelines(good)
    return ids


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('suite', nargs='?'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default=None); ap.add_argument('--shard', default='0/1')
    a = ap.parse_args()
    os.makedirs(f'{KIT}/refs', exist_ok=True)
    si, sn = map(int, a.shard.split('/'))
    if a.mode == 'refs':
        eng = load_engine()
        its = load_items(a.suite); its = its[:a.limit] if a.limit else its; its = its[si::sn]
        out = a.out or f'{KIT}/refs/{a.suite}.hobson.jsonl'
        done = _done_ids(out)
        t0 = time.time()
        with open(out, 'a') as f:
            for i, it in enumerate(its):
                if it['id'] in done: continue
                t1 = time.time()
                r = engine_eval(eng, it['state'], it['questions'])
                f.write(json.dumps({'id': it['id'], 'hobson': r, 'ms': round((time.time() - t1) * 1000, 1)}) + '\n'); f.flush()
                if i % 25 == 0: print(a.suite, i, len(its), '%.0fs' % (time.time() - t0), flush=True)
    elif a.mode == 'base':
        p = load_P()
        its = load_items(a.suite); its = its[:a.limit] if a.limit else its; its = its[si::sn]
        out = a.out or f'{KIT}/refs/{a.suite}.base.jsonl'
        done = _done_ids(out)
        t0 = time.time()
        with open(out, 'a') as f:
            for i, it in enumerate(its):
                if it['id'] in done: continue
                t1 = time.time()
                rec = {'id': it['id'], 'q': {}}
                for qn in it['questions']:
                    rec['q'][qn] = baselines(p, it, qn)
                rec['ms'] = round((time.time() - t1) * 1000, 1)
                f.write(json.dumps(rec) + '\n'); f.flush()
                if i % 10 == 0: print(a.suite, i, len(its), '%.0fs' % (time.time() - t0), 'mem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9), flush=True)
    elif a.mode == 'fidelity':
        eng = load_engine(max_length=4096)  # exactly as deployed
        recs = [json.loads(l) for l in open(f'{KIT}/suites/fidelity.jsonl')]
        recs = recs[:int(a.suite)] if a.suite else recs
        out = f'{KIT}/refs/fidelity.jsonl'
        with open(out, 'w') as f:
            for i, r in enumerate(recs):
                got = engine_eval(eng, r['state'], r['questions'])
                f.write(json.dumps({'id': r['id'], 'got': got, 'stored': r['stored']}) + '\n'); f.flush()
                if i % 50 == 0: print('fid', i, flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()
