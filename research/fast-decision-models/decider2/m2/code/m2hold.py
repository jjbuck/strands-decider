"""M2: accuracy per task on v19's held-out split (training/data/train_v5.holdout.jsonl; train side, no evalkit items), J3's probe
(dt_holdout.py): ruletaker depth 3 / depth 5 / natlang test deduction chained across the state. hobson (teacher path) vs M2 layouts.
python m2hold.py OUT.json --configs 'name=spec|...' [--ckpt CK] [--n 150]"""
import os, sys, json, random, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/m2'), os.path.expanduser('~/work/evalkit')]
import torch
from m2lib import M2, Cfg
from strands_decider.prompting import render_state
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--configs', required=True); ap.add_argument('--ckpt', default='')
ap.add_argument('--n', type=int, default=150); ap.add_argument('--tasks', default='')
a = ap.parse_args()
m = M2(); m.free_hf(); m.setup()
if a.ckpt:
    import m2train_util as TU; TU.load_student(m, a.ckpt)
else: m.head = m.head0
CF = [(c.split('=', 1)[0], Cfg(c.split('=', 1)[1])) for c in a.configs.split('|')]
rows = [json.loads(l) for l in open(os.path.expanduser('~/work/training/data/train_v5.holdout.jsonl'))]
random.Random(11).shuffle(rows)
byt = collections.defaultdict(list)
for r in rows:
    if a.tasks and r['task'] not in a.tasks.split(','): continue
    if len(byt[r['task']]) < a.n: byt[r['task']].append(r)


def v5_q(r):
    ins = r['instructions']
    if r['kind'] in ('choice', 'noul'): return {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


res = {}
with torch.inference_mode():
    for task, rs in sorted(byt.items()):
        c = collections.Counter(); segn = 0
        for r in rs:
            qd, gold = v5_q(r); st = render_state(r['state'])
            p = m.prep_q(st, qd); gi = p['rq'].slot_labels.index(gold); s = p['s']
            sv = (m.lora, m.head); m.lora = None; m.head = m.head0
            lt = m.statefirst_logits(s, [p['q']], [p])[0]; m.lora, m.head = sv
            c['hobson'] += int(lt.argmax()) == gi
            segc = {}
            for nm, cfg in CF:
                if cfg.gran not in segc: segc[cfg.gran] = m.segs_for(st, s, cfg.gran)
                it = m.build(s, segc[cfg.gran], [p], cfg)
                lg = m.logits_m2(it, m.fwd_m2(it))[0]
                c[nm] += int(lg.argmax()) == gi
                if nm == CF[0][0]: segn += it.nseg
        res[task] = {k: v / len(rs) for k, v in c.items()}; res[task]['n'] = len(rs); res[task]['mean_segments'] = segn / len(rs)
        print(task, {k: round(v, 3) for k, v in res[task].items()}, flush=True)
json.dump(res, open(os.path.expanduser(a.out), 'w'), indent=1)
