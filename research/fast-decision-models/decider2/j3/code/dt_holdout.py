"""J3: accuracy per task on v19's own held-out split (training/data/train_v5.holdout.jsonl; train-side data, no evalkit items) for hobson
(teacher path, LoRA off) and a DT checkpoint at several splits. Probes which capabilities need deep state processing (e.g. ruletaker depth 3/5).
python dt_holdout.py OUT.json --ckpt CK --splits 8,12 [--layout seqs|set] [--n 150]"""
import os, sys, json, random, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/j3'), os.path.expanduser('~/work/evalkit')]
import torch
from dtlib import DT
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ckpt', default=''); ap.add_argument('--splits', default='8,12')
ap.add_argument('--layout', default='seqs'); ap.add_argument('--n', type=int, default=150); ap.add_argument('--untrained', default='')
a = ap.parse_args()
m = DT(); m.free_hf(); eng = m.p.eng
if a.ckpt: m.dt_load(a.ckpt)
else: m.head = m.head0
if a.layout == 'set': import dtset
rows = [json.loads(l) for l in open(os.path.expanduser('~/work/training/data/train_v5.holdout.jsonl'))]
random.Random(11).shuffle(rows)
byt = collections.defaultdict(list)
for r in rows:
    if len(byt[r['task']]) < a.n: byt[r['task']].append(r)


def v5_q(r):
    ins = r['instructions']
    if r['kind'] in ('choice', 'noul'): return {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


SPL = [int(x) for x in a.splits.split(',')]
res = {}
with torch.inference_mode():
    for task, rs in sorted(byt.items()):
        c = collections.Counter()
        for r in rs:
            qd, gold = v5_q(r); rq = render_question(ta.validate_python(qd))
            s, qs = eng._fit(render_state(r['state']), [rq.text])
            pr = dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd); gi = rq.slot_labels.index(gold)
            sv = m.lora; m.lora = None; lt = m.teacher_logits(s, [qs[0]], [pr])[0]; m.lora = sv
            c['hobson'] += int(lt.argmax()) == gi
            for L in SPL:
                lg = (dtset.set_logits(m, s, [pr], L, 'A') if a.layout == 'set' else m.dt_logits(s, [qs[0]], [pr], L, 'A'))[0]
                c[f'L{L}'] += int(lg.argmax()) == gi
            if a.untrained:          # training-free DT (hobson weights, no LoRA / memory adapters) at these splits
                sv = (m.lora, m.mem, m.head); m.lora = None; m.mem = None; m.head = m.head0
                for L in [int(x) for x in a.untrained.split(',')]:
                    c[f'tf{L}'] += int(m.dt_logits(s, [qs[0]], [pr], L, 'A')[0].argmax()) == gi
                m.lora, m.mem, m.head = sv
        res[task] = {k: v / len(rs) for k, v in c.items()}; res[task]['n'] = len(rs)
        print(task, {k: round(v, 3) for k, v in res[task].items()}, flush=True)
json.dump(res, open(os.path.expanduser(a.out), 'w'), indent=1)
