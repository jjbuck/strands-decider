"""J3 post-hoc calibration (Brooker's recipe: one temperature per question kind, fitted on held-out TRAIN-side data, never on eval items).
Runs a checkpoint on N rows per kind of training/data/train_v5.holdout.jsonl, and fits alpha_kind = T0/T1 (p' ~ p^alpha on the deployed
probabilities, equivalent to a new temperature) by grid search on gold NLL. Also reports hobson's own NLL / ECE on the same rows.
python dt_calib.py OUT.json --ckpt CK --Ls 8 [--layout set] [--n 300]"""
import os, sys, json, random, argparse, collections, math
sys.path[:0] = [os.path.expanduser('~/work/j3'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch
from dtlib import DT
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ckpt', default=''); ap.add_argument('--Ls', type=int, default=8)
ap.add_argument('--bridge', default='A'); ap.add_argument('--layout', default='seqs'); ap.add_argument('--n', type=int, default=300)
a = ap.parse_args()
m = DT(); m.free_hf(); eng = m.p.eng
if a.ckpt: m.dt_load(a.ckpt)
else: m.head = m.head0
if a.layout == 'set': import dtset
rows = [json.loads(l) for l in open(os.path.expanduser('~/work/training/data/train_v5.holdout.jsonl'))]
random.Random(5).shuffle(rows)
byk = collections.defaultdict(list)
for r in rows:
    if len(byk[r['kind']]) < a.n and len(str(r['state'])) < 12000: byk[r['kind']].append(r)


def v5_q(r):
    ins = r['instructions']
    if r['kind'] in ('choice', 'noul'): return {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}, r['options'][r['label']][0]
    return {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}, str(r['label'])


res = {}
with torch.inference_mode():
    for kind, rs in byk.items():
        lp_m = []; lp_h = []
        for r in rs:
            qd, gold = v5_q(r); q = ta.validate_python(qd); rq = render_question(q)
            s, qs = eng._fit(render_state(r['state']), [rq.text])
            pr = dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd); gi = rq.slot_labels.index(gold)
            lg = (dtset.set_logits(m, s, [pr], a.Ls, a.bridge) if a.layout == 'set' else m.dt_logits(s, [qs[0]], [pr], a.Ls, a.bridge))[0]
            if a.ckpt:
                sv = m.lora; m.lora = None; lt = m.teacher_logits(s, [qs[0]], [pr])[0]; m.lora = sv
            else: lt = lg
            lp_m.append((torch.log_softmax(lg.float(), -1).tolist(), gi)); lp_h.append((torch.log_softmax(lt.float(), -1).tolist(), gi))

        def nll(lps, al):
            tot = 0.0
            for lp, gi in lps:
                z = np.array(lp) * al; z = z - z.max(); tot += -(z[gi] - math.log(np.exp(z).sum()))
            return tot / len(lps)
        grid = [round(x, 2) for x in np.arange(0.3, 2.51, 0.05)]
        best = min(grid, key=lambda al: nll(lp_m, al))
        acc = float(np.mean([int(np.argmax(lp)) == gi for lp, gi in lp_m])); acch = float(np.mean([int(np.argmax(lp)) == gi for lp, gi in lp_h]))
        res[kind] = dict(n=len(rs), alpha=best, nll_before=nll(lp_m, 1.0), nll_after=nll(lp_m, best), acc=acc, hob_nll=nll(lp_h, 1.0), hob_acc=acch,
                         hob_alpha=min(grid, key=lambda al: nll(lp_h, al)))
        print(kind, res[kind], flush=True)
json.dump(res, open(os.path.expanduser(a.out), 'w'), indent=1)
