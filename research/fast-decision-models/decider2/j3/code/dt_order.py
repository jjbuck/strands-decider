"""J3 option-order invariance: decisions and probabilities under option rotations (noul / choice: rotations r = 1, 2, 3 mod K, distinct and
non-identity; score: rubric reversal, reported separately). Every variant is its own question branch over the same state (one forward per item).
  python dt_order.py OUT.json [--ckpt CK --Ls 8 --bridge A --layout seqs|set] [--suites REAL-agree,JB-all]
--ckpt '' with --Ls 24 = hobson. Writes per-question records + a summary (unchanged fraction, mean |dp| over labels, mean TV) to OUT.json."""
import os, sys, json, time, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/j3'), os.path.expanduser('~/work/evalkit')]
import numpy as np, torch
import evalkit as EK
from dtlib import DT
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ckpt', default=''); ap.add_argument('--Ls', type=int, default=24)
ap.add_argument('--bridge', default='A'); ap.add_argument('--layout', default='seqs'); ap.add_argument('--suites', default='REAL-agree,JB-all')
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
m = DT(); m.free_hf(); eng = m.p.eng
if a.ckpt: m.dt_load(a.ckpt)
else: m.head = m.head0
if a.layout == 'set':
    import dtset


def prep(state_text, qd, order=None):
    q = ta.validate_python(qd); rq = render_question(q, option_order=order) if order is not None else render_question(q)
    s, qs = eng._fit(state_text, [rq.text])
    return dict(s=s, q=qs[0], opt=eng._option_idx([rq], 0)[0].tolist(), rq=rq, qd=qd, order=order)


def variants(qd):
    K = 2 if qd['type'] == 'noul' else len(qd['criteria'])
    if qd['type'] == 'score': return [('rev', list(reversed(range(K))))]
    out = []; seen = {tuple(range(K))}
    for r in (1, 2, 3):
        o = tuple((i + r) % K for i in range(K))
        if o in seen: continue
        seen.add(o); out.append((f'rot{r}', list(o)))
    return out


def dist(p, lg):
    pr_ = torch.softmax(lg.float(), -1).tolist()
    return {lab: pr_[i] for i, lab in enumerate(p['rq'].slot_labels)}


byitem = collections.OrderedDict()
for suite, iid, qn, st, spec in EK.all_question_items(a.suites.split(',')):
    byitem.setdefault(iid, dict(state=st, qs=collections.OrderedDict(), suite=suite))['qs'][qn] = spec
items = list(byitem.items())[:a.limit] if a.limit else list(byitem.items())
recs = []; t0 = time.time()
with torch.inference_mode():
    for k, (iid, e) in enumerate(items):
        st = render_state(e['state']); jobs = []
        for qn, qd in e['qs'].items():
            jobs.append((qn, 'orig', prep(st, qd)))
            for vn, o in variants(qd): jobs.append((qn, vn, prep(st, qd, o)))
        prs = [j[2] for j in jobs]
        s = min((p['s'] for p in prs), key=len)
        if a.layout == 'set':
            lgs = dtset.set_logits(m, s, prs, a.Ls, a.bridge)
        else:
            lgs = m.dt_logits(s, [p['q'] for p in prs], prs, a.Ls, a.bridge)
        per = collections.defaultdict(dict)
        for (qn, vn, p), lg in zip(jobs, lgs): per[qn][vn] = dist(p, lg)
        for qn, d in per.items():
            recs.append(dict(id=iid, q=qn, suite=e['suite'], kind=e['qs'][qn]['type'], d=d))
        if k % 100 == 0: print(k, len(items), f'{time.time() - t0:.0f}s', flush=True)


def summ(rs):
    unch = []; dp = []; tv = []
    for r in rs:
        o = r['d']['orig']; ao = max(o, key=o.get)
        for vn, v in r['d'].items():
            if vn == 'orig': continue
            unch.append(max(v, key=v.get) == ao); dp.append(float(np.mean([abs(v[l] - o[l]) for l in o]))); tv.append(0.5 * sum(abs(v[l] - o[l]) for l in o))
    return dict(n_q=len(rs), n_var=len(unch), unchanged=float(np.mean(unch)) if unch else None, mean_abs_dp=float(np.mean(dp)) if dp else None,
                mean_tv=float(np.mean(tv)) if tv else None)


S = {'rotations(noul+choice)': summ([r for r in recs if r['kind'] != 'score']), 'reversal(score)': summ([r for r in recs if r['kind'] == 'score'])}
for su in a.suites.split(','):
    S[f'{su} rotations'] = summ([r for r in recs if r['suite'] == su and r['kind'] != 'score'])
for kd in ('noul', 'choice'):
    S[f'{kd} rotations'] = summ([r for r in recs if r['kind'] == kd])
print(json.dumps(S, indent=1), flush=True)
json.dump(dict(summary=S, args=vars(a), recs=recs), open(os.path.expanduser(a.out), 'w'))
