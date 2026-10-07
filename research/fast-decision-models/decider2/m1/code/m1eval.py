"""M1 evaluation (reporting only).
  python m1eval.py evalkit OUT.jsonl (--ckpt CK | --untrained VARIANT) [--suites a,b] [--teacher 1]
        every evalkit question (3,227) through the student (or the in-process teacher); resumable JSONL; writes OUT.preds.json at the end
  python m1eval.py holdout OUT.json (--ckpt CK | --untrained VARIANT) [--n 300]
        train_v5.holdout.jsonl (v19's own held-out split, train-side data): accuracy per task for hobson (teacher path) and the student,
        paired per row (J3's dt_holdout.py sampling: shuffle seed 11, first n rows per task)"""
import os, sys, json, random, argparse, collections, time
sys.path[:0] = [os.path.expanduser('~/work/m1')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import m1lib as ML

ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('out')
ap.add_argument('--ckpt', default=''); ap.add_argument('--untrained', default=''); ap.add_argument('--tau_json', default='~/work/m1/untrained.json')
ap.add_argument('--suites', default=''); ap.add_argument('--teacher', type=int, default=0); ap.add_argument('--n', type=int, default=300)
ap.add_argument('--maxq', type=int, default=8)
a = ap.parse_args()
VARIANTS = {
    'plain': dict(conv=True, rope=True, beta=False, decay=False), 'beta': dict(conv=True, rope=True, beta=True, decay=False),
    'decay': dict(conv=True, rope=True, beta=False, decay=True), 'full': dict(conv=True, rope=True, beta=True, decay=True), 'full136': dict(conv=True, rope=True, beta=True, decay=True, slots=0),
    'full_norope': dict(conv=True, rope=False, beta=True, decay=True), 'full_noconv': dict(conv=False, rope=True, beta=True, decay=True)}
m = ML.M1()
if a.ckpt:
    meta = m.load_trainable(os.path.expanduser(a.ckpt)); print('loaded', a.ckpt, meta, flush=True)
elif a.untrained:
    loc = json.load(open(os.path.expanduser(a.tau_json)))['local']['best'][a.untrained]['layers']
    m.cfg = VARIANTS[a.untrained]; m.convert(ML.GDN, tau={int(i): v['tau'] for i, v in loc.items()})
    print('untrained', a.untrained, m.cfg, flush=True)
OUT = os.path.expanduser(a.out)

if a.mode == 'evalkit':
    ML.eval_suites(m, a.suites.split(',') if a.suites else None, OUT, student=not a.teacher, maxq=a.maxq)
    json.dump(ML.jsonl_to_preds(OUT), open(OUT.replace('.jsonl', '') + '.preds.json', 'w'))
    print('done', flush=True); sys.exit(0)

if a.mode == 'holdout':
    rows = [json.loads(l) for l in open(os.path.expanduser('~/work/training/data/train_v5.holdout.jsonl'))]
    random.Random(11).shuffle(rows)
    byt = collections.defaultdict(list)
    for r in rows:
        if len(byt[r['task']]) < a.n: byt[r['task']].append(r)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    with torch.no_grad():
        for task, rs in sorted(byt.items()):
            if task in res: continue
            per = []
            for r in rs:
                qd, gold = ML.v5_q(r)
                rq = m.prep(r['state'], [qd]); gi = rq['qs'][0]['labels'].index(gold)
                (lt, _), _ = m.run(rq, student=False, keep=())
                (ls, _), _ = m.run(rq, student=True, keep=())
                per.append([int(int(lt[0].argmax()) == gi), int(int(ls[0].argmax()) == gi), int(int(lt[0].argmax()) == int(ls[0].argmax()))])
            res[task] = dict(n=len(per), hobson=sum(x[0] for x in per) / len(per), model=sum(x[1] for x in per) / len(per),
                             agree=sum(x[2] for x in per) / len(per), rows=per)
            print(task, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[task].items() if k != 'rows'}, f'{time.time() - t0:.0f}s', flush=True)
            json.dump(res, open(OUT, 'w'))
    sys.exit(0)
