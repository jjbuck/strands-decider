"""eval queue: run the highest-priority job whose checkpoints exist; repeat until all are done.  python evq.py"""
import os, time, subprocess, json, sys
W = os.path.expanduser('~/work/j4/')
FAST = 'JB-all,REAL-agree,CF,CF-probe'
FULL = 'JB-all,REAL-agree,LONG,CF,CF-probe'
JOBS = [  # name, stack, suites (priority order)
    ('a_r12000', 'ck/a/r12000.pt', FULL), ('b_r12000', 'ck/dp/final.pt,ck/b/r12000.pt', FULL),
    ('c_final', 'ck/c/final.pt', FULL),
    ('a_r1000', 'ck/a/r1000.pt', FAST), ('b_r1000', 'ck/dp/final.pt,ck/b/r1000.pt', FAST),
    ('a_r4000', 'ck/a/r4000.pt', FAST), ('b_r4000', 'ck/dp/final.pt,ck/b/r4000.pt', FAST),
    ('d_r4000', 'ck/ntp/final.pt,ck/d/r4000.pt', 'JB-all,REAL-label,CF,CF-probe'), ('d_r1000', 'ck/ntp/final.pt,ck/d/r1000.pt', 'JB-all,REAL-label,CF,CF-probe'),
]
extra = W + 'evq_extra.json'
while True:
    jobs = JOBS + (json.load(open(extra)) if os.path.exists(extra) else [])
    todo = [j for j in jobs if not os.path.exists(W + f'preds/{j[0]}.json')]
    if not todo: print('ALL EVALS DONE', flush=True); break
    ready = [j for j in todo if all(os.path.exists(W + p) for p in j[1].split(','))]
    busy = subprocess.call("pgrep -f '[p]ython evalj4[.]py' > /dev/null || pgrep -f '[j]4lat.py' > /dev/null", shell=True) == 0
    if not ready or busy: time.sleep(30); continue
    name, stack, suites = ready[0]
    print('===', name, time.strftime('%T'), flush=True)
    rc = subprocess.call(f'cd {W} && python evalj4.py preds/{name}.json --stack {stack} --suites {suites} > logs/ev_{name}.log 2>&1', shell=True)
    print('=== end', name, time.strftime('%T'), 'rc', rc, flush=True)
    if rc != 0: time.sleep(60)
