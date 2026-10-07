"""Q4: all 3,227 evalkit questions through H2's deployed kernels (J15's runner, hobson layout) with the activation/weight code width changed,
to measure at decision level what Strassen's lost bit costs (per-token scales; the pair-shared scale is measured at GEMM level in q4bit.py).
python q4eval.py OUT.jsonl --prec PREC --qmax8 63 --qmax4 3 [--codes ...] [--b13 l0+read]
  --qmax8 / --qmax4 patch qrt.QMAX before the runtime is built: RTN weight codes (per channel, absmax for int8, MSE clip search for int4) and
  activation codes (per token) both use the new qmax. --b13 runs the exact eliminations (q4rt.QRT4) on the same questions."""
import sys, os, json, time, argparse, torch
sys.path[:0] = [os.path.expanduser('~/work/q4'), os.path.expanduser('~/work/j15'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'),
                os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import qrt as Q

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--prec', default='map:~/work/h1/precmap_w8a8_b8.json:w8a8'); ap.add_argument('--codes', default='')
ap.add_argument('--qmax8', type=float, default=127.0); ap.add_argument('--qmax4', type=float, default=7.0); ap.add_argument('--b13', default='')
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--inv', action='store_true')
a = ap.parse_args()
Q.QMAX[8] = a.qmax8; Q.QMAX[4] = a.qmax4
import evalkit as EK
from kitrun import prep_question, probdict, load_P
from lean2 import Lean2
from j15run import load_codes
import q4rt as R
P = load_P()
ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval()
m = R.QRT4(ln, head=head, prec=a.prec, wcodes=load_codes(a.codes)); m.tune = False
import gc
for d in ln.layers:
    for k_ in ('Wgu_il', 'Win_f', 'Wgu_f'): d.pop(k_, None)
gc.collect(); torch.cuda.empty_cache()
m.prep_l23()
if a.b13:
    if 'l0' in a.b13: m.build_l0()
    m.l0 = 'l0' in a.b13; m.l23 = 'read' if 'read' in a.b13 else ('kv' if 'kv' in a.b13 else 'none')
m.set_inv23(a.inv)
print('built', a.prec, 'inv23', a.inv, 'qmax', Q.QMAX, 'b13', a.b13, 'mem', round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
items = list(EK.all_question_items(None))
if a.limit: items = items[:a.limit]
byid = {}
for s in ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
    for it in EK.load_suite(s): byid[it['id']] = it
out = os.path.expanduser(a.out); done = set()
if os.path.exists(out):
    for l in open(out): r = json.loads(l); done.add((r['id'], r['q']))
t0 = time.time(); n = 0; buf = []
with torch.inference_mode():
    for suite, iid, qn, st, spec in items:
        if (iid, qn) in done: continue
        pr = prep_question(P, byid[iid], qn)
        ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
        rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
        lay = Q.Lay('single', T); m.q0 = pr['q0']; m.set_read(lay, rows.tolist())
        hn = m.forward(ids, lay)
        h = m.unrot(hn[rows]).float()
        lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
        pd = probdict(pr['rq'], torch.softmax(lg, -1).tolist())
        buf.append(json.dumps(dict(suite=suite, id=iid, q=qn, T=T, kind=pr['rq'].kind, probs=pd, logits=[round(x, 5) for x in lg.tolist()]))); n += 1
        if n % 200 == 0:
            with open(out, 'a') as f:
                for l in buf: f.write(l + '\n')
            buf.clear(); print(n, f'{time.time() - t0:.0f}s', flush=True)
with open(out, 'a') as f:
    for l in buf: f.write(l + '\n')
print('done', n, f'{time.time() - t0:.0f}s', flush=True)
