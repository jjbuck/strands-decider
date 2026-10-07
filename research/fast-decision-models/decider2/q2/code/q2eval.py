"""Score every evalkit question through Q2's runtime (hobson layout: state + one question; references' exact token ids).
python q2eval.py NAME FMT [suites]   FMT: w4q8 | w4 | w8 | map:<json>  (QRT2) or h2:<prec> (H2 QRT, e.g. h2:map:~/work/h1/precmap_w8a8_b8.json:w8a8)
Writes ~/work/q2/preds/preds_NAME.jsonl {suite,id,q,T,probs,logits}; resumable. GPTQ codes from $CODES (default H1-recipe w8/w4)."""
import sys, os, json, time, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import evalkit as EK
from kitrun import load_P, prep_question, probdict
from lean2 import Lean2
import qrt as Q, q2rt as R
name, fmt = sys.argv[1], sys.argv[2]
suites = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] != 'all' else None
CODES = os.environ.get('CODES', 'w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt')
wc = {kv.split('=')[0]: torch.load(os.path.expanduser(kv.split('=')[1])) for kv in CODES.split(',')} if CODES else None
P = load_P()
fold = fmt == 'h2:bf16'
ln = Lean2(P.tm, fuse='fold' if fold else ''); head = P.model.head.float().eval()
if fmt.startswith('h2:'):
    m = Q.QRT(ln, head=head, prec=fmt[3:], wcodes=None if fold else wc); m.tune = False
elif fmt.startswith('c:'):
    m = R.QRT2C(ln, head=head, fmt=fmt[2:], wcodes=wc, tune=False); m.tune = False
elif fmt.startswith('d:'):     # B8: state rows whose token contains a digit run int8 too (QRT2 one-launch path with rowmask)
    m = R.QRT2(ln, head=head, fmt=fmt[2:], wcodes=wc, tune=False); m.tune = False
    V = len(P.tok); dig = torch.zeros(V + 1024, dtype=torch.bool)
    for t in range(V):
        try:
            if any(ch.isdigit() for ch in P.tok.decode([t])): dig[t] = True
        except Exception: pass
    DIG = dig.cuda(); print('digit-bearing vocab ids', int(dig.sum()), 'of', V, flush=True)
else:
    m = R.QRT2(ln, head=head, fmt=fmt, wcodes=wc, tune=False); m.tune = False
del wc; torch.cuda.empty_cache()
items = list(EK.all_question_items(suites))
byid = {}
for s in (suites or ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']):
    for it in EK.load_suite(s): byid[it['id']] = it
os.makedirs(os.path.expanduser('~/work/q2/preds'), exist_ok=True)
out = os.path.expanduser(f'~/work/q2/preds/preds_{name}.jsonl')
done = set()
if os.path.exists(out):
    for l in open(out): r = json.loads(l); done.add((r['id'], r['q']))
print('questions', len(items), 'done', len(done), flush=True)
t0 = time.time(); n = 0; nd = 0; ns = 0
with open(out, 'a') as f, torch.inference_mode():
    for suite, iid, qn, st, spec in items:
        if (iid, qn) in done: continue
        pr = prep_question(P, byid[iid], qn)
        ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
        m.q0 = pr['q0']
        if fmt.startswith('d:'):
            is8 = DIG[ids].clone(); is8[pr['q0']:] = True
            m.pos = None; m.set_rowmask(is8); nd += int(is8[:pr['q0']].sum()); ns += pr['q0']
        hn = m.forward(ids, Q.Lay('single', T))
        rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
        h = m.unrot(hn[rows]).float()
        lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
        pd = probdict(pr['rq'], torch.softmax(lg, -1).tolist())
        f.write(json.dumps(dict(suite=suite, id=iid, q=qn, T=T, probs=pd, logits=[round(x, 5) for x in lg.tolist()])) + '\n'); n += 1
        if n % 200 == 0: f.flush(); print(name, n, f'{time.time() - t0:.0f}s', flush=True)
print('done', name, n, f'{time.time() - t0:.0f}s', 'int8 state-row share', round(nd / max(ns, 1), 4), flush=True)
