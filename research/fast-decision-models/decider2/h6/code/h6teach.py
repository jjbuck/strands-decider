"""H6 training set + teacher: train-split real requests (evalkit/train_pool.jsonl minus split.json eval tasks) and train_v5 gold rows.
Per example: hobson token ids (plib.P.prep: state s, question q, option offsets), state truncated to MAXS tokens (first quarter + last part,
as h1qat), teacher = bf16 hobson STATE-FIRST answer distribution from H2's bf16 runtime on s+q. Saved to ~/work/h6/teach.pt (list of dicts).
python h6teach.py N_POOL N_V5"""
import sys, os, json, random, time, torch
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from kitrun import load_P
from lean2 import Lean2
import qrt6 as Q6
from qrt import Lay
NP_, NV = int(sys.argv[1]), int(sys.argv[2]); MAXS = 1536
random.seed(6)
EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
pool = []
with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
    for l in f:
        r = json.loads(l)
        if r['task'] in EV or r['n_state_tok'] < 200: continue
        pool.append(dict(state=r['state'], questions=r['questions'], task=r['task']))
v5 = []
with open(os.path.expanduser('~/work/training/data/train_v5.jsonl')) as f:
    for li, l in enumerate(f):
        if li % 10 == 0: v5.append(json.loads(l))
random.shuffle(pool); random.shuffle(v5)
print('pool', len(pool), 'v5', len(v5), flush=True)
P = load_P(); ln = Lean2(P.tm, fuse='fold'); head = P.model.head.float().eval()
m = Q6.QRT6(ln, head=head, prec='bf16'); m.tune = False
tok = P.tok
out = []; t0 = time.time()
srcs = [('pool', r) for r in pool[:NP_]] + [('v5', r) for r in v5[:NV]]
random.shuffle(srcs)
for j, (kind, r) in enumerate(srcs):
    if kind == 'v5':
        qd = {'type': 'choice', 'instructions': random.choice([r['instructions']] + r.get('instruction_variants', [])[:2]), 'criteria': {n: d_ for n, d_ in r['options']}}
        gold = r['label']; qn = 'v5'
    else:
        qn = random.choice(sorted(r['questions'])); qd = r['questions'][qn]; gold = None
    try:
        pr = P.prep(r['state'], qd)
    except Exception as e:
        print('skip', e); continue
    s = pr['s']
    if len(s) > MAXS: s = s[:MAXS // 4] + s[-(MAXS - MAXS // 4):]
    q = pr['q']
    if j == 0: print('last q token', repr(tok.decode(q[-1:])), repr(tok.decode(q[-3:])), q[-1], flush=True)
    ids = torch.tensor(s + q, device='cuda'); T = ids.shape[0]
    with torch.inference_mode():
        hn = m.forward(ids, Lay('single', T))
        rows = torch.tensor([T - 1] + [len(s) + o for o in pr['opt']], device='cuda')
        h = m.unrot(hn[rows]).float()
        lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
        tp = torch.softmax(lg, -1).cpu()
    out.append(dict(kind=kind, qn=qn, s=s, q=q, opt=pr['opt'], n=pr['rq'].n_slots, temp=P.temp_for(pr['rq'].kind), tp=tp, gold=gold))
    if j % 500 == 0: print(j, f'{time.time() - t0:.0f}s', flush=True)
torch.save(out, os.path.expanduser('~/work/h6/teach.pt'))
print('saved', len(out), f'{time.time() - t0:.0f}s', flush=True)
