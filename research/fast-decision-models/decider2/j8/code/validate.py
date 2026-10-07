"""validate.py (CPU box): hob.py (fp32 / bf16) against the stock strands-decider engine on CPU (fp32, unmerged PEFT) and against the GPU refs."""
import os, sys, json, time, glob
sys.path.insert(0, os.path.expanduser('~/work/j8')); sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
torch.set_num_threads(int(os.environ.get('NT', '16')))
import hob

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
items = [json.loads(l) for l in open(os.path.expanduser('~/work/j8/ids.jsonl'))]
refs = {}
for s in ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
    for l in open(os.path.expanduser(f'~/work/evalkit/refs/{s}.hobson.jsonl')):
        r = json.loads(l); refs[r['id']] = r['hobson']
W, hs, cfg = hob.load_weights()
head = hob.Head(hs, cfg)
print('head keys', list(hs.keys()), flush=True)
short = [it for it in items if len(it['s']) + len(it['q']) < 700 and it['suite'] == 'REAL-agree'][:N]


def run(model, it):
    ids = torch.tensor(it['s'] + it['q'])
    sel = torch.tensor([len(it['s']) + o for o in it['opt']] + [len(ids) - 1])
    with torch.inference_mode():
        rows = model(ids, sel)
    return head.probs(rows, it['kind'])


res = {}
for dt in (torch.float32, torch.bfloat16):
    m = hob.Hob(W, dtype=dt).eval()
    out = []
    for it in short:
        t = time.time(); p = run(m, it); ms = (time.time() - t) * 1000
        ref = refs[it['id']][it['qn']]
        rp = torch.tensor([ref[l] for l in it['labels']])
        out.append(dict(id=it['id'], qn=it['qn'], L=len(it['s']) + len(it['q']), p=p.tolist(), ref=rp.tolist(), dmax=float((p - rp).abs().max()), ms=round(ms)))
        print(dt, out[-1]['L'], 'dmax %.4f' % out[-1]['dmax'], 'agree', int(p.argmax() == rp.argmax()), '%d ms' % ms, flush=True)
    res[str(dt)] = out
    del m
# stock engine on CPU (fp32) for the same items
from strands_decider.infer import load_engine
from kitrun import engine_eval, ckpt
eng = load_engine(ckpt(), device='cpu'); eng.model.config.max_length = 16384
st = []
suites = {}
for s in ['REAL-agree']:
    for l in open(os.path.expanduser(f'~/work/evalkit/suites/{s}.jsonl')):
        r = json.loads(l); suites[r['id']] = r
for it in short:
    r = suites[it['id']]
    t = time.time(); got = engine_eval(eng, r['state'], {it['qn']: r['questions'][it['qn']]}); ms = (time.time() - t) * 1000
    gp = torch.tensor([got[it['qn']][l] for l in it['labels']])
    st.append(dict(id=it['id'], p=gp.tolist(), ms=round(ms)))
    print('stock', 'ms', round(ms), flush=True)
res['stock'] = st
for k in ('torch.float32', 'torch.bfloat16'):
    d = [max(abs(a - b) for a, b in zip(x['p'], y['p'])) for x, y in zip(res[k], st)]
    print(k, 'vs stock-CPU-fp32 max|dp|', ['%.5f' % v for v in d])
json.dump(res, open(os.path.expanduser('~/work/j8/validate.json'), 'w'))
