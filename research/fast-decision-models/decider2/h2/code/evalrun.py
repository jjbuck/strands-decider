"""H2 correctness: every evalkit question through the H2 runtime at each precision (hobson layout: state + one question, exactly the
references' token ids via kitrun.prep_question). Writes preds_<prec>.jsonl rows {suite,id,q,probs,logits}; resumable.
python evalrun.py bf16,w8a8,w4a8,w4a4 [suites] [tag]"""
import sys, os, json, time, torch
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import evalkit as EK
from kitrun import load_P, prep_question, probdict
from lean2 import Lean2
import qrt as Q
precs = sys.argv[1].split(';') if ';' in sys.argv[1] else sys.argv[1].split(','); suites = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] != 'all' else None
tag = sys.argv[3] if len(sys.argv) > 3 else ''
P = load_P()
ln = Lean2(P.tm, fuse='fold'); head = P.model.head.float().eval()
items = list(EK.all_question_items(suites))
byid = {}
for s in (suites or ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']):
    for it in EK.load_suite(s): byid[it['id']] = it
print('questions', len(items), flush=True)
for prec in precs:
    pname = os.path.basename(prec.split(':')[1]).replace('.json', '') if prec.startswith('map:') else prec
    wc = None
    if os.environ.get('CODES'):     # one file, or 'w8=path,w4=path'
        cs_ = os.environ['CODES']
        if '=' in cs_:
            wc = {kv.split('=')[0]: torch.load(os.path.expanduser(kv.split('=')[1])) for kv in cs_.split(',')}
            pname += '_gptq_' + '_'.join(sorted(wc))
        else:
            wc = torch.load(os.path.expanduser(cs_)); pname += '_' + os.path.basename(cs_).replace('.pt', '')
    out = os.path.expanduser(f'~/work/h2/preds_{pname}{tag}.jsonl')
    done = set()
    if os.path.exists(out):
        for l in open(out): r = json.loads(l); done.add((r['id'], r['q']))
    m = Q.QRT(ln, head=head, prec=prec, ohead=os.environ.get('OHEAD') == '1', wcodes=wc); m.tune = False
    t0 = time.time(); n = 0
    with open(out, 'a') as f, torch.inference_mode():
        for suite, iid, qn, st, spec in items:
            if (iid, qn) in done: continue
            pr = prep_question(P, byid[iid], qn)
            ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
            hn = m.forward(ids, Q.Lay('single', T))
            rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
            h = m.unrot(hn[rows]).float()
            lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
            pd = probdict(pr['rq'], torch.softmax(lg, -1).tolist())
            f.write(json.dumps(dict(suite=suite, id=iid, q=qn, T=T, probs=pd, logits=[round(x, 5) for x in lg.tolist()])) + '\n'); n += 1
            if n % 200 == 0: f.flush(); print(prec, n, f'{time.time() - t0:.0f}s', flush=True)
    print('done', prec, n, f'{time.time() - t0:.0f}s', flush=True)
    del m; torch.cuda.empty_cache()
