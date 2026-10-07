"""Q3 emulated scoring: every evalkit question (3,227) through the Q3 emulation (integer-exact W4A4 student, or the bf16 teacher), hobson's
shared-prefix layout (questions of one item share the state pass). Writes ~/work/q3/preds/TAG.jsonl rows {suite,id,q,T,probs};
optionally exports the deployed-kernel codes {(i,k): (q int8, s fp32)} for H2's QRT (evalrun.py CODES=...).
python q3eval.py --ck PATH|teacher|gptq --tag TAG [--suites all] [--export PATH] [--qrow_ab 8]"""
import os, sys, json, time, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/q3'), os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import q3lib as QL
import evalkit as EK

ap = argparse.ArgumentParser(); ap.add_argument('--ck', required=True); ap.add_argument('--tag', required=True); ap.add_argument('--suites', default='all')
ap.add_argument('--export', default=''); ap.add_argument('--qrow_ab', type=int, default=0); ap.add_argument('--maxq', type=int, default=8)
ap.add_argument('--max_rows', type=int, default=14000); ap.add_argument('--q8', default=''); ap.add_argument('--f1', type=int, default=0); ap.add_argument('--gptq', default='~/work/q3/gptq4.pt'); ap.add_argument('--export_only', type=int, default=0)
a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(min(1.0, 21.0 * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory))
W = os.path.expanduser('~/work/q3'); os.makedirs(f'{W}/preds', exist_ok=True)
m = QL.Q3()
if a.qrow_ab: m.qrow_ab = a.qrow_ab
student = a.ck != 'teacher'
if student:
    if a.ck == 'gptq':
        init = torch.load(os.path.expanduser(a.gptq), map_location='cpu')
        src = {key: dict(q=e['q'], s=e['s']) for key, e in init.items()}; del init
    else:
        sd = torch.load(os.path.expanduser(a.ck), map_location='cpu')
        src = {}
        for key, e in sd['gemms'].items():
            if e['mode'] == 'lat': src[key] = dict(q=torch.round(e['A'].float() / e['s'][:, None]).clamp(-7, 7).to(torch.int8), s=e['s'])
            elif e['mode'] == 'code': src[key] = dict(q=e['q'], s=e['s'])
            elif e['mode'] == 'soft':
                src[key] = dict(q=(e['cf'].float() + (QL.hsoft(e['V']) >= 0.5).float()).clamp(-7, 7).to(torch.int8), s=e['s'])
        del sd
    q8src = torch.load(os.path.expanduser(a.q8), map_location='cpu') if a.q8 else None
    m.set_student('code', src, trainable=False, q8src=q8src); del q8src
    QL.F1 = bool(a.f1)
    if a.export:
        torch.save({key: (e['q'].to(torch.int8), e['s'].float()) for key, e in src.items()}, os.path.expanduser(a.export)); print('exported', a.export, flush=True)
    del src
    if a.export_only: sys.exit(0)
suites = None if a.suites == 'all' else a.suites.split(',')
items = list(EK.all_question_items(suites))
by = collections.OrderedDict()
for s, iid, q, st, spec in items: by.setdefault(iid, []).append((s, q, st, spec))
out = f'{W}/preds/{a.tag}.jsonl'
done = set()
if os.path.exists(out):
    for l in open(out): r = json.loads(l); done.add((r['id'], r['q']))
print('questions', len(items), 'items', len(by), 'done', len(done), flush=True)
t0 = time.time(); n = 0
with open(out, 'a') as f, torch.no_grad():
    for iid, qs in by.items():
        qs = [x for x in qs if (iid, x[1]) not in done]
        if not qs: continue
        state = qs[0][2]
        chunks = [qs[i:i + a.maxq] for i in range(0, len(qs), a.maxq)]
        for ch in chunks:
            rq = m.prep(state, [x[3] for x in ch])
            groups = [list(range(len(ch)))]
            if len(rq['s']) + sum(len(x['q']) for x in rq['qs']) > a.max_rows: groups = [[j] for j in range(len(ch))]
            for g in groups:
                rqg = dict(s=rq['s'], qs=[rq['qs'][j] for j in g])
                (lp, _), lay = m.run(rqg, student=student, keep=())
                for j, l_ in zip(g, lp):
                    p = l_.float().exp().tolist(); labs = rq['qs'][j]['labels']
                    f.write(json.dumps(dict(suite=ch[j][0], id=iid, q=ch[j][1], T=len(rq['s']) + len(rq['qs'][j]['q']), probs={lab: p[k] for k, lab in enumerate(labs)})) + '\n')
                    n += 1
        if n % 200 < len(qs): f.flush(); print(a.tag, n, f'{time.time() - t0:.0f}s', flush=True)
print('done', a.tag, n, f'{time.time() - t0:.0f}s', flush=True)
