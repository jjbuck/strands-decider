"""H6 eval: every evalkit question through the deployed H2/H6 kernels (QRT6), in one of these layouts:
  plain      hobson state-first: [state][question], one sequence (= H2 evalrun)
  plainqb    plain, question rows in bf16, state rows low-bit (row split at q0)
  schema     schema-first, uniform low-bit: [question minus its final '<answer>' (3 tokens): compiled bundle][state]['<answer>' slot, 3 tokens]
  schemamix  schema-first, bundle compiled in bf16 (free: once per deployment), state rows low-bit, answer slot bf16
  schemapq   schema-first, bundle bf16, state AND slot low-bit
Options are read from the compiled bundle rows (schema*), the decision from the slot row.
python h6eval.py OUT.jsonl --prec map:...|w4a4|bf16 --codes w8=..,w4=.. --lrot ckpt.pt --layout schemamix [--suites a,b] [--limit n]"""
import sys, os, json, time, argparse, collections, torch
sys.path[:0] = [os.path.expanduser('~/work/h6'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--prec', default='bf16'); ap.add_argument('--codes', default=''); ap.add_argument('--lrot', default='')
ap.add_argument('--layout', default='schemamix'); ap.add_argument('--suites', default='all'); ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--force_rot', action='store_true'); ap.add_argument('--head', default='')
a = ap.parse_args()
import evalkit as EK
from kitrun import load_P, prep_question, probdict
from lean2 import Lean2
import qrt6 as Q6
from qrt import Lay

P = load_P()
will_fold = a.prec == 'bf16' and not a.lrot and not a.force_rot
ln = Lean2(P.tm, fuse='fold' if will_fold else ''); head = P.model.head.float().eval()
if not will_fold:
    import gc
    for d in ln.layers:
        for k_ in ('Wgu_il', 'Win_f', 'Wgu_f'): d.pop(k_, None)
    P.model.torso = None; P.tm = None
    for at in ('torso', 'tm', 'model'):
        if hasattr(ln, at) and at != 'layers': setattr(ln, at, None)
    gc.collect(); torch.cuda.empty_cache()
    print('mem before build', round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
if a.lrot:
    _sd = torch.load(os.path.expanduser(a.lrot), map_location='cuda')
    if '_head' in _sd:
        with torch.inference_mode(): head.load_state_dict(_sd['_head'])
        print('trained head loaded', flush=True)
    del _sd
suites = None if a.suites == 'all' else a.suites.split(',')
items = list(EK.all_question_items(suites))
if a.limit: items = items[:a.limit]
byid = {}
for s in (suites or ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']):
    for it in EK.load_suite(s): byid[it['id']] = it
out = os.path.expanduser(a.out)
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l); done.add((r['id'], r['q']))
lay_kind = a.layout
need_split = lay_kind in ('plainqb', 'plainqbr', 'schemamix', 'schemapq', 'setsmix')
m = Q6.QRT6(ln, head=head, prec=a.prec, aclip4=float(os.environ.get('AC4', '0.9')), lrot=Q6.load_lrot(a.lrot), wcodes=Q6.load_codes(a.codes), split=need_split,
            force_rot=a.force_rot or bool(a.lrot), slim=not will_fold); m.tune = False
if not m.fold: m.slim(P)
print('mem after build', round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
print('model built', a.prec, a.layout, a.lrot, 'fold' if m.fold else 'rot', 'ba16', getattr(m, 'ba16', None), 'aclip4', m.aclip[4], flush=True)

# prepare + group by question prefix (schema layouts compile each distinct bundle once)
todo = []
for suite, iid, qn, st, spec in items:
    if (iid, qn) in done: continue
    pr = prep_question(P, byid[iid], qn)
    todo.append((suite, iid, qn, pr))
groups = collections.OrderedDict()
for t in todo:
    key = tuple(t[3]['q']) if (lay_kind.startswith('schema') or lay_kind.startswith('sets')) else len(groups)
    groups.setdefault(key, []).append(t)
print('questions', len(todo), 'groups', len(groups), flush=True)


def head_probs(hdec, hopt, pr):
    lg = (m.head(hdec[None], hopt[None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
    return probdict(pr['rq'], torch.softmax(lg, -1).tolist()), lg


t0 = time.time(); n = 0
with open(out, 'a') as f, torch.inference_mode():
    for key, grp in groups.items():
        if lay_kind.startswith('schema'):
            q = grp[0][3]['q']; pre = q[:-3]; slot = q[-3:]
            Tmax = max(len(t[3]['s']) for t in grp) + 3
            cache = m.compile_prefix(torch.tensor(pre, device='cuda'), Tmax, r0=(0 if lay_kind in ('schemamix', 'schemapq') else None))
            Pn = len(pre)
            for suite, iid, qn, pr in grp:
                Ls = len(pr['s']); ids = torch.tensor(pr['s'] + slot, device='cuda')
                lay = Lay('schema', Ls, nslots=3, P=Pn)
                hn = m.forward(ids, lay, cache, r0=(Ls if lay_kind == 'schemamix' else None))
                hdec = m.unrot(hn[Ls + 2:Ls + 3]).float()[0]
                hopt = cache['hP'][torch.tensor(pr['opt'], device='cuda')].float()
                pd, lg = head_probs(hdec, hopt, pr)
                f.write(json.dumps(dict(suite=suite, id=iid, q=qn, T=Ls + 3 + Pn, probs=pd, logits=[round(x, 5) for x in lg.tolist()])) + '\n'); n += 1
            del cache
        elif lay_kind.startswith('sets'):
            # H7 'slot sets' (1 question per sequence): [q[:-3] bundle, compiled][state][option-end tokens of q][<answer>]; options read from
            # the re-emitted (state-aware) rows, decision from the last row. setsmix: bundle bf16, state low-bit, slot-set rows bf16.
            q = grp[0][3]['q']; pre = q[:-3]; opt = grp[0][3]['opt']; slot = [q[o] for o in opt] + q[-3:]; Kn = len(opt)
            Tmax = max(len(t[3]['s']) for t in grp) + len(slot)
            cache = m.compile_prefix(torch.tensor(pre, device='cuda'), Tmax, r0=(0 if lay_kind == 'setsmix' else None))
            Pn = len(pre)
            for suite, iid, qn, pr in grp:
                Ls = len(pr['s']); ids = torch.tensor(pr['s'] + slot, device='cuda')
                lay = Lay('schema', Ls, nslots=len(slot), P=Pn)
                hn = m.forward(ids, lay, cache, r0=(Ls if lay_kind == 'setsmix' else None))
                h = m.unrot(hn[Ls:Ls + len(slot)]).float()
                pd, lg = head_probs(h[-1], h[:Kn], pr)
                f.write(json.dumps(dict(suite=suite, id=iid, q=qn, T=Ls + len(slot) + Pn, probs=pd, logits=[round(x, 5) for x in lg.tolist()])) + '\n'); n += 1
            del cache
        else:
            for suite, iid, qn, pr in grp:
                ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
                lr_ = None
                if lay_kind == 'plainqbr':   # question rows bf16 EXCEPT the readout rows (option rows + '<answer>'), which stay low-bit
                    lr_ = torch.tensor(sorted(set([pr['q0'] + o for o in pr['opt']] + [T - 3, T - 2, T - 1])), device='cuda')
                hn = m.forward(ids, Lay('single', T), r0=(pr['q0'] if lay_kind in ('plainqb', 'plainqbr') else None), lowrows=lr_)
                rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
                h = m.unrot(hn[rows]).float()
                pd, lg = head_probs(h[0], h[1:], pr)
                f.write(json.dumps(dict(suite=suite, id=iid, q=qn, T=T, probs=pd, logits=[round(x, 5) for x in lg.tolist()])) + '\n'); n += 1
        if n % 200 < len(grp): f.flush(); print(n, f'{time.time() - t0:.0f}s', flush=True)
print('done', n, f'{time.time() - t0:.0f}s', flush=True)
