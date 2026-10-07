"""K2 self-tests of stackrt against H2's QRT forward (no K1 code needed).
python t_stack.py PREC   (PREC bf16 | w8a8)  -> ~/work/k2/res/t_stack_PREC.json
  base1 : no flags, 1 question, real items: stackrt vs QRT 'single' layout (same kernels) -> argmax agreement, max |dp|
  packed: no flags, all questions of a REAL-agree item as branches vs QRT 'packed' layout
  xsplit: graph A (0..15) then graph B (16..23) vs one full pass (must be identical)
  combos: every D,C,V,Q,X combination captures as a CUDA graph and replays (random deployment objects); prints one timing each
"""
import os, sys, json, time, itertools, random
sys.path.insert(0, os.path.expanduser('~/work/k2'))
import torch
import stackrt as S
from stackrt import Spec, Branch, Req, StackRT

PREC = sys.argv[1] if len(sys.argv) > 1 else 'bf16'
OUT = os.path.expanduser(f'~/work/k2/res/t_stack_{PREC}.json'); os.makedirs(os.path.dirname(OUT), exist_ok=True)
import evalkit as EK, qrt as Q
from kitrun import prep_question

P, m = S.load_base(PREC)
eh = S.ExitHead(P.model.head).cuda().float().eval()
res = {}


def ref_single(pr):
    ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
    hn = m.forward(ids, Q.Lay('single', T))
    rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
    h = m.unrot(hn[rows]).float()
    lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots]
    return torch.softmax(lg, -1)


def ref_packed(prs):
    s = prs[0]['s']; Ls = len(s); ids = torch.tensor(s + [t for p in prs for t in p['q']], device='cuda')
    lay = Q.Lay('packed', Ls, [len(p['q']) for p in prs])
    hn = m.forward(ids, lay)
    out = []
    for (a, b), p in zip(lay.seg, prs):
        rows = torch.tensor([b - 1] + [a + o for o in p['opt']], device='cuda')
        h = m.unrot(hn[rows]).float()
        lg = (m.head(h[:1], h[1:][None]) / P.temp_for(p['rq'].kind))[0, :p['rq'].n_slots]
        out.append(torch.softmax(lg, -1))
    return out


def spec_of(prs, **kw):
    s = prs[0]['s']; Ls = len(s)
    br = [Branch('ctx', range(Ls, Ls + len(p['q'])), p['opt'], P.temp_for(p['rq'].kind), p['rq'].n_slots, ids=p['q']) for p in prs]
    return Spec(s, range(Ls), br, **kw)


rt = StackRT(P, m, eh=eh)
items = EK.load_suite('REAL-agree')
random.Random(0).shuffle(items)
jb = EK.load_suite('JB-all')[:6]
dps = []; agree = 0; n = 0
with torch.inference_mode():
    for it in items[:10] + jb:
        q = list(it['questions'])[0]
        pr = prep_question(P, it, q)
        if len(pr['s']) + len(pr['q']) > 3000: continue
        a = ref_single(pr)
        rq = Req(rt, spec_of([pr]))
        b = rq.run_full()[0, :pr['rq'].n_slots]
        dps.append(float((a - b).abs().max())); agree += int(a.argmax() == b.argmax()); n += 1
res['base1'] = dict(n=n, agree=agree, dp_max=max(dps), dp_med=sorted(dps)[len(dps) // 2])
print('base1', res['base1'], flush=True)

dps = []; agree = 0; n = 0
with torch.inference_mode():
    for it in [x for x in items if len(x['questions']) >= 3][:6]:
        prs = [prep_question(P, it, q) for q in it['questions']]
        if len(prs[0]['s']) > 2500: continue
        A = ref_packed(prs)
        rq = Req(rt, spec_of(prs)); B = rq.run_full()
        for j, (a, p) in enumerate(zip(A, prs)):
            b = B[j, :p['rq'].n_slots]; dps.append(float((a - b).abs().max())); agree += int(a.argmax() == b.argmax()); n += 1
res['packed'] = dict(n=n, agree=agree, dp_max=max(dps), dp_med=sorted(dps)[len(dps) // 2])
print('packed', res['packed'], flush=True)

# X split continuity, also with D (random memory adapters) and Q (random deltas)
qt = S.QTab.random([('qa', 2), ('qb', 5)], seed=1)
rt.qtab = qt
rt.mem = {'full': S.Mem(m, groups=[[11, 15, 19, 23]]), 'split': S.Mem(m, groups=[[11, 15], [19, 23]])}
dps = {}
with torch.inference_mode():
    it = [x for x in items if len(x['questions']) >= 2][0]
    prs = [prep_question(P, it, q) for q in list(it['questions'])[:2]]
    Ls = len(prs[0]['s'])
    for D in (False, True):
        for withq in (False, True):
            sp = spec_of(prs, D=D)
            if withq:
                L0 = Ls + sum(len(p['q']) for p in prs)
                sp.branches += [Branch('slot', range(Ls, Ls + 3), [0, 1], 1.0, 2, name='qa'), Branch('slot', range(Ls, Ls + 6), [0, 1, 2, 3, 4], 1.0, 5, name='qb')]
            full = Req(rt, sp).run_full()
            sp.X = True
            r2 = Req(rt, sp); r2.run_a(); fb = r2.run_b()
            dps[f'D{int(D)}Q{int(withq)}'] = float((full - fb).abs().max())
res['xsplit'] = dps
print('xsplit', dps, flush=True)

# all D,C,V,Q,X combos capture + replay (random deployment objects, T = 400 state rows, 4 questions)
Vbase = P.tok.__len__()
ext = (Vbase, torch.randn(16384, 2048) * 0.02)
rtv = StackRT(P, m, qtab=qt, mem=rt.mem, eh=eh, ext=ext)
libs = {False: S.Lib.random(512, S.ATT, seed=2), True: S.Lib.random(512, S.ATT, seed=3)}
pr = prep_question(P, items[0], list(items[0]['questions'])[0])
qids = pr['q']
combo = {}
for D, C, V, Qf, X in itertools.product((0, 1), repeat=5):
    Ts = 400; Pc = 184 if C else 0; Ll = Ts - Pc
    if V: Ll = int(round(Ll / 1.97))
    live = torch.randint(1000, 100000, (Ll,)).tolist()
    if V: live = [t if j % 2 else Vbase + (t % 16384) for j, t in enumerate(live)]
    br = []
    if Qf:
        p0 = Ts
        br.append(Branch('slot', range(p0, p0 + 3), [0, 1], 1.0, 2, name='qa'))
        br.append(Branch('slot', range(p0, p0 + 6), [0, 1, 2, 3, 4], 1.0, 5, name='qb'))
    else:
        for _ in range(2): br.append(Branch('ctx', range(Ts, Ts + len(qids)), pr['opt'], 1.0, pr['rq'].n_slots, ids=qids))
    prefix = dict(rows=list(range(Pc)), pos=list(range(Pc))) if C else None
    sp = Spec(live, list(range(Pc, Pc + Ll)), br, prefix=prefix, D=bool(D), X=bool(X))
    r = Req(rtv if V else rt, sp, lib=libs[bool(D)] if C else None)
    with torch.inference_mode():
        g, out = S.capture(r.run_a if X else r.run_full)
        pool = S.fresh_pool(r, 8)
        w = S.wall_full(r, g, out, pool)
        if X:
            gb, ob = S.capture(r.run_b); wb = S.wall_full(r, gb, ob, pool); w['b_median'] = wb['median']
    key = f'D{D}C{C}V{V}Q{Qf}X{X}'
    combo[key] = w; print(key, w, flush=True)
    del g, out, r; torch.cuda.empty_cache()
res['combos'] = combo
json.dump(res, open(OUT, 'w'), indent=1)
print('done', OUT)
