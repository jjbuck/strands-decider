"""stacklib checks (box). python test_stack.py OUT.json [ref|cross|combos|grad|all]
ref    : flags off, adapters off, vs hobson-v19's references (refs/REAL-agree.hobson.jsonl): argmax on 60 real questions, max |dp|
         (+ the same with a zero-init student: must be bit-identical to the adapters-off forward)
cross  : each flag against the source agent's own code with untrained / zero adapters: D vs J3 dtlib (Ls 8, bridge A), V vs J7 j7lib
         (16k, mean-of-constituents init), C vs J9 j9lib (layout R, comp skip; J9 also feeds block rows into the stream's conv history,
         switched on here via Stack.C_CONV_BLOCKS)
combos : all 16 combinations of D, C, V, Q with random adapters on 8 real requests: finite, and multi-question == single-question per question
grad   : one backward per combination on a 6000-token train-like sequence: peak memory, seconds"""
import os, sys, json, time, random, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch
import stacklib as SL
import evalkit as EK
from strands_decider.prompting import render_state
OUT = sys.argv[1]; WHAT = sys.argv[2] if len(sys.argv) > 2 else 'all'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(): json.dump(res, open(OUT, 'w'), indent=1)


def norm(d): s = sum(d.values()); return {k: v / s for k, v in d.items()}


items = EK.load_suite('REAL-agree')
refs = EK.load_refs('REAL-agree')
random.Random(0).shuffle(items)
sel = []; nq = 0
for it in items:
    if nq >= 60: break
    sel.append(it); nq += len(it['questions'])

if WHAT in ('ref', 'all', 'combos', 'cross'):
    m = SL.Stack(); m.free_hf(); m.load_assets()
if WHAT in ('ref', 'all'):
    t0 = time.time(); ag = 0; n = 0; mx = 0.0; dps = []
    P0 = {}
    for it in sel:
        p = m.predict_item(it, '')
        P0[it['id']] = p
        for q, d in p.items():
            h = norm(refs[it['id']]['hobson'][q]); n += 1
            ag += max(d, key=d.get) == max(h, key=h.get)
            dp = max(abs(d[k] - h[k]) for k in d); dps.append(dp); mx = max(mx, dp)
    dps.sort()
    res['ref'] = dict(n=n, argmax_agree=ag, max_dp=mx, median_dp=dps[len(dps) // 2], p90_dp=dps[int(.9 * len(dps))], secs=round(time.time() - t0))
    print('ref', res['ref'], flush=True)
    # zero-init student (no flags) must equal the adapters-off forward exactly
    m.add_student('')
    mx2 = 0.0
    for it in sel[:10]:
        p = m.predict_item(it, '')
        for q, d in p.items(): mx2 = max(mx2, max(abs(d[k] - P0[it['id']][q][k]) for k in d))
    res['ref']['zero_student_max_dp_vs_off'] = mx2
    print('zero student', mx2, flush=True); save()
    m.lora = None; m.head = None

if WHAT in ('cross', 'all'):
    sys.path.insert(0, os.path.expanduser('~/work/k1/src'))
    out = {}
    # ---- D vs J3 dtlib (untrained: no LoRA, no memory adapters, hobson head)
    from dtlib import DT
    dt = DT.__new__(DT); dt.__dict__.update(m.__dict__); dt.mem = None; dt.lora = None; dt.head = None
    mx = 0.0; ag = 0; n = 0
    with torch.inference_mode(), m.teacher():
        for it in sel[:12]:
            b = m.item_base(it)
            o = m.forward(m.view(b, 'D'))
            prs = [dict(opt=q['opt'], rq=q['rq']) for q in b.qs]
            lj = dt.dt_logits(b.s, [q['q'] for q in b.qs], prs, 8, 'A')
            for a_, c_ in zip(o['logits'], lj):
                pa = torch.softmax(a_.float(), -1); pc = torch.softmax(c_.float(), -1)
                mx = max(mx, float((pa - pc).abs().max())); ag += int(pa.argmax() == pc.argmax()); n += 1
    out['D_vs_dtlib'] = dict(n=n, argmax_agree=ag, max_dp=mx); print('D', out['D_vs_dtlib'], flush=True)
    # ---- V vs J7 j7lib (zero-init super embeddings = mean of constituents), one question per sequence
    from j7lib import J7, Prep
    m.add_student('V'); m.lora = None
    j7 = J7.__new__(J7); j7.__dict__.update(m.__dict__); j7.chans = None; j7.V = m.sup_tok.V
    j7.sup = m.sup; supers = {'16k': m.sup_tok}; PP = Prep(m.eng, supers)
    mx = 0.0; ag = 0; n = 0
    with torch.inference_mode():
        for it in sel[:12]:
            for qn in it['questions']:
                b = m.item_base(it, [qn])
                o = m.forward(m.view(b, 'V'))
                pr = PP.base(render_state(it['state']), it['questions'][qn])
                bb = PP.build(pr, level='16k')
                l7 = j7.logits1(bb, head=m.head0)
                pa = torch.softmax(o['logits'][0].float(), -1); pc = torch.softmax(l7.float(), -1)
                mx = max(mx, float((pa - pc).abs().max())); ag += int(pa.argmax() == pc.argmax()); n += 1
    out['V_vs_j7lib'] = dict(n=n, argmax_agree=ag, max_dp=mx); print('V', out['V_vs_j7lib'], flush=True)
    m.sup = None; m.head = None
    # ---- C vs J9 j9lib (layout R, comp skip, untrained), J9's conv history convention
    import j9lib as J
    j9 = J.J9.__new__(J.J9); j9.__dict__.update(m.__dict__); j9.compile_only = False; j9.lora = None; j9.head = None; j9.setup_u(m.tok)
    SL.Stack.C_CONV_BLOCKS = True
    mx = 0.0; ag = 0; n = 0; nb = 0
    with torch.inference_mode(), m.teacher():
        for it in [x for x in items if x['domain'] == 'banking_knowledge'][:12]:
            for qn in list(it['questions'])[:2]:
                b = m.item_base(it, [qn])
                v = m.view(b, 'C'); nb += len(v.blocks)
                o = m.forward(v)
                req = J.tokenize_pieces(m.tok, J.pieces(b.st_text, m.ll))
                Pl = J.build_plan(j9, req, b.qs[0]['q'], 'R', m.dev)
                l9 = j9.logits_c(Pl, j9.fwd_c(Pl, comp='skip'), b.qs[0]['opt'], b.qs[0]['kind'])
                pa = torch.softmax(o['logits'][0].float(), -1); pc = torch.softmax(l9[:b.qs[0]['rq'].n_slots].float(), -1)
                mx = max(mx, float((pa - pc).abs().max())); ag += int(pa.argmax() == pc.argmax()); n += 1
    SL.Stack.C_CONV_BLOCKS = False
    out['C_vs_j9lib'] = dict(n=n, blocks=nb, argmax_agree=ag, max_dp=mx); print('C', out['C_vs_j9lib'], flush=True)
    res['cross'] = out; save()

if WHAT in ('combos', 'all'):
    m.add_student('DCVQ'); m.randomize_adapters(0.02, seed=1)
    out = {}
    reqs = [x for x in items if x['domain'] == 'banking_knowledge' and len(x['questions']) > 1][:6] + [x for x in items if x['domain'] == 'retail'][:2]
    for r in range(0, 5):
        for fl in itertools.combinations('DCVQ', r):
            f = ''.join(fl); t0 = time.time(); bad = 0; mx = 0.0; nslot = 0; nblk = 0
            for it in reqs:
                with torch.inference_mode():
                    b = m.item_base(it); v = m.view(b, f); nblk += len(v.blocks); nslot += sum(br['kind'] == 'slot' for br in v.br)
                    o = m.forward(v)
                    for j, qd in enumerate(b.qs):
                        lg = o['logits'][j]
                        if not torch.isfinite(lg).all(): bad += 1
                        b1 = m.prep(b.st_text, [(qd['name'], qd['spec'], None)])
                        o1 = m.forward(m.view(b1, f))
                        mx = max(mx, float((torch.softmax(lg.float(), -1) - torch.softmax(o1['logits'][0].float(), -1)).abs().max()))
            out[f or '-'] = dict(nonfinite=bad, multi_vs_single_max_dp=mx, blocks=nblk, slot_branches=nslot, secs=round(time.time() - t0, 1))
            print(f or '-', out[f or '-'], flush=True)
    res['combos'] = out; save()

if WHAT in ('grad', 'all'):
    del m; torch.cuda.empty_cache()
    m = SL.Stack(); m.detach_inference(); g = m.add_student('DCVQ')
    pool = [json.loads(l) for _, l in zip(range(3000), open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')))]
    r = max([x for x in pool if x['domain'] == 'banking_knowledge'], key=lambda x: min(x['n_state_tok'], 5800) * (x['n_state_tok'] < 9000))
    out = {}
    for f in ['', 'D', 'C', 'V', 'Q', 'DCVQ']:
        st = SL.trunc_text(m.tok, render_state(r['state']), 5600)
        ql = [(q, r['questions'][q], None) for q in sorted(r['questions'])[:4]]
        b = m.prep(st, ql)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad(), m.teacher(): ot = m.forward(m.view(b, ''), keep=SL.HID_LAYERS)
        o = m.forward(m.view(b, f), ckpt=True, keep=SL.HID_LAYERS)
        loss = sum(o['logits'][j].float().logsumexp(-1) for j in range(len(b.qs))) + sum(((o['kept'][i].float() - ot['kept'][i].float()) ** 2).mean() for i in SL.HID_LAYERS)
        loss.backward()
        torch.cuda.synchronize()
        gn = {k: float(sum(p.grad.norm() ** 2 for p in ps if p.grad is not None) ** .5) for k, ps in g.items()}
        for ps in g.values():
            for p in ps: p.grad = None
        out[f or '-'] = dict(rows=o['P'].T, teacher_tok=len(b.s) + sum(len(q['q']) for q in b.qs), secs=round(time.time() - t0, 2),
                             peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), grad_norms={k: round(v, 5) for k, v in gn.items()})
        print(f or '-', out[f or '-'], flush=True)
    res['grad'] = out; save()
print('done', flush=True)
