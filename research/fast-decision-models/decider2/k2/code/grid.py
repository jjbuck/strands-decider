"""K2 exact-length latency grid: every D,C,V,Q,X combination (P chosen per process), state T tokens, three question sets.
python grid.py PREC [TS] [QSETS]      PREC bf16 | w8a8     -> ~/work/k2/res/lat_grid_PREC.jsonl (resumable)
Discipline: CUDA graph per exact shape, exclusive GPU, fresh random live-state ids every rep (H2D + replay + D2H), 3 warm-ups + 15 timed,
median and p95; kernel-category split of one replay (torch profiler) for the projections.
Settings: C compiled share 0.46 of the state's Qwen tokens (C at share 0 is the C=0 graph: no compiled rows); V state and compiled rows = Qwen
tokens / RS (measured median 16k ratio), in-context questions merged with the 16k tokenizer when available (else / RQ); Q makes deployed
questions K+1 slot rows (JevBench questions are never deployed); X times graph A (layers 0-15 + exit head = 'always exit') and graph B
(layers 16-23 for all questions = the continuation).
"""
import os, sys, json, time, itertools
sys.path.insert(0, os.path.expanduser('~/work/k2'))
import torch
import stackrt as S
from stackrt import Spec, Branch, Req, StackRT

PREC = sys.argv[1]
TS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else '32,64,128,256,400,1000,2000,4000').split(',')]
QS = (sys.argv[3] if len(sys.argv) > 3 else 'JB1,BK4,BK15').split(',')
OUT = os.path.expanduser(f'~/work/k2/res/lat_grid_{PREC}.jsonl'); os.makedirs(os.path.dirname(OUT), exist_ok=True)
CFG = json.load(open(os.path.expanduser('~/work/k2/grid_cfg.json')))     # question sets (token ids, opt, K), ratios, frame length
RS = CFG['ratio_state']; RQ = CFG.get('ratio_q', 1.31); SHARE = CFG.get('share', 0.46); FRAME = CFG.get('frame', 45)
REPS = int(os.environ.get('REPS', 15)); WARM = 3
PROF = os.environ.get('PROF', '1') == '1'

done = set()
if os.path.exists(OUT):
    for l in open(OUT): done.add(json.loads(l)['key'])

P, m = S.load_base(PREC)
eh = S.ExitHead(P.model.head).cuda().float().eval()
Vbase = len(P.tok)
names = sorted({q['name'] for qs in CFG['qsets'].values() for q in qs if q.get('deployed')})
Kof = {q['name']: q['K'] for qs in CFG['qsets'].values() for q in qs}
qt = S.QTab.random([(n, Kof[n]) for n in names], seed=1)
mem = {'full': S.Mem(m, groups=[[11, 15, 19, 23]]), 'split': S.Mem(m, groups=[[11, 15], [19, 23]])}
libs = {False: S.Lib.random(2048, S.ATT, seed=2), True: S.Lib.random(2048, S.ATT, seed=3)}
ext = (Vbase, torch.randn(16384, 2048) * 0.02)
S.free_bf16_copies(m)
rt = StackRT(P, m, qtab=qt, mem=mem, eh=eh, ext=ext)
print('built', PREC, 'mem GB', round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)


def build_spec(qset, T, D, C, V, Q, X):
    c = int(round(SHARE * T)) if C else 0
    live_q = T - c
    nl = max(1, int(round(live_q / RS))) if V else live_q
    nc = max(1, int(round(c / RS))) if (V and c) else c
    npre = min(nl, int(round(FRAME / RS)) if V else FRAME) if C else 0
    hi = Vbase + 16384 if V else 100000
    live = torch.randint(1000, hi, (nl,)).tolist()
    # runtime positions (Qwen positions): pre, blocks, post; V spaces rows by RS
    sp = RS if V else 1.0
    pos_pre = [j * sp for j in range(npre)]
    pos_blk = [npre * sp + j * sp for j in range(nc)]
    pos_post = [(npre + nc + j) * sp for j in range(nl - npre)]
    Pend = float(T)
    br = []; ctx = []; slots = []
    for q in CFG['qsets'][qset]:
        if Q and q.get('deployed'):
            K = q['K']; slots.append(Branch('slot', [Pend + k for k in range(K + 1)], list(range(K)), 1.0, K, name=q['name']))
        else:
            if V and q.get('ids16k'):
                ids = q['ids16k']; opt = q['opt16k']; pos = [Pend + e for e in q['ends16k']]
            elif V:
                n = max(len(q['opt']) + 1, int(round(len(q['ids']) / RQ))); ids = q['ids'][:n - 1] + [q['ids'][-1]]
                opt = [min(n - 2, int(round(o / RQ))) for o in q['opt']]; pos = [Pend + j * RQ for j in range(n)]
            else:
                ids = q['ids']; opt = q['opt']; pos = [Pend + j for j in range(len(ids))]
            ctx.append(Branch('ctx', pos, opt, 1.0, len(opt), ids=ids))
    prefix = dict(rows=list(range(nc)), pos=pos_blk) if nc else None
    return Spec(live, pos_pre + pos_post, ctx + slots, prefix=prefix, D=bool(D), X=bool(X), npre=npre), dict(live=nl, prefix=nc, npre=npre)


def emit(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')


for qset in QS:
    for T in TS:
        for D, C, V, Q, X in itertools.product((0, 1), repeat=5):
            if Q and not any(q.get('deployed') for q in CFG['qsets'][qset]): continue      # Q has no effect: same graph as Q=0
            key = f'{PREC}|{qset}|{T}|D{D}C{C}V{V}Q{Q}X{X}'
            if key in done: continue
            torch.manual_seed(hash(key) % 2 ** 31)
            spec, geo = build_spec(qset, T, D, C, V, Q, X)
            r = Req(rt, spec, lib=libs[bool(D)] if C else None)
            rec = dict(key=key, prec=PREC, qset=qset, T=T, D=D, C=C, V=V, Q=Q, X=X, share=SHARE if C else 0.0, rows=dict(geo, branch=r.R, total=r.T))
            try:
                with torch.inference_mode():
                    pool = S.fresh_pool(r, REPS + WARM, hi_ext=(Vbase + 16384) if V else None)
                    if not X:
                        g, out = S.capture(r.run_full); rec['full'] = S.wall_full(r, g, out, pool, WARM)
                        if PROF: rec['kern'] = S.kprof(g)
                        del g, out
                    else:
                        ga, oa = S.capture(r.run_a); rec['A'] = S.wall_full(r, ga, oa, pool, WARM)
                        gb, ob = S.capture(r.run_b); rec['B'] = S.wall_full(r, gb, ob, pool, WARM)
                        rec['AB'] = S.wall_casc(r, ga, oa, gb, ob, pool, lambda mg: True, WARM)        # never exits: A + host check + B
                        if PROF: rec['kernA'] = S.kprof(ga); rec['kernB'] = S.kprof(gb)
                        del ga, oa, gb, ob
            except torch.cuda.OutOfMemoryError as e:
                rec['error'] = 'OOM'
            rec['mem_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            emit(rec)
            del r; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print('GRID DONE', PREC, flush=True)
