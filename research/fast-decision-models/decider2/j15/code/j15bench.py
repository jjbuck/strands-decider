"""J15 latency (A10G, exclusive GPU): CUDA-graph replay of full / prefix / suffix forwards of the deployed H2 kernels, hobson layout,
state of exactly T tokens + one real question (details_match, 125 tokens, = H2's 1q), fresh random state ids per rep (H2D + replay + D2H).

python j15bench.py grid  PRECNAME=PREC[,..] TS [modes]   modes: full | pre:K | post:K  (K = layer boundary)  -> res_bench.jsonl
python j15bench.py casc  DRAFT VERIFIER TAU N            real requests (REAL-agree / LONG / JB-all sample): draft graph -> host margin -> verifier graph if margin < tau
"""
import sys, os, json, time, statistics as st, random, torch
sys.path[:0] = [os.path.expanduser('~/work/j15'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import qrt as Q
from j15run import QRTJ, load_codes
import collections
from torch.profiler import profile, ProfilerActivity


def kcat(n):
    nl = n.lower()
    if 'flash' in nl or 'fmha' in nl or 'attention' in nl or 'efficient' in nl: return 'attn'
    if 'cutlass' in nl or '_gemm_k' in nl or 'gemm' in nl or 'cublas' in nl or 'sm80_xmma' in nl or 'ampere_' in nl or '_tgemm' in nl or '_mm_k' in nl: return 'gemm'
    if 'chunk' in nl or 'recompute_w_u' in nl or 'kkt' in nl or 'solve' in nl or 'fwd_kernel_h' in nl: return 'gdn_core'
    if any(k in nl for k in ('_swiglu_k', '_gnorm_k', '_agate_k', '_addq_k', '_conv_k', '_aprep_k')): return 'glue'
    return 'other'


def kprof(g):
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    agg = collections.defaultdict(float); n = 0
    for e in p.events():
        if e.device_type.name != 'CUDA': continue
        dt = (e.device_time if hasattr(e, 'device_time') else e.cuda_time) / 1000
        agg[kcat(e.name)] += dt; n += 1
    r = {k: round(v, 3) for k, v in agg.items()}; r['kernels'] = n; r['busy'] = round(sum(agg.values()), 3)
    r['nongemm'] = round(r['busy'] - r.get('gemm', 0.0), 3)
    return r
C = 'w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt'
PRECS = {'bf16': 'bf16', 'b8': 'map:~/work/h1/precmap_w8a8_b8.json:w8a8', 'w4a4': 'w4a4', 'k48': 'map:~/work/h1/precmap_w4a4_k48.json:w4a4',
         'k64': 'map:~/work/h6/precmap_w4a4_k64.json:w4a4', 'w8a8': 'w8a8'}
OUT = os.path.expanduser(os.environ.get('J15OUT', '~/work/j15/res_bench.jsonl'))


def capture(fn, warm=3):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm): out = fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    torch.cuda.synchronize()
    return g, out


class ExitHead(torch.nn.Module):     # timing stand-in with the trained exit head's shape (LayerNorm + q/k 2048->256 + unrotation)
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(2048); self.q = torch.nn.Linear(2048, 256); self.k = torch.nn.Linear(2048, 256)
    def forward(self, d, o):
        return (self.k(self.norm(o)) @ self.q(self.norm(d)).unsqueeze(-1)).squeeze(-1) * 256 ** -0.5


class Req:
    def __init__(self, m, ids_q, opt, q0_off, Ts, temp, mode, dev='cuda', ids_state=None):
        self.m = m; self.Ts = Ts; self.mode = mode
        T = Ts + len(ids_q); self.T = T
        self.lay = Q.Lay('single', T)
        st_ids = torch.randint(1000, 100000, (Ts,), device=dev) if ids_state is None else torch.tensor(ids_state, device=dev)
        self.ids = torch.cat([st_ids, torch.tensor(ids_q, device=dev)])
        self.rows = torch.tensor([T - 1] + [Ts + o for o in opt], device=dev)
        self.temp = temp
        self.x0 = torch.randn(T, 2048, device=dev).to(torch.bfloat16) * 0.1
        self.eh = ExitHead().to(dev).float()

    def run(self):
        m = self.m
        if self.mode == 'full':
            hn = m.forward(self.ids, self.lay)
        elif self.mode.startswith('pre:'):
            k = int(self.mode[4:]); m.forward(self.ids, self.lay, i1=k)
            h = m.unrot_raw(m.xres[self.rows])
            lg = self.eh(h[:1], h[1:][None]) / self.temp
            return torch.softmax(lg, -1)
        else:
            k = int(self.mode[5:]); x = self.x0.clone(); hn = m.forward(self.ids, self.lay, x0=x, i0=k)
        h = m.unrot(hn[self.rows]).float()
        lg = m.head(h[:1], h[1:][None]) / self.temp
        return torch.softmax(lg, -1)


def wall(req, g, out, reps=20, warm=3, fresh=True):
    Ts = req.Ts
    pool = [torch.randint(1000, 100000, (Ts,), dtype=torch.long).pin_memory() for _ in range(reps + warm)]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory()
    ts = []
    for x in pool:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if fresh: req.ids[:Ts].copy_(x, non_blocking=True)
        g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[warm:])
    return dict(median=round(st.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3), n=len(ts))


def build(ln, head, name):
    m = QRTJ(ln, head=head, prec=PRECS[name], wcodes=load_codes(C) if name != 'bf16' else None); m.tune = True
    return m


def load_ln(fold):
    from kitrun import load_P
    from lean2 import Lean2
    P = load_P()
    ln = Lean2(P.tm, fuse='fold' if fold else ''); head = P.model.head.float().eval()
    return P, ln, head


if __name__ == '__main__':
    cmd = sys.argv[1]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT): done.add(json.loads(l)['key'])
    bq = torch.load(os.path.expanduser('~/work/j15/q1.pt'))      # details_match: ids, opt, temp
    if cmd == 'grid':
        names = sys.argv[2].split(','); Tss = [int(x) for x in sys.argv[3].split(',')]; modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['full']
        P, ln, head = load_ln('bf16' in names)
        for name in names:
            keys = [f'{name}|{T}|{md}' for T in Tss for md in modes]
            if all(k in done for k in keys): continue
            m = build(ln, head, name)
            for T in Tss:
                for md in modes:
                    key = f'{name}|{T}|{md}'
                    if key in done: continue
                    req = Req(m, bq['ids'], bq['opt'], 0, T, bq['temp'], md)
                    with torch.inference_mode():
                        g, out = capture(req.run)
                        w = wall(req, g, out)
                        kp = kprof(g) if (T in (256, 1000, 4000) and md == 'full') else None
                    rec = dict(key=key, prec=name, T=T, mode=md, rows=req.T, wall=w, kern=kp, mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                    print(json.dumps(rec), flush=True)
                    with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')
                    del g, out, req; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            del m; torch.cuda.empty_cache()
    elif cmd == 'casc':
        # end-to-end cascade on real requests at their exact lengths
        import evalkit as EK
        from kitrun import prep_question
        dname, vname, tau, N = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
        tag = sys.argv[6] if len(sys.argv) > 6 else ''
        P, ln, head = load_ln(False)
        D = build(ln, head, dname); V = build(ln, head, vname)
        rng = random.Random(0)
        items = []
        for s, share in (('REAL-agree', 0.45), ('LONG', 0.15), ('JB-all', 0.40)):
            its = [(s, it, q) for it in EK.load_suite(s) for q in it['questions']]
            rng.shuffle(its); items += its[:int(N * share)]
        outp = os.path.expanduser(f'~/work/j15/res_casc{tag}.jsonl')
        for s, it, q in items:
            pr = prep_question(P, it, q)
            Ts = len(pr['s'])
            with torch.inference_mode():
                rd = Req(D, pr['q'], pr['opt'], 0, Ts, P.temp_for(pr['rq'].kind), 'full', ids_state=pr['s'])
                rv = Req(V, pr['q'], pr['opt'], 0, Ts, P.temp_for(pr['rq'].kind), 'full', ids_state=pr['s'])
                gd, od = capture(rd.run); gv, ov = capture(rv.run)
                hd = torch.empty(od.shape, dtype=od.dtype).pin_memory(); hv = torch.empty(ov.shape, dtype=ov.dtype).pin_memory()
                tsc = []; tsd = []; tsv = []; nd = 0
                for r in range(15):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    gd.replay(); hd.copy_(od, non_blocking=True); torch.cuda.synchronize()
                    p = sorted(hd[0, :pr['rq'].n_slots].tolist(), reverse=True); mg = p[0] - (p[1] if len(p) > 1 else 0)
                    t1 = time.perf_counter()
                    if mg < tau:
                        gv.replay(); hv.copy_(ov, non_blocking=True); torch.cuda.synchronize(); nd += 1
                    t2 = time.perf_counter()
                    tsc.append((t2 - t0) * 1000); tsd.append((t1 - t0) * 1000)
                for r in range(8):
                    torch.cuda.synchronize(); t0 = time.perf_counter(); gv.replay(); hv.copy_(ov, non_blocking=True); torch.cuda.synchronize(); tsv.append((time.perf_counter() - t0) * 1000)
            rec = dict(suite=s, id=it['id'], q=q, T=Ts + len(pr['q']), margin=round(mg, 4), deferred=mg < tau,
                       casc=round(st.median(tsc[3:]), 3), draft=round(st.median(tsd[3:]), 3), verifier=round(st.median(tsv[2:]), 3))
            print(json.dumps(rec), flush=True)
            with open(outp, 'a') as f: f.write(json.dumps(rec) + '\n')
            del gd, gv, od, ov, rd, rv; torch.cuda.empty_cache()


def dcasc_main():
    """end-to-end depth-speculative cascade on real requests at exact lengths: segment graphs [0,L1) + exit head L1, [L1,L2) + head L2, ...,
    [Lk,24) + hobson head; host checks each exit margin against its DEV-fitted tau. python j15bench.py dcasc L1:tau1,L2:tau2 N [tag]"""
    import evalkit as EK
    from kitrun import prep_question
    import torch.nn.functional as F
    from j15learn import EH
    spec = [(int(a.split(':')[0]), float(a.split(':')[1])) for a in sys.argv[2].split(',')]; N = int(sys.argv[3]); tag = sys.argv[4] if len(sys.argv) > 4 else ''
    P, ln, head = load_ln(False)
    m = build(ln, head, 'b8')
    heads = {}
    for L, _ in spec:
        import copy
        h = EH(copy.deepcopy(P.model.head).float().cpu()); h.load_state_dict(torch.load(os.path.expanduser(f'~/work/j15/preds/exithead_L{L}.pt'))); heads[L] = h.cuda().eval()
    bounds = [0] + [L for L, _ in spec] + [24]
    rng = random.Random(0); items = []
    for s, share in (('REAL-agree', 0.45), ('LONG', 0.15), ('JB-all', 0.40)):
        its = [(s, it, q) for it in EK.load_suite(s) for q in it['questions']]
        rng.shuffle(its); items += its[:int(N * share)]
    outp = os.path.expanduser(f'~/work/j15/res_dcasc{tag}.jsonl')
    for s, it, q in items:
        pr = prep_question(P, it, q); Ts = len(pr['s']); T = Ts + len(pr['q'])
        lay = Q.Lay('single', T); ids = torch.tensor(pr['s'] + pr['q'], device='cuda')
        rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda'); temp = P.temp_for(pr['rq'].kind); n = pr['rq'].n_slots
        xbuf = torch.empty(T, 2048, device='cuda', dtype=torch.bfloat16)
        segs = []
        with torch.inference_mode():
            for j in range(len(bounds) - 1):
                a, b = bounds[j], bounds[j + 1]
                def fn(a=a, b=b):
                    if a == 0: xbuf.copy_(F.embedding(ids, m.embed))
                    hn = m.forward(ids, lay, x0=xbuf, i0=a, i1=b)
                    if b < 24:
                        h = m.unrot_raw(xbuf[rows]); lg = heads[b](h[:1], h[1:][None]) / temp
                    else:
                        h = m.unrot(hn[rows]).float(); lg = m.head(h[:1], h[1:][None]) / temp
                    return torch.softmax(lg, -1)
                segs.append(capture(fn))
            gf, of = capture(lambda: (xbuf.copy_(F.embedding(ids, m.embed)), m.unrot(m.forward(ids, lay, x0=xbuf)[rows]).float())[1])
            hosts = [torch.empty(o.shape, dtype=o.dtype).pin_memory() for _, o in segs]
            tsc = []; tsf = []; ex = None
            for r in range(15):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                for j, (g, o) in enumerate(segs):
                    g.replay(); hosts[j].copy_(o, non_blocking=True); torch.cuda.synchronize()
                    if j < len(spec):
                        p = sorted(hosts[j][0, :n].tolist(), reverse=True); mg = p[0] - (p[1] if len(p) > 1 else 0)
                        if mg >= spec[j][1]: ex = bounds[j + 1]; break
                    else: ex = 24
                tsc.append((time.perf_counter() - t0) * 1000)
            for r in range(10):
                torch.cuda.synchronize(); t0 = time.perf_counter(); gf.replay(); torch.cuda.synchronize(); tsf.append((time.perf_counter() - t0) * 1000)
        rec = dict(suite=s, id=it['id'], q=q, T=T, exit=ex, casc=round(st.median(tsc[3:]), 3), b8=round(st.median(tsf[2:]), 3))
        print(json.dumps(rec), flush=True)
        with open(outp, 'a') as f: f.write(json.dumps(rec) + '\n')
        del segs, gf, of, xbuf; torch.cuda.empty_cache()


if __name__ == '__main__' and sys.argv[1] == 'dcasc':
    dcasc_main()
