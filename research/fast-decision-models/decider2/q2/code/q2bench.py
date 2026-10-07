"""Q2 end-to-end latency (A10G, exclusive GPU): CUDA-graph replay of the full forward + pointer head, hobson layout, fresh random state ids
per rep (H2D + replay + D2H), 3 warm + REPS timed, median and p95 (ms). Bundles from H2's prep_q.py (1q = details_match 125 tok, 15q = 3720 tok).
python q2bench.py NAMES TS BUNDLES [reps]     NAMES from: b8 (H2 deployed W8A8-GPTQ-b8 map), w4a4 (H2 all int4), w8a8 (H2 all int8),
                                              w4q8 (Q2 row-role), w8q2 (Q2 kernels, all int8)"""
import os, sys, json, time, statistics as st, collections, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'),
                os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import qrt as Q
import q2rt as R
from torch.profiler import profile, ProfilerActivity
OUT = os.path.expanduser(os.environ.get('Q2E2E', '~/work/q2/res_e2e.jsonl'))
CODES = os.environ.get('CODES', 'w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt')


def capture(fn, warm=3):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warm): out = fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    torch.cuda.synchronize()
    return g, out


def kcat(n):
    nl = n.lower()
    if 'flash' in nl or 'fmha' in nl or 'attention' in nl or 'efficient' in nl: return 'attn'
    if 'k_one' in nl or 'k_two' in nl or 'cutlass' in nl or 'gemm' in nl or 'cublas' in nl or 'sm80_xmma' in nl or 'ampere_' in nl: return 'gemm'
    if 'chunk' in nl or 'recompute_w_u' in nl or 'kkt' in nl or 'solve' in nl or 'fwd_kernel_h' in nl: return 'gdn_core'
    if any(k in nl for k in ('_swiglu_k', '_gnorm_k', '_agate_k', '_addq_k', '_conv_k', '_aprep_k')): return 'glue'
    return 'other'


def kprof(g):
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    agg = collections.defaultdict(float); sub = collections.defaultdict(float); n = 0
    for e in p.events():
        if e.device_type.name != 'CUDA': continue
        dt = (e.device_time if hasattr(e, 'device_time') else e.cuda_time) / 1000
        c = kcat(e.name); agg[c] += dt; n += 1
        if c == 'gemm':
            nl = e.name.lower()
            sub['gemm_q2' if ('k_one' in nl or 'k_two' in nl) else ('gemm_cut' if 'cutlass' in nl else 'gemm_other')] += dt
    r = {k: round(v, 3) for k, v in agg.items()}; r['kernels'] = n; r['busy'] = round(sum(agg.values()), 3)
    r.update({k: round(v, 3) for k, v in sub.items()})
    return r


class Req:
    def __init__(self, m, Ts, qs, dev='cuda'):
        self.m = m; self.Ts = Ts; Mq = len(qs)
        if hasattr(m, 'pos'): m.pos = None          # B8 partition is per request shape
        Kmax = max(len(q['opt']) for q in qs)
        self.temps = torch.tensor([q['temp'] for q in qs], device=dev)
        self.nsl = torch.tensor([q['n_slots'] for q in qs], device=dev)
        self.lay = Q.Lay('single', Ts + len(qs[0]['ids'])) if Mq == 1 else Q.Lay('packed', Ts, qlens=[len(q['ids']) for q in qs])
        tail = [t for q in qs for t in q['ids']]
        pr = []; orow = []; s = Ts
        for q in qs:
            pr.append(s + len(q['ids']) - 1)
            orow.append([s + j for j in q['opt']] + [s + q['opt'][0]] * (Kmax - len(q['opt']))); s += len(q['ids'])
        self.pool_rows = torch.tensor(pr, device=dev); self.opt_rows = torch.tensor(orow, device=dev)
        self.ids = torch.cat([torch.randint(1000, 100000, (Ts,), device=dev), torch.tensor(tail, device=dev)])
        self.kmask = torch.arange(Kmax, device=dev)[None, :] < self.nsl[:, None]

    @torch.no_grad()
    def run(self):
        m = self.m
        m.q0 = self.Ts
        if getattr(m, 'x8frac', 0) > 0:        # B8 timing: a fixed number of random state rows in int8 (fresh positions per replay)
            T = self.lay.T; n8s = int(round(m.x8frac * self.Ts))
            r = torch.rand(self.Ts, device=self.ids.device)
            th = torch.topk(r, n8s).values[-1] if n8s > 0 else 2.0
            is8 = torch.cat([r >= th, torch.ones(T - self.Ts, dtype=torch.bool, device=r.device)])
            m.set_rowmask(is8) if m.pos is None else m._upd_mask(is8)
        hn = m.forward(self.ids, self.lay, None)
        dec = m.unrot(hn[self.pool_rows]).float()
        opts = m.unrot(hn[self.opt_rows.reshape(-1)]).float().reshape(self.opt_rows.shape[0], self.opt_rows.shape[1], -1)
        lg = m.head(dec, opts) / self.temps[:, None]
        lg = lg.masked_fill(~self.kmask, float('-inf'))
        return torch.softmax(lg, -1)


def wall(req, g, out, reps=20, warm=3):
    Ts = req.Ts
    pool = [torch.randint(1000, 100000, (Ts,), dtype=torch.long).pin_memory() for _ in range(reps + warm)]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory()
    ts = []
    for x in pool:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        req.ids[:Ts].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[warm:])
    return dict(median=round(st.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3), n=len(ts))


def load_codes(spec):
    if not spec: return None
    return {kv.split('=')[0]: torch.load(os.path.expanduser(kv.split('=')[1])) for kv in spec.split(',')}


def build(name, ln, head, wc):
    if name == 'b8': m = Q.QRT(ln, head=head, prec='map:~/work/h1/precmap_w8a8_b8.json:w8a8', wcodes=wc)
    elif name in ('w4a4', 'w8a8'): m = Q.QRT(ln, head=head, prec=name, wcodes=wc)
    elif name == 'w4q8': m = R.QRT2(ln, head=head, fmt='w4q8', wcodes=wc)
    elif name == 'w8q2': m = R.QRT2(ln, head=head, fmt='w8', wcodes=wc)
    elif name == 'w4q2': m = R.QRT2(ln, head=head, fmt='w4', wcodes=wc)
    elif name.startswith('rrm'): m = R.QRT2(ln, head=head, fmt='w4q8', wcodes=wc); m.x8frac = float(name[3:])
    elif name == 'w4q8c': m = R.QRT2C(ln, head=head, fmt='w4q8', wcodes=wc)
    elif name.startswith('map:'): m = R.QRT2(ln, head=head, fmt=name, wcodes=wc)
    elif name.startswith('cmap:'): m = R.QRT2C(ln, head=head, fmt=name[1:], wcodes=wc)
    else: raise ValueError(name)
    m.tune = True
    return m


if __name__ == '__main__':
    names = sys.argv[1].split(','); Tss = [int(x) for x in sys.argv[2].split(',')]; bnames = sys.argv[3].split(',')
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT): done.add(json.loads(l)['key'])
    bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt'))
    from kitrun import load_P
    from lean2 import Lean2
    P = load_P()
    ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval()
    wc = load_codes(CODES)
    for name in names:
        if all(f'{name}|{T}|{b}' in done for T in Tss for b in bnames): continue
        m = build(name, ln, head, wc)
        for T in Tss:
            for bn in bnames:
                key = f'{name}|{T}|{bn}'
                if key in done: continue
                req = Req(m, T, bundles[bn])
                g, out = capture(req.run)
                w = wall(req, g, out, reps=reps)
                kp = kprof(g)
                rec = dict(key=key, name=name, T=T, bundle=bn, rows=req.lay.T, q0=T, wall=w, kern=kp, mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                print(json.dumps(rec), flush=True)
                with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')
                del g, out, req; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        del m; torch.cuda.empty_cache()
