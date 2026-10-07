"""H2 end-to-end latency matrix on the A10G (exclusive GPU).
python bench.py PRECS TS BUNDLES MODES [tag]     e.g.  python bench.py bf16,w8a8,w4a8,w4a4 1000,4000 1q,4q,15q plain,schema
Per config: CUDA graph of [embed .. 24 layers .. final norm .. pointer head .. softmax] for every question; wall time per request =
H2D of fresh random state ids (pinned) + graph replay + D2H of the answer probabilities, 3 warm-up + 20 timed reps, median and p95.
plain  = hobson's layout: state then question; 1 question = one sequence, >1 = shared state + one branch per question (packed).
schema = compiled schema (this-that-model schema-first): [question bundle P: cached once][state][N one-token answer slots].
Results appended to res_bench<tag>.jsonl (resumable); one profiled replay per config gives the GEMM / non-GEMM kernel split."""
import sys, os, json, time, glob, statistics as st, collections, torch
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
from prof_d1 import capture
from lean2 import Lean2
import qrt as Q
from torch.profiler import profile, ProfilerActivity

def load_model():
    from strands_decider.modeling import StrandsDeciderModel
    from common import CKPT
    m = StrandsDeciderModel.load(CKPT)
    t = m.torso.merge_and_unload().eval().cuda().to(torch.bfloat16)
    head = m.head.cuda().float().eval()
    return t, head

def kcat(n):
    nl = n.lower()
    if 'flash' in nl or 'fmha' in nl or 'attention' in nl or 'efficient' in nl: return 'attn'
    if 'cutlass' in nl or '_gemm_k' in nl or 'gemm' in nl or 'cublas' in nl or 'sm80_xmma' in nl or 'ampere_' in nl: return 'gemm'
    if 'flash' in nl or 'fmha' in nl or 'attention' in nl: return 'attn'
    if 'chunk' in nl or 'recompute_w_u' in nl or 'kkt' in nl or 'solve' in nl or 'fwd_kernel_h' in nl: return 'gdn_core'
    if any(k in nl for k in ('_swiglu_k', '_gnorm_k', '_agate_k', '_addq_k', '_conv_k', '_aprep_k')): return 'glue'
    return 'other'

class Req:
    """one request type: layout, static ids, readout indices"""
    def __init__(self, m, Ts, qs, mode, dev='cuda'):
        self.m = m; self.Ts = Ts; self.qs = qs; self.mode = mode; Mq = len(qs)
        Kmax = max(len(q['opt']) for q in qs)
        self.temps = torch.tensor([q['temp'] for q in qs], device=dev)
        self.nsl = torch.tensor([q['n_slots'] for q in qs], device=dev)
        if mode == 'schema':
            pre = [t for q in qs for t in q['ids']]
            self.P = len(pre)
            self.cache = m.compile_prefix(torch.tensor(pre, device=dev), Ts + Mq)
            self.lay = Q.Lay('schema', Ts, nslots=Mq, P=self.P)
            tail = [q['ids'][-1] for q in qs]                         # one answer-slot token per question
            self.pool_rows = torch.arange(Ts, Ts + Mq, device=dev)
            offs = []; o = 0
            for q in qs:
                offs.append([o + j for j in q['opt']] + [o + q['opt'][0]] * (Kmax - len(q['opt']))); o += len(q['ids'])
            self.opt_rows = torch.tensor(offs, device=dev)            # rows of the cached prefix hidden
            self.hP = self.cache['hP']
        else:
            self.cache = None; self.P = 0
            if Mq == 1:
                self.lay = Q.Lay('single', Ts + len(qs[0]['ids']))
            else:
                self.lay = Q.Lay('packed', Ts, qlens=[len(q['ids']) for q in qs])
            tail = [t for q in qs for t in q['ids']]
            pr = []; orow = []; s = Ts
            for q in qs:
                pr.append(s + len(q['ids']) - 1)
                orow.append([s + j for j in q['opt']] + [s + q['opt'][0]] * (Kmax - len(q['opt']))); s += len(q['ids'])
            self.pool_rows = torch.tensor(pr, device=dev); self.opt_rows = torch.tensor(orow, device=dev)
        self.ids = torch.cat([torch.randint(1000, 100000, (Ts,), device=dev), torch.tensor(tail, device=dev)])
        self.kmask = torch.arange(Kmax, device=dev)[None, :] < self.nsl[:, None]

    def run(self):
        m = self.m
        hn = m.forward(self.ids, self.lay, self.cache)
        dec = m.unrot(hn[self.pool_rows]).float()
        if self.mode == 'schema':
            opts = self.hP[self.opt_rows].float()
        else:
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

def kprof(g):
    for _ in range(2): g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        g.replay(); torch.cuda.synchronize()
    agg = collections.defaultdict(float); n = 0; byname = collections.defaultdict(float)
    for e in p.events():
        if e.device_type.name != 'CUDA': continue
        dt = (e.device_time if hasattr(e, 'device_time') else e.cuda_time) / 1000
        agg[kcat(e.name)] += dt; n += 1; byname[e.name[:70]] += dt
    r = {k: round(v, 3) for k, v in agg.items()}; r['kernels'] = n; r['busy'] = round(sum(agg.values()), 3)
    r['top'] = [(k, round(v, 3)) for k, v in sorted(byname.items(), key=lambda kv: -kv[1])[:14]]
    r['nongemm'] = round(r['busy'] - r.get('gemm', 0.0), 3)
    return r

if __name__ == '__main__':
    precs = sys.argv[1].split(';') if ';' in sys.argv[1] else sys.argv[1].split(','); Tss = [int(x) for x in sys.argv[2].split(',')]; bnames = sys.argv[3].split(','); modes = sys.argv[4].split(',')
    tag = sys.argv[5] if len(sys.argv) > 5 else ''
    out_path = os.path.expanduser(f'~/work/h2/res_bench{tag}.jsonl')
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path): done.add(json.loads(l)['key'])
    bundles = torch.load(os.path.expanduser('~/work/h2/bundles.pt'))
    torso, head = load_model()
    ln = Lean2(torso, fuse='fold'); del torso; torch.cuda.empty_cache()
    for prec in precs:
        pn0 = prec.split('/')[-1].replace('.json', '') if prec.startswith('map:') else prec
        if all(f'{pn0}|{T}|{b}|{md}' in done for T in Tss for b in bnames for md in modes): continue
        m = Q.QRT(ln, head=head, prec=prec, ohead=os.environ.get('OHEAD') == '1')
        pname = prec.split('/')[-1].replace('.json', '') if prec.startswith('map:') else prec
        for T in Tss:
            for bn in bnames:
                for md in modes:
                    key = f'{pname}|{T}|{bn}|{md}'
                    if key in done: continue
                    req = Req(m, T, bundles[bn], md)
                    g, out = capture(req.run)
                    w = wall(req, g, out)
                    kp = kprof(g)
                    rec = dict(key=key, prec=pname, T=T, bundle=bn, mode=md, rows=req.lay.T, P=req.P, nq=len(bundles[bn]), wall=w, kern=kp,
                               mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                    print(json.dumps(rec), flush=True)
                    with open(out_path, 'a') as f: f.write(json.dumps(rec) + '\n')
                    del g, out, req; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        del m; torch.cuda.empty_cache()
