"""Step 4 latency: dense lean2 (all fusions, fold) vs mixed-width layers, full 24 layers, CUDA graph, exact lengths, 1 question.
Request = T state tokens + 125 question tokens (full width).  Full rows = question + 4 sinks + round(f * n_chunks) random 32-token
chunks; the rest thin in layers >= k.  Wall = H2D fresh ids (pinned) + graph replay + D2H probs, 3 warm + NREP timed, median/p95.
python lat.py check            correctness: mixed vs tw.py thin path on real questions (same thin rows), dense vs dense
python lat.py bench OUT.json [Ts] [cfgs]"""
import os, sys, json, time, math, random, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
from plib import P
from lean2 import Lean2
from mixed import Mixed
FUSE = 'addrms,gnorm,silu,prep,conv,gemm_swiglu,fold'
QN = 125
NREP = int(os.environ.get('NREP', '20'))


def make_perm(Ts, T, f, seed, dev):
    ch = [(s, min(s + 32, Ts)) for s in range(0, Ts, 32)]
    m = int(math.floor(f * len(ch) + 0.5))
    rng = random.Random(seed); sel = sorted(rng.sample(range(len(ch)), m)) if m else []
    full = set(range(min(4, Ts))) | set(range(Ts, T))
    for j in sel: full.update(range(*ch[j]))
    fl = sorted(full); th = [t for t in range(T) if t not in full]
    perm = torch.tensor(fl + th, device=dev)
    inv = torch.empty(T, dtype=torch.long); inv[torch.tensor(fl + th)] = torch.arange(T)
    return perm, len(fl), th, inv.to(dev)


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    return g, out


def wall(g, out, ids_buf, Ts, reps=NREP, warm=3):
    pool = [torch.randint(1000, 100000, (Ts,), dtype=torch.long).pin_memory() for _ in range(reps + warm)]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory()
    ts = []; gs = []
    for x in pool:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids_buf[:Ts].copy_(x, non_blocking=True)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record()
        host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000); gs.append(e0.elapsed_time(e1))
    ts = sorted(ts[warm:]); gs = sorted(gs[warm:])
    q = lambda v, p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]
    return dict(median=round(st.median(ts), 3), p95=round(q(ts, .95), 3), gpu_median=round(st.median(gs), 3), n=len(ts))


def main():
    mode = sys.argv[1]
    p = P(); p.model.config.max_length = 16384
    R = Lean2(p.tm, fuse=FUSE); dev = R.dev; head = p.model.head
    for prm in p.model.parameters(): prm.requires_grad_(False)
    if mode == 'check':
        from tw import TW
        import evalkit as EK
        bz = torch.load(os.path.expanduser('~/work/j13/bases.pt'), map_location='cpu')['out']
        tw = TW(bases=os.path.expanduser('~/work/j13/bases.pt'), p=p, modes=('out',))
        res = []
        its = [it for it in EK.load_suite('REAL-agree') if 500 < it['n_state_tok'] < 2500][:6]
        for k, r in ((4, 256), (8, 512)):
            mx = Mixed(R, k, r, U=bz)
            for it in its:
                qn = next(iter(it['questions']))
                pr = tw.prep(it, qn); q0 = pr['q0']; T = len(pr['ids'])
                ids = torch.tensor(pr['ids'], device=dev)
                perm, Mf, th, inv = make_perm(q0, T, 0.3, 7, dev)
                with torch.no_grad():
                    x, cos, sin = tw.embed_rope(pr['ids'])
                    pl = tw.thin_plan('out', r, th, T)
                    xo, _ = tw.run(x, cos, sin, plans={l: pl for l in range(k, tw.NL)})
                    p_tw = tw.head(xo, pr).exp()
                    xd, _ = tw.run(x, cos, sin); p_twd = tw.head(xd, pr).exp()
                    rows = pr['opt_abs'] + [T - 1]
                    def hd(h):
                        lg = head(h[-1:].float(), h[:-1].float()[None]) / p.temp_for(pr['rq'].kind)
                        return torch.softmax(lg[0, :pr['rq'].n_slots], -1)
                    mx.forward(ids, perm, Mf, srows=inv[rows])          # tuning pass (garbage)
                    p_mx = hd(mx.forward(ids, perm, Mf, srows=inv[rows]))
                    p_ld = hd(mx.forward(ids, rows=torch.tensor(rows, device=dev)))
                    pid_ = torch.arange(T, device=dev)
                    p_id = hd(mx.forward(ids, pid_, T, srows=torch.tensor(rows, device=dev)))
                tv = lambda a, b: float(0.5 * (a - b).abs().sum())
                rec = dict(k=k, r=r, T=T, Mf=Mf, tv_mixed_vs_tw=tv(p_mx, p_tw), tv_dense_lean2_vs_tw=tv(p_ld, p_twd),
                           tv_identityperm_vs_dense=tv(p_id, p_ld), tv_thin_vs_dense_tw=tv(p_tw, p_twd),
                           same_argmax=int(p_mx.argmax() == p_tw.argmax()))
                print(json.dumps(rec), flush=True); res.append(rec)
        json.dump(res, open(os.path.expanduser('~/work/j13/lat_check.json'), 'w'), indent=1)
        return
    out = sys.argv[2]
    Ts_list = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else '256,1000,4000').split(',')]
    cfgs = (sys.argv[4] if len(sys.argv) > 4 else 'dense,4:256:0.3,4:512:0.3,8:128:0.3,4:256:0.5,4:128:0.5,4:256:0.0').split(',')
    res = json.load(open(out)) if os.path.exists(out) else {}
    mxs = {}
    with torch.no_grad():
        for Ts in Ts_list:
            T = Ts + QN
            ids_buf = torch.randint(1000, 100000, (T,), device=dev)
            rows = torch.tensor([T - 9, T - 5, T - 1], device=dev)
            for c in cfgs:
                key = f'{c}|{Ts}'
                if key in res: continue
                if c == 'dense':
                    mx = mxs.setdefault((4, 256), Mixed(R, 4, 256))
                    fn = lambda: torch.softmax(head(*(lambda h: (h[-1:].float(), h[:-1].float()[None]))(mx.forward(ids_buf, rows=rows))), -1)
                    rec = dict(cfg=c, Ts=Ts, T=T, Mf=T, flops=1.0)
                else:
                    parts = c.split(':'); k, r, f = int(parts[0]), int(parts[1]), float(parts[2])
                    segs = [(k, r)] + [tuple(int(v) for v in sg.split('r')) for sg in parts[3:]]   # e.g. 4:512:0.3:13r64
                    mx = mxs.setdefault(tuple(segs), Mixed(R, k, r, segs=segs))
                    perm, Mf, th, inv = make_perm(Ts, T, f, 11, dev)
                    srows = inv[rows]
                    fn = lambda: torch.softmax(head(*(lambda h: (h[-1:].float(), h[:-1].float()[None]))(mx.forward(ids_buf, perm, Mf, srows=srows))), -1)
                    fn()   # tune
                    nl = len(R.layers)
                    dm = [sum(a.numel() for a in (R.layers[i]['Win'], R.layers[i]['Wo'], R.layers[i]['Wgu'], R.layers[i]['Wd'])) for i in range(nl)]
                    tm = [(mx.rl.get(i, r)) * sum(sum(R.layers[i][w].shape) for w in ('Win', 'Wo', 'Wgu', 'Wd')) for i in range(nl)]
                    fl = sum(T * dm[i] if i < k else Mf * dm[i] + (T - Mf) * tm[i] for i in range(nl)) / (T * sum(dm))
                    rec = dict(cfg=c, Ts=Ts, T=T, Mf=Mf, flops=round(fl, 4))
                g, o = capture(fn)
                rec.update(wall(g, o, ids_buf, Ts))
                del g, o; torch.cuda.empty_cache()
                res[key] = rec; print(json.dumps(rec), flush=True)
                json.dump(res, open(out, 'w'), indent=1)
    print('done')


if __name__ == '__main__':
    main()
