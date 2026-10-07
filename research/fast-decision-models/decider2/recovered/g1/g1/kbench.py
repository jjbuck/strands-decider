"""G1 kernel bench (A10G): dense cuBLAS vs BTT/Monarch (bmm+permute reference, Triton stage-1 with permuted store + cuBLAS/Triton stage 2),
bf16 and fp16-accumulate, at hobson's projection shapes, M = 1000 / 4000 rows.
Timing: CUDA graph of NCALL back-to-back calls over rotating fresh inputs; 3 warm + REPS timed replays (CUDA events, synced);
per-call time = replay / NCALL; median and p95 over replays.
usage: python kbench.py OUT.jsonl [quick]"""
import sys, os, json, math, time, statistics
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btt as B

dev = 'cuda'
NCALL, REPS, NBUF = 20, 15, 4
out_path = sys.argv[1]
quick = len(sys.argv) > 2 and sys.argv[2] == 'quick'
fo = open(out_path, 'a')


def timeit(fn, xs):
    """fn(x) for x in rotating buffers inside one CUDA graph."""
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for x in xs: fn(x)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in range(NCALL): fn(xs[c % len(xs)])
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000 / NCALL)
    ts.sort()
    return statistics.median(ts), ts[int(math.ceil(0.95 * len(ts))) - 1]


def emit(**kw):
    fo.write(json.dumps(kw) + '\n'); fo.flush()
    print(' '.join(f'{k}={v}' for k, v in kw.items()), flush=True)


SHAPES = [('mlp_gate_up', 2048, 12288, 2), ('mlp_up', 2048, 6144, 1), ('mlp_down', 6144, 2048, 1), ('gdn_in', 2048, 8192, 1),
          ('attn_in', 2048, 5120, 1), ('out_proj', 2048, 2048, 1)]
if quick: SHAPES = SHAPES[:3]
MS = [1000, 4000]
BS = [1, 2, 4, 8, 16]
FRACS = [0.5, 0.25, 0.125]


def run_shape(name, n_in, n_out, bo_mult, M):
    torch.manual_seed(0)
    res = {}
    for dt in ('bf16', 'fp16acc'):
        dtype = torch.bfloat16 if dt == 'bf16' else torch.float16
        torch.backends.cuda.matmul.allow_fp16_accumulation = (dt == 'fp16acc')
        xs = [torch.randn(M, n_in, device=dev, dtype=dtype) for _ in range(NBUF)]
        W = torch.randn(n_out, n_in, device=dev, dtype=dtype) / math.sqrt(n_in)
        Wt = W.t()
        md, p95 = timeit(lambda x: x @ Wt, xs)
        dflop = 2 * M * n_in * n_out
        emit(shape=name, M=M, dtype=dt, kind='dense', b=0, r=0, frac=1.0, variant='cublas', med_us=round(md, 2), p95_us=round(p95, 2),
             tflops=round(dflop / md / 1e6, 1))
        res[(dt, 'dense')] = md
        cfgs = [(b, f, None) for b in BS for f in FRACS] + [(b, None, min(n_in, n_out) // (b * b)) for b in (2, 4, 8, 16)]
        for b, f, rr in cfgs:
            b_in, b_out = b, b * bo_mult
            # FLOP fraction defined on the (per-matrix) BTT with b_in = b_out/bo_mult = b
            if rr is None:
                r = max(1, int(round(f * n_in * (n_out // bo_mult) / (b * (n_in + n_out // bo_mult)))))
                r = max(8, int(round(r / 8)) * 8) if r >= 8 else r
                kind = 'btt'
            else:
                r = rr; kind = 'monarch'
            if r < 1: continue
            mac = B.macs(n_in, n_out, b_in, b_out, r)
            fr = mac / (n_in * n_out)
            if fr >= 1.0: continue
            if b == 1:
                if kind == 'monarch': continue
                V = torch.randn(r, n_in, device=dev, dtype=dtype) / math.sqrt(n_in)
                U = torch.randn(n_out, r, device=dev, dtype=dtype) / math.sqrt(r)
                Vt, Ut = V.t(), U.t()
                md, p95 = timeit(lambda x: (x @ Vt) @ Ut, xs)
                emit(shape=name, M=M, dtype=dt, kind='lowrank', b=1, r=r, frac=round(fr, 4), variant='cublas2', med_us=round(md, 2),
                     p95_us=round(p95, 2), speedup=round(res[(dt, 'dense')] / md, 2), flop_x=round(1 / fr, 2))
                continue
            R, L = B.init_factors(n_in, n_out, b_in, b_out, r, dtype=dtype)
            # correctness vs reference
            yr = B.btt_ref(xs[0], R, L).float()
            for be in ('cublas', 'triton'):
                y = B.btt(xs[0], R, L, backend=be, acc16=(dt == 'fp16acc' and be == 'triton')).float()
                err = float((y - yr).norm() / yr.norm())
                if err > 2e-2: print('ERR', name, b, r, be, err, flush=True)
            variants = [('bmm_permute', lambda x: B.btt_ref(x, R, L)),
                        ('tri1_cublas2', lambda x: B.btt(x, R, L, backend='cublas', acc16=(dt == 'fp16acc'))),
                        ('tri1_tri2', lambda x: B.btt(x, R, L, backend='triton', acc16=(dt == 'fp16acc')))]
            if dt == 'fp16acc': variants = variants[1:]
            for vn, fn in variants:
                md, p95 = timeit(fn, xs)
                emit(shape=name, M=M, dtype=dt, kind=kind, b=b, r=r, frac=round(fr, 4), variant=vn, med_us=round(md, 2), p95_us=round(p95, 2),
                     speedup=round(res[(dt, 'dense')] / md, 2), flop_x=round(1 / fr, 2))
            del R, L
        torch.backends.cuda.matmul.allow_fp16_accumulation = False


t0 = time.time()
for name, n_in, n_out, bm in SHAPES:
    for M in MS:
        run_shape(name, n_in, n_out, bm, M)
print('done', round(time.time() - t0), 's', flush=True)
