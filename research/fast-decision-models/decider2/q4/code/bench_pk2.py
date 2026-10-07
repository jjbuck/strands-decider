"""Q4 B14 prototype v2 timing (q4pk2: all rows in one M tile, wide N tiles, split-K by int32 atomics; outputs int32): the 96 W8A8 GEMMs of one hobson-v19 forward (all int8; b8's 8 bf16 GEMMs counted as int8) at M rows,
(a) as 96 kernel launches captured in one CUDA graph (H2 CUTLASS s8 / J5 s8, best config per shape at this M), and
(b) as ONE persistent launch (q4pk.cu: resident CTAs, grid barrier between GEMMs, L2 prefetch of the next GEMM's first K stages).
Inputs are fixed int8 buffers (no glue between GEMMs): this isolates the GEMM side of the short-decision floor.
python bench_pk.py Ms -> ~/work/q4/res_pk.jsonl  (median / p95 of 30 replays, us)"""
import os, sys, json, ctypes, statistics, torch
sys.path[:0] = [os.path.expanduser('~/work/q4'), os.path.expanduser('~/work/h2')]
import qgemm as QG
dev = 'cuda'
OUT = os.path.expanduser(os.environ.get('Q4OUT', '~/work/q4/res_pk2.jsonl'))
BMN = {0: (160, 128), 1: (160, 64), 2: (256, 128), 3: (256, 64), 4: (128, 128), 5: (64, 128), 6: (160, 128)}


class GDesc(ctypes.Structure):
    _fields_ = [('A', ctypes.c_void_p), ('W', ctypes.c_void_p), ('out', ctypes.c_void_p), ('M', ctypes.c_int), ('N', ctypes.c_int), ('K', ctypes.c_int),
                ('lda', ctypes.c_int), ('ldw', ctypes.c_int), ('ldo', ctypes.c_int), ('alpha', ctypes.c_float), ('sk', ctypes.c_int),
                ('tm', ctypes.c_int), ('tn', ctypes.c_int), ('items', ctypes.c_int)]


lib = ctypes.CDLL(os.path.expanduser('~/work/q4/libq4pk2.so'))
lib.q4pk_run.restype = ctypes.c_int
lib.q4pk_run.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
assert lib.q4pk_sizeof() == ctypes.sizeof(GDesc)
j5 = ctypes.CDLL(os.path.expanduser('~/work/h2/libg2s4x.so')); j5.s8_gemm.restype = ctypes.c_int
j5.s8_gemm.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]


def layer_shapes():
    out = []
    for i in range(24):
        att = (i % 4 == 3)
        out += [(8224 if not att else 5120, 2048), (2048, 2048), (12288, 2048), (2048, 6144)]
    return out


def tm_(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return round(statistics.median(ts), 1), round(ts[int(0.95 * (len(ts) - 1))], 1)


def main(Ms):
    shapes = layer_shapes()
    W = [torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8) for (N, K) in shapes]
    wbytes = sum(w.numel() for w in W)
    for M in Ms:
        A = {K: torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8) for K in (2048, 6144)}
        O = [torch.empty(M, N, device=dev, dtype=torch.float16) for (N, K) in shapes]
        st = torch.cuda.current_stream().cuda_stream
        rec = dict(M=M, n_gemm=len(shapes), weight_MB=round(wbytes / 1e6, 1), gop=round(2 * M * wbytes / 1e9, 1))
        # per-shape best separate kernel (H2 CUTLASS cfg 0-4, J5 cfg 0-10)
        best = {}
        for (N, K) in set(shapes):
            w = W[shapes.index((N, K))]; o = O[shapes.index((N, K))]
            cands = [(('h2', c), (lambda c=c: QG.gemm('s8', A[K], w, 2 ** -10, c, out=o))) for c in range(5)]
            cands += [(('j5', c), (lambda c=c: j5.s8_gemm(A[K].data_ptr(), w.data_ptr(), o.data_ptr(), M, N, K, 2 ** -10, c, st))) for c in range(11)]
            bb = None
            for lab, fn in cands:
                try:
                    fn(); torch.cuda.synchronize(); t = tm_(fn, reps=10)[0]
                except Exception: continue
                if bb is None or t < bb[1]: bb = (lab, t)
            best[(N, K)] = bb
        rec['per_shape_best_us'] = {f'{N}x{K}': [best[(N, K)][0], best[(N, K)][1]] for (N, K) in best}
        rec['sum_of_isolated_us'] = round(sum(best[s][1] for s in shapes), 1)
        def seq():
            for g, (N, K) in enumerate(shapes):
                (lab, c), _ = best[(N, K)]
                if lab == 'h2': QG.gemm('s8', A[K], W[g], 2 ** -10, c, out=O[g])
                else: j5.s8_gemm(A[K].data_ptr(), W[g].data_ptr(), O[g].data_ptr(), M, N, K, 2 ** -10, c, torch.cuda.current_stream().cuda_stream)
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            for _ in range(2): seq()
        torch.cuda.current_stream().wait_stream(s_); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): seq()
        rec['graph_96_kernels'] = tm_(lambda: g.replay())
        # persistent v2: int32 outputs; split-K per GEMM = smallest divisor of the K-tile count giving >= target x 80 items
        Oi = [torch.empty(M, N, device=dev, dtype=torch.int32) for (N, K) in shapes]
        bar = torch.zeros(2, device=dev, dtype=torch.int32)
        prog = (GDesc * len(shapes))(); dprog = torch.empty(ctypes.sizeof(GDesc) * len(shapes), device=dev, dtype=torch.uint8)
        res = {}
        for cfg in range(7):
            BM, BN = BMN[cfg]
            for tgt in (0, 1, 2):
                for pf in (0, 4):
                    for g_, (N, K) in enumerate(shapes):
                        d = prog[g_]; d.A = A[K].data_ptr(); d.W = W[g_].data_ptr(); d.out = Oi[g_].data_ptr(); d.M = M; d.N = N; d.K = K
                        d.lda = K; d.ldw = K; d.ldo = N; d.alpha = 2 ** -10
                        base = ((M + BM - 1) // BM) * ((N + BN - 1) // BN); kt = K // 128; sk = 1
                        if tgt > 0:
                            for dv in range(1, kt + 1):
                                if kt % dv == 0 and kt // dv >= 2:
                                    sk = dv
                                    if base * dv >= tgt * 80: break
                        d.sk = sk
                    def run(cfg=cfg, pf=pf):
                        rc = lib.q4pk_run(cfg, ctypes.addressof(prog), dprog.data_ptr(), len(shapes), bar.data_ptr(), pf, 0, torch.cuda.current_stream().cuda_stream)
                        if rc: raise RuntimeError(rc)
                    key = f'c{cfg}_sk{tgt}_pf{pf}'
                    try:
                        run(); torch.cuda.synchronize(); res[key] = tm_(run)
                        ref = (A[2048].double() @ W[2].double().t())
                        res[key + '_exact'] = bool(torch.equal(Oi[2].double(), ref)) and bool(torch.equal(Oi[3].double(), A[6144].double() @ W[3].double().t()))
                    except Exception as ex:
                        res[key] = str(ex)
        rec['persistent'] = res
        okv = [v for k_, v in res.items() if isinstance(v, tuple) and res.get(k_ + '_exact')]
        rec['persistent_best'] = min(okv) if okv else None
        rec['floor_weights_us_at_600GBs'] = round(wbytes / 600e9 * 1e6, 1)
        print(json.dumps(rec), flush=True)
        with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')
        del A, O, Oi; torch.cuda.empty_cache()


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1].split(',')])
