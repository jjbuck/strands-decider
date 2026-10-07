"""Q2 GEMM timing at hobson-v19's shapes (A10G, exclusive GPU).
python bench_gemm.py base  Ms        -> cuBLAS bf16, H2 CUTLASS s8/s4 (+EVT swiglu), J5 s8 (g2s4x), Q2 s4/s8/bf16, best config each
python bench_gemm.py var   Ms        -> Q2 variants: B1 tails (bf16 / int8, r = 32..256) + Z GEMM, group g64/g128, B9 skip, B5 splits, row-role
Timing: CUDA events around one call, 8 warm-up, 30 timed, input buffers rotated over 3 sets; median and p95 in microseconds."""
import os, sys, json, statistics, ctypes, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
import q2k as Q
import qgemm as QG
dev = 'cuda'
SH = {'gdn_in': (8224, 2048), 'attn_in': (5120, 2048), 'out': (2048, 2048), 'gate_up': (12288, 2048), 'down': (2048, 6144)}
COUNT = {'gdn_in': 18, 'attn_in': 6, 'out': 24, 'gate_up': 24, 'down': 24}
OUT = os.path.expanduser(os.environ.get('Q2OUT', '~/work/q2/res_gemm.jsonl'))


def tm(fn, reps=30, warm=8):
    for w in range(warm): fn(w)
    torch.cuda.synchronize(); ts = []
    for i in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(i); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return round(statistics.median(ts), 2), round(ts[int(0.95 * (len(ts) - 1))], 2)


def best(cands):
    """cands: list of (label, fn(i)) -> (label, med, p95) of the fastest by median"""
    out = None
    for lab, fn in cands:
        try:
            fn(0); torch.cuda.synchronize()
            m, p = tm(fn)
        except Exception as ex:
            if os.environ.get('Q2DBG'): print('cand fail', lab, repr(ex)[:200], flush=True)
            continue
        if out is None or m < out[1]: out = (lab, m, p)
    return out


def emit(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, 'a') as f: f.write(json.dumps(rec) + '\n')


class Bufs:
    def __init__(self, M, N, K, nset=3):
        self.M, self.N, self.K = M, N, K
        g = lambda *s, lo=-7, hi=8: torch.randint(lo, hi, s, device=dev, dtype=torch.int8)
        self.a4 = [Q.pack4(g(M, K)) for _ in range(nset)]; self.w4 = Q.pack4(g(N, K))
        self.a8 = [g(M, K, lo=-127, hi=128) for _ in range(nset)]; self.w8 = g(N, K, lo=-127, hi=128)
        self.ab = [torch.randn(M, K, device=dev, dtype=torch.bfloat16) for _ in range(nset)]; self.wb = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        self.sa = torch.rand(M, device=dev) + 0.5; self.sw = torch.rand(N, device=dev) + 0.5
        self.o = torch.empty(M, N, device=dev, dtype=torch.bfloat16); self.oh = torch.empty(M, N // 2, device=dev, dtype=torch.bfloat16)
        self.o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
        self.nset = nset


def q2c(var, cfgs, mk):
    return [(f'{var}/c{c}', (lambda i, c=c: Q.run(var, c, mk(i)))) for c in cfgs]


CF = [0, 1, 2, 3, 4, 5, 6, 7]


def base(Ms):
    for M in Ms:
        for name, (N, K) in SH.items():
            b = Bufs(M, N, K); sw = name == 'gate_up'
            r = dict(kind='base', shape=name, M=M, N=N, K=K)
            r['bf16_cublas'] = best([('cublas', lambda i: torch.mm(b.ab[i % 3], b.wb.t()))])
            a8 = lambda kind, A, B: QG.alpha_for(kind, K) if hasattr(QG, 'alpha_for') else 1.0 / 1024
            r['h2_s8'] = best([(f'cut{c}', (lambda i, c=c: QG.gemm('s8', b.a8[i % 3], b.w8, 2 ** -10, c, out=b.o16))) for c in range(5)])
            r['h2_s4'] = best([(f'cut{c}', (lambda i, c=c: QG.gemm('s4', b.a4[i % 3], b.w4, 0.5, c, out=b.o16))) for c in range(5)])
            if sw:
                r['h2_s8_evt'] = best([(f'evt{c}', (lambda i, c=c: QG.swiglu_gemm('s8', b.a8[i % 3], b.w8, b.sa, b.sw, cfg=c, out=b.oh))) for c in (1, 3)])
                r['h2_s4_evt'] = best([(f'evt{c}', (lambda i, c=c: QG.swiglu_gemm('s4', b.a4[i % 3], b.w4, b.sa, b.sw, cfg=c, out=b.oh))) for c in (0, 3)])
            try:
                lib = ctypes.CDLL(os.path.expanduser('~/work/h2/libg2s4x.so'))
                f = lib.s8_gemm; f.restype = ctypes.c_int; f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
                def j5(i, c):
                    rc = f(b.a8[i % 3].data_ptr(), b.w8.data_ptr(), b.o16.data_ptr(), M, N, K, 2 ** -10, c, torch.cuda.current_stream().cuda_stream)
                    if rc: raise RuntimeError(rc)
                r['j5_s8'] = best([(f'j5c{c}', (lambda i, c=c: j5(i, c))) for c in range(11)])
            except OSError:
                pass
            epi = 'swiglu' if sw else 'bf16'; ob = b.oh if sw else b.o
            r['q2_s4'] = best(q2c('s4', CF, lambda i: Q.prob(b.a4[i % 3], b.w4, 's4', ob, b.sa, b.sw, epi=epi, N=N)))
            r['q2_s8'] = best(q2c('s8', CF, lambda i: Q.prob(b.a8[i % 3], b.w8, 's8', ob, b.sa, b.sw, epi=epi, N=N)))
            r['q2_s4_fp16a'] = best(q2c('s4', CF, lambda i: Q.prob(b.a4[i % 3], b.w4, 's4', b.o16, epi='fp16a', alpha=0.5)))
            r['q2_s8_fp16a'] = best(q2c('s8', CF, lambda i: Q.prob(b.a8[i % 3], b.w8, 's8', b.o16, epi='fp16a', alpha=2 ** -10)))
            r['q2_bf16'] = best(q2c('bf16', CF, lambda i: Q.prob(b.ab[i % 3], b.wb, 'bf16', ob, epi=epi, N=N)))
            emit(r)
            del b; torch.cuda.empty_cache()


def var(Ms):
    for M in Ms:
        for name, (N, K) in SH.items():
            b = Bufs(M, N, K); sw = name == 'gate_up'
            epi = 'swiglu' if sw else 'bf16'; ob = b.oh if sw else b.o
            r = dict(kind='var', shape=name, M=M, N=N, K=K)
            r['q2_s4'] = best(q2c('s4', CF, lambda i: Q.prob(b.a4[i % 3], b.w4, 's4', ob, b.sa, b.sw, epi=epi, N=N)))
            # B1: bf16 tail r in 32..256, int8 tail r in 64..256; Z = dX @ A by cuBLAS (bf16 in, bf16 out)
            dX = [torch.randn(M, K, device=dev, dtype=torch.bfloat16) * 0.01 for _ in range(3)]
            for rr in (32, 64, 128, 256):
                Z = torch.randn(M, rr, device=dev, dtype=torch.bfloat16); Bt = torch.randn(N, rr, device=dev, dtype=torch.bfloat16)
                Ar = torch.randn(K, rr, device=dev, dtype=torch.bfloat16)
                r[f'b1_tbf16_r{rr}'] = best(q2c('s4_tbf16', CF, lambda i: Q.prob(b.a4[i % 3], b.w4, 's4', ob, b.sa, b.sw, A1=Z, B1=Bt, epi=epi, N=N)))
                r[f'z_cublas_r{rr}'] = best([('mm', lambda i: torch.mm(dX[i % 3], Ar))])
                Zc = torch.empty(M, rr, device=dev, dtype=torch.bfloat16)
                ArT = Ar.t().contiguous()
                r[f'z_q2_r{rr}'] = best(q2c('bf16', CF, lambda i: Q.prob(dX[i % 3], ArT, 'bf16', Zc)))
                if rr >= 64:
                    Z8 = torch.randint(-127, 128, (M, rr), device=dev, dtype=torch.int8); B8 = torch.randint(-127, 128, (N, rr), device=dev, dtype=torch.int8)
                    sz = torch.rand(M, device=dev); sb8 = torch.rand(N, device=dev)
                    r[f'b1_ts8_r{rr}'] = best(q2c('s4_ts8', [0, 1, 3, 4, 5], lambda i: Q.prob(b.a4[i % 3], b.w4, 's4', ob, b.sa, b.sw, A1=Z8, B1=B8, sa1=sz, sb1=sb8, epi=epi, N=N)))
            # group scales
            for g in (64, 128):
                G = K // g
                gsa = torch.rand(M, G, device=dev); gsw = torch.rand(N, G, device=dev)
                r[f'g{g}'] = best(q2c(f's4g{g}', CF, lambda i: Q.prob(b.a4[i % 3], b.w4, 's4', ob, gsa=gsa, gsb=gsw, epi=epi, N=N)))
            # B9 tile skipping, f = 0.75, 0.5
            nkb = K // 128
            for f in (0.75, 0.5):
                nk = max(1, int(round(f * nkb)))
                r[f'skip{f}'] = best(q2c('s4skip', [0, 1, 3, 4, 5], lambda i: Q.prob(b.a4[i % 3], b.w4, 's4', ob, b.sa, b.sw, kscale=nkb / nk, seed=i, nkeep=nk, nkb=nkb, epi=epi, N=N)))
            # B5 / ResQ: int8 slice K8 + int4 slice K4 (K4 + K8 <= K); report K8 = 128 with K4 = K - 128, and the B5 examples
            for (k8, k4) in [(128, K - 128), (256, K - 256), (K // 4, K // 2), (K // 8, (5 * K) // 8), (0, (3 * K) // 4)]:
                k4 = (k4 // 128) * 128
                A4s = [a[:, : k4 // 2] for a in b.a4]; W4s = b.w4[:, : k4 // 2]
                if k8 > 0:
                    A8s = b.a8[0][:, :k8]; W8s = b.w8[:, :k8]
                    r[f'split_k8_{k8}_k4_{k4}'] = best(q2c('s4_ts8', [0, 1, 3, 4, 5], lambda i: Q.prob(A4s[i % 3], W4s, 's4', ob, b.sa, b.sw, A1=A8s, B1=W8s, sa1=b.sa, sb1=b.sw, epi=epi, N=N)))
                else:
                    r[f'split_k8_0_k4_{k4}'] = best(q2c('s4', CF, lambda i: Q.prob(A4s[i % 3], W4s, 's4', ob, b.sa, b.sw, epi=epi, N=N)))
            # row-role: state rows int4, last 125 rows int8 (1 question); and an arbitrary 10% int8 partition (B8)
            for lab, nq in (('rowrole_q125', 125), ('rowrole_10pct', max(16, M // 10))):
                if nq >= M: continue
                q0 = M - nq
                rm = torch.randperm(M, device=dev).to(torch.int32) if lab.endswith('pct') else torch.arange(M, device=dev, dtype=torch.int32)
                a4s = [a[:q0] for a in b.a4]; a8s = b.a8[0][q0:]
                r[lab] = best([(f'rr/c{c}', (lambda i, c=c: Q.run('s4', c, Q.prob(a4s[i % 3], b.w4, 's4', ob, b.sa, b.sw, epi=epi, N=N, rowmap=rm[:q0]),
                                                                    Q.prob(a8s, b.w8, 's8', ob, b.sa, b.sw, epi=epi, N=N, rowmap=rm[q0:])))) for c in (0, 1, 3, 4, 5)])
            emit(r)
            del b; torch.cuda.empty_cache()


if __name__ == '__main__':
    cmd = sys.argv[1]; Ms = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [140, 400, 1125, 4125]
    {'base': base, 'var': var}[cmd](Ms)
