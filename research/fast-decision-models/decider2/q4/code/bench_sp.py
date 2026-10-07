"""Q4 B11: 2:4 sparse int8 / int4 (custom mma.sp, q4sp.cu) against dense kernels at hobson-v19's GEMM shapes (A10G, exclusive GPU).
python bench_sp.py Ms [shapes]
Per (shape, M), best config of each kernel family by median (CUDA events around one call, 8 warm-up, 30 timed, inputs rotated over 3 sets):
  dense:  h2 CUTLASS s8 / s4 (fp16a epilogue, H2's deployed GEMMs; EVT SwiGLU for gate_up), J5 s8 (g2s4x), Q2 q2gemm s8 / s4 (bf16 / SwiGLU epilogue),
          Q4 dense control d8 / d4 (the same main loop as the sparse kernel, bf16 / SwiGLU epilogue)
  sparse: Q4 sp8 / sp4 (bf16 / SwiGLU epilogue; sp8 also fp16a = drop-in for QRT), CUTLASS 2.x GemmSparseUniversal (Q2's q2sp.cu copy)
Results: ~/work/q4/res_sparse.jsonl (one record per (shape, M)); values [label, median_us, p95_us]."""
import os, sys, json, ctypes, statistics, torch
sys.path[:0] = [os.path.expanduser('~/work/q4'), os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
import q4sp as S
import test_sp as TS
import q2k as Q
import qgemm as QG
dev = 'cuda'
SH = {'gdn_in': (8224, 2048), 'attn_in': (5120, 2048), 'out': (2048, 2048), 'gate_up': (12288, 2048), 'down': (2048, 6144)}
OUT = os.path.expanduser(os.environ.get('Q4OUT', '~/work/q4/res_sparse.jsonl'))
CF = list(range(16))


def tm(fn, reps=30, warm=8):
    for w in range(warm): fn(w)
    torch.cuda.synchronize(); ts = []
    for i in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(i); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return round(statistics.median(ts), 2), round(ts[int(0.95 * (len(ts) - 1))], 2)


def best(cands):
    out = None
    for lab, fn in cands:
        try:
            fn(0); torch.cuda.synchronize()
            m, p = tm(fn)
        except Exception as ex:
            if os.environ.get('Q4DBG'): print('cand fail', lab, repr(ex)[:200], flush=True)
            continue
        if out is None or m < out[1]: out = [lab, m, p]
    return out


_sp = None
def cutlass_sp():
    global _sp
    if _sp is None:
        p = os.path.expanduser('~/work/q4/libq2sp.so')
        if not os.path.exists(p): return None
        _sp = ctypes.CDLL(p)
        for f in (_sp.sp_s8, _sp.sp_s4):
            f.restype = ctypes.c_int; f.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        for f in (_sp.sp_s8_ebytes, _sp.sp_s4_ebytes):
            f.restype = ctypes.c_longlong; f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return _sp


_j5 = None
def j5lib():
    global _j5
    if _j5 is None:
        p = os.path.expanduser('~/work/h2/libg2s4x.so')
        if not os.path.exists(p): return None
        _j5 = ctypes.CDLL(p); f = _j5.s8_gemm; f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    return _j5


def emit(r):
    print(json.dumps(r), flush=True)
    with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')


def bench(Ms, shapes):
    TS.set_layout()
    for M in Ms:
        for name in shapes:
            N, K = SH[name]; sw = name == 'gate_up'
            epi = 'swiglu' if sw else 'bf16'
            r = dict(shape=name, M=M, N=N, K=K)
            g = lambda *s, q=7: torch.randint(-q, q + 1, s, device=dev, dtype=torch.int8)
            x8 = [g(M, K, q=127) for _ in range(3)]; x4 = [S.pack4(g(M, K)) for _ in range(3)]
            w8 = g(N, K, q=127); w4q = g(N, K); w4 = S.pack4(w4q)
            sa = torch.rand(M, device=dev) + 0.5; sb = torch.rand(N, device=dev) + 0.5
            ob = torch.empty(M, N // 2 if sw else N, device=dev, dtype=torch.bfloat16)
            o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
            oh = torch.empty(M, N // 2, device=dev, dtype=torch.bfloat16)
            Wc8, E8, _ = S.compress(w8, S.mask24_mag(w8, 'sp8'), 'sp8')
            Wc4, E4, _ = S.compress(w4q, S.mask24_mag(w4q, 'sp4'), 'sp4')
            if os.environ.get('ONLY') != 'cs':
                # ---- dense
                r['h2_s8'] = best([(f'cut{c}', (lambda i, c=c: QG.gemm('s8', x8[i % 3], w8, 2 ** -10, c, out=o16))) for c in range(5)])
                r['h2_s4'] = best([(f'cut{c}', (lambda i, c=c: QG.gemm('s4', x4[i % 3], w4, 0.5, c, out=o16))) for c in range(5)])
                if sw:
                    r['h2_s8_evt'] = best([(f'evt{c}', (lambda i, c=c: QG.swiglu_gemm('s8', x8[i % 3], w8, sa, sb, cfg=c, out=oh))) for c in (0, 1, 2, 3)])
                    r['h2_s4_evt'] = best([(f'evt{c}', (lambda i, c=c: QG.swiglu_gemm('s4', x4[i % 3], w4, sa, sb, cfg=c, out=oh))) for c in (0, 1, 2, 3)])
                j5 = j5lib()
                if j5 is not None:
                    def j5f(i, c):
                        rc = j5.s8_gemm(x8[i % 3].data_ptr(), w8.data_ptr(), o16.data_ptr(), M, N, K, 2 ** -10, c, torch.cuda.current_stream().cuda_stream)
                        if rc: raise RuntimeError(rc)
                    r['j5_s8'] = best([(f'j5c{c}', (lambda i, c=c: j5f(i, c))) for c in range(11)])
                r['q2_s8'] = best([(f'q{c}', (lambda i, c=c: Q.run('s8', c, Q.prob(x8[i % 3], w8, 's8', ob, sa, sb, epi=epi, N=N)))) for c in range(10)])
                r['q2_s4'] = best([(f'q{c}', (lambda i, c=c: Q.run('s4', c, Q.prob(x4[i % 3], w4, 's4', ob, sa, sb, epi=epi, N=N)))) for c in range(10)])
                r['q4_d8'] = best([(f'd{c}', (lambda i, c=c: S.gemm('d8', c, w8, None, x8[i % 3], ob, N, M, K, sa=sa, sb=sb, epi=epi))) for c in CF])
                r['q4_d4'] = best([(f'd{c}', (lambda i, c=c: S.gemm('d4', c, w4, None, x4[i % 3], ob, N, M, K, sa=sa, sb=sb, epi=epi))) for c in CF])
                # ---- sparse (magnitude 2:4 mask; speed does not depend on which positions are kept)
                r['q4_sp8'] = best([(f's{c}', (lambda i, c=c: S.gemm('sp8', c, Wc8, E8, x8[i % 3], ob, N, M, K, sa=sa, sb=sb, epi=epi))) for c in CF])
                r['q4_sp4'] = best([(f's{c}', (lambda i, c=c: S.gemm('sp4', c, Wc4, E4, x4[i % 3], ob, N, M, K, sa=sa, sb=sb, epi=epi))) for c in CF])
                r['q4_sp8_fp16a'] = best([(f's{c}', (lambda i, c=c: S.gemm('sp8', c, Wc8, E8, x8[i % 3], o16, N, M, K, epi='fp16a', alpha=2 ** -10))) for c in CF])
                r['q4_sp4_fp16a'] = best([(f's{c}', (lambda i, c=c: S.gemm('sp4', c, Wc4, E4, x4[i % 3], o16, N, M, K, epi='fp16a', alpha=0.5))) for c in CF])
            cs = cutlass_sp()
            if cs is not None:
                Mp = (M + 15) // 16 * 16          # CUTLASS needs the token count aligned (fp16 output, 128-bit accesses)
                ot = torch.empty(N, Mp, device=dev, dtype=torch.float16)
                pad = lambda a: torch.cat([a, torch.zeros(Mp - M, a.shape[1], device=dev, dtype=a.dtype)], 0).contiguous()
                for kind, Wc, xs, al in (('s8', Wc8, [pad(a) for a in x8], 2 ** -10), ('s4', Wc4, [pad(a) for a in x4], 0.5)):
                    fe = cs.sp_s8_ebytes if kind == 's8' else cs.sp_s4_ebytes
                    nb = max(fe(N, K, c) for c in range(16))
                    Ec = torch.full((nb // 4 + 64,), 0x44444444, device=dev, dtype=torch.int32)
                    f = cs.sp_s8 if kind == 's8' else cs.sp_s4
                    def cf(i, c, f=f, Wc=Wc, xs=xs, Ec=Ec, al=al):
                        rc = f(Wc.data_ptr(), xs[i % 3].data_ptr(), ot.data_ptr(), Ec.data_ptr(), N, Mp, K, al, c, torch.cuda.current_stream().cuda_stream)
                        if rc: raise RuntimeError(rc)
                    r[f'cutlass_sp_{kind}'] = best([(f'cs{c}', (lambda i, c=c, cf=cf: cf(i, c))) for c in range(16)])
            emit(r)
            del x8, x4, w8, w4, w4q, Wc8, Wc4, E8, E4; torch.cuda.empty_cache()


if __name__ == '__main__':
    Ms = [int(x) for x in sys.argv[1].split(',')]
    shapes = sys.argv[2].split(',') if len(sys.argv) > 2 else list(SH)
    bench(Ms, shapes)
