"""Q2 B11: 2:4 sparse int8 / int4 (CUTLASS mma.sp) vs dense (H2 CUTLASS, Q2) at hobson's shapes.  python bench_sp.py [Ts]"""
import os, sys, json, ctypes, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/h2')]
import q2k as Q, qgemm as QG
from bench_gemm import tm, best, SH
dev = 'cuda'
OUT = os.path.expanduser('~/work/q2/res_sparse.jsonl')
lib = ctypes.CDLL(os.path.expanduser('~/work/q2/libq2sp.so'))
for f in (lib.sp_s8, lib.sp_s4):
    f.restype = ctypes.c_int; f.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
for f in (lib.sp_s8_ebytes, lib.sp_s4_ebytes):
    f.restype = ctypes.c_longlong; f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]


def sp(kind, Wc, X, Y, E, Nout, T, K, alpha, cfg):
    f = lib.sp_s8 if kind == 's8' else lib.sp_s4
    rc = f(Wc.data_ptr(), X.data_ptr(), Y.data_ptr(), E.data_ptr(), Nout, T, K, alpha, cfg, torch.cuda.current_stream().cuda_stream)
    if rc: raise RuntimeError(f'sp {kind} cfg{cfg} rc {rc}')


def ebuf(kind, Nout, K):
    nb = max((lib.sp_s8_ebytes if kind == 's8' else lib.sp_s4_ebytes)(Nout, K, c) for c in range(10))
    return torch.full((nb // 4 + 64,), 0x44444444, device=dev, dtype=torch.int32)


def check():
    torch.manual_seed(0)
    for kind, qm in (('s8', 127), ('s4', 7)):
        Nout, T, K = 512, 136, 1024
        c = torch.randint(-qm, qm + 1, (Nout, K // 2), device=dev, dtype=torch.int8)
        x = torch.randint(-qm, qm + 1, (T, K), device=dev, dtype=torch.int8)
        Wu = torch.zeros(Nout, K // 4, 4, device=dev, dtype=torch.int8); Wu[:, :, 0] = c[:, 0::2]; Wu[:, :, 1] = c[:, 1::2]; Wu = Wu.reshape(Nout, K)
        ref = (Wu.double() @ x.double().t())
        E = ebuf(kind, Nout, K)
        Wc = c if kind == 's8' else Q.pack4(c); X = x if kind == 's8' else Q.pack4(x)
        for cfg in range(10):
            Y = torch.empty(Nout, T, device=dev, dtype=torch.float16)
            alpha = 1.0 / 1024 if kind == 's8' else 1.0
            try:
                sp(kind, Wc, X, Y, E, Nout, T, K, alpha, cfg); torch.cuda.synchronize()
                err = ((Y.double() / alpha - ref).abs().max() / ref.abs().max()).item()
                print(f'{kind} cfg{cfg}: max rel err {err:.2e}', flush=True)
            except Exception as e:
                print(kind, cfg, e)


def emit(r):
    print(json.dumps(r), flush=True)
    with open(OUT, 'a') as f: f.write(json.dumps(r) + '\n')


def bench(Ts):
    for T in Ts:
        for name, (Nout, K) in SH.items():
            r = dict(shape=name, T=T, N=Nout, K=K)
            x8 = [torch.randint(-127, 128, (T, K), device=dev, dtype=torch.int8) for _ in range(3)]
            x4 = [Q.pack4(torch.randint(-7, 8, (T, K), device=dev, dtype=torch.int8)) for _ in range(3)]
            w8 = torch.randint(-127, 128, (Nout, K), device=dev, dtype=torch.int8); w4 = Q.pack4(torch.randint(-7, 8, (Nout, K), device=dev, dtype=torch.int8))
            c8 = torch.randint(-127, 128, (Nout, K // 2), device=dev, dtype=torch.int8); c4 = Q.pack4(torch.randint(-7, 8, (Nout, K // 2), device=dev, dtype=torch.int8))
            o16 = torch.empty(T, Nout, device=dev, dtype=torch.float16); ot = torch.empty(Nout, T, device=dev, dtype=torch.float16)
            E8 = ebuf('s8', Nout, K); E4 = ebuf('s4', Nout, K)
            r['dense_s8_cutlass'] = best([(f'c{c}', (lambda i, c=c: QG.gemm('s8', x8[i % 3], w8, 2 ** -10, c, out=o16))) for c in range(5)])
            r['dense_s4_cutlass'] = best([(f'c{c}', (lambda i, c=c: QG.gemm('s4', x4[i % 3], w4, 0.5, c, out=o16))) for c in range(5)])
            r['dense_s8_q2'] = best([(f'q{c}', (lambda i, c=c: Q.run('s8', c, Q.prob(x8[i % 3], w8, 's8', o16, epi='fp16a', alpha=2 ** -10)))) for c in range(10)])
            r['dense_s4_q2'] = best([(f'q{c}', (lambda i, c=c: Q.run('s4', c, Q.prob(x4[i % 3], w4, 's4', o16, epi='fp16a', alpha=0.5)))) for c in range(10)])
            r['sparse_s8'] = best([(f's{c}', (lambda i, c=c: sp('s8', c8, x8[i % 3], ot, E8, Nout, T, K, 2 ** -10, c))) for c in range(10)])
            r['sparse_s4'] = best([(f's{c}', (lambda i, c=c: sp('s4', c4, x4[i % 3], ot, E4, Nout, T, K, 0.5, c))) for c in range(10)])
            emit(r)


if __name__ == '__main__':
    if sys.argv[1] == 'check': check()
    else: bench([int(t) for t in sys.argv[2].split(',')])
