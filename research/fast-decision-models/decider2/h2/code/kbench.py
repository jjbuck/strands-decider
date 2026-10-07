"""G2 kernel microbenchmarks on the 2B projection shapes (A10G).
  python kbench.py build      -> compile the CUTLASS int4/int8 extension, correctness check
  python kbench.py prec       -> bf16 cuBLAS vs int8 (cuBLAS _int_mm, CUTLASS s8, d2 Triton fused-dequant) vs int4 (CUTLASS s4), M=128..4096
  python kbench.py nest       -> nested leading-sub-block GEMMs (width 1, 1/2, 1/4, 1/8) and per-class split execution at T=1000/4000
Timing: CUDA events, 30 warm reps after 8 warm-up, fresh input buffers rotated, median and p95 (us)."""
import os, sys, json, statistics, subprocess, time
import torch
W = os.path.expanduser('~/work/h2')
sys.path.insert(0, W); sys.path.insert(0, os.path.expanduser('~/work/h2'))
SH = {"gdn_in": (8224, 2048), "attn_in": (5120, 2048), "out": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}


def tm(fn, reps=30, warm=8):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return statistics.median(ts), ts[int(0.95 * (len(ts) - 1))]


class _Ext:
    def __init__(self, lib):
        import ctypes
        self.lib = lib
        for f in (lib.s4_gemm, lib.s8_gemm):
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]

    def _call(self, f, A, B, K, alpha, cfg, out=None):
        M, N = A.shape[0], B.shape[0]
        C = out if out is not None else torch.empty(M, N, device=A.device, dtype=torch.float16)
        rc = f(A.data_ptr(), B.data_ptr(), C.data_ptr(), M, N, K, float(alpha), int(cfg), torch.cuda.current_stream().cuda_stream)
        if rc != 0: raise RuntimeError(f'cutlass rc {rc}')
        return C

    def s4_gemm(self, A, B, alpha, cfg, out=None):  # A [M, K/2] uint8, B [N, K/2] uint8
        return self._call(self.lib.s4_gemm, A, B, A.shape[1] * 2, alpha, cfg, out)

    def s8_gemm(self, A, B, alpha, cfg, out=None):
        return self._call(self.lib.s8_gemm, A, B, A.shape[1], alpha, cfg, out)


def ext():
    import ctypes
    cut = os.path.expanduser('~/work/cutlass')
    if not os.path.exists(cut + '/include/cutlass/cutlass.h'):
        subprocess.run(['git', 'clone', '-q', '--depth', '1', '--branch', 'v3.5.1', 'https://github.com/NVIDIA/cutlass.git', cut], check=True)
    so = W + '/libg2s4.so'
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(W + '/g2s4.cu'):
        t0 = time.time()
        r = subprocess.run(['/usr/local/cuda/bin/nvcc', '-O3', '-std=c++17', '-gencode=arch=compute_86,code=sm_86', '--expt-relaxed-constexpr', '-DNDEBUG',
                            '-shared', '-Xcompiler', '-fPIC', '-I', cut + '/include', '-I', cut + '/tools/util/include', '-o', so, W + '/g2s4.cu'],
                           capture_output=True, text=True)
        print('nvcc rc', r.returncode, f'{time.time()-t0:.0f}s', r.stderr[-3000:], flush=True)
        if r.returncode: raise SystemExit('build failed')
    return _Ext(ctypes.CDLL(so))


def pack4(q):  # int tensor in [-8,7], [..., K] -> uint8 [..., K/2], element 2j in the low nibble
    q = q.to(torch.int16)
    lo = q[..., 0::2] & 0xF; hi = q[..., 1::2] & 0xF
    return (lo | (hi << 4)).to(torch.uint8).contiguous()


def check(m):
    torch.manual_seed(0)
    for (M, N, K) in [(256, 512, 2048), (1000, 2048, 6144), (77, 8224, 2048)]:
        a = torch.randint(-7, 8, (M, K), device='cuda'); b = torch.randint(-7, 8, (N, K), device='cuda')
        ref = a.double() @ b.double().t()
        for cfg in range(5):
            try:
                c = m.s4_gemm(pack4(a), pack4(b), 1.0 / 64, cfg).double() * 64
                print(f's4 M={M} N={N} K={K} cfg{cfg}: max rel err {((c - ref).abs().max() / ref.abs().max()).item():.2e}', flush=True)
            except Exception as ex: print('s4 cfg', cfg, 'FAILED', str(ex)[:200], flush=True)
        a8 = torch.randint(-127, 128, (M, K), device='cuda'); b8 = torch.randint(-127, 128, (N, K), device='cuda')
        ref8 = a8.double() @ b8.double().t()
        for cfg in range(5):
            try:
                c = m.s8_gemm(a8.to(torch.int8).contiguous(), b8.to(torch.int8).contiguous(), 1.0 / 4096, cfg).double() * 4096
                print(f's8 M={M} cfg{cfg}: max rel err {((c - ref8).abs().max() / ref8.abs().max()).item():.2e}', flush=True)
            except Exception as ex: print('s8 cfg', cfg, 'FAILED', str(ex)[:200], flush=True)


def prec(m):
    import int8mm as I8
    CF8 = [(128, 128, 64, 4, 3), (128, 128, 128, 8, 3), (128, 256, 64, 8, 3), (64, 128, 128, 4, 3), (256, 128, 64, 8, 3)]
    out = []
    for name, (N, K) in SH.items():
        w = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
        w8 = torch.randint(-127, 128, (N, K), device='cuda').to(torch.int8).contiguous()
        w4 = pack4(torch.randint(-7, 8, (N, K), device='cuda'))
        sw = torch.rand(N, device='cuda') * 0.01
        for M in (128, 256, 1024, 4096):
            nb = 4
            xs = [torch.randn(M, K, device='cuda').bfloat16() for _ in range(nb)]
            x8 = [torch.randint(-127, 128, (M, K), device='cuda').to(torch.int8).contiguous() for _ in range(nb)]
            x4 = [pack4(torch.randint(-7, 8, (M, K), device='cuda')) for _ in range(nb)]
            sa = torch.rand(M, device='cuda') * 0.01
            it = [0]
            def nxt(L):
                it[0] = (it[0] + 1) % nb; return L[it[0]]
            r = dict(shape=name, M=M, N=N, K=K)
            r['bf16'] = tm(lambda: nxt(xs) @ w.t())
            r['int_mm'] = tm(lambda: torch._int_mm(nxt(x8), w8.t()))
            best = None
            for c in CF8:
                try: t = tm(lambda: I8.i8mm(nxt(x8), sa, w8, sw, c))
                except Exception: continue
                if best is None or t[0] < best[0][0]: best = (t, c)
            r['tri_i8_dq'] = best[0]
            for nm, fn, L, wq in (('cut_s8', m.s8_gemm, x8, w8), ('cut_s4', m.s4_gemm, x4, w4)):
                bb = None
                for cfg in range(5):
                    try: t = tm(lambda: fn(nxt(L), wq, 1.0, cfg))
                    except Exception: continue
                    if bb is None or t[0] < bb[0][0]: bb = (t, cfg)
                r[nm] = bb[0] if bb else None; r[nm + '_cfg'] = bb[1] if bb else None
            fl = 2 * M * N * K
            line = f'{name:8s} M={M:5d} ' + ' '.join(f'{k} {v[0]:7.0f}us ({fl / v[0] / 1e6:5.1f}T)' for k, v in r.items() if isinstance(v, tuple))
            print(line, flush=True); out.append(r)
    json.dump(out, open(W + '/res_prec.json', 'w'), indent=0)


def nest():
    """nested width: class c uses the leading sub-block of each weight. Rows already grouped by class (contiguous)."""
    out = []
    F_ = 6144; D = 2048
    Wgu = (torch.randn(2 * F_, D, device='cuda') * 0.02).bfloat16(); Wd = (torch.randn(D, F_, device='cuda') * 0.02).bfloat16()
    Win = (torch.randn(8224, D, device='cuda') * 0.02).bfloat16(); Wo = (torch.randn(D, D, device='cuda') * 0.02).bfloat16()
    def layer_gemms(M, w, xs):
        """one GDN layer's 4 GEMMs for M rows at width w (head-major in_proj, interleaved gate/up)"""
        nin = int(8224 * w) // 16 * 16 if w < 1 else 8224; nh = int(D * w); nf = int(F_ * w)
        x = xs
        def f():
            p = x @ Win[:nin].t()
            o = x[:, :nh] @ Wo[:, :nh].t() if w < 1 else x @ Wo.t()
            m = x @ Wgu[:2 * nf].t()
            d = m[:, :nf] @ Wd[:, :nf].t()
            return d
        return f
    for T in (1000, 4000):
        xsT = torch.randn(T, D, device='cuda').bfloat16()
        base = tm(layer_gemms(T, 1.0, xsT))
        r = dict(T=T, dense=base)
        print(f'T={T} dense layer GEMMs {base[0]:.0f}us', flush=True)
        for w in (0.5, 0.25, 0.125):
            t = tm(layer_gemms(T, w, xsT)); r[f'all@{w}'] = t
            print(f'  all rows @ width {w}: {t[0]:.0f}us = {t[0] / base[0]:.3f} of dense (FLOP ratio ~{w})', flush=True)
        for split in ((0.1, 0, 0.9), (0.1, 0.2, 0.7), (0.25, 0, 0.75), (0.2, 0, 0.8)):
            n_full = int(T * split[0]); n_half = int(T * split[1]); n_q = T - n_full - n_half
            fs = []
            if n_full: fs.append(layer_gemms(n_full, 1.0, xsT[:n_full]))
            if n_half: fs.append(layer_gemms(n_half, 0.5, xsT[:n_half]))
            if n_q: fs.append(layer_gemms(n_q, 0.25, xsT[:n_q]))
            t = tm(lambda: [f() for f in fs]); flr = split[0] + 0.5 * split[1] + 0.25 * split[2]
            r[f'split{split}'] = t
            print(f'  split full/half/quarter {split}: {t[0]:.0f}us = {t[0] / base[0]:.3f} of dense (FLOP ratio {flr:.3f})', flush=True)
            # same split + 125 question rows in the full class
            fs2 = [layer_gemms(n_full + 125, 1.0, torch.randn(n_full + 125, D, device='cuda').bfloat16())] + fs[1:]
            t2 = tm(lambda: [f() for f in fs2])
            print(f'    + 125 question rows in the full class: {t2[0]:.0f}us', flush=True); r[f'split{split}+q125'] = t2
        # gather/scatter overhead (rows grouped by class from token order)
        idx = torch.randperm(T, device='cuda')
        g = tm(lambda: xsT.index_select(0, idx)); r['gather_row2048'] = g
        print(f'  gather of [{T},2048] rows: {g[0]:.1f}us', flush=True)
        out.append(r)
    json.dump(out, open(W + '/res_nest.json', 'w'), indent=0, default=str)


def square(m):
    out = {}
    for n in (4096, 8192):
        a = torch.randn(n, n, device='cuda').bfloat16(); b = torch.randn(n, n, device='cuda').bfloat16()
        a16, b16 = a.half(), b.half()
        a8 = torch.randint(-127, 128, (n, n), device='cuda').to(torch.int8); b8 = torch.randint(-127, 128, (n, n), device='cuda').to(torch.int8)
        a4 = pack4(torch.randint(-7, 8, (n, n), device='cuda')); b4 = pack4(torch.randint(-7, 8, (n, n), device='cuda'))
        c16 = torch.empty(n, n, device='cuda', dtype=torch.float16)
        fl = 2 * n ** 3
        r = dict(bf16=tm(lambda: a @ b.t(), reps=10), fp16=tm(lambda: a16 @ b16.t(), reps=10), int_mm=tm(lambda: torch._int_mm(a8, b8.t()), reps=10))
        for cfg in range(5):
            try: r[f's8c{cfg}'] = tm(lambda: m.s8_gemm(a8, b8, 1.0, cfg, out=c16), reps=10)
            except Exception as ex: print('s8', cfg, ex)
            try: r[f's4c{cfg}'] = tm(lambda: m.s4_gemm(a4, b4, 1.0, cfg, out=c16), reps=10)
            except Exception as ex: print('s4', cfg, ex)
        print(f'n={n}: ' + ' '.join(f'{k} {fl / v[0] / 1e6:.1f}T' for k, v in r.items()), flush=True)
        out[n] = {k: fl / v[0] / 1e6 for k, v in r.items()}
    json.dump(out, open(W + '/res_square.json', 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
    if cmd == 'build': check(ext())
    elif cmd == 'square': square(ext())
    elif cmd == 'prec': prec(ext())
    elif cmd == 'nest': nest()
