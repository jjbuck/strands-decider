import sys, os, ctypes, torch, statistics as st
sys.path[:0] = [os.path.expanduser('~/work/h2')]
import qgemm as QG
lib = ctypes.CDLL(os.path.expanduser('~/work/h2/libh2evt.so'))
for fn in (lib.s4_swiglu, lib.s8_swiglu):
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
torch.manual_seed(0); dev = 'cuda'
def ref(acc, ra, cs):
    y = (acc.float() * ra[:, None]) * cs[None, :]
    g = y[:, 0::2].bfloat16().float(); u = y[:, 1::2].bfloat16().float()
    s = (g / (1 + torch.exp(-g))).bfloat16().float()
    return (s * u).bfloat16()
for kind, M in (('s4', 1000), ('s4', 1015), ('s8', 1000), ('s4', 4000)):
    N, K = 12288, 2048
    qmax = 7 if kind == 's4' else 127
    a = torch.randint(-qmax, qmax + 1, (M, K), device=dev); b = torch.randint(-qmax, qmax + 1, (N, K), device=dev)
    acc = (a.double() @ b.double().t())
    ra = torch.rand(M, device=dev) * 0.02 + 0.001; cs = torch.rand(N, device=dev) * (0.05 / qmax)
    A = QG.pack4(a) if kind == 's4' else a.to(torch.int8).contiguous(); B = QG.pack4(b) if kind == 's4' else b.to(torch.int8).contiguous()
    out = torch.zeros(M, N // 2, device=dev, dtype=torch.bfloat16)
    f = lib.s4_swiglu if kind == 's4' else lib.s8_swiglu
    for cfg in ((0, 3) if kind == 's4' else (1, 3)):
        out.zero_()
        rc = f(A.data_ptr(), B.data_ptr(), out.data_ptr(), ra.data_ptr(), cs.data_ptr(), M, N, K, N // 2, cfg, torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        r = ref(acc, ra, cs)
        d = (out.float() - r.float()).abs(); rel = d.max().item() / r.float().abs().max().item()
        print(f'{kind} M={M} cfg{cfg} rc={rc}: max rel err {rel:.2e}, exact frac {(d == 0).float().mean().item():.4f}', flush=True)
        if rc == 0:
            ts = []
            for _ in range(20):
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
                f(A.data_ptr(), B.data_ptr(), out.data_ptr(), ra.data_ptr(), cs.data_ptr(), M, N, K, N // 2, cfg, torch.cuda.current_stream().cuda_stream)
                e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
            C = torch.empty(M, N, device=dev, dtype=torch.float16); ts2 = []
            for _ in range(20):
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
                QG.gemm(kind, A, B, 1.0 / 64, 3, out=C); e1.record(); e1.synchronize(); ts2.append(e0.elapsed_time(e1) * 1000)
            print(f'    evt swiglu gemm {st.median(ts):.0f} us vs plain fp16-out gemm {st.median(ts2):.0f} us (contended GPU)', flush=True)
