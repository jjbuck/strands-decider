import sys, os, torch, statistics as st
sys.path[:0] = [os.path.expanduser('~/work/h2')]
import qk as K, rot as RT, triton
dev = 'cuda'; M = K.mats(dev)
def tmk(f, n=30):
    for _ in range(5): f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10): f()
    ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 100)
    return st.median(ts)
R4 = RT.Rot(6144, 1236, dev)
for T in (1000, 4000):
    GU = (torch.randn(T, 12288, device=dev) * 100).half(); csg = torch.rand(12288, device=dev) * 0.01; ra = torch.rand(T, device=dev) * 0.01
    q6 = torch.empty(T, 3072, device=dev, dtype=torch.uint8); s = torch.empty(T, device=dev)
    for nw in (4, 8, 16):
        for stg in (1, 2, 3):
            try:
                t = tmk(lambda: K._swiglu_k[(T,)](GU, ra, csg, R4.sign, M['P12T'], M['H16'], M['H32'], q6, s, GU, T, DQ=True, OUTQ=True, QMAX=7., CLIP=.9, BITS=4, PREC=3, ROWS=1, num_warps=nw, num_stages=stg))
                print(f'T={T} swiglu nw{nw} stages{stg}: {t:.1f} us  ({(T * 12288 * 2 + T * 3072) / t / 1e3:.0f} GB/s)', flush=True)
            except Exception as ex: print('fail', nw, stg, str(ex)[:100])
    o = torch.randn(T, 16, 128, device=dev).bfloat16(); P = (torch.randn(T, 8224, device=dev) * 100).half(); csP = torch.rand(8224, device=dev) * 0.01
    gw = (torch.rand(128, device=dev) + 0.5).bfloat16(); q = torch.empty(T, 1024, device=dev, dtype=torch.uint8); R2 = RT.Rot(2048, 1235, dev)
    for nw in (4, 8, 16):
        t = tmk(lambda: K._gnorm_k[(T,)](o, P[:, 6144:], ra, csP, gw, R2.sign, M['H32'], M['H64'], M['H128'], q, s, o, T, P.stride(0), 1e-6, DQ=True, OUTQ=True, HAD=1, QMAX=7., CLIP=.9, BITS=4, PREC=3, ROWS=1, num_warps=nw))
        print(f'T={T} gnorm nw{nw}: {t:.1f} us', flush=True)
    x = (torch.randn(T, 2048, device=dev) * 2).bfloat16(); d = (torch.randn(T, 2048, device=dev) * 300).half(); qa = torch.empty(T, 1024, device=dev, dtype=torch.uint8); cs = torch.rand(2048, device=dev) * .01
    for R_ in (1, 2, 4):
        for nw in (4, 8):
            t = tmk(lambda: K._addq_k[(triton.cdiv(T, R_),)](x, d, ra, cs, qa, s, x, T, 1e-6, HAS_D=True, DQ=True, OUTQ=True, QMAX=7., CLIP=.9, BITS=4, R=R_, num_warps=nw))
            print(f'T={T} addq R{R_} nw{nw}: {t:.1f} us', flush=True)
