import sys, os, torch, statistics as st
sys.path[:0] = [os.path.expanduser('~/work/h2')]
import qk as K, rot as RT
dev = 'cuda'; T = 1000; M = K.mats(dev)
def tmk(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record(); f(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
    return st.median(ts)
o = torch.randn(T, 16, 128, device=dev).bfloat16(); P = (torch.randn(T, 8224, device=dev) * 100).half(); csP = torch.rand(8224, device=dev) * 0.01
gw = (torch.rand(128, device=dev) + 0.5).bfloat16(); ra = torch.rand(T, device=dev) * 0.01; s = torch.empty(T, device=dev)
R2 = RT.Rot(2048, 1235, dev); q = torch.empty(T, 1024, device=dev, dtype=torch.uint8); yb = torch.empty(T, 2048, device=dev, dtype=torch.bfloat16)
for had in (0, 1, 2):
    for prec in (1, 2):
        for nw in (2, 4, 8):
            t = tmk(lambda: K._gnorm_k[(T,)](o, P[:, 6144:], ra, csP, gw, R2.sign, M['H32'], M['H64'], M['H128'], q, s, o, T, P.stride(0), 1e-6, DQ=True, OUTQ=True, HAD=had, QMAX=7., CLIP=.9, BITS=4, PREC=prec, num_warps=nw))
            print(f'gnorm had{had} prec{prec} nw{nw}: {t:.1f} us', flush=True)
t = tmk(lambda: K._gnorm_k[(T,)](o, P[:, 6144:], ra, csP, gw, R2.sign, M['H32'], M['H64'], M['H128'], q, s, yb, T, P.stride(0), 1e-6, DQ=True, OUTQ=False, HAD=1, QMAX=7., CLIP=.9, BITS=4, num_warps=8))
print('gnorm no quant (bf16 out)', t)
