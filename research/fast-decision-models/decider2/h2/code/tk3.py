import sys, os, torch, statistics as st, triton, triton.language as tl
sys.path[:0] = [os.path.expanduser('~/work/h2')]
import qk as K
dev = 'cuda'; T = 1000; M = K.mats(dev)
def tmk(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record(); f(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
    return st.median(ts)
@triton.jit
def k2d(O, Z, Y, zs, A: tl.constexpr, B: tl.constexpr, HAD: tl.constexpr, H64p, H32p):
    r = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, A)[:, None]; j = tl.arange(0, B)[None, :]
    o = tl.load(O + r * 2048 + i * B + j).to(tl.float32)
    z = tl.load(Z + r * zs + i * B + j).to(tl.float32)
    y = o * z * tl.sigmoid(z)
    if HAD:
        H64 = K._ldm(H64p, 64); H32 = K._ldm(H32p, 32)
        y = tl.dot(y, H64, input_precision="tf32")
        y = tl.dot(H32, y, input_precision="tf32")
    tl.store(Y + r * 2048 + i * B + j, y.to(tl.bfloat16))
@triton.jit
def k1d(O, Z, Y, zs):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, 2048)
    o = tl.load(O + r * 2048 + c).to(tl.float32)
    z = tl.load(Z + r * zs + c).to(tl.float32)
    y = o * z * tl.sigmoid(z)
    tl.store(Y + r * 2048 + c, y.to(tl.bfloat16))
o = torch.randn(T, 2048, device=dev).bfloat16(); P = torch.randn(T, 8224, device=dev).half(); y = torch.empty(T, 2048, device=dev, dtype=torch.bfloat16)
for nw in (4, 8):
    print('1d', nw, tmk(lambda: k1d[(T,)](o, P[:, 6144:], y, P.stride(0), num_warps=nw)))
    print('2d 16x128', nw, tmk(lambda: k2d[(T,)](o, P[:, 6144:], y, P.stride(0), A=16, B=128, HAD=False, H64p=M['H64'], H32p=M['H32'], num_warps=nw)))
    print('2d 32x64', nw, tmk(lambda: k2d[(T,)](o, P[:, 6144:], y, P.stride(0), A=32, B=64, HAD=False, H64p=M['H64'], H32p=M['H32'], num_warps=nw)))
    print('2d 32x64 had', nw, tmk(lambda: k2d[(T,)](o, P[:, 6144:], y, P.stride(0), A=32, B=64, HAD=True, H64p=M['H64'], H32p=M['H32'], num_warps=nw)))
big = torch.randn(64, 1024, 1024, device=dev); big2 = torch.empty_like(big)
print("copy 8MB", tmk(lambda: y.copy_(o)), "copy 512MB", tmk(lambda: big2.copy_(big)))
for nw in (4, 8):
    for TT in (4000, 16000):
        o2 = torch.randn(TT, 2048, device=dev).bfloat16(); P2 = torch.randn(TT, 8224, device=dev).half(); y2 = torch.empty(TT, 2048, device=dev, dtype=torch.bfloat16)
        print("1d T", TT, nw, tmk(lambda: k1d[(TT,)](o2, P2[:, 6144:], y2, P2.stride(0), num_warps=nw)))
