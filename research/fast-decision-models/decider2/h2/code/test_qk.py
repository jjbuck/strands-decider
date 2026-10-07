import sys, os, torch, torch.nn.functional as F
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g')]
import qk as K, rot as RT
torch.manual_seed(0); dev = 'cuda'
M = K.mats(dev)
def unpack4(b):
    lo = (b & 15).to(torch.int16); hi = ((b >> 4) & 15).to(torch.int16)
    lo = torch.where(lo >= 8, lo - 16, lo); hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], -1).reshape(b.shape[0], -1)
def qref(x, qmax, clip):
    s = x.abs().amax(-1).clamp_min(1e-8) / qmax * clip
    return torch.round(x / s[:, None]).clamp(-qmax, qmax), s
def cmp(name, q, s, qr, sr, bits):
    qq = unpack4(q) if bits == 4 else q.to(torch.int16)
    mis = (qq.float() != qr).float().mean().item(); off = (qq.float() - qr).abs().max().item()
    print(f'{name:28s} bits{bits}: code mismatch {mis:.2e} (max |d| {off:.0f}), scale rel err {((s - sr).abs() / sr).max().item():.2e}', flush=True)
T = 1000
for bits, qmax, clip in ((8, 127., 1.), (4, 7., .9)):
    # K5 addq
    x = (torch.randn(T, 2048, device=dev) * 2).bfloat16(); d = (torch.randn(T, 2048, device=dev) * 300).half()
    ra = torch.rand(T, device=dev) * 0.01 + 1e-3; cs = torch.rand(2048, device=dev) * 0.01
    y = (d.float() * ra[:, None] * cs[None, :]).bfloat16(); x2 = (x.float() + y.float()).bfloat16()
    xf = x2.float(); xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    qr, sr = qref(xn, qmax, clip)
    xx = x.clone(); q = torch.empty(T, 2048 if bits == 8 else 1024, device=dev, dtype=torch.int8 if bits == 8 else torch.uint8); s = torch.empty(T, device=dev)
    K.addq(xx, d, ra, cs, q, s, qmax=qmax, clip=clip, bits=bits)
    print('addq residual exact:', torch.equal(xx, x2))
    cmp('addq', q, s, qr, sr, bits)
    # K3 gnorm + R2 full / per-head
    o = torch.randn(T, 16, 128, device=dev).bfloat16(); P = (torch.randn(T, 8224, device=dev) * 100).half(); csP = torch.rand(8224, device=dev) * 0.01
    gw = (torch.rand(128, device=dev) + 0.5).bfloat16()
    of = o.reshape(-1, 128).float(); of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + 1e-6)
    z = (P[:, 6144:8192].float() * ra[:, None] * csP[None, 6144:8192]).bfloat16()
    yy = ((gw * of.to(torch.bfloat16)).float() * F.silu(z.reshape(-1, 128).float())).bfloat16().reshape(T, 2048)
    for had, R2 in ((1, RT.Rot(2048, 1235, dev)), (2, RT.Rot(2048, 1235, dev, block=128))):
        qr, sr = qref(R2(yy.float()), qmax, clip)
        q.zero_()
        K._gnorm_k[((T + 3) // 4,)](o, P[:, 6144:], ra, csP, gw, R2.sign, M['H32'], M['H64'], M['H128'], q, s, o, T, P.stride(0), 1e-6,
                                                  DQ=True, OUTQ=True, HAD=had, QMAX=qmax, CLIP=clip, BITS=bits, num_warps=8)
        cmp(f'gnorm had{had}', q, s, qr, sr, bits)
    # K3b attention gate
    P5 = (torch.randn(T, 5120, device=dev) * 100).half(); cs5 = torch.rand(5120, device=dev) * 0.01
    oa = torch.randn(8, T, 256, device=dev).bfloat16()
    gate = (P5.float() * ra[:, None] * cs5[None, :]).bfloat16()[:, :4096].reshape(T, 8, 512)[..., 256:]
    ya = (oa.transpose(0, 1) * torch.sigmoid(gate)).reshape(T, 2048)
    for had, R2 in ((1, RT.Rot(2048, 1235, dev)), (2, RT.Rot(2048, 1235, dev, block=256))):
        qr, sr = qref(R2(ya.float()), qmax, clip)
        K._agate_k[((T + 3) // 4,)](oa, P5, ra, cs5, R2.sign, M['H32'], M['H64'], M['H16'], q, s, oa, T, P5.stride(0), oa.stride(0), oa.stride(1),
                                   DQ=True, OUTQ=True, HAD=had, QMAX=qmax, CLIP=clip, BITS=bits, num_warps=8)
        cmp(f'agate had{had}', q, s, qr, sr, bits)
    # K4 swiglu + R4
    GU = (torch.randn(T, 12288, device=dev) * 100).half(); csg = torch.rand(12288, device=dev) * 0.01
    gu = (GU.float() * ra[:, None] * csg[None, :]).bfloat16()
    m = F.silu(gu[:, :6144]) * gu[:, 6144:]
    R4 = RT.Rot(6144, 1236, dev)
    qr, sr = qref(R4(m.float()), qmax, clip)
    q6 = torch.empty(T, 6144 if bits == 8 else 3072, device=dev, dtype=torch.int8 if bits == 8 else torch.uint8)
    K._swiglu_k[((T + 3) // 4,)](GU, ra, csg, R4.sign, M['P12T'], M['H16'], M['H32'], q6, s, GU, T, DQ=True, OUTQ=True, QMAX=qmax, CLIP=clip, BITS=bits, num_warps=8)
    cmp('swiglu R4', q6, s, qr, sr, bits)
# conv vs lean2 conv_l2 (DQ=False, no tails)
import lean2 as L2
proj = (torch.randn(T, 8224, device=dev)).bfloat16(); w = (torch.randn(6144, 4, device=dev) * 0.3).bfloat16()
ref = L2.conv_l2(proj, w)
prev = torch.stack([torch.arange(T, device=dev) - 3 + j for j in range(3)], 1).int(); prev = torch.where(prev < 0, -1, prev).int().contiguous()
tail = torch.zeros(1, 6144, device=dev)
out, ab = K.conv(proj, None, None, w, prev, tail, dq=False)
print('conv vs lean2 max abs diff', (out.float() - ref.float()).abs().max().item(), 'ab exact', torch.equal(ab, proj[:, 8192:]))
# tails: second half with tail from the first half
T2 = 600; p2 = proj[T2:].contiguous()
prev2 = torch.stack([torch.arange(T - T2, device=dev) - 3 + j for j in range(3)], 1)
prev2 = torch.where(prev2 < 0, -2 - (prev2 + 3), prev2).int().contiguous()
tail2 = proj[T2 - 3:T2, :6144].float().contiguous()
o2, _ = K.conv(p2, None, None, w, prev2, tail2, dq=False)
print('conv with tail vs full max abs diff', (o2.float() - ref[:, T2:].float()).abs().max().item())
# aprep vs lean2 attn_prep
pa = torch.randn(T, 5120, device=dev).bfloat16(); qn = torch.randn(256, device=dev) * 0.1; kn = torch.randn(256, device=dev) * 0.1
inv = 1.0 / (10000000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=dev) / 64))
fr = torch.arange(T, device=dev).float()[:, None] * inv[None, :]; fr = torch.cat([fr, fr], -1); cos, sin = fr.cos().bfloat16().contiguous(), fr.sin().bfloat16().contiguous()
q0, k0, g0 = L2.attn_prep(pa, qn, kn, cos, sin, 1e-6)
kb = torch.zeros(T + 7, 2, 256, device=dev, dtype=torch.bfloat16); vb = torch.zeros_like(kb)
qq = K.aprep(pa, None, None, qn, kn, cos, sin, kb, vb, 7, dq=False)
print('aprep q exact', torch.equal(qq, q0), 'k exact', torch.equal(kb[7:], k0), 'v exact', torch.equal(vb[7:], pa[:, 4608:].reshape(T, 2, 256)))
# ---- timing of the glue kernels at T=1000 (us per call) for PREC 0 (tf32x3) / 1 (tf32) / 2 (fp16)
import statistics as st
def tmk(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record(); f(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000)
    return st.median(ts)
R2 = RT.Rot(2048, 1235, dev); R4 = RT.Rot(6144, 1236, dev)
for bits in (4, 8):
    qmax, clip = (7., .9) if bits == 4 else (127., 1.)
    q = torch.empty(T, 2048 if bits == 8 else 1024, device=dev, dtype=torch.int8 if bits == 8 else torch.uint8)
    q6 = torch.empty(T, 6144 if bits == 8 else 3072, device=dev, dtype=torch.int8 if bits == 8 else torch.uint8)
    for prec in (1, 2, 3):
        for nw in (8,):
            t1 = tmk(lambda: K._swiglu_k[((T + 3) // 4,)](GU, ra, csg, R4.sign, M['P12T'], M['H16'], M['H32'], q6, s, GU, T, DQ=True, OUTQ=True, QMAX=qmax, CLIP=clip, BITS=bits, PREC=prec, num_warps=nw))
            t2 = tmk(lambda: K._gnorm_k[((T + 3) // 4,)](o, P[:, 6144:], ra, csP, gw, R2.sign, M['H32'], M['H64'], M['H128'], q, s, o, T, P.stride(0), 1e-6, DQ=True, OUTQ=True, HAD=1, QMAX=qmax, CLIP=clip, BITS=bits, PREC=prec, num_warps=nw))
            t3 = tmk(lambda: K._gnorm_k[((T + 3) // 4,)](o, P[:, 6144:], ra, csP, gw, R2.sign, M['H32'], M['H64'], M['H128'], q, s, o, T, P.stride(0), 1e-6, DQ=True, OUTQ=True, HAD=2, QMAX=qmax, CLIP=clip, BITS=bits, PREC=prec, num_warps=nw))
            t4 = tmk(lambda: K._agate_k[((T + 3) // 4,)](oa, P5, ra, cs5, R2.sign, M['H32'], M['H64'], M['H16'], q, s, oa, T, P5.stride(0), oa.stride(0), oa.stride(1), DQ=True, OUTQ=True, HAD=1, QMAX=qmax, CLIP=clip, BITS=bits, PREC=prec, num_warps=nw))
            # accuracy of swiglu codes at this precision
            K._swiglu_k[((T + 3) // 4,)](GU, ra, csg, R4.sign, M['P12T'], M['H16'], M['H32'], q6, s, GU, T, DQ=True, OUTQ=True, QMAX=qmax, CLIP=clip, BITS=bits, PREC=prec, num_warps=nw)
            qr, sr = qref(R4(m.float()), qmax, clip); qq = unpack4(q6) if bits == 4 else q6.to(torch.int16)
            print(f'bits{bits} prec{prec} nw{nw}: swiglu {t1:6.1f}  gnorm-full {t2:6.1f}  gnorm-head {t3:6.1f}  agate {t4:6.1f} us; swiglu code mismatch {(qq.float() != qr).float().mean().item():.1e}', flush=True)
x = (torch.randn(T, 2048, device=dev) * 2).bfloat16(); d = (torch.randn(T, 2048, device=dev) * 300).half(); qa = torch.empty(T, 1024, device=dev, dtype=torch.uint8)
for R_ in (1, 2, 4):
    print('addq R', R_, tmk(lambda: K.addq(x, d, ra, cs, qa, s, qmax=7., clip=.9, bits=4, R=R_)))
pr = torch.randn(T, 8224, device=dev).half()
print('conv', tmk(lambda: K.conv(pr, ra, csP, w, prev, tail, dq=True)))
