import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/j5')]
import sk
torch.manual_seed(0); dev = 'cuda'
def rel(a, b): return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))
print([ (M, sk.blocks(M)) for M in (16, 37, 140, 172, 236, 364, 461, 508, 557, 829)])
for (M, N, K) in [(140, 2048, 6144), (364, 12288, 2048), (37, 8224, 2048), (508, 2048, 2048), (461, 5120, 2048)]:
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16); W = torch.randn(N, K, device=dev) * 0.02
    w = sk.QW('bf16', W.to(torch.bfloat16).contiguous()); ref = A.float() @ W.to(torch.bfloat16).float().t()
    s8 = W.abs().amax(1) / 127; q8 = torch.round(W / s8[:, None]).clamp(-127, 127)
    w8 = sk.QW('w8', q8.to(torch.int8).contiguous(), s8.contiguous())
    Aq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); ra = torch.rand(M, device=dev)
    refi = Aq.double() @ q8.double().t()
    ss = torch.rand(M, device=dev) * K + 1; rs = torch.rsqrt(ss / 2048 + 1e-6)
    for cfg in [(32, 64, 1, 4, 2, 256), (64, 32, 4, 8, 2, 128), (32, 64, 2, 8, 3, 256)]:
        if M * cfg[0] > 64 * 600: cfg = (32,) + cfg[1:]
        try:
            e0 = rel(sk.launch_mt(A, w, 0, cfg=cfg), ref)
        except Exception as ex:
            print(M, N, K, cfg, 'skip', repr(ex)[:80]); continue
        e1 = rel(sk.launch_mt(A, w, 0, ss=ss, cfg=cfg), ref * rs[:, None])
        R = torch.randn(M, N, device=dev, dtype=torch.bfloat16); R0 = R.clone(); sso = torch.zeros(M, device=dev)
        sk.launch_mt(A, w, 3, res=R, ssout=sso, cfg=cfg); refR = (R0.float() + ref.to(torch.bfloat16).float()).to(torch.bfloat16); e2 = rel(R, refR)
        cfgi = (cfg[0], 128, cfg[2], cfg[3], 2, cfg[5])
        e3 = rel(sk.launch_mt(Aq, w8, 4, alpha=1 / 1024, scale=False, cfg=cfgi).float() * 1024, refi)
        y = refi.float() * ra[:, None] * s8[None, :]; g_, u_ = y[:, 0::2], y[:, 1::2]
        e4 = rel(sk.launch_mt(Aq, w8, 5, ra=ra, cfg=cfgi), torch.nn.functional.silu(g_.to(torch.bfloat16).float()).to(torch.bfloat16).float() * u_.to(torch.bfloat16).float())
        print(M, N, K, cfg, 'bf16 %.1e rs %.1e resid %.1e i8 %.1e i8swi %.1e' % (e0, e1, e2, e3, e4), flush=True)
