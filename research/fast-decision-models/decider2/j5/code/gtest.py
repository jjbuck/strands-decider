"""correctness of sk._sk against torch references (random data), every mode / epilogue / split."""
import os, sys, torch
sys.path[:0] = [os.path.expanduser('~/work/j5')]
os.environ['SKTUNE'] = '0'
import sk
torch.manual_seed(0); dev = 'cuda'
def rel(a, b): return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-9))
for (M, N, K) in [(140, 2048, 6144), (37, 12288, 2048), (172, 8224, 2048), (16, 2048, 2048)]:
    A = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=dev) * 0.02
    for split in (1, 4):
        for BM, BN in ((32, 128), (64, 64), (16, 256)):
            cfg = (BM, BN, 64, split, 4, 3)
            if (K // split) % 128: continue
            # bf16
            w = sk.QW('bf16', W.to(torch.bfloat16).contiguous())
            ref = A.float() @ W.to(torch.bfloat16).float().t()
            c = sk.launch(A, w, 0, cfg=cfg); e0 = rel(c, ref)
            # w8 per-channel (A bf16)
            s8 = W.abs().amax(1) / 127; q8 = torch.round(W / s8[:, None]).clamp(-127, 127)
            w8 = sk.QW('w8', q8.to(torch.int8).contiguous(), s8.contiguous())
            ref8 = A.float() @ (q8 * s8[:, None]).t()
            c8 = sk.launch(A, w8, 0, cfg=(BM, BN, 128, split, 4, 3)); e1 = rel(c8, ref8)
            # w4 g128 / g64 asym
            es = []
            for G in (128, 64):
                Wg = W.reshape(N, K // G, G); mx = Wg.amax(-1); mn = Wg.amin(-1); sc = (mx - mn) / 15; z = torch.round(-mn / sc).clamp(0, 15)
                q = (torch.round(Wg / sc[..., None]) + z[..., None]).clamp(0, 15)
                dq = ((q - z[..., None]) * sc[..., None]).reshape(N, K)
                w4 = sk.QW('w4', sk.pack_blocked4(q.reshape(N, K)), sc.contiguous(), z.to(torch.uint8).contiguous(), G=G)
                assert torch.equal(sk.unpack_blocked4(w4.w).float(), q.reshape(N, K))
                ref4 = A.float() @ dq.t()
                es.append(rel(sk.launch(A, w4, 0, cfg=(BM, BN, 128, split, 4, 3)), ref4))
            # epilogues on bf16: rowscale + swiglu, residual+sumsq
            ss = torch.rand(M, device=dev) * K + 1
            rs = torch.rsqrt(ss / 2048 + 1e-6)
            refrs = ref * rs[:, None]
            crs = sk.launch(A, w, 0, ss=ss, cfg=cfg); e2 = rel(crs, refrs)
            g_, u_ = refrs[:, 0::2], refrs[:, 1::2]
            refsw = (torch.nn.functional.silu(g_.to(torch.bfloat16).float()).to(torch.bfloat16).float() * u_.to(torch.bfloat16).float())
            csw = sk.launch(A, w, 1, ss=ss, cfg=cfg); e3 = rel(csw, refsw)
            R = torch.randn(M, N, device=dev, dtype=torch.bfloat16); R0 = R.clone(); sso = torch.zeros(M, device=dev)
            sk.launch(A, w, 3, res=R, ssout=sso, cfg=cfg)
            refR = (R0.float() + ref.to(torch.bfloat16).float()).to(torch.bfloat16)
            e4 = rel(R, refR); e5 = rel(sso, refR.float().pow(2).sum(1))
            # int8 activations: W8A8 raw (EPI4) and W4A8 g128 scaled
            Aq = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); ra = torch.rand(M, device=dev)
            refi = (Aq.double() @ q8.double().t())
            ci = sk.launch(Aq, w8, 4, alpha=1 / 1024, scale=False, cfg=(BM, BN, 128, split, 4, 3)); e6 = rel(ci.float() * 1024, refi)
            ci5 = sk.launch(Aq, w8, 5, ra=ra, cfg=(BM, BN, 128, split, 4, 3))
            y = refi.float() * ra[:, None] * s8[None, :]; g_, u_ = y[:, 0::2], y[:, 1::2]
            e7 = rel(ci5, torch.nn.functional.silu(g_.to(torch.bfloat16).float()).to(torch.bfloat16).float() * u_.to(torch.bfloat16).float())
            print(M, N, K, 'split', split, 'BM,BN', BM, BN, 'bf16 %.1e w8 %.1e w4g128 %.1e w4g64 %.1e rs %.1e swiglu %.1e resid %.1e ss %.1e i8 %.1e i8swi %.1e' % (e0, e1, es[0], es[1], e2, e3, e4, e5, e6, e7), flush=True)
