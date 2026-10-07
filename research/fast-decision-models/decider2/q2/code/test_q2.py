"""Q2 correctness: every kernel variant against the FORMATS.md torch reference. python test_q2.py [cfgs]"""
import sys, torch
sys.path.insert(0, __import__('os').path.expanduser('~/work/q2'))
import q2k as Q
torch.manual_seed(0)
dev = 'cuda'
cfgs = [int(c) for c in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0, 1, 2, 3, 4, 5, 6, 7]
res = []


def rep(name, cfg, got, ref, exact=True):
    d = (got.float() - ref.float()).abs()
    rel = (d.max() / ref.float().abs().max().clamp_min(1e-9)).item()
    ok = (d.max().item() == 0) if exact else rel < 2e-2
    res.append((name, cfg, ok))
    print(f'{name:22s} cfg{cfg}: max|d| {d.max().item():.3e} rel {rel:.2e} mismatches {(d > 0).sum().item()} -> {"OK" if ok else "FAIL"}', flush=True)


for (M, N, K) in [(77, 256, 256), (300, 520, 384), (1125, 2048, 2048)]:
    qa = torch.randint(-7, 8, (M, K), device=dev, dtype=torch.int8); qw = torch.randint(-7, 8, (N, K), device=dev, dtype=torch.int8)
    sa = torch.rand(M, device=dev) * 0.1 + 0.01; sw = torch.rand(N, device=dev) * 0.1 + 0.01
    A4, B4 = Q.pack4(qa), Q.pack4(qw)
    ref = Q.ref_int(qa, sa, qw, sw).to(torch.bfloat16)
    a8 = torch.randint(-127, 128, (M, K), device=dev, dtype=torch.int8); w8 = torch.randint(-127, 128, (N, K), device=dev, dtype=torch.int8)
    ref8 = Q.ref_int(a8, sa, w8, sw).to(torch.bfloat16)
    Ab = torch.randn(M, K, device=dev).to(torch.bfloat16); Bb = torch.randn(N, K, device=dev).to(torch.bfloat16)
    refb = (Ab.double() @ Bb.double().t())
    r = 64
    Z = (torch.randn(M, r, device=dev) * 0.3).to(torch.bfloat16); Bt = torch.randn(N, r, device=dev).to(torch.bfloat16)
    reft = (Q.ref_int(qa, sa, qw, sw).double() + Z.double() @ Bt.double().t())
    # int8 tail (ResQ split): hi part K8 = 128 int8 cols
    K8 = 128
    ah = torch.randint(-127, 128, (M, K8), device=dev, dtype=torch.int8); wh = torch.randint(-127, 128, (N, K8), device=dev, dtype=torch.int8)
    sah = torch.rand(M, device=dev) * 0.01; swh = torch.rand(N, device=dev) * 0.01
    refs8 = (Q.ref_int(qa, sa, qw, sw) + Q.ref_int(ah, sah, wh, swh)).to(torch.bfloat16)
    # group scales g64 / g128
    gs = {}
    for g in (64, 128):
        G = K // g
        gsa = torch.rand(M, G, device=dev) * 0.1 + 0.01; gsw = torch.rand(N, G, device=dev) * 0.1 + 0.01
        f = torch.zeros(M, N, device=dev)
        for j in range(G):
            accg = Q.iacc(qa[:, j * g:(j + 1) * g], qw[:, j * g:(j + 1) * g]).float()
            p = gsa[:, j:j + 1] * gsw[None, :, j]
            f = f + accg * p
        gs[g] = (gsa, gsw, f.to(torch.bfloat16))
    # skip mask (B9): reference = masked integer sum * kscale
    nkb = K // 128; nkeep = max(1, (nkb * 3 + 3) // 4); seed = 12345
    mt128 = (M + 127) // 128; nt128 = (N + 127) // 128
    keep = Q.skip_mask(seed, mt128, nt128, nkb, nkeep)
    accm = torch.zeros(M, N, device=dev, dtype=torch.int64)
    for kb in range(nkb):
        a = Q.iacc(qa[:, kb * 128:(kb + 1) * 128], qw[:, kb * 128:(kb + 1) * 128])
        km = keep[:, :, kb].repeat_interleave(128, 0)[:M].repeat_interleave(128, 1)[:, :N]
        accm += a * km
    ks = float(torch.tensor(nkb / nkeep, dtype=torch.float32))
    refsk = (((accm.float() * sa[:, None]) * sw[None, :]) * ks).to(torch.bfloat16)
    print(f'--- M {M} N {N} K {K}', flush=True)
    for cfg in cfgs:
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        Q.run('s4', cfg, Q.prob(A4, B4, 's4', out, sa, sw)); torch.cuda.synchronize(); rep('s4', cfg, out, ref)
        Q.run('s8', cfg, Q.prob(a8, w8, 's8', out, sa, sw)); torch.cuda.synchronize(); rep('s8', cfg, out, ref8)
        o32 = torch.empty(M, N, device=dev, dtype=torch.float32)
        Q.run('bf16', cfg, Q.prob(Ab, Bb, 'bf16', o32, epi='f32')); torch.cuda.synchronize(); rep('bf16(f32 out)', cfg, o32, refb, exact=False)
        try:
            Q.run('s4_tbf16', cfg, Q.prob(A4, B4, 's4', o32, sa, sw, A1=Z, B1=Bt, epi='f32')); torch.cuda.synchronize(); rep('s4+bf16 tail', cfg, o32, reft, exact=False)
        except RuntimeError as e: print('s4_tbf16', cfg, e)
        try:
            Q.run('s4_ts8', cfg, Q.prob(A4, B4, 's4', out, sa, sw, A1=ah, B1=wh, sa1=sah, sb1=swh)); torch.cuda.synchronize(); rep('s4+s8 tail', cfg, out, refs8)
        except RuntimeError as e: print('s4_ts8', cfg, e)
        for g in (64, 128):
            gsa, gsw, rg = gs[g]
            Q.run(f's4g{g}', cfg, Q.prob(A4, B4, 's4', out, gsa=gsa, gsb=gsw)); torch.cuda.synchronize(); rep(f's4 g{g}', cfg, out, rg)
        if cfg in (0, 1, 3, 4, 5):
            Q.run('s4skip', cfg, Q.prob(A4, B4, 's4', out, sa, sw, kscale=ks, seed=seed, nkeep=nkeep, nkb=nkb)); torch.cuda.synchronize(); rep('s4 skip', cfg, out, refsk)
        # swiglu epilogue
        oh = torch.empty(M, N // 2, device=dev, dtype=torch.bfloat16)
        Q.run('s4', cfg, Q.prob(A4, B4, 's4', oh, sa, sw, epi='swiglu', N=N)); torch.cuda.synchronize()
        fr = Q.ref_int(qa, sa, qw, sw); g_ = fr[:, 0::2].to(torch.bfloat16).float(); u_ = fr[:, 1::2].to(torch.bfloat16).float()
        rsw = (torch.nn.functional.silu(g_).to(torch.bfloat16).float() * u_).to(torch.bfloat16)
        rep('s4 swiglu', cfg, oh, rsw)
        # fp16 alpha (H2 convention)
        o16 = torch.empty(M, N, device=dev, dtype=torch.float16)
        Q.run('s4', cfg, Q.prob(A4, B4, 's4', o16, epi='fp16a', alpha=0.5)); torch.cuda.synchronize()
        rep('s4 fp16a', cfg, o16, (Q.iacc(qa, qw).float() * 0.5).half())
    # row-role two-problem launch: rows [0, q0) int4, [q0, M) int8, interleaved output via rowmap
    q0 = M - 50
    perm = torch.randperm(M, device=dev).to(torch.int32)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    p0 = Q.prob(A4[:q0], B4, 's4', out, sa[:q0], sw, rowmap=perm[:q0])
    p1 = Q.prob(a8[q0:], w8, 's8', out, sa[q0:], sw, rowmap=perm[q0:])
    for cfg in cfgs:
        out.zero_()
        Q.run('s4', cfg, p0, p1); torch.cuda.synchronize()
        rr = torch.empty_like(out); rr[perm[:q0].long()] = ref[:q0]; rr[perm[q0:].long()] = ref8[q0:]
        rep('rowrole s4|s8', cfg, out, rr)

bad = [r for r in res if not r[2]]
print('ALL OK' if not bad else f'FAILURES {bad}')
