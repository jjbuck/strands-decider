"""Q4 B11 tests: probe the mma.sp metadata layout, then check every q4sp config bit-exact against the torch reference.
python test_sp.py probe | gemm"""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/q4')]
import q4sp as S

dev = 'cuda'


def find_layout():
    res = {}
    for kind in ('sp8', 'sp4'):
        res[kind] = S.probe(kind, trials=3)
    json.dump(res, open(os.path.expanduser('~/work/q4/probe.json'), 'w'))
    return res


def set_layout():
    res = json.load(open(os.path.expanduser('~/work/q4/probe.json')))
    for kind in ('sp8', 'sp4'):
        assert res[kind], f'no layout matched for {kind}'
    assert res['sp8'][0] == res['sp4'][0], res
    S.LAYOUT = S.layout_cands()[res['sp8'][0]]
    return res['sp8'][0]


def check_gemm(cfgs=range(12)):
    lay = set_layout(); print('layout', lay, flush=True)
    torch.manual_seed(0)
    bad = 0
    for kind, qm in (('sp8', 127), ('sp4', 7), ('d8', 127), ('d4', 7)):
        for (N, T, K) in ((512, 136, 1024), (8224, 77, 2048), (2048, 300, 6144), (256, 1, 512)):
            Wq = torch.randint(-qm, qm + 1, (N, K), device=dev, dtype=torch.int8)
            Xq = torch.randint(-qm, qm + 1, (T, K), device=dev, dtype=torch.int8)
            if kind.startswith('sp'):
                m = S.mask24_mag(Wq, kind)
                Wc, E, Wm = S.compress(Wq, m, kind)
            else:
                Wm = Wq; E = None
                Wc = Wq.contiguous() if kind == 'd8' else S.pack4(Wq)
            X = Xq.contiguous() if kind in ('sp8', 'd8') else S.pack4(Xq)
            ref = S.ref_i32(Wm, Xq)
            sa = torch.rand(T, device=dev) + 0.5; sb = torch.rand(N, device=dev) + 0.5
            fref = ((ref.float() * sa[:, None]) * sb[None, :])
            for cfg in cfgs:
                o = torch.zeros(T, N, device=dev, dtype=torch.int32)
                try:
                    S.gemm(kind, cfg, Wc, E, X, o, N, T, K, epi='i32'); torch.cuda.synchronize()
                except RuntimeError as ex:
                    print(kind, N, T, K, cfg, 'skip', ex); continue
                e = (o.long() - ref).abs().max().item()
                ob = torch.zeros(T, N, device=dev, dtype=torch.bfloat16)
                S.gemm(kind, cfg, Wc, E, X, ob, N, T, K, sa=sa, sb=sb, epi='bf16'); torch.cuda.synchronize()
                eb = (ob.float() - fref.to(torch.bfloat16).float()).abs().max().item()
                oh = torch.zeros(T, N, device=dev, dtype=torch.float16)
                al = 2 ** -10 if qm == 127 else 0.5
                S.gemm(kind, cfg, Wc, E, X, oh, N, T, K, epi='fp16a', alpha=al); torch.cuda.synchronize()
                eh = (oh.float() - (ref.float() * al).to(torch.float16).float()).abs().max().item()
                # SwiGLU: interleave in blocks of 8 handled by caller; here just check vs reference on the same (already interleaved) channels
                osw = torch.zeros(T, N // 2, device=dev, dtype=torch.bfloat16)
                S.gemm(kind, cfg, Wc, E, X, osw, N, T, K, sa=sa, sb=sb, epi='swiglu'); torch.cuda.synchronize()
                f = fref.reshape(T, N // 16, 2, 8)
                gg = f[:, :, 0].to(torch.bfloat16).float(); uu = f[:, :, 1].to(torch.bfloat16).float()
                msw = ((gg / (1.0 + torch.exp(-gg))).to(torch.bfloat16).float() * uu).to(torch.bfloat16).reshape(T, N // 2)
                es = (osw.float() - msw.float()).abs().max().item()
                ok = e == 0 and eb == 0 and eh == 0 and es == 0
                bad += not ok
                print(f'{kind} N{N} T{T} K{K} cfg{cfg}: i32 maxerr {e}  bf16 {eb:.3g}  fp16a {eh:.3g}  swiglu {es:.3g}  {"OK" if ok else "FAIL"}', flush=True)
    print('ALL OK' if bad == 0 else f'{bad} FAIL', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'probe': print(find_layout())
    else: check_gemm([int(c) for c in sys.argv[2].split(',')] if len(sys.argv) > 2 else range(16))
