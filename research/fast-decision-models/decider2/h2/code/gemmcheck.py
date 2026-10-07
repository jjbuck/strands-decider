import sys, os, json, statistics
sys.path.insert(0, os.path.expanduser('~/work/h2'))
import torch, qgemm as Q
torch.manual_seed(0)
def tm(f, reps=30, warm=8):
    for _ in range(warm): f()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    ts.sort(); return statistics.median(ts)
for (M, N, K) in [(256, 512, 2048), (1000, 2048, 6144), (77, 8224, 2048), (1015, 12288, 2048)]:
    a8 = torch.randint(-127, 128, (M, K), device='cuda'); b4 = torch.randint(-7, 8, (N, K), device='cuda')
    ref = a8.double() @ b4.double().t()
    for cfg in range(5):
        try:
            c = Q.gemm('s8s4', a8.to(torch.int8).contiguous(), Q.pack4(b4), 1.0 / 256, cfg).double() * 256
            print(f's8s4 M={M} N={N} K={K} cfg{cfg}: max rel err {((c - ref).abs().max() / ref.abs().max()).item():.2e}', flush=True)
        except Exception as ex: print('s8s4 cfg', cfg, 'FAILED', str(ex)[:200], flush=True)
    a4 = torch.randint(-7, 8, (M, K), device='cuda'); ref4 = a4.double() @ b4.double().t()
    c = Q.gemm('s4', Q.pack4(a4), Q.pack4(b4), 1.0 / 16, 3).double() * 16
    print(f's4 cfg3 rel err {((c - ref4).abs().max() / ref4.abs().max()).item():.2e}')
SH = {"gdn_in": (8224, 2048), "attn_in": (5120, 2048), "out": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}
out = []
for name, (N, K) in SH.items():
    w = (torch.randn(N, K, device='cuda') * 0.02).bfloat16()
    w8 = torch.randint(-127, 128, (N, K), device='cuda').to(torch.int8).contiguous()
    w4 = Q.pack4(torch.randint(-7, 8, (N, K), device='cuda'))
    for M in (1000, 1015, 4000, 4015):
        xs = [torch.randn(M, K, device='cuda').bfloat16() for _ in range(3)]
        x8 = [torch.randint(-127, 128, (M, K), device='cuda').to(torch.int8).contiguous() for _ in range(3)]
        x4 = [Q.pack4(torch.randint(-7, 8, (M, K), device='cuda')) for _ in range(3)]
        it = [0]
        def nx(L): it[0] = (it[0] + 1) % 3; return L[it[0]]
        r = dict(shape=name, M=M, N=N, K=K, bf16=tm(lambda: nx(xs) @ w.t()))
        for kind, L, wq in (('s8', x8, w8), ('s8s4', x8, w4), ('s4', x4, w4)):
            best = None
            for cfg in range(5):
                try: t = tm(lambda: Q.gemm(kind, nx(L), wq, 1.0, cfg))
                except Exception as ex: continue
                if best is None or t < best[0]: best = (t, cfg)
            r[kind] = best[0]; r[kind + '_cfg'] = best[1]
        print(f"{name:8s} M={M:5d} bf16 {r['bf16']:7.0f} s8 {r['s8']:7.0f}(c{r['s8_cfg']}) s8s4 {r['s8s4']:7.0f}(c{r['s8s4_cfg']}) s4 {r['s4']:7.0f}(c{r['s4_cfg']})  x: {r['bf16']/r['s8']:.2f} {r['bf16']/r['s8s4']:.2f} {r['bf16']/r['s4']:.2f}", flush=True)
        out.append(r)
json.dump(out, open(os.path.expanduser('~/work/h2/res_gemm.json'), 'w'), indent=0)
