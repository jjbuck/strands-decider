"""per-kernel fixed cost of the int8 GEMMs at short M: n separate GEMMs of depth K vs one GEMM of depth n*K (same FLOPs and bytes)."""
import os, sys, torch, json
sys.path[:0] = [os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/j5')]
import qgemm as QG
dev = 'cuda'; torch.manual_seed(0)
def tg(fn, n=20):
    fn(); torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n): fn()
    e1.record(); e1.synchronize(); return e0.elapsed_time(e1) / n * 1000
out = {}
for M in (16, 140, 364):
    for (nm, N, K) in [('gate_up', 12288, 2048), ('out', 2048, 2048), ('gdn_in', 8224, 2048)]:
        n = 4
        A = torch.randint(-50, 50, (M, K), device=dev, dtype=torch.int8)
        Bs = [torch.randint(-50, 50, (N, K), device=dev, dtype=torch.int8) for _ in range(n)]
        A4 = torch.randint(-50, 50, (M, n * K), device=dev, dtype=torch.int8); B4 = torch.randint(-50, 50, (N, n * K), device=dev, dtype=torch.int8)
        best = {}
        for c in range(11):
            try:
                t1 = tg(lambda: [QG.gemm('s8', A, B, 1 / 4096, c) for B in Bs]) / n
                t4 = tg(lambda: QG.gemm('s8', A4, B4, 1 / 4096, c))
                if 'sep' not in best or t1 < best['sep']: best['sep'] = t1; best['cfg_sep'] = c
                if 'deep' not in best or t4 < best['deep']: best['deep'] = t4; best['cfg_deep'] = c
            except Exception: pass
        best['per_kernel_fixed_us'] = round((n * best['sep'] - best['deep']) / (n - 1), 2)
        best['sep'] = round(best['sep'], 2); best['deep_per_K'] = round(best['deep'] / n, 2); best['deep'] = round(best['deep'], 2)
        out[f'{nm}|{M}'] = best; print(nm, M, best, flush=True)
json.dump(out, open(os.path.expanduser('~/work/j5/res_ramp.json'), 'w'), indent=1)
