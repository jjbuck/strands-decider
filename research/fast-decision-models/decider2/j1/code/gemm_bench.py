"""Per-layer GEMM floor for the encoder at small M (exclusive GPU): best Triton tile (with the runtime's fused epilogue) vs cuBLAS plain,
for the 4 projections; x26 gives the GEMM-only time of a request. Also bare weight-streaming time (copy of 4.05 GB)."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encrt as R
dev = 'cuda'
rt = R.EncRT([], None, torch.ones(R.D, device=dev), None)
shapes = dict(qkv=(4096, 2304, 0, True), o=(2304, 2048, 0, False), gu=(18432, 2304, 1, True), d=(2304, 9216, 0, False))
out = {}
for M in (156, 220, 348, 492, 1092, 4092):
    row = {}
    for nm, (N, K, epi, fold) in shapes.items():
        t, kind, c = rt.pick(M, N, K, epi, fold)
        a = torch.randn(M, K, device=dev, dtype=torch.bfloat16); b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        for _ in range(3): a @ b.t()
        e0.record()
        for _ in range(20): a @ b.t()
        e1.record(); torch.cuda.synchronize(); tc = e0.elapsed_time(e1) / 20
        row[nm] = dict(best_ms=round(t, 4), best=kind, cfg=c, cublas_plain_ms=round(tc, 4))
    tot = sum(v['best_ms'] for v in row.values()) * 26; totc = sum(v['cublas_plain_ms'] for v in row.values()) * 26
    fl = 2 * M * sum(N * K for N, K, _, _ in shapes.values()) * 26
    row['x26_best_ms'] = round(tot, 2); row['x26_cublas_ms'] = round(totc, 2); row['tflops_best'] = round(fl / tot / 1e9, 1)
    out[M] = row; print(M, {k: (v if not isinstance(v, dict) else (v['best'], v['best_ms'], v['cublas_plain_ms'])) for k, v in row.items()}, flush=True)
x = torch.empty(int(4.05e9 // 2), dtype=torch.bfloat16, device=dev); y = torch.empty_like(x)
e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
y.copy_(x); e0.record(); y.copy_(x); e1.record(); torch.cuda.synchronize()
out['stream_4.05GB_copy_ms'] = e0.elapsed_time(e1); out['read_floor_ms_est'] = e0.elapsed_time(e1) / 2
print('copy 4.05GB (read+write) ms', out['stream_4.05GB_copy_ms'], flush=True)
json.dump(out, open('gemm_bench.json', 'w'), indent=1)
