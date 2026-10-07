"""q8_cpu.py: AMX int8 GEMM via oneDNN qlinear (prepacked int8 weights, u8/s8 activations) vs prepacked bf16 mkldnn linear."""
import os, time, json, torch
torch.set_num_threads(int(os.environ.get('NT', '16')))


def bench(fn, reps=20):
    for _ in range(5): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]


res = {}
for M in (256, 1000, 4000):
    for K, N in ((2048, 8224), (2048, 12288), (6144, 2048)):
        fl = 2 * M * K * N; r = {}
        x = torch.randn(M, K).bfloat16(); w = (torch.randn(N, K) * 0.02).bfloat16()
        try:  # prepacked bf16 (what inductor freezing uses)
            wp = torch.ops.mkldnn._reorder_linear_weight(w, M)
            t = bench(lambda: torch.ops.mkldnn._linear_pointwise(x, wp, None, 'none', [], ''))
            r['bf16_packed_tflops'] = round(fl / t / 1e12, 2)
        except Exception as e:
            r['bf16p_err'] = str(e)[:100]
        try:
            wq = torch.randint(-127, 127, (N, K), dtype=torch.int8)
            ws = torch.rand(N) * 0.01; wz = torch.zeros(N, dtype=torch.int64)
            packed = torch.ops.onednn.qlinear_prepack(wq.t().contiguous().t(), [M, K]) if False else torch.ops.onednn.qlinear_prepack(wq, [M, K])
            xq = torch.randint(0, 255, (M, K), dtype=torch.uint8)
            def f():
                return torch.ops.onednn.qlinear_pointwise(xq, 0.05, 128, packed, ws, wz, None, 1.0, 0, torch.bfloat16, 'none', [], '')
            t = bench(f)
            r['int8_onednn_tops'] = round(fl / t / 1e12, 2)
        except Exception as e:
            r['int8_err'] = str(e)[:160]
        res[f'{M}x{K}x{N}'] = r; print(M, K, N, r, flush=True)
json.dump(res, open(os.path.expanduser('~/work/j8/q8_cpu.json'), 'w'), indent=1)
