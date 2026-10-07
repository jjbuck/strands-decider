"""gemm_cpu.py: AMX GEMM rates at hobson's shapes (bf16 via oneDNN matmul, int8 via torch._int_mm and onednn qlinear)."""
import os, sys, time, json, torch
NT = int(os.environ.get('NT', '16')); torch.set_num_threads(NT)
res = {}


def bench(fn, reps=10):
    for _ in range(3): fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    ts.sort(); return ts[len(ts) // 2]


shapes = [(2048, 8224), (2048, 12288), (6144, 2048), (2048, 2048)]
for M in (64, 256, 1000, 4000):
    for K, N in shapes:
        a = torch.randn(M, K).bfloat16(); w = torch.randn(N, K).bfloat16()
        t = bench(lambda: a @ w.t())
        fl = 2 * M * K * N
        r = dict(bf16_ms=round(t * 1e3, 3), bf16_tflops=round(fl / t / 1e12, 2))
        try:
            ai = torch.randint(-127, 127, (M, K), dtype=torch.int8); wi = torch.randint(-127, 127, (K, N), dtype=torch.int8)
            ti = bench(lambda: torch._int_mm(ai, wi))
            r.update(int8mm_ms=round(ti * 1e3, 3), int8mm_tops=round(fl / ti / 1e12, 2))
        except Exception as e:
            r['int8mm_err'] = str(e)[:80]
        try:
            from torch.ao.nn.quantized.dynamic import Linear as DQL
            torch.backends.quantized.engine = 'onednn'
            lin = torch.nn.Linear(K, N, bias=False)
            dq = torch.ao.quantization.quantize_dynamic(torch.nn.Sequential(lin), {torch.nn.Linear}, dtype=torch.qint8)
            af = a.float()
            td = bench(lambda: dq(af))
            r.update(dyn_q8_ms=round(td * 1e3, 3), dyn_q8_tops=round(fl / td / 1e12, 2))
        except Exception as e:
            r['dynq_err'] = str(e)[:80]
        res[f'{M}x{K}x{N}'] = r
        print(M, K, N, r, flush=True)
json.dump(res, open(os.path.expanduser(f'~/work/j8/gemm_cpu_nt{NT}.json'), 'w'), indent=1)
