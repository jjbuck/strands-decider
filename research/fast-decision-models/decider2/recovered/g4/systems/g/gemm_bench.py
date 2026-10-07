import torch, time, statistics as st, json
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
def bench(fn, reps=30, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return st.median(ts)
res = {}
print("--- dense bf16 / fp16 GEMM: time (ms) and TFLOPS; y = x @ W^T, W [N,K]")
shapes = {"mlp_gate_up(2B)": (12288, 2048), "mlp_down(2B)": (2048, 6144), "gdn_in(2B)": (8224, 2048), "big": (8192, 8192)}
for name, (N, K) in shapes.items():
    for dt in (torch.bfloat16, torch.float16):
        W = torch.randn(N, K, device=dev, dtype=dt) * 0.02
        for M in (16, 64, 256, 512, 1024, 2048, 4096):
            x = torch.randn(M, K, device=dev, dtype=dt)
            t = bench(lambda: x @ W.t())
            fl = 2 * M * N * K; wb = N * K * 2
            print(f"{name:18s} {str(dt)[6:]:9s} M={M:5d} {t*1000:8.1f} us  {fl/t/1e9:7.1f} TFLOPS  {wb/t/1e6:7.1f} GB/s", flush=True)
            res[f"{name}_{str(dt)[6:]}_M{M}"] = dict(us=round(t*1000,1), tflops=round(fl/t/1e9,1))
print("--- memory bandwidth (device copy)")
a = torch.empty(1<<28, dtype=torch.uint8, device=dev); b = torch.empty_like(a)
t = bench(lambda: b.copy_(a)); print(f"copy 256MB: {t:.3f} ms -> {2*a.numel()/t/1e6:.0f} GB/s read+write")
print("--- 2:4 semi-structured sparse (fp16/bf16)")
from torch.sparse import to_sparse_semi_structured
for name, (N, K) in list(shapes.items())[:3]:
    for dt in (torch.float16, torch.bfloat16):
        W = torch.randn(N, K, device=dev, dtype=dt) * 0.02
        mask = torch.tensor([1, 1, 0, 0], device=dev, dtype=torch.bool).tile((N, K // 4))
        Wm = (W * mask)
        try:
            Ws = to_sparse_semi_structured(Wm)
        except Exception as e:
            print("sparse convert failed", name, dt, repr(e)[:120]); continue
        for M in (16, 64, 256, 1024, 2048, 4096):
            x = torch.randn(M, K, device=dev, dtype=dt)
            td = bench(lambda: x @ Wm.t())
            try:
                ts = bench(lambda: torch.nn.functional.linear(x, Ws))
            except Exception as e:
                print("sparse mm failed", repr(e)[:100]); break
            print(f"{name:18s} {str(dt)[6:]:9s} M={M:5d} dense {td*1000:7.1f} us  sparse24 {ts*1000:7.1f} us  speedup {td/ts:.2f}x", flush=True)
            res[f"sp_{name}_{str(dt)[6:]}_M{M}"] = dict(dense_us=round(td*1000,1), sparse_us=round(ts*1000,1))
print("--- int8: _int_mm (W8A8 tensor cores) and weight-only int8pack vs bf16")
for name, (N, K) in list(shapes.items())[:3]:
    W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
    Wq = (W / W.abs().amax(1, keepdim=True) * 127).round().to(torch.int8); sc = (W.abs().amax(1) / 127).to(torch.bfloat16)
    for M in (16, 64, 256, 1024, 2048, 4096):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        td = bench(lambda: x @ W.t())
        xq = torch.randint(-127, 127, (max(M, 17), K), device=dev, dtype=torch.int8)
        try:
            ti = bench(lambda: torch._int_mm(xq, Wq.t()))
        except Exception as e:
            ti = float("nan"); print("int_mm failed", repr(e)[:100])
        try:
            tw = bench(lambda: torch._weight_int8pack_mm(x, Wq, sc))
        except Exception as e:
            tw = float("nan"); print("int8pack failed", repr(e)[:100])
        # W8A8 incl dynamic per-token activation quant + dequant epilogue (torch ops, unfused)
        def w8a8():
            s = x.abs().amax(1, keepdim=True).float() / 127
            xi = (x / s).round().clamp(-127, 127).to(torch.int8)
            y = torch._int_mm(xi if M > 16 else torch.cat([xi, xi[:1].expand(17 - M, -1)]), Wq.t())[:M]
            return (y.float() * s * sc.float()).to(torch.bfloat16)
        try:
            tq = bench(w8a8)
        except Exception as e:
            tq = float("nan")
        print(f"{name:18s} M={M:5d} bf16 {td*1000:7.1f} us | int_mm {ti*1000:7.1f} us ({td/ti:.2f}x) | w8 int8pack {tw*1000:7.1f} us ({td/tw:.2f}x) | w8a8 unfused {tq*1000:7.1f} us", flush=True)
json.dump(res, open("gemm_bench.json", "w"), indent=1)
