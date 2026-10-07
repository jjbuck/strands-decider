"""Contention-robust diagnostics (safe under the shared lock): CUPTI kernel durations, kernel counts, CPU op counts.
NOT latency measurements: sum-of-kernel-time excludes launch gaps and host time."""
import sys, time, statistics as st, torch, json
from torch.profiler import profile, ProfilerActivity
from common import *
def kprof(fn, reps=7, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    sums, cnts, cpu_ops = [], [], []
    last = None
    for _ in range(reps):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            fn(); torch.cuda.synchronize()
        ev = p.events()
        k = [e for e in ev if e.device_type.name == "CUDA"]
        sums.append(sum(e.device_time for e in k) / 1000); cnts.append(len(k))
        cpu_ops.append(sum(1 for e in ev if e.device_type.name == "CPU" and e.name.startswith("aten::")))
        last = k
    return dict(gpu_busy_ms=round(st.median(sums), 2), kernels=int(st.median(cnts)), aten_ops=int(st.median(cpu_ops))), last
if __name__ == "__main__":
    which = sys.argv[1]
    res = {}
    if which == "gemm":
        dev = "cuda"
        for name, (N, K) in {"gate_up": (12288, 2048), "down": (2048, 6144), "gdn_in": (8224, 2048)}.items():
            W = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
            for M in (16, 64, 256, 512, 1024, 2048, 4096):
                x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
                r, _ = kprof(lambda: x @ W.t())
                t = r["gpu_busy_ms"]; fl = 2 * M * N * K
                print(f"bf16 {name:8s} M={M:5d} kernel {t*1000:8.1f} us  {fl/t/1e9:6.1f} TFLOPS  wt {N*K*2/t/1e6:6.0f} GB/s", flush=True)
                res[f"bf16_{name}_M{M}"] = dict(us=round(t*1000, 1), tflops=round(fl/t/1e9, 1))
        json.dump(res, open("prof2_gemm.json", "w"), indent=1)
    if which == "gemm2":   # int8 / sparse / fp16 kernel times (profiler-based)
        dev = "cuda"
        from torch.sparse import to_sparse_semi_structured
        for name, (N, K) in {"gate_up": (12288, 2048), "down": (2048, 6144), "gdn_in": (8224, 2048)}.items():
            W = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
            mask = torch.tensor([1, 1, 0, 0], device=dev, dtype=torch.bool).tile((N, K // 4))
            Wm = W * mask
            try: Ws = to_sparse_semi_structured(Wm)
            except Exception as e: Ws = None; print("sparse convert fail", repr(e)[:100])
            Wq = (W / W.abs().amax(1, keepdim=True) * 127).round().to(torch.int8); sc = (W.abs().amax(1) / 127).to(torch.bfloat16)
            Wb = W.to(torch.bfloat16)
            for M in (16, 64, 256, 1024, 2048):
                x = torch.randn(M, K, device=dev, dtype=torch.float16); xb = x.to(torch.bfloat16)
                xq = torch.randint(-127, 127, (max(M, 32), K), device=dev, dtype=torch.int8)
                out = {}
                out["fp16"] = kprof(lambda: x @ W.t())[0]["gpu_busy_ms"]
                out["bf16"] = kprof(lambda: xb @ Wb.t())[0]["gpu_busy_ms"]
                try: out["int_mm"] = kprof(lambda: torch._int_mm(xq, Wq.t()))[0]["gpu_busy_ms"]
                except Exception as e: out["int_mm"] = float("nan")
                try: out["w8pack"] = kprof(lambda: torch._weight_int8pack_mm(xb, Wq, sc))[0]["gpu_busy_ms"]
                except Exception as e: out["w8pack"] = float("nan")
                if Ws is not None:
                    try: out["sp24"] = kprof(lambda: torch.nn.functional.linear(x, Ws))[0]["gpu_busy_ms"]
                    except Exception as e: out["sp24"] = float("nan")
                fl = 2 * M * N * K
                print(f"{name:8s} M={M:5d} " + " | ".join(f"{k} {v*1000:7.1f}us" for k, v in out.items()) + f" | bf16 TFLOPS {fl/out['bf16']/1e9:.1f}", flush=True)
    if which == "model":
        from strands_decider.infer import load_engine
        from lean import Lean
        from runner import FastEngine
        from packed import PackedEngine
        eng = load_engine(CKPT, device="cuda"); tok = eng.model.tokenizer; mk = make_state_fn(tok)
        for n in (64, 1000):
            it = iter([mk(n, f"p{i}") for i in range(40)])
            r, k = kprof(lambda: eng.ask(next(it), qs(1)), reps=5)
            print(f"stock(unmerged LoRA) tokens={n}:", r, flush=True)
        torso = eng.model.torso.merge_and_unload().eval(); eng.model.torso = torso
        for n in (64, 1000):
            it = iter([mk(n, f"q{i}") for i in range(40)])
            r, k = kprof(lambda: eng.ask(next(it), qs(1)), reps=5)
            print(f"stock(merged LoRA) tokens={n}:", r, flush=True)
        ln = Lean(torso)
        for n in (64, 256, 1000, 2000, 4000):
            ids = torch.randint(1000, 100000, (1, n), device="cuda")
            r, k = kprof(lambda: ln.forward(ids), reps=5)
            byname = {}
            for e in k: byname[e.name[:50]] = byname.get(e.name[:50], 0) + e.device_time / 1000
            top = sorted(byname.items(), key=lambda kv: -kv[1])[:6]
            print(f"lean fwd tokens={n}:", r, "top:", [(a, round(b, 2)) for a, b in top], flush=True)
        pe = PackedEngine(eng, ln, graph=False)
        for n in (64, 1000):
            it = iter([mk(n, f"r{i}") for i in range(40)])
            r, k = kprof(lambda: pe.ask(next(it), qs(4)), reps=5)
            print(f"packed q=4 tokens={n}:", r, flush=True)
