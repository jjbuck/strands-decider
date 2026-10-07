"""Uncontended-GPU-time ESTIMATE under a shared GPU: profile CUDA-graph replays with CUPTI; per-kernel-instance MIN over reps
(preemption/time-slicing only ever inflates a duration), plus min inter-kernel gap. Not a wall-clock latency measurement."""
import sys, time, statistics as st, torch, json
from torch.profiler import profile, ProfilerActivity
from common import *
def est(fn, reps=14, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    per = []
    for _ in range(reps):
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn(); torch.cuda.synchronize()
        ks = sorted([(e.time_range.start, e.time_range.end, e.name) for e in p.events() if e.device_type.name == "CUDA" and e.time_range.end > e.time_range.start])
        per.append(ks)
    n = min(len(k) for k in per)
    cnt = st.median([len(k) for k in per])
    cons = [k for k in per if len(k) == n][: reps]
    dur = [min(k[i][1] - k[i][0] for k in cons) for i in range(n)]
    gaps = [min(max(0, k[i + 1][0] - k[i][1]) for k in cons) for i in range(n - 1)]
    span = [k[-1][1] - k[0][0] for k in cons]
    def catof(nm):
        l = nm.lower()
        if any(x in l for x in ("gemm", "cutlass", "xmma", "cublas", "s16816", "s1688")): return "gemm"
        if any(x in l for x in ("flash", "efficient_attention", "fmha", "attention")): return "attn"
        if "conv" in l: return "conv"
        if any(x in l for x in ("chunk", "solve_tril", "recompute", "fwd_prepare", "kkt", "gated_delta", "cumsum", "l2norm", "fused_recurrent")): return "fla_gdn"
        if "triton" in l: return "triton_fused"
        return "elementwise/other"
    cats = {}
    for i in range(n): cats[catof(cons[0][i][2])] = cats.get(catof(cons[0][i][2]), 0) + dur[i] / 1000
    return dict(cat_ms={k: round(v, 2) for k, v in sorted(cats.items(), key=lambda kv: -kv[1])}, kernels=n, busy_ms=round(sum(dur) / 1000, 2), gaps_ms=round(sum(gaps) / 1000, 2), est_ms=round((sum(dur) + sum(gaps)) / 1000, 2),
                span_min_ms=round(min(span) / 1000, 2), span_med_ms=round(st.median(span) / 1000, 2))
def cap(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): out = fn()
    return g, out
if __name__ == "__main__":
    what = sys.argv[1]; toks = [int(x) for x in sys.argv[2].split(",")]
    from strands_decider.infer import load_engine
    import transformers
    from lean import Lean
    from strands_decider.modeling import StrandsDeciderModel
    res = {}
    class _E: pass
    eng = _E(); eng.model = None
    def _load_merged():
        m = StrandsDeciderModel.load(CKPT)           # CPU
        t = m.torso.merge_and_unload().eval()
        m.torso = t; eng.model = m
        return t.cuda()
    if what in ("2b", "2b_compile", "2b_hf"):
        torso = _load_merged()
        if what == "2b_hf":
            f = lambda ids: torso(input_ids=ids, use_cache=False).last_hidden_state
        else:
            if what == "2b_compile":
                import torch._dynamo; torch._dynamo.config.cache_size_limit = 64
            ln = Lean(torso, compile_glue=(what == "2b_compile")); f = ln.forward
        for n in toks:
            ids = torch.randint(1000, 100000, (1, n), device="cuda")
            g, out = cap(lambda: f(ids))
            r = est(g.replay); res[f"{what}_T{n}"] = r; print(what, "tokens", n, r, flush=True)
            del g, out
    if what == "2b_packed":
        from packed import PackedEngine
        torso = _load_merged()
        ln = Lean(torso)
        for n in toks:
            s_ids = torch.randint(1000, 100000, (n,), device="cuda"); n_s = torch.tensor(n - 7, device="cuda")
            b_ids = torch.randint(1000, 100000, (4, 64), device="cuda")
            g, out = cap(lambda: ln.forward_packed(s_ids, n_s, b_ids)[1])
            r = est(g.replay); res[f"packed4_T{n}"] = r; print("packed q=4 state tokens", n, "+4x64", r, flush=True)
            del g, out
    if what in ("0.8b", "0.6b"):
        if what == "0.8b":
            name = "Qwen/Qwen3.5-0.8B"; cfg = transformers.AutoConfig.from_pretrained(name)
            torso = transformers.Qwen3_5ForCausalLM.from_pretrained(name, config=cfg.get_text_config(), dtype=torch.bfloat16).cuda().eval().model
            f = Lean(torso).forward
        else:
            torso = transformers.AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).cuda().eval()
            f = lambda ids: torso(input_ids=ids, use_cache=False).last_hidden_state
        for n in toks:
            ids = torch.randint(1000, 100000, (1, n), device="cuda")
            g, out = cap(lambda: f(ids))
            r = est(g.replay); res[f"{what}_T{n}"] = r; print(what, "tokens", n, r, flush=True)
            del g, out
    json.dump(res, open(f"prof3_{what}.json", "w"), indent=1)
