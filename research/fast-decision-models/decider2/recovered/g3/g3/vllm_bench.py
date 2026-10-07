"""vLLM fused_moe (Triton, the production grouped-GEMM kernel) on G3's recorded real routings, same shapes; run in ~/vvenv.
usage: ~/vvenv/bin/python vllm_bench.py results/route_MODEL_T.pt  -> per-layer expert time (CUDA graph over all layers)"""
import os, sys, time, json, statistics as st, torch
IMPL = sys.argv[2] if len(sys.argv) > 2 else "vllm"
if IMPL == "vllm":
    from vllm.model_executor.layers.fused_moe import fused_experts
else:
    import os, types; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import moe_lean as ML
R = torch.load(sys.argv[1]); T, E, k, d, ffn = R["T"], R["E"], R["k"], R["d"], R["ffn"]
L = len(R["route"])
w1 = [torch.randn(E, 2 * ffn, d, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(L)]
w2 = [torch.randn(E, d, ffn, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(L)]
x = torch.randn(T, d, device="cuda", dtype=torch.bfloat16)
rt = [(w.view(T, k).cuda().float(), i.view(T, k).cuda().to(torch.int32)) for w, i in R["route"]]
if os.environ.get("BALANCED"):
    rt = [(torch.full((T, k), 1.0 / k, device="cuda"), (torch.arange(T * k, device="cuda") % E).view(T, k).to(torch.int32)) for _ in rt]
if IMPL != "vllm":
    g = types.SimpleNamespace(E=E, k=k, d=d, ffn=ffn, act="silu")
    ln = ML.LeanMoE.__new__(ML.LeanMoE); ln.g = g; ln.impl = "triton"; ln.alias = False; ln.cfg = json.loads(os.environ.get("CFG", "{}")); ln.dev = "cuda"
def fn():
    y = x
    for l in range(L):
        if IMPL == "vllm":
            y = fused_experts(x, w1[l], w2[l], rt[l][0], rt[l][1])
        else:
            y = ln.experts(dict(Wgu=w1[l], Wd=w2[l]), x, rt[l][0].reshape(-1).contiguous(), rt[l][1].reshape(-1).long().contiguous(), T)
    return y
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): fn()
torch.cuda.current_stream().wait_stream(s)
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr): fn()
g_ = gr
ts = []
for i in range(23):
    torch.cuda.synchronize(); t0 = time.perf_counter(); gr.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
ts = sorted(ts[3:])
if IMPL != "vllm" and os.environ.get("SPLIT"):
    print(json.dumps(ML.kernel_split(gr)), flush=True)
ver = __import__("vllm").__version__ if IMPL == "vllm" else "g3-triton"
print(json.dumps(dict(route=sys.argv[1], impl=ver, cfg=os.environ.get("CFG", ""), layers=L, T=T, experts_total_ms=round(st.median(ts), 3), p95=round(ts[int(.95 * 19)], 3),
                      per_layer_ms=round(st.median(ts) / L, 4))), flush=True)
