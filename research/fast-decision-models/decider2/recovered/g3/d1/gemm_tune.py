import sys, os, torch, statistics as st, json
sys.path.insert(0, os.path.expanduser("~/work/d1"))
from lean2 import tgemm
def bench(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    return st.median(ts)
cfgs = [(128,128,32,4,4),(128,128,64,4,3),(128,256,32,8,3),(256,128,32,8,3),(128,128,32,8,4),(64,128,64,4,4),(128,64,64,4,4),(64,256,32,4,4)]
shapes = {"in_gdn": (8224, 2048), "out": (2048, 2048), "gate_up": (12288, 2048), "down": (2048, 6144)}
res = {}
for M in [int(x) for x in sys.argv[1].split(",")]:
    for nm, (N, K) in shapes.items():
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * .02
        tc = bench(lambda: a @ b.t())
        best = None
        for c in cfgs:
            try:
                t = bench(lambda: tgemm(a, b, cfg=c))
            except Exception as e:
                continue
            if best is None or t < best[0]: best = (t, c)
        fl = 2 * M * N * K
        line = f"M={M} {nm}: cublas {tc*1000:.0f}us {fl/tc/1e9:.1f}TF | triton best {best[0]*1000:.0f}us {fl/best[0]/1e9:.1f}TF {best[1]}"
        if nm == "gate_up":
            tsw = bench(lambda: tgemm(a, b, epi=1, cfg=best[1]))
            ss = torch.rand(M, device="cuda") * 2048
            tf = bench(lambda: tgemm(a, b, epi=1, ss=ss, cfg=best[1]))
            line += f" | swiglu-epi {tsw*1000:.0f}us | swiglu+rowscale {tf*1000:.0f}us"
        if nm in ("out", "down"):
            r = torch.randn(M, N, device="cuda", dtype=torch.bfloat16); so = torch.zeros(M, device="cuda")
            tr = bench(lambda: tgemm(a, b, epi=3, res=r, ssout=so, cfg=best[1]))
            line += f" | resid+sumsq-epi {tr*1000:.0f}us"
        res[f"{nm}_{M}"] = dict(cublas_us=tc*1000, triton_us=best[0]*1000, cfg=best[1])
        print(line, flush=True)
# correctness of epilogues
M, N, K = 1000, 12288, 2048
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * .02
I = N // 2; Wil = torch.stack([W[:I], W[I:]], 1).reshape(N, K).contiguous()
ref = torch.nn.functional.silu(a @ W[:I].t()) * (a @ W[I:].t())
out = tgemm(a, Wil, epi=1)
print("swiglu rel err", ((out.float() - ref.float()).norm() / ref.float().norm()).item())
b = torch.randn(2048, K, device="cuda", dtype=torch.bfloat16) * .02
print("plain rel err", ((tgemm(a, b).float() - (a @ b.t()).float()).norm() / (a @ b.t()).float().norm()).item())
