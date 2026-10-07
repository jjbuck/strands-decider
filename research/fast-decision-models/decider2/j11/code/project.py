"""per-point latency projection [S]: t = t_gemm/ratio_compute + (t_meas - t_gemm)/ratio_bw, t_gemm = slope(ms/GFLOP from the A10G fit) x GFLOPs.
compute ratios as FAST_DECISION_MODEL sec. 9 (fp16 accumulation on GeForce): 3090 1.92x, 4090 3.85x, 5090 5.27x; bandwidth: 936/1008/1792 vs 600 GB/s."""
import json, glob, os, numpy as np
J = os.path.expanduser("~/decider2/j11"); RC = {"3090": 1.92, "4090": 3.85, "5090": 5.27}; RB = {"3090": 1.56, "4090": 1.68, "5090": 2.99}
res = {}
for f in glob.glob(f"{J}/lat_[SM].json"):
    pts = {}
    for r in json.load(open(f)):
        if r["ms"] is not None: pts.setdefault((r["arm"], r["d"], r["L"], r["Q"]), []).append(r)
    for k, v in pts.items():
        v.sort(key=lambda r: r["T"]); g = np.array([r["gflops"] for r in v]); t = np.array([r["ms"] for r in v])
        slope = float(np.polyfit(g[-4:], t[-4:], 1)[0])  # compute-bound slope from the long lengths
        for r in v:
            tg = min(r["ms"], slope * r["gflops"]); rest = r["ms"] - tg
            res.setdefault(str(k), {})[r["T"]] = dict(A10G=r["ms"], p95=r["p95"], gflops=r["gflops"], **{gpu: tg / RC[gpu] + rest / RB[gpu] for gpu in RC})
json.dump(res, open(f"{J}/lat_proj.json", "w"), indent=1)
for k, v in sorted(res.items()):
    print(k, " ".join(f"T{T}: {d['A10G']:.2f}/{d['3090']:.2f}/{d['4090']:.2f}/{d['5090']:.2f}" for T, d in sorted(v.items(), key=lambda x: int(x[0])) if int(T) in (64, 256, 1000, 4000, 8000)))
