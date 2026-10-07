import sys, os, json, torch
sys.path.insert(0, os.path.expanduser("~/work/d1"))
from prof_d1 import load_torso, capture, graph_profile, _trace_kernels
from lean2 import Lean2
toks = [int(x) for x in sys.argv[1].split(",")]; fuse = sys.argv[2]
torso = load_torso(); ln = Lean2(torso, fuse=fuse)
def cat(n):
    l = n.lower()
    if "_gemm_k" in l or "gemm" in l or "cutlass" in l or "s16816" in l or "s1688" in l: return "gemm"
    if "flash" in l: return "sdpa"
    if "_conv_l2" in l: return "conv_l2"
    if "_add_rms" in l or "_gnorm" in l: return "norms"
    if "_attn_prep" in l: return "attn_prep"
    if "_silu_mul" in l: return "silu_mul"
    if any(x in l for x in ("chunk", "recompute", "kkt", "gate", "beta", "solve", "cumsum", "l2norm")): return "fla_gdn"
    return "other"
out = {}
for T in toks:
    ids = torch.randint(1000, 100000, (1, T), device="cuda")
    g, o = capture(lambda: ln.forward(ids))
    r = graph_profile(g, f"fused_{T}")
    ks, _ = _trace_kernels(f"/tmp/d1_trace_fused_{T}.json")
    c = {}; lt = {}
    for k in ks:
        a = k.get("args", {}); gr = a.get("grid", [1, 1, 1]); nb = gr[0] * gr[1] * gr[2]
        cc = cat(k["name"]); c[cc] = c.get(cc, 0) + k["dur"] / 1000
        if nb < 80: lt[cc] = lt.get(cc, 0) + k["dur"] / 1000
    r["cat_ms"] = {k: round(v, 2) for k, v in sorted(c.items(), key=lambda kv: -kv[1])}
    r["cat_ms_fewer_ctas_than_SMs"] = {k: round(v, 2) for k, v in lt.items()}
    fl = 2.745e9 * T; r["gemm_tflops"] = round(fl / (c["gemm"] / 1000) / 1e12, 1)
    other = {}
    for k in ks:
        if cat(k["name"]) in ("other", "fla_gdn"): other[k["name"][:60]] = other.get(k["name"][:60], 0) + k["dur"] / 1000
    r["other_top"] = sorted([(n, round(v, 3)) for n, v in other.items()], key=lambda x: -x[1])[:10]
    out[T] = r; print(T, json.dumps(r), flush=True)
    del g, o; torch.cuda.empty_cache()
json.dump(out, open(os.path.expanduser(f"~/work/d1/prof_fused_{fuse.replace(',', '_')}.json"), "w"), indent=1)
