"""d1: uncontended profile of the lean 2B prefill (box a8, exclusive).
(a) eager forward with record_function stage labels -> per-stage kernel time (CUPTI), per-GEMM TFLOPS
(b) CUDA-graph replay -> busy, span, inter-kernel gaps, kernel count, grid sizes (wave quantization)
(c) wall-clock graph replay latency (fresh ids each rep)
(d) bare-GEMM skeleton: the same 24-layer GEMM sequence alone (isolated + graphed)
usage: python prof_d1.py 256,1000,4000 [variant]
"""
import sys, os, json, time, statistics as st, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/work/systems/g")); sys.path.insert(0, os.path.expanduser("~/work/d1"))
from torch.profiler import profile, ProfilerActivity, record_function as RF
from common import CKPT
import lean as LM
from lean import Lean
OUT = {}

def load_torso():
    from strands_decider.modeling import StrandsDeciderModel
    m = StrandsDeciderModel.load(CKPT)
    t = m.torso.merge_and_unload().eval()
    return t.cuda().to(torch.bfloat16)

def fwd_annot(ln, ids):
    B, T = ids.shape
    with RF("S:embed_rope"):
        x = F.embedding(ids, ln.embed).reshape(B * T, -1)
        pos = torch.arange(T, device=ln.dev, dtype=torch.float32)
        fr = pos[:, None] * ln.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype), fr.sin().to(x.dtype)
        h = LM._rms_zc(x, ln.layers[0]["in_norm"], ln.eps)
    n = len(ln.layers)
    for i, d in enumerate(ln.layers):
        lin = d["type"] == "linear_attention"
        with RF("G:in_gdn" if lin else "G:in_attn"):
            proj = h @ d["Win"].t()
        if lin:
            with RF("S:gdn_glue"):
                z, beta, g = LM._lin_glue(proj, d["A_log"], d["dt_bias"])
                qkv = proj[:, :6144].reshape(B, T, 6144)
            with RF("S:gdn_conv"):
                qkv = LM.fla_conv(qkv, d["conv_w"], None, activation="silu")
                qkv = qkv[0] if isinstance(qkv, tuple) else qkv
            with RF("S:gdn_chunk"):
                q, k, v = qkv.split(2048, dim=-1)
                q = q.reshape(B, T, 16, 128); k = k.reshape(B, T, 16, 128); v = v.reshape(B, T, 16, 128)
                o, _ = LM.chunk_gated_delta_rule(q, k, v, g.reshape(B, T, 16), beta.reshape(B, T, 16), use_qk_l2norm_in_kernel=True)
            with RF("S:gdn_gnorm"):
                o = LM._gated_norm(o.reshape(-1, 128), z.reshape(-1, 128), d["gn_w"], ln.eps).reshape(B * T, 2048)
        else:
            with RF("S:attn_prep"):
                q, k, v, gate = LM._attn_prep(proj, d["qn"], d["kn"], cos, sin, ln.eps)
                qh = q.reshape(B, T, 8, 256).transpose(1, 2); kh = k.reshape(B, T, 2, 256).transpose(1, 2); vh = v.reshape(B, T, 2, 256).transpose(1, 2)
            with RF("S:attn_sdpa"):
                o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
            with RF("S:attn_gate"):
                o = o.transpose(1, 2).reshape(B * T, 2048) * gate
        with RF("G:out"):
            d_out = o @ d["Wo"].t()
        with RF("S:add_rms"):
            x, h2 = LM._add_rms(x, d_out, d["post_norm"], ln.eps)
        with RF("G:gate_up"):
            gu = h2 @ d["Wgu"].t()
        with RF("S:silu_mul"):
            m = LM._silu_mul(gu, d["I"])
        with RF("G:down"):
            d_mlp = m @ d["Wd"].t()
        nw = ln.layers[i + 1]["in_norm"] if i + 1 < n else ln.norm_w
        with RF("S:add_rms"):
            x, h = LM._add_rms(x, d_mlp, nw, ln.eps)
    return h.reshape(B, T, -1)

GEMM = {"G:in_gdn": (8224, 2048, 18), "G:in_attn": (5120, 2048, 6), "G:out": (2048, 2048, 24), "G:gate_up": (12288, 2048, 24), "G:down": (2048, 6144, 24)}

def _trace_kernels(path):
    tr = json.load(open(path))
    ks = sorted([e for e in tr["traceEvents"] if e.get("cat") == "kernel"], key=lambda e: e["ts"])
    ann = sorted([e for e in tr["traceEvents"] if e.get("cat") == "gpu_user_annotation"], key=lambda e: e["ts"])
    return ks, ann

def stage_profile(fn, T, label):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    path = f"/tmp/d1_eager_{label}_{T}.json"; p.export_chrome_trace(path)
    ks, ann = _trace_kernels(path)
    import bisect
    starts = [a["ts"] for a in ann]
    labels = []
    for k in ks:
        i = bisect.bisect_right(starts, k["ts"] + 1e-3) - 1
        lab = "?"
        while i >= 0:
            a = ann[i]
            if a["ts"] <= k["ts"] + 1e-3 and k["ts"] + k["dur"] <= a["ts"] + a["dur"] + 1e-3: lab = a["name"]; break
            if a["ts"] + a["dur"] < k["ts"] - 5000: break
            i -= 1
        labels.append(lab)
    return ks, labels

def stage_table(ks, labels, T, nsm=80):
    stages = {}
    for k, lab in zip(ks, labels):
        a = k.get("args", {}); gr = a.get("grid", [1, 1, 1]); nb = gr[0] * gr[1] * gr[2]
        s = stages.setdefault(lab, dict(ms=0.0, n=0, n_lt_sm=0, ms_lt_sm=0.0))
        s["ms"] += k["dur"] / 1000; s["n"] += 1
        if nb < nsm: s["n_lt_sm"] += 1; s["ms_lt_sm"] += k["dur"] / 1000
    for s in stages.values():
        s["ms"] = round(s["ms"], 3); s["ms_lt_sm"] = round(s["ms_lt_sm"], 3)
    res = {"stages": dict(sorted(stages.items(), key=lambda kv: -kv[1]["ms"]))}
    gt = {}
    for kname, (N, K, c) in GEMM.items():
        if kname in stages:
            gt[kname] = dict(ms=stages[kname]["ms"], tflops=round(2 * T * N * K * c / (stages[kname]["ms"] / 1000) / 1e12, 1))
    res["gemm"] = gt
    res["gemm_ms"] = round(sum(v["ms"] for k, v in stages.items() if k.startswith("G:")), 2)
    res["nongemm_ms"] = round(sum(v["ms"] for k, v in stages.items() if not k.startswith("G:")), 2)
    kern = {}
    for k, lab in zip(ks, labels):
        key = (lab + " | " + k["name"])[:110]; kern[key] = kern.get(key, 0) + k["dur"] / 1000
    res["top_kernels"] = [(k, round(v, 3)) for k, v in sorted(kern.items(), key=lambda kv: -kv[1])[:30]]
    return res

def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): out = fn()
    return g, out

def graph_profile(g, tag):
    for _ in range(3): g.replay()
    torch.cuda.synchronize()
    spans, busys, ns, gaps = [], [], [], []
    for r in range(5):
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            g.replay(); torch.cuda.synchronize()
        ks = sorted([(e.time_range.start, e.time_range.end) for e in p.events() if e.device_type.name == "CUDA" and e.time_range.end > e.time_range.start])
        busy = sum(b - a for a, b in ks); span = ks[-1][1] - ks[0][0]
        gp = sum(max(0, ks[i + 1][0] - ks[i][1]) for i in range(len(ks) - 1))
        spans.append(span / 1000); busys.append(busy / 1000); ns.append(len(ks)); gaps.append(gp / 1000)
        if r == 4:
            p.export_chrome_trace(f"/tmp/d1_trace_{tag}.json")
    return dict(kernels=int(st.median(ns)), busy_ms=round(st.median(busys), 2), span_ms=round(st.median(spans), 2), gaps_ms=round(st.median(gaps), 3))

def wave_stats(tag):
    tr = json.load(open(f"/tmp/d1_trace_{tag}.json"))
    ev = [e for e in tr["traceEvents"] if e.get("cat") == "kernel"]
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    tot = 0; lost = 0; small = 0; small_t = 0
    for e in ev:
        a = e.get("args", {}); gr = a.get("grid", [1, 1, 1]); bps = a.get("blocks per SM", None)
        nb = gr[0] * gr[1] * gr[2]; dur = e["dur"]
        tot += dur
        occ = a.get("est. achieved occupancy %", None)
        # waves = nb / (nsm * resident blocks per SM); 'blocks per SM' in CUPTI = nb/nsm
        if nb < nsm: small += 1; small_t += dur
    return dict(n=len(ev), kernels_with_fewer_blocks_than_SMs=small, their_ms=round(small_t / 1000, 2))

def wall(g, ids_static, T, reps=20):
    pool = [torch.randint(1000, 100000, (1, T), dtype=torch.long).pin_memory() for _ in range(reps + 3)]
    ts = []
    for i, x in enumerate(pool):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids_static.copy_(x, non_blocking=True); g.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = sorted(ts[3:])
    return dict(median=round(st.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1))], 2), n=len(ts))

def bare_gemm(T):
    dev = "cuda"; r = {}
    Ws = {k: torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for k, (N, K, c) in GEMM.items()}
    tot = 0; fl = 0
    for k, (N, K, c) in GEMM.items():
        x = torch.randn(T, K, device=dev, dtype=torch.bfloat16); W = Ws[k]
        for _ in range(5): x @ W.t()
        torch.cuda.synchronize(); ts = []
        for _ in range(30):
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); x @ W.t(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        t = st.median(ts); tot += t * c; fl += 2 * T * N * K * c
        r[k] = dict(us=round(t * 1000, 1), tflops=round(2 * T * N * K / t / 1e9, 1))
    r["isolated_sum_ms"] = round(tot, 2); r["isolated_tflops"] = round(fl / tot / 1e9, 1)
    # graphed skeleton: real layer order, distinct weights per layer (no L2 reuse cheating)
    types = ["linear_attention"] * 24
    for i in (3, 7, 11, 15, 19, 23): types[i] = "full_attention"
    Wl = []
    for t in types:
        Wl.append(dict(Win=torch.randn(8224 if t == "linear_attention" else 5120, 2048, device=dev, dtype=torch.bfloat16) * .02,
                       Wo=torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16) * .02, Wgu=torch.randn(12288, 2048, device=dev, dtype=torch.bfloat16) * .02,
                       Wd=torch.randn(2048, 6144, device=dev, dtype=torch.bfloat16) * .02))
    h = torch.randn(T, 2048, device=dev, dtype=torch.bfloat16); o = torch.randn(T, 2048, device=dev, dtype=torch.bfloat16); m = torch.randn(T, 6144, device=dev, dtype=torch.bfloat16)
    def skel():
        y = None
        for d in Wl:
            a = h @ d["Win"].t(); b = o @ d["Wo"].t(); c = h @ d["Wgu"].t(); y = m @ d["Wd"].t()
        return y
    g, _ = capture(skel)
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(15):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
    t = st.median(ts)
    r["graphed_skeleton_ms"] = round(t, 2); r["graphed_skeleton_tflops"] = round(fl / t / 1e9, 1)
    del Wl, g; torch.cuda.empty_cache()
    return r

if __name__ == "__main__":
    toks = [int(x) for x in sys.argv[1].split(",")]
    variant = sys.argv[2] if len(sys.argv) > 2 else "base"
    torso = load_torso()
    if variant == "base":
        ln = Lean(torso)
    elif variant == "compile":
        import torch._dynamo; torch._dynamo.config.cache_size_limit = 64
        ln = Lean(torso, compile_glue=True)
    print("mem GB", torch.cuda.memory_allocated() / 1e9, flush=True)
    for T in toks:
        ids = torch.randint(1000, 100000, (1, T), device="cuda")
        r = {}
        eks, elab = stage_profile(lambda: fwd_annot(ln, ids), T, variant)
        r["stage_eager"] = stage_table(eks, elab, T)
        g, out = capture(lambda: ln.forward(ids))
        r["graph"] = graph_profile(g, f"{variant}_{T}")
        gks, _ = _trace_kernels(f"/tmp/d1_trace_{variant}_{T}.json")
        names_e = [k["name"] for k in eks]; names_g = [k["name"] for k in gks]
        r["graph"]["eager_kernels"] = len(eks); r["graph"]["seq_match"] = names_e == names_g
        if names_e == names_g:
            r["stage_graph"] = stage_table(gks, elab, T)
        else:
            r["stage_graph"] = "kernel sequences differ"
        r["waves"] = wave_stats(f"{variant}_{T}")
        r["wall"] = wall(g, ids, T)
        del g, out; torch.cuda.empty_cache()
        r["bare_gemm"] = bare_gemm(T)
        gem = r["bare_gemm"]
        fl = 2.745e9 * T
        r["model_tflops_wall"] = round(fl / (r["wall"]["median"] / 1000) / 1e12, 1)
        r["frac_of_bare_skeleton"] = round(gem["graphed_skeleton_ms"] / r["wall"]["median"], 3)
        OUT[f"{variant}_T{T}"] = r
        rr = {k: v for k, v in r.items() if k not in ("stage_eager",)}
        if isinstance(rr.get("stage_graph"), dict): rr["stage_graph"] = {k: v for k, v in rr["stage_graph"].items() if k != "top_kernels"}
        else: rr["stage_eager"] = {k: v for k, v in r["stage_eager"].items() if k != "top_kernels"}
        print(json.dumps({f"{variant}_T{T}": rr}), flush=True)
    json.dump(OUT, open(os.path.expanduser(f"~/work/d1/prof_{variant}_{'_'.join(map(str, toks))}.json"), "w"), indent=1)
