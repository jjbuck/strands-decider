"""G3 latency bench (run under box.sh --timing). Batch 1, exact T, CUDA graph, fresh REAL token sequences per rep (MoE latency depends
on routing), torch.cuda.synchronize, 20 warm reps, median/p95.
usage: python bench_lat.py MODEL [T ...] [--check] [--impls triton,gmm] [--extra]   -> results/lat_MODEL.json (appends)
--extra: alias (all experts share expert 0's weights: compute without weight DRAM traffic), balanced synthetic routing, weight-stream floor.
"""
import os, sys, json, argparse, random, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
import moe as M
import moe_lean as ML
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("T", nargs="*", type=int, default=[1000, 4000])
ap.add_argument("--check", action="store_true"); ap.add_argument("--impls", default="triton"); ap.add_argument("--extra", action="store_true")
ap.add_argument("--layers", type=int, default=0); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--ckpt", default="")
a = ap.parse_args()

g = M.load_cfg(a.model)
W = M.load_weights(g, layers=a.layers or None)
if a.ckpt:
    t = M.MoETorso(g, W).cuda(); t.lora.load_state_dict(torch.load(os.path.join(a.ckpt, "lora.pt"))); W = t.merged(inplace=True); del t
tok = AutoTokenizer.from_pretrained(M.MODELS.get(a.model, a.model))
# real token pool: rendered train-pool states (train split only), concatenated, cut into fresh windows
from strands_decider.prompting import render_state
pool_txt = []
with open(os.path.expanduser("~/work/evalkit/train_pool.jsonl")) as f:
    for i, l in enumerate(f):
        if i % 50: continue
        pool_txt.append(render_state(json.loads(l)["state"]))
        if len(pool_txt) >= 120: break
ids_all = []
for s in pool_txt: ids_all += tok(s, add_special_tokens=False)["input_ids"]
print("pool tokens", len(ids_all), flush=True)
tot, act = M.body_params(g)
res = dict(model=a.model, repo=M.MODELS.get(a.model), body_total=tot, body_active=act, layers=len(W["layers"]), gpu=torch.cuda.get_device_name(0),
           grouped_mm=M.has_grouped_mm())
print(json.dumps(res), flush=True)
os.makedirs('results', exist_ok=True)


def windows(T, n):
    rng = random.Random(T)
    out = []
    for _ in range(n):
        s = rng.randrange(0, len(ids_all) - T)
        out.append(torch.tensor(ids_all[s:s + T], dtype=torch.long).view(1, T).pin_memory())
    return out


for T in a.T:
    pool = windows(T, a.reps + 3)
    ids = pool[0].cuda()
    bn1 = 64 if g.ffn % 128 else 128
    CANDS = [dict(BM=64, BN1=bn1, BK1=64, BN2=64, BK2=64, w1=4, s1=3, w2=4, s2=3, nf=1), dict(BM=128, BN1=bn1, BK1=64, BN2=64, BK2=64, w1=4, s1=3, w2=4, s2=3, nf=1),
             dict(BM=32, BN1=64, BK1=64, BN2=128, BK2=64, w1=4, s1=3, w2=4, s2=3, nf=1)]
    best = None
    for ci, impl in enumerate(["triton"] * len(CANDS) + [x for x in a.impls.split(",") if x != "triton"]):
        if impl == "gmm": continue   # torch._grouped_mm falls back to host-synchronising code on sm86: not CUDA-graph capturable
        ln = ML.LeanMoE(g, W, impl=impl, cfg=CANDS[ci] if impl == "triton" else None)
        if a.check and impl == "triton" and ci == 0:
            ref = M.MoETorso(g, W, lora=False).eval()
            with torch.no_grad():
                r = ref(ids).last_hidden_state[0].float(); o = ln.forward(ids).float()
            cos = torch.nn.functional.cosine_similarity(r, o, dim=-1)
            rel = ((r - o).norm() / r.norm()).item()
            print(json.dumps(dict(T=T, check_cos_mean=round(cos.mean().item(), 6), check_cos_min=round(cos.min().item(), 6), rel=round(rel, 5))), flush=True)
            del ref
            ln.record = []
            with torch.no_grad(): ln.forward(ids)
            torch.save(dict(route=ln.record, T=T, E=g.E, k=g.k, d=g.d, ffn=g.ffn), f"results/route_{a.model}_{T}.pt"); ln.record = None
            cnt = torch.stack([torch.bincount(i, minlength=g.E) for _, i in torch.load(f"results/route_{a.model}_{T}.pt")["route"]]).float()
            print(json.dumps(dict(T=T, load_max_over_mean=round((cnt.max(1).values / cnt.mean(1)).mean().item(), 3),
                                  load_cv=round((cnt.std(1) / cnt.mean(1)).mean().item(), 3), experts_hit=round((cnt > 0).float().sum(1).mean().item(), 2))), flush=True)
        gr, out = ML.capture(lambda: ln.forward(ids))
        w = ML.wall(gr, ids, pool, a.reps)
        ks = ML.kernel_split(gr)
        e = dict(T=T, impl=impl, cfg=CANDS[ci] if impl == "triton" else None, **{f"wall_{k}": v for k, v in w.items()}, kernels=ks)
        print(json.dumps(e), flush=True); res[f"{impl}{ci}_{T}"] = e
        if impl == "triton" and (best is None or w["median"] < best[0]): best = (w["median"], CANDS[ci])
        del gr, out
        torch.cuda.empty_cache()
    res[f"best_{T}"] = dict(ms=best[0], cfg=best[1]); print(json.dumps(dict(T=T, best=res[f"best_{T}"])), flush=True)
    if a.extra:
        ln = ML.LeanMoE(g, W, impl="triton", alias=True, cfg=best[1])
        gr, out = ML.capture(lambda: ln.forward(ids)); w = ML.wall(gr, ids, pool, a.reps)
        e = dict(T=T, impl="alias_compute_only", **{f"wall_{k}": v for k, v in w.items()}, kernels=ML.kernel_split(gr)); print(json.dumps(e), flush=True); res[f"alias_{T}"] = e
        del gr, out
        S = T * g.k
        bal = (torch.arange(S, device="cuda") % g.E).to(torch.long)
        ln = ML.LeanMoE(g, W, impl="triton", synth_route=bal, cfg=best[1])
        gr, out = ML.capture(lambda: ln.forward(ids)); w = ML.wall(gr, ids, pool, a.reps)
        e = dict(T=T, impl="balanced_routing", **{f"wall_{k}": v for k, v in w.items()}, kernels=ML.kernel_split(gr)); print(json.dumps(e), flush=True); res[f"balanced_{T}"] = e
        del gr, out
        torch.cuda.empty_cache()
if a.extra:
    ln = ML.LeanMoE(g, W)
    fn = ln.stream_graph(); gr, out = ML.capture(fn)
    w = ML.wall(gr, torch.zeros(1, 1, dtype=torch.long, device="cuda"), [torch.zeros(1, 1, dtype=torch.long).pin_memory()], a.reps)
    nb = ln.weight_bytes()
    e = dict(stream_bytes=nb, stream_ms=w["median"], stream_GBps=round(nb / w["median"] / 1e6, 1)); print(json.dumps(e), flush=True); res["stream"] = e
json.dump(res, open(f"results/lat_{a.model}{'_' + os.path.basename(a.ckpt.rstrip('/')) if a.ckpt else ''}.json", "w"), indent=1)
