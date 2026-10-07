"""d1: torch.compile paths on the whole forward (static shapes, our CUDA graph around it). usage: python compile_bench.py MODE TOKS [lean2fuse]"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.expanduser("~/work/d1"))
import torch._dynamo, torch._inductor.config as ic
torch._dynamo.config.cache_size_limit = 64
from prof_d1 import load_torso, capture, wall
from lean import Lean
from lean2 import Lean2
mode = sys.argv[1]; toks = [int(x) for x in sys.argv[2].split(",")]
fuse = sys.argv[3] if len(sys.argv) > 3 else None
torso = load_torso()
ref = Lean(torso)
ln = Lean2(torso, fuse=fuse) if fuse is not None else ref
if mode == "max-autotune-no-cudagraphs":
    ic.max_autotune_gemm_backends = "TRITON,ATEN"
fwd = torch.compile(ln.forward, mode=None if mode == "default" else mode, dynamic=False)
res = {}
torch.manual_seed(0)
idsv = torch.randint(1000, 100000, (1, 1000), device="cuda")
for T in toks:
    ids = torch.randint(1000, 100000, (1, T), device="cuda")
    t0 = time.time()
    with torch.inference_mode():
        fwd(ids); torch.cuda.synchronize()
    ct = time.time() - t0
    g, out = capture(lambda: fwd(ids))
    r = wall(g, ids, T, reps=20); r["compile_s"] = round(ct, 1)
    r["tflops"] = round(2.745e9 * T / (r["median"] / 1000) / 1e12, 1)
    if T == 1000:
        with torch.inference_mode():
            ids.copy_(idsv); g.replay(); h = out.float(); href = ref.forward(idsv).float()
        cos = torch.nn.functional.cosine_similarity(h[0], href[0], dim=-1)
        r["acc"] = dict(cos_mean=round(cos.mean().item(), 6), cos_min=round(cos.min().item(), 5), rel=round(((h - href).norm() / href.norm()).item(), 5))
    res[T] = r; print(mode, fuse, T, json.dumps(r), flush=True)
    del g, out; torch.cuda.empty_cache()
json.dump(res, open(os.path.expanduser(f"~/work/d1/compile_{mode}_{fuse}.json"), "w"), indent=1)
