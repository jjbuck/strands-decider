"""d1: correctness + wall timing of lean2 fusion steps. usage: python bench_d1.py 256,1000,4000 step1;step2;...  (each step = FUSE list)"""
import sys, os, json, time, statistics as st, torch
sys.path.insert(0, os.path.expanduser("~/work/d1"))
from prof_d1 import load_torso, capture, wall
from lean import Lean
from lean2 import Lean2
toks = [int(x) for x in sys.argv[1].split(",")]
steps = sys.argv[2].split(";")
torso = load_torso()
ln = Lean2(torso, fuse="")
ref = Lean(torso)
res = {}
torch.manual_seed(0)
idsv = torch.randint(1000, 100000, (1, 1000), device="cuda")
with torch.inference_mode():
    href = ref.forward(idsv).float()
for s in steps:
    ln.fuse = set(s.split(",")) - {""}
    with torch.inference_mode():
        h = ln.forward(idsv).float()
    cos = torch.nn.functional.cosine_similarity(h[0], href[0], dim=-1)
    rel = ((h - href).norm() / href.norm()).item()
    acc = dict(cos_mean=round(cos.mean().item(), 6), cos_min=round(cos.min().item(), 5), rel_err=round(rel, 5))
    r = {"acc": acc}
    for T in toks:
        ids = torch.randint(1000, 100000, (1, T), device="cuda")
        g, out = capture(lambda: ln.forward(ids))
        r[T] = wall(g, ids, T, reps=20)
        r[T]["tflops"] = round(2.745e9 * T / (r[T]["median"] / 1000) / 1e12, 1)
        del g, out; torch.cuda.empty_cache()
    res[s or "base"] = r
    print(s or "base", json.dumps(r), flush=True)
json.dump(res, open(os.path.expanduser(f"~/work/d1/bench_{int(time.time())}.json"), "w"), indent=1)
