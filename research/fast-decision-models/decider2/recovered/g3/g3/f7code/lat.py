"""F7 latency: trained decider (LoRA merged) through the d1 fused runtime (lean2, fuse=gnorm,prep,conv,fold), CUDA graph,
batch 1, exact T, fresh random ids per rep (pinned H2D copy inside the timed region), torch.cuda.synchronize, n=20 warm reps.
usage: python lat.py q08 CKPT | trunc12 CKPT | hobson | base08"""
import sys, os, glob, json, torch, transformers
sys.path.insert(0, os.path.expanduser("~/work/d1")); sys.path.insert(0, os.path.expanduser("~/work/systems/g"))
sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
from prof_d1 import capture, wall
from lean2 import Lean2
from strands_decider.modeling import StrandsDeciderModel, build_head, StrandsDeciderConfig
from peft import PeftModel
arch = sys.argv[1]; ck = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
LAYERS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as MD
if ck:
    torso, head, c, meta = MD.load_ckpt(ck); arch = f"{meta['arch']}{meta.get('layers')}"
else:  # untrained shapes (latency only): q08 / hob with LAYERS (0 = all)
    torso, hob = MD.base_torso(arch, LAYERS); head = hob.head if hob is not None else None; arch = f"{arch}{len(torso.layers)}_untrained"
torso = torso.cuda().eval()
P = sum(p.numel() for n, p in torso.named_parameters() if "embed" not in n)
ln = Lean2(torso, fuse="gnorm,prep,conv,fold")
res = dict(arch=arch, ckpt=ck, body_params=P, layers=len(torso.layers), hidden=torso.config.hidden_size, gpu=torch.cuda.get_device_name(0))
if head is not None: head = head.cuda().eval()
K = 2
for T in (256, 1000, 4000):
    ids = torch.randint(1000, 100000, (1, T), device="cuda")
    g, o = capture(lambda: ln.forward(ids)); res[f"torso_{T}"] = wall(g, ids, T)
    if T == 1000:  # kernel split for the projection: Triton GEMMs vs everything else (one replay under the profiler, x5)
        from torch.profiler import profile, ProfilerActivity
        tot = {"gemm": 0.0, "other": 0.0}
        for _ in range(5):
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                g.replay(); torch.cuda.synchronize()
            for e in p.events():
                if e.device_type.name == "CUDA" and e.time_range.end > e.time_range.start:
                    tot["gemm" if "gemm" in e.name.lower() else "other"] += (e.time_range.end - e.time_range.start) / 1000 / 5
        res["kernel_split_1000_ms"] = {k: round(v, 2) for k, v in tot.items()}
    del g, o
    if head is not None:
        opt = torch.tensor([T - 10, T - 5], device="cuda")
        def full():
            h = ln.forward(ids)[0].float()
            return head(h[-1:], h[opt][None])
        g, o = capture(full); res[f"torso+head_{T}"] = wall(g, ids, T); del g, o
    torch.cuda.empty_cache()
print(json.dumps(res), flush=True)
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/lat_{arch}.json", "w"), indent=1)
