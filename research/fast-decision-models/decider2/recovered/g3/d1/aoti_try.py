import sys, os, time, torch, traceback
sys.path.insert(0, os.path.expanduser("~/work/d1"))
from prof_d1 import load_torso, capture, wall
from lean import Lean
from lean2 import Lean2
torso = load_torso()
T = 1000
for name, ln in (("lean_base", Lean(torso)), ("lean2_fold", Lean2(torso, fuse="gnorm,prep,conv,fold"))):
    class M(torch.nn.Module):
        def forward(self, ids): return ln.forward(ids)
    ids = torch.randint(1000, 100000, (1, T), device="cuda")
    t0 = time.time()
    try:
        with torch.inference_mode():
            ep = torch.export.export(M(), (ids,), strict=False)
        print(name, "export ok", round(time.time() - t0, 1), "s", flush=True)
        t0 = time.time()
        path = torch._inductor.aoti_compile_and_package(ep, package_path=os.path.expanduser(f"~/work/d1/{name}.pt2"))
        print(name, "aoti compile ok", round(time.time() - t0, 1), "s", flush=True)
        f = torch._inductor.aoti_load_package(path)
        with torch.inference_mode():
            out = f(ids)
            g, o = capture(lambda: f(ids))
            print(name, "aoti+graph wall", wall(g, ids, T), flush=True)
    except Exception as e:
        print(name, "FAILED:", type(e).__name__, str(e).split("\n")[0][:400], flush=True)
