import sys, os, json, torch, time, statistics as st
sys.path.insert(0, os.path.expanduser("~/work/systems/g")); sys.path.insert(0, os.path.expanduser("~/work/d1"))
from common import CKPT, qs
from strands_decider.infer import load_engine
from runner import FastEngine
from prof_d1 import capture
from lean2 import Lean2
from torch.profiler import profile, ProfilerActivity
eng = load_engine(CKPT, device="cuda"); tok = eng.tok
torso = eng.model.torso.merge_and_unload().eval(); eng.model.torso = torso
ln = Lean2(torso, fuse="gnorm,prep,conv,fold")
fe = FastEngine(eng, ln.forward, graph=True)
reqs = json.load(open(os.path.expanduser("~/work/tokens/real_reqs.json")))
s = tok.decode(tok._tokenizer.encode(reqs[0]["state"], add_special_tokens=False).ids[:960])
Q = qs(1)
for _ in range(5): fe.ask(s, Q)
print("fe gpu_ms", [round(x, 2) for x in fe.gpu_ms[-3:]], "keys", list(fe.graphs.keys()))
st_ = fe.graphs[list(fe.graphs.keys())[0]]
with profile(activities=[ProfilerActivity.CUDA]) as p:
    st_["g"].replay(); torch.cuda.synchronize()
agg = {}
for e in p.events():
    if e.device_type.name == "CUDA": agg[e.name[:70]] = agg.get(e.name[:70], 0) + (e.time_range.end - e.time_range.start) / 1000
for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:12]: print(round(v, 2), k)
ids = st_["ids"]
g, out = capture(lambda: ln.forward(ids))
for _ in range(3): g.replay()
torch.cuda.synchronize(); a = time.perf_counter(); g.replay(); torch.cuda.synchronize(); print("forward-only graph same ids", (time.perf_counter() - a) * 1e3, ids.shape)
