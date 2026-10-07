import sys, time, torch, json
from common import *
from strands_decider.infer import load_engine
eng = load_engine(CKPT, device="cuda")
tok = eng.model.tokenizer
mk = make_state_fn(tok)
torso = eng.model.torso
print(type(torso))
nparam = sum(p.numel() for p in torso.parameters()); print("params (incl lora)", nparam/1e6)
def phase(label, n, nq):
    r = timeit(lambda: eng.ask(mk(n, time.time()), qs(nq)), reps=12)
    print(label, f"tokens={n} q={nq}", r, flush=True)
# 1. stock
phase("stock", 64, 1); phase("stock", 1000, 1)
# 2. profile stock at 64
from torch.profiler import profile, ProfilerActivity
s = mk(64, "p0"); q = qs(1); eng.ask(s, q)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    torch.cuda.synchronize(); t=time.perf_counter(); eng.ask(mk(64,"p1"), q); torch.cuda.synchronize(); wall=(time.perf_counter()-t)*1000
ev = prof.key_averages()
tot_cuda = sum(e.device_time_total for e in ev if e.device_type.name=="CUDA" ) if False else None
kern = [e for e in ev if e.device_type.name == "CUDA"] if hasattr(ev[0], "device_type") else []
print("wall ms", wall)
n_k = sum(e.count for e in kern); gpu_us = sum(e.self_device_time_total for e in kern)
print("cuda kernels launched:", n_k, "sum kernel time ms:", gpu_us/1000)
for e in sorted(kern, key=lambda e: -e.self_device_time_total)[:12]:
    print(f"  {e.key[:70]:70s} n={e.count:5d} total={e.self_device_time_total/1000:.2f}ms")
cpu_ops = [e for e in ev if e.device_type.name == "CPU"]
print("top CPU ops by self time")
for e in sorted(cpu_ops, key=lambda e: -e.self_cpu_time_total)[:12]:
    print(f"  {e.key[:60]:60s} n={e.count:5d} self_cpu={e.self_cpu_time_total/1000:.2f}ms")
# 3. merge LoRA
eng.model.torso = torso.merge_and_unload()
phase("merged", 64, 1); phase("merged", 1000, 1)
