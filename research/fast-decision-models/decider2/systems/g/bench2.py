import sys, time, json, torch
from common import *
from runner import *
from strands_decider.infer import load_engine
from lean import Lean
variants = sys.argv[1].split(","); toks = [int(x) for x in sys.argv[2].split(",")]; nqs = [int(x) for x in sys.argv[3].split(",")]
eng = load_engine(CKPT, device="cuda"); tok = eng.model.tokenizer; mk = make_state_fn(tok)
torso = eng.model.torso.merge_and_unload().eval()
eng.model.torso = torso
res = {}
REPS = int(__import__("os").environ.get("REPS", "15"))
def run(label, fn):
    for nq in nqs:
        for n in toks:
            states = [mk(n, f"s{i}_{time.time()}") for i in range(REPS + 2)]
            it = iter(states); Q = qs(nq)
            r = timeit(lambda: fn(next(it), Q), reps=REPS)
            g = getattr(run, "fe", None)
            if g is not None and g.gpu_ms:
                import statistics as _s
                r["gpu_replay_median"] = round(_s.median(g.gpu_ms[-REPS:]), 2); g.gpu_ms.clear()
            res[f"{label}_q{nq}_t{n}"] = r; print(label, f"q={nq} tokens={n}", r, flush=True)
for v in variants:
    if v == "merged": run(v, eng.ask)
    elif v == "hfgraph":
        fe = FastEngine(eng, lambda ids: torso(input_ids=ids, use_cache=False).last_hidden_state, graph=True); run.fe = fe; run(v, fe.ask)
    elif v == "hfeager":
        fe = FastEngine(eng, lambda ids: torso(input_ids=ids, use_cache=False).last_hidden_state, graph=False); run.fe = fe; run(v, fe.ask)
    elif v == "lean_eager":
        ln = Lean(torso); fe = FastEngine(eng, ln.forward, graph=False); run.fe = fe; run(v, fe.ask)
    elif v == "lean_graph":
        ln = Lean(torso); fe = FastEngine(eng, ln.forward, graph=True); run.fe = fe; run(v, fe.ask)
    elif v == "lean_compile_graph":
        import torch._dynamo; torch._dynamo.config.cache_size_limit = 64
        ln = Lean(torso, compile_glue=True); fe = FastEngine(eng, ln.forward, graph=True); run.fe = fe; run(v, fe.ask)
json.dump(res, open(f"bench2_{'_'.join(variants)}.json", "w"), indent=1)
