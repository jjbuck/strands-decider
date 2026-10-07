import sys, time, json, os, statistics as st, torch
from common import *
from runner import *
from packed import *
from strands_decider.infer import load_engine
from strands_decider.modeling import PointerHead
from lean import Lean
model = sys.argv[1]; modes = sys.argv[2].split(","); toks = [int(x) for x in sys.argv[3].split(",")]
REPS = int(os.environ.get("REPS", "15"))
eng = load_engine(CKPT, device="cuda"); tok = eng.model.tokenizer; mk = make_state_fn(tok)
if model == "2b":
    torso = eng.model.torso.merge_and_unload().eval()
elif model == "0.6b":
    import transformers
    torso = transformers.AutoModel.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).cuda().eval()
    eng.model.head = PointerHead(torso.config.hidden_size, 256).float().cuda().eval()
else:
    import transformers
    name = {"0.8b": "Qwen/Qwen3.5-0.8B"}[model]
    cfg = transformers.AutoConfig.from_pretrained(name)
    lm = transformers.Qwen3_5ForCausalLM.from_pretrained(name, config=cfg.get_text_config(), dtype=torch.bfloat16).cuda().eval()
    torso = lm.model
    eng.model.head = PointerHead(torso.config.hidden_size, 256).float().cuda().eval()
eng.model.torso = torso
ln = Lean(torso) if model != '0.6b' else None
res = {}
def pre(n, nq):
    Q = qs(nq); return iter([mk(n, f"s{i}_{time.time()}") for i in range(REPS + 3)]), Q
def rep(label, n, nq, fn, fe):
    it, Q = pre(n, nq)
    r = timeit(lambda: fn(next(it), Q), reps=REPS)
    if fe.gpu_ms:
        r["gpu_replay_median"] = round(st.median(fe.gpu_ms[-REPS:]), 2); fe.gpu_ms.clear()
    res[f"{model}_{label}_q{nq}_t{n}"] = r; print(model, label, f"q={nq} tokens={n}", r, flush=True)
for mode in modes:
    if mode == "single":
        fe = FastEngine(eng, ln.forward, graph=True)
        for n in toks: rep("lean_graph", n, 1, fe.ask, fe)
    if mode == "hfsingle":
        f = (lambda ids: torso(input_ids=ids % 150000, use_cache=False).last_hidden_state)
        fe = FastEngine(eng, f, graph=True)
        for n in toks: rep("hf_graph", n, 1, fe.ask, fe)
    if mode == "hfrows4":
        f = (lambda ids: torso(input_ids=ids % 150000, use_cache=False).last_hidden_state)
        fe = FastEngine(eng, f, graph=True)
        for n in toks: rep("hf_graph_rows", n, 4, fe.ask, fe)
    if mode == "concat4":   # 4 questions as 4 full-prompt rows (what the stock single-pass path would do)
        fe = FastEngine(eng, ln.forward, graph=True)
        for n in toks: rep("lean_graph_rows", n, 4, fe.ask, fe)
    if mode == "packed4":
        pe = PackedEngine(eng, ln, graph=True)
        for n in toks: rep("packed", n, 4, pe.ask, pe)
    if mode == "packed2":
        pe = PackedEngine(eng, ln, graph=True)
        for n in toks: rep("packed", n, 2, pe.ask, pe)
    if mode == "host":   # host-side pieces at ~n tokens
        fe = FastEngine(eng, ln.forward, graph=True)
        for n in toks:
            it, Q = pre(n, 1); states = list(it)
            def prep(s): return fe._prep(s, Q)
            ts = []
            for s in states[3:3 + REPS]:
                t = time.perf_counter(); prep(s); ts.append((time.perf_counter() - t) * 1000)
            res[f"{model}_hostprep_t{n}"] = round(st.median(ts), 3); print(model, "host prep (render+tokenize+offsets) ms", n, res[f"{model}_hostprep_t{n}"], flush=True)
            # tokenizer only, fast no-offset
            ts = []
            for s in states[3:3 + REPS]:
                t = time.perf_counter(); tok(s, add_special_tokens=False); ts.append((time.perf_counter() - t) * 1000)
            print(model, "tokenize state only (no offsets) ms", n, round(st.median(ts), 3), flush=True)
    if mode == "throughput":
        for L in toks:
            for B in (1, 2, 4, 8, 16, 32):
                if B * L > 32768: continue
                ids = torch.randint(1000, 100000, (B, L), device="cuda")
                torch.cuda.synchronize()
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s), torch.inference_mode():
                    for _ in range(2): ln.forward(ids)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(g): out = ln.forward(ids)
                t = []
                for _ in range(10):
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                    e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); t.append(e0.elapsed_time(e1))
                m = st.median(t)
                res[f"{model}_thr_L{L}_B{B}"] = dict(ms=round(m, 2), req_s=round(B / m * 1000, 1), tok_s=round(B * L / m * 1000))
                print(model, f"throughput L={L} B={B}: {m:.2f} ms  {B/m*1000:.1f} req/s  {B*L/m*1000:.0f} tok/s", flush=True)
                del g, out; torch.cuda.empty_cache()
json.dump(res, open(f"bench3_{model}_{'_'.join(modes)}.json", "w"), indent=1)
