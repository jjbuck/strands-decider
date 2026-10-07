"""d1 host side: tokenization / prep / H2D at ~1000 and ~4000 tokens on REAL banking states, and the e2e request path.
usage: python host_d1.py [e2e]"""
import sys, os, json, time, statistics as st, re
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import torch
sys.path.insert(0, os.path.expanduser("~/work/systems/g")); sys.path.insert(0, os.path.expanduser("~/work/d1"))
from common import CKPT, qs
from strands_decider.prompting import render_state
reqs = json.load(open(os.path.expanduser("~/work/tokens/real_reqs.json")))
E2E = len(sys.argv) > 1 and sys.argv[1] == "e2e"

def med(ts): ts = sorted(ts); return dict(median=round(st.median(ts), 3), p95=round(ts[int(0.95 * (len(ts) - 1))], 3))

def par_chunks(text, k):
    # split at line starts (after '\n', next char not whitespace) into ~k equal pieces: BPE-safe for the Qwen pre-tokenizer
    if k <= 1: return [text]
    n = len(text); cuts = [0]
    for j in range(1, k):
        p = text.find("\n", j * n // k)
        while p != -1 and p + 1 < n and text[p + 1].isspace(): p = text.find("\n", p + 1)
        if p == -1 or p + 1 >= n: break
        if p + 1 > cuts[-1]: cuts.append(p + 1)
    cuts.append(n)
    return [text[a:b] for a, b in zip(cuts[:-1], cuts[1:]) if b > a]

def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKPT)
    rt = tok._tokenizer if hasattr(tok, "_tokenizer") else tok.backend_tokenizer
    spec = tok("x", add_special_tokens=True)["input_ids"] != tok("x", add_special_tokens=False)["input_ids"]
    print("special tokens added for state:", spec, flush=True)
    out = {}
    for target in (1000, 4000):
        texts = []
        for r in reqs:
            s = render_state(r["state"]) if not isinstance(r["state"], str) else r["state"]
            ids = rt.encode(s, add_special_tokens=False).ids
            if len(ids) >= target:
                texts.append(tok.decode(ids[:target]))  # real text truncated to ~target tokens
            if len(texts) >= 24: break
        res = {}
        # 1. HF __call__ single (what _fit does for the state)
        ts = []
        for t in texts: a = time.perf_counter(); tok(t, add_special_tokens=True, truncation=True, max_length=8000)["input_ids"]; ts.append((time.perf_counter() - a) * 1e3)
        res["hf_call"] = med(ts)
        ts = []
        for t in texts: a = time.perf_counter(); tok(t, add_special_tokens=False, return_offsets_mapping=True)["input_ids"]; ts.append((time.perf_counter() - a) * 1e3)
        res["hf_call_offsets"] = med(ts)
        # 2. raw Rust encode (no BatchEncoding wrapping)
        ts = []
        for t in texts: a = time.perf_counter(); rt.encode(t, add_special_tokens=False).ids; ts.append((time.perf_counter() - a) * 1e3)
        res["rust_encode"] = med(ts)
        # 3. parallel chunked encode_batch (Rayon threads), exactness check
        for k in (2, 4, 8, 16):
            ts = []; bad = 0
            for t in texts:
                ref = rt.encode(t, add_special_tokens=False).ids
                a = time.perf_counter()
                ch = par_chunks(t, k)
                ids = [i for e in rt.encode_batch(ch, add_special_tokens=False) for i in e.ids]
                ts.append((time.perf_counter() - a) * 1e3)
                bad += ids != ref
            res[f"par{k}"] = dict(**med(ts), mismatches=bad, n=len(texts))
        # 4. list -> tensor -> GPU
        ids = rt.encode(texts[0], add_special_tokens=False).ids
        dst = torch.zeros(len(ids), dtype=torch.long, device="cuda"); dst32 = torch.zeros(len(ids), dtype=torch.int32, device="cuda")
        pin = torch.empty(len(ids), dtype=torch.long).pin_memory(); pin32 = torch.empty(len(ids), dtype=torch.int32).pin_memory()
        def h2d_pageable(): dst.copy_(torch.tensor(ids, dtype=torch.long)); torch.cuda.synchronize()
        def h2d_pinned(): pin.copy_(torch.tensor(ids, dtype=torch.long)); dst.copy_(pin, non_blocking=True); torch.cuda.synchronize()
        import numpy as np
        def h2d_np32(): pin32.numpy()[:] = np.fromiter(ids, dtype=np.int32, count=len(ids)); dst32.copy_(pin32, non_blocking=True); torch.cuda.synchronize()
        for nm, f in (("h2d_pageable_int64", h2d_pageable), ("h2d_pinned_int64", h2d_pinned), ("h2d_pinned_np_int32", h2d_np32)):
            for _ in range(5): f()
            ts = []
            for _ in range(30): a = time.perf_counter(); f(); ts.append((time.perf_counter() - a) * 1e3)
            res[nm] = med(ts)
        out[target] = res
        print(target, json.dumps(res), flush=True)
    json.dump(out, open(os.path.expanduser("~/work/d1/host_tok.json"), "w"), indent=1)

def e2e():
    """Full request path: FastEngine.ask (render+tokenize+offsets+H2D+graph replay incl. pointer head+D2H+answer)."""
    from strands_decider.infer import load_engine
    from runner import FastEngine
    from lean import Lean
    from lean2 import Lean2
    eng = load_engine(CKPT, device="cuda"); tok = eng.tok
    torso = eng.model.torso.merge_and_unload().eval(); eng.model.torso = torso
    rt = tok._tokenizer if hasattr(tok, "_tokenizer") else tok.backend_tokenizer
    Q = qs(1)
    states = {}
    for target in (1000, 4000):
        L = []
        for r in reqs:
            ids = rt.encode(r["state"] if isinstance(r["state"], str) else render_state(r["state"]), add_special_tokens=False).ids
            if len(ids) >= target: L.append(tok.decode(ids[:target - 75]))
            if len(L) >= 16: break
        states[target] = L
    orig_fit = eng._fit
    def fast_fit(state_text, question_texts):
        max_len = eng.model.config.max_length
        enc = tok(question_texts, add_special_tokens=False, return_offsets_mapping=True)
        q = enc["input_ids"]; offs = enc["offset_mapping"]
        longest = max(len(x) for x in q)
        reserve = min(longest, max(1, int(max_len * eng.cfg.max_question_fraction)))
        cut = [max(0, len(x) - reserve) for x in q]
        eng._last_offsets = [o[c:] for o, c in zip(offs, cut)]
        q = [x[c:] for x, c in zip(q, cut)]
        budget = max(1, max_len - reserve)
        ch = par_chunks(state_text, 8)
        s = [i for e in rt.encode_batch(ch, add_special_tokens=False) for i in e.ids][:budget]
        return s, q
    res = {}
    import runner
    std_b = list(runner.BUCKETS); fine_b = list(range(64, 8193, 64))
    l2 = Lean2(torso, fuse="gnorm,prep,conv,fold")
    l2s = Lean2(torso, fuse="addrms,gnorm,silu,prep,conv,gemm_swiglu")
    auto = lambda ids: (l2 if ids.shape[1] >= 512 else l2s).forward(ids)
    for label, fwd, fit, bk in (("lean_base", Lean(torso).forward, orig_fit, std_b), ("lean2+fasthost+64buckets", auto, fast_fit, fine_b)):
        eng._fit = fit; runner.BUCKETS[:] = bk
        fe = FastEngine(eng, fwd, graph=True)
        for target, L in states.items():
            # equality of answers between paths is checked by comparing the probabilities of the first state
            for s in L[:2]: fe.ask(s, Q)
            ts, hs = [], []
            for s in L[2:]:
                torch.cuda.synchronize(); a = time.perf_counter(); fe._prep(s, Q); hs.append((time.perf_counter() - a) * 1e3)
                torch.cuda.synchronize(); a = time.perf_counter(); ans = fe.ask(s, Q); torch.cuda.synchronize(); ts.append((time.perf_counter() - a) * 1e3)
            g = st.median(fe.gpu_ms[-len(ts):]); fe.gpu_ms.clear()
            res[f"{label}_{target}"] = dict(e2e=med(ts), host_prep=med(hs), graph_gpu_ms=round(g, 2), buckets=[k[1] for k in fe.graphs], answer=str(ans)[:120])
            print(label, target, json.dumps(res[f"{label}_{target}"]), flush=True)
        del fe; torch.cuda.empty_cache()
    json.dump(res, open(os.path.expanduser("~/work/d1/host_e2e.json"), "w"), indent=1)

if __name__ == "__main__":
    e2e() if E2E else main()
