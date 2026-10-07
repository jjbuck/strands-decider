"""Latency of the hobson-v19 torso truncated to L layers, 1 question, cold input (fresh text each call), A10G. eager vs CUDA-graph (bucketed padded length)."""
import sys, statistics as st
from lib import *
import os
base = ("user: Hi, I'd like to return the blue jacket from order #W1234567 and exchange the boots for a size 10. "
        "assistant: Sure, I can help with that. Let me look up your order. tool: {\"order_id\": \"#W1234567\", \"items\": "
        "[{\"name\": \"jacket\", \"color\": \"blue\", \"price\": 129.99}, {\"name\": \"boots\", \"size\": 9, \"price\": 89.5}], \"status\": \"delivered\"} ")
def make_state(tok, n):
    ids = tok(base * (n // 60 + 2), add_special_tokens=False)["input_ids"][:n]
    return tok.decode(ids)
eng = load(); R = Runner(eng); tm = eng.model.torso; head = eng.model.head; tok = eng.tok
dev = eng.device
Ls = [int(x) for x in sys.argv[1].split(",")]; Ns = [int(x) for x in sys.argv[2].split(",")]; REPS = int(sys.argv[3]); MODE = sys.argv[4]
q = jb_question({"question": {"type": "choice", "instructions": "Which option best describes the user's intent?", "criteria": {"a": "first option", "b": "second option", "c": "third option"}}})
KMAX = 4
def sync(): torch.cuda.synchronize()
def pct(ts, p): ts = sorted(ts); return ts[int(p * (len(ts) - 1))]

def forward_body(ids, L, pos3, masks_cache):
    emb = tm.embed_tokens(ids); B, T = ids.shape
    pe = tm.rotary_emb(emb, pos3)
    h = emb
    for i in range(L):
        o = tm.layers[i](h, position_embeddings=pe, attention_mask=masks_cache[R.types[i]], position_ids=pos3[0], past_key_values=None, use_cache=False)
        h = o[0] if isinstance(o, tuple) else o
    return tm.norm(h)

class Graphed:
    def __init__(self, L, Tb):
        self.L, self.Tb = L, Tb
        self.ids = torch.zeros(1, Tb, dtype=torch.long, device=dev); self.ans = torch.zeros(1, dtype=torch.long, device=dev); self.opt = torch.zeros(KMAX, dtype=torch.long, device=dev)
        pos = torch.arange(Tb, device=dev).view(1, 1, -1).expand(4, 1, -1).contiguous()
        emb0 = tm.embed_tokens(self.ids); am = torch.ones(1, Tb, dtype=torch.long, device=dev)
        mk = dict(config=R.cfg, inputs_embeds=emb0, attention_mask=am, past_key_values=None, position_ids=pos[0])
        self.masks = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
        self.pos = pos
        def body():
            hid = forward_body(self.ids, L, self.pos[1:] if False else self.pos[1:], self.masks)
            pooled = hid[0, self.ans].float(); options = hid[0, self.opt].float().unsqueeze(0)
            return head(pooled, options)[0]
        self.body = body
        s = torch.cuda.Stream()
        with torch.inference_mode():
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3): self.body()
            torch.cuda.current_stream().wait_stream(s); sync()
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g): self.out = self.body()
    def __call__(self, ids_list, ans, opt):
        n = len(ids_list); self.ids.zero_(); self.ids[0, :n] = torch.tensor(ids_list, device=dev)
        self.ans.fill_(ans); k = len(opt); self.opt.fill_(0); self.opt[:k] = torch.tensor(opt, device=dev)
        self.g.replay(); return self.out[:k]

res = {}
for N in Ns:
    for L in Ls:
        # prepare cold inputs
        ins = []
        for r in range(REPS + 3):
            st_ = make_state(tok, N) + f" session {r}-{random.random()}"
            ins.append(st_)
        if MODE == "graph":
            e0 = encode(eng, ins[0], q); Tb = ((e0["T"] + 63) // 64) * 64 + 64
            try:
                G = Graphed(L, Tb)
            except Exception as ex:
                print("graph capture failed", repr(ex)[:300]); continue
        ts_total = []; ts_tok = []; ts_gpu = []
        for r, s in enumerate(ins):
            sync(); t0 = time.perf_counter()
            e = encode(eng, s, q)
            t1 = time.perf_counter()
            with torch.inference_mode():
                if MODE == "eager":
                    out = R.run(e["ids"], upto=L); lg = R.head_logits(out["final"], e["opt"]).cpu()
                else:
                    lg = G(e["ids"][0].tolist(), e["T"] - 1, e["opt"].tolist()).cpu()
            sync(); t2 = time.perf_counter()
            if r >= 3: ts_total.append((t2 - t0) * 1e3); ts_tok.append((t1 - t0) * 1e3); ts_gpu.append((t2 - t1) * 1e3)
        key = f"{MODE}_L{L}_N{N}"
        res[key] = dict(T=e["T"], median=st.median(ts_total), p95=pct(ts_total, .95), tok_med=st.median(ts_tok), gpu_med=st.median(ts_gpu), min=min(ts_total))
        print(f"{MODE} L={L:2d} tokens={e['T']:5d}: total median {res[key]['median']:7.1f} ms p95 {res[key]['p95']:7.1f} | tokenize+render {res[key]['tok_med']:5.1f} | model+head {res[key]['gpu_med']:7.1f} (min total {res[key]['min']:.1f})", flush=True)
        if MODE == "graph": del G; torch.cuda.empty_cache()
json.dump(res, open(f"bench_depth_{MODE}.json", "w"), indent=1)
