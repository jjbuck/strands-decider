import time, torch
from runner import *
from strands_decider.prompting import render_question, render_state
from strands_decider.infer import _to_answer, _option_token_index
from strands_decider.modeling import masked_log_softmax, apply_temperature
LQ_BUCKETS = [32, 48, 64, 96, 128, 192, 256, 384, 512]
def qbucket(n):
    for b in LQ_BUCKETS:
        if b >= n: return b
    return n
class PackedEngine:
    """state + M question branches in ONE pass (weights read once), CUDA-graphed per (Ls_bucket, M, Lq_bucket)."""
    def __init__(self, eng, lean, graph=True, maxK=8):
        self.eng, self.lean, self.graph, self.maxK = eng, lean, graph, maxK
        self.head = eng.model.head.eval(); self.cfgm = eng.model.config; self.tok = eng.tok
        self.graphs = {}; self.dev = "cuda"; self.gpu_ms = []
    def _core(self, s_ids, n_s, b_ids, ans, opt, nsl, temp):
        _, hb = self.lean.forward_packed(s_ids, n_s, b_ids)      # [M, Lq, d]
        M = b_ids.shape[0]; ar = torch.arange(M, device=b_ids.device)
        pooled = hb[ar, ans].float(); options = hb[ar[:, None], opt].float()
        logits = apply_temperature(self.head(pooled, options), temp)
        return masked_log_softmax(logits, nsl).exp()
    def _get(self, Ls, M, Lq):
        key = (Ls, M, Lq)
        if key in self.graphs: return self.graphs[key]
        d = self.dev
        st = dict(s_ids=torch.zeros(Ls, dtype=torch.long, device=d), n_s=torch.tensor(Ls, dtype=torch.long, device=d),
                  b_ids=torch.zeros(M, Lq, dtype=torch.long, device=d), ans=torch.zeros(M, dtype=torch.long, device=d),
                  opt=torch.zeros(M, self.maxK, dtype=torch.long, device=d), nsl=torch.full((M,), 2, dtype=torch.long, device=d), temp=torch.ones(M, device=d))
        if self.graph:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.inference_mode():
                for _ in range(3): self._core(**st)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g):
                st["out"] = self._core(**st)
            st["g"] = g
        self.graphs[key] = st
        return st
    @torch.inference_mode()
    def ask(self, state, questions):
        eng = self.eng
        names = list(questions.keys())
        rendered = [render_question(questions[n]) for n in names]
        s, q = eng._fit(render_state(state), [r.text for r in rendered])
        offs = eng._last_offsets
        M = len(q); Ls = bucket(len(s)); Lq = qbucket(max(len(x) for x in q))
        st = self._get(Ls, M, Lq)
        pad = self.tok.pad_token_id or 0
        ans = [len(x) - 1 for x in q]
        opts = [[i for i in _option_token_index(o, r.option_spans, 0)] for r, o in zip(rendered, offs)]
        cfg = self.cfgm; by = getattr(cfg, "temperature_by_kind", None) or {}
        temp = torch.tensor([float(by.get(r.kind, cfg.temperature)) for r in rendered])
        nsl = [r.n_slots for r in rendered]
        flat = torch.tensor(s + [pad] * (Ls - len(s)) + [len(s)] + [t for x in q for t in (x + [pad] * (Lq - len(x)))] + ans
                            + [t for o in opts for t in (o + [0] * (self.maxK - len(o)))] + nsl, dtype=torch.long).to(self.dev)
        o = 0
        st["s_ids"].copy_(flat[o:o + Ls]); o += Ls
        st["n_s"].copy_(flat[o:o + 1][0]); o += 1
        st["b_ids"].copy_(flat[o:o + M * Lq].view(M, Lq)); o += M * Lq
        st["ans"].copy_(flat[o:o + M]); o += M
        st["opt"].copy_(flat[o:o + M * self.maxK].view(M, self.maxK)); o += M * self.maxK
        st["nsl"].copy_(flat[o:o + M]); st["temp"].copy_(temp, non_blocking=True)
        if self.graph:
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); st["g"].replay(); e1.record(); probs = st["out"]
        else:
            probs = self._core(st["s_ids"], st["n_s"], st["b_ids"], st["ans"], st["opt"], st["nsl"], st["temp"])
        probs = probs.cpu()
        if self.graph: self.gpu_ms.append(e0.elapsed_time(e1))
        return {n: _to_answer(r, probs[i, :r.n_slots].tolist(), ordinal_smoothing=cfg.ordinal_smoothing) for i, (n, r) in enumerate(zip(names, rendered))}
