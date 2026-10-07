import time, torch, torch.nn.functional as F
from strands_decider.prompting import render_question, render_state
from strands_decider.infer import _to_answer
from strands_decider.modeling import gather_options, masked_log_softmax, apply_temperature
BUCKETS = [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192]
def bucket(n):
    for b in BUCKETS:
        if b >= n: return b
    return n

class FastEngine:
    """Single-pass engine for N questions, one row per question (full prompt per row), optional CUDA graph."""
    def __init__(self, eng, fwd, graph=True, maxK=8):
        self.eng, self.fwd, self.graph, self.maxK = eng, fwd, graph, maxK
        self.head = eng.model.head.eval()
        self.cfgm = eng.model.config
        self.graphs = {}
        self.dev = "cuda"
        self.tok = eng.tok
        self.gpu_ms = []
    def _prep(self, state, questions):
        eng = self.eng
        names = list(questions.keys())
        rendered = [render_question(questions[n]) for n in names]
        state_text = render_state(state)
        s, q = eng._fit(state_text, [r.text for r in rendered])
        offs = eng._last_offsets
        from strands_decider.infer import _option_token_index
        rows, ans, opts, nsl, kinds = [], [], [], [], []
        for r, qi, o in zip(rendered, q, offs):
            ids = s + qi
            rows.append(ids); ans.append(len(ids) - 1)
            opts.append([len(s) + i for i in _option_token_index(o, r.option_spans, 0)])
            nsl.append(r.n_slots); kinds.append(r.kind)
        return names, rendered, rows, ans, opts, nsl, kinds
    def _core(self, ids, ans, opt, nsl, temp):
        hidden = self.fwd(ids)                      # [B, L, d]
        B = ids.shape[0]
        ar = torch.arange(B, device=ids.device)
        pooled = hidden[ar, ans].float()
        options = hidden[ar[:, None], opt].float()
        logits = self.head(pooled, options)
        logits = apply_temperature(logits, temp)
        return masked_log_softmax(logits, nsl).exp()
    def _get(self, B, L):
        key = (B, L)
        if key in self.graphs: return self.graphs[key]
        st = dict(ids=torch.zeros(B, L, dtype=torch.long, device=self.dev), ans=torch.zeros(B, dtype=torch.long, device=self.dev),
                  opt=torch.zeros(B, self.maxK, dtype=torch.long, device=self.dev), nsl=torch.full((B,), 2, dtype=torch.long, device=self.dev),
                  temp=torch.ones(B, device=self.dev))
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
        names, rendered, rows, ans, opts, nsl, kinds = self._prep(state, questions)
        B = len(rows); L = bucket(max(len(r) for r in rows))
        st = self._get(B, L)
        pad = self.tok.pad_token_id or 0
        ids = torch.tensor([r + [pad] * (L - len(r)) for r in rows], dtype=torch.long)
        optp = torch.tensor([o + [0] * (self.maxK - len(o)) for o in opts], dtype=torch.long)
        cfg = self.cfgm; by = getattr(cfg, "temperature_by_kind", None) or {}
        temp = torch.tensor([float(by.get(k, cfg.temperature)) for k in kinds])
        packed = torch.cat([ids.reshape(-1), torch.tensor(ans), optp.reshape(-1), torch.tensor(nsl)]).to(self.dev, non_blocking=False)
        st["ids"].copy_(packed[:B*L].view(B, L)); o = B*L
        st["ans"].copy_(packed[o:o+B]); o += B
        st["opt"].copy_(packed[o:o+B*self.maxK].view(B, self.maxK)); o += B*self.maxK
        st["nsl"].copy_(packed[o:o+B]); st["temp"].copy_(temp, non_blocking=True)
        if self.graph:
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); st["g"].replay(); e1.record(); probs = st["out"]
        else:
            probs = self._core(st["ids"], st["ans"], st["opt"], st["nsl"], st["temp"])
        probs = probs.cpu()
        if self.graph: self.gpu_ms.append(e0.elapsed_time(e1))
        return {n: _to_answer(r, probs[i, :r.n_slots].tolist(), ordinal_smoothing=cfg.ordinal_smoothing) for i, (n, r) in enumerate(zip(names, rendered))}
