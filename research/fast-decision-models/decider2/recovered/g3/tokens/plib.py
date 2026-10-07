"""Shared library: hobson-v19 on CUDA with in-network token eviction (training-free).  Topic 'tokens'.

Prompt = state tokens (first q0) + question tokens (rest). Eviction happens after layer k on STATE tokens only;
question tokens (which carry options + <answer>) are never dropped, so the pointer readout is intact.
"""
import os, sys, json, glob, math, time, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F
from pydantic import TypeAdapter
from strands_decider.infer import load_engine
from strands_decider.prompting import render_state, render_question
from strands_decider.modeling import masked_log_softmax
import strands_decider.schema as SC
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

FULL_ATTN = (3, 7, 11, 15, 19, 23)
SINK = 4
ta = TypeAdapter(SC.Question)


class P:
    def __init__(self, dev="cuda"):
        ck = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*"))[0]
        self.eng = load_engine(ck, device=dev)
        self.model = self.eng.model
        tm = self.model.torso
        if hasattr(tm, "merge_and_unload"):
            tm = tm.merge_and_unload(); self.model.torso = tm
        tm.eval()
        self.tm = tm; self.dev = dev; self.NL = tm.config.num_hidden_layers; self.tok = self.eng.tok

    # ---------- prompt ----------
    def prep(self, state, qdict):
        q = ta.validate_python(qdict)
        rq = render_question(q)
        s, qs = self.eng._fit(render_state(state), [rq.text])
        opt = self.eng._option_idx([rq], 0)[0].tolist()
        return dict(s=s, q=qs[0], opt=opt, rq=rq, q0=len(s), L=len(s) + len(qs[0]))

    def temp_for(self, kind):
        cfg = self.model.config
        return float((getattr(cfg, "temperature_by_kind", None) or {}).get(kind, cfg.temperature))

    @torch.inference_mode()
    def finish(self, h, opt_abs, rq):
        h = self.tm.norm(h)
        pooled = h[:, -1].float()
        options = h[:, opt_abs].float()
        logits = self.model.head(pooled, options) / self.temp_for(rq.kind)
        lp = masked_log_softmax(logits, torch.tensor([rq.n_slots], device=self.dev))
        return lp.exp()[0, :rq.n_slots]

    @torch.inference_mode()
    def layer(self, i, h, cos, sin):
        return self.tm.layers[i](h, position_embeddings=(cos, sin), attention_mask=None, position_ids=None, past_key_values=None)

    @torch.inference_mode()
    def attn_scores(self, i, hin, cos, sin, nq):
        """mean (over heads, question rows) attention of the last nq rows onto all earlier rows at full-attn layer i."""
        at = self.tm.layers[i].self_attn
        x = self.tm.layers[i].input_layernorm(hin)
        B, L, _ = x.shape
        q, _g = torch.chunk(at.q_proj(x[:, L - nq:]).view(B, nq, -1, at.head_dim * 2), 2, dim=-1)
        q = at.q_norm(q.reshape(B, nq, -1, at.head_dim)).transpose(1, 2)
        k = at.k_norm(at.k_proj(x).view(B, L, -1, at.head_dim)).transpose(1, 2)
        # rotary on q rows uses their own positions
        cq, sq = cos[:, L - nq:], sin[:, L - nq:]
        q = apply_rotary_pos_emb(q, q, cq, sq)[0]
        k = apply_rotary_pos_emb(k, k, cos, sin)[0]
        H, KV = q.shape[1], k.shape[1]
        k = k.repeat_interleave(H // KV, dim=1)
        lg = (q.float() @ k.float().transpose(-1, -2)) * at.scaling          # [B,H,nq,L]
        kp = torch.arange(L, device=hin.device)
        qp = torch.arange(L - nq, L, device=hin.device)
        lg = lg.masked_fill(kp[None, :] > qp[:, None], float("-inf"))
        p = torch.softmax(lg, -1)
        return p[0, :, :, :L - nq].mean(dim=(0, 1))                        # [n_state_alive]

    # ---------- base pass with stored layer inputs ----------
    @torch.inference_mode()
    def base(self, pr):
        ids = torch.tensor([pr["s"] + pr["q"]], device=self.dev)
        emb = self.tm.embed_tokens(ids)
        L = ids.shape[1]
        pos = torch.arange(L, device=self.dev).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.tm.rotary_emb(emb, pos)
        hs = [emb]
        h = emb
        for i in range(self.NL):
            h = self.layer(i, h, cos, sin)
            hs.append(h)
        pr.update(hs=hs, cos=cos, sin=sin)
        n_q = L - pr["q0"]
        opt_abs = [pr["q0"] + o for o in pr["opt"]]
        return self.finish(h, opt_abs, pr["rq"])

    # ---------- pruned pass ----------
    @torch.inference_mode()
    def pruned(self, pr, sched, scorer="qattn", sink=SINK, rng=None, ids_from_scratch=False, qmode='all'):
        """sched: list of (k, keep) applied after layer k; keep<=1 is a fraction of the ORIGINAL state length, >1 an absolute count.
        Uses stored layer inputs in pr['hs'] (base() must have been called) unless ids_from_scratch."""
        q0, L = pr["q0"], pr["L"]
        nq = L - q0
        k0 = sched[0][0]
        h = pr["hs"][k0 + 1]
        cos, sin = pr["cos"], pr["sin"]
        alive_state = torch.arange(q0, device=self.dev)           # original indices of alive state tokens, sorted
        hin_first = pr["hs"][k0]
        stage = {k: m for k, m in sched}
        for i in range(k0, self.NL):
            if i > k0:
                hin = h
                h = self.layer(i, h, cos, sin)
            else:
                hin = hin_first
                # h already = output of layer k0 (full set)
            if i in stage:
                m = stage[i]
                ns = alive_state.numel()
                tgt = int(round(m * q0)) if m <= 1 else int(m)
                tgt = max(tgt, sink) if sink > 0 else max(tgt, 0)
                tgt = min(tgt, ns)
                if tgt < ns:
                    if scorer == "qattn":
                        sc = self.attn_scores(i, hin, cos, sin, nq)          # over alive state rows
                    elif scorer == "random":
                        sc = torch.rand(ns, device=self.dev, generator=rng)
                    elif scorer == "tail":
                        sc = torch.arange(ns, device=self.dev, dtype=torch.float32)
                    elif scorer == "head":
                        sc = -torch.arange(ns, device=self.dev, dtype=torch.float32)
                    else:
                        raise ValueError(scorer)
                    if sink > 0:
                        sc = sc.clone(); sc[:sink] = float("inf")
                    top = torch.topk(sc, tgt).indices.sort().values if tgt > 0 else torch.zeros(0, dtype=torch.long, device=self.dev)
                    keep_rows = torch.cat([top, torch.arange(ns, ns + nq, device=self.dev)])
                    h = h[:, keep_rows]
                    cos, sin = cos[:, keep_rows], sin[:, keep_rows]
                    alive_state = alive_state[top]
        n_alive = alive_state.numel()
        opt_abs = [n_alive + o for o in pr["opt"]]
        return self.finish(h, opt_abs, pr["rq"])

    @torch.inference_mode()
    def pruned_qr(self, pr, k, qlayer=None, scorer="qattn", state_keep=0.0, sink=0):
        """State tokens evicted after layer k (keep fraction state_keep); question tokens other than the K option-end rows and the final <answer> row
        evicted after layer qlayer (default k). Everything else as in pruned()."""
        qlayer = k if qlayer is None else qlayer
        q0, L = pr["q0"], pr["L"]; nq = L - q0
        h = pr["hs"][k + 1]; cos, sin = pr["cos"], pr["sin"]
        n_alive = q0
        # state eviction at k
        tgt = max(int(round(state_keep * q0)), sink)
        if tgt < q0:
            if tgt > 0 and scorer == "qattn":
                sc = self.attn_scores(k, pr["hs"][k], cos, sin, nq).clone()
                if sink > 0: sc[:sink] = float("inf")
                top = torch.topk(sc, tgt).indices.sort().values
            else:
                top = torch.arange(tgt, device=self.dev)
            rows = torch.cat([top, torch.arange(q0, L, device=self.dev)])
            h = h[:, rows]; cos = cos[:, rows]; sin = sin[:, rows]; n_alive = tgt
        qkeep = sorted(set(pr["opt"] + [nq - 1]))
        for i in range(k + 1, self.NL):
            if i == qlayer + 1 and len(qkeep) < nq:
                rows = torch.cat([torch.arange(n_alive, device=self.dev), n_alive + torch.tensor(qkeep, device=self.dev)])
                h = h[:, rows]; cos = cos[:, rows]; sin = sin[:, rows]
                qmap = {o: j for j, o in enumerate(qkeep)}; cur_opt = [n_alive + qmap[o] for o in pr["opt"]]
            h = self.layer(i, h, cos, sin)
        if qlayer >= self.NL - 1 or len(qkeep) >= nq:
            cur_opt = [n_alive + o for o in pr["opt"]]
        return self.finish(h, cur_opt, pr["rq"])


def _stale_layer(self, i, hq, hs_state, cos_s, sin_s, cos_q, sin_q):
    """Full-attention layer i for question rows only; keys/values of the (dead) state tokens come from their STALE residual hs_state."""
    layer = self.tm.layers[i]; at = layer.self_attn
    x = layer.input_layernorm(hq); xs = layer.input_layernorm(hs_state)
    B, Lq, _ = x.shape; Ls = xs.shape[1]; hd = at.head_dim
    q, gate = torch.chunk(at.q_proj(x).view(B, Lq, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(B, Lq, -1)
    q = at.q_norm(q.reshape(B, Lq, -1, hd)).transpose(1, 2)
    kq = at.k_norm(at.k_proj(x).view(B, Lq, -1, hd)).transpose(1, 2)
    vq = at.v_proj(x).view(B, Lq, -1, hd).transpose(1, 2)
    ks = at.k_norm(at.k_proj(xs).view(B, Ls, -1, hd)).transpose(1, 2)
    vs = at.v_proj(xs).view(B, Ls, -1, hd).transpose(1, 2)
    q, kq = apply_rotary_pos_emb(q, kq, cos_q, sin_q)
    ks = apply_rotary_pos_emb(ks, ks, cos_s, sin_s)[0]
    K = torch.cat([ks, kq], 2); V = torch.cat([vs, vq], 2)
    allow = torch.cat([torch.ones(Lq, Ls, dtype=torch.bool, device=hq.device), torch.tril(torch.ones(Lq, Lq, dtype=torch.bool, device=hq.device))], 1)
    o = F.scaled_dot_product_attention(q, K, V, attn_mask=allow, scale=at.scaling, enable_gqa=True)
    o = o.transpose(1, 2).reshape(B, Lq, -1) * torch.sigmoid(gate)
    h = hq + at.o_proj(o)
    return h + layer.mlp(layer.post_attention_layernorm(h))


@torch.inference_mode()
def pruned_stale(self, pr, k, stale_from=None, qro_layer=None):
    """State tokens stop after layer k (no further compute) but later FULL-attention layers still read K/V projected from their stale residual h_k.
    Linear-attention (DeltaNet) layers see question rows only. Optionally drop non-readout question rows after qro_layer."""
    q0, L = pr["q0"], pr["L"]; nq = L - q0
    hstate = pr["hs"][k + 1][:, :q0]; hq = pr["hs"][k + 1][:, q0:]
    cos_s, sin_s = pr["cos"][:, :q0], pr["sin"][:, :q0]; cos_q, sin_q = pr["cos"][:, q0:], pr["sin"][:, q0:]
    qkeep = sorted(set(pr["opt"] + [nq - 1])); cur_opt = list(pr["opt"])
    for i in range(k + 1, self.NL):
        if qro_layer is not None and i == qro_layer + 1 and len(qkeep) < nq:
            idx = torch.tensor(qkeep, device=self.dev)
            hq = hq[:, idx]; cos_q = cos_q[:, idx]; sin_q = sin_q[:, idx]
            qmap = {o: j for j, o in enumerate(qkeep)}; cur_opt = [qmap[o] for o in pr["opt"]]
        if i in FULL_ATTN:
            hq = self._stale_layer(i, hq, hstate, cos_s, sin_s, cos_q, sin_q)
        else:
            hq = self.layer(i, hq, cos_q, sin_q)
    return self.finish(hq, cur_opt, pr["rq"])

P._stale_layer = _stale_layer
P.pruned_stale = pruned_stale
