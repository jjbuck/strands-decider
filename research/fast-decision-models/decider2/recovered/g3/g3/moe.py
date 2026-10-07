"""G3: self-contained MoE decoder torso (granitemoe, smallthinker, mixtral) for training (LoRA) and as the reference for the
lean inference runtime. No remote code: weights are read from the HF safetensors and mapped to plain tensors.

Expert compute: tokens are sorted by expert and run through grouped GEMMs (torch._grouped_mm when available, else a per-expert
loop). LoRA r on q,k,v,o and per-expert LoRA on every expert projection (gate/up fused, down); router frozen.
"""
import os, json, glob, math, types
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open

HUB = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = {
    "st4b": "Tiiny/SmallThinker-4BA0.6B-Instruct",
    "gr3b": "ibm-granite/granite-3.1-3b-a800m-base",
    "gr1b": "ibm-granite/granite-3.1-1b-a400m-base",
    "euro": "utter-project/EuroMoE-2.6B-A0.6B-2512",
}


def snap(repo):
    d = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*")
    if not d:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo)
    return d[0]


class Cfg(types.SimpleNamespace):
    pass


def load_cfg(name):
    path = snap(MODELS.get(name, name))
    c = json.load(open(f"{path}/config.json"))
    mt = c["model_type"]
    g = Cfg(path=path, kind=mt, d=c["hidden_size"], L=c["num_hidden_layers"], H=c["num_attention_heads"],
            KV=c["num_key_value_heads"], eps=c.get("rms_norm_eps", 1e-6), theta=float(c.get("rope_theta", 10000.0)),
            vocab=c["vocab_size"], hidden_size=c["hidden_size"])
    g.hd = c.get("head_dim") or g.d // g.H
    if mt == "granitemoe":
        g.E, g.k, g.ffn = c["num_local_experts"], c["num_experts_per_tok"], c["intermediate_size"]
        g.act, g.gate_fn, g.router_pre = "silu", "softmax", False
        g.emb_mult, g.res_mult, g.attn_scale = c["embedding_multiplier"], c["residual_multiplier"], c["attention_multiplier"]
    elif mt == "smallthinker":
        g.E, g.k, g.ffn = c["moe_num_primary_experts"], c["moe_num_active_primary_experts"], c["moe_ffn_hidden_size"]
        g.act = "relu"; g.gate_fn = "softmax" if c.get("moe_primary_router_apply_softmax", False) else "sigmoid_norm"
        g.router_pre = True; g.emb_mult, g.res_mult, g.attn_scale = 1.0, 1.0, g.hd ** -0.5
        assert all(c.get("rope_layout", [1] * g.L)) and not any(c.get("sliding_window_layout", [0] * g.L)), "4B variant: RoPE everywhere, no SWA"
    elif mt == "mixtral":
        g.E, g.k, g.ffn = c["num_local_experts"], c["num_experts_per_tok"], c["intermediate_size"]
        g.act, g.gate_fn, g.router_pre = "silu", "softmax", False
        g.emb_mult, g.res_mult, g.attn_scale = 1.0, 1.0, g.hd ** -0.5
    else:
        raise ValueError(mt)
    return g


def _reader(path):
    idx = f"{path}/model.safetensors.index.json"
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    hs = [safe_open(f"{path}/{f}", "pt") for f in files]
    keymap = {k: h for h in hs for k in h.keys()}
    return lambda k: keymap[k].get_tensor(k)


def load_weights(g, dev="cuda", dtype=torch.bfloat16, layers=None):
    """-> dict with 'embed', 'norm', 'layers': [ {ln1, ln2, Wq, Wk, Wv, Wo, Wr [E,d], Wgu [E,2ffn,d] (gate rows then up rows), Wd [E,d,ffn]} ]"""
    get = _reader(g.path)
    T = lambda k: get(k).to(dev, dtype)
    W = dict(embed=T("model.embed_tokens.weight"), norm=T("model.norm.weight"), layers=[])
    for i in range(g.L if layers is None else layers):
        p = f"model.layers.{i}."
        d = dict(ln1=T(p + "input_layernorm.weight"), ln2=T(p + "post_attention_layernorm.weight"),
                 Wq=T(p + "self_attn.q_proj.weight"), Wk=T(p + "self_attn.k_proj.weight"), Wv=T(p + "self_attn.v_proj.weight"),
                 Wo=T(p + "self_attn.o_proj.weight"))
        if g.kind == "granitemoe":
            d["Wr"] = T(p + "block_sparse_moe.router.layer.weight")
            d["Wgu"] = T(p + "block_sparse_moe.input_linear.weight")          # [E, 2ffn, d]: first half -> act
            d["Wd"] = T(p + "block_sparse_moe.output_linear.weight")          # [E, d, ffn]
        elif g.kind == "smallthinker":
            d["Wr"] = T(p + "block_sparse_moe.primary_router.weight")
            q = p + "block_sparse_moe.experts."
            d["Wgu"] = torch.stack([torch.cat([get(f"{q}{e}.gate.weight"), get(f"{q}{e}.up.weight")], 0) for e in range(g.E)]).to(dev, dtype)
            d["Wd"] = torch.stack([get(f"{q}{e}.down.weight") for e in range(g.E)]).to(dev, dtype)
        elif g.kind == "mixtral":
            d["Wr"] = T(p + "block_sparse_moe.gate.weight")
            q = p + "block_sparse_moe.experts."
            d["Wgu"] = torch.stack([torch.cat([get(f"{q}{e}.w1.weight"), get(f"{q}{e}.w3.weight")], 0) for e in range(g.E)]).to(dev, dtype)
            d["Wd"] = torch.stack([get(f"{q}{e}.w2.weight") for e in range(g.E)]).to(dev, dtype)
        W["layers"].append(d)
    return W


# ------------------------------------------------------------------ grouped GEMM (rows sorted by expert)
_GMM = None


def has_grouped_mm():
    global _GMM
    if _GMM is None:
        try:
            a = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16); b = torch.randn(2, 32, 48, device="cuda", dtype=torch.bfloat16)
            offs = torch.tensor([16, 64], device="cuda", dtype=torch.int32)
            y = torch._grouped_mm(a, b, offs=offs)
            ref = torch.cat([a[:16] @ b[0], a[16:] @ b[1]])
            _GMM = bool((y.float() - ref.float()).abs().max() < 0.5)
        except Exception as e:
            print("grouped_mm unavailable:", repr(e)[:200]); _GMM = False
    return _GMM


def gmm(x, Wt, offs, counts=None):
    """x [M, K] sorted by expert; Wt [E, K, N] (a transposed view is fine); offs int32 cumulative ends [E] -> [M, N]"""
    if has_grouped_mm():
        return torch._grouped_mm(x, Wt, offs=offs)
    outs, s = [], 0
    for e, c in enumerate(counts.tolist()):
        if c: outs.append(x[s:s + c] @ Wt[e])
        s += c
    return torch.cat(outs) if outs else x.new_zeros(0, Wt.shape[-1])


def _rms(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _act_mul(h, ffn: int, relu: bool):
    g = h[:, :ffn]
    return (F.relu(g) if relu else F.silu(g)) * h[:, ffn:]


def _rot(x, cos, sin):
    hh = x.shape[-1] // 2
    xr = torch.cat([-x[..., hh:], x[..., :hh]], -1)
    return x * cos + xr * sin


def _gate_mul(y, w):
    return y * w.to(y.dtype)[:, None]


if os.environ.get("G3_COMPILE", "1") == "1":
    rms = torch.compile(_rms, dynamic=True); act_mul = torch.compile(_act_mul, dynamic=True)
    rot_c = torch.compile(_rot, dynamic=True); gate_mul = torch.compile(_gate_mul, dynamic=True)
else:
    rms, act_mul, rot_c, gate_mul = _rms, _act_mul, _rot, _gate_mul


def rope_cs(T, hd, theta, dev, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    fr = torch.arange(T, device=dev, dtype=torch.float32)[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rot(x, cos, sin):  # x [B, h, T, hd]
    h = x.shape[-1] // 2
    xr = torch.cat([-x[..., h:], x[..., :h]], -1)
    return x * cos + xr * sin


def route(g, logits):
    """logits fp32 [N, E] -> (gates [N,k] fp32, idx [N,k])"""
    v, i = logits.topk(g.k, dim=-1)
    if g.gate_fn == "softmax":
        w = torch.softmax(v, -1)
    else:
        w = torch.sigmoid(v); w = w / w.sum(-1, keepdim=True)
    return w, i


class LoRA(nn.Module):
    def __init__(self, din, dout, r, alpha, E=0):
        super().__init__()
        shp = (E,) if E else ()
        self.A = nn.Parameter(torch.empty(*shp, r, din)); self.B = nn.Parameter(torch.zeros(*shp, dout, r))
        a = self.A.view(-1, din)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.scale = alpha / r


class MoETorso(nn.Module):
    """forward(input_ids [B,T] right-padded, attention_mask [B,T]) -> namespace(last_hidden_state [B,T,d])"""

    def __init__(self, g, W, r=16, alpha=32, lora=True):
        super().__init__()
        self.g, self.W = g, W
        self.config = types.SimpleNamespace(hidden_size=g.d)
        self.lora = nn.ModuleList()
        self.use_lora = lora
        self.ckpt = False
        if lora:
            qd, kd = g.H * g.hd, g.KV * g.hd
            for _ in range(len(W["layers"])):
                self.lora.append(nn.ModuleDict(dict(q=LoRA(g.d, qd, r, alpha), k=LoRA(g.d, kd, r, alpha), v=LoRA(g.d, kd, r, alpha),
                                                    o=LoRA(qd, g.d, r, alpha), gu=LoRA(g.d, 2 * g.ffn, r, alpha, E=g.E),
                                                    dn=LoRA(g.ffn, g.d, r, alpha, E=g.E))))
        self.counts = []

    def lin(self, x, Wt, lo):
        y = x @ Wt.t()
        if lo is not None:
            y = y + (x @ lo.A.t().to(x.dtype)) @ lo.B.t().to(x.dtype) * lo.scale
        return y

    def moe(self, i, x, r_in):
        if getattr(self, "tg", False): return self.moe_tg(i, x, r_in)
        g, d = self.g, self.W["layers"][i]
        lo = self.lora[i] if self.use_lora else None
        logits = (r_in @ d["Wr"].t()).float()
        w, idx = route(g, logits)
        fe = idx.reshape(-1)
        order = torch.argsort(fe, stable=True)
        tok = order // g.k
        counts = torch.bincount(fe, minlength=g.E)
        offs = torch.cumsum(counts, 0).to(torch.int32)
        xs = x[tok]
        h = gmm(xs, d["Wgu"].transpose(1, 2), offs, counts)
        if lo is not None:
            h = h + gmm(gmm(xs, lo["gu"].A.to(xs.dtype).transpose(1, 2), offs, counts), lo["gu"].B.to(xs.dtype).transpose(1, 2), offs, counts) * lo["gu"].scale
        a = (F.silu(h[:, :g.ffn]) if g.act == "silu" else F.relu(h[:, :g.ffn])) * h[:, g.ffn:]
        y = gmm(a, d["Wd"].transpose(1, 2), offs, counts)
        if lo is not None:
            y = y + gmm(gmm(a, lo["dn"].A.to(a.dtype).transpose(1, 2), offs, counts), lo["dn"].B.to(a.dtype).transpose(1, 2), offs, counts) * lo["dn"].scale
        y = y * w.reshape(-1)[order].to(y.dtype)[:, None]
        out = torch.zeros_like(x).index_add_(0, tok, y)
        if not torch.is_grad_enabled() or not self.training:
            self.counts.append(counts)
        return out

    def moe_tg(self, i, x, r_in):
        """same math as moe(), on the padded sorted layout with the Triton grouped GEMM (gg.py)"""
        import gg as G
        g, d = self.g, self.W["layers"][i]
        lo = self.lora[i] if self.use_lora else None
        T = x.shape[0]
        logits = (r_in @ d["Wr"].t()).float()
        w, idx = route(g, logits)
        tok_pad, slot_pad, meta, counts = G.align(idx.reshape(-1), g.k, g.E, T)
        xs = G.Gather.apply(x, tok_pad, meta["inv"], g.k)
        h = G.GG.apply(xs, d["Wgu"], meta)
        if lo is not None:
            h = G.GG.apply(G.GG.apply(xs, lo["gu"].A, meta), lo["gu"].B, meta, h, lo["gu"].scale)
        a = act_mul(h, g.ffn, g.act == "relu")
        y = G.GG.apply(a, d["Wd"], meta)
        if lo is not None:
            y = G.GG.apply(G.GG.apply(a, lo["dn"].A, meta), lo["dn"].B, meta, y, lo["dn"].scale)
        wp = torch.cat([w.reshape(-1), w.new_zeros(1)])[slot_pad.clamp_min(-1)]  # slot -1 -> appended zero
        y = gate_mul(y, wp)
        out = G.Combine.apply(y, meta["inv"], T, g.k)
        if not torch.is_grad_enabled() or not self.training:
            self.counts.append(counts)
        return out

    def layer(self, i, x, mask_flat, cos, sin):
        g, d = self.g, self.W["layers"][i]
        lo = self.lora[i] if self.use_lora else None
        B, T, _ = x.shape
        h = rms(x, d["ln1"], g.eps)
        q = self.lin(h, d["Wq"], lo["q"] if lo else None).view(B, T, g.H, g.hd).transpose(1, 2)
        k = self.lin(h, d["Wk"], lo["k"] if lo else None).view(B, T, g.KV, g.hd).transpose(1, 2)
        v = self.lin(h, d["Wv"], lo["v"] if lo else None).view(B, T, g.KV, g.hd).transpose(1, 2)
        q, k = rot_c(q, cos, sin), rot_c(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=g.H != g.KV, scale=g.attn_scale)
        o = o.transpose(1, 2).reshape(B, T, g.H * g.hd)
        r_in = x
        x = x + self.lin(o, d["Wo"], lo["o"] if lo else None) * g.res_mult
        h2 = rms(x, d["ln2"], g.eps)
        xf = h2.reshape(B * T, -1); rf = (r_in if g.router_pre else h2).reshape(B * T, -1)
        if mask_flat is not None:
            sel = mask_flat.nonzero().squeeze(1)
            m = self.moe(i, xf[sel], rf[sel])
            mo = torch.zeros_like(xf).index_copy_(0, sel, m)
        else:
            mo = self.moe(i, xf, rf)
        return x + mo.view(B, T, -1) * g.res_mult

    def forward(self, input_ids, attention_mask=None, **kw):
        g = self.g
        B, T = input_ids.shape
        x = F.embedding(input_ids, self.W["embed"])
        if g.emb_mult != 1.0: x = x * g.emb_mult
        cos, sin = rope_cs(T, g.hd, g.theta, x.device, x.dtype)
        mask_flat = None
        if attention_mask is not None and not bool(attention_mask.all()):
            mask_flat = attention_mask.reshape(-1).bool()
        self.counts = []
        for i in range(len(self.W["layers"])):
            if self.ckpt and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(self.layer, i, x, mask_flat, cos, sin, use_reentrant=False)
            else:
                x = self.layer(i, x, mask_flat, cos, sin)
        x = rms(x, self.W["norm"], g.eps)
        return types.SimpleNamespace(last_hidden_state=x)

    @torch.no_grad()
    def merged(self, inplace=False):
        """weights with the LoRA folded in (for the lean runtime / eval)."""
        if not self.use_lora: return self.W
        out = dict(embed=self.W["embed"], norm=self.W["norm"], layers=[])
        for i, d in enumerate(self.W["layers"]):
            lo = self.lora[i]; n = d if inplace else dict(d)
            for key, nm in (("Wq", "q"), ("Wk", "k"), ("Wv", "v"), ("Wo", "o")):
                L = lo[nm]; n[key] = (d[key].float() + (L.B.float() @ L.A.float()) * L.scale).to(d[key].dtype)
            for key, nm in (("Wgu", "gu"), ("Wd", "dn")):
                L = lo[nm]; n[key] = (d[key].float() + torch.bmm(L.B.float(), L.A.float()) * L.scale).to(d[key].dtype)
            out["layers"].append(n)
        return out


def body_params(g):
    att = g.d * (g.H * g.hd) * 2 + g.d * (g.KV * g.hd) * 2
    exp = 3 * g.d * g.ffn
    tot = g.L * (att + g.E * exp + g.d * g.E)
    act = g.L * (att + g.k * exp + g.d * g.E)
    return tot, act
