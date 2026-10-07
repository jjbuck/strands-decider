"""validate moe.MoETorso against the HF implementation (granitemoe native; smallthinker remote code with import shims)"""
import os, sys, json, torch, typing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe as M
name = sys.argv[1]
import transformers, transformers.cache_utils as CU, transformers.utils as TU
if not hasattr(CU, "HybridCache"): CU.HybridCache = CU.DynamicCache
if not hasattr(TU, "LossKwargs"):
    class LossKwargs(typing.TypedDict, total=False): pass
    TU.LossKwargs = LossKwargs
import transformers.modeling_rope_utils as RU
if "default" not in RU.ROPE_INIT_FUNCTIONS:
    def _default(config, device=None, **kw):
        hd = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return 1.0 / (config.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / hd)), 1.0
    RU.ROPE_INIT_FUNCTIONS["default"] = _default
g = M.load_cfg(name)
tok = transformers.AutoTokenizer.from_pretrained(M.MODELS[name])
txt = open(os.path.expanduser("~/work/sd/README.md")).read()[:6000]
ids = tok(txt, return_tensors="pt")["input_ids"][:, :int(sys.argv[2]) if len(sys.argv) > 2 else 512].cuda()
hf = transformers.AutoModelForCausalLM.from_pretrained(M.MODELS[name], dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa").cuda().eval()
with torch.no_grad():
    out = hf.model(input_ids=ids, output_hidden_states=False)
    r = out.last_hidden_state[0].float()
    hf.config._attn_implementation = "eager"
    for m in hf.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"): m.config._attn_implementation = "eager"
    r2 = hf.model(input_ids=ids).last_hidden_state[0].float()
    c2 = torch.nn.functional.cosine_similarity(r, r2, dim=-1)
    print(json.dumps(dict(hf_sdpa_vs_eager_cos_mean=round(c2.mean().item(), 6), cos_min=round(c2.min().item(), 6), rel=round(((r - r2).norm() / r.norm()).item(), 5))))
del hf; torch.cuda.empty_cache()
W = M.load_weights(g)
t = M.MoETorso(g, W, lora=False).eval()
with torch.no_grad():
    o = t(ids).last_hidden_state[0].float()
cos = torch.nn.functional.cosine_similarity(r, o, dim=-1)
print('worst positions', cos.argsort()[:8].tolist(), [round(x, 4) for x in cos.sort().values[:8].tolist()])
print(json.dumps(dict(model=name, T=ids.shape[1], cos_mean=round(cos.mean().item(), 6), cos_min=round(cos.min().item(), 6), rel=round(((r - o).norm() / r.norm()).item(), 5),
                      grouped_mm=M.has_grouped_mm())))
