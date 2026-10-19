"""Fold one QuaRot-style randomized Hadamard rotation Q = D H (2048 x 2048) into the residual stream of the merged
Strands 2B (Qwen3.5) checkpoint. No runtime transform: the GGUF converts and runs like an unrotated model.

  python fold_quarot.py --src /opt/lam/merged-2b --out /opt/lam/rot/merged-qr [--seed 20261019]

Residual stream x (row vector) becomes x Q. Qwen3.5's RMSNorm is x_hat * (1 + w) (transformers Qwen3_5RMSNorm; the
GGUF converter writes 1 + w), and rms(x Q) = rms(x), so with an identity norm a reader sees x_hat Q.
  readers  (q/k/v_proj; GDN in_proj_qkv, in_proj_z, in_proj_b, in_proj_a; mlp gate/up_proj), W [out, in]:
           W <- W diag(1 + w_norm) Q, then that norm's w <- 0 (identity in the (1 + w) form)
  writers  (self_attn.o_proj, linear_attn.out_proj, mlp.down_proj), W [d, in]:  W <- Q^T W
  embedding E [vocab, d]: E <- E Q
  final norm: w <- 0; the engine maps the returned x_hat Q back with h = (x_hat Q) Q^T * (1 + w_final)
           (q_and_scale.npz: q = Q, scale = 1 + w_final, both float64), see lcpp_engine.py DEC_UNROTATE.
Untouched (they act on projection outputs, not on the residual stream): per-head q_norm / k_norm, the GDN conv1d,
A_log, dt_bias and gated output norm (linear_attn.norm, weight * x_hat * silu(z)). The model has no linear biases
(checked: asserts no *.bias in the text layers). Not made consistent: the tied LM head (it would need
diag(1 + w_final) folded into the shared embedding) and the MTP block (mtp.*); neither runs in the decider's
embeddings-mode forward (the prefill patch returns before the LM head, MTP is a separate graph type).
All products are formed in float64 and stored in float32.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20261019)
a = ap.parse_args()
torch.set_num_threads(32)


def hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if not f.name.endswith(".safetensors") and f.name != "hadamard_packing.json":
        shutil.copy(f, out / f.name)  # config, tokenizer, the shard index (names unchanged)
cfg = json.load(open(src / "config.json"))
tc = cfg.get("text_config", cfg)
d, n_layer, types = tc["hidden_size"], tc["num_hidden_layers"], tc["layer_types"]

shards = sorted(src.glob("*.safetensors"))
assert len(shards) == 1, shards  # the merge writes one shard; keeps the index trivial
sd = load_file(shards[0])
P = "model.language_model."
assert P + "embed_tokens.weight" in sd
assert not [k for k in sd if k.startswith(P) and k.endswith(".bias")], "unexpected biases"

g = torch.Generator().manual_seed(a.seed)
D = (torch.randint(0, 2, (d,), generator=g) * 2 - 1).to(torch.float64)
Q = D[:, None] * hadamard(d)  # Q = diag(D) H
assert torch.allclose(Q @ Q.T, torch.eye(d, dtype=torch.float64), atol=1e-12)
done = []


def scale(norm_key):
    w = sd[norm_key].double()
    sd[norm_key] = torch.zeros_like(sd[norm_key])  # identity: 1 + 0
    return 1.0 + w


def read(key, s):
    w = sd[key].double()
    assert w.shape[1] == d, (key, w.shape)
    sd[key] = ((w * s[None, :]) @ Q).float().contiguous()
    done.append((key, "reader"))


def write(key):
    w = sd[key].double()
    assert w.shape[0] == d, (key, w.shape)
    sd[key] = (Q.T @ w).float().contiguous()
    done.append((key, "writer"))


for i in range(n_layer):
    L = f"{P}layers.{i}."
    s_in = scale(L + "input_layernorm.weight")
    if types[i] == "linear_attention":
        for t in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            read(L + f"linear_attn.{t}.weight", s_in)
        write(L + "linear_attn.out_proj.weight")
    else:
        for t in ("q_proj", "k_proj", "v_proj"):
            read(L + f"self_attn.{t}.weight", s_in)
        write(L + "self_attn.o_proj.weight")
    s_post = scale(L + "post_attention_layernorm.weight")
    for t in ("gate_proj", "up_proj"):
        read(L + f"mlp.{t}.weight", s_post)
    write(L + "mlp.down_proj.weight")
    # every 2-D text-layer weight must have been handled (norm vectors and GDN 1-D/conv params are excluded)
    left = [k for k in sd if k.startswith(L) and sd[k].ndim == 2 and (k, "reader") not in done and (k, "writer") not in done]
    assert not left, left

E = sd[P + "embed_tokens.weight"]
sd[P + "embed_tokens.weight"] = (E.double() @ Q).float().contiguous()
done.append((P + "embed_tokens.weight", "embedding"))
g_final = scale(P + "norm.weight")
assert "lm_head.weight" not in sd and tc.get("tie_word_embeddings", cfg.get("tie_word_embeddings")), "untied head"
save_file(sd, out / shards[0].name, metadata={"format": "pt"})
np.savez(out / "q_and_scale.npz", q=Q.numpy(), scale=g_final.numpy())
rep = {"seed": a.seed, "d": d, "readers": sum(r == "reader" for _, r in done),
       "writers": sum(r == "writer" for _, r in done), "embedding": 1,
       "norms_set_identity": 2 * n_layer + 1, "final_scale_range": [float(g_final.min()), float(g_final.max())]}
json.dump(rep, open(out / "quarot_report.json", "w"), indent=1)
print(json.dumps(rep))
