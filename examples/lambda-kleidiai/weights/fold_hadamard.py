"""Fold a signed block Hadamard rotation into the merged Strands 2B matmul weights, for llama.cpp's prism.hadamard path.

  python fold_hadamard.py --src /opt/lam/merged-2b --out /opt/lam/merged-2b-h128 --block 128

For every folded weight W [out, in]: W' = W D H, where D = diag(signs[in]) and H is the normalized Sylvester-Walsh
Hadamard applied block-wise along the input axis. llama.cpp applies the same D H to each folded matmul's input at
runtime, so x W^T = (x D H)(W D H)^T exactly. The rotation spreads activation outliers across each block, which is what
per-row int8 activation quantisation (KleidiAI's Q8_0 path) needs. Writes hadamard_packing.json (the converter's
manifest, schema 1: untied output, explicit signs per input width).
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--block", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

FOLD = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def hadamard(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n ** 0.5


H = hadamard(a.block)
g = torch.Generator().manual_seed(a.seed)
signs = {}


def fold(w):
    o, i = w.shape
    assert i % a.block == 0, (w.shape, a.block)
    if i not in signs:
        signs[i] = (torch.randint(0, 2, (i,), generator=g) * 2 - 1).to(torch.float64)
    x = (w.double() * signs[i]).reshape(o, i // a.block, a.block) @ H
    return x.reshape(o, i).to(w.dtype).contiguous()


src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if not f.name.endswith(".safetensors") and f.name != "hadamard_packing.json":
        shutil.copy(f, out / f.name)
records = []
for sh in sorted(src.glob("*.safetensors")):
    sd = load_file(sh)
    for k in list(sd):
        if (k.startswith(("model.language_model.layers.", "model.layers.")) and k.endswith(".weight")
                and any(k.endswith(t + ".weight") for t in FOLD)):
            sd[k] = fold(sd[k])
            records.append({"name": k, "axis": -1, "role": "fold-before-matmul"})
    save_file(sd, out / sh.name, metadata={"format": "pt"})
manifest = {"schema_version": 1, "kind": "hadamard-weight-fold", "status": "requires-matching-runtime",
            "transform": {"name": "normalized-signed-sylvester-walsh-hadamard", "block_size": a.block,
                          "sign_mode": "explicit"},
            "signs": {str(k): [int(v) for v in s.tolist()] for k, s in signs.items()},
            "tensors": records, "tied_output": False}
json.dump(manifest, open(out / "hadamard_packing.json", "w"))
print(f"folded {len(records)} weights, block {a.block}, sign widths {sorted(signs)}")
