"""box/fold_hadamard.py with two switches for the rotation-placement ablation: which matmuls to fold, and whether the
random signs are used. Same fold (W' = W D H, block-wise along the input axis), same manifest (schema 1), same seed
handling, so `--block 512` with the defaults reproduces the deployed fold exactly.

  python fold_hadamard_sel.py --src DIR --out DIR --block 512                                   # A1 (deployed)
  python fold_hadamard_sel.py --src DIR --out DIR --block 2048                                  # A2
  python fold_hadamard_sel.py --src DIR --out DIR --block 512 --sign-mode identity              # A3 (D = I)
  python fold_hadamard_sel.py --src QR  --out DIR --block 2048 --only mlp.down_proj             # A5
  python fold_hadamard_sel.py --src QR  --out DIR --block 2048 \\
      --only mlp.down_proj,self_attn.o_proj,linear_attn.out_proj                                 # A6
  python fold_hadamard_sel.py --src QR  --out DIR --block 512                                   # A7

The source may itself be residual-rotated (rotabl/fold_quarot.py): this fold acts on matmul input axes only, which
commutes with the residual rotation's output-axis fold on the writers. Non-safetensors files (q_and_scale.npz,
quarot_report.json) are copied along.
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
ap.add_argument("--only", default="", help="comma list of FOLD suffixes (default: all of FOLD)")
ap.add_argument("--sign-mode", choices=["explicit", "identity"], default="explicit")
a = ap.parse_args()
torch.set_num_threads(32)

FOLD = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
only = tuple(x for x in a.only.split(",") if x)
assert all(x in FOLD for x in only), only
targets = only or FOLD


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
    if i not in signs:  # drawn in the same order as box/fold_hadamard.py, whatever the sign mode
        signs[i] = (torch.randint(0, 2, (i,), generator=g) * 2 - 1).to(torch.float64)
    s = signs[i] if a.sign_mode == "explicit" else torch.ones(i, dtype=torch.float64)
    x = (w.double() * s).reshape(o, i // a.block, a.block) @ H
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
                and any(k.endswith(t + ".weight") for t in targets)):
            sd[k] = fold(sd[k])
            records.append({"name": k, "axis": -1, "role": "fold-before-matmul"})
    save_file(sd, out / sh.name, metadata={"format": "pt"})
transform = {"name": "normalized-signed-sylvester-walsh-hadamard", "block_size": a.block, "sign_mode": a.sign_mode}
manifest = {"schema_version": 1, "kind": "hadamard-weight-fold", "status": "requires-matching-runtime",
            "transform": transform, "tensors": records, "tied_output": False}
if a.sign_mode == "explicit":
    manifest["signs"] = {str(k): [int(v) for v in s.tolist()] for k, s in signs.items()}
json.dump(manifest, open(out / "hadamard_packing.json", "w"))
print(f"folded {len(records)} weights ({','.join(targets)}), block {a.block}, sign mode {a.sign_mode}, "
      f"sign widths {sorted(signs)}")
