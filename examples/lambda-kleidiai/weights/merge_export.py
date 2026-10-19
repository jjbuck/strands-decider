"""Fold the Strands 2B LoRA into its Qwen3.5 base and write an HF checkpoint that llama.cpp's converter reads.

  python merge_export.py [--ckpt StrandsAgents/strands-decider-2B-qwen3.5-v1-2610] --out /opt/lam/merged-2b

W' = W + (alpha / r) B A, formed in fp32 and stored in fp32 (the converter's --outtype then rounds once). Every
other tensor, the config and the tokenizer are the base checkpoint's, at the revision the Strands loader pins.
Writes merge_report.json: merged count, and the adapter / base keys that did not pair (must be empty).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from strands_decider.modeling import StrandsDeciderConfig, base_revision, checkpoint_dir, config_path

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ck = checkpoint_dir(a.ckpt)
cfg = StrandsDeciderConfig.from_json(config_path(ck))
rev = base_revision(ck, cfg)
base = Path(checkpoint_dir(cfg.base_model, rev))
print("checkpoint", ck, "\nbase", cfg.base_model, rev, base)
acfg = json.load(open(os.path.join(ck, "lora", "adapter_config.json")))
assert not acfg.get("use_dora") and not acfg.get("modules_to_save") and not acfg.get("rank_pattern"), acfg
scale = acfg["lora_alpha"] / (acfg["r"] ** 0.5 if acfg.get("use_rslora") else acfg["r"])
ad = load_file(os.path.join(ck, "lora", "adapter_model.safetensors"))
stems = {k.removesuffix(".lora_A.weight") for k in ad if k.endswith(".lora_A.weight")}
assert set(ad) == {s + x for s in stems for x in (".lora_A.weight", ".lora_B.weight")}, sorted(ad)[:5]

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for f in base.iterdir():
    if not f.name.endswith(".safetensors"):
        shutil.copy(f, out / f.name)
shards = sorted(base.glob("*.safetensors"))
want = {s.removeprefix("base_model.model."): s for s in stems}  # e.g. layers.3.linear_attn.in_proj_qkv
merged, seen = 0, set()
for sh in shards:
    sd = load_file(sh)
    for k in list(sd):
        if not k.endswith(".weight"):
            continue
        stem = k.removesuffix(".weight")
        # text-decoder keys only: model.language_model.<path> (multimodal) or model.<path>
        for pre in ("model.language_model.", "model."):
            if stem.startswith(pre) and stem[len(pre):] in want:
                s = want[stem[len(pre):]]
                A = ad[s + ".lora_A.weight"].float()
                B = ad[s + ".lora_B.weight"].float()
                sd[k] = (sd[k].float() + scale * (B @ A)).contiguous()
                merged += 1
                seen.add(s)
                break
    save_file(sd, out / sh.name, metadata={"format": "pt"})
rep = {"merged": merged, "scale": scale, "unpaired_adapter": sorted(stems - seen), "base_revision": rev,
       "checkpoint": str(ck)}
json.dump(rep, open(out / "merge_report.json", "w"), indent=1)
print(json.dumps({k: (v if k != "unpaired_adapter" else v[:5]) for k, v in rep.items()}))
assert not rep["unpaired_adapter"]
