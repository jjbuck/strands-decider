"""Shared loader: merged torso (+ trained LoRA merged) and pointer head for every F7 arm.
arch 'q08' = Qwen3.5-0.8B-Base (first `layers` layers if layers>0), 'trunc' / 'hob' = hobson-v19 merged (first `layers` layers; 0 = all 24)."""
import os, json, glob, torch, transformers
HOB = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*"))[0]


def cut(torso, layers):
    if layers and layers < len(torso.layers):
        torso.layers = torch.nn.ModuleList(list(torso.layers)[:layers]); torso.config.num_hidden_layers = layers
        torso.config.layer_types = list(torso.config.layer_types)[:layers]
    return torso


def base_torso(arch, layers=0):
    if arch == "q08":
        cfg = transformers.AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
        t = transformers.Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", config=cfg.get_text_config(), dtype=torch.bfloat16).model
        return cut(t, layers), None
    import sys; sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
    from strands_decider.modeling import StrandsDeciderModel
    hob = StrandsDeciderModel.load(HOB)
    return cut(hob.torso.merge_and_unload(), layers), hob


def load_ckpt(ck):
    """-> (merged torso, head, config, meta) for a checkpoint saved by train.py"""
    import sys; sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
    from strands_decider.modeling import StrandsDeciderConfig, build_head
    from peft import PeftModel
    meta = json.load(open(os.path.join(ck, "f7_meta.json")))
    arch = meta["arch"]; layers = int(meta.get("layers") or 0)
    torso, _ = base_torso("q08" if arch == "q08" else "hob", layers)
    torso = PeftModel.from_pretrained(torso, os.path.join(ck, "lora")).merge_and_unload()
    c = StrandsDeciderConfig.from_json(os.path.join(ck, "strands_decider_config.json"))
    head = build_head(c, torso.config.hidden_size); head.load_state_dict(torch.load(os.path.join(ck, "slot_head.pt")))
    return torso, head, c, meta
