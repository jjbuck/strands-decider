import sys, os, json, torch, transformers
sys.path.insert(0, os.path.expanduser("~/work/d1")); sys.path.insert(0, os.path.expanduser("~/work/systems/g"))
from prof_d1 import capture, wall
from lean import Lean
from lean2 import Lean2
name = "Qwen/Qwen3.5-0.8B"; cfg = transformers.AutoConfig.from_pretrained(name)
torso = transformers.Qwen3_5ForCausalLM.from_pretrained(name, config=cfg.get_text_config(), dtype=torch.bfloat16).cuda().eval().model
P = sum(p.numel() for n, p in torso.named_parameters() if "embed" not in n)
ref = Lean(torso); ln = Lean2(torso, fuse="")
torch.manual_seed(0); idsv = torch.randint(1000, 100000, (1, 1000), device="cuda")
with torch.inference_mode():
    hr = ref.forward(idsv).float()
    hf = torso(input_ids=idsv, use_cache=False).last_hidden_state.float()
print("lean vs HF rel", round(((hr - hf).norm() / hf.norm()).item(), 5), flush=True)
for s in ["addrms", "addrms,gnorm", "addrms,gnorm,silu", "addrms,gnorm,silu,prep", "addrms,gnorm,silu,prep,conv", "addrms,gnorm,silu,prep,conv,gemm_swiglu", "gnorm,prep,conv,fold"]:
    ln.fuse = set(s.split(","))
    with torch.inference_mode(): h = ln.forward(idsv).float()
    r = {"rel_vs_lean": round(((h - hr).norm() / hr.norm()).item(), 5), "rel_vs_hf": round(((h - hf).norm() / hf.norm()).item(), 5)}
    if s.endswith("fold"):
        for T in (256, 1000, 4000):
            ids = torch.randint(1000, 100000, (1, T), device="cuda"); g, o = capture(lambda: ln.forward(ids)); r[T] = wall(g, ids, T); del g, o
    print(s, json.dumps(r), flush=True)
