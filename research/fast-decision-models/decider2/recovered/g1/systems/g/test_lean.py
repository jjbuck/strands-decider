import time, torch, sys
from common import *
from strands_decider.infer import load_engine
from lean import Lean
eng = load_engine(CKPT, device="cuda")
tok = eng.model.tokenizer; mk = make_state_fn(tok)
torso = eng.model.torso.merge_and_unload()
torso.eval()
ids = tok(mk(300, "x"), return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
print(ids.shape)
with torch.no_grad():
    ref = torso(input_ids=ids, use_cache=False).last_hidden_state
ln = Lean(torso)
with torch.no_grad():
    out = ln.forward(ids)
print("ref", ref.float().abs().mean().item(), "diff mean", (out.float()-ref.float()).abs().mean().item(), "max", (out.float()-ref.float()).abs().max().item())
cos = torch.nn.functional.cosine_similarity(out.float().reshape(-1,2048), ref.float().reshape(-1,2048), dim=-1)
print("cos min", cos.min().item(), "mean", cos.mean().item())
