"""sanity: next-token loss of moe.MoETorso with the tied LM head on real text (granite: logits / logits_scaling)."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe as M, transformers
name = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
g = M.load_cfg(name); cfg = json.load(open(g.path + "/config.json"))
tok = transformers.AutoTokenizer.from_pretrained(M.MODELS[name])
txt = open(os.path.expanduser("~/work/sd/README.md")).read()
ids = tok(txt, return_tensors="pt")["input_ids"][:, :T].cuda()
W = M.load_weights(g); t = M.MoETorso(g, W, lora=False).eval()
with torch.no_grad():
    h = t(ids).last_hidden_state[0]
    lg = (h @ W["embed"].t()).float() / cfg.get("logits_scaling", 1.0)
    loss = torch.nn.functional.cross_entropy(lg[:-1], ids[0, 1:]).item()
cnt = torch.stack(t.counts).float()
print(json.dumps(dict(model=name, T=ids.shape[1], nll=round(loss, 3), load_max_over_mean=round((cnt.max(1).values / cnt.mean(1)).mean().item(), 2))))
