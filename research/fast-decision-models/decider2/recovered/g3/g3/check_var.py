import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe as M, transformers
name = "st4b"; T = 1024
g = M.load_cfg(name)
tok = transformers.AutoTokenizer.from_pretrained(M.MODELS[name])
txt = open(os.path.expanduser("~/work/sd/README.md")).read()
ids = tok(txt, return_tensors="pt")["input_ids"][:, :T].cuda()
W = M.load_weights(g)
for gf, pre in (("sigmoid_norm", True), ("softmax", True), ("sigmoid_norm", False)):
    g.gate_fn, g.router_pre = gf, pre
    t = M.MoETorso(g, W, lora=False).eval()
    with torch.no_grad():
        h = t(ids).last_hidden_state[0]; lg = (h @ W["embed"].t()).float()
        print(gf, pre, round(torch.nn.functional.cross_entropy(lg[:-1], ids[0, 1:]).item(), 4), flush=True)
