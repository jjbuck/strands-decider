"""exactness of option-order invariance: batch-1 fp32 forward of canonical vs rotated renderings of the same kit questions.
  python invcheck.py runs/S_slot [n]  -> prints max/mean |dp| and decision-unchanged rate"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as Mo, train as T
run = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
meta = json.load(open(f"{run}/meta.json")); vmap = np.load(f"{run}/vmap.npy")
m = Mo.build(meta["arm"], meta["V"], meta["d"], meta["L"], R=meta.get("R", 3)).cuda().float()
m.load_state_dict(torch.load(f"{run}/final.pt", weights_only=False)["model"], strict=os.environ.get("J11_NOBIND") != "1"); m.eval()
kit = T.load_rows("kit"); by = {}
for r in kit:
    by.setdefault((r["meta"]["iid"], r["meta"]["q"]), {})[r["meta"]["rot"]] = r
keys = [k for k, v in by.items() if None in v and len(v) > 1 and len(by[k][None]["ids"]) < 3000][:n]
dps = []; same = []
with torch.no_grad():
    for k in keys:
        out = {}
        for tag, r in by[k].items():
            b = T.collate(meta["arm"], [r], vmap, "cuda"); lg = m(b)[-1]
            p = torch.softmax(lg.float()[0, :r["n"]], -1).tolist(); out[tag] = dict(zip(r["meta"]["labels"], p))
        c = out[None]; ca = max(c, key=c.get)
        for tag, pr in out.items():
            if tag is None: continue
            dps.append(max(abs(pr[l] - c[l]) for l in c)); same.append(max(pr, key=pr.get) == ca)
print(json.dumps(dict(run=run, n_pairs=len(dps), max_abs_dp=float(np.max(dps)), mean_abs_dp=float(np.mean(dps)), unchanged=float(np.mean(same)))))
