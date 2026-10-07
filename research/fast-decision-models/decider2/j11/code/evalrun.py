"""J11 evaluation of one trained run: evalkit (canonical + option rotations), synthetic test (iid + held-out families, + rotations), lceval.
  python evalrun.py runs/S_dec  -> runs/S_dec/{preds_kit.json, eval.json}
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as Mo, train as T


def ece(conf, corr, nb=15):
    conf = np.asarray(conf); corr = np.asarray(corr, float); e = 0.0
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / nb)
        if m.any(): e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(e)


def invariance(rows, preds, key):
    """decisions unchanged across all option rotations, and mean |dp| (aligned by label), vs the canonical rendering."""
    g = {}
    for r, p in zip(rows, preds):
        if p is None: continue
        g.setdefault(key(r), {})[r["meta"]["rot"]] = dict(zip(r["meta"]["labels"], p))
    same = []; dps = []
    for k, d in g.items():
        if None not in d or len(d) < 2: continue
        c = d[None]; ca = max(c, key=c.get)
        for tag, pr in d.items():
            if tag is None: continue
            dps.append(np.mean([abs(pr[l] - c[l]) for l in c]))
        same.append(all(max(pr, key=pr.get) == ca for pr in d.values()))
    return dict(n=len(same), unchanged=float(np.mean(same)) if same else None, mean_abs_dp=float(np.mean(dps)) if dps else None)


def main():
    run = sys.argv[1]; dev = "cuda"
    meta = json.load(open(f"{run}/meta.json")); vmap = np.load(f"{run}/vmap.npy")
    model = Mo.build(meta["arm"], meta["V"], meta["d"], meta["L"], R=meta.get("R", 3)).to(dev)
    strict = os.environ.get("J11_NOBIND") != "1"  # v1 vslot (trained before the binding priors existed): no b_id/b_rec, no slot_h
    model.load_state_dict(torch.load(f"{run}/final.pt", weights_only=False)["model"], strict=strict); model.eval()
    arm = meta["arm"]; out = {}; t0 = time.time()
    # synthetic
    st = T.load_rows("synth_test"); ps = T.predict(model, arm, st, vmap, dev)
    can = [(r, p) for r, p in zip(st, ps) if r["meta"]["rot"] is None and p is not None]
    out["synth"] = T.synth_acc([p for _, p in can], [r for r, _ in can])
    out["synth_iid"] = float(np.mean([v for k, v in out["synth"].items() if not k.startswith("H_")]))
    out["synth_held"] = float(np.mean([v for k, v in out["synth"].items() if k.startswith("H_")]))
    conf = [max(p) for _, p in can]; corr = [int(np.argmax(p) == r["label"]) for r, p in can]
    out["synth_ece"] = ece(conf, corr)
    out["synth_brier"] = float(np.mean([sum((pi - (1.0 if i == r["label"] else 0.0)) ** 2 for i, pi in enumerate(p)) for r, p in can]))
    out["synth_inv"] = invariance(st, ps, lambda r: r["meta"]["id"])
    print("synth", json.dumps(out["synth"]), out["synth_iid"], out["synth_held"], f"{time.time() - t0:.0f}s", flush=True)
    # lceval (corpus val rows with hobson teacher + gold)
    lc = T.load_rows("lceval"); pl = T.predict(model, arm, lc, vmap, dev)
    ok = [(r, p) for r, p in zip(lc, pl) if p is not None]
    out["lceval_gold"] = float(np.mean([np.argmax(p) == r["label"] for r, p in ok]))
    out["lceval_agree_hobson"] = float(np.mean([np.argmax(p) == int(np.argmax(r["t"])) for r, p in ok if r.get("t") is not None]))
    # evalkit
    kit = T.load_rows("kit"); pk = T.predict(model, arm, kit, vmap, dev, tok_budget=12000)
    preds = {}
    for r, p in zip(kit, pk):
        if p is None or r["meta"]["rot"] is not None: continue
        preds.setdefault(r["meta"]["iid"], {})[r["meta"]["q"]] = dict(zip(r["meta"]["labels"], p))
    json.dump(preds, open(f"{run}/preds_kit.json", "w"))
    rot = {}
    for r, p in zip(kit, pk):
        if p is None or r["meta"]["rot"] is None: continue
        rot.setdefault(r["meta"]["rot"], {}).setdefault(r["meta"]["iid"], {})[r["meta"]["q"]] = dict(zip(r["meta"]["labels"], p))
    json.dump(rot, open(f"{run}/preds_kit_rot.json", "w"))
    out["kit_inv"] = invariance(kit, pk, lambda r: (r["meta"]["iid"], r["meta"]["q"]))
    out["eval_s"] = time.time() - t0
    json.dump(out, open(f"{run}/eval.json", "w"), indent=1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
