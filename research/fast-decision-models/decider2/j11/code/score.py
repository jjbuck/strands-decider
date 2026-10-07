"""J11 laptop-side scoring (pure python + numpy): evalkit suites, ECE/Brier against ground truth, McNemar vs hobson, order invariance.
  python3 score.py RUN_DIR [RUN_DIR ...]  -> RUN_DIR/score.json and a compact table on stdout
"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.expanduser("~/decider2/evalkit"))
import evalkit as EK

GT_SUITES = ["JB-all", "JB-hard", "CF", "CF-probe", "REAL-label"]


def ece(conf, corr, nb=15):
    conf = np.asarray(conf); corr = np.asarray(corr, float); e = 0.0
    for lo in np.linspace(0, 1, nb + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / nb)
        if m.any(): e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(e)


def gt_metrics(suite, preds):
    rows = [r for r in EK._qrows(suite, preds) if r["exp"] is not None]
    out = {}
    for who in ("pred", "hob"):
        rr = [r for r in rows if r[who] is not None]
        if not rr: continue
        conf = [max(r[who].values()) for r in rr]; corr = [EK._arg(r[who]) == r["exp"] for r in rr]
        brier = [sum((v - (1.0 if k == r["exp"] else 0.0)) ** 2 for k, v in r[who].items()) for r in rr]
        out[who] = dict(n=len(rr), acc=float(np.mean(corr)), ece=ece(conf, corr), brier=float(np.mean(brier)))
    # McNemar (exact binomial) model vs hobson, paired on questions both cover
    pr = [r for r in rows if r["pred"] is not None and r["hob"] is not None]
    b = sum(EK._arg(r["pred"]) == r["exp"] and EK._arg(r["hob"]) != r["exp"] for r in pr)
    c = sum(EK._arg(r["pred"]) != r["exp"] and EK._arg(r["hob"]) == r["exp"] for r in pr)
    n = b + c; k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    out["mcnemar"] = dict(model_only=b, hobson_only=c, p=p)
    return out


def invariance(rot, base, margin=0.1):
    """decision unchanged under every rotation; mean |dp|; and unchanged among canonical decisions with top-2 margin > `margin`
    (near-tied outputs flip on bf16 batch-padding noise alone, so the confident subset is the meaningful one)."""
    same = []; dps = []; conf = []
    for iid, qs in base.items():
        for q, c in qs.items():
            rs = [rot[t][iid][q] for t in rot if iid in rot[t] and q in rot[t][iid]]
            if not rs: continue
            ca = max(c, key=c.get); v = sorted(c.values(), reverse=True)
            ok = all(max(r, key=r.get) == ca for r in rs); same.append(ok)
            if len(v) > 1 and v[0] - v[1] > margin: conf.append(ok)
            dps += [np.mean([abs(r[l] - c[l]) for l in c]) for r in rs]
    return dict(n=len(same), unchanged=float(np.mean(same)), mean_abs_dp=float(np.mean(dps)),
                n_conf=len(conf), unchanged_conf=float(np.mean(conf)) if conf else None)


def score(run):
    preds = json.load(open(f"{run}/preds_kit.json"))
    out = {"suites": {}, "gt": {}}
    for s in ["JB-all", "JB-hard", "REAL-agree", "LONG", "CF", "CF-probe", "SHUF", "REAL-label"]:
        sc = EK.score(s, preds); out["suites"][s] = sc.get("model", sc)
    for s in GT_SUITES: out["gt"][s] = gt_metrics(s, preds)
    if os.path.exists(f"{run}/preds_kit_rot.json"): out["kit_inv"] = invariance(json.load(open(f"{run}/preds_kit_rot.json")), preds)
    if os.path.exists(f"{run}/eval.json"): out["eval"] = json.load(open(f"{run}/eval.json"))
    json.dump(out, open(f"{run}/score.json", "w"), indent=1)
    return out


def row(run, o):
    S = o["suites"]; G = o["gt"]; e = o.get("eval", {})
    g = lambda s, k: S.get(s, {}).get(k, float("nan"))
    return (f"{os.path.basename(run):14s} JBall {g('JB-all', 'acc'):.3f} JBhard {g('JB-hard', 'acc'):.3f} REAL agree_sd {g('REAL-agree', 'agree_sd'):.3f} "
            f"LONG agree_sd {g('LONG', 'agree_sd'):.3f} CF acc/flip {g('CF', 'acc'):.3f}/{g('CF', 'flip'):.3f} probe acc/flip {g('CF-probe', 'acc'):.3f}/{g('CF-probe', 'flip'):.3f} "
            f"RL {g('REAL-label', 'acc'):.3f} | synth iid {e.get('synth_iid', float('nan')):.3f} held {e.get('synth_held', float('nan')):.3f} "
            f"| inv {o.get('kit_inv', {}).get('unchanged', float('nan')):.3f} ECE(CF) {G['CF']['pred']['ece']:.3f}")


if __name__ == "__main__":
    for run in sys.argv[1:]:
        o = score(run); print(row(run, o))
