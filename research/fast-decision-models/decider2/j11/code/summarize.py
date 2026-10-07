"""J11 laptop summary: learning curves (synthetic acc vs training PF) per run, final table, latency fits + projections.
  python3 summarize.py ~/decider2/j11/runs ~/decider2/j11/lat_*.json
"""
import sys, os, json, glob
import numpy as np

RATIO = {"A10G": 1.0, "3090": 52.7 / 27.4, "4090": 52.7 / 13.7, "5090": 52.7 / 10.0}  # compute-term ratios used in FAST_DECISION_MODEL sec. 9


def curves(rd):
    out = {}
    for d in sorted(glob.glob(f"{rd}/*/log.jsonl")):
        name = os.path.basename(os.path.dirname(d)); ev = []
        for l in open(d):
            j = json.loads(l)
            if "eval" in j: e = j["eval"]; ev.append((e["pf"], e["mean_iid"], e["mean_held"], e["synth"]))
        out[name] = ev
    return out


def latfit(files):
    res = {}
    for f in files:
        for r in json.load(open(f)):
            if r["ms"] is None: continue
            res.setdefault((r["arm"], r["d"], r["L"], r["Q"]), []).append((r["T"], r["gflops"], r["ms"], r["p95"]))
    fits = {}
    for k, v in res.items():
        v.sort(); g = np.array([x[1] for x in v]); t = np.array([x[2] for x in v])
        A = np.vstack([np.ones_like(g), g]).T; c, *_ = np.linalg.lstsq(A, t, rcond=None)
        fits[k] = dict(fixed_ms=float(c[0]), ms_per_gflop=float(c[1]), pts=v,
                       proj={gpu: {T: float(c[0] + (gf * c[1]) / r) for T, gf, _, _ in v} for gpu, r in RATIO.items()})
    return fits


if __name__ == "__main__":
    rd = sys.argv[1]
    for name, ev in curves(rd).items():
        print(name, " ".join(f"[{pf:.1f}PF iid {a:.3f} held {h:.3f}]" for pf, a, h, _ in ev))
    fits = latfit(sys.argv[2:])
    for k, f in sorted(fits.items()):
        print(k, f"fixed {f['fixed_ms']:.2f} ms + {f['ms_per_gflop'] * 1000:.2f} us/GFLOP;", " ".join(f"T{T}:{ms:.2f}" for T, _, ms, _ in f["pts"]))
    json.dump({str(k): v for k, v in fits.items()}, open(os.path.join(rd, "..", "latfits.json"), "w"), indent=1)
