"""quality vs batch-1 latency (1000 tokens) for hobson, F7 dense controls and the G3 MoE decider. usage: python3 plot_g3.py points.json out.png"""
import sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
P = json.load(open(sys.argv[1]))
mets = [("REAL-agree", "agree_sd", "REAL agree_sd (bar 0.95)", 0.95), ("CF", "flip_given_hobson", "CF fgh (bar 0.90)", 0.90),
        ("JB-hard", "acc", "JB-hard accuracy", None), ("LONG", "agree_sd", "LONG agree_sd (bar 0.95)", 0.95)]
fig, axs = plt.subplots(2, 4, figsize=(17, 7.5), sharey="col")
for row, lat_key, xl in ((0, "lat_a10g", "A10G latency @1000 tok, ms (measured)"), (1, "lat_3090", "RTX 3090 latency @1000 tok, ms (projected, fp16-acc)")):
    for j, (su, m, title, bar) in enumerate(mets):
        ax = axs[row, j]
        for p in P:
            y = p.get("scores", {}).get(su, {}).get(m)
            x = p.get(lat_key)
            if y is None or x is None: continue
            moe = p.get("moe", False)
            ax.scatter([x], [y], s=70, marker="D" if moe else "o", color="#c0392b" if moe else ("#555555" if p["name"] == "hobson" else "#2e86c1"), zorder=3)
            ax.annotate(p["name"], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
        if bar: ax.axhline(bar, color="#999999", lw=0.8, ls="--")
        ax.set_title(title, fontsize=10); ax.set_xlabel(xl, fontsize=8); ax.grid(alpha=0.3)
        lim = 0.4 * [p for p in P if p["name"] == "hobson"][0][lat_key]
        ax.axvline(lim, color="#999999", lw=0.8, ls=":")
fig.suptitle("Decision quality vs batch-1 prefill latency (dotted: 0.4x hobson latency; diamonds: MoE)", fontsize=11)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=130)
print("saved", sys.argv[2])
