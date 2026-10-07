import json, os, glob
J = os.path.expanduser("~/decider2/j11"); Ts = [64, 128, 256, 400, 1000, 2000, 4000, 8000]
rows = {}
for f in sorted(glob.glob(f"{J}/lat_[SMX]*.json")):
    if "proj" in f: continue
    for r in json.load(open(f)): rows.setdefault((r["d"], r["L"], r["arm"]), {})[(r["Q"], r["T"])] = r
print("| model | " + " | ".join(str(t) for t in Ts) + " | GF@1000 |"); print("|" + "---|" * (len(Ts) + 2))
for (d, L, arm), v in sorted(rows.items(), key=lambda x: (x[0][0], ["dec", "slot", "vslot", "belief"].index(x[0][2]))):
    cells = []
    for t in Ts:
        a = v.get((1, t)); b = v.get((4, t))
        cells.append(f"{a['ms']:.2f} ({b['ms']:.2f})" if a and b and a["ms"] and b["ms"] else "-")
    print(f"| {arm} d{d}/L{L} | " + " | ".join(cells) + f" | {v[(1, 1000)]['gflops']:.0f} |")
