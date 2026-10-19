"""Answer parity of a candidate run against a reference run (parity_bench.py outputs).

  python compare.py ref.json cand.json   -> argmax agreement, TV mean / max (0.5 sum |p - p_ref|) per set
"""
import json
import sys


def dist(ans):
    if "noul" in ans and ans.get("probabilities") is None:
        return {"true": ans["noul"], "false": 1 - ans["noul"]}
    return ans["probabilities"]


ref = {r["id"]: r for r in json.load(open(sys.argv[1]))["rows"] if "error" not in r}
cand = json.load(open(sys.argv[2]))["rows"]
by = {}
for c in cand:
    r = ref.get(c["id"])
    if r is None or "error" in c:
        continue
    p, q = dist(r["answer"]), dist(c["answer"])
    tv = 0.5 * sum(abs(p[k] - q.get(k, 0.0)) for k in p)
    agree = max(p, key=p.get) == max(q, key=q.get)
    by.setdefault(c["set"], []).append((agree, tv))
allv = [x for v in by.values() for x in v]
for k, v in list(by.items()) + [("ALL", allv)]:
    print(f"{k:8s} n={len(v):3d} argmax {sum(a for a, _ in v)}/{len(v)}  TV mean {sum(t for _, t in v)/len(v):.4f}"
          f"  max {max(t for _, t in v):.4f}")
