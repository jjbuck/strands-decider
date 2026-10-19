"""Shared-state fork vs per-question encoding on multi-question suite rows: answer agreement and latency.

  python fork_test.py GGUF THREADS LIB [N_ROWS]
"""
import gzip
import json
import sys
import time
from dataclasses import replace

sys.path.insert(0, "/opt/lam/engine")
from lcpp_engine import load_llamacpp_engine  # noqa: E402
from strands_decider.schema import SystemOneRequest  # noqa: E402

gguf, threads, lib = sys.argv[1], int(sys.argv[2]), sys.argv[3]
n_rows = int(sys.argv[4]) if len(sys.argv) > 4 else 40
eng = load_llamacpp_engine("StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", gguf, lib, threads, n_seq=17)


def soq(q):
    out = {"type": q["type"], "instructions": q.get("instructions")}
    if q["type"] == "noul":
        c = {k: v for k, v in (q.get("criteria") or {}).items() if k in ("true", "false") and v not in (None, "")}
        if c:
            out["criteria"] = c
    else:
        out["criteria"] = q["criteria"]
    return out


rows = []
for suite in ("bev", "di"):
    for line in gzip.open(f"/opt/lam/suites/{suite}.jsonl.gz", "rt"):
        r = json.loads(line)
        if 3 <= len(r["questions"]) <= 32 and len(r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])) > 400:
            rows.append(r)
        if len(rows) >= n_rows * (1 if suite == "bev" else 2):
            break


def dist(a):
    return {"true": a["noul"], "false": 1 - a["noul"]} if a.get("probabilities") is None else a["probabilities"]


worst, agree, n, t_fork, t_sep, nq = 0.0, 0, 0, 0.0, 0.0, 0
for r in rows:
    req = SystemOneRequest(state=r["state"], questions={k: soq(q) for k, q in r["questions"].items()})
    out = {}
    for mode in ("sep", "fork"):
        eng.cfg = replace(eng.cfg, use_prefix_cache=(mode == "fork"))
        try:
            eng.evaluate(req)  # warm this shape
            t = time.perf_counter()
            out[mode] = eng.evaluate(req)
            dt = time.perf_counter() - t
        except ValueError:
            break
        if mode == "fork":
            t_fork += dt
        else:
            t_sep += dt
    if len(out) < 2:
        continue
    nq += len(r["questions"])
    for k in r["questions"]:
        a, b = dist(out["sep"].answers[k].model_dump()), dist(out["fork"].answers[k].model_dump())
        worst = max(worst, max(abs(a[x] - b[x]) for x in a))
        agree += max(a, key=a.get) == max(b, key=b.get)
        n += 1
print(f"rows {len(rows)} questions {n}: argmax agreement {agree}/{n}, max |dp| {worst:.4f}; "
      f"latency per request sep {t_sep/max(1,len(rows))*1e3:.0f} ms vs fork {t_fork/max(1,len(rows))*1e3:.0f} ms "
      f"({t_sep/max(t_fork,1e-9):.2f}x)")
