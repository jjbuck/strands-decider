"""Latency and answers of one Strands 2B engine on trade-study rows, for parity and speed comparisons.

  python parity_bench.py --engine torch [--threads 6] --out ref.json
  python parity_bench.py --engine lcpp --gguf /opt/lam/gguf/s2b-Q8_0.gguf --threads 6 --out q8.json
  --sets mixM1:60,mixM2:60,len256:20,len1k:20,len4k:5   (first N rows of s3://your-bucket/requests/<set>)

Each row is one request with one question, as the trade study sends them (adapter.build_http). Records per row:
id, set, input tokens, wall seconds of engine.evaluate (render + tokenise + forward + head), and the answer.
Two warm-up requests run first and are not recorded.
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["torch", "lcpp"], required=True)
ap.add_argument("--ckpt", default="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610")
ap.add_argument("--gguf")
ap.add_argument("--lib", default="/opt/lam/libdecshim.so")
ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--sets", default="mixM1:60,mixM2:60,len256:20,len1k:20")
ap.add_argument("--req-dir", default="/opt/lam/req")
ap.add_argument("--torch-mode", default="fp32", choices=["fp32", "bf16", "fp32-bf16math"],
                help="fp32: the engine as released on CPU; bf16: keep the bf16 torso (oneDNN AMX/BFMMLA kernels); "
                     "fp32-bf16math: fp32 weights, matmuls in bf16 (oneDNN implicit down-conversion)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, "/opt/lam/engine")
import torch  # noqa: E402

from strands_decider.schema import SystemOneRequest  # noqa: E402

torch.set_num_threads(a.threads)


def system_one_question(q):  # trade/runtimes/strands/adapter.py
    out = {"type": q["type"], "instructions": q.get("instructions")}
    if q["type"] == "noul":
        c = {k: v for k, v in (q.get("criteria") or {}).items() if k in ("true", "false") and v not in (None, "")}
        if c:
            out["criteria"] = c
    else:
        out["criteria"] = q["criteria"]
    return out


rows = []
for spec in a.sets.split(","):
    name, n = spec.split(":")
    p = Path(a.req_dir) / f"{name}.jsonl.gz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"aws s3 cp --quiet s3://your-bucket/requests/{name}.jsonl.gz {p}")
    with gzip.open(p, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= int(n):
                break
            r = json.loads(line)
            r["_set"] = name
            rows.append(r)

t0 = time.time()
if a.engine == "torch":
    from strands_decider import infer
    if a.torch_mode == "bf16":
        infer.SystemOneEngine._upcast_torso_for_cpu = lambda self: None
    if a.torch_mode == "fp32-bf16math":
        torch.set_float32_matmul_precision("medium")
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
    eng = infer.load_engine(a.ckpt, device="cpu")
else:
    from lcpp_engine import load_llamacpp_engine
    eng = load_llamacpp_engine(a.ckpt, a.gguf, a.lib, a.threads)
load_s = time.time() - t0
print(f"loaded {a.engine} in {load_s:.1f}s", flush=True)


def req(r):
    return SystemOneRequest(state=r["state"], questions={"q": system_one_question(r["question"])})


for r in rows[:4]:
    try:
        eng.evaluate(req(r))
    except ValueError:
        pass
out = {"engine": a.engine, "torch_mode": a.torch_mode, "gguf": a.gguf, "threads": a.threads, "load_s": load_s,
       "env": {k: v for k, v in os.environ.items() if k.startswith(("ONEDNN", "DNNL", "OMP", "GGML", "DEC_"))}, "rows": []}
for r in rows:
    q = req(r)
    t = time.perf_counter()
    try:
        resp = eng.evaluate(q)
    except ValueError as e:  # the engine refuses e.g. a prompt truncated through its option list, as it does served
        out["rows"].append({"id": r["id"], "set": r["_set"], "error": str(e)[:200]})
        continue
    dt = time.perf_counter() - t
    ans = resp.answers["q"].model_dump()
    out["rows"].append({"id": r["id"], "set": r["_set"], "tokens": resp.usage.input_tokens, "s": dt, "answer": ans})
json.dump(out, open(a.out, "w"))
print("refused:", sum("error" in x for x in out["rows"]), flush=True)
by = {}
for x in out["rows"]:
    if "error" in x:
        continue
    by.setdefault(x["set"], []).append(x)
for k, v in by.items():
    s = sorted(x["s"] for x in v)
    tk = sorted(x["tokens"] for x in v)
    print(f"{k:8s} n={len(v):3d} tokens p50={tk[len(tk)//2]:5d}  latency p50={s[len(s)//2]*1e3:7.0f} ms  "
          f"p95={s[min(len(s)-1, int(len(s)*0.95))]*1e3:7.0f} ms  max={s[-1]*1e3:7.0f} ms", flush=True)
