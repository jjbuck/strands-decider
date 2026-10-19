"""Phase 2 on Lambda, measured as PR #77 measured: 10 concurrent requests at a freshly published version (each one a new
execution environment), then 10 warm requests one at a time, then 10 warm requests concurrently.

  python snap_bench.py --fn strands-decider-2b-snap --qualifier 3 --rows len256_10.json --tag on-r1 --out on-r1.json

Each request is one trade len256 row as a direct invocation (the event is the System One request). Records per
request: client round trip; the function's own latency_ms (engine.evaluate); from the log tail (LogType=Tail): the
host's CPU part from the handler's log line and the REPORT fields (Duration, Init Duration or Restore Duration, Max
Memory Used); FunctionError and its message if any.
"""
import argparse
import base64
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--fn", required=True)
ap.add_argument("--qualifier", required=True)
ap.add_argument("--rows", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--region", default="us-east-2")
a = ap.parse_args()

lam = boto3.client("lambda", region_name=a.region, config=Config(read_timeout=900, connect_timeout=10,
                                                                 retries={"max_attempts": 0}, max_pool_connections=32))
rows = json.load(open(a.rows))


def report(log):
    out = {}
    for line in log.splitlines():
        if line.startswith("REPORT"):
            for part in line.split("\t")[1:]:
                if ":" in part:
                    k, v = part.split(":", 1)
                    m = re.match(r"\s*([\d.]+)", v)
                    if m:
                        out[k.strip()] = float(m.group(1))
    return out


def invoke(i, phase):
    req = rows[i % len(rows)]["request"]
    t0 = time.time()
    t = time.perf_counter()
    r = lam.invoke(FunctionName=a.fn, Qualifier=a.qualifier, Payload=json.dumps(req).encode(), LogType="Tail")
    body = r["Payload"].read()
    rt = time.perf_counter() - t
    log = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
    rec = {"phase": phase, "i": i, "row": rows[i % len(rows)]["id"], "t_start": t0, "rt_s": rt,
           "status": r["StatusCode"], "error": r.get("FunctionError"), "report": report(log)}
    m = re.findall(r"cpu_part=(\S+)", log)
    rec["cpu_part"] = m[-1] if m else None
    try:
        p = json.loads(body)
    except ValueError:
        p = {"raw": body[:400].decode(errors="replace")}
    if rec["error"]:
        rec["payload"] = p
        rec["log_tail"] = log[-1500:]
    else:
        rec.update(latency_ms=p.get("latency_ms"), tokens=p["usage"]["input_tokens"], answer=p["answers"]["q"])
    return rec


def summary(recs, name):
    ok = [r for r in recs if not r["error"]]
    rt = [r["rt_s"] for r in ok]
    lat = [r["latency_ms"] / 1e3 for r in ok]
    dur = [r["report"].get("Duration", 0) / 1e3 for r in ok]
    init = [r["report"].get("Restore Duration", r["report"].get("Init Duration", 0)) / 1e3 for r in ok]
    cpus = {}
    for r in recs:
        cpus[r["cpu_part"]] = cpus.get(r["cpu_part"], 0) + 1
    s = (f"{a.tag} {name}: ok {len(ok)}/{len(recs)}  round trip median {statistics.median(rt):.2f} s "
         f"(min {min(rt):.2f}, max {max(rt):.2f})  in-function engine median {statistics.median(lat):.3f} s  "
         f"REPORT Duration median {statistics.median(dur):.2f} s  init/restore median {statistics.median(init):.2f} s  "
         f"cpu parts {cpus}") if ok else f"{a.tag} {name}: ok 0/{len(recs)} {[r.get('payload') for r in recs][:2]}"
    print(s, flush=True)
    return s


out = {"fn": a.fn, "qualifier": a.qualifier, "tag": a.tag, "rows": [r["id"] for r in rows], "requests": [],
       "summary": []}
with ThreadPoolExecutor(10) as ex:
    cold = list(ex.map(lambda i: invoke(i, "cold"), range(10)))
out["requests"] += cold
out["summary"].append(summary(cold, "cold x10 concurrent"))
seq = [invoke(i, "warm-seq") for i in range(10)]
out["requests"] += seq
out["summary"].append(summary(seq, "warm x10 sequential"))
with ThreadPoolExecutor(10) as ex:
    wc = list(ex.map(lambda i: invoke(i, "warm-conc"), range(10)))
out["requests"] += wc
out["summary"].append(summary(wc, "warm x10 concurrent"))
json.dump(out, open(a.out, "w"), indent=1)
