"""Send trade-study rows to the Lambda decider one at a time and record client and engine latency plus answers.

  python drive.py --fn strands-decider-2b --sets mixM1:60,mixM2:59,len256:20,len1k:20 --out lam_std.json
Rows are streamed from s3://your-bucket/requests/ (nothing is written to local disk but the output JSON).
Each request is one question, as the trade study sends them. 'client_s' is the synchronous Invoke round trip from
this machine; 'engine_s' is measured inside the function (render + tokenise + forward + head).
"""
import argparse
import gzip
import io
import json
import os
import time

import boto3
from botocore.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--fn", required=True)
ap.add_argument("--sets", default="mixM1:60,mixM2:59,len256:20,len1k:20")
ap.add_argument("--out", required=True)
ap.add_argument("--region", default="us-east-2")
ap.add_argument("--conc", type=int, default=1, help="concurrent requests (each row still one request)")
a = ap.parse_args()

s3 = boto3.client("s3", region_name=a.region)
lam = boto3.client("lambda", region_name=a.region, config=Config(read_timeout=300, retries={"max_attempts": 0},
                                                                  max_pool_connections=max(10, a.conc)))


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
    body = s3.get_object(Bucket=os.environ.get("DECIDER_BUCKET", "your-bucket"), Key=f"requests/{name}.jsonl.gz")["Body"].read()
    for i, line in enumerate(gzip.open(io.BytesIO(body), "rt")):
        if i >= int(n):
            break
        r = json.loads(line)
        rows.append((name, r["id"], {"state": r["state"], "questions": {"q": system_one_question(r["question"])}}))

out = {"fn": a.fn, "conc": a.conc, "rows": []}


def one(item):
    name, rid, req = item
    throttles = 0
    while True:  # Managed Instances throttles (TooManyRequests) when no environment has a free slot
        t = time.perf_counter()
        try:
            resp = lam.invoke(FunctionName=a.fn, Payload=json.dumps(req).encode())
            break
        except lam.exceptions.TooManyRequestsException:
            throttles += 1
            time.sleep(min(2.0, 0.2 * throttles))
    dt = time.perf_counter() - t
    p = json.loads(resp["Payload"].read())
    rec = {"id": rid, "set": name, "client_s": dt, "throttles": throttles}
    if "answers" in p:
        rec.update(answer=p["answers"]["q"], tokens=p["usage"]["input_tokens"], **p["timing"])
    else:
        rec["error"] = str(p)[:300]
    return rec


from concurrent.futures import ThreadPoolExecutor  # noqa: E402
with ThreadPoolExecutor(a.conc) as ex:
    for rec in ex.map(one, rows):
        out["rows"].append(rec)
        print(rec["set"], rec["id"][:10], f"{rec['client_s']*1e3:.0f} ms", f"engine {rec.get('engine_s', 0)*1e3:.0f} ms",
              rec.get("cpu", ""), rec.get("error", "")[:120], flush=True)
json.dump(out, open(a.out, "w"))
