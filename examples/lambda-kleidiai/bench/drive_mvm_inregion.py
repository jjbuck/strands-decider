"""Drive a running MicroVM from the in-region client Lambda (strands-decider-mvm-client), in batches, and save in drive.py's
format. The auth token is minted here (60 min) and passed in the invoke payload only; it is not logged.

  python drive_mvm_inregion.py --microvm microvm-... --sets mixM1:60,mixM2:59,len256:20,len1k:20 --out mvm_inregion.json
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
ap.add_argument("--microvm", required=True)
ap.add_argument("--sets", default="mixM1:60,mixM2:59,len256:20,len1k:20")
ap.add_argument("--batch", type=int, default=30)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = "us-east-2"
mvm = boto3.client("lambda-microvms", region_name=R)
lam = boto3.client("lambda", region_name=R, config=Config(read_timeout=900, retries={"max_attempts": 0}))
s3 = boto3.client("s3", region_name=R)
host = mvm.get_microvm(microvmIdentifier=a.microvm)["endpoint"]


def token():  # minted per batch, so runs longer than the token's 60 min keep working
    return mvm.create_microvm_auth_token(microvmIdentifier=a.microvm, expirationInMinutes=60,
                                         allowedPorts=[{"allPorts": {}}])["authToken"]


def soq(q):
    out = {"type": q["type"], "instructions": q.get("instructions")}
    if q["type"] == "noul":
        c = {k: v for k, v in (q.get("criteria") or {}).items() if k in ("true", "false") and v not in (None, "")}
        if c:
            out["criteria"] = c
    else:
        out["criteria"] = q["criteria"]
    return out


reqs = []
for spec in a.sets.split(","):
    name, n = spec.split(":")
    body = s3.get_object(Bucket=os.environ.get("DECIDER_BUCKET", "your-bucket"), Key=f"requests/{name}.jsonl.gz")["Body"].read()
    for i, line in enumerate(gzip.open(io.BytesIO(body), "rt")):
        if i >= int(n):
            break
        r = json.loads(line)
        reqs.append({"id": r["id"], "set": name, "body": {"state": r["state"], "questions": {"q": soq(r["question"])}}})
rows = []
for i in range(0, len(reqs), a.batch):
    t0 = time.time()
    p = lam.invoke(FunctionName="strands-decider-mvm-client",
                   Payload=json.dumps({"host": host, "auth": token(), "requests": reqs[i:i + a.batch]}).encode())
    res = json.loads(p["Payload"].read())
    if isinstance(res, dict):
        raise SystemExit(str(res)[:500])
    for r in res:
        r["batch_t0"] = t0
    rows += res
    json.dump({"microvm": a.microvm, "caller": "lambda us-east-2, keep-alive", "rows": rows}, open(a.out, "w"))
    print(f"{len(rows)}/{len(reqs)}", flush=True)
print("rows", len(rows), "errors", sum("error" in r for r in rows))
