"""Cold start on standard Lambda: set an env change (fresh sandboxes), then N concurrent first invocations.

  python cold_test.py --fn strands-decider-2b --env KEY=VAL,... --n 10 --out cold_<tag>.json
Records each first invocation's client wall time (cold start + one decision) and a second wave (warm).
"""
import argparse
import concurrent.futures as cf
import json
import os
import time

import boto3
from botocore.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--fn", default="strands-decider-2b")
ap.add_argument("--env", default="", help="KEY=VAL,... merged into the function's current environment")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--out", required=True)
ap.add_argument("--request", default="", help="trade set whose first row is the request (default: the one-line ticket)")
a = ap.parse_args()
lam = boto3.client("lambda", region_name="us-east-2", config=Config(read_timeout=900, max_pool_connections=a.n,
                                                                     retries={"max_attempts": 0}))
env = lam.get_function_configuration(FunctionName=a.fn).get("Environment", {}).get("Variables", {})
env.update(dict(kv.split("=", 1) for kv in a.env.split(",") if kv))
env["COLD_TAG"] = str(time.time())
lam.update_function_configuration(FunctionName=a.fn, Environment={"Variables": env})
lam.get_waiter("function_updated_v2").wait(FunctionName=a.fn)
REQ = {"state": {"ticket": "Refund issued but invoice still shows the old total."},
       "questions": {"q": {"type": "choice", "instructions": "Route this ticket.",
                           "criteria": {"billing": "payment or invoice", "technical": "product defect", "account": "login"}}}}


def trade_request(name):  # first row of a trade-study set, as a System One request (the drivers' conversion)
    import gzip, io
    r = json.loads(gzip.open(io.BytesIO(boto3.client("s3", region_name="us-east-2").get_object(
        Bucket=os.environ.get("DECIDER_BUCKET", "your-bucket"), Key=f"requests/{name}.jsonl.gz")["Body"].read()), "rt").readline())
    q = r["question"]
    out = {"type": q["type"], "instructions": q.get("instructions")}
    if q["type"] == "noul":
        c = {k: v for k, v in (q.get("criteria") or {}).items() if k in ("true", "false") and v not in (None, "")}
        if c:
            out["criteria"] = c
    else:
        out["criteria"] = q["criteria"]
    return {"state": r["state"], "questions": {"q": out}}


if a.request:
    REQ = trade_request(a.request)


def call(_):
    t = time.perf_counter()
    p = json.loads(lam.invoke(FunctionName=a.fn, Payload=json.dumps(REQ))["Payload"].read())
    return (round(time.perf_counter() - t, 2), p.get("timing", {}).get("init_s"), p.get("timing", {}).get("cpu"), p.get("errorType"),
            p.get("timing", {}).get("engine_s"), p.get("usage", {}).get("input_tokens"))


out = {}
for wave in ("cold", "warm"):
    with cf.ThreadPoolExecutor(a.n) as ex:
        out[wave] = list(ex.map(call, range(a.n)))
    v = sorted(x[0] for x in out[wave])
    print(wave, "client s p50", v[len(v) // 2], "min", v[0], "max", v[-1], "errors", sum(1 for x in out[wave] if x[3]), flush=True)
json.dump({"env": env, **out}, open(a.out, "w"), indent=1)
