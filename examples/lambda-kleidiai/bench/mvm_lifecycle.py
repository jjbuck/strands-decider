"""MicroVM lifecycle latency: fresh RunMicrovm -> RUNNING -> first answer, and suspend -> auto-resume -> first answer.

  python mvm_lifecycle.py --image strands-decider-2b-mvm --fresh 5 --resume 5 --out mvm_lifecycle.json
Each fresh trial starts a new MicroVM from the snapshot and terminates it after three requests. Resume trials use one
MicroVM: suspend (API), wait until SUSPENDED plus 20 s, then send a request (autoResumeEnabled wakes it).
Requests go from this machine over HTTPS (new connection each), so they include the client's network round trip.
"""
import argparse
import json
import os
import time
import urllib.request

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--fresh", type=int, default=5)
ap.add_argument("--resume", type=int, default=5)
ap.add_argument("--out", required=True)
ap.add_argument("--request", default="", help="trade set whose first row is the request (default: the one-line ticket)")
a = ap.parse_args()
R = "us-east-2"
c = boto3.client("lambda-microvms", region_name=R)
ACCOUNT = boto3.client("sts", region_name=R).get_caller_identity()["Account"]
ARN = f"arn:aws:lambda:{R}:{ACCOUNT}:microvm-image:{a.image}"
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


def run():
    r = c.run_microvm(imageIdentifier=ARN,
                      ingressNetworkConnectors=[f"arn:aws:lambda:{R}:aws:network-connector:aws-network-connector:ALL_INGRESS"],
                      idlePolicy={"autoResumeEnabled": True, "maxIdleDurationSeconds": 900, "suspendedDurationSeconds": 900},
                      maximumDurationInSeconds=3600)
    return r["microvmId"]


def wait(mid, state):
    while (g := c.get_microvm(microvmIdentifier=mid))["state"] != state:
        time.sleep(0.2)
    return g


def post(host, auth):
    t = time.perf_counter()
    req = urllib.request.Request(f"https://{host}/decide", data=json.dumps(REQ).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **auth})
    p = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return round(time.perf_counter() - t, 3), p["timing"]["engine_s"], p.get("usage", {}).get("input_tokens")


out = {"fresh": [], "resume": []}
for i in range(a.fresh):
    t0 = time.time(); mid = run(); g = wait(mid, "RUNNING"); t1 = time.time()
    auth = c.create_microvm_auth_token(microvmIdentifier=mid, expirationInMinutes=10, allowedPorts=[{"allPorts": {}}])["authToken"]
    reqs = [post(g["endpoint"], auth) for _ in range(3)]
    out["fresh"].append({"run_to_running_s": round(t1 - t0, 2), "requests": reqs})
    print("fresh", out["fresh"][-1], flush=True)
    c.terminate_microvm(microvmIdentifier=mid)
mid = run(); g = wait(mid, "RUNNING")
auth = c.create_microvm_auth_token(microvmIdentifier=mid, expirationInMinutes=60, allowedPorts=[{"allPorts": {}}])["authToken"]
post(g["endpoint"], auth)
for i in range(a.resume):
    t = time.time(); c.suspend_microvm(microvmIdentifier=mid); wait(mid, "SUSPENDED"); ts = time.time() - t
    time.sleep(20)
    reqs = [post(g["endpoint"], auth) for _ in range(3)]
    out["resume"].append({"suspend_s": round(ts, 2), "requests": reqs})
    print("resume", out["resume"][-1], flush=True)
c.terminate_microvm(microvmIdentifier=mid)
json.dump(out, open(a.out, "w"), indent=1)
