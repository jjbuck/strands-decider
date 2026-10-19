"""Create and manage the v2 zip functions (strands-decider-2b-v2, -v2q). Runs from the operator's machine (PowerUser).

  python deploy.py create NAME --gguf-key KEY [--unrotate s3://...npz] [--ubatch N]
  python deploy.py env NAME KEY=VAL ...          # merge into $LATEST's environment, wait until applied
  python deploy.py publish NAME                  # new COLD_TAG (forces a new version), publish, wait for the snapshot
  python deploy.py alias NAME VERSION            # create or move alias "live"
  python deploy.py prune NAME                    # delete every published version except the one "live" points to
  python deploy.py show NAME

Function settings: python3.12, arm64, 10,240 MB, 512 MB ephemeral storage (SnapStart's limit), 900 s, role
LAMBDA_ROLE, SnapStart on published versions, handler app.handler, code s3://your-bucket/lambda/results/
v2/function.zip, tags Project=strands-decider, Name=<NAME>. Environment: MODEL_BUCKET / MODEL_KEY (the GGUF,
streamed into a memfd by PR #77's loader), DEC_KV=f32, DEC_UBATCH, N_SEQ=17, and for v2q DEC_UNROTATE_S3.
"""
import argparse
import json
import os
import sys
import time

import boto3

R, B = "us-east-2", os.environ.get("DECIDER_BUCKET", "your-bucket")
ROLE = os.environ["DECIDER_LAMBDA_ROLE_ARN"]  # execution role: S3 read of the GGUF, CloudWatch Logs
ZIP_KEY = "lambda/results/v2/function.zip"
lam = boto3.client("lambda", region_name=R)


def wait_updated(name, qualifier=None):
    while True:
        c = lam.get_function_configuration(FunctionName=name, **({"Qualifier": qualifier} if qualifier else {}))
        if c["State"] == "Pending" or c.get("LastUpdateStatus") == "InProgress":
            time.sleep(3)
            continue
        if c["State"] != "Active" or c.get("LastUpdateStatus") not in (None, "Successful"):
            sys.exit(f"{name}:{qualifier}: State {c['State']} {c.get('StateReason')} LastUpdate "
                     f"{c.get('LastUpdateStatus')} {c.get('LastUpdateStatusReason')}")
        return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["create", "env", "publish", "alias", "prune", "show"])
    ap.add_argument("name")
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--gguf-key")
    ap.add_argument("--unrotate")
    ap.add_argument("--ubatch", default="512")
    a = ap.parse_args()
    n = a.name
    assert n.startswith("strands-decider-2b-v2"), n
    if a.cmd == "create":
        env = {"MODEL_BUCKET": B, "MODEL_KEY": a.gguf_key, "DEC_KV": "f32", "DEC_UBATCH": a.ubatch, "N_SEQ": "17"}
        if a.unrotate:
            env["DEC_UNROTATE_S3"] = a.unrotate
        r = lam.create_function(
            FunctionName=n, Runtime="python3.12", Architectures=["arm64"], Handler="app.handler", Role=ROLE,
            Code={"S3Bucket": B, "S3Key": ZIP_KEY}, MemorySize=10240, EphemeralStorage={"Size": 512}, Timeout=900,
            SnapStart={"ApplyOn": "PublishedVersions"}, Environment={"Variables": env},
            Description="Strands Decider 2B on llama.cpp (portable KleidiAI build, f74e2fa6a + patch_llama.py), PR #77 "
                        "torso-free SnapStart packaging; trade/lambda/v2",
            Tags={"Project": "strands-decider", "Name": n})
        print("created", r["FunctionArn"], "code sha256", r["CodeSha256"], "size", r["CodeSize"])
        c = wait_updated(n)
        print(json.dumps({k: c.get(k) for k in ("State", "LastUpdateStatus", "CodeSha256", "Environment")}))
    elif a.cmd == "env":
        c = lam.get_function_configuration(FunctionName=n)
        env = dict(c.get("Environment", {}).get("Variables", {}))
        for kv in a.rest:
            k, v = kv.split("=", 1)
            if v == "":
                env.pop(k, None)
            else:
                env[k] = v
        lam.update_function_configuration(FunctionName=n, Environment={"Variables": env})
        c = wait_updated(n)
        print("env", c["Environment"]["Variables"])
    elif a.cmd == "publish":
        c = lam.get_function_configuration(FunctionName=n)
        env = dict(c["Environment"]["Variables"], COLD_TAG=f"{time.time():.3f}")
        lam.update_function_configuration(FunctionName=n, Environment={"Variables": env})
        wait_updated(n)
        t0 = time.time()
        v = lam.publish_version(FunctionName=n, Description=f"COLD_TAG {env['COLD_TAG']}")["Version"]
        c = wait_updated(n, v)
        print("published", n, "version", v, f"in {time.time() - t0:.0f} s (snapshot ready)", "SnapStart",
              c.get("SnapStart"), "env", c["Environment"]["Variables"])
    elif a.cmd == "alias":
        v = a.rest[0]
        try:
            r = lam.update_alias(FunctionName=n, Name="live", FunctionVersion=v)
        except lam.exceptions.ResourceNotFoundException:
            r = lam.create_alias(FunctionName=n, Name="live", FunctionVersion=v)
        print("alias", r["AliasArn"], "->", r["FunctionVersion"])
    elif a.cmd == "prune":
        live = lam.get_alias(FunctionName=n, Name="live")["FunctionVersion"]
        vs = [v["Version"] for p in lam.get_paginator("list_versions_by_function").paginate(FunctionName=n)
              for v in p["Versions"] if v["Version"] != "$LATEST"]
        for v in vs:
            if v != live:
                lam.delete_function(FunctionName=n, Qualifier=v)
                print("deleted", n, "version", v)
        print("kept version", live, "(live)")
    else:
        c = lam.get_function_configuration(FunctionName=n)
        vs = [(v["Version"], v.get("Description")) for p in lam.get_paginator("list_versions_by_function").paginate(
            FunctionName=n) for v in p["Versions"]]
        al = lam.list_aliases(FunctionName=n)["Aliases"]
        print(json.dumps({"config": {k: c.get(k) for k in ("Runtime", "Architectures", "MemorySize", "Timeout",
                                                         "EphemeralStorage", "SnapStart", "Handler", "CodeSha256",
                                                         "CodeSize", "Environment", "Role")},
                          "versions": vs, "aliases": [(x["Name"], x["FunctionVersion"]) for x in al],
                          "concurrency": lam.get_function_concurrency(FunctionName=n).get(
                              "ReservedConcurrentExecutions"),
                          "tags": lam.list_tags(Resource=c["FunctionArn"])["Tags"]}, indent=1))


main()
