"""Deploy the pinned lambda-kleidiai baseline: one bucket, one role, two functions with the same zip.

  python deploy.py up        --build build/ [--prefix lkai-p1] [--region us-east-2]
  python deploy.py upload    --build build/                 # weights + zip to the bucket (run where build/ is)
  python deploy.py functions --build build/                 # role, log group, both functions, runtime pin
  python deploy.py show
  python deploy.py teardown  [--all]                        # versions (SnapStart cache billing); --all: everything

Functions (identical code, config and environment except LKAI_PHASES):
  <prefix>-inst   LKAI_PHASES=1  phase records on stdout (baseline/phases.py)
  <prefix>-rep    LKAI_PHASES=0  Lambda's REPORT line only: the overhead control
Both: python3.12, arm64, 10,240 MB, 512 MB ephemeral storage (SnapStart's limit), 900 s, SnapStart on published
versions, handler lkai_entry.handler, the runtime version pinned (RuntimeManagementConfig Manual) to the one the
first invocation reports, env MODEL_BUCKET/MODEL_KEY/DEC_KV=f32/DEC_UBATCH=512/N_SEQ=17 as snapstart/deploy.py.
The role reads only the GGUF object and writes only its own log groups. The bucket blocks public access, enforces
TLS and SSE-S3. State (names, ARNs, sha256s) goes to build/deployment.json, which bench.py reads.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time

import boto3
from botocore.exceptions import ClientError

ARMS = {"inst": "1", "rep": "0"}
TAGS = {"Project": "strands-decider", "Experiment": "lambda-kleidiai-P1"}
RUNTIME = "python3.12"
MEMORY = 10240


def _state_path(build: str) -> str:
    return os.path.join(build, "deployment.json")


def load_state(build: str) -> dict:
    p = _state_path(build)
    return json.load(open(p)) if os.path.exists(p) else {}


def save_state(build: str, st: dict) -> None:
    tmp = _state_path(build) + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, _state_path(build))


def wait_updated(lam, name: str, qualifier: str | None = None) -> dict:
    """Wait until a function/configuration is usable and a published version's snapshot is ready."""
    q = {"Qualifier": qualifier} if qualifier else {}
    while True:
        c = lam.get_function_configuration(FunctionName=name, **q)
        if c["State"] == "Pending" or c.get("LastUpdateStatus") == "InProgress":
            time.sleep(3)
            continue
        if c["State"] != "Active" or c.get("LastUpdateStatus") not in (None, "Successful"):
            sys.exit(
                f"{name}:{qualifier}: {c['State']} {c.get('StateReason')} {c.get('LastUpdateStatusReason')}"
            )
        if qualifier and c.get("SnapStart", {}).get("ApplyOn") == "PublishedVersions":
            status = c["SnapStart"].get("OptimizationStatus")
            if status == "Off":
                time.sleep(3)
                continue
            if status != "On":
                sys.exit(f"{name}:{qualifier}: unexpected SnapStart optimization status {status!r}")
        return c


def ensure_bucket(s3, region: str, st: dict, prefix: str) -> str:
    if st.get("bucket"):
        return st["bucket"]
    name = f"{prefix}-{region}-{os.urandom(4).hex()}"
    s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": region})
    s3.put_public_access_block(
        Bucket=name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=name,
        ServerSideEncryptionConfiguration={
            "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
        },
    )
    s3.put_bucket_policy(
        Bucket=name,
        Policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "TLSOnly",
                        "Effect": "Deny",
                        "Principal": "*",
                        "Action": "s3:*",
                        "Resource": [f"arn:aws:s3:::{name}", f"arn:aws:s3:::{name}/*"],
                        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                    }
                ],
            }
        ),
    )
    s3.put_bucket_tagging(
        Bucket=name, Tagging={"TagSet": [{"Key": k, "Value": v} for k, v in TAGS.items()]}
    )
    st["bucket"] = name
    return name


def upload(sess, build: str, st: dict) -> None:
    s3 = sess.client("s3")
    m = json.load(open(os.path.join(build, "manifest.json")))
    bucket = ensure_bucket(s3, sess.region_name, st, st["prefix"])
    g, z = m["model"]["gguf_sha256"], m["function_zip"]["sha256"]
    items = [
        (os.path.join(build, "model", "s2b-h512-Q8_0.gguf"), f"weights/{g}/s2b-h512-Q8_0.gguf"),
        (os.path.join(build, "function.zip"), f"code/{z}/function.zip"),
        (os.path.join(build, "manifest.json"), f"code/{z}/manifest.json"),
    ]
    for path, key in items:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print("present", key)
            continue
        except ClientError:
            pass
        t = time.time()
        s3.upload_file(path, bucket, key)
        print(f"uploaded {key} {os.path.getsize(path) / 2**20:.0f} MiB in {time.time() - t:.0f} s")
    st.update(model_key=items[0][1], zip_key=items[1][1], manifest=m)


def ensure_role(iam, st: dict, bucket: str) -> str:
    name = f"{st['prefix']}-fn"
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(
            RoleName=name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Tags=[{"Key": k, "Value": v} for k, v in TAGS.items()],
        )["Role"]["Arn"]
        time.sleep(10)  # IAM propagation before Lambda can assume it
    region = st["region"]
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket}/{st['model_key']}",
            },
            {
                "Effect": "Allow",
                "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [
                    f"arn:aws:logs:{region}:*:log-group:/aws/lambda/{st['prefix']}-{a}:*"
                    for a in ARMS
                ],
            },
        ],
    }
    iam.put_role_policy(
        RoleName=name, PolicyName="lkai-baseline", PolicyDocument=json.dumps(policy)
    )
    st["role"] = arn
    return arn


def runtime_version_arn(log: str) -> str | None:
    m = re.search(r"Runtime Version ARN: (\S+)", log)
    return m.group(1) if m else None


def functions(sess, st: dict) -> None:
    lam, logs, iam = sess.client("lambda"), sess.client("logs"), sess.client("iam")
    role = ensure_role(iam, st, st["bucket"])
    st.setdefault("functions", {})
    for arm, flag in ARMS.items():
        name = f"{st['prefix']}-{arm}"
        group = f"/aws/lambda/{name}"
        try:
            logs.create_log_group(logGroupName=group, tags=TAGS)
        except logs.exceptions.ResourceAlreadyExistsException:
            pass
        logs.put_retention_policy(logGroupName=group, retentionInDays=14)
        env = {
            "MODEL_BUCKET": st["bucket"],
            "MODEL_KEY": st["model_key"],
            "DEC_KV": "f32",
            "DEC_UBATCH": "512",
            "N_SEQ": "17",
            "LKAI_PHASES": flag,
            "COLD_TAG": "0",
        }
        cfg = dict(
            FunctionName=name,
            Runtime=RUNTIME,
            Architectures=["arm64"],
            Handler="lkai_entry.handler",
            Role=role,
            MemorySize=MEMORY,
            EphemeralStorage={"Size": 512},
            Timeout=900,
            SnapStart={"ApplyOn": "PublishedVersions"},
            Environment={"Variables": env},
            Description=f"lambda-kleidiai P1 baseline ({arm}); zip {st['manifest']['function_zip']['sha256'][:12]}",
        )
        for attempt in range(6):
            try:
                try:
                    lam.get_function(FunctionName=name)
                    lam.update_function_code(
                        FunctionName=name, S3Bucket=st["bucket"], S3Key=st["zip_key"]
                    )
                    wait_updated(lam, name)
                    lam.update_function_configuration(
                        **{k: v for k, v in cfg.items() if k != "Architectures"}
                    )
                except lam.exceptions.ResourceNotFoundException:
                    lam.create_function(
                        **cfg, Code={"S3Bucket": st["bucket"], "S3Key": st["zip_key"]}, Tags=TAGS
                    )
                break
            except lam.exceptions.InvalidParameterValueException as e:  # role not assumable yet
                if attempt == 5:
                    raise
                print("retrying:", e)
                time.sleep(10)
        c = wait_updated(lam, name)
        st["functions"][arm] = {
            "name": name,
            "arn": c["FunctionArn"],
            "code_sha256_b64": c["CodeSha256"],
            "code_size": c["CodeSize"],
        }
        print("function", name, c["CodeSha256"], c["CodeSize"])
    pin_runtime(lam, st, logs)


def pin_runtime(lam, st: dict, logs=None) -> None:
    """Invoke $LATEST once (on-demand: no snapshot) and pin both functions to the runtime version it ran on.

    The version comes from the log tail, the function configuration (RuntimeVersionConfig) or the INIT_START line
    in CloudWatch Logs; the 4 KB tail of a cold invoke is usually too short to still hold INIT_START."""
    name = st["functions"]["inst"]["name"]
    t0 = int(time.time() * 1000) - 5000
    r = lam.invoke(
        FunctionName=name,
        LogType="Tail",
        Payload=json.dumps(
            {"state": "warm up", "questions": {"q": {"type": "noul", "instructions": "Ready?"}}}
        ).encode(),
    )
    log = base64.b64decode(r.get("LogResult", "")).decode(errors="replace")
    body = r["Payload"].read()
    if r.get("FunctionError"):
        sys.exit(f"smoke invoke failed: {body[:600]!r}\n{log[-1500:]}")
    print("smoke invoke ok:", body[:300].decode(errors="replace"))
    arn = runtime_version_arn(log)
    if not arn:
        cfg = lam.get_function_configuration(FunctionName=name)
        arn = (cfg.get("RuntimeVersionConfig") or {}).get("RuntimeVersionArn")
    for _ in range(12):
        if arn or logs is None:
            break
        time.sleep(5)
        ev = logs.filter_log_events(
            logGroupName=f"/aws/lambda/{name}", startTime=t0, filterPattern='"Runtime Version ARN"'
        ).get("events", [])
        arn = next((a for a in (runtime_version_arn(e["message"]) for e in ev) if a), None)
    if not arn:
        sys.exit("runtime version not in the log tail, configuration or INIT_START; pin it by hand")
    for f in st["functions"].values():
        lam.put_runtime_management_config(
            FunctionName=f["name"], UpdateRuntimeOn="Manual", RuntimeVersionArn=arn
        )
    st["runtime_version_arn"] = arn
    print("runtime pinned", arn)


def teardown(sess, st: dict, everything: bool) -> None:
    lam = sess.client("lambda")
    for f in st.get("functions", {}).values():
        try:
            for p in lam.get_paginator("list_versions_by_function").paginate(
                FunctionName=f["name"]
            ):
                for v in p["Versions"]:
                    if v["Version"] != "$LATEST":
                        lam.delete_function(FunctionName=f["name"], Qualifier=v["Version"])
                        print("deleted", f["name"], v["Version"])
            if everything:
                lam.delete_function(FunctionName=f["name"])
                print("deleted", f["name"])
        except lam.exceptions.ResourceNotFoundException:
            pass
    if not everything:
        return
    iam = sess.client("iam")
    try:
        iam.delete_role_policy(RoleName=f"{st['prefix']}-fn", PolicyName="lkai-baseline")
        iam.delete_role(RoleName=f"{st['prefix']}-fn")
    except iam.exceptions.NoSuchEntityException:
        pass
    if st.get("bucket"):
        b = sess.resource("s3").Bucket(st["bucket"])
        b.objects.all().delete()
        b.delete()
        print("deleted bucket", st["bucket"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "upload", "functions", "show", "teardown"])
    ap.add_argument(
        "--build", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    )
    ap.add_argument("--prefix", default="lkai-p1")
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    st = load_state(a.build)
    st.setdefault("prefix", a.prefix)
    st.setdefault("region", a.region)
    if st["region"] != a.region:
        sys.exit(f"{_state_path(a.build)} is for {st['region']}")
    try:
        if a.cmd in ("up", "upload"):
            upload(sess, a.build, st)
        if a.cmd in ("up", "functions"):
            if "model_key" not in st:
                sys.exit("run upload first")
            functions(sess, st)
        if a.cmd == "teardown":
            teardown(sess, st, a.all)
    finally:
        save_state(a.build, st)
    if a.cmd == "show":
        print(json.dumps({k: v for k, v in st.items() if k != "manifest"}, indent=1))


def b64_to_hex(b64: str) -> str:
    """Lambda's CodeSha256 (base64) as hex, to compare with manifest.json's function_zip.sha256."""
    return base64.b64decode(b64).hex()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    main()
