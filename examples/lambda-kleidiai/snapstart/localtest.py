"""Box test of the v2 zip as Lambda runs it (inside amazonlinux:2023, python3.12, PYTHONPATH=/task:/pydeps where /pydeps
holds only boto3, which the Lambda runtime provides). Imports app (PR #77's handler loads the engine at import, as under
SnapStart; MODEL_PATH is a local GGUF), then runs one native_eval shard through app.handler and two interactive requests.

  python3.12 localtest.py s3://.../in/core2k/shard_00002.jsonl.gz s3://.../out/v2local/<tag>/shard_00002.jsonl.gz
"""
import json
import sys
import time

t0 = time.time()
import app  # noqa: E402

print("import+load", round(time.time() - t0, 1), "s", flush=True)
print(json.dumps(app._engine.SHIM.init_info), flush=True)
r = app.handler({"state": "Order 1182 arrived with a cracked screen. The customer wants a refund today.",
                 "questions": {"q": {"type": "choice", "instructions": "Route the ticket.",
                                     "criteria": {"refunds": "money back", "shipping": "delivery problems",
                                                  "tech": "device faults"}},
                               "u": {"type": "noul", "instructions": "Is the customer upset?"}}}, None)
print("interactive 2q:", json.dumps(r)[:600], flush=True)
r = app.handler({"state": "Is the sky green?", "questions": {"q": {"type": "noul", "instructions": "Is it true?"}}},
                None)
print("interactive 1q:", json.dumps(r)[:400], flush=True)
if len(sys.argv) > 2:
    t = time.time()
    st = app.handler({"mode": "native_eval", "in": sys.argv[1], "out": sys.argv[2]}, None)
    print("native_eval", json.dumps(st), round(time.time() - t, 1), "s", flush=True)
print("LOCALTEST_DONE", flush=True)
