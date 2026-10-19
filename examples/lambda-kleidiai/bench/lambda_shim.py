"""HTTP front for the Lambda decider, so harnesses written against `strands-decider serve` can score the deployed function.

  python lambda_shim.py [--port 8000] [--function strands-decider-2b]

POST /v1/systemone: the body goes to lambda:Invoke unchanged; the response is the function's SystemOneResponse, with
`latency_ms` set to the in-function engine time as `serve` reports it, and the round trip in `x_roundtrip_ms`.
A ValueError in the engine (e.g. too many options) returns 422, as `serve` does. GET /health mirrors serve's fields.
Every request is logged to --log as one JSON line (status, engine and round-trip ms, sandbox CPU, input tokens).
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
from botocore.config import Config

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--function", default="strands-decider-2b")
ap.add_argument("--region", default="us-east-2")
ap.add_argument("--log", default="shim.jsonl")
a = ap.parse_args()
lam = boto3.client("lambda", region_name=a.region,
                   config=Config(read_timeout=900, connect_timeout=10, retries={"max_attempts": 8, "mode": "adaptive"},
                                 max_pool_connections=64))
lock = threading.Lock()
HEALTH = {"status": "ok", "model": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
          "checkpoint": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", "base_model": "Qwen/Qwen3.5-2B-Base",
          "max_length": 4096, "device": f"lambda:{a.function}", "prefix_cache": True, "vision": False}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, HEALTH) if self.path.startswith("/health") else self._send(404, {"detail": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/systemone"):
            return self._send(404, {"detail": "not found"})
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        t = time.perf_counter()
        r = lam.invoke(FunctionName=a.function, Payload=body)
        rt = (time.perf_counter() - t) * 1000
        p = json.loads(r["Payload"].read())
        rec = {"t": time.time(), "roundtrip_ms": round(rt, 1)}
        if r.get("FunctionError"):
            msg = str(p.get("errorMessage", p))[:500]
            code = 422 if p.get("errorType") in ("ValueError", "ValidationError") else 502
            rec.update(status=code, error=msg)
            self._send(code, {"detail": msg})
        else:
            tm = p.pop("timing", {})
            p["latency_ms"] = round(1000 * tm.get("engine_s", 0), 2)
            p["x_roundtrip_ms"] = round(rt, 1)
            rec.update(status=200, engine_ms=p["latency_ms"], cpu=tm.get("cpu"), tokens=p.get("usage", {}).get("input_tokens"))
            self._send(200, p)
        with lock, open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
