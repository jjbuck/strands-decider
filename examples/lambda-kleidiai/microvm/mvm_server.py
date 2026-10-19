"""Lambda MicroVM server: Strands Decider 2B on llama.cpp behind HTTP, loaded before the snapshot.

The model loads in a background thread at start. Lambda's /ready build hook gets 503 until the load and a warm-up
request finish, then 200, and Lambda snapshots the VM with the weights resident. /validate runs sample requests on
the restored VM, so Lambda learns which snapshot pages a request touches and prefetches them on later starts.
POST /decide takes a System One request and returns the response with engine timing.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"eng": None, "err": None, "init_s": None, "stages": []}
T0 = time.time()


def stage(msg):
    STATE["stages"].append([round(time.time() - T0, 1), msg])
    print(f"[mvm] {time.time() - T0:.1f}s {msg}", flush=True)
LOCK = threading.Lock()
SAMPLE = {"state": {"ticket": "Customer reports the invoice total is wrong after a refund."},
          "questions": {"q": {"type": "choice", "instructions": "Route this ticket.",
                              "criteria": {"billing": "payment or invoice issue", "technical": "product defect",
                                           "account": "login or profile"}}}}


def load():
    t0 = time.time()
    try:
        stage(f"load start, cpus={os.cpu_count()}, gguf={os.environ['GGUF']}")
        from lcpp_engine import load_llamacpp_engine
        from strands_decider.schema import SystemOneRequest
        stage("imported engine")
        eng = load_llamacpp_engine(os.environ["CKPT_DIR"], os.environ["GGUF"], os.environ["SHIM"],
                                   int(os.environ.get("THREADS") or os.cpu_count()),
                                   n_seq=int(os.environ.get("N_SEQ", "17")))
        stage("model loaded")
        eng.evaluate(SystemOneRequest(**SAMPLE))
        stage("warm-up done")
        STATE.update(eng=eng, req=SystemOneRequest, init_s=time.time() - t0)
    except Exception as e:  # reported by /ready and /health
        STATE["err"] = repr(e)
        stage(f"load failed: {e!r}")


def decide(body):
    with LOCK:
        t = time.perf_counter()
        resp = STATE["eng"].evaluate(STATE["req"](**body))
        dt = time.perf_counter() - t
    out = resp.model_dump()
    out["timing"] = {"engine_s": round(dt, 4), "init_s": round(STATE["init_s"], 2), "cpus": os.cpu_count()}
    return out


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        ready = STATE["eng"] is not None
        if self.path.endswith("/runtime/v1/ready"):
            if os.environ.get("READY_EARLY") == "1":  # debug builds: snapshot before the model loads
                return self._send(200, {"ready": ready})
            if STATE["err"]:
                return self._send(500, {"ready": False, "err": STATE["err"]})
            return self._send(200 if ready else 503, {"ready": ready, "err": STATE["err"]})
        if self.path.endswith("/runtime/v1/validate"):
            if not ready:
                return self._send(503, {"ready": False})
            for _ in range(3):
                decide(SAMPLE)
            return self._send(200, {"validated": True})
        if "/aws/lambda-microvms/" in self.path:  # other lifecycle hooks (run, resume, suspend, terminate)
            return self._send(200, {"ok": True})
        return self._send(200, {"ready": ready, "err": STATE["err"], "init_s": STATE["init_s"], "cpus": os.cpu_count(),
                                "stages": STATE["stages"]})

    def do_POST(self):
        if "/aws/lambda-microvms/" in self.path:  # lifecycle hooks may arrive as POST; answer them like GET
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n:
                self.rfile.read(n)
            return self.do_GET()
        if STATE["eng"] is None:
            return self._send(503, {"ready": False, "err": STATE["err"]})
        n = int(self.headers.get("Content-Length", 0))
        try:
            return self._send(200, decide(json.loads(self.rfile.read(n))))
        except Exception as e:
            return self._send(400, {"error": repr(e)[:500]})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=load, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", 8080))), H).serve_forever()
