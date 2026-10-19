"""In-region caller for a Lambda MicroVM decider: POST each request over one keep-alive HTTPS connection."""
import http.client
import json
import time


def handler(event, context):
    host, hdr = event["host"], event["auth"]
    conn = http.client.HTTPSConnection(host, timeout=300)
    out = []
    for req in event["requests"]:
        body = json.dumps(req["body"])
        t = time.perf_counter()
        conn.request("POST", "/decide", body=body, headers={"Content-Type": "application/json", **hdr})
        r = conn.getresponse()
        p = json.loads(r.read())
        dt = time.perf_counter() - t
        out.append({"id": req["id"], "set": req["set"], "client_s": dt, "status": r.status,
                    **({"answer": p["answers"]["q"], "tokens": p["usage"]["input_tokens"],
                        "engine_s": p["timing"]["engine_s"], "cpu": f"mvm-{p['timing']['cpus']}cpu"}
                       if "answers" in p else {"error": str(p)[:200]})})
    return out
