"""Lambda entry point (handler "app.handler") of strands-decider-2b-v2 and -v2q: PR #77's handler.py, unchanged, plus
two things around it.

1. {"mode": "native_eval", "in": "s3://.../shard.jsonl.gz", "out": "s3://.../out/shard.jsonl.gz"} runs eval_np.run_shard
   on the loaded engine (the deployed image's eval mode, torch-free); the result is the shard's stats.
2. A direct invocation's response also carries "timing" {"engine_s", "init_s", "threads", "cpu"}, the deployed
   handler's field, so client/drive.py and bench/lambda_shim.py run unchanged. engine_s is PR #77's latency_ms / 1000
   (engine.evaluate), and cpu is the CPU part of the host serving this request (a SnapStart snapshot restores on any).
Everything else, including the function-URL routes, the SnapStart load at import and the memfd weights, is PR #77's.
"""
import engine as _engine
import handler as _pr77


def handler(event, context):
    if isinstance(event, dict) and event.get("mode") == "native_eval":
        eng, _ = _pr77._ENGINE.result()
        from eval_np import run_shard
        return run_shard(eng, _engine.SHIM, event["in"], event["out"], _pr77._cpu_part())
    out = _pr77.handler(event, context)
    if isinstance(out, dict) and "latency_ms" in out and "statusCode" not in out:
        _, init_s = _pr77._ENGINE.result()
        out["timing"] = {"engine_s": round(out["latency_ms"] / 1000, 4), "init_s": init_s, "threads": _pr77.THREADS,
                         "cpu": _pr77._cpu_part()}
    return out
