"""Lambda handler: Strands Decider 2B on llama.cpp (CPU), one decision request per invocation.

Event: a System One request ({"state": ..., "questions": {...}}), either as the event itself or as a function-URL /
API Gateway JSON "body". Returns the System One response plus timing: engine_s (render + tokenise + forward + head)
and init_s (this sandbox's model load), and the sandbox's CPU model.
"""
import json
import os
import threading
import time

T0 = time.time()


def _stage(name):
    print(f"[init] {name} {time.time() - T0:.1f}s", flush=True)


def _prefetch(path, workers=16, chunk=64 << 20):
    """Read the weights file in parallel ranges before llama.cpp maps it: the container image is loaded lazily, and
    llama.cpp's page-by-page faults through it are far slower than large concurrent reads."""
    from concurrent.futures import ThreadPoolExecutor
    size = os.path.getsize(path)

    def rd(off):
        with open(path, "rb", buffering=0) as fh:
            fh.seek(off)
            left = min(chunk, size - off)
            while left > 0:
                b = fh.read(min(left, 8 << 20))
                if not b:
                    break
                left -= len(b)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(rd, range(0, size, chunk)))


def _from_s3(uri, dest, workers=32, part=16 << 20):
    """Parallel ranged GETs from S3 into local ephemeral storage (/tmp)."""
    import boto3
    from boto3.s3.transfer import TransferConfig
    bucket, key = uri[5:].split("/", 1)
    cfg = TransferConfig(max_concurrency=workers, multipart_chunksize=part, multipart_threshold=part)
    boto3.client("s3").download_file(bucket, key, dest, Config=cfg)


def _weights():
    if os.environ.get("GGUF_S3"):  # weights from S3, not the lazily loaded image (measured: 2 GB took 256 s that way)
        dest = f"/tmp/model-{os.getpid()}.gguf.part"
        if not os.path.exists("/tmp/model.gguf"):
            _from_s3(os.environ["GGUF_S3"], dest)
            os.replace(dest, "/tmp/model.gguf")  # /tmp is shared by Managed Instances' worker processes
        os.environ["GGUF"] = "/tmp/model.gguf"
        _stage("downloaded weights from S3")
    elif os.environ.get("PREFETCH", "0") == "1":
        _prefetch(os.environ["GGUF"])
        _stage("prefetched weights")


THREADS = int(os.environ.get("THREADS") or os.cpu_count())
CPU = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith(("model name", "CPU part"))), "?")
ENG, INIT_S, INIT_ERR = None, None, None
_READY = threading.Event()


def _load():
    """Load the engine and run one warm-up decision. INIT_MODE picks where this runs:
    import      in the init phase (Lambda's 10 s init limit kills it, and init reruns inside the first invoke)
    background  in a thread started at import, so the init phase returns at once and the first invoke waits
    lazy        inside the first invoke (Lambda Managed Instances' Python runtime runs requests in worker processes)"""
    global ENG, INIT_S, INIT_ERR
    try:
        _weights()
        from lcpp_engine import load_llamacpp_engine
        from strands_decider.schema import SystemOneRequest
        _stage("imported engine (torch, transformers)")
        eng = load_llamacpp_engine(os.environ["CKPT_DIR"], os.environ["GGUF"], os.environ["SHIM"], THREADS,
                                   mmap=os.environ.get("MMAP", "1") == "1", n_seq=int(os.environ.get("N_SEQ", "17")))
        _stage("loaded model")
        eng.evaluate(SystemOneRequest(state="warm-up",
                                      questions={"q": {"type": "noul", "instructions": "Is this a warm-up?"}}))
        ENG, INIT_S = eng, time.time() - T0
        _stage("warm-up done")
    except Exception as e:  # surfaced by the handler
        INIT_ERR = repr(e)
    finally:
        _READY.set()


INIT_MODE = os.environ.get("INIT_MODE", "import")
if INIT_MODE == "import":
    _load()
elif INIT_MODE == "background":
    threading.Thread(target=_load, daemon=True).start()


def handler(event, context):
    if INIT_MODE == "lazy" and not _READY.is_set():
        _load()
    _READY.wait()
    if INIT_ERR:
        raise RuntimeError(f"engine failed to load: {INIT_ERR}")
    if event.get("mode") == "native_eval":  # accuracy runs: an S3 shard of suite rows in, jevscale results out
        from eval_native import run_shard
        return run_shard(ENG, event["in"], event["out"], CPU)
    req = json.loads(event["body"]) if isinstance(event.get("body"), str) else event
    t = time.perf_counter()
    from strands_decider.schema import SystemOneRequest
    resp = ENG.evaluate(SystemOneRequest(**req))
    dt = time.perf_counter() - t
    out = resp.model_dump()
    out["timing"] = {"engine_s": round(dt, 4), "init_s": round(INIT_S, 2), "threads": THREADS, "cpu": CPU}
    return out
