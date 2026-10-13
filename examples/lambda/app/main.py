"""The System One API on AWS Lambda, served by llama.cpp (Q8_0) without torch.

Same request and response schema as `strands-decider serve`. Everything heavy (the
GGUF from S3 into memory, the model load, a warm-up forward) runs at import, so with
SnapStart it runs once when a version is published and is restored from the snapshot.
"""

from __future__ import annotations

import logging
import os
import time

import loader
from engine import DeciderEngine
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("decider.app")

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("MODEL_NAME", "strands-decider-2B-qwen3.5-v1-2610-q8_0")
THREADS = int(os.environ.get("THREADS", "0")) or (os.cpu_count() or 1)

_t0 = time.perf_counter()
ENGINE = DeciderEngine(
    lib_path=os.path.join(HERE, "libs", "libdecider_hidden.so"),
    gguf_path=loader.download_to_memfd(os.environ["MODEL_BUCKET"], os.environ["MODEL_KEY"]),
    assets_dir=os.path.join(HERE, "assets"),
    threads=THREADS,
)
# Pay the first-forward costs before the snapshot is taken.
ENGINE.evaluate(
    {
        "state": "warm up",
        "questions": {"q": {"type": "noul", "instructions": "Is this a warm-up?"}},
    },
    MODEL_NAME,
)
INIT_SECONDS = round(time.perf_counter() - _t0, 2)
log.info("model ready in %.1fs, n_embd=%d, threads=%d", INIT_SECONDS, ENGINE.n_embd, THREADS)


def _cpu_part() -> str:
    """CPU part of the host serving this request: 0xd0c is Graviton2, 0xd40 Graviton3.

    Read per request, because a SnapStart snapshot can restore on either.
    """
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("CPU part"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "?"


app = FastAPI(title="strands-decider on AWS Lambda (llama.cpp)")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "backend": "llama.cpp",
        "quant": "Q8_0",
        "threads": THREADS,
        "init_seconds": INIT_SECONDS,
        "cpu_part": _cpu_part(),
    }


@app.post("/v1/systemone")
def systemone(request: dict) -> JSONResponse:
    started = time.perf_counter()
    try:
        payload = ENGINE.evaluate(request, MODEL_NAME)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    payload["cpu_part"] = _cpu_part()
    log.info(
        "latency_ms=%s cpu_part=%s input_tokens=%s",
        payload["latency_ms"],
        payload["cpu_part"],
        payload["usage"]["input_tokens"],
    )
    return JSONResponse(content=payload)
